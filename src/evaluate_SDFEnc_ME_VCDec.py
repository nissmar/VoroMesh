import os
os.environ['OMP_NUM_THREADS'] = '16'
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import numpy as np
import yaml
import io
import argparse
from tqdm import tqdm
from time import time

from datasets.cloud_transformation import ComposeCloudTransformation
from datasets.sample_transformation import ComposeSampleTransformation
from datasets.ABCDataset import ABCDataset, abc_collate_fn
from networks.sdf_encoders import LocalSDFFE_ME, GlobFeatEnc
from networks.vc_decoders import VCDec, VCOccDec
from utils.networks import cnt_params, AverageMeter
from utils.custom_voronoi import Voroloss_opt, VoronoiValues, get_clean_shape


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str,
                        help='Path to config file in YAML format.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    return parser


def percentage_valid_edges(nt):
    '''nt: triangle faces, Nx3 array of indices'''
    edges = np.concatenate((nt[:, :2], nt[:, 1:], nt[:, ::2]))
    s_e = edges.min(1)+(nt.max()+1)*edges.max(1)
    _, c = np.unique(s_e, return_counts=True)
    return (c == 2).mean()


def export_obj(nv: np.ndarray, nf: np.ndarray, name: str):
    if name[:-4] != ".obj":
        name += ".obj"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    # file.write("o {} \n".format(name))
    for e in nv:
        file.write("v {} {} {}\n".format(*e))
    file.write("\n")
    for face in nf:
        file.write("f " + " ".join([str(fi + 1) for fi in face]) + "\n")
    file.write("\n")


def evaluate_and_reconstruct(dataset, iterator):
    data_time = AverageMeter()
    inf_time = AverageMeter()
    batch_time = AverageMeter()
    VL = AverageMeter()
    CE = AverageMeter()

    encoder.eval()
    if config['vcdec_in_glob']:
        glob_encoder.eval()
    vcdec.eval()
    occdec.eval()
    torch.set_grad_enabled(False)

    obj_ind = 0
    end = time()
    for i, batch in enumerate(iterator):
        torch.cuda.empty_cache()
        if i >= len(iterator):
            break
        data_time.update(time() - end)
        end = time()

        input_sparse_xyz = batch['input_sparse_xyz'].to(device)
        input_sparse_sdf = (
            batch['input_sparse_sdf'].to(device) *
            torch.repeat_interleave(batch['input_gs'].to(device) / config['grid_size'][0],
                                    batch['input_sparse_sdf_idx_size'].to(device),
                                    dim=0)
        ).unsqueeze(1)
        input_sparse_sdf_idx = batch['input_sparse_sdf_idx'].to(device).int()
        input_sdf = ME.SparseTensor(input_sparse_sdf, input_sparse_sdf_idx)
        grid_features = encoder(input_sdf)

        if config['vcdec_in_glob'] and config['vcdec_p_dim']:
            vc = vcdec(
                grid_features.features,
                query=input_sparse_xyz,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(device),
                                                   dim=0)
            )
        elif config['vcdec_in_glob']:
            vc = vcdec(
                grid_features.features,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(device),
                                                   dim=0)
            )
        elif config['vcdec_p_dim']:
            vc = vcdec(grid_features.features, query=input_sparse_xyz)
        else:
            vc = vcdec(grid_features.features)
        vc /= torch.repeat_interleave(
            batch['input_gs'].to(device),
            batch['input_sparse_sdf_idx_size'].to(device), 
            dim=0
        ).unsqueeze(1)

        if config['vcdec_vc_tanh']:
            vc *= config['vcdec_vc_tanh_scale']

        vc += input_sparse_xyz

        if config['vcdec_in_glob'] and config['vcdec_p_dim']:
            vc_occ = occdec(
                grid_features.features, query=vc,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(device),
                                                   dim=0)
            )
        elif config['vcdec_in_glob']:
            vc_occ = occdec(
                grid_features.features,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(device),
                                                   dim=0)
            )
        elif config['vcdec_p_dim']:
            vc_occ = occdec(grid_features.features, query=vc)
        else:
            vc_occ = occdec(grid_features.features)
        inf_time.update(time() - end)

        if batch.get('gt_vg_occ', None) is not None:
            ce = celoss(vc_occ, batch['gt_vg_occ'].to(device))
            ce = ce.item()
        else:
            ce = 0
        gt_clouds = batch['gt_cloud_glob'].to(device)
        l = 0
        r = 0
        vl = 0
        for j in range(len(batch['input_sparse_sdf_idx_size'])):
            r += batch['input_sparse_sdf_idx_size'][j]
            vcj, vcj_occ = vc[l:r], vc_occ[l:r]
            vl += voroloss(gt_clouds[j].unsqueeze(0), vcj.unsqueeze(0)).sum()
            l += batch['input_sparse_sdf_idx_size'][j]

            vvj = VoronoiValues(vcj.cpu().numpy(), torch.argmax(vcj_occ, dim=1).cpu().numpy())
            vvj.clean_useless_generators()
            torch.save(vvj, '{}/test_{}.pt'.format(config['path2samples'], obj_ind + j))
            # voro_v, voro_f, voro_err = vvj.to_mesh(return_errors=True)
            voro_v, voro_f = get_clean_shape(vvj, batch['input_sdf'][j, 0].numpy(),
                                             mask_scale=2, grid_s=batch['input_gs'][j].numpy() + 1)

            # is_good_edges = percentage_valid_edges(voro_f)

            if config['sample_rescale']:
                voro_v /= config['sample_rescale_scale']

            export_obj(voro_v, voro_f, '{}/test_{}'.format(config['path2samples'], obj_ind + j))
        vl /= len(batch['input_sparse_sdf_idx_size'])

        batch_time.update(time() - end)
        # VL.update(vl.item(), batch['input_sdf'].shape[0])
        VL.update(vl, batch['input_gs'].shape[0])
        CE.update(ce, batch['input_gs'].shape[0])

        # if (i + 1) % (config['num_workers'] // 2) == 0:
        line = '[{0}/{1}]'.format(i + 1, len(iterator))
        line += '\tData {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            data_time=data_time)
        line += '\tInference {inf_time.val:.3f} ({inf_time.avg:.3f})'.format(
            inf_time=inf_time)
        line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            batch_time=batch_time)
        line += '\tVoroloss {VL.val:.3f} ({VL.avg:.3f})'.format(VL=VL)
        line += '\tCE {CE.val:.3f} ({CE.avg:.3f})'.format(CE=CE)
        tqdm.write(line)

        obj_ind += len(batch['input_sparse_sdf_idx_size'])
        end = time()


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    with io.open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.Loader)
    print('Config file loaded.')

    device = torch.device(args.gpu)
    torch.cuda.set_device(device)
    torch.multiprocessing.set_sharing_strategy('file_system')

    input_pc_transform = ComposeCloudTransformation(**config)
    sample_transform = ComposeSampleTransformation(**config)
    train_dataset = ABCDataset(
        config['tr_obj_dir'], config['tr_surf_dir'], config['tr_sdf_dir'], config['tr_vgocc_dir'], val=False,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        return_sdf_grid=True, return_gt_mesh=True, return_vgocc=True,
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform
    )
    eval_dataset = ABCDataset(
        config['val_obj_dir'], config['val_surf_dir'], config['val_sdf_dir'], config['val_vgocc_dir'], val=True,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        return_sdf_grid=True, return_gt_mesh=True, return_vgocc=True,
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform,
    )
    print('Datasets init: done.')

    train_iterator = DataLoader(
        train_dataset, collate_fn=abc_collate_fn,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], drop_last=False
    )
    eval_iterator = DataLoader(
        eval_dataset, collate_fn=abc_collate_fn,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], drop_last=False
    )
    print('Iterator init: done.')

    encoder = LocalSDFFE_ME(
        config['in_dim'], config['enc_feat_dim']).to(device)
    if config['vcdec_in_glob']:
        maxpooler = ME.MinkowskiGlobalMaxPooling().to(device)
        glob_encoder = GlobFeatEnc(
            config['enc_feat_dim'][-1], [config['enc_feat_dim'][-1], config['enc_feat_dim'][-1]]).to(device)
    vcdec = VCDec(config['vcdec_p_dim'], config['enc_feat_dim'][-1], config['vcdec_feat_dim'], in_glob=config['vcdec_in_glob'],
                  vc_per_query=config['vcdec_vc_per_query'], vc_tanh=config['vcdec_vc_tanh'],
                  pe=config['vcdec_pe'], pe_feat_dim=config['vcdec_pe_feat_dim'],
                  film=config['vcdec_film'], film_std=config['vcdec_film_std']).to(device)
    occdec = VCOccDec(config['vcdec_p_dim'], config['enc_feat_dim'][-1], config['vcdec_feat_dim'], in_glob=config['vcdec_in_glob'],
                      pe=config['vcdec_pe'], pe_feat_dim=config['vcdec_pe_feat_dim'],
                      film=config['vcdec_film'], film_std=config['vcdec_film_std']).to(device)
    print('Model init: done.')
    print('Total number of parameters in encoder: {}'.format(
        cnt_params(encoder.parameters())))
    if config['vcdec_in_glob']:
        print('Total number of parameters in global encoder: {}'.format(
            cnt_params(glob_encoder.parameters())))
    print('Total number of parameters in vgdec: {}'.format(
        cnt_params(vcdec.parameters())))
    print('Total number of parameters in occdec: {}'.format(
        cnt_params(occdec.parameters())))

    voroloss = Voroloss_opt().to(device)
    celoss = torch.nn.CrossEntropyLoss().to(device)

    path2checkpoint = os.path.join(config['path2save'], config['model_name'])
    checkpoint = torch.load(path2checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state'])
    if config['vcdec_in_glob']:
        glob_encoder.load_state_dict(checkpoint['glob_encoder_state'])
    vcdec.load_state_dict(checkpoint['vcdec_state'])
    occdec.load_state_dict(checkpoint['occdec_state'])
    del (checkpoint)
    print('Model {} loaded.'.format(path2checkpoint))

    if not os.path.exists(config['path2samples']):
        os.makedirs(config['path2samples'])

    evaluate_and_reconstruct(eval_dataset, eval_iterator)
