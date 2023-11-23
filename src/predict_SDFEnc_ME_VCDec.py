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

from networks.sdf_encoders import LocalSDFFE_ME, GlobFeatEnc
from networks.vc_decoders import VCDec, VCOccDec
from utils.networks import cnt_params, AverageMeter
from utils.custom_voronoi import Voroloss_opt, VoronoiValues, get_clean_shape


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str,
                        help='Path to config file in YAML format.')
    parser.add_argument('path', type=str,
                        help='Path to SDF file.')
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


def reconstruct(sample):
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
    torch.cuda.empty_cache()

    input_sparse_xyz = sample['input_sparse_xyz'].to(device)
    input_sparse_sdf = (sample['input_sparse_sdf'].to(device) * sample['input_gs'].to(device) / 64.).unsqueeze(1)
    input_sparse_sdf_idx = torch.cat(
        [torch.zeros((sample['input_sparse_sdf_idx'].shape[0], 1)),
         sample['input_sparse_sdf_idx']], dim=1
    ).to(device).int()

    input_sdf = ME.SparseTensor(input_sparse_sdf, input_sparse_sdf_idx)
    grid_features = encoder(input_sdf)

    if config['vcdec_in_glob'] and config['vcdec_p_dim']:
        vc = vcdec(
            grid_features.features,
            query=input_sparse_xyz,
            g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                               sample['input_sparse_sdf_idx_size'].to(device),
                                               dim=0)
        )
    elif config['vcdec_in_glob']:
        vc = vcdec(
            grid_features.features,
            g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                               sample['input_sparse_sdf_idx_size'].to(device),
                                               dim=0)
        )
    elif config['vcdec_p_dim']:
        vc = vcdec(grid_features.features, query=input_sparse_xyz)
    else:
        vc = vcdec(grid_features.features)
    vc /= sample['input_gs'].to(device)

    if config['vcdec_vc_tanh']:
        vc *= config['vcdec_vc_tanh_scale']

    vc += input_sparse_xyz

    if config['vcdec_in_glob'] and config['vcdec_p_dim']:
        vc_occ = occdec(
            grid_features.features, query=vc,
            g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                               sample['input_sparse_sdf_idx_size'].to(device),
                                               dim=0)
        )
    elif config['vcdec_in_glob']:
        vc_occ = occdec(
            grid_features.features,
            g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                               sample['input_sparse_sdf_idx_size'].to(device),
                                               dim=0)
        )
    elif config['vcdec_p_dim']:
        vc_occ = occdec(grid_features.features, query=vc)
    else:
        vc_occ = occdec(grid_features.features)

    vv = VoronoiValues(vc.cpu().numpy(), torch.argmax(vc_occ, dim=1).cpu().numpy())
    vv.clean_useless_generators()
    torch.save(vv, '{}/test_sample.pt'.format(config['path2samples']))
    voro_v, voro_f = get_clean_shape(vv, sample['input_sdf'][0].numpy(),
                                     mask_scale=2, grid_s=sample['input_gs'].numpy() + 1)

    if config['sample_rescale']:
        voro_v /= config['sample_rescale_scale']

    export_obj(voro_v, voro_f, '{}/test_sample'.format(config['path2samples']))


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    with io.open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.Loader)
    print('Config file loaded.')

    device = torch.device(args.gpu)
    torch.cuda.set_device(device)
    torch.multiprocessing.set_sharing_strategy('file_system')

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

    torchify = lambda x: torch.as_tensor(x)

    grid_sdf = np.load(args.path)['sdf_vox']
    print(grid_sdf.max(), grid_sdf.min())
    grid_xyz = np.stack(np.meshgrid(
        np.linspace(config['v_start'], config['v_end'], num=grid_sdf.shape[-1]),
        np.linspace(config['v_start'], config['v_end'], num=grid_sdf.shape[-1]),
        np.linspace(config['v_start'], config['v_end'], num=grid_sdf.shape[-1]),
        indexing='ij'
    ), axis=-1)

    sample = {}
    sample['input_gs'] = grid_sdf.shape[-1] - 1
    sample['input_xyz'] = np.expand_dims(np.float32(grid_xyz).copy(), 0)
    sample['input_sdf'] = np.expand_dims(np.float32(grid_sdf).copy(), 0)
    sample['min_dist'] = np.float32(-(config['truncation_cell_dist'] * np.sqrt(3)) / (grid_sdf.shape[-1] - 1))
    sample['max_dist'] = np.float32( (config['truncation_cell_dist'] * np.sqrt(3)) / (grid_sdf.shape[-1] - 1))
    neg_mask = grid_sdf < sample['min_dist']
    pos_mask = grid_sdf > sample['max_dist']
    grid_sdf[neg_mask] = 0.
    grid_sdf[pos_mask] = 0.
    sample['input_sparse_sdf_idx'] = np.stack((grid_sdf.nonzero()), axis=-1).astype(int)
    sample['input_sparse_sdf_idx_size'] = np.array(sample['input_sparse_sdf_idx'].shape[0])
    sample['input_sparse_xyz'] = np.float32(grid_xyz[
        sample['input_sparse_sdf_idx'][:, 0],
        sample['input_sparse_sdf_idx'][:, 1],
        sample['input_sparse_sdf_idx'][:, 2],
    ])
    sample['input_sparse_sdf'] = np.float32(grid_sdf[
        sample['input_sparse_sdf_idx'][:, 0],
        sample['input_sparse_sdf_idx'][:, 1],
        sample['input_sparse_sdf_idx'][:, 2],
    ])

    if config['sample_rescale']:
        for key in sample.keys():
            if key in ['min_dist', 'max_dist', 'input_sparse_xyz', 'input_sparse_sdf']:
                sample[key] *= config['sample_rescale_scale']

    for key, value in sample.items():
        sample[key] = torchify(value)

    reconstruct(sample)
