from networks.vc_decoders import VCDec, VCOccDec
from networks.sdf_encoders import LocalSDFFE_ME, GlobFeatEnc
from utils.networks import cnt_params, AverageMeter, save_model
from datasets.sample_transformation import ComposeSampleTransformation
from datasets.cloud_transformation import ComposeCloudTransformation
from datasets.ABCDataset import ABCDataset, abc_collate_fn
import MinkowskiEngine as ME
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from time import time
import yaml
import io
import argparse
import os
os.environ['OMP_NUM_THREADS'] = '16'


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str,
                        help='Path to config file in YAML format.')
    parser.add_argument('--n_epochs', type=int, default=40,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float,
                        default=0.000256, help='Learning rate.')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Second moment weihght in adam optimizer.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--resume', action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')
    return parser


def training_step():
    data_time = AverageMeter()
    batch_time = AverageMeter()
    CE = AverageMeter()

    encoder.eval()
    if config['vcdec_in_glob']:
        glob_encoder.eval()
    vcdec.eval()
    occdec.train()
    torch.set_grad_enabled(True)

    end = time()
    for i, batch in enumerate(train_iterator):
        torch.cuda.empty_cache()
        if i >= len(train_iterator):
            break
        data_time.update(time() - end)
        end = time()

        with torch.no_grad():
            input_sparse_xyz = batch['input_sparse_xyz'].to(device)
            input_sparse_sdf = (
                batch['input_sparse_sdf'].to(device) *
                torch.repeat_interleave(batch['input_gs'].to(device) / 32.,
                                        batch['input_sparse_sdf_idx_size'].to(
                                            device),
                                        dim=0)
            ).unsqueeze(1)
            input_sparse_sdf_idx = batch['input_sparse_sdf_idx'].to(
                device).int()
            input_sdf = ME.SparseTensor(input_sparse_sdf, input_sparse_sdf_idx)
            grid_features = encoder(input_sdf)

            if config['vcdec_in_glob'] and config['vcdec_p_dim']:
                vc = vcdec(
                    grid_features.features,
                    query=input_sparse_xyz,
                    g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                       batch['input_sparse_sdf_idx_size'].to(
                                                           device),
                                                       dim=0)
                )
            elif config['vcdec_in_glob']:
                vc = vcdec(
                    grid_features.features,
                    g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                       batch['input_sparse_sdf_idx_size'].to(
                                                           device),
                                                       dim=0)
                )
            elif config['vcdec_p_dim']:
                vc = vcdec(grid_features.features, query=input_sparse_xyz)
            else:
                vc = vcdec(grid_features.features)
            vc /= torch.repeat_interleave(batch['input_gs'].to(
                device), batch['input_sparse_sdf_idx_size'].to(device), dim=0).unsqueeze(1)

            if config['vcdec_vc_tanh']:
                vc *= config['vcdec_vc_tanh_scale']

            vc += input_sparse_xyz

        if config['vcdec_in_glob'] and config['vcdec_p_dim']:
            vc_occ = occdec(
                grid_features.features, query=vc,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(
                                                       device),
                                                   dim=0)
            )
        elif config['vcdec_in_glob']:
            vc_occ = occdec(
                grid_features.features,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(
                                                       device),
                                                   dim=0)
            )
        elif config['vcdec_p_dim']:
            vc_occ = occdec(grid_features.features, query=vc)
        else:
            vc_occ = occdec(grid_features.features)

        loss = celoss(vc_occ, batch['gt_vg_occ'].to(device))
        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time() - end)
        CE.update(loss.item(), batch['input_gs'].shape[0])

        # if (i + 1) % (config['num_workers'] // 2) == 0:
        line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1,
                                              i + 1, len(train_iterator))
        line += '\tData {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            data_time=data_time)
        line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            batch_time=batch_time)
        line += '\tCE {CE.val:.3f} ({CE.avg:.3f})'.format(CE=CE)
        tqdm.write(line)

        end = time()


def evaluation_step():
    pass


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    with io.open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.Loader)
    config['resume'] = True if args.resume else False
    config['resume_optimizer'] = True if args.resume_optimizer else False
    print('Config file loaded.')

    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    input_pc_transform = ComposeCloudTransformation(**config)
    sample_transform = ComposeSampleTransformation(**config)
    train_dataset = ABCDataset(
        config['tr_obj_dir'], config['tr_surf_dir'], config['tr_sdf_dir'], config['tr_vgocc_dir'], val=False,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        return_vgocc=True,
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform,
        exclude_nonwt_shapes=True
    )
    eval_dataset = ABCDataset(
        config['val_obj_dir'], config['val_surf_dir'], config['val_sdf_dir'], config['val_vgocc_dir'], val=True,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        return_gt_mesh=True, return_vgocc=True,
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform,
        exclude_nonwt_shapes=True
    )
    print('Datasets init: done.')

    train_iterator = DataLoader(
        train_dataset, collate_fn=abc_collate_fn,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], drop_last=True
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

    celoss = torch.nn.CrossEntropyLoss().to(device)

    optimizer = AdamW([{'params': occdec.parameters()}],
                      lr=config['lr'], weight_decay=config['wd'],
                      betas=(config['beta1'], config['beta2']), amsgrad=config['amsgrad'])
    print('Optimizer init: done.')

    if not config['resume']:
        path2checkpoint = os.path.join(
            config['path2save'], config['model_name'])
        checkpoint = torch.load(path2checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state'])
        if config['vcdec_in_glob']:
            glob_encoder.load_state_dict(checkpoint['glob_encoder_state'])
        vcdec.load_state_dict(checkpoint['vcdec_state'])
        voroloss_epoch = checkpoint['voroloss_epoch']
        voroloss_opt_state = checkpoint['voroloss_optimizer_state']
        del (checkpoint)
        print('Model {} loaded.'.format(path2checkpoint))
        cur_epoch = 0
        cur_iter = 0
    else:
        path2checkpoint = os.path.join(
            config['path2save'], config['model_name'])
        checkpoint = torch.load(path2checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state'])
        if config['vcdec_in_glob']:
            glob_encoder.load_state_dict(checkpoint['glob_encoder_state'])
        vcdec.load_state_dict(checkpoint['vcdec_state'])
        occdec.load_state_dict(checkpoint['occdec_state'])
        voroloss_epoch = checkpoint['voroloss_epoch']
        voroloss_opt_state = checkpoint['voroloss_optimizer_state']
        cur_epoch = checkpoint['ce_epoch']
        if config['resume_optimizer']:
            optimizer.load_state_dict(checkpoint['ce_optimizer_state'])
        del (checkpoint)
        print('Model {} loaded.'.format(path2checkpoint))

    for epoch in tqdm(range(cur_epoch, config['n_epochs']), initial=cur_epoch, total=config['n_epochs']):
        training_step()

        save_model({
            'encoder_state': encoder.state_dict(),
            'glob_encoder_state': glob_encoder.state_dict() if config['vcdec_in_glob'] else None,
            'vcdec_state': vcdec.state_dict(),
            'occdec_state': occdec.state_dict(),
            'voroloss_epoch': voroloss_epoch,
            'voroloss_optimizer_state': voroloss_opt_state,
            'ce_epoch': epoch + 1,
            'ce_optimizer_state': optimizer.state_dict()
        }, os.path.join(config['path2save'], config['model_name']))
        tqdm.write('Model saved to: {}'.format(
            os.path.join(config['path2save'], config['model_name'])))

        if (epoch + 1) % 5 == 0:
            evaluation_step()
