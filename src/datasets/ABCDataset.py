from itertools import product
from datetime import datetime
from glob import glob
from pathlib import Path
import os
import collections
import numpy as np
import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NON_WT_TRAIN = [
    '00000908', '00000500', '00002737', '00003763', '00003694', '00003695', '00003181', '00003107', '00003773', '00004062',
    '00004268', '00004270', '00000922', '00004956', '00003770', '00005146', '00000653', '00003864', '00001650', '00006952',
    '00001504', '00007490', '00003860', '00006842', '00007445', '00005927', '00004817', '00006112', '00004917', '00007047',
    '00007837', '00007886', '00005248', '00002742', '00003932', '00002981', '00004322', '00005679', '00005372', '00001993',
    '00004756', '00005981', '00007557', '00003950', '00006752', '00000944', '00004424', '00007395', '00004930', '00005732',
    '00006424', '00001330', '00004081', '00004197', '00001835', '00005615', '00002663', '00005495', '00003092', '00005609',
    '00005850', '00002126', '00003342', '00005926', '00005799', '00005365', '00006407', '00006803', '00007097', '00005441',
    '00007364', '00006276', '00006409', '00005373', '00001644', '00006630', '00005740', '00001272', '00000069', '00006453',
    '00001060', '00003002', '00000515', '00002284', '00000029', '00005239', '00002876', '00002603', '00007363', '00007606',
    '00003734', '00000186', '00005812', '00002872', '00000931', '00000941', '00007747', '00002549', '00003986', '00002527',
    '00005997', '00003351', '00006443', '00005404', '00002488', '00003824', '00003896', '00002122', '00007449', '00004083',
    '00000733', '00004534', '00007136', '00004399', '00002017', '00001498', '00007726', '00002874', '00000756', '00006802',
    '00005867', '00005811', '00002285', '00002932', '00004306', '00007485', '00006434', '00003159', '00007385', '00007903',
    '00003158', '00002875', '00003895', '00000127', '00000742', '00004948', '00004441', '00006235', '00004160', '00007835',
    '00007392', '00006825', '00000744', '00001806', '00005668', '00003690', '00002534', '00005067', '00004290', '00003524',
    '00002459', '00001625', '00001992', '00000965', '00000213', '00007279', '00004264', '00000428', '00002018', '00007722',
    '00006129', '00001917', '00007234', '00002864', '00007709', '00005412', '00005870', '00005452', '00005123', '00005411',
    '00004593', '00006373', '00006809', '00007229', '00002873', '00007487', '00000114', '00003179', '00006314', '00007771',
    '00004483', '00002871', '00004942', '00006869', '00000053', '00002460', '00005307', '00006316', '00006648', '00007230',
    '00004949', '00000349', '00006559', '00000074', '00001018', '00000347', '00002219', '00003312', '00006798', '00000970',
    '00005366', '00001312', '00006985', '00000350', '00006799', '00003875', '00007564', '00002388', '00006130', '00000036',
    '00005430', '00000772', '00004910', '00005458', '00006793', '00000080', '00002882', '00005700', '00006485', '00007223',
    '00002566', '00003125', '00000833', '00003154', '00000016', '00007586', '00006256', '00005843', '00001661', '00006210',
    '00001308', '00006376', '00000771', '00004818', '00007224', '00005059', '00004190', '00002879', '00000720', '00001343',
    '00007013', '00006548', '00003142', '00002374', '00002426', '00000081', '00003961', '00000641', '00003018', '00007092',
    '00006147', '00005592', '00006478', '00004269', '00001457', '00004173', '00007708', '00005462', '00000642', '00006957',
    '00005518', '00006695', '00005121', '00002020', '00001270', '00002208', '00004097', '00003667', '00001815', '00000363',
    '00000567', '00007649', '00003754', '00002114', '00001477', '00002880', '00002396', '00006431', '00006650', '00006068',
    '00000340', '00006815', '00004174', '00004271', '00004014', '00003016', '00000015', '00002482', '00006083', '00001580',
    '00001115', '00000600', '00002878', '00007167', '00001384', '00001329', '00001302', '00002159', '00003343', '00006142',
    '00006479', '00003796', '00000804', '00007070', '00003982', '00001223', '00004193', '00002870', '00000089', '00007388',
    '00002869', '00004983', '00000495', '00000828', '00000512', '00000652', '00001840', '00000824', '00004912', '00000716',
    '00007711', '00006791', '00007650', '00001458', '00005176', '00000278', '00003481', '00002956', '00002780', '00007434',
    '00003354', '00003432', '00006830', '00001771', '00000028', '00006576', '00000577', '00004114', '00004588', '00006961',
    '00001446', '00005621', '00001083', '00000504', '00002925', '00002867', '00005298', '00000706', '00006644', '00007133',
    '00000240', '00001128', '00007483', '00002541', '00004589', '00003716', '00002868', '00005892', '00000743', '00001922',
    '00004045', '00007403', '00002546', '00002636', '00000966', '00007163', '00000745', '00001154', '00007386', '00003846',
    '00003365', '00004213', '00003929', '00003023', '00003063', '00000303', '00005501', '00006331', '00007488', '00003632',
    '00004027', '00000947', '00002914', '00001337', '00006000', '00006579', '00006372', '00001304', '00004231', '00007412',
    '00001711', '00007396', '00006354', '00005699', '00000950', '00004168', '00007221', '00007418', '00005784', '00002245',
    '00004464', '00001328', '00000121', '00005333', '00000075', '00001176', '00003397', '00004243', '00006143', '00005773',
    '00007398', '00006552', '00005063', '00007074', '00007397', '00001274', '00000430', '00006298', '00007220', '00007222',
    '00002086', '00002085', '00007286', '00007419', '00006658', '00007215', '00007219', '00000739', '00007213', '00007216',
    '00001897', '00007218', '00005666', '00007211', '00002012', '00000017', '00007550'
]

NON_WT_VAL = [
    '00007904', '00007924', '00007990', '00008007', '00008052', '00008036', '00008054', '00008037', '00008040', '00008278',
    '00008279', '00008311', '00008320', '00008253', '00008295', '00008121', '00007923', '00008385', '00008404', '00008345',
    '00007925', '00008467', '00008444', '00008506', '00008465', '00008490', '00008312', '00008531', '00008534', '00008545',
    '00008577', '00008580', '00008612', '00008603', '00008470', '00008610', '00008626', '00008611', '00008627', '00008601',
    '00008596', '00008718', '00008829', '00008759', '00008837', '00008966', '00008535', '00008625', '00009084', '00008778',
    '00009091', '00009092', '00008999', '00009064', '00008670', '00009008', '00008974', '00009101', '00009110', '00009062',
    '00009162', '00009211', '00009208', '00009247', '00009262', '00009324', '00009319', '00009321', '00009354', '00009385',
    '00009380', '00009344', '00009435', '00009483', '00009482', '00009487', '00009518', '00009570', '00009533', '00009578',
    '00009563', '00009572', '00009619', '00009541', '00009622', '00009641', '00009626', '00009649', '00009616', '00009717',
    '00009701', '00009655', '00009715', '00009402', '00009667', '00009536', '00009711', '00008830', '00009789', '00009773',
    '00009769', '00009880', '00009881', '00009883', '00009882', '00009884', '00009861', '00009637', '00009143'
]


def sample_uniform_cloud(vertices_c, faces_vc, size=2**10):
    polygons = vertices_c[faces_vc]
    cross = np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])
    areas = np.sqrt((cross**2).sum(1)) / 2.0

    probs = areas / areas.sum()
    p_sample = np.random.choice(np.arange(polygons.shape[0]), size=size, p=probs)

    sampled_polygons = polygons[p_sample]
    s1 = np.random.random((size, 1)).astype(np.float32)
    s2 = np.random.random((size, 1)).astype(np.float32)
    cond = (s1 + s2) > 1.
    s1[cond] = 1. - s1[cond]
    s2[cond] = 1. - s2[cond]
    sample = (sampled_polygons[:, 0] +
              s1 * (sampled_polygons[:, 1] - sampled_polygons[:, 0]) +
              s2 * (sampled_polygons[:, 2] - sampled_polygons[:, 0])).astype(np.float32)
    return sample


class ABCDataset(Dataset):
    def __init__(self, obj_dir, surf_dir, sdf_dir, vgocc_dir, val=False,
                 input_type='sdf',
                 grid_size=[32], v_start=-0.5, v_end=0.5,
                 return_sdf_grid=False, return_gt_mesh=False, return_vgocc=False,
                 n_input_pc=4096, input_pc_transform=None,
                 knn_input=8, local_input=False, pooling_radius=2,
                 truncate_sdf=False, truncation_cell_dist=3, sparse_sdf=False,
                 sample_grid_points=True, sample_grid_size=4096,
                 sample_grid_add_noise=True, sample_grid_add_noise_scale=1.0,
                 sample_gt_pc=False, n_gt_pc=50000,
                 sample_transform=None, return_crust_generators=False,
                 exclude_nonwt_shapes=False):
        super(ABCDataset, self).__init__()
        self.obj_files = [i for i in sorted(glob(os.path.join(obj_dir, '*.obj')))]
        self.surf_files = [i for i in sorted(glob(os.path.join(surf_dir, '*.npy')))]

        self.vgocc_dir = vgocc_dir
        self.sdf_files = []
        self.vgocc_files = []
        for gs in grid_size:
            self.sdf_files += [os.path.join(sdf_dir + str(gs), Path(i).stem + '.npz') for i in self.surf_files]
            if vgocc_dir is not None:
                self.vgocc_files += [i for i in sorted(glob(os.path.join(vgocc_dir + str(gs), '*.npz')))]
            if len(self.vgocc_files) == 0:
                self.vgocc_dir = None

        self.obj_files *= len(grid_size)
        self.surf_files *= len(grid_size)
        assert \
            ((len(self.obj_files) != 0) and (len(self.surf_files) != 0) and (len(self.sdf_files) !=0)),\
            "No objects in the data directory"

        if exclude_nonwt_shapes:
            k = 0
            for i in range(len(self.obj_files)):
                if self.obj_files[-(i - k + 1)][-12:-4] in (NON_WT_TRAIN + NON_WT_VAL):
                    del self.obj_files[-(i - k + 1)]
                    del self.surf_files[-(i - k + 1)]
                    del self.sdf_files[-(i - k + 1)]
                    del self.vgocc_files[-(i - k + 1)]
                    k += 1

        self.num_instances = len(self.obj_files)
        self.val = val

        self.input_type = input_type

        self.grid_size = grid_size
        self.v_start = v_start
        self.v_end = v_end

        self.return_sdf_grid = return_sdf_grid
        self.return_gt_mesh = return_gt_mesh
        self.return_vgocc = return_vgocc

        self.n_input_pc = n_input_pc
        self.input_pc_transform = input_pc_transform

        self.knn_input = knn_input
        self.local_input = local_input
        self.pooling_radius = pooling_radius

        self.truncate_sdf = truncate_sdf
        self.truncation_cell_dist = truncation_cell_dist
        self.sparse_sdf = sparse_sdf

        self.sample_grid_points = sample_grid_points
        self.sample_grid_size = sample_grid_size
        self.sample_grid_add_noise = sample_grid_add_noise
        self.sample_grid_add_noise_scale = sample_grid_add_noise_scale

        self.n_gt_pc = n_gt_pc

        self.sample_transform = sample_transform
        self.return_crust_generators = return_crust_generators

        self.exclude_nonwt_shapes = exclude_nonwt_shapes

    def __len__(self):
        return self.num_instances

    def to_int_coord(self, pc_xyz, v_start=-0.5, v_end=0.5, grid_size=32):
        min_val = -1 * v_start
        max_val = v_end - v_start
        return np.floor((grid_size * ((pc_xyz + min_val) / max_val))).astype(np.int32)

    def to_xyz_coord(self, pc_int, v_start=-0.5, v_end=0.5, grid_size=32):
        min_val = -1 * v_start
        max_val = v_end - v_start
        return (max_val * (pc_int / grid_size) - min_val).astype(np.float32)

    def get_pc_features(self, pc_xyz):
        kd_tree = KDTree(pc_xyz)
        _, pc_KNN_idx = kd_tree.query(pc_xyz, k=self.knn_input)
        pc_KNN_idx = pc_KNN_idx.reshape(-1)
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        if self.local_input:
            pc_KNN_xyz = pc_KNN_xyz.reshape(-1, self.knn_input, 3) - pc_xyz.reshape(-1, 1, 3)
            pc_KNN_xyz = pc_KNN_xyz.reshape(-1, 3)
        return pc_KNN_xyz, pc_KNN_idx

    def get_grid_features(self, pc_xyz, grid_size=32, pooling_radius=2):
        kd_tree = KDTree(pc_xyz)
        pc_int = self.to_int_coord(pc_xyz, v_start=self.v_start, v_end=self.v_end, grid_size=grid_size)
        
        if pooling_radius == -1:
            grid_int = np.stack(np.meshgrid(
                np.arange(grid_size + 1),
                np.arange(grid_size + 1),
                np.arange(grid_size + 1),
                indexing='ij'
            ), axis=-1).reshape(-1, 3)
            grid_xyz = np.stack(np.meshgrid(
                np.linspace(self.v_start, self.v_end, num=grid_size + 1),
                np.linspace(self.v_start, self.v_end, num=grid_size + 1),
                np.linspace(self.v_start, self.v_end, num=grid_size + 1),
                indexing='ij'
            ), axis=-1).reshape(-1, 3)
        else:
            grid_int = set()
            nbh = np.arange(-pooling_radius, pooling_radius + 1)
            for i, j, k in product(nbh, nbh, nbh):
                grid_int |= set(map(tuple, np.clip(pc_int + np.array([[i, j, k]]), 0, grid_size)))
            grid_int = np.array(list(grid_int))
            grid_xyz = self.to_xyz_coord(grid_int, v_start=self.v_start, v_end=self.v_end, grid_size=grid_size)

        _, grid_KNN_idx = kd_tree.query(grid_xyz, k=self.knn_input)
        grid_KNN_idx = grid_KNN_idx.reshape(-1)
        grid_KNN_xyz = pc_xyz[grid_KNN_idx]
        if self.local_input:
            grid_KNN_xyz = grid_KNN_xyz.reshape(-1, self.knn_input, 3) - grid_xyz.reshape(-1, 1, 3)
            grid_KNN_xyz = grid_KNN_xyz.reshape(-1, 3)
        return grid_KNN_xyz, grid_KNN_idx, grid_int

    def __getitem__(self, index):
        surf_obj = self.obj_files[index]
        surf_npy = self.surf_files[index]
        sdf_npz = self.sdf_files[index]
        cur_gs = int(sdf_npz.split('/')[-2][3:])
        sample = {}

        meshin = trimesh.load_mesh(surf_obj)
        vertices = np.array(meshin.vertices)
        faces = np.array(meshin.faces)

        if self.return_gt_mesh:
            sample['gt_mesh_vertices'] = np.float32(vertices)
            sample['gt_mesh_faces'] = faces

        if self.return_vgocc and self.vgocc_dir is not None:
            vgocc_npz = self.vgocc_files[index]
            sample['gt_vg_occ'] = np.load(vgocc_npz)['gt_occ'].astype(np.int64)

        if self.input_type == 'sdf':
            sample['input_gs'] = np.array(cur_gs, dtype=np.int)

            grid_xyz = np.stack(np.meshgrid(
                np.linspace(self.v_start, self.v_end, num=cur_gs + 1),
                np.linspace(self.v_start, self.v_end, num=cur_gs + 1),
                np.linspace(self.v_start, self.v_end, num=cur_gs + 1),
                indexing='ij'
            ), axis=-1)
            grid_sdf = np.load(sdf_npz)['sdf_vox']
            if self.return_sdf_grid:
                sample['input_xyz'] = np.expand_dims(np.float32(grid_xyz).copy(), 0)
                sample['input_sdf'] = np.expand_dims(np.float32(grid_sdf).copy(), 0)

            if self.return_crust_generators:
                thinb = np.float32(np.sqrt(3) / cur_gs)
                thinc = (np.fabs(grid_sdf) < thinb).astype(int)
                with torch.no_grad():
                    outerc = F.conv3d(
                        torch.from_numpy(thinc).unsqueeze(0).unsqueeze(0),
                        weight=torch.ones((1, 1, 3, 3, 3), dtype=torch.long), padding='same'
                    )
                outerc = (outerc.numpy() > 0).astype(int) - thinc
                outerc_idx = np.stack((outerc.nonzero()), axis=-1).astype(int)
                sample['crust_vg_size'] = np.array(outerc_idx.shape[0])
                sample['crust_vg'] = grid_xyz[outerc_idx[:, 0], outerc_idx[:, 1], outerc_idx[:, 2]]
                sample['crust_vg_occ'] = (grid_sdf[outerc_idx[:, 0], outerc_idx[:, 1], outerc_idx[:, 2]] < 0).astype(int)

            if self.truncate_sdf:
                sample['min_dist'] = np.float32(-(self.truncation_cell_dist * np.sqrt(3)) / cur_gs)
                sample['max_dist'] = np.float32( (self.truncation_cell_dist * np.sqrt(3)) / cur_gs)
                neg_mask = grid_sdf < sample['min_dist']
                pos_mask = grid_sdf > sample['max_dist']
                grid_sdf[neg_mask] = sample['min_dist']
                grid_sdf[pos_mask] = sample['max_dist']

            if self.input_type == 'sdf' and self.sparse_sdf:
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

        if self.val:
            opc = np.load(surf_npy)[:self.n_gt_pc, :3]
        else:
            opc = sample_uniform_cloud(vertices, faces, size=self.n_gt_pc)
        sample['gt_cloud_glob'] = np.float32(opc)

        if self.sample_transform:
            sample = self.sample_transform(sample)

        return sample


def abc_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return abc_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, collections.abc.Mapping):
        output = {}
        for key in elem:
            if key == 'gt_mesh_vertices':
                output[key] = [abc_collate_fn(d[key]) for d in batch]
            elif key == 'gt_mesh_faces':
                output[key] = [abc_collate_fn(d[key]) for d in batch]
            elif key == 'gt_vg_occ':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'crust_vg':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'crust_vg_occ':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'input_cloud_loc_idx':
                output[key] = abc_collate_fn([d[key] + i * d['input_cloud_glob'].shape[0] for i, d in enumerate(batch)]).view(-1)
            elif key == 'input_cloud_grid_loc':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'input_cloud_grid_loc_idx':
                output[key] = torch.cat([
                    abc_collate_fn(d[key] + i * d['input_cloud_glob'].shape[0]) for i, d in enumerate(batch)
                ], dim=0)
            elif key == 'input_cloud_grid_int':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
                output['input_cloud_grid_int_size'] = torch.cat([
                    abc_collate_fn(np.array([d[key].shape[0]])) for d in batch
                ], dim=0)
            elif key == 'input_sparse_xyz':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'input_sparse_sdf':
                output[key] = torch.cat([abc_collate_fn(d[key]) for d in batch], dim=0)
            elif key == 'input_sparse_sdf_idx':
                output[key] = torch.cat([abc_collate_fn(
                    np.hstack((i * np.ones((d[key].shape[0], 1), dtype=np.int32), d[key]))
                ) for i, d in enumerate(batch)], dim=0)
            else:
                output[key] = abc_collate_fn([d[key] for d in batch])

        return output
