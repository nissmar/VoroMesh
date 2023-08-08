import numpy as np
from torchvision.transforms import Compose


class Rescale(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('sample_rescale_scale', 2.0))

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['gt_mesh_vertices', 'gt_cloud_glob', 'min_dist', 'max_dist',
                       'input_sparse_xyz', 'input_sparse_sdf', 'crust_vg']:
                sample[key] *= self.scale
        return sample


class RandomAxesPermutation(object):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.get('in_dim', 3)

    def __call__(self, sample):
        order = np.arange(self.in_dim)
        np.random.shuffle(order)

        if 'input_cloud_glob' in sample:
            sample['input_cloud_glob'] = sample['input_cloud_glob'][:, order]
        if 'gt_cloud_glob' in sample:
            sample['gt_cloud_glob'] = sample['gt_cloud_glob'][:, order]
        if 'input_sdf' in sample:
            sample['input_sdf'] = np.transpose(sample['input_sdf'], axes=order)

        return sample


class RandomAxesFlip(object):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.get('in_dim', 3)

    def __call__(self, sample):
        sign = 2. * np.float32(np.random.rand(self.in_dim) > 0.5) - 1.
        mask = (sign < 0).nonzero()[0]

        if len(mask) > 0:
            if 'input_cloud_glob' in sample:
                sample['input_cloud_glob'] *= sign.reshape(1, -1)
            if 'gt_cloud_glob' in sample:
                sample['gt_cloud_glob'] *= sign.reshape(1, -1)
            if 'input_sdf' in sample:
                sample['input_sdf'] = np.flip(sample['input_sdf'], axis=mask).copy()

        return sample


def ComposeSampleTransformation(**kwargs):
    sample_transformation = []
    if kwargs.get('sample_rescale'):
        sample_transformation.append(Rescale(**kwargs))
    if kwargs.get('sample_axes_permutation'):
        sample_transformation.append(RandomAxesPermutation(**kwargs))
    if kwargs.get('sample_axes_flip'):
        sample_transformation.append(RandomAxesFlip(**kwargs))

    if len(sample_transformation) == 0:
        return None
    else:
        return Compose(sample_transformation)
