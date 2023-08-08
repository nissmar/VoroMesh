import numpy as np
from torchvision.transforms import Compose


class AddGaussianNoise(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('cloud_gaussian_noise_scale'))

    def __call__(self, cloud):
        cloud += np.random.normal(size=cloud,
                                  scale=self.scale).astype(np.float32)
        return cloud


def ComposeCloudTransformation(**kwargs):
    cloud_transformation = []
    if kwargs.get('cloud_gaussian_noise'):
        cloud_transformation.append(AddNoise2Cloud(**kwargs))

    if len(cloud_transformation) == 0:
        return None
    else:
        return Compose(cloud_transformation)
