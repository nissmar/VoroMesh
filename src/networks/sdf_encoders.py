from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


class conv_resnet_block(nn.Module):
    def __init__(self, feat_dim):
        super(conv_resnet_block, self).__init__()
        self.feat_dim = feat_dim
        self.rc = nn.Sequential(OrderedDict([
            ('conv_0_bn', nn.BatchNorm3d(self.feat_dim)),
            ('conv_0_lrelu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ('conv_1', nn.Conv3d(self.feat_dim, self.feat_dim, 3, stride=1, padding=1, bias=False)),
            ('conv_1_bn', nn.BatchNorm3d(self.feat_dim)),
            ('conv_1_lrelu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ('conv_2', nn.Conv3d(self.feat_dim, self.feat_dim, 1, stride=1, padding=0, bias=False)),
        ]))
        self.lrelu = nn.Sequential(OrderedDict([
            ('conv_2_bn', nn.BatchNorm3d(self.feat_dim)),
            ('conv_2_lrelu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
        ]))

    def forward(self, input):
        return self.lrelu(input + self.rc(input))


class LocalSDFFE(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(LocalSDFFE, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim

        self.features = nn.Sequential(OrderedDict([]))
        for i in range(len(self.feat_dim)):
        	cur_dim = self.in_dim if i == 0 else self.feat_dim[i - 1]
        	next_dim = self.feat_dim[i]
        	self.features.append(nn.Conv3d(cur_dim, next_dim, 3, stride=1, padding=1, bias=False))
        	self.features.append(conv_resnet_block(next_dim))
        self.features.append(nn.Conv3d(next_dim, next_dim, 3, stride=1, padding=1, bias=False))

    def forward(self, sdf):
        return self.features(sdf)


class conv_resnet_me_block(nn.Module):
    def __init__(self, feat_dim):
        super(conv_resnet_me_block, self).__init__()
        self.feat_dim = feat_dim
        self.rc = nn.Sequential(OrderedDict([
            ('conv_0_bn', ME.MinkowskiBatchNorm(self.feat_dim)),
            ('conv_0_prelu', ME.MinkowskiPReLU(init=0.01)),
            ('conv_1', ME.MinkowskiConvolution(self.feat_dim, self.feat_dim, kernel_size=3, stride=1, bias=False, dimension=3)),
            ('conv_1_bn', ME.MinkowskiBatchNorm(self.feat_dim)),
            ('conv_1_prelu', ME.MinkowskiPReLU(init=0.01)),
            ('conv_2', ME.MinkowskiConvolution(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, bias=False, dimension=3))
        ]))
        self.prelu = nn.Sequential(OrderedDict([
            ('conv_2_bn', ME.MinkowskiBatchNorm(self.feat_dim)),
            ('conv_2_prelu', ME.MinkowskiPReLU(init=0.01))
        ]))

    def forward(self, input):
        return self.prelu(input + self.rc(input))


class LocalSDFFE_ME(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(LocalSDFFE_ME, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim

        self.features = nn.Sequential(OrderedDict([]))
        for i in range(len(self.feat_dim)):
        	cur_dim = self.in_dim if i == 0 else self.feat_dim[i - 1]
        	next_dim = self.feat_dim[i]
        	self.features.append(ME.MinkowskiConvolution(cur_dim, next_dim, kernel_size=3, stride=1, bias=False, dimension=3))
        	self.features.append(conv_resnet_me_block(next_dim))
        self.features.append(ME.MinkowskiConvolution(next_dim, next_dim, kernel_size=3, stride=1, bias=False, dimension=3))

    def forward(self, sdf):
        return self.features(sdf)


class GlobFeatEnc(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(GlobFeatEnc, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim

        self.features = nn.Sequential(OrderedDict([]))
        for i in range(len(self.feat_dim)):
            cur_dim = self.in_dim if i == 0 else self.feat_dim[i - 1]
            next_dim = self.feat_dim[i]
            self.features.append(nn.BatchNorm1d(cur_dim))
            self.features.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            self.features.append(nn.Linear(cur_dim, next_dim, bias=True if i == len(self.feat_dim) - 1 else False))

    def forward(self, input):
        return self.features(input)