from collections import OrderedDict
from math import sqrt

import torch
import torch.nn as nn


class Sine(nn.Module):
  def __init__(self, w=30.0):
    super().__init__()
    self.w = w

  def forward(self, input):
    return torch.sin(self.w * input)

  def extra_repr(self):
    return 'w={}'.format(self.w)


class resnet_block_film(nn.Module):
    def __init__(self, feat_dim, film=False):
        super(resnet_block_film, self).__init__()
        self.feat_dim = feat_dim
        self.film = film

        self.rc = nn.Sequential(OrderedDict([
            ('rn_0_bn', nn.BatchNorm1d(feat_dim)),
            ('rn_0_lrelu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ('rn_1', nn.Linear(feat_dim, feat_dim, bias=False)),
            ('rn_1_bn', nn.BatchNorm1d(feat_dim)),
            ('rn_1_lrelu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ('rn_2', nn.Linear(feat_dim, feat_dim, bias=False)),
        ]))

        self.features = nn.BatchNorm1d(feat_dim, affine=False if film else True)

    def forward(self, input):
        return self.features(input + self.rc(input))


class VCDec(nn.Module):
    def __init__(self, in_dim, in_feat_dim, out_feat_dim, in_glob=False, vc_per_query=1, vc_tanh=True, pe=False, pe_feat_dim=64, pred_std=0.01, film=False, film_std=0.001):
        super(VCDec, self).__init__()
        self.in_dim = in_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.in_glob = in_glob
        self.vc_per_query = vc_per_query
        self.vc_tanh = vc_tanh
        self.pe_feat_dim = pe_feat_dim
        self.pe = None
        self.film = film
        self.film_w = None
        self.film_b = None
        self.pred_std = pred_std
        self.film_std = film_std

        if pe:
            self.pe = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(in_dim, pe_feat_dim, bias=True)),
                ('fc_1_sin', Sine()),
                ('fc_2', nn.Linear(pe_feat_dim, pe_feat_dim, bias=True)),
                ('fc_2_sin', Sine()),
            ]))

            with torch.no_grad():
                for i in range(len(self.pe) // 2):
                    self.pe[2 * i].weight.uniform_(-sqrt(6. / pe_feat_dim), sqrt(6. / pe_feat_dim))
                    nn.init.constant_(self.pe[2 * i].bias.data, 0.0)

        all_feat_dim = (pe_feat_dim if self.pe else in_dim) + (2 if in_glob else 1) * in_feat_dim

        self.first_fc = nn.Linear(all_feat_dim, out_feat_dim[0], bias=False)
        self.resnet = nn.ModuleList([
            resnet_block_film(out_feat_dim[i], film=self.film)
        for i in range(len(out_feat_dim))])
        self.fc = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc_{}_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('fc_{}'.format(i + 1), nn.Linear(out_feat_dim[i],
                								  out_feat_dim[i + 1] if i != (len(out_feat_dim) - 1) else 3 * self.vc_per_query,
                								  bias=False if i != (len(out_feat_dim) - 1) else True))
            ]))
        for i in range(len(out_feat_dim))])

        with torch.no_grad():
            self.fc[-1][-1].weight.normal_(std=pred_std)
            nn.init.constant_(self.fc[-1][-1].bias.data, 0.0)

        if self.film:
            self.film_w = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('film_w_{}_1'.format(i + 1), nn.Linear(all_feat_dim, out_feat_dim[i], bias=False)),
                    ('film_w_{}_1_bn'.format(i + 1), nn.BatchNorm1d(out_feat_dim[i])),
                    ('film_w_{}_1_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    ('film_w_{}_2'.format(i + 1), nn.Linear(out_feat_dim[i], out_feat_dim[i], bias=True)),
                ]))
            for i in range(len(self.resnet))])
            with torch.no_grad():
                for i in range(len(self.resnet)):
                    self.film_w[i][-1].weight.normal_(std=film_std)
                    nn.init.constant_(self.film_w[i][-1].bias.data, 0.0)

            self.film_b = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('film_b_{}_1'.format(i + 1), nn.Linear(all_feat_dim, out_feat_dim[i], bias=False)),
                    ('film_b_{}_1_bn'.format(i + 1), nn.BatchNorm1d(out_feat_dim[i])),
                    ('film_b_{}_1_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    ('film_b_{}_2'.format(i + 1), nn.Linear(out_feat_dim[i], out_feat_dim[i], bias=True)),
                ]))
            for i in range(len(self.resnet))])
            with torch.no_grad():
                for i in range(len(self.resnet)):
                    self.film_b[i][-1].weight.normal_(std=film_std)
                    nn.init.constant_(self.film_b[i][-1].bias.data, 0.0)

    def forward(self, l_features, query=None, g_features=None):
        ### Positional encoding with SIREN features ###
        if query is not None:
            pos = query
            if self.pe:
                pos = self.pe(2.0 * pos)

        if query is not None:
            if g_features is not None:
                features = torch.cat([pos, l_features, g_features], dim=1)
            else:
                features = torch.cat([pos, l_features], dim=1)
        else:
            if g_features is not None:
                features = torch.cat([l_features, g_features], dim=1)
            else:
                features = l_features

        out = self.first_fc(features)
        for i in range(len(self.resnet)):
            out = self.resnet[i](out)
            if self.film:
                out = torch.add(self.film_w[i](features), 1.0) * out + self.film_b[i](features)
            out = self.fc[i](out)

        if self.vc_tanh:
        	return torch.tanh(out)
        else:
        	return out


class VCOccDec(nn.Module):
    def __init__(self, in_dim, in_feat_dim, out_feat_dim, in_glob=False, pe=False, pe_feat_dim=64, film=False, film_std=0.001):
        super(VCOccDec, self).__init__()
        self.in_dim = in_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.in_glob = in_glob
        self.pe_feat_dim = pe_feat_dim
        self.pe = None
        self.film = film
        self.film_w = None
        self.film_b = None
        self.film_std = film_std

        if pe:
            self.pe = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(in_dim, pe_feat_dim, bias=True)),
                ('fc_1_sin', Sine()),
                ('fc_2', nn.Linear(pe_feat_dim, pe_feat_dim, bias=True)),
                ('fc_2_sin', Sine()),
            ]))

            with torch.no_grad():
                for i in range(len(self.pe) // 2):
                    self.pe[2 * i].weight.uniform_(-sqrt(6. / pe_feat_dim), sqrt(6. / pe_feat_dim))
                    nn.init.constant_(self.pe[2 * i].bias.data, 0.0)

        all_feat_dim = (pe_feat_dim if self.pe else in_dim) + (2 if in_glob else 1) * in_feat_dim

        self.first_fc = nn.Linear(all_feat_dim, out_feat_dim[0], bias=False)
        self.resnet = nn.ModuleList([
            resnet_block_film(out_feat_dim[i], film=True if film else False)
        for i in range(len(out_feat_dim))])
        self.fc = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc_{}_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('fc_{}'.format(i + 1), nn.Linear(out_feat_dim[i],
                                                  out_feat_dim[i + 1] if i != (len(out_feat_dim) - 1) else 2,
                                                  bias=False if i != (len(out_feat_dim) - 1) else True))
            ]))
        for i in range(len(out_feat_dim))])

        if film:
            self.film_w = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('film_w_{}_1'.format(i + 1), nn.Linear(in_feat_dim, out_feat_dim, bias=False)),
                    ('film_w_{}_1_bn'.format(i + 1), nn.BatchNorm1d(out_feat_dim)),
                    ('film_w_{}_1_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    ('film_w_{}_2'.format(i + 1), nn.Linear(out_feat_dim, out_feat_dim, bias=True)),
                ]))
            for i in range(len(self.resnet))])
            with torch.no_grad():
                for i in range(len(self.resnet)):
                    self.film_w[i][-1].weight.normal_(std=film_std)
                    nn.init.constant_(self.film_w[i][-1].bias.data, 0.0)

            self.film_b = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('film_b_{}_1'.format(i + 1), nn.Linear(in_feat_dim, out_feat_dim, bias=False)),
                    ('film_b_{}_1_bn'.format(i + 1), nn.BatchNorm1d(out_feat_dim)),
                    ('film_b_{}_1_lrelu'.format(i + 1), nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    ('film_b_{}_2'.format(i + 1), nn.Linear(out_feat_dim, out_feat_dim, bias=True)),
                ]))
            for i in range(len(self.resnet))])
            with torch.no_grad():
                for i in range(len(self.resnet)):
                    self.film_b[i][-1].weight.normal_(std=film_std)
                    nn.init.constant_(self.film_b[i][-1].bias.data, 0.0)

    def forward(self, l_features, query=None, g_features=None):
        ### Positional encoding with SIREN features ###
        if query is not None:
            pos = query
            if self.pe:
                pos = self.pe(2.0 * pos)

        if query is not None:
            if g_features is not None:
                features = torch.cat([pos, l_features, g_features], dim=1)
            else:
                features = torch.cat([pos, l_features], dim=1)
        else:
            if g_features is not None:
                features = torch.cat([l_features, g_features], dim=1)
            else:
                features = l_features

        out = self.first_fc(features)
        for i in range(len(self.resnet)):
            out = self.resnet[i](out)
            if self.film:
                out = torch.add(self.film_w[i](features), 1.0) * out + self.film_b[i](features)
            out = self.fc[i](out)

        return out
