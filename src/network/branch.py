from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv

        self.action = nn.Sequential(
            nn.Conv3d(input_channel, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel * 2, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk):
        output = {}
        output_wh = []
        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        input_action = torch.stack(input_chunk, dim=2)
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output_ac = self.action(input_action)
        b,_,_,h,w = output_ac.shape
        output_ac = output_ac.view(b, -1, h, w)
        hm_input_chunk = torch.cat([input_chunk, output_ac], dim=1)
        output['hm'] = self.hm(hm_input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['wh'] = output_wh
        return output
