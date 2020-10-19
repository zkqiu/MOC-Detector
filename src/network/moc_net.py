from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet

from .recognition import build_rec_backbone

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}

class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K, rec, modality, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.backbone = backbone[arch](num_layers)
        self.rec_backbone = build_rec_backbone(rec, 24, K, modality)
        self.branch = MOC_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)

    def forward(self, input, rec_input):
        if self.flip_test:
            assert(self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            chunk1_attention = self.rec_backbone(rec_input)
            chunk2_attention = self.rec_backbone(rec_input)

            return [self.branch(chunk1, chunk1_attention), self.branch(chunk2, chunk2_attention)]
        else:
            chunk = [self.backbone(input[i]) for i in range(self.K)]
            attention = self.rec_backbone(rec_input)

            return [self.branch(chunk, attention)]
