import torch
from escnn import gspaces
from escnn import nn
import numpy as np

class EquiHandObs(torch.nn.Module):
    def __init__(self, num_subgroups, filter_sizes, filter_counts):
    
        super(EquiHandObs, self).__init__()
        self.num_subgroups = num_subgroups
        self.filter_sizes = filter_sizes
        self.filter_counts = filter_counts

        self.r2_act = gspaces.flipRot2dOnR2(N=num_subgroups)

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr])
        self.block1 = self.make_block(in_type=in_type, out_type=out_type, filter_size=self.filter_sizes[0])

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr])
        self.block2 = self.make_block(in_type=in_type, out_type=out_type, filter_size=self.filter_sizes[1])
        
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr])
        self.block3 = self.make_block(in_type=in_type, out_type=out_type, filter_size=self.filter_sizes[2])

        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr])
        self.block4 = self.make_block(in_type=in_type, out_type=out_type, filter_size=self.filter_sizes[3])

        self.gpool = nn.GroupPooling(out_type)

    def forward(self, input):
        x = nn.GeometricTensor(input, self.input_type)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.gpool(x)
        x = x.tensor

        return x

    def make_block(self, in_type, out_type, filter_size):
        conv = nn.R2Conv(in_type=in_type, out_type=out_type, kernel_size=filter_size, padding=(filter_size-1)//2, bias=False, initialize=True)
        batchnorm = nn.InnerBatchNorm(out_type)
        activation_last = nn.ELU(out_type, inplace=True)
        pool = nn.PointwiseMaxPool(out_type, 2)
        return nn.SequentialModule(conv, batchnorm, activation_last, pool)
