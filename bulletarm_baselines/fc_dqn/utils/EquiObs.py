import torch
from escnn import gspaces
from escnn import nn

class EquiObs(torch.nn.Module):
    def __init__(self, num_subgroups, filter_counts=[16, 32, 64, 128, 256, 128], dim_out=128):
        super(EquiObs, self).__init__()
        self.filter_counts = filter_counts
        self.dim_out = dim_out

        self.r2_act = gspaces.flipRot2dOnR2(N=num_subgroups)
        # self.r2_act = gspaces.rot2dOnR2(N=num_subgroups)

        self.input_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        self.obs_encoder = torch.nn.Sequential(
            ### 128x128 ###
            nn.R2Conv(self.input_type,  
                    nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=1, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr]), 2),

            ### 64x64 ###
            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[0]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=1, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr]), 2),

            ### 32x32 ###
            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[1]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=1, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr]), 2),

            ### 16x16 ###
            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[2]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=1, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr]), 2),

            ### 8x8 ###
            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[3]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.filter_counts[4]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=1, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[4]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[4]*[self.r2_act.regular_repr]), inplace=True), 

            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[4]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.filter_counts[5]*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=0, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.filter_counts[5]*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.filter_counts[5]*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.filter_counts[5]*[self.r2_act.regular_repr]), 2),

            ### 3x3 ###
            nn.R2Conv(nn.FieldType(self.r2_act, self.filter_counts[5]*[self.r2_act.regular_repr]),
                    nn.FieldType(self.r2_act, self.dim_out*[self.r2_act.regular_repr]),
                    kernel_size=3, stride=1, padding=0, initialize=True),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, self.dim_out*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, self.dim_out*[self.r2_act.regular_repr]), inplace=True),
        )

        self.gpool = nn.GroupPooling(nn.FieldType(self.r2_act, self.dim_out*[self.r2_act.regular_repr]))

    def forward(self, input):
        x = nn.GeometricTensor(input, self.input_type)
        x = self.obs_encoder(x)
        x = self.gpool(x)
        x = x.tensor
        return x