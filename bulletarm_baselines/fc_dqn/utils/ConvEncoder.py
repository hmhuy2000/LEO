import numpy as np
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):

    def __init__(self, config):

        super(ConvEncoder, self).__init__()

        c = config
        self.input_size = c["input_size"]
        self.filter_sizes = c["filter_size"]
        self.filter_counts = c["filter_counts"]
        self.strides = c["strides"]
        self.use_batch_norm = c["use_batch_norm"]
        self.activation_last = c["activation_last"]
        self.flat_output = c["flat_output"]

        assert len(self.filter_sizes) == len(self.filter_counts) == len(self.strides)

        self.use_norm = self.use_batch_norm
        self.output_size = self.get_output_size_()
        if self.flat_output:
            self.output_size = int(np.prod(self.output_size))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # create conv and padding layers
        convs, pads = self.make_convs_()
        self.convs = nn.ModuleList(convs)
        self.pads = nn.ModuleList(pads)

        # maybe create norm layers
        self.norms = None
        if self.use_batch_norm:
            self.norms = nn.ModuleList(self.make_bns_())

        # initialize variables
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if not self.use_norm:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        size = x.size()
        assert len(size) == 4

        for i in range(len(self.filter_sizes)):

            last = i == len(self.filter_sizes) - 1

            x = self.convs[i](self.pads[i](x))

            if not last or self.activation_last:

                if self.use_norm:
                    x = self.norms[i](x)

                x = self.relu(x)
                x = self.pool(x)

        if self.flat_output:
            x = torch.flatten(x, start_dim=1)

        return x

    def make_convs_(self):

        convs = []
        pads = []

        for i in range(len(self.filter_sizes)):
            if i == 0:
                channels = self.input_size[2]
            else:
                channels = self.filter_counts[i - 1]

            convs.append(
                nn.Conv2d(
                    channels, self.filter_counts[i], kernel_size=self.filter_sizes[i], stride=self.strides[i],
                    bias=not self.use_norm
                )
            )

            pads.append(self.get_padding_layer_(*self.get_same_padding_(self.filter_sizes[i], self.strides[i])))

        return convs, pads

    def make_bns_(self):

        bns = []

        if self.activation_last:
            count = len(self.filter_sizes)
        else:
            count = len(self.filter_sizes) - 1

        for i in range(count):
            bns.append(nn.BatchNorm2d(self.filter_counts[i]))

        return bns

    def get_same_padding_(self, kernel_size, stride):

        assert kernel_size >= stride
        total_padding = kernel_size - stride

        p1 = int(np.ceil(total_padding / 2))
        p2 = int(np.floor(total_padding / 2))
        assert p1 + p2 == total_padding

        return p1, p2

    def get_padding_layer_(self, p1, p2):

        return nn.ZeroPad2d((p1, p2, p1, p2))

    def get_output_size_(self):

        total_stride = int(np.prod(self.strides))
        width = self.input_size[0] // total_stride
        height = self.input_size[1] // total_stride

        if len(self.filter_counts) == 0:
            channels = self.input_size[2]
        else:
            channels = self.filter_counts[-1]

        return width, height, channels

class CNNOBSEncoder(nn.Module):
    def __init__(self, filter_counts=[16, 32, 64, 128, 256, 128], dim_out=128):
        super(CNNOBSEncoder, self).__init__()
        
        self.obs_encoder = nn.Sequential(
            ### 128x128###
            nn.Conv2d(1, filter_counts[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 64x64 ###
            nn.Conv2d(filter_counts[0], filter_counts[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 32x32 ###
            nn.Conv2d(filter_counts[1], filter_counts[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[2]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 16x16 ###
            nn.Conv2d(filter_counts[2], filter_counts[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[3]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 8x8 ###
            nn.Conv2d(filter_counts[3], filter_counts[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[4]),
            nn.GELU(),
            
            nn.Conv2d(filter_counts[4], filter_counts[5], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(filter_counts[5]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 3x3 ###
            nn.Conv2d(filter_counts[5], dim_out, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),   
        )
        self.init_weights()
        
    def forward(self, x):
        return self.obs_encoder(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                

class CNNHandObsEncoder(nn.Module):
    def __init__(self, filter_counts=[32, 64, 128], dim_out=128):
        super(CNNHandObsEncoder, self).__init__()
        
        self.hand_obs_encoder = nn.Sequential(
            ### 24x24 ###
            nn.Conv2d(1, filter_counts[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 12x12 ###
            nn.Conv2d(filter_counts[0], filter_counts[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ### 6x6 ###                                                                  
            nn.Conv2d(filter_counts[1], filter_counts[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_counts[2]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
  
            ### 3x3 ###
            nn.Conv2d(filter_counts[2], dim_out, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
            
        )
        self.init_weights()

    def forward(self, x):
        return self.hand_obs_encoder(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)