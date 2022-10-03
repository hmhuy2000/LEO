import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                if not last:
                    x = F.max_pool2d(x, 2)

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
