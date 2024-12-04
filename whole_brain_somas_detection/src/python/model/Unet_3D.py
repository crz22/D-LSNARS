import torch
import torch.nn as nn

# Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),#3 1
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        nn.BatchNorm3d(out_dim),
        activation, )


class UNet_3D(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet_3D, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.up_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_1 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_2 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.trans_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)

        # Output
        self.out = nn.Sequential(nn.Conv3d(self.num_filters,out_dim,kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # Down sampling
        # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
        down_1 = self.down_1(x)  # -> [6, 16, 32, 64, 64]
        pool_1 = self.pool_1(down_1)  # -> [6, 16, 16, 32, 32]

        down_2 = self.down_2(pool_1)  # -> [6, 32, 16, 32, 32]
        pool_2 = self.pool_2(down_2)  # -> [6, 32, 8, 16, 16]

        down_3 = self.down_3(pool_2)  # -> [6, 64, 8, 16, 16]
        pool_3 = self.pool_3(down_3)  # -> [6, 64, 4, 8, 8]

        down_4 = self.down_4(pool_3)  # -> [6, 128, 4, 8, 8]

        # Up sampling
        up_1 = self.up_1(down_4)  # -> [6, 64, 8, 16, 16]
        concat_1 = torch.cat([up_1, down_3], dim=1)  # -> [6, 64, 16, 32, 32]
        trans_1 = self.trans_1(concat_1)

        up_2 = self.up_2(trans_1)  # -> [6, 32, 16, 32, 32]
        concat_2 = torch.cat([up_2, down_2], dim=1)  # -> [6, 32, 32, 64, 64]
        trans_2 = self.trans_2(concat_2)  # -> [6, 16, 32, 64, 64]

        up_3 = self.up_3(trans_2)  # -> [6, 16, 32, 64, 64]
        concat_3 = torch.cat([up_3, down_1], dim=1)
        trans_3 = self.trans_3(concat_3)

        # Output
        out = self.out(trans_3)  # -> [6, 2, 32, 64, 64]
        return out