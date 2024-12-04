import torch
import torch.nn as nn
from torch.nn import functional as F

# Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ReflectionPad3d(1),
        nn.Conv3d(in_dim, out_dim, kernel_size=(3,3,3), stride=1),#3 1
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
        conv_block_3d(out_dim,out_dim, activation))

class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels,padding=1,dilation=1, kernel_size=3)
        self.bn = nn.ModuleList([nn.BatchNorm3d(in_channels), nn.BatchNorm3d(in_channels), nn.BatchNorm3d(in_channels)])

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_channels, out_channels=in_channels,kernel_size=1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels=in_channels//2, out_channels=2, kernel_size=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels=in_channels // 2, out_channels=2, kernel_size=3)
        )

        self.conv_last = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,stride=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv3d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv3d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.bn[2](branches_3)

        feat = torch.cat([branches_1, branches_2], dim=1)
        # feat=feat_cat.detach()
        att = self.layer1(feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)
        # feat=feat_cat.detach()
        att1 = self.layer2(feat1)
        att1 = F.softmax(att1, dim=1)

        att_1_2 = att1[:, 0, :, :].unsqueeze(1)
        att_3 = att1[:, 1, :, :].unsqueeze(1)

        ax = self.relu(self.gamma * (att_1_2 * fusion_1_2 + att_3 * branches_3) + (1 - self.gamma) * x)
        ax = self.conv_last(ax)
        return ax

class S_UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(S_UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)
        self.sap = SAPblock(self.num_filters * 8)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)

        # Output
        # self.out = conv_block_3d(self.num_filters, out_dim, activation)
        self.out = nn.Sequential(nn.ReflectionPad3d(1),
                                 nn.Conv3d(self.num_filters, out_dim,kernel_size=3,stride=1))
                                 #nn.BatchNorm3d(out_dim),
                                 #nn.Sigmoid())

    def forward(self, x):
        # Down sampling
        # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
        #x -> [6, 1, 32, 64, 64]
        down_1 = self.down_1(x)  # -> [6, 16, 32, 64, 64]
        # print(down_1.shape)
        pool_1 = self.pool_1(down_1)  # -> [6, 16, 16, 32, 32]
        # print(pool_1.shape)

        down_2 = self.down_2(pool_1)  # -> [6, 32, 16, 32, 32]
        pool_2 = self.pool_2(down_2)  # -> [6, 32, 8, 16, 16]

        down_3 = self.down_3(pool_2)  # -> [6, 64, 8, 16, 16]
        pool_3 = self.pool_3(down_3)  # -> [6, 64, 4, 8, 8]
        # print(pool_3.shape)
        down_4 = self.down_4(pool_3)  # -> [6, 128, 4, 8, 8]
        s = self.sap(down_4)

        # Up sampling
        trans_1 = self.trans_1(s)  # -> [6, 64, 8, 16, 16]
        # print(trans_1.shape,down_3.shape)
        concat_1 = torch.cat([trans_1, down_3], dim=1)  # -> [6, 128, 8, 16, 16]
        up_1 = self.up_1(concat_1)  # -> [6, 64, 8, 16, 16]

        trans_2 = self.trans_2(up_1)  # -> [6, 32, 16, 32, 32]
        concat_2 = torch.cat([trans_2, down_2], dim=1)  # -> [6, 64, 16, 32, 32]
        up_2 = self.up_2(concat_2)  # -> [6, 32, 16, 32, 32]

        trans_3 = self.trans_3(up_2)  # -> [6, 16, 32, 64, 64]
        concat_3 = torch.cat([trans_3, down_1], dim=1)  # -> [6, 32, 32, 64, 64]
        up_3 = self.up_3(concat_3)  # -> [6, 16, 32, 64, 64]

        # Output
        out = self.out(up_3)  # -> [6, 2, 32, 64, 64]
        # print('out',out.shape)
        return out
