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


class Attention_block_3d(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_3d, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):    #g:64*64*512   x:64*64*512
        # print(g.shape)
        # print(x.shape)
        g1 = self.W_g(g)     #g1(64*64*256)
        # print(g1.shape)
        x1 = self.W_x(x)     #x1(64*64*256)
        # print(x1.shape)
        psi = self.relu(g1 + x1)   #psi(64*64*256)
        # print("psi的尺寸")
        # print(psi.shape)
        psi = self.psi(psi)      #64*64*1
        # print(psi.shape)
        out = x * psi   #out（64*64*512）
        # print('********最后的尺寸')
        # print(out.shape)
        return out

class Att_UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(Att_UNet, self).__init__()

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
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)

        # Output
        self.out = nn.Sequential(nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1))

        self.Att3 = Attention_block_3d(F_g=self.num_filters * 4, F_l=self.num_filters * 4,F_int=self.num_filters * 2)
        self.Att2 = Attention_block_3d(F_g=self.num_filters * 2, F_l=self.num_filters * 2,F_int=self.num_filters)
        self.Att1 = Attention_block_3d(F_g=self.num_filters, F_l=self.num_filters,F_int=8)

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
        # x3 = self.Att3(down_3)
        # x2 = self.Att2(down_2)
        # x1 = self.Att1(down_1)

        # Up sampling
        trans_1 = self.trans_1(down_4)  # -> [6, 64, 8, 16, 16]
        # print(trans_1.shape,down_3.shape)
        x3 = self.Att3(trans_1,down_3)
        concat_1 = torch.cat([x3, trans_1], dim=1)  # -> [6, 128, 8, 16, 16]
        up_1 = self.up_1(concat_1)  # -> [6, 64, 8, 16, 16]

        trans_2 = self.trans_2(up_1)  # -> [6, 32, 16, 32, 32]
        x2 = self.Att2(trans_2, down_2)
        concat_2 = torch.cat([x2, trans_2], dim=1)  # -> [6, 64, 16, 32, 32]
        up_2 = self.up_2(concat_2)  # -> [6, 32, 16, 32, 32]

        trans_3 = self.trans_3(up_2)  # -> [6, 16, 32, 64, 64]
        x1 = self.Att1(trans_3, down_1)
        concat_3 = torch.cat([x1, trans_3], dim=1)  # -> [6, 32, 32, 64, 64]
        up_3 = self.up_3(concat_3)  # -> [6, 16, 32, 64, 64]

        # Output
        out = self.out(up_3)  # -> [6, 2, 32, 64, 64]
        # print('out',out.shape)
        return out