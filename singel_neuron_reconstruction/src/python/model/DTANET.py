import torch
import torch.nn as nn
from model.baselayer import Channel_att,Spatial_att,Fusion_Spatial_att,CSFM,Fusion_Spatial_att2,Fusion_Spatial_att3

# Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=(3,3,3), stride=1, padding=(1,1,1),padding_mode='reflect'),#3 1
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
        nn.Conv3d(out_dim, out_dim, kernel_size=(3,3,3), stride=1, padding=(1,1,1),padding_mode='reflect'),
        nn.BatchNorm3d(out_dim),
        activation, )

class Multiscale_Feature_Encoder(nn.Module):
    def __init__(self,in_dim, num_filters):
        super(Multiscale_Feature_Encoder,self).__init__()
        self.in_dim = in_dim
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

        # Attention module
        self.CSFM_1 = CSFM(self.num_filters)
        self.CSFM_2 = CSFM(self.num_filters * 2)
        self.CSFM_3 = CSFM(self.num_filters * 4)
        self.CSFM_4 = CSFM(self.num_filters * 8)

    def forward(self,x):
        # Down sampling
        # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
        down_1 = self.down_1(x)  # -> [32, 64, 64, 64]
        down_1 = self.CSFM_1(down_1) + down_1
        pool_1 = self.pool_1(down_1)  # -> [32, 32, 32, 32]

        down_2 = self.down_2(pool_1)  # -> [64, 32, 32, 32]
        down_2 = self.CSFM_2(down_2) + down_2
        pool_2 = self.pool_2(down_2)  # -> [64, 16, 16, 16]

        down_3 = self.down_3(pool_2)  # -> [128, 16, 16, 16]
        down_3 = self.CSFM_3(down_3) + down_3
        pool_3 = self.pool_3(down_3)  # -> [128, 8, 8, 8]

        down_4 = self.down_4(pool_3)  # -> [256, 8, 8, 8]
        down_4 = self.CSFM_4(down_4) + down_4

        return [down_1,down_2,down_3,down_4]

class External_Features_Classifier(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(External_Features_Classifier,self).__init__()
        activation = nn.ReLU(inplace=True)
        self.conv1 = conv_block_2_3d(in_dim,in_dim*2,activation)  #->[512, 8,8,8]
        self.pool1 = max_pooling_3d() #->[512,4,4,4]
        self.conv2 = nn.Conv3d(in_dim*2,out_dim,kernel_size=1,stride=1)  #-> [out_dim,4,4,4]
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  #->[out_dim,1,1,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self,feature):
        f1 = self.conv1(feature)
        f1 = self.pool1(f1)

        f2 = self.conv2(f1)
        f2 = self.avg_pool(f2)
        f2 = f2.view(f2.shape[0],f2.shape[1])

        out = self.sigmoid(f2)
        return out

class Parameter_Adaptive_Decoder(nn.Module):
    def __init__(self,num_filters,out_dim):
        super(Parameter_Adaptive_Decoder,self).__init__()
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)
        # Up sampling
        self.up_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_1 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_2 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.trans_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation)

        # Output
        self.out = nn.Sequential(nn.Conv3d(self.num_filters, out_dim, kernel_size=3, stride=1, padding=1),
                                 nn.Sigmoid())

    def forward(self,feature):
        down_1, down_2, down_3, down_4 = feature
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

class DTANET(nn.Module):
    def __init__(self, in_dim, class_num, num_filters):
        super(DTANET, self).__init__()
        self.MSFE = Multiscale_Feature_Encoder(in_dim,num_filters)
        self.EFC = External_Features_Classifier(num_filters*8,class_num)
        self.PAD_WS = Parameter_Adaptive_Decoder(num_filters, out_dim=1)
        self.PAD_SN = Parameter_Adaptive_Decoder(num_filters, out_dim=1)

    def forward(self, x):
        feature = self.MSFE(x)
        class_out = self.EFC(feature[-1]) #->[class_num]
        class_out = torch.argmax(class_out,dim=1)
        if class_out == 0:
            out = self.PAD_WS(feature)
        elif class_out == 1:
        #else:
            out = self.PAD_SN(feature)

        return out

    def class_forward(self,x):
        feature = self.MSFE(x)
        class_out = self.EFC(feature[-1])
        return class_out

    def segment_forward(self,x,class_label):
        feature = self.MSFE(x)
        out_WS = self.PAD_WS(feature)
        out_SN = self.PAD_SN(feature)
        class_label_reshape = class_label.reshape(class_label.shape[0],class_label.shape[1],1,1,1)
        out = out_WS*class_label_reshape[:,0,:,:,:]+out_SN*class_label_reshape[:,1,:,:,:]
        return out




