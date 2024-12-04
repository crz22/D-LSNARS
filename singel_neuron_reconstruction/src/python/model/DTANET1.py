import torch
import torch.nn as nn
import torch.nn.functional as F
from model.baselayer1 import *

class MSFE(nn.Module):
    def __init__(self):
        super(MSFE, self).__init__()
        #[1,64,64,64]
        self.layer1 = nn.Sequential(DSConv3d(1,32),
                                    DSConv3d(32,32))
        self.att_layer1 = CSFM(32)
        #[32,64,64,64]
        self.layer2 = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                    DSConv3d(32,64),#)
                                    DSConv3d(64,64))
        self.att_layer2 = CSFM(64)
        #[64,32,32,32]
        self.layer3 = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                    DSConv3d(64,128),#)
                                    DSConv3d(128,128))
        self.att_layer3 = CSFM(128)
        #[128,16,16,16]
        self.layer4 = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                    DSConv3d(128,128),#)
                                    DSConv3d(128,128))
        self.att_layer4 = CSFM(128)
        #[128,8,8,8]
    def forward(self,x):
        y1 = self.layer1(x)
        y1 = self.att_layer1(y1)

        y2 = self.layer2(y1)
        y2 = self.att_layer2(y2)

        y3 = self.layer3(y2)
        y3 = self.att_layer3(y3)

        y4 = self.layer4(y3)
        y4 = self.att_layer4(y4)

        mid_feature = [y1,y2,y3,y4]
        return mid_feature

class EFC_NET(nn.Module):
    def __init__(self,class_num=2):
        super(EFC_NET, self).__init__()
        #[128,8,8,8]
        self.layer1 = nn.Sequential(Conv3D(128,256,ksize=3,stride=1,pad=1,norm='batch',actf='relu'),
                                    nn.MaxPool3d(kernel_size=2,stride=2))
        #[256,4,4,4]
        self.layer2 = Conv3D(256,256,ksize=3,stride=1,pad=1,norm='batch',actf='relu')
        self.layer3 = Conv3D(256,class_num,ksize=1,stride=1)
    def forward(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        out = torch.mean(y3,dim=[2,3,4])
        return out

class PSAD_NET(nn.Module):
    def __init__(self):
        super(PSAD_NET, self).__init__()
        #[128,8,8,8]
        self.up_layer1 = UpConcat(128,128)
        #[128,16,16,16]
        self.up_layer2 = UpConcat(128,64)
        #[64,32,32,32]
        self.up_layer3 = UpConcat(64,32)
        #[32,64,64,64]
        self.out_layer = nn.Sequential(Conv3D(32,32,ksize=3,stride=1,pad=1,norm='batch',actf='relu'),
                                       Conv3D(32,1,ksize=3,stride=1,pad=1,actf='sigmoid'))

    def forward(self, feature_map):
        mid_feat1, mid_feat2, mid_feat3, mid_feat4 = feature_map
        up_feat1 = self.up_layer1(mid_feat4, mid_feat3)
        up_feat2 = self.up_layer2(up_feat1, mid_feat2)
        up_feat3 = self.up_layer3(up_feat2, mid_feat1)
        output = self.out_layer(up_feat3)
        return output

class DTANET(nn.Module):
    def __init__(self,class_num=2):
        super(DTANET, self).__init__()
        self.MSFE = MSFE()
        self.EFC = EFC_NET(class_num)
        self.PASD_WS = PSAD_NET()
        self.PASD_SN = PSAD_NET()
        self.Softmax = nn.Softmax(dim=1)

    def forward(self,inputs):
        feature_map = self.MSFE(inputs)
        class_out = self.EFC(feature_map[-1])
        class_out = self.Softmax(class_out)
        class_out = torch.argmax(class_out,dim=1)
        if class_out == 0:
            output = self.PASD_WS(feature_map)
        else:
            output = self.PASD_SN(feature_map)
        return output

    def classify_forward(self,inputs):
        feature_map = self.MSFE(inputs)
        class_out = self.EFC(feature_map[-1])
        return class_out

    def segment_forward(self,inputs,clabel):
        feature_map = self.MSFE(inputs)
        batch_size = inputs.shape[0]
        for i in range(batch_size):
            feature_map1 = [feature_map[0][i].unsqueeze(dim=0),feature_map[1][i].unsqueeze(dim=0),
                            feature_map[2][i].unsqueeze(dim=0),feature_map[3][i].unsqueeze(dim=0)]
            if clabel[i] == 0:
                output1 = self.PASD_WS(feature_map1)
            else:
                output1 = self.PASD_SN(feature_map1)

            if i == 0:
                output = output1
            else:
                output = torch.cat((output,output1),dim=0)
        return output
