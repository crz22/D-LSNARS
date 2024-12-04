import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find("Conv3d") != -1 or classname.find("ConvTranspose3d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class Conv3D(nn.Module):
    def __init__(self,Cin,Cout,ksize,stride,
                 pad = 0,
                 norm = 'None',
                 actf = 'None'):
        super(Conv3D, self).__init__()
        model = []
        model += [nn.ReflectionPad3d(pad)]
        model += [nn.Conv3d(Cin, Cout, ksize, stride, padding=0)]

        if norm != 'None':
            model += [self.normlayer(norm, Cout)]

        if actf != 'None':
            model += [self.actfunction(actf)]

        self.model = nn.Sequential(*model)
        self.model.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)

    def normlayer(self, mode, channel):
        if mode == 'batch':
            return nn.BatchNorm3d(channel)
        else:
            assert mode == 'None', 'Normlayer initial False'

    def actfunction(self, mode):
        if mode == 'relu':
            return nn.ReLU(inplace=True)
        elif mode == 'sigmoid':
            return nn.Sigmoid()
        else:
            assert mode == 'None', 'actfunction initial False'

#depthwise separable convolution
class DSConv3d(nn.Module):
    def __init__(self,Cin,Cout,ksize=3,stride=1,pad=1):
        super(DSConv3d, self).__init__()
        # Replace the first Conv3d layer with a MobileNetV3 depthwise separable convolution
        self.conv1 = nn.Sequential(nn.ReflectionPad3d(pad),
                                   nn.Conv3d(Cin, Cout, ksize, stride, groups=Cin),
                                   nn.BatchNorm3d(Cout),
                                   nn.ReLU())
        # Replace the second Conv3d layer with a MobileNetV3 pointwise convolution
        self.conv2 = nn.Sequential(nn.Conv3d(Cout, Cout, kernel_size=1),
                                   nn.BatchNorm3d(Cout),
                                   nn.ReLU())
        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

#Channel Spatial Fusion Module (Attention block)
class CSFM(nn.Module):
    def __init__(self,dim):
        super(CSFM, self).__init__()
        self.channel_att_layer = Channel_att(dim)
        self.spatial_att_layer = Spatial_att(dim)
    def forward(self,x):
        out = self.channel_att_layer(x)
        out = self.spatial_att_layer(out)
        return out

#channel attention block
class Channel_att(nn.Module):
    def __init__(self,dim,ratio=8):
        super(Channel_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=1)
        self.fc_layer = nn.Sequential(nn.Conv3d(dim,dim//ratio,kernel_size=1,bias=False),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(dim//ratio,dim,kernel_size=1,bias=False),)
        self.fc_layer.apply(weights_init_normal)
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        channel_weight = self.fc_layer(x_avg)+self.fc_layer(x_max)
        channel_weight = self.Sigmoid(channel_weight)
        out = x * channel_weight+x
        return out

#spatial attention block
class Spatial_att(nn.Module):
    def __init__(self,dim,ratio=8):
        super(Spatial_att, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d((2, 2, 3, 3, 3, 3)),
                                            nn.Conv3d(2,1, kernel_size=(7,7,5), bias=False))
        self.spatial_layer2 = nn.Sequential(nn.ReflectionPad3d((1, 1, 2, 2, 2, 2)),
                                            nn.Conv3d(2,1, kernel_size=(5,5,3), bias=False))
        self.spatial_layer3 = nn.Sequential(nn.ReflectionPad3d((0, 0, 1, 1, 1, 1)),
                                            nn.Conv3d(2,1, kernel_size=(3,3,1), bias=False))
        self.fusion_layer = nn.Sequential(nn.Conv3d(3,1,kernel_size=1,bias=False),
                                          nn.Sigmoid())
        self.fusion_layer.apply(weights_init_normal)
    def channel_pool(self,y):
        y_avg = torch.mean(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)
        y_out = torch.cat([y_avg, y_max], dim=1)
        return y_out

    def forward(self,x):
        s_in = self.channel_pool(x)
        s1 = self.spatial_layer1(s_in)
        s2 = self.spatial_layer2(s_in)
        s3 = self.spatial_layer3(s_in)

        s_cat = torch.cat((s1,s2,s3),dim=1)
        spatial_weight = self.fusion_layer(s_cat)
        out = x*spatial_weight+x
        return out

class UpConcat(nn.Module):
    def __init__(self,dim,dim_cat):
        super(UpConcat, self).__init__()
        self.up_layer1 = nn.ConvTranspose3d(dim,dim_cat,kernel_size=2,stride=2)
        self.conv_layer1 = Conv3D(dim_cat*2,dim_cat,ksize=3,stride=1,pad=1,norm='batch',actf='relu')
        self.conv_layer2 = Conv3D(dim_cat,dim_cat,ksize=3,stride=1,pad=1,norm='batch',actf='relu')
        self.up_layer1.apply(weights_init_normal)
    def forward(self,x,x_cat):
        y1 = self.up_layer1(x)
        #print(x.shape,x_cat.shape)
        y1 = torch.cat([y1,x_cat],dim=1)
        out = self.conv_layer1(y1)
        out = self.conv_layer2(out)
        return out
