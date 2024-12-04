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

#Channel Spatial Fusion Module (Attention block)
class CSFM(nn.Module):
    def __init__(self,dim):
        super(CSFM, self).__init__()
        self.channel_att_layer = Channel_att(dim)
        self.spatial_att_layer = Fusion_Spatial_att()
    def forward(self,x):
        out = self.channel_att_layer(x)
        out = self.spatial_att_layer(out)
        return out

class Channel_att(nn.Module):
    def __init__(self,dim,ratio=16):
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
        out = x * channel_weight
        return out

#spatial attention block
class Spatial_att(nn.Module):
    def __init__(self):
        super(Spatial_att, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d((3, 3, 3, 3, 3, 3)),
                                            nn.Conv3d(2,1, kernel_size=7, bias=False))
        self.Sigmoid = nn.Sigmoid()
        #self.fusion_layer.apply(weights_init_normal)
    def channel_pool(self,y):
        y_avg = torch.mean(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)
        y_out = torch.cat([y_avg, y_max], dim=1)
        return y_out

    def forward(self,x):
        s_in = self.channel_pool(x)
        s1 = self.spatial_layer1(s_in)
        spatial_weight = self.Sigmoid(s1)
        out = x*spatial_weight
        return out

class Fusion_Spatial_att(nn.Module):
    def __init__(self):
        super(Fusion_Spatial_att, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d(( 3, 3, 3, 3, 2, 2)),
                                            nn.Conv3d(2,1, kernel_size=(5,7,7), bias=False))
        self.spatial_layer2 = nn.Sequential(nn.ReflectionPad3d((2, 2, 2, 2 ,1, 1)),
                                            nn.Conv3d(2,1, kernel_size=(3,5,5), bias=False))
        self.spatial_layer3 = nn.Sequential(nn.ReflectionPad3d((1, 1, 1, 1, 0, 0 )),
                                            nn.Conv3d(2, 1, kernel_size=(1, 3, 3), bias=False))
        self.Sigmoid = nn.Sigmoid()
        #self.fusion_layer.apply(weights_init_normal)
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
        #print(s1.shape,s2.shape,s3.shape)
        spatial_weight = self.Sigmoid(s1+s2+s3)
        out = x*spatial_weight
        return out

class Fusion_Spatial_att2(nn.Module):
    def __init__(self):
        super(Fusion_Spatial_att2, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d((2, 2 , 3, 3, 3, 3)),
                                            nn.Conv3d(2,1, kernel_size=(7,7,5), bias=False))
        self.spatial_layer2 = nn.Sequential(nn.ReflectionPad3d((1, 1, 2, 2, 2, 2 )),
                                            nn.Conv3d(2,1, kernel_size=(5,5,3), bias=False))
        self.spatial_layer3 = nn.Sequential(nn.ReflectionPad3d((0, 0, 1, 1, 1, 1)),
                                            nn.Conv3d(2, 1, kernel_size=(3,3,1), bias=False))
        self.Sigmoid = nn.Sigmoid()
        #self.fusion_layer.apply(weights_init_normal)
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
        #print(s1.shape,s2.shape,s3.shape)
        spatial_weight = self.Sigmoid(s1+s2+s3)
        out = x*spatial_weight
        return out

class Fusion_Spatial_att3(nn.Module):
    def __init__(self):
        super(Fusion_Spatial_att3, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d((2, 2 , 3, 3, 3, 3)),
                                            nn.Conv3d(2,1, kernel_size=(7,7,5), bias=False))
        self.spatial_layer2 = nn.Sequential(nn.ReflectionPad3d((1, 1, 2, 2, 2, 2 )),
                                            nn.Conv3d(2,1, kernel_size=(5,5,3), bias=False))
        self.spatial_layer3 = nn.Sequential(nn.ReflectionPad3d((0, 0, 1, 1, 1, 1)),
                                            nn.Conv3d(2, 1, kernel_size=(3,3,1), bias=False))
        self.fusion_layer = nn.Sequential(nn.Conv3d(3,1,kernel_size=1,stride=1,bias=False),
                                          nn.Sigmoid())
        self.Sigmoid = nn.Sigmoid()
        #self.fusion_layer.apply(weights_init_normal)
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
        #print(s1.shape,s2.shape,s3.shape)
        s = torch.cat((s1,s2,s3),dim=1)
        spatial_weight = self.fusion_layer(s)
        out = x*spatial_weight
        return out

class Fusion_Spatial_att4(nn.Module):
    def __init__(self):
        super(Fusion_Spatial_att4, self).__init__()
        self.spatial_layer1 = nn.Sequential(nn.ReflectionPad3d(( 3, 3, 3, 3, 2, 2)),
                                            nn.Conv3d(2,1, kernel_size=(5,7,7), bias=False))
        self.spatial_layer2 = nn.Sequential(nn.ReflectionPad3d((2, 2, 2, 2 ,1, 1)),
                                            nn.Conv3d(2,1, kernel_size=(3,5,5), bias=False))
        self.spatial_layer3 = nn.Sequential(nn.ReflectionPad3d((1, 1, 1, 1, 0, 0 )),
                                            nn.Conv3d(2, 1, kernel_size=(1, 3, 3), bias=False))
        self.fusion_layer = nn.Sequential(nn.Conv3d(3, 1, kernel_size=1, stride=1, bias=False),
                                          nn.Sigmoid())
        self.Sigmoid = nn.Sigmoid()
        #self.fusion_layer.apply(weights_init_normal)
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
        s = torch.cat((s1, s2, s3), dim=1)
        spatial_weight = self.fusion_layer(s)
        out = x*spatial_weight
        return out