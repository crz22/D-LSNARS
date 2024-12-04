# -*- coding: UTF-8 -*-

import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm3d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_2D(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm2d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CenterlineNet(nn.Module):
    def __init__(self, NUM_ACTIONS):
        super(CenterlineNet, self).__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        #self.n_classes = n_classes
        self.layer7 = nn.Conv3d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out

class CenterlineNet_Discrimintor(nn.Module):
    def __init__(self, NUM_ACTIONS):
        super(CenterlineNet_Discrimintor, self).__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        #self.n_classes = n_classes
        self.layer7 = nn.Conv3d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block(32, 64, 3, stride=1, p_size=0),
            conv_block(64, 64, 1, stride=1, p_size=0),
            nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)         
        )              


    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out = self.layer5(out)
        # print(out.shape)
        out = self.layer6(out)
        # print(out.shape)
        out = self.layer7(out)   
        # print(out.shape)

        return out, out_dis

class CenterlineNet_Discrimintor_2D_Radii(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(CenterlineNet_Discrimintor_2D_Radii, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        # self.layer5 = conv_block_2D(32, 64, 3, stride=1, p_size=0)
        # self.layer6 = conv_block_2D(64, 64, 1, stride=1, p_size=0)
        # #self.n_classes = n_classes
        # self.layer7 = nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),                     
        )        
        self.dis_out = nn.Conv2d(64, 2+1, kernel_size=1, stride=1, padding=0)
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.down_sampling(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out.shape)
        out_dis = self.dis_out(out_dis)
        # print(out.shape)
        out = self.tracker(out)
        # print(out.shape)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)        

        return out, out_dis
    
class CenterlineNet_Discrimintor_2D_Radii_32(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(CenterlineNet_Discrimintor_2D_Radii_32, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        # self.layer5 = conv_block_2D(32, 64, 3, stride=1, p_size=0)
        # self.layer6 = conv_block_2D(64, 64, 1, stride=1, p_size=0)
        # #self.n_classes = n_classes
        # self.layer7 = nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),                     
        )        
        self.dis_out = nn.Conv2d(64, 2+1, kernel_size=1, stride=1, padding=0)
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        # out = self.down_sampling(out)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out_dis = self.dis_out(out_dis)
        # print(out_dis.shape)
        out = self.tracker(out)
        # print(out.shape)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)        

        return out, out_dis
    
class CenterlineNet_Discrimintor_2D_Radii_16(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(CenterlineNet_Discrimintor_2D_Radii_16, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=1, p_size=1)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=1, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        # self.layer5 = conv_block_2D(32, 64, 3, stride=1, p_size=0)
        # self.layer6 = conv_block_2D(64, 64, 1, stride=1, p_size=0)
        # #self.n_classes = n_classes
        # self.layer7 = nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            nn.AvgPool2d(4, stride=1),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),                     
        )        
        self.dis_out = nn.Conv2d(64, 2+1, kernel_size=1, stride=1, padding=0)
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            nn.AvgPool2d(4, stride=1),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        # out = self.down_sampling(out)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out_dis = self.dis_out(out_dis)
        # print(out_dis.shape)
        out = self.tracker(out)
        # print(out.shape)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)        

        return out, out_dis

class CenterlineNet_Discrimintor_2D_Radii_128(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(CenterlineNet_Discrimintor_2D_Radii_128, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        # self.layer5 = conv_block_2D(32, 64, 3, stride=1, p_size=0)
        # self.layer6 = conv_block_2D(64, 64, 1, stride=1, p_size=0)
        # #self.n_classes = n_classes
        # self.layer7 = nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),                     
        )        
        self.dis_out = nn.Conv2d(64, 2+1, kernel_size=1, stride=1, padding=0)
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)

        out = self.down_sampling(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.down_sampling(out)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out_dis = self.dis_out(out_dis)
        # print(out_dis.shape)
        out = self.tracker(out)
        # print(out.shape)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)        

        return out, out_dis

class CenterlineNet_Discrimintor_2D(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(CenterlineNet_Discrimintor_2D, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        # self.layer5 = conv_block_2D(32, 64, 3, stride=1, p_size=0)
        # self.layer6 = conv_block_2D(64, 64, 1, stride=1, p_size=0)
        # #self.n_classes = n_classes
        # self.layer7 = nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)         
        )
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.down_sampling(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out = self.tracker(out)
        # print(out.shape)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)        

        return out, out_dis