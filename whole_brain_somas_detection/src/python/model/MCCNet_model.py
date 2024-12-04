import torch
import torch.nn as nn
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.startswith('Linear'):
        m.weight.data.normal_(0.0, 0.02)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 6*128*128
        )

    def forward(self, input):
            return self.conv(input)

class ClassNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ClassNet, self).__init__()

        self.conv1 = DoubleConv(3,8)
        self.conv2 = DoubleConv(8, 16)
        self.conv3 = DoubleConv(16, 32)
        self.conv4 = DoubleConv(32, 64)
        self.conv5 = DoubleConv(64, 128)
        self.conv6 = DoubleConv(128, 256)
        self.conv7 = DoubleConv(256, 512)  #512*2*2
        self.fcn1 = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),)
        self.fcn6 = nn.Linear(512, num_classes)
    def forward(self, x):
        # print('x', x.shape)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c7_out = c7.reshape(c7.size(0), -1)
        fc1 = self.fcn1(c7_out)
        fc6 = self.fcn6(fc1)
        return fc6




