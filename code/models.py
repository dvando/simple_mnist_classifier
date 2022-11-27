import torch
import torch.nn as nn
import torchvision
import cv2
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

def _cbr_layer(in_channels, out_channels, kernel_size, groups=1, stride=1, activation=True):
        if activation:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                        groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2),
                        groups=groups,
                        bias=False),
                nn.BatchNorm2d(out_channels, affine=True, eps=1e-5, momentum=0.1))

class TributeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cbr1 = _cbr_layer(1, 16, 3, stride=1)
        self.cbr2 = _cbr_layer(16, 32, 3, stride=1)
        self.cbr3 = _cbr_layer(32, 48, 3, stride=1)
        self.cbr4 = _cbr_layer(48, 64, 3, stride=1)
        self.cbr5 = _cbr_layer(64, 80, 3, stride=1)
        self.cbr6 = _cbr_layer(80, 96, 3, stride=1)
        self.cbr7 = _cbr_layer(96, 112, 3, stride=1)
        self.cbr8 = _cbr_layer(112, 128, 3, stride=1)
        self.pre_out = nn.Conv2d(128, 64, 5, stride=1, bias=True)
        self.out = nn.Conv2d(64, 10, 5, stride=1, bias=True)
        self.pool = nn.AvgPool2d(4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cbr1(x)
        feat1 = x
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        feat2 = x
        x = self.cbr5(x)
        x = self.cbr6(x)
        x = self.cbr7(x)
        feat3 = x
        x = self.cbr8(x)
        x = self.pre_out(x)
        x = self.softmax(x)
        feat4 = x
        x = self.out(x)
        x = self.pool(x)
        

        return feat1, feat2, feat3, feat4, x


if __name__ == '__main__':
    model = TributeNet().cuda()
    dummy = torch.randn(1, 1, 28, 28).cuda()
    summary(model, (1, 28, 28))
    out = model(dummy)
    writer = SummaryWriter(f'logs/tes')
    writer.add_graph(model, dummy)
