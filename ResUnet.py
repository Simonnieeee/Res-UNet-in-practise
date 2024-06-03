import torch
import torch.nn as nn 

## This is a Residual Block class that will be used in the ResUnet model
class res_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=64):
        super(ResUnet, self).__init__()
        self.encoder1 = res_Block(in_channels, num_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = res_Block(num_filters, num_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = res_Block(num_filters*2, num_filters*4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = res_Block(num_filters*4, num_filters*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = res_Block(num_filters*8, num_filters*16)

        self.upconv4 = nn.ConvTranspose2d(num_filters*16, num_filters*8, 2, 2)
        self.decoder4 = res_Block(num_filters*16, num_filters*8)
        self.upconv3 = nn.ConvTranspose2d(num_filters*8, num_filters*4, 2, 2)
        self.decoder3 = res_Block(num_filters*8, num_filters*4)
        self.upconv2 = nn.ConvTranspose2d(num_filters*4, num_filters*2, 2, 2)
        self.decoder2 = res_Block(num_filters*4, num_filters*2)
        self.upconv1 = nn.ConvTranspose2d(num_filters*2, num_filters, 2, 2)
        self.decoder1 = res_Block(num_filters*2, num_filters)

        self.conv = nn.Conv2d(num_filters, out_channels, 1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        pool1 = self.pool1(encoder1)
        encoder2 = self.encoder2(pool1)
        pool2 = self.pool2(encoder2)
        encoder3 = self.encoder3(pool2)
        pool3 = self.pool3(encoder3)
        encoder4 = self.encoder4(pool3)
        pool4 = self.pool4(encoder4)

        bottleneck = self.bottleneck(pool4)

        upconv4 = self.upconv4(bottleneck)
        concat4 = torch.cat([upconv4, encoder4], 1)
        decoder4 = self.decoder4(concat4)
        upconv3 = self.upconv3(decoder4)
        concat3 = torch.cat([upconv3, encoder3], 1)
        decoder3 = self.decoder3(concat3)
        upconv2 = self.upconv2(decoder3)
        concat2 = torch.cat([upconv2, encoder2], 1)
        decoder2 = self.decoder2(concat2)
        upconv1 = self.upconv1(decoder2)
        concat1 = torch.cat([upconv1, encoder1], 1)
        decoder1 = self.decoder1(concat1)

        out = self.conv(decoder1)
        out = torch.softmax(out, dim=1)  # 应用 softmax

        return out