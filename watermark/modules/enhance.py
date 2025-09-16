import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNet(nn.Module):
    """Standard UNet architecture with configurable depth and activation"""
    def __init__(self, in_channels, out_channels, base_channels=32, output_act='tanh'):
        super().__init__()
        
        # Downsampling path
        self.down1 = self._make_block(in_channels, base_channels)
        self.down2 = self._make_down_block(base_channels, base_channels*2)
        self.down3 = self._make_down_block(base_channels*2, base_channels*4)
        self.down4 = self._make_down_block(base_channels*4, base_channels*8)
        self.down5 = self._make_down_block(base_channels*8, base_channels*16)
        
        # Upsampling path
        self.up4 = self._make_up_block(base_channels*16, base_channels*8)
        self.up3 = self._make_up_block(base_channels*8, base_channels*4)
        self.up2 = self._make_up_block(base_channels*4, base_channels*2)
        self.up1 = self._make_up_block(base_channels*2, base_channels)
        
        # Output layer
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        self.activation = self._get_activation(output_act)
        
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._make_block(in_ch, out_ch)
        )
    
    def _make_up_block(self, in_ch, out_ch):
        return UpBlock(in_ch, out_ch)
    
    def _get_activation(self, name):
        return {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity(),
        }.get(name, nn.Softplus())
    
    def forward(self, x):
        # Encoding
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Decoding
        u4 = self.up4(d4, d5)
        u3 = self.up3(d3, u4)
        u2 = self.up2(d2, u3)
        u1 = self.up1(d1, u2)
        
        return self.activation(self.output(u1))
    

class OLDUNet(nn.Module):
    """Standard UNet architecture with configurable depth and activation"""
    def __init__(self, in_channels, out_channels, base_channels=32, output_act='tanh'):
        super().__init__()
        
        # Downsampling path
        self.down1 = ConvDoubleBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels*2)
        self.down3 = DownBlock(base_channels*2, base_channels*4)
        self.down4 = DownBlock(base_channels*4, base_channels*8)
        self.down5 = DownBlock(base_channels*8, base_channels*16)
        
        # Upsampling path
        self.up4 = UpBlock(base_channels*16, base_channels*8)
        self.up3 = UpBlock(base_channels*8, base_channels*4)
        self.up2 = UpBlock(base_channels*4, base_channels*2)
        self.up1 = UpBlock(base_channels*2, base_channels)
        
        # Output layer
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        self.activation = self._get_activation(output_act)
        
    def _get_activation(self, name):
        return {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity(),
        }.get(name, None)
    
    def forward(self, x):
        # Encoding
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Decoding
        u4 = self.up4(d4, d5)
        u3 = self.up3(d3, d4)
        u2 = self.up2(d2, u3)
        u1 = self.up1(d1, u2)
        
        if self.activation:
            return self.activation(self.output(u1))
        else:
            return self.output(u1)


class DownBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvDoubleBlock(in_num_ch, out_num_ch, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class UpBlock(nn.Module):
    """UNet upsampling block with bilinear interpolation"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, in_ch//2, 2)
        )
        self.conv = ConvDoubleBlock(in_ch, out_ch)

    def forward(self, x_down, x_up):
        x_up = self.up(x_up)
        x_up = F.pad(x_up, (0,1,0,1), mode='replicate')
        x = torch.cat([x_down, x_up], dim=1)
        return self.conv(x)
    

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3):
        super(ConvDoubleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    # Test models
    unet = UNet(16, 16)
    x = torch.rand(1, 16, 32, 32)
    print(f"Output shape: {unet(x).shape}")