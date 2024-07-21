import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu = True, downsampling = True, **kwargs):
        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode = "reflect", **kwargs)
            if downsampling else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True) if use_relu else nn.Identity()
        )
    
    def forward(self, x):
        return self.convblock(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.resblock = nn.Sequential(
            ConvolutionalBlock(channels, channels, kernel_size = 3, padding = 1),
            ConvolutionalBlock(channels, channels, use_relu = False, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.resblock(x) + x
    
class Generator(nn.Module):
    def __init__(self, img_channels, filters = [64, 128, 256], num_residuals = 9):
        super().__init__()

        self.downconv = nn.Sequential(
            ConvolutionalBlock(img_channels, filters[0], kernel_size = 7, stride = 1, padding = 3),
            ConvolutionalBlock(filters[0], filters[1], kernel_size = 3, stride = 2, padding = 1),
            ConvolutionalBlock(filters[1], filters[2], kernel_size = 3, stride = 2, padding = 1)
        )

        self.resblocks = [ResidualBlock(filters[2]) for _ in range(num_residuals)]
        self.residual_blocks = nn.Sequential(*self.resblocks)

        self.upconv = nn.Sequential(
            ConvolutionalBlock(filters[2], filters[1], downsampling = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            ConvolutionalBlock(filters[1], filters[0], downsampling = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        )

        self.last = nn.Conv2d(filters[0], img_channels, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect")

    def forward(self, x):
        x = self.downconv(x)
        x = self.residual_blocks(x)
        x = self.upconv(x)
        return torch.tanh(self.last(x))