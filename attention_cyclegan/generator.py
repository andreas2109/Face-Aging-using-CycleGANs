import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True, downsampling=True, **kwargs):
        super().__init__()

        self.convblock = nn.Sequential(
             nn_utils.spectral_norm(nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs))
            if downsampling else  nn_utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, **kwargs)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity()
        )

    def forward(self, x):
        return self.convblock(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.resblock = nn.Sequential(
            ConvolutionalBlock(channels, channels, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, use_relu=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.resblock(x) + x
    
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy / (proj_key.size(-1) ** 0.5))

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        
        return out, attention
    
class Generator(nn.Module):
    def __init__(self, img_channels, filters=[64, 128, 256], num_residuals=9):
        super().__init__()

        self.downconv = nn.Sequential(
            ConvolutionalBlock(img_channels, filters[0], kernel_size=7, stride=1, padding=3),
            ConvolutionalBlock(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            ConvolutionalBlock(filters[1], filters[2], kernel_size=3, stride=2, padding=1)
        )

        self.attention1 = SelfAttention(filters[2])

        self.resblocks = nn.Sequential(*[ResidualBlock(filters[2]) for _ in range(num_residuals)])

        self.upconv = nn.Sequential(
            ConvolutionalBlock(filters[2], filters[1], downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvolutionalBlock(filters[1], filters[0], downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.last =  nn_utils.spectral_norm(nn.Conv2d(filters[0], img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"))

    def forward(self, x):
        x = self.downconv(x)
        x, attn1 = self.attention1(x)
        x = self.resblocks(x)
        x = self.upconv(x)
        out = torch.tanh(self.last(x))
        return out, attn1