import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.discblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode = "reflect", **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True)
        )

    def forward(self, x):
        return self.discblock(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_channels, filters = [64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, filters[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.discblocks = nn.Sequential(
            DiscBlock(filters[0], filters[1], kernel_size = 4, stride = 2),
            DiscBlock(filters[1], filters[2], kernel_size = 4, stride = 2),
            DiscBlock(filters[2], filters[3], kernel_size = 4, stride = 1)
        )

        self.last = nn.Conv2d(filters[3], 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = "reflect")


    def forward(self, x):
        x = self.initial(x)
        x = self.discblocks(x)
        return torch.sigmoid(self.last(x))