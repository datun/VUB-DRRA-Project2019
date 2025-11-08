import torch.nn as nn
import torch.nn.functional as fn


class ESPCN(nn.Module):
    def __init__(self, upscale, nChannel):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(nChannel, 64, kernel_size=5, padding=5//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=3//2)
        self.conv3 = nn.Conv2d(32, nChannel * (upscale**2), kernel_size=3, padding=3//2)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = fn.leaky_relu(self.conv1(x),0.01)
        x = fn.leaky_relu(self.conv2(x),0.01)
        x = fn.leaky_relu(self.pixel_shuffle(self.conv3(x)),0.01)
        return x