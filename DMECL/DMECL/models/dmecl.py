import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class DMECL(nn.Module):
    def __init__(self, num_classes=10, projection_dim=64):
        super().__init__()
        encoder = resnet50(weights=None)
        old_conv = encoder.conv1
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            encoder.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.decode3 = DecoderBlock(2048, 1024, 512)
        self.decode2 = DecoderBlock(512, 512, 256)
        self.decode1 = DecoderBlock(256, 256, 128)
        self.decode0 = DecoderBlock(128, 64, 64)
        self.decode_full = DecoderBlock(64, 1, 32)
        self.segment = nn.Conv2d(32, num_classes, 1)
        self.project = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, projection_dim, 1),
        )

    def forward(self, image):
        stem = self.stem(image)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.decode3(x4, x3)
        x = self.decode2(x, x2)
        x = self.decode1(x, x1)
        x = self.decode0(x, stem)
        x = self.decode_full(x, image)
        return {"logits": self.segment(x), "features": self.project(x)}
