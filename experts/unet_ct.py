# models/unet_raindrop.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1): 
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def crop_to_match(self, encoder_feat, decoder_feat):
        _, _, h, w = decoder_feat.shape
        return T.CenterCrop([h, w])(encoder_feat)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.crop_to_match(enc4, dec4)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))

        dec3 = self.upconv3(dec4)
        enc3 = self.crop_to_match(enc3, dec3)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.upconv2(dec3)
        enc2 = self.crop_to_match(enc2, dec2)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.upconv1(dec2)
        enc1 = self.crop_to_match(enc1, dec1)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))

        return F.interpolate(self.out_conv(dec1), size=x.shape[2:], mode="bilinear", align_corners=False)