# -*- coding: utf-8 -*-

# Jonas Braun
# jonas.braun@epfl.ch
# 22.02.2021

# copied from Semih Günel's repo https://github.com/NeLy-EPFL/Drosoph2PRegistration

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable


class Warper(nn.Module):
    def __init__(self):
        super(Warper, self).__init__()

    def forward(self, input_img, input_grid):
        warp = input_grid.permute(0, 2, 3, 1)
        output = F.grid_sample(input_img, warp)
        return output


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


# integrate over the predicted grid offset to get the grid(deformation field)
class GridSpatialIntegral(nn.Module):
    def __init__(self, wx, wy=None):
        super(GridSpatialIntegral, self).__init__()
        self.wx = wx
        self.wy = wy if wy is not None else wx
        self.filterx = torch.cuda.FloatTensor(1, 1, 1, self.wx).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1, 1, self.wy, 1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        self.filterx.cuda()
        self.filtery.cuda()

    def forward(self, input_diffgrid):
        fullx = F.conv_transpose2d(
            input_diffgrid[:, 0, :, :].unsqueeze(1), self.filterx, stride=1, padding=0
        )
        fully = F.conv_transpose2d(
            input_diffgrid[:, 1, :, :].unsqueeze(1), self.filtery, stride=1, padding=0
        )
        output_grid = torch.cat(
            (fullx[:, :, 0 : self.wy, 0 : self.wx], fully[:, :, 0 : self.wy, 0 : self.wx]),
            1,
        )
        return output_grid


class UNetSmall(nn.Module):
    def __init__(self, num_channels=1, out_channels=1):
        super(UNetSmall, self).__init__()
        num_feat = [32, 64, 128, 256]

        self.down1 = nn.Sequential(Conv3x3Small(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_feat[0]),
            Conv3x3Small(num_feat[0], num_feat[1]),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_feat[1]),
            Conv3x3Small(num_feat[1], num_feat[2]),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_feat[2]),
            Conv3x3Small(num_feat[2], num_feat[3]),
            nn.BatchNorm2d(num_feat[3]),
        )

        self.up1 = UpSample(num_feat[3], num_feat[2])
        self.upconv1 = nn.Sequential(
            Conv3x3Small(num_feat[3] + num_feat[2], num_feat[2]),
            nn.BatchNorm2d(num_feat[2]),
        )

        self.up2 = UpSample(num_feat[2], num_feat[1])
        self.upconv2 = nn.Sequential(
            Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1]),
            nn.BatchNorm2d(num_feat[1]),
        )

        self.up3 = UpSample(num_feat[1], num_feat[0])
        self.upconv3 = nn.Sequential(
            Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]),
            nn.BatchNorm2d(num_feat[0]),
        )

        self.final = nn.Sequential(nn.Conv2d(num_feat[0], out_channels, kernel_size=1))

    def forward(self, inputs, return_features=False):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        bottom_feat = self.bottom(down3_feat)

        # print(bottom_feat.size())
        up1_feat = self.up1(bottom_feat, down3_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)
        # print(up1_feat.size())
        up2_feat = self.up2(up1_feat, down2_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)
        # print(up2_feat.size())
        up3_feat = self.up3(up2_feat, down1_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())

        outputs = self.final(up3_feat)

        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ELU()
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

