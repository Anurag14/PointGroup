import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from models.resnet import ResNetBase


class Res16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(in_channels, out_channels, config, D, **kwargs)

    def network_initialization(self, in_channels, out_channels, config, D):
        bn_momentum = 0.1

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], bn_momentum=bn_momentum)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], bn_momentum=bn_momentum)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], bn_momentum=bn_momentum)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], bn_momentum=bn_momentum)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK, self.PLANES[4], self.LAYERS[4], bn_momentum=bn_momentum)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK, self.PLANES[5], self.LAYERS[5], bn_momentum=bn_momentum)

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK, self.PLANES[6], self.LAYERS[6], bn_momentum=bn_momentum)

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK, self.PLANES[7], self.LAYERS[7], bn_momentum=bn_momentum)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_b1p1 = self.relu(out)

        # tensor_stride: 1->2
        out = self.conv1p1s2(out_b1p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # tensor_stride: 2->4
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # tensor_stride: 4->8
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride: 8->16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride: 16->8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride: 8->4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride: 4->2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride: 2->1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p1)
        out = self.block8(out)

        out = self.final(out)

        return out


class Res16UNet14(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14B(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14C(Res16UNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class SmallUNet(ResNetBase):
    BLOCK = BasicBlock
    PLANES = None
    LAYERS = (2, 2, 2)

    def __init__(self, n_planes, config, D=3, **kwargs):
        self.PLANES = (n_planes, 2*n_planes, n_planes)
        super().__init__(None, None, config, D, **kwargs)

    def network_initialization(self, in_channels, out_channels, config, D):
        bn_momentum = 0.1

        self.inplanes = self.PLANES[0]
        self.block1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], bn_momentum=bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], bn_momentum=bn_momentum)

        self.convtr2p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[1], kernel_size=2, stride=2, dimension=D)
        self.bntr2 = ME.MinkowskiBatchNorm(self.PLANES[1], momentum=bn_momentum)

        self.inplanes = self.PLANES[1] + self.PLANES[0] * self.BLOCK.expansion
        self.block3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], bn_momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):

        out_b1p1 = self.block1(x)

        # tensor_stride: 1->2
        out = self.conv1p1s2(out_b1p1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block2(out)

        # tensor_stride: 2->1
        out = self.convtr2p2s2(out)
        out = self.bntr2(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p1)
        out = self.block3(out)

        return out
