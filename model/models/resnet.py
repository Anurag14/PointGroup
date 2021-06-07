import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class ResNetBase(ME.MinkowskiNetwork):
    BLOCK = None
    PLANES = (64, 128, 256, 512)
    LAYERS = (2, 2, 2, 2)
    INIT_DIM = 64

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(D=D)
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, config, D):
        # Note: bn_momentum is set for all BN layers except the skip-connection in the res-block
        bn_momentum = 0.1

        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2, bn_momentum=bn_momentum)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, stride=1, bias=True, dimension=D)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion, momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                bn_momentum=bn_momentum,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, bn_momentum=bn_momentum, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)    # tensor_stride: 1->2

        x = self.layer1(x)  # tensor_stride: 2->4
        x = self.layer2(x)  # tensor_stride: 4->8
        x = self.layer3(x)  # tensor_stride: 8->16
        x = self.layer4(x)  # tensor_stride: 16->32
        x = self.final(x)
        return x


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
