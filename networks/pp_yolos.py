import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from Pytorch.photo_detection.photo_detection_YOLOv3.PP_YOLO.networks.DCN import DeformConv2D


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.bottleneck = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
class SPP(nn.Module):
    def __init__(self, pool_sizes=[1, 5, 9, 13]):
        super(SPP, self).__init__()  # 3种卷积核的最大池化
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        # MaxPool2d（卷积核，步长，填充）

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]  # 从0开始不要最后一个数值
        features = torch.cat(features + [x], dim=1)  # 3次特征和直接连接线的融合叠加

        return features


#   卷积块
#   CONV+BATCHNORM+ACTIVATION
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


#   最后获得yolov4的输出   一个3x3的卷积 + 一个1x1的卷积   （特征的整合 -> yolov4的检测结果）
def yolo_head(filters_list, out_filter):  # （[2*输入，通道数]，输入）
    m = nn.Sequential(
        BaseConv(filters_list[0], filters_list[1], 3),  # 卷积 + bn + Leak_Relu
        # DropBlock2D(block_size=3, drop_prob=0.3),  # 加入DropBlock2D 防止过拟合
        nn.Conv2d(filters_list[1], out_filter, 1, 1, 0, bias=True),  # 单纯的卷积层 nn.Conv2d(输入，输入/2，卷积核，步长)
    )
    return m


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv_bn_relu = BaseConv(3, 64, 7, 2)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layers = [4, 3, 6, 3]

        self.layer1 = self._make_layer(64, 4)
        self.layer2 = self._make_layer(128, 3, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None  # 初始化变量
        if stride != 1 or self.inplanes != 4 * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, 4 * planes, 1, stride, bias=False),
                nn.BatchNorm2d(4 * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c5, c4, c3


def ResNets(pretrained, **kwargs):
    model = ResNet()
    if pretrained:
        model.load_state_dict(torch.load(pretrained))  # 加载预训练模型
    return model


class ResNet50(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(ResNet50, self).__init__()
        self.backbone = ResNets(None)
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        self.spp = SPP()
        self.latlayer = nn.Sequential(
            nn.Conv2d(1280, 256, 1, 1, 0),
            # DropBlock2D(block_size=3, drop_prob=0.3),  # 加入DropBlock2D 防止过拟合
        )
        self.latlayer1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, 1, 0),
            # DropBlock2D(block_size=3, drop_prob=0.3),  # 加入DropBlock2D 防止过拟合
        )
        self.latlayer2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            # DropBlock2D(block_size=3, drop_prob=0.3),  # 加入DropBlock2D 防止过拟合
        )

        final_out_filter = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, 512], final_out_filter)
        self.yolo_head2 = yolo_head([256, 256], final_out_filter)
        self.yolo_head1 = yolo_head([256, 128], final_out_filter)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c5, c4, c3 = self.backbone(x)
        p5 = self.toplayer(c5)
        p5 = self.spp(p5)
        p5 = self.latlayer(p5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))

        p5 = self.yolo_head3(p5)
        p4 = self.yolo_head2(p4)
        p3 = self.yolo_head1(p3)
        return p5, p4, p3
