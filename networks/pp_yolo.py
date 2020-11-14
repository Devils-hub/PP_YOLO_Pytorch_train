import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class Bottleneck(nn.Module):
    expansion = 4

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


#   卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):  # （输入，输入/2）
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BaseConv(in_channels, out_channels, 1),  # conv2d(输入，输入/2，卷积核，步长)
            nn.Upsample(scale_factor=2, mode='nearest')  # Upsample(放大的倍数，上采样方法'最近邻') 最近邻是默认的
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


#   卷积块
#   CONV+BATCHNORM+ACTIVATION
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#   Darknet的结构块的组成部分
#   内部堆叠的残差块
class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()

        self.block = nn.Sequential(
            BaseConv(channels, channels, 1, 1),
            BaseConv(channels, channels, 3, 1),
            BaseConv(channels, channels, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


#   五次卷积块
def make_five_conv(filters_list, in_filters):  # （[输出，输出*2]，输入）
    m = nn.Sequential(
        BaseConv(in_filters, filters_list[0], 1),  # conv2d(输入，输入/2，卷积核，步长)
        BaseConv(filters_list[0], filters_list[1], 3),
        BaseConv(filters_list[1], filters_list[0], 1),
        BaseConv(filters_list[0], filters_list[1], 3),
        BaseConv(filters_list[1], filters_list[0], 1),
    )
    return m


#   最后获得yolov4的输出   一个3x3的卷积 + 一个1x1的卷积   （特征的整合 -> yolov4的检测结果）
def yolo_head(filters_list, out_filter):  # （[2*输入，通道数]，输入）
    m = nn.Sequential(
        BaseConv(filters_list[0], filters_list[1], 3),  # 卷积 + bn + Leak_Relu
        nn.Conv2d(filters_list[1], out_filter, 1, 1, 0, bias=True),  # 单纯的卷积层 nn.Conv2d(输入，输入/2，卷积核，步长)
    )
    return m


class ResNet50s(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(ResNet50s, self).__init__()
        # self.conv = BaseConv(3, 64, 7, 2)
        # self.maxpooling = nn.MaxPool2d(3, 2, 1)
        self.conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(3, 2, 1)

        # self.conv1 = nn.Sequential(*[Resblock(64) for _ in range(3)])  # 3层卷积
        # self.conv2 = nn.Sequential(*[Resblock(128) for _ in range(4)])  # 4层卷积
        # self.conv3 = nn.Sequential(*[Resblock(256) for _ in range(6)])  # 6层卷积
        # self.conv4 = nn.Sequential(*[Resblock(512) for _ in range(3)])  # 3层卷积
        self.conv1 = nn.Sequential(*[Resblock(256) for _ in range(3)])  # 3层卷积
        self.conv2 = nn.Sequential(*[Resblock(512) for _ in range(4)])  # 4层卷积
        self.conv3 = nn.Sequential(*[Resblock(1024) for _ in range(6)])  # 6层卷积
        self.conv4 = nn.Sequential(*[Resblock(2048) for _ in range(3)])  # 3层卷积

        final_out_filter = num_anchors * (5 + num_classes)

        self.latlayer1 = nn.Conv2d(2048, 256, 1, 1, 0)
        self.upsample1 = Upsample(2048, 1024)
        self.yolo1 = yolo_head([2048, 256], final_out_filter)

        self.latlayer2 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.upsample2 = Upsample(1024, 512)
        self.yolo2 = yolo_head([1024, 256], final_out_filter)

        self.latlayer3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.upsample3 = Upsample(512, 256)
        self.yolo3 = yolo_head([512, 256], final_out_filter)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        maxpooling = self.maxpooling(relu)
        # maxpooling = self.maxpooling(conv1)
        conv2 = self.conv1(maxpooling)
        conv3 = self.conv3(conv2)
        conv4 = self.conv2(conv3)
        conv5 = self.conv4(conv4)

        out3 = self.latlayer3(conv5)
        out3s = self.upsample3(out3)
        head3 = self.yolo3(out3)

        out2 = self.latlayer2(out3s)
        out2 = self.upsample2(out2)
        head2 = self.yolo2(out2)

        out1 = self.latlayer1(out2)
        out1 = self.upsample1(out1)
        head1 = self.yolo1(out1)

        return head3, head2, head1


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # block即为Bottleneck模型，layers可控制传入的Bottleneck
        self.inplanes = 64  # 初始输入通道数为64
        super(ResNet, self).__init__()  # 可见ResNet也是nn.Module的子类
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 四层残差块,64为这一层输入的通道数,layer[0]表示有几个残差块
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = self.nn.AvgPool2d(7)  # 这里默认stride为7
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # _make_layer方法用来构建ResNet网络中的4个blocks
    # _make_layer方法的第一个输入block是Bottleneck类
    # 第二个输入是该blocks输出的channels
    # 第三个输入是每个blocks中包含多少个residual子结构
    def _make_layer(self, block, planes, blocks, stride=1):  # 这里的blocks相当于layers

        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维
        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        downsample = None
        # 这个步骤是为了匹配维度
        # 在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # stride不为1时，残差结构输出纬度变化
        # 输入通道数不为输出通道数的四分之一，也需要downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes,
                                                 planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion), )

        layers = []

        # 只在这里传递了stride=2的参数，因而一个box_block中的图片大小只在第一次除以2
        layers.append(block(self.inplanes, planes, stride, downsample))  # 将每个blocks的第一个residual结构保存在layers列表中
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 该部分是将每个blocks的剩下residual结构保存在layers列表中，这样就完成了一个blocks的构造
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size[0], -1)  # 将输出结果展成一行
        # x = self.fc(x)
        #
        # return x
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        return x3, x2, x1


def ResNet50(pretrained, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(torch.load(pretrained))  # 加载预训练模型
    return model
