import math
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'ResNet1d',
    'resnet18',
    'resnet34',
    'resnet26',
    'resnet50',
    'resnet101',
    'resnet152',
    'wide_resnet50',
    'wide_resnet101',
    'resnext50',
    'resnext101',
    'resnext152',
    'seresnet18',
    'seresnet34',
    'seresnet26',
    'seresnet50',
    'seresnet101',
    'seresnet152',
    'seresnext50',
    'seresnext101',
    'seresnext152',
]


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L170
class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(drop_prob={round(self.drop_prob, 3):0.3f})'


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite1d(nn.Module):
    '''
    Squeeze and Excite Module.
    '''
    def __init__(self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8):
        super(SqueezeExcite1d, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_down = nn.Conv1d(channels, rd_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv1d(rd_channels, channels, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_se = self.global_pool(x)
        x_se = self.conv_down(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv_up(x_se)
        return x * self.sig(x_se)


class BasicBlock1d(nn.Module):
    '''
    ResNet Block Module.
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 use_se_layer=False, drop_path_rate=0., scale_by_keep=True):
        super(BasicBlock1d, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, outplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate, scale_by_keep)

        # Squeeze and excitation
        self.se_layer = SqueezeExcite1d(outplanes) if use_se_layer else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.se_layer(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.drop_path(x) + residual
        x = self.relu(x)

        return x


class Bottleneck1d(nn.Module):
    '''
    ResNet Block Module (with Bottleneck).
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, use_se_layer=True,
                 drop_path_rate=0., scale_by_keep=True):
        super(Bottleneck1d, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv1d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate, scale_by_keep)

        # Squeeze and excitation
        self.se_layer = SqueezeExcite1d(outplanes) if use_se_layer else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.se_layer(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.drop_path(x) + residual
        x = self.relu(x)

        return x


class ResNet1d(nn.Module):

    def __init__(self, num_leads, block, num_classes, num_blocks_lst, channels=[64, 128, 256, 512],
                 cardinality=1, base_width=64, use_se_layer=False, **kwargs):
        super(ResNet1d, self).__init__()
        self.block = block

        self.expansion = self.block.expansion
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.drop_path_rate = kwargs.get('drop_path_rate', 0.)
        self.drop_out_rate = kwargs.get('drop_out_rate', 0.)

        self.inplanes = 64
        self.total_block_count = sum(num_blocks_lst)
        self.blocks_made = 0

        self.conv1 = nn.Conv1d(self.num_leads, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channels[0], num_blocks_lst[0], stride=1, cardinality=cardinality,
                                       base_width=base_width, use_se_layer=use_se_layer)
        self.layer2 = self._make_layer(channels[1], num_blocks_lst[1], stride=2, cardinality=cardinality,
                                       base_width=base_width, use_se_layer=use_se_layer)
        self.layer3 = self._make_layer(channels[2], num_blocks_lst[2], stride=2, cardinality=cardinality,
                                       base_width=base_width, use_se_layer=use_se_layer)
        self.layer4 = self._make_layer(channels[3], num_blocks_lst[3], stride=2, cardinality=cardinality,
                                       base_width=base_width, use_se_layer=use_se_layer)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3] * self.expansion, self.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, num_blocks, stride=1, cardinality=1, base_width=64, use_se_layer=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )
        layers = []
        block_id = self.blocks_made
        drop_path_rate = self.drop_path_rate * block_id / self.total_block_count
        layers.append(self.block(self.inplanes, planes, stride, downsample, cardinality=cardinality,
                                 base_width=base_width, use_se_layer=use_se_layer, drop_path_rate=drop_path_rate))
        self.inplanes = planes * self.expansion
        for i in range(1, num_blocks):
            block_id = self.blocks_made + i
            drop_path_rate = self.drop_path_rate * block_id / self.total_block_count
            layers.append(self.block(self.inplanes, planes, cardinality=cardinality, base_width=base_width,
                                     use_se_layer=use_se_layer, drop_path_rate=drop_path_rate))
        self.blocks_made += num_blocks
        return nn.Sequential(*layers)

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, feature_only=False):
        x = self.get_feature(x)
        if feature_only:
            return x
        if self.drop_out_rate > 0.:
            x = F.dropout(x, p=self.drop_out_rate, training=self.training)
        x = self.fc(x)
        return x

def _create_resnet(num_leads, **kwargs):
    return ResNet1d(num_leads, **kwargs)

def resnet18(num_leads, num_classes, **kwargs):
    model_args = dict(block=BasicBlock1d, num_classes=num_classes, num_blocks_lst=[2, 2, 2, 2], **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnet34(num_leads, num_classes, **kwargs):
    model_args = dict(block=BasicBlock1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnet26(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[2, 2, 2, 2], **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnet50(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnet101(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 23, 3], **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnet152(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 8, 36, 3], **kwargs)
    return _create_resnet(num_leads, **model_args)

def wide_resnet50(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], base_width=128,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def wide_resnet101(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 23, 3], base_width=128,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnext50(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnext101(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 23, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet(num_leads, **model_args)

def resnext152(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 8, 36, 3], cardinality=32,
                      base_width=4, **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet18(num_leads, num_classes, **kwargs):
    model_args = dict(block=BasicBlock1d, num_classes=num_classes, num_blocks_lst=[2, 2, 2, 2], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet34(num_leads, num_classes, **kwargs):
    model_args = dict(block=BasicBlock1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet26(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[2, 2, 2, 2], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet50(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet101(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 23, 3], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnet152(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 8, 36, 3], use_se_layer=True,
                      **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnext50(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 6, 3], cardinality=32,
                      base_width=4, use_se_layer=True, **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnext101(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 4, 23, 3], cardinality=32,
                      base_width=4, use_se_layer=True, **kwargs)
    return _create_resnet(num_leads, **model_args)

def seresnext152(num_leads, num_classes, **kwargs):
    model_args = dict(block=Bottleneck1d, num_classes=num_classes, num_blocks_lst=[3, 8, 36, 3], cardinality=32,
                      base_width=4, use_se_layer=True, **kwargs)
    return _create_resnet(num_leads, **model_args)
