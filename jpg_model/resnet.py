import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCustom(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetCustom, self).__init__()
        self.in_planes = 4

        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        # 残差块
        self.layer1 = self._make_layer(block, 4, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 8, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2)
        # 全连接层
        self.linear = nn.Linear(32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

