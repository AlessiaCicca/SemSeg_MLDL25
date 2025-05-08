import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
import os

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for p in self.bn1.parameters():
            p.requires_grad = False

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for p in self.bn2.parameters():
            p.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for p in self.bn3.parameters():
            p.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1,
                          padding=padding, dilation=dilation, bias=True)
            )

        for m in self.conv2d_list:
            nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(1, len(self.conv2d_list)):
            out += self.conv2d_list[i](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level=False):
        super(ResNetMulti, self).__init__()
        self.inplanes = 64
        self.multi_level = multi_level

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for p in self.bn1.parameters():
            p.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par)
            )
            for p in downsample[1].parameters():
                p.requires_grad = False

        layers = [block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        if self.training:
            return x, None, None
        else:
            return x

    def get_1x_lr_params_no_scale(self):
        for module in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

    def get_10x_lr_params(self):
        return self.layer6.parameters()

    def optim_parameters(self, lr):
        return [
            {'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
            {'params': self.get_10x_lr_params(), 'lr': 10 * lr}
        ]

#FATTO DA NOI CAPIRE - è PER Backbone: R101 (pre-trained on ImageNet) [2]
#VEDERE SE FATTO GIUSTO
def get_deeplab_v2(num_classes=19, pretrain=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)

    pretrain_model_path = "deepLab_resenet_petrained_imagenet.pth"
    file_id = "1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    if pretrain and not os.path.exists(pretrain_model_path):
        print(">>> Downloading pretrained model...")
        gdown.download(download_url, pretrain_model_path, quiet=False)

    if pretrain:
        print(">>> Loading pretrained weights...")
        saved_state_dict = torch.load(pretrain_model_path, map_location='cpu')
        new_params = model.state_dict()
        for k, v in saved_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' if trained with DataParallel
            if k in new_params and new_params[k].size() == v.size():
                new_params[k] = v
        model.load_state_dict(new_params, strict=False)

    return model
