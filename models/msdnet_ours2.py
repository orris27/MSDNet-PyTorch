import torch.nn as nn
import torch
import math
import pdb

class LastLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LastLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        output = super(LastLinear, self).forward(x)
        return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckSE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=0, downsample=None, has_se=False):
        super(BottleneckSE, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.has_se = has_se
        if self.has_se:
            self.se = SELayer(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.has_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ClassifierModuleOurs(nn.Module):
    def __init__(self, in_channel, hidden_channel, down_scale, num_btn, expansion, num_classes):
        super(ClassifierModuleOurs, self).__init__()
        assert num_btn < 10
        self.num_btn = num_btn
        if self.num_btn == 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            #self.linear = nn.Linear(in_channel, num_classes)
            self.linear = LastLinear(in_channel, num_classes)
        else:
            if down_scale == 1:
                k, s, p = (3, 1, 1)
            elif down_scale == 2:
                k, s, p = (4, 2, 1)
            elif down_scale == 4:
                k, s, p = (6, 4, 1)
            else:
                raise ValueError
            self.scale = nn.Sequential(nn.Conv2d(in_channel, hidden_channel, kernel_size=k, stride=s, padding=p),
                        nn.BatchNorm2d(hidden_channel),
                        nn.ReLU())


            #self.bn1 = BottleneckSE(hidden_channel, hidden_channel// 2, kernel_size=3, stride=2, padding=1,
            #            downsample=nn.Sequential(conv1x1(hidden_channel, hidden_channel * 2, stride=2),
            #                                    nn.BatchNorm2d(hidden_channel * 2)), has_se=True)
            #num_btn = self.num_btn - 1

            self.btns = []
            for i in range(num_btn):
                #self.btns.append(BottleneckSE(hidden_channel * 2, hidden_channel// 2, kernel_size=3, stride=1, padding=1,
                #            downsample=nn.Sequential(conv1x1(hidden_channel * 2, hidden_channel * 2, stride=1),
                #                                    nn.BatchNorm2d(hidden_channel * 2)), has_se=True))
                self.btns.append(BottleneckSE(hidden_channel, hidden_channel// 4, kernel_size=3, stride=1, padding=1,
                            downsample=nn.Sequential(conv1x1(hidden_channel, hidden_channel, stride=1),
                                                    nn.BatchNorm2d(hidden_channel)), has_se=True))
            self.btns = nn.Sequential(*self.btns)
            #self.linear = nn.Linear(hidden_channel * 2 * expansion, num_classes)
            #self.linear = LastLinear(hidden_channel * 2 * expansion, num_classes)
            self.linear = LastLinear(hidden_channel * expansion, num_classes)


    def forward(self, x):
        if self.num_btn == 0:
            out = self.avgpool(x)
        else:
            out = self.scale(x) # 256x14x14
            #out = self.bn1(out) # 512x7x7
            out = self.btns(out) # 512x7x7
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)

class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        #self.linear = nn.Linear(channel, num_classes)
        self.linear = LastLinear(channel, num_classes)

    #def forward(self, x):
    #    res = self.m(x[-1])
    #    res = res.view(res.size(0), -1)
    #    return self.linear(res)

    def forward(self, x):
        res = self.m(x)
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        #down_scales = [4, 4, 2, 2, 1, 1]
        #down_scales = [1, 1, 1, 1, 1, 1]
        down_scales = [2, 2, 2, 2, 2, 2]
        self.bottlenecks = [3, 3, 2, 2, 1, 1]
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            #print('m:', m, 'nIn:', nIn)
            print('nIn:', nIn, 'nIn * ..:', nIn * args.grFactor[-1])
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                #self.classifier.append(
                #    self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
                # (ClassifierModule(in_channel=64 * block.expansion, hidden_channel=256 * block.expansion, down_scale=4, num_btn=bottlenecks[0], expansion=7*7, num_classes=num_classes))
                if i != self.nBlocks - 1:
                    self.classifier.append(
                        #ClassifierModuleOurs(in_channel=nIn * args.grFactor[-1], hidden_channel=128, down_scale=down_scales[i], num_btn=self.bottlenecks[i], expansion=4*4, num_classes=100))
                        ClassifierModuleOurs(in_channel=nIn * args.grFactor[-1], hidden_channel=32, down_scale=down_scales[i], num_btn=self.bottlenecks[i], expansion=4*4, num_classes=100))
                else:
                    self.classifier.append(
                        self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
            elif args.data.startswith('cifar10'):
                assert False
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
            elif args.data == 'ImageNet':
                assert False
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            #for x_ in x:
            #    print(x_.shape)
            #print(x[-1].shape)
            #res.append(self.classifier[i](x))
            res.append(self.classifier[i](x[-1]))
        return res

