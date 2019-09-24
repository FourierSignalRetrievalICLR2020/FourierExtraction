import torch
import numpy                  as np
import torch.nn               as nn
import torch.optim            as optim
import torch.nn.functional    as F
import torchvision.transforms as transforms
import torch.utils.checkpoint as cp
import torchvision
import sys
from .                        import configcifar as cf
sys.path.append('..')
from   complexmodels          import ComplexConv, get_real, get_imag
from   complexnorm            import ComplexBN as CBN
from   complexnorm            import ComplexLN as CLN
from   torch.autograd         import Variable
import math
import os
import time


# Copied from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def complexconv3x3(in_planes, out_planes, stride=(1, 1), dilation=(1, 1), padding=(1, 1),
                   init_criterion='he', weight_init='complex', seed=2018):
    """3x3 complex convolution with padding"""
    return ComplexConv(in_channels=in_planes, out_channels=out_planes, 
                       kernel_size=(3, 3), stride=stride,
                       dilation=dilation, padding=padding, bias=False,
                       init_criterion=init_criterion, weight_init=weight_init,
                       seed=seed)


def complexconv1x1(in_planes, out_planes, stride=(1, 1), dilation=(1, 1), padding=(0, 0),
                   init_criterion='he', weight_init='complex', seed=2018):
    """3x3 complex convolution with padding"""
    return ComplexConv(in_channels=in_planes, out_channels=out_planes, 
                       kernel_size=(1, 1), stride=stride,
                       dilation=dilation, padding=padding, bias=False,
                       init_criterion=init_criterion, weight_init=weight_init,
                       seed=seed)


def conv3x3(in_planes, out_planes, stride=(1, 1), padding=(1, 1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


# The following is taken from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py#L12
def _bn_act_conv(norm, relu, conv):
    def comp_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return comp_function


class RealBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4, stochastic_drop=0.5, seed=2018):
        super(RealBottleneck, self).__init__()
        self.expansion  = expansion
        self.conv1      = conv1x1(inplanes, planes)
        self.bn1        = nn.BatchNorm2d(   planes)
        self.conv2      = conv3x3(planes,   planes, stride)
        self.bn2        = nn.BatchNorm2d(   planes)
        self.conv3      = conv1x1(planes,   planes * self.expansion)
        self.bn3        = nn.BatchNorm2d(   planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride
        self.stochastic_drop = stochastic_drop
        self.rng             = np.random.RandomState(seed)

    def forward(self, x):
        residual = x
        if self.stochastic_drop is not None:
            perform_computation = self.rng.binomial(size=1, n=1, p=self.stochastic_drop)[0]
        else:
            perform_computation = True
        if self.training == False or perform_computation:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            if self.training == False and self.stochastic_drop is not None:
                out *= self.stochastic_drop
            if self.downsample is not None:
                residual     = self.downsample(x)
            out     += residual
            out      = self.relu(out)
            return out
        elif self.downsample is     None:
            return x
        elif self.downsample is not None:
            return self.downsample(x)


class RealBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, stochastic_drop=0.5, seed=2018):
        super(RealBasicBlock, self).__init__()
        self.conv1           = conv3x3(inplanes, planes, stride)
        self.bn1             = nn.BatchNorm2d(   planes)
        self.relu            = nn.ReLU(inplace=True)
        self.conv2           = conv3x3(planes,   planes)
        self.bn2             = nn.BatchNorm2d(   planes)
        self.downsample      = downsample
        self.stride          = stride
        self.stochastic_drop = stochastic_drop
        self.rng             = np.random.RandomState(seed)

    def forward(self, x):
        residual     =              x
        if self.stochastic_drop is not None:
            perform_computation = self.rng.binomial(size=1, n=1, p=self.stochastic_drop)[0]
        else:
            perform_computation = True

        if self.training == False or perform_computation:
            out      = self.bn1(    x)
            out      = self.relu( out)
            out      = self.conv1(out)
            out      = self.bn2(  out)
            out      = self.relu( out)
            out      = self.conv2(out)
            if self.training == False and self.stochastic_drop is not None:
                out *= self.stochastic_drop
            if self.downsample is not None:
                residual     = self.downsample(x)
            out     += residual
            return out
        elif self.downsample is     None:
            return x
        elif self.downsample is not None:
            return self.downsample(x)



class learnConcatRealImagBlock(nn.Module):
    """Learn initial imaginary component for input."""
    def __init__(self, inplanes):
        super(learnConcatRealImagBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn2   = nn.BatchNorm2d(inplanes)
        self.conv2 = conv3x3(inplanes, inplanes)

    def forward(self, x):
        out = self.bn1(    x)
        out = self.relu( out)
        out = self.conv1(out)
        out = self.bn2(  out)
        out = self.relu( out)
        out = self.conv2(out)
        out = torch.cat([x, out], dim=1)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None,
                 init_criterion='he', weight_init='complex', seed=2018,
                 stochastic_drop=0.5, dimbn=1, eps=1e-4, momentum=0.1, scale=True,
                 center=True, track_running_stats=True, normalization='layernorm'):
        super(BasicBlock, self).__init__()
        if normalization == 'bn':
            self.bn1 = CBN(
                num_complex_features=inplanes, dim=dimbn, eps=eps, momentum=momentum,
                scale=scale, center=center, track_running_stats=track_running_stats
            )
            self.bn2 = CBN(
                num_complex_features=planes,   dim=dimbn, eps=eps, momentum=momentum,
                scale=scale, center=center, track_running_stats=track_running_stats
            )
        elif normalization == 'layernorm':
            self.bn1 = CLN(
                num_complex_features=inplanes, dim=dimbn, eps=eps,
                scale=scale, center=center
            )
            self.bn2 = CLN(
                num_complex_features=planes,   dim=dimbn, eps=eps,
                scale=scale, center=center
            )
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = complexconv3x3(inplanes, planes, stride=stride,
                                    init_criterion=init_criterion,
                                    weight_init=weight_init, seed=seed)
        self.conv2 = complexconv3x3(planes, planes, stride=(1, 1),
                                    init_criterion=init_criterion,
                                    weight_init=weight_init, seed=seed)
        self.downsample      = downsample
        self.stride          = stride
        self.stochastic_drop = stochastic_drop
        self.rng             = np.random.RandomState(seed)

    def forward(self, x):
        residual     =              x
        if self.stochastic_drop is not None:
            perform_computation = self.rng.binomial(size=1, n=1, p=self.stochastic_drop)[0]
        else:
            perform_computation = True

        if self.training == False or perform_computation:
            #####
            """#this is added to test the block without cbn:
            out      = self.relu(    x)
            #we have to remove it after testing
            #####"""
            out      = self.bn1(    x)
            out      = self.relu( out)
            out      = self.conv1(out)
            out      = self.bn2(  out)
            out      = self.relu( out)
            out      = self.conv2(out)
            if self.training == False and self.stochastic_drop is not None:
                out *= self.stochastic_drop
            if self.downsample is not None:
                residual   = self.downsample(x)
                real_out   = get_real(out,            input_type='convolution')
                imag_out   = get_imag(out,            input_type='convolution')
                real_res   = get_real(residual,       input_type='convolution')
                imag_res   = get_imag(residual,       input_type='convolution')
                out_real   = torch.cat([real_res, real_out], dim=1)
                out_imag   = torch.cat([imag_res, imag_out], dim=1)
                out        = torch.cat([out_real, out_imag], dim=1)
            else:
                out       += residual
                #if we do not want to have identity connection then return out without adding the residual part.
            return out
        elif self.downsample is     None:
            return x
        elif self.downsample is not None:
            raise Exception(
                "dowsampling in a resnet block cannot be performed while a stochastic depth"
                " is activated during training. Found perform_computation == " + str(perform_computation) +
                " and downsample is not None."
            )


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, init_criterion='he',
                 weight_init='complex', expansion=2, typebtl='down', seed=2018,
                 stochastic_drop=0.5, dimbn=1, eps=1e-4, momentum=0.1, scale=True,
                 center=True, track_running_stats=True, normalization='layernorm'):
        super(Bottleneck, self).__init__()
        self.expansion  = expansion
        self.typebtl    = typebtl
        self.bottleneck_size = planes * self.expansion
        self.stochastic_drop = stochastic_drop
        self.rng             = np.random.RandomState(seed)
        self.normalization   = normalization
        if   self.normalization == 'bn':
            self.bn1 = CBN(
                num_complex_features=inplanes, dim=dimbn, eps=eps, momentum=momentum,
                scale=scale, center=center, track_running_stats=track_running_stats
            )
        elif self.normalization == 'layernorm':
            self.bn1 = CLN(
                num_complex_features=inplanes, dim=dimbn, eps=eps,
                scale=scale, center=center
            )
        self.conv1 = ComplexConv(
            in_channels=inplanes, out_channels=inplanes,
            kernel_size=(3, 3), stride=(1, 1),
            dilation=(1, 1), padding=(1, 1),
            bias=False, init_criterion=init_criterion,
            weight_init=weight_init, seed=seed
        )
        if   self.normalization == 'bn':
            self.bn2 = CBN(
                num_complex_features=inplanes, dim=dimbn, eps=eps, momentum=momentum,
                scale=scale, center=center, track_running_stats=track_running_stats
            )
        elif self.normalization == 'layernorm':
             self.bn2 = CLN(
                num_complex_features=inplanes, dim=dimbn, eps=eps,
                scale=scale, center=center
            )
        self.conv2 = ComplexConv(
            in_channels=inplanes, out_channels=self.bottleneck_size, kernel_size=(1, 1), stride=stride,
            padding=(0, 0), dilation=(1, 1), bias=False, init_criterion=init_criterion,
            weight_init=weight_init, seed=seed
        )
        if   self.normalization == 'bn':
            self.bn3 = CBN(
                num_complex_features=self.bottleneck_size,
                dim=dimbn, eps=eps, momentum=momentum,
                scale=scale, center=center,
                track_running_stats=track_running_stats
            )
        elif self.normalization == 'layernorm':
            self.bn3 = CLN(
                num_complex_features=self.bottleneck_size,
                dim=dimbn, eps=eps,
                scale=scale, center=center
            )
        self.conv3 = ComplexConv(  # we can try to replace the kernel here by a 3x3 instead of 1x1
            in_channels=self.bottleneck_size,
            out_channels=planes,
            kernel_size=(3, 3), stride=(1, 1),
            dilation=(1, 1), padding=(1, 1),
            bias=False, init_criterion=init_criterion,
            weight_init=weight_init, seed=seed
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.stochastic_drop is not None:
            perform_computation = self.rng.binomial(size=1, n=1, p=self.stochastic_drop)[0]
        else:
            perform_computation = True
        if self.training == False or perform_computation:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.conv3(out)
            if self.training == False and self.stochastic_drop is not None:
                out *= self.stochastic_drop
            if self.downsample is not None:
                residual   = self.downsample(x)
                real_out   = get_real(out,            input_type='convolution')
                imag_out   = get_imag(out,            input_type='convolution')
                real_res   = get_real(residual,       input_type='convolution')
                imag_res   = get_imag(residual,       input_type='convolution')
                out_real   = torch.cat([real_res, real_out], dim=1)
                out_imag   = torch.cat([imag_res, imag_out], dim=1)
                out        = torch.cat([out_real, out_imag], dim=1)
            else:
                out       += residual
            return out
        elif self.downsample is     None:
            return x
        elif self.downsample is not None:
            raise Exception(
                "dowsampling in a resnet block cannot be performed while a stochastic depth"
                " is activated during training. Found perform_computation == " + str(perform_computation) +
                " and downsample is not None."
            )


class BasicDenseBlock(nn.Module):
    def __init__(self, inplanes, growth_rate, stride=(1, 1), downsample=None, init_criterion='he',
                 weight_init='complex', expansion=4, bottleneck=False, efficient=True, seed=2018,
                 dimbn=1, eps=1e-4, momentum=0.1, scale=True, center=True, track_running_stats=True):
        super(BasicDenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.stride      = stride
        self.expansion   = expansion
        self.bottleneck  = bottleneck
        self.downsample  = downsample
        if self.bottleneck:
            self.bottleneck_size = inplanes * self.expansion
        else:
            self.bottleneck_size = inplanes
        self.bn1         = CBN(
            num_complex_features=inplanes, dim=dimbn, eps=eps, momentum=momentum,
            scale=scale, center=center, track_running_stats=track_running_stats
        )
        if self.bottleneck:
            self.convbtl = complexconv1x1(inplanes, self.bottleneck_size, stride=(1, 1), padding=(0, 0),
                                          init_criterion=init_criterion, weight_init=weight_init, seed=seed)
            self.bnbtl   = CBN(
                num_complex_features=self.bottleneck_size, dim=dimbn, eps=eps, momentum=momentum,
            scale=scale, center=center, track_running_stats=track_running_stats
            )
        lastconv         = complexconv3x3 if self.stride not in {2, (2, 2)} else complexconv1x1
        padding          = (0, 0) if lastconv == complexconv1x1 else (1, 1)
        output_fmaps     = self.growth_rate if self.downsample is None else self.growth_rate // 2  # self.downsample should output also self.growth_rate // 2 fmaps
        self.conv1       = lastconv(self.bottleneck_size, output_fmaps, stride=self.stride, padding=padding,
                                    init_criterion=init_criterion, weight_init=weight_init, seed=seed)
        self.relu1       = nn.ReLU(inplace=True)
        self.relubtl     = nn.ReLU(inplace=True)
        self.efficient   = efficient

    def forward(self, *x):
        if self.bottleneck:
            block_function   = _bn_act_conv(self.bn1, self.relu1, self.convbtl)
        else:
            block_function   = _bn_act_conv(self.bn1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in x[0]):
            out1 = cp.checkpoint(block_function, *(x[0] + x[1]))
        else:
            out1 = block_function(*(x[0] + x[1]))
        if self.bottleneck:
            out2 = self.conv1(self.relubtl(self.bnbtl(out1)))
            return out2
        else:
            return out1


class RealBasicDenseBlock(nn.Module):
    def __init__(self, inplanes, growth_rate, stride=(1, 1), downsample=None,
                 expansion=4, bottleneck=False, efficient=True):
        super(RealBasicDenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.stride      = stride
        self.expansion   = expansion
        self.bottleneck  = bottleneck
        self.downsample  = downsample
        if self.bottleneck:
            self.bottleneck_size = inplanes * self.expansion
        else:
            self.bottleneck_size = inplanes
        self.bn1         = nn.BatchNorm2d(
            num_features=inplanes, eps=1e-4, momentum=0.1,
            affine=True, track_running_stats=True
        )
        if self.bottleneck:
            self.convbtl = conv1x1(inplanes, self.bottleneck_size, stride=(1, 1), padding=(0, 0))
            self.bnbtl   = nn.BatchNorm2d(
                num_features=self.bottleneck_size, eps=1e-4, momentum=0.1,
                affine=True, track_running_stats=True
            )
        lastconv         = conv3x3 if self.stride not in {2, (2, 2)} else conv1x1
        padding          = (0, 0) if lastconv == complexconv1x1 else (1, 1)
        output_fmaps     = self.growth_rate
        self.conv1       = lastconv(self.bottleneck_size, output_fmaps, stride=self.stride, padding=padding)
        self.relu1       = nn.ReLU(inplace=True)
        self.relubtl     = nn.ReLU(inplace=True)
        self.efficient   = efficient

    def forward(self, *x):
        if self.bottleneck:
            block_function   = _bn_act_conv(self.bn1, self.relu1, self.convbtl)
        else:
            block_function   = _bn_act_conv(self.bn1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in x):
            out1 = cp.checkpoint(block_function, *x)
        else:
            out1 = block_function(*x)
        if self.bottleneck:
            out2 = self.conv1(self.relubtl(self.bnbtl(out1)))
            return out2
        else:
            return out1


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.layer0 = learnConcatRealImagBlock(inplanes=3)
        self.bn1 = CBN(
            num_complex_features=3, dim=1, eps=1e-4, momentum=0.1,
            scale=True, center=True, track_running_stats=True
        )
        self.conv1 = ComplexConv(
            in_channels=3, out_channels=self.inplanes,
            kernel_size=(1, 1), stride=(1, 1),
            dilation=(1, 1), padding=(0, 0),
            bias=False, init_criterion='he',
            weight_init='complex', seed=None
        )
        self.layer1  = self._make_layer(block, self.inplanes,     blocks=layers[0], stride=(1, 1))
        self.layer2  = self._make_layer(block, self.inplanes,     blocks=layers[1], stride=(2, 2))
        self.layer3  = self._make_layer(block, self.inplanes * 2, blocks=layers[2], stride=(1, 1))
        self.layer4  = self._make_layer(block, self.inplanes * 2, blocks=layers[3], stride=(2, 2))
        self.layer5  = self._make_layer(block, self.inplanes * 4, blocks=layers[4], stride=(1, 1))
        self.avgpool = nn.AvgPool2d(8, stride=(1, 1))
        self.fc      = nn.Linear(self.inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None
        if stride != (1, 1):
            downsample = ComplexConv(
                in_channels=planes, out_channels=planes,
                kernel_size=(1, 1), stride=stride,
                dilation=(1, 1), padding=(0, 0),
                bias=False, init_criterion='he',
                weight_init='complex', seed=None
            ) # convolution of type valid

        layers = []
        layers.append(block(planes, planes, stride, downsample))
        for i in range(1, blocks):
            if downsample is None:
                layers.append(block(planes, planes))
            else:
                layers.append(block(planes * 2, planes * 2))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.bn1(    out)
        out = self.conv1(  out)
        out = self.layer1( out)
        out = self.layer2( out)
        out = self.layer3( out)
        out = self.layer4( out)
        out = self.layer5( out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def complexresnet(**kwargs):
    """Constructs a complex ResNet model.
    """
    model = ResNet(BasicBlock, [2, 1, 3, 1, 6], **kwargs)
    return model


# The following main function is taken from:
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py
def train(epoch, net, trainloader, num_epochs, batch_size, criterion, trainset, lr=0.1):
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    lr         = cf.deep_complex_lr(epoch)
    optimizer  = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    use_cuda   = torch.cuda.is_available()

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()          # GPU settings
        optimizer.zero_grad()
        inputs, targets     = Variable(inputs), Variable(targets)
        outputs             = net(inputs)                            # Forward Propagation
        loss                = criterion(outputs, targets)            # Loss
        loss.backward()                                              # Backward Propagation
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1) # Clipping gradient norm
        optimizer.step()                                             # Optimizer update
        torch.cuda.empty_cache()                                     # Empty Memory cache
        train_loss  += loss.data.item()                              # counting accumulative loss
        _, predicted = torch.max(outputs.data, 1)
        total       += targets.size(0)
        correct     += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data.item(), (100.*correct/total)))
        sys.stdout.flush()

def test(epoch, net, testloader, criterion, best_acc):
    net.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    use_cuda  = torch.cuda.is_available()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets     = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs         = net(inputs)
            loss            = criterion(outputs, targets)
            torch.cuda.empty_cache()
            test_loss      += loss.data.item()
            _, predicted    = torch.max(outputs.data, 1)
            total          += targets.size(0)
            correct        += predicted.eq(targets.data).cpu().sum()
    
    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.4f%%" %(epoch, loss.data.item(), acc))

    if acc > best_acc:
        print('| Best model...\t\t\tTop1 = %.4f%%' %(acc))
        best_acc = acc

    return best_acc


def main():
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    lr       = 0.1
    start_epoch, num_epochs, batch_size, optim_type = 0, 200, 64, 'SGD'

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    """
    Resnet paper:
        '4 pixels are padded on each side,
        and a 32×32 crop is randomly sampled from the padded
        image or its horizontal flip. For testing, we only evaluate
        the single view of the original 32×32 image.' 
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # the input size would be 32 x 32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
    ])

    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True,  transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    print('\n[Phase 2] : Model setup')
    print('| Building the network ...')
    net = complexresnet()
    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        criterion = nn.CrossEntropyLoss()
    net_total_params     = sum(p.numel() for p in net.parameters())
    net_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total     Number of parameters of the model: " + str(net_total_params))
    print("Trainable Number of parameters in the model: " + str(net_trainable_params))

    # Training
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(lr))
    print('| Optimizer = ' + optim_type)

    elapsed_time = 0
    best_acc     = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(           epoch, net, trainloader, num_epochs, batch_size, criterion, trainset, lr)
        best_acc = test( epoch, net, testloader,  criterion,  best_acc)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' %(best_acc))


if __name__ == "__main__":
    main()
