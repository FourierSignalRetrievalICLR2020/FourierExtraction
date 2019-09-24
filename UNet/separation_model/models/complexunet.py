import torch
import numpy                  as np
import torch.nn               as nn
import torch.nn.functional    as F
import sys
from   numpy.random       import RandomState
sys.path.append('../..')
from   complexmodels      import get_real, get_imag, get_modulus, ComplexConv
from   complexmodels      import complexops, realops, complex_product
from   complexmodels      import independent_filters_init_real as independent_filters_init
from   complexnorm        import ComplexBN, ComplexLN
from   resnet             import BasicBlock, Bottleneck, BasicDenseBlock, RealBasicBlock, RealBottleneck, RealBasicDenseBlock
from   torch.nn           import init



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_residual_unet_block(block, planes, nb_output_fmaps, blocks, stride=(1, 1),
                             space='complex', init_criterion='he', weight_init='complex',
                             seed=2018, stochastic_drop=0.5):
    downsample = None
    if stride not in {1, (1, 1)} or nb_output_fmaps != planes:
        if space == 'complex':
            downsample = ComplexConv(
                in_channels=planes, out_channels=nb_output_fmaps // 2,
                kernel_size=(1, 1), stride=stride,
                dilation=(1, 1), padding=(0, 0),
                bias=False, init_criterion=init_criterion,
                weight_init=weight_init, seed=seed
            ) # convolution of type valid
        elif space == 'real':
            downsample = nn.Conv2d(
                in_channels=planes, out_channels=nb_output_fmaps,
                kernel_size=(1, 1), stride=stride,
                dilation=(1, 1), padding=(0, 0),
                bias=False
            ) # convolution of type valid
    layers = []
    if downsample is None:
        if space == 'complex':
            layers.append(block(planes, nb_output_fmaps,      stride, downsample,
                            init_criterion=init_criterion, weight_init=weight_init,
                            stochastic_drop=stochastic_drop))
        elif space == 'real':
            layers.append(block(planes, nb_output_fmaps,      stride, downsample,       stochastic_drop=stochastic_drop))
    else:
        if space == 'complex':
            layers.append(block(planes, nb_output_fmaps // 2, stride, downsample,
                                init_criterion=init_criterion, weight_init=weight_init, stochastic_drop=None))
        elif space == 'real':
            layers.append(block(planes, nb_output_fmaps,      stride, downsample,       stochastic_drop=stochastic_drop))
    for i in range(1, blocks):
        if downsample is None or space == 'real':
            layers.append(block(nb_output_fmaps, nb_output_fmaps, stochastic_drop=stochastic_drop,
                                init_criterion=init_criterion, weight_init=weight_init))
        elif space == 'complex':
            layers.append(block(nb_output_fmaps // 2, nb_output_fmaps // 2,
                                init_criterion=init_criterion, weight_init=weight_init,
                                stochastic_drop=None))

    return nn.Sequential(*layers)


def make_dense_unet_block(block, inplanes, nb_output_fmaps, growth_rate, blocks, stride=(1, 1), seed=2018):
    downsample = None
    if stride not in {1, (1, 1)}:
        downsample = ComplexConv(
            in_channels=inplanes,
            out_channels=nb_output_fmaps // 2,
            kernel_size=(1, 1), stride=stride,  # stride here is (2, 2)
            dilation=(1, 1), padding=(0, 0),
            bias=False, init_criterion='he',
            weight_init='complex', seed=seed
        ) # convolution of type valid

    layers = []
    layers.append(block(inplanes=inplanes, growth_rate=growth_rate, stride=stride, downsample=downsample))
    first_block_inplanes = (inplanes + growth_rate) if downsample is None else growth_rate
    for i in range(1, blocks):
        layers.append(block(first_block_inplanes + ((i - 1) * growth_rate), growth_rate))
    layers.append(TransitionDense(first_block_inplanes + ((blocks - 1) * growth_rate), nb_output_fmaps))
    return nn.Sequential(*layers)


def complex_cat(x, y):
    d        = {'input_type':'convolution', 'channels_axis':1}
    out_real = torch.cat([get_real(x, **d), get_real(y, **d)], dim=1)
    out_imag = torch.cat([get_imag(x, **d), get_imag(y, **d)], dim=1)
    return torch.cat([out_real, out_imag], dim=1)


def get_complex_tuple(x):
    d        = {'input_type':'convolution', 'channels_axis':1}
    x_real   = get_real(x, **d)
    x_imag   = get_imag(x, **d)
    nb_fmaps = x_real.size(1) // 2
    s1       = torch.cat([x_real[:, :nb_fmaps ], x_imag[:, :nb_fmaps ]], dim=1)
    s2       = torch.cat([x_real[:,  nb_fmaps:], x_imag[:,  nb_fmaps:]], dim=1)
    return s1, s2


class CopiesGenerator(nn.Module):
    # this is the class for the keys generator
    # Inspired from FiLM (as FiLM could be seen as a special case of a mask generator)
    # Copies Generator could be performed at each block of the UNet where
    # we use generate copies from the parallel layer and concatenate them
    # to the previously inferred output. But let's try it first on the
    # last UNet block.
    def __init__(self, in_channels, nb_copies, nb_speakers=2, kernel_size=(3, 3),
                 stride=(1, 1), dilation=(1, 1), padding=(1, 1),
                 bias=True, init_criterion='he', weight_init='unitary',
                 seed=2018, dropcopy=None):
        super(CopiesGenerator, self).__init__()
        self.in_channels    = in_channels
        self.nb_speakers    = nb_speakers
        self.nb_copies      = nb_copies
        self.out_channels   = self.nb_copies    * self.nb_speakers  # out_channels is the nb of copies x nb_speakers.
        self.dropcopy       = dropcopy
        self.w_dgamma_beta  = ComplexConv(
            in_channels     = self.in_channels,
            out_channels    = self.out_channels * 2,
            kernel_size     = kernel_size,
            stride          = stride,
            dilation        = dilation,
            padding         = padding,
            bias            = bias,
            init_criterion  = init_criterion,
            weight_init     = weight_init,
            seed            = seed
        )
        self.seed           = seed
        self.rng            = np.random.RandomState(self.seed)
    def forward(self, inp):
        assert len(inp) == 2
        x_cond           = inp[0]
        x_to_film        = inp[1]
        dgamma_beta      = self.w_dgamma_beta(x_cond)
        dgamma, beta     = get_complex_tuple(dgamma_beta)
        d                = {'input_type':'convolution', 'channels_axis':1}
        repeated_input   = torch.cat([get_real(x_to_film, **d).repeat(1, self.out_channels, 1, 1),
                                      get_imag(x_to_film, **d).repeat(1, self.out_channels, 1, 1)], dim=1)
        if self.dropcopy is not None and self.dropcopy != 0:
            if self.training:
                dropcopies_mask     = self.rng.binomial(size=(1, self.out_channels, 1, 1), n=1, p=1-self.dropcopy)
                dropcopies_mask     = torch.from_numpy(dropcopies_mask).type_as(repeated_input).repeat(1, 2, 1, 1) # the same dropout mask is used for real and imaginary
                return complex_product(torch.ones_like(dgamma) + dgamma, repeated_input * dropcopies_mask,     input_type='convolution') + beta
            else:
                return complex_product(torch.ones_like(dgamma) + dgamma, repeated_input * (1 - self.dropcopy), input_type='convolution') + beta
        else:
            return     complex_product(torch.ones_like(dgamma) + dgamma, repeated_input,                       input_type='convolution') + beta


class LearnMasks(nn.Module):
    # VERY IMPORTANT: This type of class is used to get the output of ONLY 1 SPEAKER!
    def __init__(self, in_channels, nb_copies, nb_speakers=2, kernel_size=(3, 3),
                 stride=(1, 1), dilation=(1, 1), padding=(1, 1),
                 bias=True, init_criterion='he', weight_init='complex',
                 seed=2018, blocktype='basic'):
        super(LearnMasks, self).__init__()
        self.in_channels    = in_channels
        self.nb_copies      = nb_copies
        self.nb_speakers    = nb_speakers
        self.out_channels   = (self.nb_copies + 1) * self.nb_speakers  # out_channels is equal here to (nb_copies+1) x nb_speakers as we cat the input to the copies 
        self.seed           = seed
        self.weight_init    = weight_init
        self.init_criterion = init_criterion
        self.blocktype      = blocktype
        self.learnmasks     = ComplexConv(
            in_channels     = self.in_channels,
            out_channels    = self.out_channels,
            kernel_size     = kernel_size,
            stride          = stride,
            dilation        = dilation,
            padding         = padding,
            bias            = bias,
            init_criterion  = init_criterion,
            weight_init     = weight_init,
            seed            = seed
        )
        block = {'basic':     BasicBlock,
                 'bottleneck':Bottleneck,
                 'dense':     BasicDenseBlock}[self.blocktype]
        self.resblock       = make_residual_unet_block(
            block, self.out_channels, self.out_channels,
            blocks          = 2,
            stride          = (1, 1),
            space           = 'complex',
            seed            = self.seed,
            weight_init     = self.weight_init,
            init_criterion  = self.init_criterion,
            stochastic_drop = None
        )

    def forward(self, x):
        return self.resblock(self.learnmasks(x))


class GetOutputFromMasks(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=(3, 3),
                 stride=(1, 1), dilation=(1, 1), padding=(1, 1),
                 bias=True, init_criterion='he', weight_init='complex',
                 average=True, seed=2018):
        super(GetOutputFromMasks, self).__init__()
        self.in_channels      = in_channels   # in_channels would be (self.nb_copies + 1) x self.nb_speakers.
        self.out_channels     = out_channels  # should be equal to nb_speakers
        self.average          = average
        if out_channels is not None:
            self.learnspeech  = ComplexConv(
                in_channels   = self.in_channels,
                out_channels  = self.out_channels,
                kernel_size   = kernel_size,
                stride        = stride,
                dilation      = dilation,
                padding       = padding,
                bias          = bias,
                init_criterion= init_criterion,
                weight_init   = weight_init,
                seed          = seed
            )

    def forward(self, inp):
        copies_and_input, masks = get_complex_tuple(inp)
        candidates              = complex_product(copies_and_input, masks, input_type='convolution')
        if self.out_channels is not None:
            if self.average == True:
                raise ValueError(
                    """An invalid argument error. out_channels can not be different than None while average is True.
                    Found average == """ + str(self.average) + """ and out_channels == """ + str(self.out_channels)
            )
            else:
                return self.learnspeech(candidates)
        elif self.average:
            d           = {'input_type':'convolution', 'channels_axis':1}
            # We can potentially perform attention here
            s1, s2      = get_complex_tuple(candidates)
            s1_real_avg = get_real(s1, **d).mean( dim=1).unsqueeze(1)
            s1_imag_avg = get_imag(s1, **d).mean( dim=1).unsqueeze(1)
            s1_avg      = torch.cat([s1_real_avg, s1_imag_avg], dim=1)
            s2_real_avg = get_real(s2, **d).mean( dim=1).unsqueeze(1)
            s2_imag_avg = get_imag(s2, **d).mean( dim=1).unsqueeze(1)
            s2_avg      = torch.cat([s2_real_avg, s2_imag_avg], dim=1)
            return complex_cat(s1_avg, s2_avg)
        else:
            raise ValueError(
                """An invalid argument error. out_channels can not be None while at the same time average is False.
                Found average == """ + str(self.average) + """ and out_channels == """ + str(self.out_channels)
            )


class DenseUNetBlock(nn.Module):
    def __init__(self, block, inplanes, nb_output_fmaps, growth_rate, blocks,
                 stride=(1, 1), efficient=True, init_criterion='he',
                 weight_init='complex', seed=2018):
        super(DenseUNetBlock, self).__init__()
        self.block           = block
        self.inplanes        = inplanes
        self.nb_output_fmaps = nb_output_fmaps
        self.growth_rate     = growth_rate
        self.blocks          = blocks
        self.stride          = stride
        self.downsample      = None
        self.efficient       = efficient
        self.init_criterion  = init_criterion
        self.weight_init     = weight_init
        self.seed            = seed
        if self.stride not in {1, (1, 1)}:
            self.downsample  = ComplexConv(
                in_channels=self.inplanes,
                out_channels=self.growth_rate // 2,
                kernel_size=(1, 1), stride=self.stride,  # stride here is (2, 2)
                dilation=(1, 1), padding=(0, 0),
                bias=False, init_criterion=self.init_criterion,
                weight_init=self.weight_init, seed=self.seed
            ) # convolution of type valid"""
        layer = self.block(inplanes=self.inplanes, growth_rate=growth_rate, stride=stride,
                           downsample=self.downsample, efficient=self.efficient,
                           weight_init=self.weight_init, init_criterion=self.init_criterion)
        self.add_module('denselayer%d' % (0 + 1), layer)
        first_block_inplanes = (self.inplanes + self.growth_rate) if self.downsample is None else self.growth_rate
        for i in range(1, self.blocks):
            layer = self.block(first_block_inplanes + ((i - 1) * self.growth_rate), self.growth_rate,
                               efficient=self.efficient, weight_init=self.weight_init,
                               init_criterion=self.init_criterion)
            self.add_module('denselayer%d' % (i + 1), layer)

        self.transition = TransitionDense(first_block_inplanes + ((self.blocks - 1) * self.growth_rate),
                                          self.nb_output_fmaps)

    def forward(self, x):
        list_real   = []
        list_imag   = []
        u_real_imag = (list_real, list_imag)
        d           = {'input_type':'convolution', 'channels_axis':1}
        list_real.append(get_real(x, **d))
        list_imag.append(get_imag(x, **d))
        for name, layer in self.named_children():
            if not isinstance(layer, TransitionDense):
                new_features = layer(*u_real_imag)
                list_real.append(get_real(new_features, **d))
                list_imag.append(get_imag(new_features, **d))
        out = torch.cat([*list_real, *list_imag], dim=1)
        return self.transition(out)


class TransitionDense(nn.Module):
    def __init__(self, num_input_features, num_output_features,
                 weight_init='complex', init_criterion='he', seed=2018):
        super(TransitionDense, self).__init__()
        self.seed       = seed
        self.bn         = ComplexBN(
            num_complex_features=num_input_features, dim=1, eps=1e-4, momentum=0.1,
            scale=True, center=True, track_running_stats=True
        )
        self.relu       = nn.ReLU(inplace=True)
        self.conv       = ComplexConv(
            in_channels=num_input_features, out_channels=num_output_features, kernel_size=(1, 1), stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), bias=False, init_criterion=init_criterion,
            weight_init=weight, seed=self.seed
        )


    def forward(self, x):
        out        = self.bn(x)
        out        = self.relu(out)
        return       self.conv(out)


class RealDenseUNetBlock(nn.Module):
    def __init__(self, block, inplanes, nb_output_fmaps,
                 growth_rate, blocks, efficient=True,
                 bottleneck=True):
        super(RealDenseUNetBlock, self).__init__()
        for i in range(blocks):
            layer = RealBasicDenseBlock(
                inplanes   =inplanes + i * growth_rate,
                growth_rate=growth_rate,
                bottleneck =bottleneck,
                efficient  =efficient
            )
            self.add_module('denselayer%d' % (i + 1), layer)

        self.transition = RealTransitionDense(inplanes + (blocks * growth_rate),
                                              nb_output_fmaps)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if not isinstance(layer, RealTransitionDense):
                new_features = layer(*features)
                features.append(new_features)
        return self.transition(torch.cat(features, 1))


class RealTransitionDense(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(RealTransitionDense, self).__init__()
        self.bn         = nn.BatchNorm2d(
            num_features=num_input_features, eps=1e-4, momentum=0.1,
            affine=True, track_running_stats=True
        )
        self.relu       = nn.ReLU(inplace=True)
        self.conv       = nn.Conv2d(
            in_channels=num_input_features, out_channels=num_output_features,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            dilation=(1, 1), bias=False
        )


    def forward(self, x):
        out        = self.bn(x)
        out        = self.relu(out)
        return       self.conv(out)


# the following is taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1=3, stride1=1,
                 padding1=1, dilation1=1, kernel_size2=3, stride2=1, padding2=1, dilation2=1,
                 bn_axis=1, eps=1e-4, momentum=0.1, scale=True, center=True, track_running_stats=True,
                 conv_bias=False, init_criterion='he', weight_init='complex', operation='convolution2d',
                 seed=2018, nb_residual_blocks=None, blocktype='basic', growth_rate=2, space='complex',
                 affine=True, efficient=True, bottleneck=True, stochastic_drop=0.5, normalization='layernorm'):
        super(double_conv, self).__init__()
        self.in_channels            = in_channels
        self.out_channels1          = out_channels1
        self.out_channels2          = out_channels2
        self.nb_residual_blocks     = nb_residual_blocks
        self.stride1                = stride1
        self.stride2                = stride2
        self.blocktype              = blocktype
        self.growth_rate            = growth_rate
        self.kernel_size1           = kernel_size1
        self.stride1                = stride1
        self.padding1               = padding1
        self.dilation1              = dilation1
        self.kernel_size2           = kernel_size2
        self.stride2                = stride2
        self.padding2               = padding2
        self.dilation2              = dilation2
        self.bn_axis                = bn_axis
        self.eps                    = eps
        self.momentum               = momentum
        self.scale                  = scale
        self.center                 = center
        self.track_running_stats    = track_running_stats
        self.conv_bias              = conv_bias
        self.init_criterion         = init_criterion
        self.weight_init            = weight_init
        self.seed                   = seed
        self.operation              = operation
        self.space                  = space
        self.affine                 = affine
        self.efficient              = efficient
        self.bottleneck             = bottleneck
        self.stochastic_drop        = stochastic_drop
        self.normalization          = normalization
        conv1_args                  = {
            'in_channels'             : self.in_channels,
            'out_channels'            : self.out_channels1,
            'kernel_size'             : self.kernel_size1,
            'stride'                  : self.stride1,
            'dilation'                : self.dilation1,
            'padding'                 : self.padding1,
            'bias'                    : self.conv_bias,
        }
        conv2_args                  = {
            'in_channels'             : self.out_channels1,
            'out_channels'            : self.out_channels2,
            'kernel_size'             : self.kernel_size2,
            'stride'                  : self.stride2,
            'dilation'                : self.dilation2,
            'padding'                 : self.padding2,
            'bias'                    : self.conv_bias,
        }
        if self.space == 'complex':
            conv1_args.update({
                'init_criterion'      : self.init_criterion,
                'weight_init'         : self.weight_init,
                'operation'           : self.operation,
                'seed'                : self.seed
            })
            conv2_args.update({
                'init_criterion'      : self.init_criterion,
                'weight_init'         : self.weight_init,
                'operation'           : self.operation,
                'seed'                : self.seed
            })
            if self.normalization == 'bn':
                bn_args                 = {
                    'dim'                 : self.bn_axis,
                    'eps'                 : self.eps,
                    'momentum'            : self.momentum,
                    'scale'               : self.scale,
                    'center'              : self.center,
                    'track_running_stats' : self.track_running_stats
                }
                bn_fn   = ComplexBN
            elif self.normalization == 'layernorm':
                bn_args                 = {
                    'dim'                 : self.bn_axis,
                    'eps'                 : self.eps,
                    'scale'               : self.scale,
                    'center'              : self.center
                }
                bn_fn                   = ComplexLN
            conv_fn = ComplexConv
        elif self.space == 'real':
            bn_fn   = nn.BatchNorm2d
            conv_fn = nn.Conv2d
            bn_args                 = {
                'eps'                 : self.eps,
                'momentum'            : self.momentum,
                'affine'              : self.affine,
                'track_running_stats' : self.track_running_stats
            }

        self.bn1                    = bn_fn(self.in_channels, **bn_args)
        self.relu1                  = nn.ReLU(inplace=True)
        self.conv1                  = conv_fn(**conv1_args)
        if self.nb_residual_blocks is not None:
            if self.space == 'complex':
                block = {'basic':     BasicBlock,
                         'bottleneck':Bottleneck,
                         'dense':     BasicDenseBlock}[self.blocktype]
            elif self.space == 'real':
                block = {'basic':     RealBasicBlock,
                         'bottleneck':RealBottleneck,
                         'dense':     RealBasicDenseBlock}[self.blocktype]
            if   block in {BasicBlock, Bottleneck, RealBasicBlock, RealBottleneck}:
                self.resblocks1     = make_residual_unet_block(block, self.out_channels1, self.out_channels1,
                                                               self.nb_residual_blocks,
                                                               stride         =(1, 1),
                                                               space          =self.space,
                                                               seed           =self.seed,
                                                               weight_init    =self.weight_init,
                                                               init_criterion =self.init_criterion,
                                                               stochastic_drop=self.stochastic_drop)
                self.resblocks2     = make_residual_unet_block(block, self.out_channels2, self.out_channels2,
                                                               self.nb_residual_blocks,
                                                               stride         =(1, 1),
                                                               space          =self.space,
                                                               seed           =self.seed,
                                                               weight_init    =self.weight_init,
                                                               init_criterion =self.init_criterion,
                                                               stochastic_drop=self.stochastic_drop)
            elif block in {BasicDenseBlock}:
                self.resblocks1     = DenseUNetBlock(block, self.out_channels1, self.out_channels1, self.growth_rate,
                                                     self.nb_residual_blocks, stride=(1, 1), seed=self.seed,
                                                     efficient=self.efficient)
                self.resblocks2     = DenseUNetBlock(block, self.out_channels2, self.out_channels2, self.growth_rate,
                                                     self.nb_residual_blocks, stride=(1, 1), seed=self.seed,
                                                     efficient=self.efficient)
            elif block in {RealBasicDenseBlock}:
                self.resblocks1     = RealDenseUNetBlock(block=block, inplanes=self.out_channels1, nb_output_fmaps=self.out_channels1,
                                                         growth_rate=self.growth_rate, blocks=self.nb_residual_blocks,
                                                         efficient=self.efficient, bottleneck=self.bottleneck)
                self.resblocks2     = RealDenseUNetBlock(block=block, inplanes=self.out_channels2, nb_output_fmaps=self.out_channels2,
                                                         growth_rate=self.growth_rate, blocks=self.nb_residual_blocks,
                                                         efficient=self.efficient, bottleneck=self.bottleneck)

        self.bn2                    = bn_fn(self.out_channels1, **bn_args)
        self.relu2                  = nn.ReLU(inplace=True)
        self.conv2                  = conv_fn(**conv2_args)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        if self.nb_residual_blocks is not None:
            x = self.resblocks1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        if self.nb_residual_blocks is not None:
            x = self.resblocks2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, weight_init='complex', nb_residual_blocks=None,
                 blocktype='basic', growth_rate=2, space='complex', init_criterion='he', seed=2018, efficient=True,
                 bottleneck=True, stochastic_drop=0.5):
        super(inconv, self).__init__()
        self.in_channels   = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.space         = space
        self.seed          = seed
        self.efficient     = efficient
        self.bottleneck    = bottleneck
        self.stochastic_drop = stochastic_drop
        self.conv          = double_conv(
            self.in_channels, self.out_channels1, self.out_channels2,
            nb_residual_blocks=nb_residual_blocks, blocktype=blocktype,
            growth_rate=growth_rate, space=self.space, seed=self.seed,
            init_criterion=init_criterion, weight_init=weight_init,
            efficient=self.efficient, bottleneck=self.bottleneck,
            stochastic_drop=self.stochastic_drop
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class down(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, weight_init='complex', nb_residual_blocks=None,
                 blocktype='basic', growth_rate=2, space='complex', init_criterion='he', seed=2018, efficient=True,
                 bottleneck=True, stochastic_drop=0.5):
        super(down, self).__init__()
        self.in_channels     = in_channels
        self.out_channels1   = out_channels1
        self.out_channels2   = out_channels2
        self.space           = space
        self.seed            = seed
        self.efficient       = efficient
        self.bottleneck      = bottleneck
        self.stochastic_drop = stochastic_drop
        self.mpconv          = double_conv(
            self.in_channels, self.out_channels1, self.out_channels2, kernel_size2=1,
            stride2=(2, 2), padding2=0, nb_residual_blocks=nb_residual_blocks,
            blocktype=blocktype, growth_rate=growth_rate, space=self.space,
            seed=self.seed, init_criterion=init_criterion, weight_init=weight_init,
            efficient=self.efficient, bottleneck=self.efficient,
            stochastic_drop=self.stochastic_drop
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, mode='bilinear', bias=True,
                 init_criterion='he', weight_init='complex', seed=None,
                 operation='convolution2d', transpose=False, nb_residual_blocks=None,
                 blocktype='basic', growth_rate=2, space='complex',
                 efficient=True, bottleneck=True, stochastic_drop=0.5):
        super(up, self).__init__()
        self.in_channels     = in_channels
        self.out_channels1   = out_channels1
        self.out_channels2   = out_channels2
        self.mode            = mode
        self.bias            = bias
        self.init_criterion  = init_criterion
        self.weight_init     = weight_init
        self.seed            = seed
        self.operation       = operation
        self.transpose       = transpose
        self.space           = space
        self.efficient       = efficient
        self.bottleneck      = bottleneck
        self.stochastic_drop = stochastic_drop

        if   self.mode  == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode=self.mode, align_corners=True)
        elif self.mode  == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=self.mode)
        elif self.mode  == 'transpose':
            self.transpose = True
            if self.space == 'complex':
                self.up = ComplexConv(
                    in_channels    = self.in_channels // 2,
                    out_channels   = self.in_channels // 2,
                    kernel_size    =                     2,
                    stride         =                     2,
                    padding        =                     0,
                    dilation       =                     1,
                    bias           =             self.bias,
                    init_criterion =   self.init_criterion,
                    weight_init    =      self.weight_init,
                    seed           =             self.seed,
                    operation      =        self.operation,
                    transpose      =        self.transpose
                )
            elif self.space == 'real':
                self.up = nn.ConvTranspose2d(
                    in_channels    = self.in_channels // 2,
                    out_channels   = self.in_channels // 2,
                    kernel_size    =                     2,
                    stride         =                     2,
                    padding        =                     0,
                    dilation       =                     1,
                    bias           =             self.bias,
                )
        else:
            raise ValueError(
                """An invalid upsampling mode. mode must be among the following set {'bilinear', 'nearest', 'transpose'}.
                   Found mode == """ + str(self.mode)
            )

        self.conv = double_conv(
            self.in_channels, self.out_channels1, self.out_channels2,
            nb_residual_blocks=nb_residual_blocks, blocktype=blocktype,
            growth_rate=growth_rate, space=self.space, seed=self.seed,
            init_criterion=self.init_criterion, weight_init=self.weight_init,
            efficient=self.efficient, bottleneck=self.bottleneck,
            stochastic_drop=self.stochastic_drop
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(
            input = x2,
            pad   = (diffX // 2,
                     int(diffX / 2),
                     diffY // 2,
                     int(diffY / 2))
        )
        if self.space == 'complex':
            d      = {'input_type':'convolution', 'channels_axis':1}
            x_real = torch.cat([get_real(x2, **d), get_real(x1, **d)], dim=1)
            x_imag = torch.cat([get_imag(x2, **d), get_imag(x1, **d)], dim=1)
            x      = torch.cat([x_real, x_imag],                       dim=1)
        elif self.space == 'real':
            x      = torch.cat([x2, x1], dim=1)

        x          = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels, nb_residual_blocks=1,
                 blocktype='basic', growth_rate=2, seed=2018, space='complex',
                 weight_init='complex', init_criterion='he', efficient=True,
                 bottleneck=True, stochastic_drop=0.5):
        super(outconv, self).__init__()
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.nb_residual_blocks = nb_residual_blocks
        self.blocktype          = blocktype
        self.growth_rate        = growth_rate
        self.seed               = seed
        self.space              = space
        self.weight_init        = weight_init
        self.init_criterion     = init_criterion
        self.efficient          = efficient
        self.bottleneck         = bottleneck
        self.stochastic_drop    = stochastic_drop

        self.resblocks = double_conv(
            in_channels         = self.in_channels,
            out_channels1       = self.out_channels,
            out_channels2       = self.out_channels,
            nb_residual_blocks  = self.nb_residual_blocks,
            blocktype           = self.blocktype,
            growth_rate         = self.growth_rate,
            seed                = self.seed,
            space               = self.space,
            weight_init         = self.weight_init,
            init_criterion      = self.init_criterion,
            efficient           = self.efficient,
            bottleneck          = self.bottleneck,
            stochastic_drop     = self.stochastic_drop
        )

    def forward(self, x):
        return self.resblocks(x)


# the following is taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, in_channels=1, nb_speakers=2, mask_act='linear', eps=1e-4, mode='bilinear',
                 nb_residual_blocks=None, start_fmaps=64, blocktype='basic', growth_rate=2,
                 space='complex', weight_init='complex', init_criterion='he', seed=2018,
                 efficient=True, bottleneck=True, nb_copies=10, avg_copies=True, stochastic_drop=0.5,
                 stochdrop_schedule=True, dropcopy=None):
        super(UNet, self).__init__()
        self.in_channels = in_channels if space == 'complex' else in_channels * 2
        self.nb_speakers = nb_speakers
        self.mask_act    = mask_act
        self.eps         = eps
        self.mode        = mode
        self.nb_residual_blocks = nb_residual_blocks
        self.blocktype   = blocktype
        self.growth_rate = growth_rate
        self.space       = space
        self.start_fmaps = start_fmaps
        self.seed        = seed
        self.weight_init = weight_init
        self.init_criterion     = init_criterion
        self.efficient   = efficient
        self.bottleneck  = bottleneck
        self.nb_copies   = nb_copies
        self.avg_copies  = avg_copies
        self.stochastic_drop    = stochastic_drop
        self.stochdrop_schedule = stochdrop_schedule
        self.dropcopy    = dropcopy

        if self.stochastic_drop is None and self.stochdrop_schedule:
            raise ValueError(
                    """An invalid argument error. stochdrop_schedule can not be different True while stochastic_drop is None.
                    Found stochastic_drop == """ + str(self.stochastic_drop) + """ and stochdrop_schedule == """ + str(self.stochdrop_schedule)
            )
        if self.stochdrop_schedule:
            self.prob_list = []
            depth          = 5
            for i in range(depth):
                self.prob_list.append(1 - (i * 1.0 / depth) * (1 - self.stochastic_drop))
            print(self.prob_list)
        elif self.stochastic_drop is not None:
            self.prob_list = []
            depth          = 5
            for i in range(depth):
                self.prob_list.append(self.stochastic_drop)
            print(self.prob_list)
        if self.stochastic_drop is None:
            self.prob_list = []
            depth          = 5
            for i in range(depth):
                self.prob_list.append(None)
            print(self.prob_list)
        self.inc   = inconv( in_channels=self.in_channels,      out_channels1=self.start_fmaps // 2, out_channels2=self.start_fmaps // 2,  weight_init=self.weight_init,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[0])
        self.down1 = down(   in_channels=self.start_fmaps // 2, out_channels1=self.start_fmaps,      out_channels2=self.start_fmaps,       weight_init=self.weight_init,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[1])
        self.down2 = down(   in_channels=self.start_fmaps,      out_channels1=self.start_fmaps *  2, out_channels2=self.start_fmaps *  2,  weight_init=self.weight_init,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[2])
        self.down3 = down(   in_channels=self.start_fmaps *  2, out_channels1=self.start_fmaps *  4, out_channels2=self.start_fmaps *  4,  weight_init=self.weight_init,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[3])
        self.down4 = down(   in_channels=self.start_fmaps *  4, out_channels1=self.start_fmaps *  4, out_channels2=self.start_fmaps *  4,  weight_init=self.weight_init,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[4])
        self.up1   = up(     in_channels=self.start_fmaps *  8, out_channels1=self.start_fmaps *  2, out_channels2=self.start_fmaps *  2,  weight_init=self.weight_init, mode=self.mode,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[4])
        self.up2   = up(     in_channels=self.start_fmaps *  4, out_channels1=self.start_fmaps,      out_channels2=self.start_fmaps,       weight_init=self.weight_init, mode=self.mode,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[3])
        self.up3   = up(     in_channels=self.start_fmaps *  2, out_channels1=self.start_fmaps // 2, out_channels2=self.start_fmaps // 2,  weight_init=self.weight_init, mode=self.mode,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[2])
        self.up4   = up(     in_channels=self.start_fmaps,      out_channels1=self.start_fmaps // 2, out_channels2=self.start_fmaps // 2,  weight_init=self.weight_init, mode=self.mode,
                             nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                             init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[1])
        if self.nb_copies is None or self.nb_copies == 0:
            self.outc  = outconv(
                in_channels=self.start_fmaps // 2, out_channels=self.nb_speakers * self.in_channels, weight_init=self.weight_init,
                nb_residual_blocks=self.nb_residual_blocks, blocktype=self.blocktype, growth_rate=self.growth_rate, space=self.space, seed=self.seed,
                init_criterion=self.init_criterion, efficient=self.efficient, bottleneck=self.bottleneck, stochastic_drop=self.prob_list[0]
            )
        else:
            self.generate_copies = CopiesGenerator(
                in_channels      = self.start_fmaps  // 2,
                nb_copies        = self.nb_copies,
                nb_speakers      = self.nb_speakers,
                init_criterion   = self.init_criterion,
                weight_init      = 'unitary',
                seed             = self.seed,
                dropcopy         = self.dropcopy
            )
            self.learnmasks      = LearnMasks(
                in_channels      = (self.start_fmaps // 2)  + (self.nb_copies  + 1) * self.nb_speakers,
                nb_copies        = self.nb_copies,  # so out_channels would be equal to (self.nb_copies + 1) * self.nb_speakers
                init_criterion   = self.init_criterion,
                weight_init      = 'unitary',
                seed             = self.seed
            )
            oc = None if self.avg_copies else self.nb_speakers  * self.in_channels
            self.get_out         = GetOutputFromMasks(
                in_channels      = (self.nb_copies   +  1)  * self.nb_speakers,
                out_channels     = oc,
                init_criterion   = self.init_criterion,
                weight_init      = self.weight_init,
                average          = self.avg_copies,
                seed             = self.seed
            )

        if space == 'real':
            self.init_realweights()

    def init_realweights(self):
        rng = RandomState(self.seed)
        real_init_dict = {'orthogonal': independent_filters_init,
                          'real'      : realops.real_init}
        init_f = real_init_dict[self.weight_init]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kernel_size, _ = complexops.get_kernel_and_weight_shape(
                    'convolution2d', m.in_channels, m.out_channels, m.kernel_size
                )
                realops.affect_conv_init(m.weight, kernel_size, init_f, rng, self.init_criterion)
                if m.bias:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        batch, _, h, w = x.shape
        x              = x.view(batch, 2, h // 2, w)
        max_h, max_w   = int(np.ceil(h / 32) * 16), int(np.ceil(w / 16) * 16)
        padded_x       = torch.zeros((batch, 2, max_h, max_w), device=device)
        padded_x[:, :, :(h // 2), :w] = x
        x1             = self.inc(padded_x)
        x2             = self.down1(x1)
        x3             = self.down2(x2)
        x4             = self.down3(x3)
        x5             = self.down4(x4)
        dec            = self.up1(x5,  x4)
        dec            = self.up2(dec, x3)
        dec            = self.up3(dec, x2)
        dec            = self.up4(dec, x1)
        if self.nb_copies is None or self.nb_copies == 0:
            dec        = self.outc(dec)
        elif self.nb_copies is not None and self.nb_copies != 0:
            copies               = self.generate_copies([dec, padded_x])
            s1_copies, s2_copies = get_complex_tuple(copies)
            s1_copies_and_input  = complex_cat(padded_x, s1_copies)
            s2_copies_and_input  = complex_cat(padded_x, s2_copies)
            copies_and_input     = complex_cat(s1_copies_and_input, s2_copies_and_input)
            inp2lm               = complex_cat(copies_and_input, dec)
            masks                = self.learnmasks(inp2lm)
            inp2getout           = complex_cat(copies_and_input, masks)
            dec                  = self.get_out(inp2getout)

        if self.mask_act == 'linear':
            dec1       = torch.cat([dec.narrow(dim=1, start=0, length=1).narrow(dim=2, start=0, length=h//2).narrow(dim=3, start=0, length=w),
                                    dec.narrow(dim=1, start=2, length=1).narrow(dim=2, start=0, length=h//2).narrow(dim=3, start=0, length=w)],
                                   dim=1)
            dec2       = torch.cat([dec.narrow(dim=1, start=1, length=1).narrow(dim=2, start=0, length=h//2).narrow(dim=3, start=0, length=w),
                                    dec.narrow(dim=1, start=3, length=1).narrow(dim=2, start=0, length=h//2).narrow(dim=3, start=0, length=w)],
                                   dim=1)
            return(dec1.view(dec1.size(0), h, w), dec2.view(dec2.size(0), h, w))
