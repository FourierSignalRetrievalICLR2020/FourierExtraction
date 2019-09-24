import torch
import scipy.linalg
import numpy              as     np
import torch.nn           as     nn
from   numpy.random       import RandomState
from   torch.nn.parameter import Parameter
from   torch.nn           import Module
from   .                  import realops
from   .complexops        import *
import sys
sys.path.append('..')
from   complexnorm        import complexbn


class ComplexLinear(Module):
    r"""Applies a complex linear transformation to the incoming data.
    Args:
        in_features:  size of each complex input sample. The effective number
            of hidden units for each of the real and imaginary inputs.
            The total effective number of input hidden units is 2 x in_features.
        out_features: size of each complex output sample. The effective number
            of hidden units for each of the real and imaginary outputs.
            The total effective number of output hidden units is 2 x out_features.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
    Shape:
        - Input:  (N, 2 * in_features) 
        - Output: (N, 2 * out_features)
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (2 * out_features)
    Examples::
        >>> m = ComplexLinear(20, 30)
        >>> input = Variable(torch.randn(128, 40))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='unitary',
                 seed=None):
        super(ComplexLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.real_weight  = Parameter(torch.Tensor(in_features, out_features))
        self.imag_weight  = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(2 * out_features))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'complex': complex_init,
                 'unitary': unitary_init}[self.weight_init]
        affect_init(self.real_weight, self.imag_weight, winit,
                    self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):
        #if input.dim() == 3:
        #    T, N, C = input.size()
        #    input = input.view(T * N, C)
        #    output = complex_linear_function.apply(input, self.real_weight, self.imag_weight, self.bias)
        #    output = output.view(T, N, output.size(1))
        #elif input.dim() == 2:
        #    output = complex_linear_function.apply(input, self.real_weight, self.imag_weight, self.bias)
        #else:
        #    raise NotImplementedError
        #return output
        
        return complex_linear(input, self.real_weight, self.imag_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features='           + str(self.in_features) \
            + ', out_features='        + str(self.out_features) \
            + ', bias='                + str(self.bias is not None) \
            + ', init_criterion='      + str(self.init_criterion) \
            + ', weight_init='         + str(self.weight_init) \
            + ', seed='                + str(self.seed) \
            + ')'


class ComplexConv(Module):
    r"""Applies a complex Convolution to the incoming data.
    Args:
        in_channels:  Number of complex input channels. The effective number
            of input channels for each of the real and imaginary inputs.
            The total effective number of input channels is 2 x in_channels.
        out_channels: size of each complex output sample. The effective number
            of output_channels for each of the real and imaginary outputs.
            The total effective number of output channels is 2 x out_channels.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True the size of the bias is (2 x out_channels, )
    Shape:
        - Input:  (N, 2 * in_channels)  + in_channel.shape()
        - Output: (N, 2 * out_channels) + out_channel.shape()
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (2 * out_features)
    Examples::
        >>> cc = ComplexConv(in_channels=2, out_channels=1, kernel_size=3, stride=1, dilation=1, padding=1)
        >>> input = Variable(torch.rand(16,4,10))
        >>> output = cc(input)
        >>> print(output.size())
        Out: torch.Size([16, 2, 4])
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation, padding, bias=True, init_criterion='glorot',
                 weight_init='unitary', seed=None, operation='convolution2d',
                 transpose=False, normalize_weight=False, eps=1e-4, **kwargs):

        super(ComplexConv, self).__init__(**kwargs)
        self.in_channels      =    in_channels
        self.out_channels     =    out_channels
        self.stride           =    stride
        self.padding          =    padding
        self.dilation         =    dilation
        self.init_criterion   =    init_criterion
        self.weight_init      =    weight_init
        self.seed             =    seed if seed is not None else 1337
        self.rng              =    RandomState(self.seed)
        self.winit            =    {'complex': complex_init,
                                    'unitary': independent_complex_filters_init}[self.weight_init]
        self.operation        = operation
        self.transpose        = transpose
        self.normalize_weight = normalize_weight
        self.eps              = eps

        (self.kernel_size,
         self.w_shape)        = get_kernel_and_weight_shape(
            self.operation, self.in_channels, self.out_channels, kernel_size, self.transpose
        )
        self.real_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.imag_weight  = Parameter(torch.Tensor(*self.w_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(2 * out_channels))
        else:
            self.register_parameter('bias', None)
        if self.normalize_weight:
            """gamma_shape   = (self.in_channels * self.out_channels,)
            self.gamma_rr = Parameter(torch.Tensor(*gamma_shape))
            self.gamma_ii = Parameter(torch.Tensor(*gamma_shape))
            self.gamma_ri = Parameter(torch.Tensor(*gamma_shape))"""
            gamma_shape   = (self.in_channels * self.out_channels * 2,)
            self.gamma    = Parameter(torch.Tensor(*gamma_shape))
        else:
            """self.gamma_rr = self.register_parameter('gamma_rr', None)
            self.gamma_ii = self.register_parameter('gamma_ii', None)
            self.gamma_ri = self.register_parameter('gamma_ri', None)"""
            self.gamma = self.register_parameter('gamma', None)
        self.reset_parameters()

    def reset_parameters(self):
        fargs = [self.winit, self.rng, self.init_criterion, self.transpose]
        affect_conv_init(self.real_weight, self.imag_weight, self.kernel_size, *fargs)
        if self.bias is not None:
           self.bias.data.zero_()
        if self.normalize_weight:
            """self.gamma_rr.data.fill_(1 / np.sqrt(2))
            self.gamma_ii.data.fill_(1 / np.sqrt(2))
            self.gamma_ri.data.zero_()"""
            self.gamma.data.fill_(1 / np.sqrt(2))

    def forward(self, input):
        convargs={'stride'   : self.stride,
                  'padding'  : self.padding,
                  'dilation' : self.dilation,
                  'transpose': self.transpose}
        if self.normalize_weight:
            nb_kernels                      = self.out_channels   *   self.in_channels
            kernel_shape_4_norm             = (nb_kernels, np.prod(   self.kernel_size))
            reshaped_w_real                 = self.real_weight.view(kernel_shape_4_norm)
            reshaped_w_imag                 = self.imag_weight.view(kernel_shape_4_norm)
            complex_weight                  = torch.cat([reshaped_w_real, reshaped_w_imag], dim=0)
            complex_weight_norm             = torch.norm(complex_weight)
            prenormalized_weight            = complex_weight      /      (complex_weight_norm + self.eps)
            normalized_weight               = self.gamma.unsqueeze(1) * prenormalized_weight
            """complex_weight_norm             = torch.norm(complex_weight)
            complex_weight_prenormalized    = complex_weight    / ( complex_weight_norm          + self.eps)
            mu                              = complex_weight_prenormalized.mean(dim=1, keepdim=True)
            reshaped_w_complex_centred      = complex_weight_prenormalized - mu"""
            """mu                              = complex_weight.mean(          dim=1, keepdim=True)
            reshaped_w_complex_centred      = complex_weight      - mu
            reshaped_w_real_centred         = reshaped_w_complex_centred[:nb_kernels ]
            reshaped_w_imag_centred         = reshaped_w_complex_centred[ nb_kernels:]
            Vrr = (reshaped_w_real_centred ** 2).mean(                      dim=1, keepdim=True) + self.eps
            Vii = (reshaped_w_imag_centred ** 2).mean(                      dim=1, keepdim=True) + self.eps
            Vri = (reshaped_w_real_centred  * reshaped_w_imag_centred).mean(dim=1, keepdim=True) + self.eps
            normalized_weight = complexbn(
                reshaped_w_complex_centred,
                Vrr, Vii, Vri,
                beta      = None,
                gamma_rr  = self.gamma_rr,
                gamma_ri  = self.gamma_ri,
                gamma_ii  = self.gamma_ii,
                scale     = True,
                center    = False,
                layernorm = False,
                dim       = 0
            )"""
            normalized_real = normalized_weight[:nb_kernels ]
            normalized_imag = normalized_weight[ nb_kernels:]
            f_real = normalized_real.view(*self.w_shape)
            f_imag = normalized_imag.view(*self.w_shape)
            return complex_conv(input,    f_real,           f_imag,        self.bias, **convargs)
        else:
            return complex_conv(input, self.real_weight, self.imag_weight, self.bias, **convargs)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='        + str(self.in_channels) \
            + ', out_channels='     + str(self.out_channels) \
            + ', bias='             + str(self.bias is not None) \
            + ', kernel_size='      + str(self.kernel_size) \
            + ', stride='           + str(self.stride) \
            + ', padding='          + str(self.padding) \
            + ', dilation='         + str(self.dilation) \
            + ', init_criterion='   + str(self.init_criterion) \
            + ', weight_init='      + str(self.weight_init) \
            + ', seed='             + str(self.seed) \
            + ', operation='        + str(self.operation) \
            + ', transpose='        + str(self.transpose) \
            + ', normalize_weight=' + str(self.normalize_weight) \
            + ', eps='              + str(self.eps) + ')'


#
# Utility functions
#
def _istuple(x):   return isinstance(x, tuple)
def _mktuple1d(x): return x if _istuple(x) else (x,)
def _mktuple2d(x): return x if _istuple(x) else (x,x)

class STFT1d(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=8, window_fn=np.hanning, stride=1,
                 padding=0, dilation=1, inverse=False):
        super(STFT1d, self).__init__()
        self.in_channels  = in_channels
        self.kernel_size  = _mktuple1d(kernel_size)
        self.window_fn    = _mktuple1d(window_fn)
        self.stride       = _mktuple1d(stride)
        self.padding      = _mktuple1d(padding)
        self.dilation     = _mktuple1d(dilation)
        self.inverse      = bool(inverse);
        
        w  = self.kernel_size[0]
        wW = np.ones(w) if self.inverse else self.window_fn[0](w)
        Fw = scipy.linalg.dft(w, "sqrtn").astype("complex128")
        
        F  = np.einsum("i,fi->fi", wW, Fw)
        F  = F.astype("complex128")
        F  = F.reshape(-1,1,w)
        F  = F.conjugate() if self.inverse else F
        
        Wr = torch.empty(*F.real.shape).copy_(torch.from_numpy(F.real))
        Wi = torch.empty(*F.imag.shape).copy_(torch.from_numpy(F.imag))
        self.register_buffer("Wr", Wr)
        self.register_buffer("Wi", Wi)
    
    def forward(self, xr, xi=None):
        inpSize = xr.shape
        B  = inpSize[0]
        if self.inverse:
            assert(xi is not None)
            xr = xr.view(B*self.in_channels, -1, *inpSize[2:])
            xi = xi.view(B*self.in_channels, -1, *inpSize[2:])
            rr = torch.nn.functional.conv_transpose1d(xr,
                                                      self.Wr,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ri = torch.nn.functional.conv_transpose1d(xr,
                                                      self.Wi,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ir = torch.nn.functional.conv_transpose1d(xi,
                                                      self.Wr,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ii = torch.nn.functional.conv_transpose1d(xi,
                                                      self.Wi,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            rr = rr.view(B, -1, *rr.shape[2:])
            ri = ri.view(B, -1, *ri.shape[2:])
            ir = ir.view(B, -1, *ir.shape[2:])
            ii = ii.view(B, -1, *ii.shape[2:])
            return rr-ii, ri+ir
        else:
            xr = xr.view(B*self.in_channels, 1, *inpSize[2:])
            rr = torch.nn.functional.conv1d(xr,
                                            self.Wr,
                                            None,
                                            stride  =self.stride,
                                            padding =self.padding,
                                            dilation=self.dilation)
            ri = torch.nn.functional.conv1d(xr,
                                            self.Wi,
                                            None,
                                            stride  =self.stride,
                                            padding =self.padding,
                                            dilation=self.dilation)
            rr = rr.view(B, -1, *rr.shape[2:])
            ri = ri.view(B, -1, *ri.shape[2:])
            
            if xi is None:
                return rr, ri
            else:
                xi = xi.view(B*self.in_channels, 1, *inpSize[2:])
                ir = torch.nn.functional.conv1d(xi,
                                                self.Wr,
                                                None,
                                                stride  =self.stride,
                                                padding =self.padding,
                                                dilation=self.dilation)
                ii = torch.nn.functional.conv1d(xi,
                                                self.Wi,
                                                None,
                                                stride  =self.stride,
                                                padding =self.padding,
                                                dilation=self.dilation)
                ir = ir.view(B, -1, *ir.shape[2:])
                ii = ii.view(B, -1, *ii.shape[2:])
                return rr-ii, ri+ir


class STFT2d(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=8, window_fn=np.hanning, stride=1,
                 padding=0, dilation=1, inverse=False):
        super(STFT2d, self).__init__()
        self.in_channels  = in_channels
        self.kernel_size  = _mktuple2d(kernel_size)
        self.window_fn    = _mktuple2d(window_fn)
        self.stride       = _mktuple2d(stride)
        self.padding      = _mktuple2d(padding)
        self.dilation     = _mktuple2d(dilation)
        self.inverse      = bool(inverse);
        
        h  = self.kernel_size[0]
        w  = self.kernel_size[1]
        wH = np.ones(h) if self.inverse else self.window_fn[0](h)
        wW = np.ones(w) if self.inverse else self.window_fn[1](w)
        Fh = scipy.linalg.dft(h, "sqrtn").astype("complex128")
        Fw = scipy.linalg.dft(w, "sqrtn").astype("complex128")
        
        F  = np.einsum("i,j,fi,gj->fgij", wH, wW, Fh, Fw)
        F  = F.astype("complex128")
        F  = F.reshape(-1,1,h,w)
        F  = F.conjugate() if self.inverse else F
        
        Wr = torch.empty(*F.real.shape).copy_(torch.from_numpy(F.real))
        Wi = torch.empty(*F.imag.shape).copy_(torch.from_numpy(F.imag))
        self.register_buffer("Wr", Wr)
        self.register_buffer("Wi", Wi)
    
    def forward(self, xr, xi=None):
        inpSize = xr.shape
        B  = inpSize[0]
        if self.inverse:
            assert(xi is not None)
            xr = xr.view(B*self.in_channels, -1, *inpSize[2:])
            xi = xi.view(B*self.in_channels, -1, *inpSize[2:])
            rr = torch.nn.functional.conv_transpose2d(xr,
                                                      self.Wr,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ri = torch.nn.functional.conv_transpose2d(xr,
                                                      self.Wi,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ir = torch.nn.functional.conv_transpose2d(xi,
                                                      self.Wr,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            ii = torch.nn.functional.conv_transpose2d(xi,
                                                      self.Wi,
                                                      None,
                                                      stride  =self.stride,
                                                      padding =self.padding,
                                                      dilation=self.dilation)
            rr = rr.view(B, -1, *rr.shape[2:])
            ri = ri.view(B, -1, *ri.shape[2:])
            ir = ir.view(B, -1, *ir.shape[2:])
            ii = ii.view(B, -1, *ii.shape[2:])
            return rr-ii, ri+ir
        else:
            xr = xr.view(B*self.in_channels, 1, *inpSize[2:])
            rr = torch.nn.functional.conv2d(xr,
                                            self.Wr,
                                            None,
                                            stride  =self.stride,
                                            padding =self.padding,
                                            dilation=self.dilation)
            ri = torch.nn.functional.conv2d(xr,
                                            self.Wi,
                                            None,
                                            stride  =self.stride,
                                            padding =self.padding,
                                            dilation=self.dilation)
            rr = rr.view(B, -1, *rr.shape[2:])
            ri = ri.view(B, -1, *ri.shape[2:])
            
            if xi is None:
                return rr, ri
            else:
                xi = xi.view(B*self.in_channels, 1, *inpSize[2:])
                ir = torch.nn.functional.conv2d(xi,
                                                self.Wr,
                                                None,
                                                stride  =self.stride,
                                                padding =self.padding,
                                                dilation=self.dilation)
                ii = torch.nn.functional.conv2d(xi,
                                                self.Wi,
                                                None,
                                                stride  =self.stride,
                                                padding =self.padding,
                                                dilation=self.dilation)
                ir = ir.view(B, -1, *ir.shape[2:])
                ii = ii.view(B, -1, *ii.shape[2:])
                return rr-ii, ri+ir

