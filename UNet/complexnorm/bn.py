import torch
import numpy                         as     np
from   torch.nn                      import Module
from   torch.nn.parameter            import Parameter
import sys
sys.path.append('..')
from   complexmodels.complexops      import get_real, get_imag


# The following code is heavily inspired from
# https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py


def complex_standardization(input_centred, Vrr, Vii, Vri,
                            layernorm=False, dim=-1):
    ndim                       = input_centred.dim()
    input_dim                  = input_centred.size(dim) // 2
    variances_broadcast        = [1] * ndim
    variances_broadcast[dim]   = input_dim
    if layernorm:
        variances_broadcast[0] = input_centred.size(0)

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >=0 because Positive-definite matrix
    tau   = Vrr + Vii
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s     = delta.sqrt()
    t     = (tau + 2 * s).sqrt()

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / ( s   * t)
    Wrr        =       ( Vii + s) * inverse_st
    Wii        =       ( Vrr + s) * inverse_st
    Wri        =       - Vri      * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr    = Wrr.view(variances_broadcast)
    broadcast_Wri    = Wri.view(variances_broadcast)
    broadcast_Wii    = Wii.view(variances_broadcast)

    cat_W_4_real     = torch.cat([broadcast_Wrr, broadcast_Wii], dim=dim)
    cat_W_4_imag     = torch.cat([broadcast_Wri, broadcast_Wri], dim=dim)
    if   dim ==  0:
        centred_real = input_centred[:input_dim ]
        centred_imag = input_centred[ input_dim:]
    elif dim ==  1  or (dim == -1 and ndim == 2):
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif dim == -1 and ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif dim == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif dim == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of dim (axis) and number of dimensions. the dim should be either 0, 1 or -1. '
            'dim: ' + str(dim) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = torch.cat([centred_imag, centred_real], dim=dim)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output


def complexbn(input_centred, Vrr, Vii, Vri, beta,
              gamma_rr, gamma_ri, gamma_ii, scale=True,
              center=True, layernorm=False, dim=-1):
    ndim                            = input_centred.dim()
    input_dim                       = input_centred.size(dim) // 2
    if scale:
        gamma_broadcast_shape       = [1] * ndim
        gamma_broadcast_shape[dim]  = input_dim
    if center:
        broadcast_beta_shape        = [1] * ndim
        broadcast_beta_shape[dim]   = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            layernorm,
            dim=dim
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag
        broadcast_gamma_rr = gamma_rr.view(gamma_broadcast_shape)
        broadcast_gamma_ri = gamma_ri.view(gamma_broadcast_shape)
        broadcast_gamma_ii = gamma_ii.view(gamma_broadcast_shape)

        cat_gamma_4_real = torch.cat([broadcast_gamma_rr, broadcast_gamma_ii], dim=dim)
        cat_gamma_4_imag = torch.cat([broadcast_gamma_ri, broadcast_gamma_ri], dim=dim)
        if    dim ==  0:
            centred_real = standardized_output[:input_dim ]
            centred_imag = standardized_output[ input_dim:]
        elif  dim ==  1  or (dim  == -1 and ndim == 2):
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif  dim == -1 and  ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif dim  == -1 and  ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif dim  == -1 and  ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of dim and number of dimensions. dim should be either 0, 1 or -1. '
                'dim: ' + str(dim)  + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output  = torch.cat([centred_imag, centred_real], dim=dim)
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            a = cat_gamma_4_real    * standardized_output
            b = cat_gamma_4_imag    * rolled_standardized_output
            return a + b   + broadcast_beta
            #return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            return input_centred    + broadcast_beta
        else:
            return input_centred


# The following mean function using a list of reduced axes is taken from:
# https://discuss.pytorch.org/t/sum-mul-over-multiple-axes/1882/8
def multi_mean(input, axes, keepdim=False):
    '''
    Performs `torch.mean` over multiple dimensions of `input`
    '''
    axes = sorted(axes)
    m    = input
    for axis in reversed(axes):
        m = m.mean(axis, keepdim)
    return m


class ComplexBatchNormalization(Module):
    def __init__(self, num_complex_features, dim=-1, eps=1e-4, momentum=0.1,
                 scale=True, center=True, track_running_stats=True,
                 decorrelation=True, **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.num_complex_features        = num_complex_features
        self.dim                         = dim
        self.eps                         = eps
        self.momentum                    = momentum
        self.scale                       = scale
        self.center                      = center
        self.track_running_stats         = track_running_stats
        self.decorrelation               = decorrelation

        if self.scale:
            self.gamma_rr = Parameter(torch.Tensor(num_complex_features))
            self.gamma_ii = Parameter(torch.Tensor(num_complex_features))
            if self.decorrelation:
                self.gamma_ri = Parameter(torch.Tensor(num_complex_features))
            else:
                self.register_parameter('gamma_ri',        None)
        else:
            # To know more also about register_parameter check
            # https://pytorch.org/docs/stable/nn.html?highlight=buffer#torch.nn.Module.register_parameter
            self.register_parameter('gamma_rr',            None)
            self.register_parameter('gamma_ii',            None)
            self.register_parameter('gamma_ri',            None)

        if self.center:
            self.beta     = Parameter(torch.Tensor(num_complex_features * 2))
        else:
            self.register_parameter('beta',                None)

        if self.track_running_stats:
            # Check https://pytorch.org/docs/stable/nn.html?highlight=buffer#torch.nn.Module.register_buffer
            # to know more about nn.register_buffer and buffers in general.

            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

            if self.scale:
                self.register_buffer(   'moving_Vrr',   torch.ones( num_complex_features) * np.sqrt(1 / 2))
                self.register_buffer(   'moving_Vii',   torch.ones( num_complex_features) * np.sqrt(1 / 2))
                if self.decorrelation:
                    self.register_buffer(   'moving_Vri',   torch.zeros(num_complex_features))
                else:
                    self.register_parameter('moving_Vri',  None)
            else:
                self.register_parameter('moving_Vrr',      None)
                self.register_parameter('moving_Vii',      None)
                self.register_parameter('moving_Vri',      None)

            if self.center:
                self.register_buffer(   'moving_mean',  torch.zeros(num_complex_features  * 2))
            else:
                self.register_parameter('moving_mean',     None)

        else:
            self.register_parameter('moving_Vrr',          None)
            self.register_parameter('moving_Vii',          None)
            self.register_parameter('moving_Vri',          None)
            self.register_parameter('moving_mean',         None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            if self.center:
                self.moving_mean.zero_()
            if self.scale:
                self.moving_Vrr.fill_(1 / np.sqrt(2))
                self.moving_Vii.fill_(1 / np.sqrt(2))
                if self.decorrelation:
                    self.moving_Vri.zero_()
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.scale:
            self.gamma_rr.data.fill_(1 / np.sqrt(2))
            self.gamma_ii.data.fill_(1 / np.sqrt(2))
            if self.decorrelation:
                self.gamma_ri.data.zero_()
        if self.center:
            self.beta.data.zero_()

    def forward(self, input):
        if not (self.scale or self.center):
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.center:
                self.moving_mean         = self.moving_mean.detach()
            if self.scale:
                self.moving_Vrr          = self.moving_Vrr.detach()
                self.moving_Vii          = self.moving_Vii.detach()
                if self.decorrelation:
                    self.moving_Vri      = self.moving_Vri.detach()
            self.num_batches_tracked   = self.num_batches_tracked.detach()
            self.num_batches_tracked  += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum

        input_shape                   = input.size()
        ndim                          = input.dim()
        reduction_axes                = list(range(ndim))
        del reduction_axes[self.dim]
        input_dim                     = input_shape[self.dim] // 2

        mu                            = multi_mean(input, reduction_axes, True)
        input_centred                 = input  - mu
        if self.scale:
            centred_squared     =  input_centred ** 2
        if self.dim    == 1    or  ndim == 2:
            if self.scale:
                centred_squared_real  = centred_squared[:, :input_dim]
                centred_squared_imag  = centred_squared[:, input_dim:]
            if self.center:
                centred_real          = input_centred[:, :input_dim]
                centred_imag          = input_centred[:, input_dim:]
        elif self.dim  == -1  and  ndim == 3:
            if self.scale:
                centred_squared_real  = centred_squared[:, :, :input_dim]
                centred_squared_imag  = centred_squared[:, :, input_dim:]
            if self.center:
                centred_real          = input_centred[:, :, :input_dim]
                centred_imag          = input_centred[:, :, input_dim:]
        elif self.dim  == -1  and  ndim == 4:
            if self.scale:
                centred_squared_real  = centred_squared[:, :, :, :input_dim]
                centred_squared_imag  = centred_squared[:, :, :, input_dim:]
            if self.center:
                centred_real          = input_centred[:, :, :, :input_dim]
                centred_imag          = input_centred[:, :, :, input_dim:]
        elif self.dim  == -1  and  ndim == 5:
            if self.scale:
                centred_squared_real  = centred_squared[:, :, :, :, :input_dim]
                centred_squared_imag  = centred_squared[:, :, :, :, input_dim:]
            if self.center:
                centred_real          = input_centred[:, :, :, :, :input_dim]
                centred_imag          = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of dim and number of dimensions. dim should be either 1 or -1. '
                'dim: ' + str(self.dim) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = multi_mean(
                centred_squared_real,        axes=reduction_axes, keepdim=True
            ) + self.eps
            Vii = multi_mean(
                centred_squared_imag,        axes=reduction_axes, keepdim=True
            ) + self.eps
            # Vri contains the real and imaginary covariance for each feature map.
            if self.decorrelation:
                Vri = multi_mean(
                    centred_real * centred_imag, axes=reduction_axes, keepdim=True
                )
            else:
                Vri = None
        else:
            Vrr = None
            Vii = None
            Vri = None
        if self.training:
            if self.track_running_stats:
                     # Pick the normalized form corresponding to the training phase when we use running stats.
                if self.center:
                    self.moving_mean       = (1 - exponential_average_factor) * self.moving_mean + exponential_average_factor * mu.view(self.moving_mean.size())
                if self.scale:
                    self.moving_Vrr        = (1 - exponential_average_factor) * self.moving_Vrr  + exponential_average_factor * Vrr.view(self.moving_Vrr.size())
                    self.moving_Vii        = (1 - exponential_average_factor) * self.moving_Vii  + exponential_average_factor * Vii.view(self.moving_Vii.size())
                    if self.decorrelation:
                         self.moving_Vri   = (1 - exponential_average_factor) * self.moving_Vri  + exponential_average_factor * Vri.view(self.moving_Vri.size())
        if self.decorrelation:
            if self.training or (not self.track_running_stats):
                input_inferred = input_centred if self.center else input
                return complexbn(
                        input_inferred, Vrr, Vii, Vri,
                        self.beta, self.gamma_rr, self.gamma_ri,
                        self.gamma_ii, self.scale, self.center,
                        layernorm=False, dim=self.dim
                    )
            else:  # which means if not self.training and self.track_running_stats
                input_inferred = input - self.moving_mean.view(mu.size()) if self.center else input
                return complexbn(
                    input_inferred, self.moving_Vrr, self.moving_Vii, self.moving_Vri,
                    self.beta, self.gamma_rr, self.gamma_ri,
                    self.gamma_ii, self.scale, self.center,
                    layernorm=False, dim=self.dim
                )
        else:
            if self.scale:
                if self.training or (not self.track_running_stats):
                    input_inferred  = input_centred if self.center else input
                    complex_std     = torch.sqrt(Vrr   + Vii)
                else:  # which means if not self.training and self.track_running_stats
                    input_inferred  = input - self.moving_mean.view(mu.size()) if self.center else input
                    complex_std     = torch.sqrt(self.moving_Vrr   + self.moving_Vii)
                gamma_broadcast_shape              = [1]       * ndim
                gamma_broadcast_shape[self.dim]    = input_dim
                inferred_real   = get_real(input_inferred, input_type='convolution', channels_axis=self.dim)
                inferred_imag   = get_imag(input_inferred, input_type='convolution', channels_axis=self.dim)
                normalized_real = self.gamma_rr.view(gamma_broadcast_shape) * (inferred_real / (complex_std.view(gamma_broadcast_shape) + self.eps))
                normalized_imag = self.gamma_ii.view(gamma_broadcast_shape) * (inferred_imag / (complex_std.view(gamma_broadcast_shape) + self.eps))
                if self.center:
                    broadcast_beta_shape           = [1]       * ndim
                    broadcast_beta_shape[self.dim] = input_dim * 2
                    return torch.cat([normalized_real, normalized_imag], dim=self.dim) + self.beta.view(broadcast_beta_shape)
                else:
                    return torch.cat([normalized_real, normalized_imag], dim=self.dim)
            elif self.center:
                if self.training or (not self.track_running_stats):
                    input_inferred  = input_centred
                else:  # which means if not self.training and self.track_running_stats
                    input_inferred  = input - self.moving_mean.view(mu.size())
                broadcast_beta_shape               = [1]       * ndim
                broadcast_beta_shape[self.dim]     = input_dim * 2
                inferred_real   = get_real(input_inferred, input_type='convolution', channels_axis=self.dim)
                inferred_imag   = get_imag(input_inferred, input_type='convolution', channels_axis=self.dim)
                return torch.cat([inferred_real, inferred_imag], dim=self.dim) + self.beta.view(broadcast_beta_shape)

    def extra_repr(self):
        return '{num_complex_features}, dim={dim}, eps={eps}, momentum={momentum}, scale={scale}, ' \
               'center={center}, track_running_stats={track_running_stats}, ' \
               'decorrelation={decorrelation}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ComplexBatchNormalization, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ComplexLayerNormalization(Module):
    def __init__(self, num_complex_features, dim=1, eps=1e-4, scale=True,
                 center=True, decorrelation=True, **kwargs):
        super(ComplexLayerNormalization, self).__init__(**kwargs)
        self.num_complex_features        = num_complex_features
        self.dim                         = dim
        self.eps                         = eps
        self.scale                       = scale
        self.center                      = center
        self.decorrelation               = decorrelation
        if self.scale:
            self.gamma_rr = Parameter(torch.Tensor(num_complex_features))
            self.gamma_ii = Parameter(torch.Tensor(num_complex_features))
            if self.decorrelation:
                self.gamma_ri = Parameter(torch.Tensor(num_complex_features))
            else:
                self.register_parameter('gamma_ri',        None)
        else:
            # To know more also about register_parameter check
            # https://pytorch.org/docs/stable/nn.html?highlight=buffer#torch.nn.Module.register_parameter
            self.register_parameter('gamma_rr',            None)
            self.register_parameter('gamma_ii',            None)
            self.register_parameter('gamma_ri',            None)

        if self.center:
            self.beta     = Parameter(torch.Tensor(num_complex_features * 2))
        else:
            self.register_parameter('beta',                None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.scale:
            self.gamma_rr.data.fill_(1 / np.sqrt(2))
            self.gamma_ii.data.fill_(1 / np.sqrt(2))
            self.gamma_ri.data.zero_()
        if self.center:
            self.beta.data.zero_()

    def forward(self, input):
        input_shape                   = input.size()
        ndim                          = input.dim()
        reduction_axes                = list(range(ndim))
        del reduction_axes[self.dim]
        del reduction_axes[0]
        input_dim                     = input_shape[self.dim] // 2
        mu                            = multi_mean(input, reduction_axes, True)

        if self.center:
            input_centred = input - mu
        else:
            input_centred = input

        centred_squared   = input_centred ** 2

        if self.dim   == 1   or  ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real         = input_centred[:, :input_dim]
            centred_imag         = input_centred[:, input_dim:]
        elif self.dim == -1 and  ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real         = input_centred[:, :, :input_dim]
            centred_imag         = input_centred[:, :, input_dim:]
        elif self.dim == -1 and  ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real         = input_centred[:, :, :, :input_dim]
            centred_imag         = input_centred[:, :, :, input_dim:]
        elif self.dim == -1 and  ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real         = input_centred[:, :, :, :, :input_dim]
            centred_imag         = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Layernorm combination of dim and number of dimensions. dim should be either 1 or -1. '
                'dim: ' + str(self.dim) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = multi_mean(
                centred_squared_real,        axes=reduction_axes, keepdim=True
            ) + self.eps
            Vii = multi_mean(
                centred_squared_imag,        axes=reduction_axes, keepdim=True
            ) + self.eps
            # Vri contains the real and imaginary covariance for each feature map.
            if self.decorrelation:
                Vri = multi_mean(
                    centred_real * centred_imag, axes=reduction_axes, keepdim=True
                )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in layernorm are set to False.')
        if self.decorrelation:
            return complexbn(
                input_centred, Vrr, Vii, Vri,
                self.beta, self.gamma_rr, self.gamma_ri,
                self.gamma_ii, self.scale, self.center,
                dim=self.dim, layernorm=True
            )
        else:
            input_inferred      = input_centred if self.center else input
            if self.scale:
                complex_std     = torch.sqrt(Vrr + Vii)
                gamma_broadcast_shape              = [1]       * ndim
                gamma_broadcast_shape[self.dim]    = input_dim
                inferred_real   = get_real(input_inferred, input_type='convolution', channels_axis=self.dim)
                inferred_imag   = get_imag(input_inferred, input_type='convolution', channels_axis=self.dim)
                normalized_real = self.gamma_rr.view(gamma_broadcast_shape) * (inferred_real / (complex_std.view(gamma_broadcast_shape) + self.eps))
                normalized_imag = self.gamma_ii.view(gamma_broadcast_shape) * (inferred_imag / (complex_std.view(gamma_broadcast_shape) + self.eps))
                if self.center:
                    broadcast_beta_shape           = [1]       * ndim
                    broadcast_beta_shape[self.dim] = input_dim * 2
                    return torch.cat([normalized_real, normalized_imag], dim=self.dim) + self.beta.view(broadcast_beta_shape)
                else:
                    return torch.cat([normalized_real, normalized_imag], dim=self.dim)
            elif self.center:
                input_inferred  = input_centred
                broadcast_beta_shape               = [1]       * ndim
                broadcast_beta_shape[self.dim]     = input_dim * 2
                inferred_real   = get_real(input_inferred, input_type='convolution', channels_axis=self.dim)
                inferred_imag   = get_imag(input_inferred, input_type='convolution', channels_axis=self.dim)
                return torch.cat([inferred_real, inferred_imag], dim=self.dim) + self.beta.view(broadcast_beta_shape)

    def extra_repr(self):
        return '{num_complex_features}, dim={dim}, eps={eps}, scale={scale}, ' \
               'center={center}'.format(**self.__dict__)
