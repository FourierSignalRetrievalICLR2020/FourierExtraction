import numpy as np
import torch

def affect_init(weight, init_func, rng, init_criterion):
    if weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(real_weight.dim()))

    # In pytorch, for real matrices, the weights have size =     (output_dim, input_dim)
    # Remember that for our complex matrices it is the opposite: (input_dim,  output_dim)
    a = init_func(in_channels=weight.size(1), out_channels=weight.size(0),
                  kernel_size=None, rng=rng, criterion=init_criterion)
    a = torch.from_numpy(a)
    weight.data = a.type_as(weight.data)
    

def affect_conv_init(weight, kernel_size, init_func, rng, init_criterion):
    if 2 >= weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(real_weight.dim()))

    a = init_func(in_channels=weight.size(1), out_channels=weight.size(0),
                  kernel_size=kernel_size, rng=rng, criterion=init_criterion)
    a = torch.from_numpy(a)
    weight.data = a.type_as(weight.data)


def independent_filters_init(in_channels, out_channels, kernel_size, rng, criterion='glorot'):
    if kernel_size is not None:
        num_rows = out_channels * in_channels
        num_cols = np.prod(kernel_size)
    else:
        num_rows = in_channels
        num_cols = out_channels
    flat_shape = (int(num_rows), int(num_cols))
    z = rng.uniform(size=flat_shape)
    u, _, v = np.linalg.svd(z)
    orthogonal_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), v.T))
    if kernel_size is not None:
        if type(kernel_size) is int:
            indep_z = np.reshape(orthogonal_z, (num_rows,) + tuple((kernel_size,)))
        else:
            indep_z = np.reshape(orthogonal_z, (num_rows,) + (*kernel_size,))
    else:
        indep_z = orthogonal_z

    receptive_field = num_cols
    if kernel_size is not None:
        fan_out = out_channels * receptive_field
        fan_in  = in_channels  * receptive_field
    else:
        fan_out = out_channels
        fan_in  = in_channels
    if   criterion == 'glorot':
        desired_var = 2. / (fan_in + fan_out)
    elif criterion == 'he':
        desired_var = 2. / (fan_in)
    else:
        raise ValueError('Invalid criterion: ' + self.criterion)
    multip_z     =  np.sqrt(desired_var / np.var(indep_z))
    scaled_z     =  multip_z * indep_z
    if kernel_size is not None:
        if type(kernel_size) is int:
            kernel_shape =  (int(out_channels), int(in_channels)) + tuple((kernel_size,))
        else:
            kernel_shape =  (int(out_channels), int(in_channels)) + (*kernel_size,)
        weight  =  np.reshape(scaled_z, kernel_shape)
    else:
        weight  =  scaled_z

    return weight.T if kernel_size is None else weight


def real_init(in_channels, out_channels, rng, kernel_size=None, criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_out = out_channels * receptive_field
        fan_in  = in_channels  * receptive_field
    else:
        fan_out = out_channels
        fan_in  = in_channels
    if criterion == 'glorot':
        s = 2. / (fan_in + fan_out)
    elif criterion == 'he':
        s = 2. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    if kernel_size is None:
        size = (in_channels, out_channels)
    else:
        if type(kernel_size) is int:
            size = (out_channels, in_channels) + tuple((kernel_size,))
        else:
            size = (out_channels, in_channels) + (*kernel_size,)
    weight = rng.normal(loc=0, scale=np.sqrt(s), size=size)
    return weight.T if kernel_size is None else weight
