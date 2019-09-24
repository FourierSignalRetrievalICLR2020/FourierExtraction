import torch
import torch.nn            as     nn
import torch.nn.functional as     F
import numpy               as     np
from   numpy.random        import RandomState


def check_input(input):
    if input.dim() not in {2, 3}:
        raise Exception(
            "complex linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1]

    if nb_hidden % 2 != 0:
        raise Exception(
            "complex Tensors have to have an even number of hidden dimensions."
            " input.size()[1] = " + str(nb_hidden)
        )


def check_conv_input(input, channels_axis=1):
    if input.dim() not in {3, 4, 5}:
        raise Exception(
            "complex convolution accepts only input of dimension 3, 4 or 5."
            " input.dim = " + str(input.dim())
        )

    nb_channels = input.size(channels_axis)
    if nb_channels % 2 != 0:
        print("input.size()" + str(input.size()))
        raise Exception(
            "complex Tensors have to have an even number of feature maps."
            " input.size()[1] = " + str(nb_channels)
        )


def get_real(input, input_type='linear', channels_axis=1):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input, channels_axis=channels_axis)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(1, 0, nb_hidden // 2) # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(2, 0, nb_hidden // 2) # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, 0, nb_featmaps // 2)


def get_imag(input, input_type='linear', channels_axis=1):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input, channels_axis=channels_axis)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(1, nb_hidden // 2, nb_hidden // 2) # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(2, nb_hidden // 2, nb_hidden // 2) # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, nb_featmaps // 2, nb_featmaps // 2)


def get_conjugate(input, input_type='linear', channels_axis=1):
    input_imag = get_imag(input, input_type, channels_axis)
    input_real = get_real(input, input_type, channels_axis)
    if   input_type == 'linear':
        return torch.cat([input_real, -input_imag], dim=-1)
    elif input_type == 'convolution':
        return torch.cat([input_real, -input_imag], dim=channels_axis)


def get_modulus(input, vector_form=False, input_type='convolution'):
    if   input_type == 'linear':
        check_input(input)
        a = get_real(input)
        b = get_imag(input)
        if vector_form:
            return torch.sqrt(a * a + b * b)
        else:
            return torch.sqrt((a * a + b * b).sum(dim=-1))
    elif input_type == 'convolution':
        check_conv_input(input)
        a = get_real(input, input_type='convolution')
        b = get_imag(input, input_type='convolution')
        if vector_form:
            return torch.sqrt((a * a + b * b))
        else:
            # we will return the modulus of each feature map
            return torch.sqrt((a * a + b * b).view(a.size(0), a.size(1), -1).sum(dim=-1))


def get_normalized(input, eps=0.001, threshold=1, input_type='linear'):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        data_modulus = get_modulus(input)
        m = data_modulus.unsqueeze(1).expand_as(input)
        mask = m < threshold
        return (input / (m + eps)) * (1 - mask.type_as(input)) + input * mask.type_as(input)
    else: # (which means if operation is convolutional)
        data_modulus = get_modulus(input, input_type='convolution')
        data_modulus = data_modulus.repeat(1, 2)
        if   input.dim() == 3:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1).expand_as(input)
        elif input.dim() == 4:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1, 1).expand_as(input)
        elif input.dim() == 5:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1, 1, 1).expand_as(input)
        
        mask = data_modulus < threshold
        return (input / (data_modulus + eps)) * (1 - mask.type_as(input)) + input * mask.type_as(input)


def complex_product(z0, z1, input_type='linear'):
    """
    Applies a complex product z0 * z1:
    Shape:
        - z0, z1: (batch_size, complex_number)
    code:
        The code is equivalent of doing the following:
        >>> (xx' - yy') + (xy' + x'y)i
    """

    # NEED TO CHECK THE SHAPE OF THE INPUT
    #check_input(input)
    
    #xx' and yy'
    r_base = z0*z1

    #(xx' - yy')
    real   = get_real(r_base, input_type) - get_imag(r_base, input_type)

    #xy' and x'y
    cat_channel = -1 if input_type == 'linear' else 1
    i_base = z0 * torch.cat([get_imag(z1, input_type), get_real(z1, input_type)], dim=cat_channel)

    #(xy' + x'y)
    imag   = get_real(i_base, input_type) + get_imag(i_base, input_type)

    return torch.cat([real, imag], dim=cat_channel)


def cosloss(estimation, reference, input_type='convolution',
            channels_axis=1, real_penalty=1, imag_penalty=1e6,
            eps=1e-4):
    cm  = complex_product(
        get_conjugate(estimation, input_type=input_type),
        reference,
        input_type=input_type
    )
    estimation_norm = torch.norm(estimation)
    reference_norm  = torch.norm(reference)
    cm_normalized   = cm / ((estimation_norm * reference_norm).clamp(eps))
    cmr             =  get_real(cm_normalized, input_type=input_type, channels_axis=channels_axis).sum()         # we want to maximize its sum                  to 1.
    cmi             = (get_imag(cm_normalized, input_type=input_type, channels_axis=channels_axis) ** 2).sum()   # we want to minimize the sum of its squares   to 0.
    return -real_penalty * cmr + imag_penalty * cmi


def istftloss(estimation, reference, stft_module, input_type='convolution', channels_axis=1, loss_type='istftl2loss'):
    estr = get_real(estimation, input_type='convolution', channels_axis=channels_axis)
    esti = get_imag(estimation, input_type='convolution', channels_axis=channels_axis)
    refr = get_real(reference,  input_type='convolution', channels_axis=channels_axis)
    refi = get_imag(reference,  input_type='convolution', channels_axis=channels_axis)

    estr = torch.cat([estr,  estr.flip(channels_axis).narrow(channels_axis, 0, estr.size(channels_axis)-1)], dim=channels_axis)
    esti = torch.cat([esti, -esti.flip(channels_axis).narrow(channels_axis, 0, esti.size(channels_axis)-1)], dim=channels_axis)
    refr = torch.cat([refr,  refr.flip(channels_axis).narrow(channels_axis, 0, refr.size(channels_axis)-1)], dim=channels_axis)
    refi = torch.cat([refi, -refi.flip(channels_axis).narrow(channels_axis, 0, refi.size(channels_axis)-1)], dim=channels_axis)

    estimation_temporal_r, estimation_temporal_i = stft_module(estr, esti)
    reference_temporal_r,  reference_temporal_i  = stft_module(refr, refi)
    if   loss_type == 'istftl2loss':
        return ((reference_temporal_r - estimation_temporal_r) ** 2 + (reference_temporal_i - estimation_temporal_i) ** 2).mean()
    elif loss_type == 'istftcosloss':
        etr = estimation_temporal_r.view(-1)
        rtr = reference_temporal_r.view( -1)
        return - F.cosine_similarity(etr, rtr, dim=0)


def complex_linear(input, real_weight, imag_weight, bias=None, **kwargs):
    """
    Applies a complex linear transformation to the incoming data:
    Shape:
        - Input:       (batch_size, nb_complex_elements_in * 2)
        - real_weight: (nb_complex_elements, nb_complex_elements_out)
        - imag_weight: (nb_complex_elements, nb_complex_elements_out)
        - Bias:        (nb_complex_elements_out * 2)
        - Output:      (batch_size, nb_complex_elements_out * 2)
    code:
        The code is equivalent of doing the following:
        >>> input_real = get_real(input)
        >>> input_imag = get_imag(input)
        >>> r = input_real.mm(real_weight) - input_imag.mm(imag_weight)
        >>> i = input_real.mm(imag_weight) + input_imag.mm(real_weight)

        >>> if bias is not None:
        >>>    return torch.cat([r, i], dim=1) + bias
        >>> else:
        >>>    return torch.cat([r, i], dim=1)
    """

    check_input(input)
    cat_kernels_4_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_kernels_4_imag = torch.cat([imag_weight,  real_weight], dim=0)
    cat_kernels_4_complex = torch.cat([cat_kernels_4_real, cat_kernels_4_imag], dim=1)
    if bias is not None:
        if input.dim() == 3:
            """if input.size()[0] != 1:
                raise Exception(
                    "Time dimension of the input different than 1."
                    " input.dim = " + str(input.dim())
                )"""
            if input.size(0) == 1:
                input = input.squeeze(0)
                return torch.addmm(bias, input, cat_kernels_4_complex)
            else:
                return input.matmul(cat_kernels_4_complex) + bias
        else:
            return torch.addmm(bias, input, cat_kernels_4_complex)
    else:
        return input.mm(cat_kernels_4_complex)


def complex_conv(input, real_weight, imag_weight, bias=None, transpose=False, **convargs):
    """
    Applies a complex convolution to the incoming data:
    Shape:
        - Input:       (batch_size, nb_complex_channels_in * 2, *signal_length)
        - real_weight: (nb_complex_channels_out, nb_complex_channels_in, *kernel_size)
        - imag_weight: (nb_complex_channels_out, nb_complex_channels_in, *kernel_size)
        - Bias:        (nb_complex_channels_out * 2)
        - Output:      (batch_size, nb_complex_channels_out * 2, *signal_out_length)
        - convArgs =   {"strides":        strides,
                        "padding":        padding,
                        "dilation_rate":  dilation}
    """
    check_conv_input(input)
    if transpose:
        cat_kernels_4_real = torch.cat([ real_weight,  imag_weight], dim=1)
        cat_kernels_4_imag = torch.cat([-imag_weight,  real_weight], dim=1)
    else:
        cat_kernels_4_real = torch.cat([real_weight, -imag_weight], dim=1)
        cat_kernels_4_imag = torch.cat([imag_weight,  real_weight], dim=1)
    cat_kernels_4_complex  = torch.cat([cat_kernels_4_real, cat_kernels_4_imag], dim=0)
    if   input.dim() == 3:
        if transpose:
            convfunc = F.conv_transpose1d
        else:
            convfunc = F.conv1d
    elif input.dim() == 4:
        if transpose:
            convfunc = F.conv_transpose2d
        else:
            convfunc = F.conv2d
    elif input.dim() == 5:
        if transpose:
            convfunc = F.conv_transpose3d
        else:
            convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))
    return convfunc(input, cat_kernels_4_complex, bias, **convargs)


def unitary_init(in_features, out_features, rng, criterion='glorot'):
    r = rng.uniform(size=(in_features, out_features))
    i = rng.uniform(size=(in_features, out_features))
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    num_rows = in_features
    num_cols = out_features
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    indep_real = unitary_z.real
    indep_imag = unitary_z.imag
    if criterion == 'glorot':
        desired_var = 1. / (in_features + out_features)
    elif criterion == 'he':
        desired_var = 1. / (in_features)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    multip_real = np.sqrt(desired_var / np.var(indep_real))
    multip_imag = np.sqrt(desired_var / np.var(indep_imag))
    weight_real = multip_real * indep_real
    weight_imag = multip_imag * indep_imag

    return (weight_real, weight_imag)


def complex_init(in_features, out_features, rng, kernel_size=None, criterion='glorot', transpose=False):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_out = out_features * receptive_field
        fan_in  = in_features  * receptive_field
    else:
        fan_out = out_features
        fan_in  = in_features
    if criterion == 'glorot':
        s = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        s = 1. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    if kernel_size is None:
        size = (in_features, out_features)
    else:
        if not transpose:
            if type(kernel_size) is int:
                size = (out_features, in_features) + tuple((kernel_size,))
            else:
                size = (out_features, in_features) + (*kernel_size,)
        else:
            if type(kernel_size) is int:
                size = (in_features, out_features) + tuple((kernel_size,))
            else:
                size = (in_features, out_features) + (*kernel_size,)
    modulus = rng.rayleigh(scale=s,                size=size)
    phase   = rng.uniform (low=-np.pi, high=np.pi, size=size)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    return (weight_real, weight_imag)


def independent_complex_filters_init(in_channels, out_channels, kernel_size, rng, criterion='glorot', transpose=False):
    if kernel_size is not None:
        num_rows = out_channels * in_channels
        num_cols = np.prod(kernel_size)
    else:
        num_rows = in_channels
        num_cols = out_channels
    flat_shape = (int(num_rows), int(num_cols))
    r = rng.uniform(size=flat_shape)
    i = rng.uniform(size=flat_shape)
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    real_unitary = unitary_z.real
    imag_unitary = unitary_z.imag
    if kernel_size is not None:
        if type(kernel_size) is int:
            indep_real = np.reshape(real_unitary, (num_rows,) + (kernel_size,))
            indep_imag = np.reshape(imag_unitary, (num_rows,) + (kernel_size,))
        else:
            indep_real = np.reshape(real_unitary, (num_rows,) + kernel_size)
            indep_imag = np.reshape(imag_unitary, (num_rows,) + kernel_size)
    else:
        indep_real = np.reshape(real_unitary, (num_rows, num_cols))
        indep_imag = np.reshape(imag_unitary, (num_rows, num_cols))

    receptive_field = num_cols
    if kernel_size is not None:
        fan_out = out_channels * receptive_field
        fan_in  = in_channels  * receptive_field
    else:
        fan_out = out_channels
        fan_in  = in_channels
    if   criterion == 'glorot':
        desired_var = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        desired_var = 1. / (fan_in)
    else:
        raise ValueError('Invalid criterion: ' + self.criterion)
    multip_real  =  np.sqrt(desired_var / np.var(indep_real))
    multip_imag  =  np.sqrt(desired_var / np.var(indep_imag))
    scaled_real  =  multip_real * indep_real
    scaled_imag  =  multip_imag * indep_imag
    if kernel_size is not None:
        if not transpose:
            if type(kernel_size) is int:
                kernel_shape = (int(out_channels), int(in_channels)) + (kernel_size,)
            else:
                kernel_shape = (int(out_channels), int(in_channels)) + kernel_size
        else:
            if type(kernel_size) is int:
                kernel_shape = (int(in_channels), int(out_channels)) + (kernel_size,)
            else:
                kernel_shape = (int(in_channels), int(out_channels)) + kernel_size
    else:
        kernel_shape =  (int(out_channels), int(in_channels))
    weight_real  =  np.reshape(scaled_real, kernel_shape)
    weight_imag  =  np.reshape(scaled_imag, kernel_shape)

    return (weight_real, weight_imag)


def affect_init(real_weight, imag_weight, init_func, rng, init_criterion):
    if real_weight.size() != imag_weight.size():
         raise ValueError('The real and imaginary weights '
                          'should have the same size . Found: '
                          + str(real_weight.size()) + ' and '
                          + str(imag_weight.size()))
    elif real_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(real_weight.dim()))

    a, b = init_func(real_weight.size(0), real_weight.size(1), rng, init_criterion)
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def affect_conv_init(real_weight, imag_weight, kernel_size, init_func, rng, init_criterion, transpose=False):
    if real_weight.size() != imag_weight.size():
         raise ValueError('The real and imaginary weights '
                          'should have the same size . Found: '
                          + str(real_weight.size()) + ' and '
                          + str(imag_weight.size()))
    elif 2 >= real_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(real_weight.dim()))

    in_channels  = real_weight.size(0) if transpose else real_weight.size(1)
    out_channels = real_weight.size(1) if transpose else real_weight.size(0)
    a, b = init_func(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        rng=rng,
        criterion=init_criterion,
        transpose=transpose
    )
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size, transpose=False):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:# in case it is 2d or 3d.
        if   operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if   operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        if transpose:
            w_shape = (in_channels, out_channels) + (*ks,)
        else:
            w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape
 


def apply_complex_mask(input, mask, dropout_type='complex', operation='linear'):
    if dropout_type == 'complex':
        input_real_masked = get_real(input, input_type=operation) * mask
        input_imag_masked = get_imag(input, input_type=operation) * mask
        return torch.cat([input_real_masked, input_imag_masked], dim=1)
    elif dropout_type == 'regular':
        return input * mask
    else:
        raise Exception("dropout_type accepts only 'complex' or 'regular'. Found dropout_type = "
                        + str(dropout_type))


def apply_complex_dropout(input, dropout_p, rng, do_dropout=True, dropout_type='complex', operation='linear'):
    size = input.data.size()
    s = []
    for i in range(input.dim()):
        s.append(size[i])
    if dropout_type == 'complex':
            s[1] = s[1] // 2
    elif dropout_type != 'regular':
        raise Exception("dropout_type accepts only 'complex' or 'regular'. Found dropout_type = "
                        + str(dropout_type))
    s = tuple(s)
    mask = create_dropout_mask(dropout_p, s, rng, input.data.type(), operation)
    return apply_complex_mask(input, mask, dropout_type, operation) / (1 - dropout_p) if do_dropout else input
