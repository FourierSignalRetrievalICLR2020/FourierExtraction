import argparse
import torch
import os
import shutil
import time
import numpy                   as     np
import torch.nn                as     nn
import torch.nn.functional     as     F
import sys
sys.path.append('..')
from models                    import ComplexUNet
from torch.optim               import Adam, SGD, RMSprop
from torch.nn.parallel         import DataParallel
from datasets                  import WSJ2MReader
from datasets.utils            import eval_sources, complex_mul
from sympy.utilities.iterables import multiset_permutations
sys.path.append('../..')
from complexmodels             import get_modulus, complex_product, cosloss, istftloss, STFT1d


class ISTFTLoss(nn.Module):
    def __init__(self, in_channels=1, kernel_size=257, window_fn=np.hanning, stride=128,
                 padding=128, dilation=1, nb_speakers=2, pil=True, loss_type='istftl2loss',
                 frequencyloss=None, real_penalty=1, imag_penalty=1e6):
        super(ISTFTLoss, self).__init__()
        self.loss_type        = loss_type
        self.input_type       = 'convolution'
        self.channels_axis    = 1
        self.nb_speakers      = nb_speakers
        self.pil              = pil
        self.stftmodule       = STFT1d(
            in_channels       = in_channels,
            kernel_size       = kernel_size,
            window_fn         = window_fn,
            stride            = stride,
            padding           = padding,
            dilation          = dilation,
            inverse           = True
        )
        self.frequencyloss    = frequencyloss
        if self.frequencyloss is not None:
            self.real_penalty = real_penalty
            self.imag_penalty = imag_penalty

    def forward(self, pred, truth, mask=None):
        if mask is not None:
            pred  = pred  * mask[:, None]
            truth = truth * mask[:, None]
        if   self.frequencyloss is None:
            return istftloss(
                pred,
                truth,
                stft_module   = self.stftmodule,
                input_type    = self.input_type,
                channels_axis = self.channels_axis,
                loss_type     = self.loss_type
            )
        elif self.frequencyloss == 'specl2loss':
            return istftloss(
                pred,
                truth,
                stft_module   = self.stftmodule,
                input_type    = self.input_type,
                channels_axis = self.channels_axis,
                loss_type     = self.loss_type
            ) + ((pred - truth) ** 2).mean()
        elif self.frequencyloss == 'speccosloss':
            return istftloss(
                pred,
                truth,
                stft_module   = self.stftmodule,
                input_type    = self.input_type,
                channels_axis = self.channels_axis,
                loss_type     = self.loss_type
            ) + cosloss(
                estimation    = pred,
                reference     = truth,
                input_type    = self.input_type,
                channels_axis = self.channels_axis,
                real_penalty  = self.real_penalty,
                imag_penalty  = self.imag_penalty,
            )


class CosLoss(nn.Module):
    def __init__(self, input_type='convolution', channels_axis=1,
                 real_penalty=1, imag_penalty=1e6, eps=1e-4,
                 nb_speakers=2, pil=True):
        super(CosLoss, self).__init__()
        self.loss_type        = 'speccosloss'
        self.input_type       = input_type
        self.channels_axis    = channels_axis
        self.real_penalty     = real_penalty
        self.imag_penalty     = imag_penalty
        self.nb_speakers      = nb_speakers
        self.pil              = pil

    def forward(self, pred, truth, mask=None):
        if mask is not None:
            pred  = pred  * mask[:, None]
            truth = truth * mask[:, None]
        return cosloss(
            estimation       = pred,
            reference        = truth,
            input_type       = self.input_type,
            channels_axis    = self.channels_axis,
            real_penalty     = self.real_penalty,
            imag_penalty     = self.imag_penalty,
        )


class SequenceLoss(nn.Module):
    def __init__(self, nb_speakers=2, pil=True):
        super(SequenceLoss, self).__init__()
        self.loss_type   = 'specl2loss'
        self.nb_speakers = nb_speakers
        self.pil         = pil

    def forward(self, pred, truth, mask=None):
        batch, seq, feature = pred.size()
        if mask is not None:
            pred   = pred   * mask[:, None]
            truth  = truth  * mask[:, None]
        return ((pred - truth) ** 2).mean()


def adjust_learning_rate(optimizer, epoch):
    if   epoch >=   0 and epoch <  10:
        lrate = 0.01
        if epoch == 0:
            print("Current learning rate value is " + str(lrate))
    elif epoch >=  10 and epoch < 100:
        lrate = 0.1
        if epoch == 10:
            print("Current learning rate value is "+str(lrate))
    elif epoch >= 100 and epoch < 120:
        lrate = 0.01
        if epoch == 100:
            print("Current learning rate value is "+str(lrate))
    elif epoch >= 120 and epoch < 150:
        lrate = 0.001
        if epoch == 120:
            print("Current learning rate value is "+str(lrate))
    elif epoch >= 150:
        lrate = 0.0001
        if epoch == 150:
            print("Current learning rate value is "+str(lrate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrate


def save_checkpoint(state, is_best, save_path='./',
                    filename='checkpoint.pth.tar'):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))

def parse_args():
    parser = argparse.ArgumentParser(description='speech-separation')
    parser.add_argument('--experiment-path', type=str,   default='/blablabla/', help='Location for writing results')
    parser.add_argument('--load-path',       type=str,   default='',                       help='File with parameter to be loaded to the model. Usually path to the latest checkpoint (default: none).')
    parser.add_argument('--data-path',       type=str,   default='/WSJ0_8k/',              help='Path to the dataset folder.')
    parser.add_argument('--loss-type',       type=str,   default='istftl2loss',            help='The type of loss to use',                                                                         choices=['specl2loss', 'speccosloss', 'istftl2loss', 'istftcosloss'])
    parser.add_argument('--freqloss',        type=str,   default=None,                     help='The type of frequency loss to use along with the istft loss.',                                    choices=['specl2loss', 'speccosloss'])
    parser.add_argument('--optimizer',       type=str,   default='sgd',                    help='The type of optimizer to use',                                                                    choices=['adam','sgd','rmsprop'])
    parser.add_argument('--upsampling-mode', type=str,   default='bilinear',               help='The type of interpolation to perform',                                                            choices=['bilinear','nearest','transpose'])
    parser.add_argument('--model-type',      type=str,   default='complexunet',            help='The type of model to use',                                                                        choices=['complexunet'])
    parser.add_argument('--blocktype',       type=str,   default='basic',                  help='the type of residual block for up and down unet layers',                                          choices=['basic', 'bottleneck', 'dense'])
    parser.add_argument('--space',           type=str,   default='complex',                help='The paramter space in which the parameters of the network are learned',                           choices=['complex', 'real'])
    parser.add_argument('--weight-init',     type=str,   default='complex',                help='The weight initialization to perform',                                                            choices=['complex', 'real', 'unitary', 'orthogonal'])
    parser.add_argument('--init-criterion',  type=str,   default='he',                     help='The weight initialization criterion to perform',                                                  choices=['he', 'glorot'])
    parser.add_argument('--learning-rate',   type=float, default=0.01,                     help='Learning rate')
    parser.add_argument('--weight-decay',    type=float, default=0.0001,                   help='Weight decay.')
    parser.add_argument('--momentum',        type=float, default=0.9,                      help='Momentum for SGD.')
    parser.add_argument('--alpha',           type=float, default=0.9,                      help='Alpha, called also Rho hyperparameter for the RMSprop optimizer.')
    parser.add_argument('--eps',             type=float, default=1e-8,                     help='the epsilon hyperparameter for RMSprop and Adam.')
    parser.add_argument('--beta1',           type=float, default=0.9,                      help='Beta1 for Adam.')
    parser.add_argument('--beta2',           type=float, default=0.999,                    help='Beta2 for Adam.')
    parser.add_argument('--clip',            type=float, default=1,                        help="Gradient norm above which we perform the clipping.")
    parser.add_argument('--stochastic-drop', type=float, default=None,                     help="Stochastic depth constant. Should be between 0 and 1.")
    parser.add_argument('--dropcopy',        type=float, default=None,                     help="Dropout ratio for the input copies that are learned. Should be between 0 and 1.")
    parser.add_argument('--real-penalty',    type=float, default=1,                        help='Penalty for the real part of the spectral cosine loss.')
    parser.add_argument('--imag-penalty',    type=float, default=1e6,                      help='Penalty for the imag part of the spectral cosine loss.')
    parser.add_argument('--efficient',                   default=False,                    help='When mentioned use the efficient implementation of densenet if blocktype is dense.',              action='store_true')
    parser.add_argument('--bottleneck',                  default=False,                    help='When mentioned use the bottleneck block of densenet if blocktype is dense.',                      action='store_true')
    parser.add_argument('--schedule',                    default=False,                    help='When mentioned use the schedule function for the learning rate.',                                 action='store_true')
    parser.add_argument('--nonesterov',                  default=True,                     help='When mentioned the nesterov momentum is not used during SGD.',                                    action='store_false')
    parser.add_argument('--rmspop-momentum',             default=False,                    help='When mentioned the momentum is used during the training with rmsprop.',                           action='store_true')
    parser.add_argument('--nopil',                       default=True,                     help='When mentioned the permutation invariant loss is not used during training.',                      action='store_false')
    parser.add_argument('--avg-copies',                  default=False,                    help='whether to average the copies of the predicted output or not',                                    action='store_true')
    parser.add_argument('--stochdrop-schedule',          default=False,                    help='whether to use the stochastic drop schedule or not',                                              action='store_true')
    parser.add_argument('--nb-resblocks',    type=int,   default=None,                     help='number of residual blocks in each of the downsampling and upsampling blocks,')
    parser.add_argument('--growth-rate',     type=int,   default=12,                       help='number of feature maps to add to the input when using densenets inside the unets blocks.')
    parser.add_argument('--start-fmaps',     type=int,   default=64,                       help='number of the number of output feature maps in the first downsampling unet block,')
    parser.add_argument('--epochs',          type=int,   default=200,                      help='Number of epochs')
    parser.add_argument('--batch-size',      type=int,   default=64,                       help='batch size for training.')
    parser.add_argument('--n-sources',       type=int,   default=2,                        help='Number of sources.')
    parser.add_argument('--print-interval',  type=int,   default=100,                      help='Print interval.')
    parser.add_argument('--seed',            type=int,   default=2018,                     help="Seed for PRNGs.")
    parser.add_argument('--nb-copies',       type=int,   default=None,                     help="Number of copies of the inferred speech for a given speaker")

    return parser.parse_args()


def main(experiment_path, load_path, data_path, loss_type, freqloss, optimizer, upsampling_mode, model_type, blocktype,
         space, weight_init, init_criterion, learning_rate, weight_decay, momentum, alpha, eps, beta1, beta2, clip,
         stochastic_drop, dropcopy, real_penalty, imag_penalty, efficient, bottleneck, schedule, nonesterov, rmspop_momentum, nopil,
         avg_copies, stochdrop_schedule, nb_resblocks, growth_rate, start_fmaps, epochs, batch_size, n_sources,
         print_interval, seed, nb_copies, **kwargs):
    # First thing we check is to see if we should load a prexisting model.
    if load_path:
        load_path   = os.path.join(experiment_path, load_path)

    chkptFilename   = os.path.join(experiment_path, 'checkpoint.pth.tar')
    isResuming      = os.path.isfile(chkptFilename) or load_path
    if isResuming and (not load_path):
        load_path   = chkptFilename
    # If experiment path is not created, we create it.
    if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

    print('...preparing dataset')
    trainset        = WSJ2MReader(data_path, 'train', random_seed=seed)
    devset          = WSJ2MReader(data_path, 'dev',   random_seed=seed)
    print('...number of training utterances {}'.format(trainset.n_examples))
    print('...number of dev utterances {}'.format(devset.n_examples))
    bptt_len        = None
    train_stream    = trainset.read(batch=batch_size, sortseq=False, normalize=False, bptt_len=bptt_len)
    dev_stream      = devset.read(  batch=batch_size, sortseq=False, normalize=False, bptt_len=bptt_len)
    print('...building model')
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'complexunet':
        model       = ComplexUNet(in_channels=1, nb_speakers=2, mode=upsampling_mode, nb_residual_blocks=nb_resblocks,
                                  start_fmaps=start_fmaps, blocktype=blocktype, growth_rate=growth_rate,
                                  space=space, seed=seed, weight_init=weight_init, init_criterion=init_criterion,
                                  efficient=efficient, bottleneck=bottleneck, nb_copies=nb_copies, avg_copies=avg_copies,
                                  stochastic_drop=stochastic_drop, stochdrop_schedule=stochdrop_schedule, dropcopy=dropcopy).to(device)
    print(model)
    if torch.cuda.device_count() > 1:
        print('...multi-gpu training')
        model       = DataParallel(model)
    # loss function
    print('...define loss function')
    if freqloss is None:
        if   loss_type == 'specl2loss':
            maskloss = SequenceLoss(nb_speakers=n_sources, pil=nopil)
        elif loss_type == 'speccosloss':
            maskloss = CosLoss(real_penalty=real_penalty,
                               imag_penalty=imag_penalty,
                               nb_speakers=n_sources, pil=nopil)
        elif loss_type in {'istftl2loss', 'istftcosloss'}:
            maskloss = ISTFTLoss(nb_speakers=n_sources, pil=nopil, loss_type=loss_type)
    else:
        print("USING TIME FREQUENCY LOSS!")
        if loss_type in {'istftl2loss', 'istftcosloss'} and freqloss in {'specl2loss', 'speccosloss'}:
            maskloss = ISTFTLoss(nb_speakers=n_sources, pil=nopil, loss_type=loss_type,
                                 frequencyloss=freqloss, real_penalty=real_penalty,
                                 imag_penalty=imag_penalty)
        else:
            raise Exception(
                "When a frequency loss is used it should be used along with a loss on the temporal signal." +
                "Found freqloss == " + str(freqloss) + " and loss_type == " + str(loss_type) + "."
            )
    maskloss = maskloss.to(device)

    # optimizer
    print('...define optimizer')
    if   optimizer == 'adam':
        optimizer = Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=learning_rate,
                         weight_decay=weight_decay,
                         betas=(beta1, beta2),
                         eps=eps)
    elif optimizer == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad,
                               model.parameters()),
                        lr=learning_rate,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        nesterov=nonesterov)
    elif optimizer == 'rmsprop':
        rmsmom = momentum if rmspop_momentum else 0
        optimizer = RMSprop(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=learning_rate,
                            alpha=alpha,
                            eps=eps,
                            weight_decay=weight_decay,
                            momentum=rmsmom)
    # initialize training arguments
    start_epoch            =  0
    start_i                =  0
    bestloss               =  np.inf
    best_epoch             =  0
    best_iter              =  0
    training_losses        =  []
    validation_losses      =  []
    trainCost              =  []
    devSDR, devSIR, devSAR =  [], [], []
    params_num             =  0
    best_devSDR            =  0.
    # perform the loading if we need to do it
    if load_path:
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            for k in checkpoint.keys():
                if k not in ['optimizer',
                             'training_losses',
                             'validation_losses',
                             'training_losses',
                             'train_reader',
                             'model_state_dict']:
                    print("    " + str(k) + ": " + str(checkpoint[k]))
            start_epoch                       =               checkpoint['start_epoch']
            clip                              =               checkpoint['clip']
            schedule                          =               checkpoint['schedule']
            epochs                            =               checkpoint['epochs']
            batch_size                        =               checkpoint['batch_size']
            print_interval                    =               checkpoint['print_interval']
            best_devSDR                       =               checkpoint['bestSDR']
            best_epoch                        =               checkpoint['best_epoch']
            trainCost                         =               checkpoint['training_losses']
            devSDR                            =               checkpoint['devSDR']
            devSIR                            =               checkpoint['devSIR']
            devSAR                            =               checkpoint['devSAR']
            trainset                          =               checkpoint['train_reader']
            devset                            =               checkpoint['dev_reader']
            maskloss                          =               checkpoint['maskloss']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(load_path))

    m = model.module if isinstance(model, DataParallel) else model

    print("    loss_type:              "      + str(maskloss.loss_type))
    if maskloss.loss_type == 'speccosloss':
        print("    real_penalty:           "      + str(maskloss.real_penalty))
        print("    imag_penalty:           "      + str(maskloss.imag_penalty))
    print("    PIL:                    "      + str(maskloss.pil))
    print("    maskloss:               "      + str(maskloss.modules))
    print("    optimizer:              "      + str(optimizer.__module__))
    print("    upsampling_mode         "      + str(m.mode))
    print("    model_type              "      + str(model_type))
    print("    weight_init             "      + str(m.weight_init))
    print("    init_criterion          "      + str(m.init_criterion))
    if isinstance(m, ComplexUNet):
        print("    space                   "  + str(m.space))
        print("    blocktype               "  + str(m.blocktype))
        print("    nb_copies               "  + str(m.nb_copies))
        print("    avg_copies              "  + str(m.avg_copies))
        print("    stochastic_drop         "  + str(m.stochastic_drop))
        print("    stochdrop_schedule      "  + str(m.stochdrop_schedule))
        print("    nb_residual_blocks      "  + str(m.nb_residual_blocks))
        print("    dropcopy                "  + str(m.dropcopy))
        if m.blocktype in {'dense'}:
            print("    growth_rate             "  + str(m.growth_rate))
            if m.space == 'real':
                print("    efficient               "      + str(m.efficient))
                print("    bottleneck              "      + str(m.bottleneck))
    print("    start_fmaps             "      + str(m.start_fmaps))
    print("    learning_rate           "      + str(optimizer.param_groups[0]['lr']))
    print("    weight_decay            "      + str(optimizer.param_groups[0]['weight_decay']))
    if isinstance(  optimizer, SGD):
        print("    nesterov                "  + str(optimizer.param_groups[0]['nesterov']))
        print("    momentum                "  + str(optimizer.param_groups[0]['momentum'])) 
    elif isinstance(optimizer, RMSprop):
        print("    alpha                   "  + str(optimizer.param_groups[0]['alpha']))
        print("    rms_momentum            "  + str(optimizer.param_groups[0]['momentum']))
        print("    eps                     "  + str(optimizer.param_groups[0]['eps']))
    elif isinstance(optimizer, Adam):
        print("    beta1                   "  + str(optimizer.param_groups[0]['betas'][0]))
        print("    beta2                   "  + str(optimizer.param_groups[0]['betas'][1]))
        print("    eps                     "  + str(optimizer.param_groups[0]['eps']))
    print("    clip:                   "      + str(clip))
    print("    schedule                "      + str(schedule))
    print("    start_epoch:            "      + str(start_epoch))
    print("    epochs:                 "      + str(epochs))
    print("    print_interval:         "      + str(print_interval))
    print("    batch_size:             "      + str(batch_size))
    print("    nb_speakers:            "      + str(maskloss.nb_speakers))
    print("    trainset_seed:          "      + str(trainset.random_seed))
    print("    model_seed:             "      + str(m.seed))

    for param in model.parameters():
        if param.requires_grad:
            params_num += np.prod(param.size())
    print("number of parameters: {}".format(params_num))

    print('...start training')
    for epoch in range(start_epoch, epochs):
        model.train()
        if schedule:
            adjust_learning_rate(optimizer, epoch)
        train_cost, total_train_cost = 0, 0
        print("------ Epoch {} ------".format(epoch))
        train_iters = 0
        for data in train_stream:
            source, target, mask, _ = (torch.FloatTensor(_data).to(device)
                                       for _data in data)
            optimizer.zero_grad()
            loss   = 0
            output = model(source[:, None]) # By providing a None in the second dimension, we actually add a new dimension so
            # source[:, None] has shape of (batch_size, 1, feature_size, nb_spectrums)
            #if torch.cuda.is_available():
            #    torch.cuda.empty_cache()
            if not maskloss.pil:
                for i in range(n_sources):
                    if nb_copies is None:
                        loss += maskloss(
                            pred    =complex_product(source, output[i], input_type='convolution'),  #complex_mul(source, output[i]),
                            truth   =target[:, i],
                            mask    =mask
                        )
                    else:
                        loss += maskloss(
                            pred    =output[i],
                            truth   =target[:, i],
                            mask    =mask
                        )
                loss.backward()
            else:
                list_losses = []
                speakers_indices = range(n_sources)
                for speakers_idx in multiset_permutations(speakers_indices):
                    loss = 0
                    j = 0
                    for i in speakers_idx:
                        if nb_copies is None:
                            loss += maskloss(
                                pred     =complex_product(source, output[j], input_type='convolution'),  # complex_mul(source, output[j]),
                                truth    =target[:, i],
                                mask     =mask
                            )
                        else:
                            loss += maskloss(
                                pred     =output[j],
                                truth    =target[:, i],
                                mask     =mask
                            )
                        j += 1
                    list_losses.append(loss)
                loss = min(torch.stack(list_losses))
                loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                                  model.parameters()), clip)
            optimizer.step()
            #if torch.cuda.is_available():
            #    torch.cuda.empty_cache()
            train_cost    += loss.item()
            train_iters   += 1
            if train_iters % print_interval == 0:
                print("train loss {} at batch \
                      {}".format(train_cost / print_interval, train_iters))
                train_cost = 0

            total_train_cost += loss.item()

        trainCost.append(total_train_cost / train_iters)
        train_stream = trainset.read(batch=batch_size, sortseq=False, normalize=False, bptt_len=bptt_len)

        print('...start evaluation')
        dev_sdr, dev_sir, dev_sar = 0, 0, 0
        dev_iters = 0
        model.eval()

        with torch.no_grad():
            for idx, data in enumerate(dev_stream):
                source, target, mask, sourcelen = (torch.FloatTensor(_data).to(device)
                                                   for _data in data)
                output   = model(source[:, None])

                #if torch.cuda.is_available():
                #    torch.cuda.empty_cache()
                output_pred  = []
                for i in range(n_sources):
                    if nb_copies is None:
                        output_pred.append(complex_product(source, output[i], input_type='convolution').cpu().data.numpy())  # complex_mul(source, output[i]).cpu().data.numpy())
                    else:
                        output_pred.append(output[i].cpu().data.numpy())

                if (idx + 1) % 10 == 0:
                    sdr, sir, sar = eval_sources(source.cpu().data.numpy(),
                                                 target.cpu().data.numpy(),
                                                 output_pred,
                                                 mask.cpu().data.numpy(),
                                                 sourcelen.cpu().data.numpy())
                    dev_sdr += sdr
                    dev_sir += sir
                    dev_sar += sar
                    print(sdr, sir, sar)
                dev_iters += 1
                #if torch.cuda.is_available():
                #    torch.cuda.empty_cache()

        devSDR.append(dev_sdr / (dev_iters // 10))
        devSIR.append(dev_sir / (dev_iters // 10))
        devSAR.append(dev_sar / (dev_iters // 10))

        print("SDR on valid set {}".format(devSDR[-1]))
        print("SIR on valid set {}".format(devSIR[-1]))
        print("SAR on valid set {}".format(devSAR[-1]))
        print("------- End -------")

        dev_stream      = devset.read(batch=batch_size, sortseq=False, normalize=False, bptt_len=bptt_len)
        if len(devSDR) == 1 or devSDR[-1] > best_devSDR:
            is_best     = True
            best_devSDR = devSDR[-1]
            best_epoch  = epoch
        else:
            is_best     = False
        state           = {
            'start_epoch'        :      epoch + 1,
            'model_state_dict'   :      model.state_dict(),
            'bestSDR'            :      best_devSDR,
            'optimizer'          :      optimizer.state_dict(),
            'clip'               :      clip,
            'schedule'           :      schedule,
            'epochs'             :      epochs,
            'batch_size'         :      batch_size,
            'print_interval'     :      print_interval,
            'best_epoch'         :      best_epoch,
            'maskloss'           :      maskloss,
            'training_losses'    :      trainCost,
            'devSDR'             :      devSDR,
            'devSIR'             :      devSIR,
            'devSAR'             :      devSAR,
            'train_reader'       :      trainset,
            'dev_reader'         :      devset
        }
        if is_best:
            print("\rSaving Best Model.\r")
            save_checkpoint(state, is_best, save_path=experiment_path)
        else:
            print("\rSaving last Model.\r")
            save_checkpoint(state, False,   save_path=experiment_path)

args = parse_args()
main(**args.__dict__)
