import numpy as np
import os
import glob
from .utils import stft, istft
from .utils import AVER_MAG


class WSJ2MReader(object):
    def __init__(self, path='/mnt/home/WSJ0_8k', subset='train',
                 random_seed=None):
        super(WSJ2MReader, self).__init__()
        self.subset = subset
        if subset == 'train':
            self.path = os.path.join(path, 'tr')
        elif subset == 'dev':
            self.path = os.path.join(path, 'cv')
        elif subset == 'test':
            self.path = os.path.join(path, 'tt')
        file_names = glob.glob(os.path.join(self.path, 'mix', '*.wav'))
        self.triplets = []
        self.n_examples = len(file_names)

        for fi in file_names:
            finame = fi.split('/')[-1]
            self.triplets.append((os.path.join(self.path, 'mix', finame),
                                  os.path.join(self.path, 's1',  finame),
                                  os.path.join(self.path, 's2',  finame)))
        self.random_seed = random_seed if random_seed is not None else 2018
        self.rng = np.random.RandomState(self.random_seed)

    def read(self, batch=1, sample_rate=8000, feature_size=258,
             sortseq=False, normalize=False, bptt_len=None):
        self.rng.shuffle(self.triplets)
        n_iters  = len(self.triplets) // batch
        leftover = len(self.triplets) % batch
        if leftover != 0:
            n_iters += 1

        for i in range(n_iters):
            sources, targets, sources_len = [], [], []
            for j in range(batch):
                source_spec, source_len = stft(self.triplets[i * batch + j][0], sample_rate)
                target_spec = [stft(raw, sample_rate)[0] for raw in
                               [self.triplets[i * batch + j][1],
                                self.triplets[i * batch + j][2]]]
                if normalize:
                    source_spec = source_spec / AVER_MAG[:, None]
                    target_spec = [spec / AVER_MAG[:, None]
                                   for spec in target_spec]

                # istft during inference
                sources_len.append(source_len)

                sources.append(source_spec)
                targets.append(target_spec)

                if i == n_iters - 1 and j == leftover - 1:
                    break

            seqlen_list = [source.shape[1] for source in sources]
            if sortseq:
                zip_list = sorted(zip(sources, targets, seqlen_list, sources_len),
                                  key=lambda x: x[2], reverse=True)
                sources, targets, seqlen_list, sources_len = zip(*zip_list)

            if bptt_len is None:
                sources_arr = np.zeros((len(seqlen_list),    feature_size, max(seqlen_list)))
                targets_arr = np.zeros((len(seqlen_list), 2, feature_size, max(seqlen_list)))
                masks_arr   = np.zeros((len(seqlen_list),                  max(seqlen_list)))
            else:
                nb_bptt     = max(seqlen_list) // bptt_len
                if max(seqlen_list) % bptt_len != 0:
                    nb_bptt += 1
                sources_arr = np.zeros((len(seqlen_list),    feature_size, nb_bptt * bptt_len))
                targets_arr = np.zeros((len(seqlen_list), 2, feature_size, nb_bptt * bptt_len))
                masks_arr   = np.zeros((len(seqlen_list),                  nb_bptt * bptt_len))

            for k in range(len(seqlen_list)):
                sources_arr[k, :, :seqlen_list[k]] = np.vstack((sources[k].real,
                                                                sources[k].imag))
                targets_arr[k, :, :, :seqlen_list[k]] = np.concatenate([
                    np.vstack((targets[k][0].real,
                              targets[k][0].imag))[None],
                    np.vstack((targets[k][1].real,
                              targets[k][1].imag))[None]], axis=0)

                masks_arr[k, :seqlen_list[k]] = 1.
            if sortseq:
                yield(sources_arr, targets_arr, masks_arr,
                      np.array(seqlen_list), np.array(sources_len))
            else:
                yield(sources_arr, targets_arr, masks_arr, np.array(sources_len))
