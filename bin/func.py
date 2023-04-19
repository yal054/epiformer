#!/bin/python

import os
import glob
import random

# pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset, ConcatDataset
from torchvision import transforms
from torchvision import datasets

## tqdm for loading bars
from tqdm import tqdm

# PyTorch Lightning
import pytorch_lightning as pl

# hdf5 
import h5py
import hdf5plugin

# computing
import math
import numpy as np
import pandas as pd
from audtorch.metrics.functional import pearsonr
from audtorch.metrics import PearsonR
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# load data
import pyBigWig
from pybedtools import BedTool
import pysam
from scipy.sparse import csr_matrix

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import subprocess
from operator import attrgetter


##############################
### load and manage dataset

class HDF5Dataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 preprocess=False,
                 if_log2=False,
                 which = None
                ):
        self.h5_path = h5_path
        self.hf = h5py.File(h5_path, 'r')
        self.length = len(self.hf['inputs/seq'])
        self.acgt2num = {'A': 0,
                         'C': 1,
                         'G': 2,
                         'T': 3} # convert to one-hot
        self.preprocess = preprocess
        self.if_log2 = if_log2
        self.which = which
        
        if self.preprocess:
            self.data = [self.getitem_fromraw(index) for index in tqdm(range(self.__len__()))]

    def checkseq(self, seq):
        legal = ('n' not in seq) and ('N' not in seq)
        return legal

    def seq2onehot(self, seq):
        seq = seq.decode()
        #legal = self.checkseq(seq)
        seq = seq.upper() # uppercase
        h = 4
        w = len(seq)
        mat = np.zeros((h, w), dtype=int)  # True or false in mat

        for i in range(w):
            if seq[i] != 'N':
                mat[self.acgt2num[seq[i]], i] = 1.
        return mat

    def getitem_fromraw(self, index):
        data_obj = {}
        
        seq = self.hf['inputs/seq'][index]
        seq = np.float32(self.seq2onehot(seq))
        data_obj["seq"] = seq
        
        seqcons = np.float32(self.hf['inputs/seqcons'][index])
        seqcons = np.expand_dims(seqcons, axis=0)
        data_obj["seqcons"] = seqcons
        
        signal = self.hf['targets/value'][index]
        signal = np.float32(signal)
        if self.if_log2:
            signal = np.log2(signal, out=np.copy(signal), where=(signal>1))
        #signal = np.float32(np.log2(signal+1))
        
        if self.which:
            data_obj["signal"] = np.expand_dims(signal[self.which, :], axis=0)
        else:
            data_obj["signal"] = signal
        
        e = "targets/label" in self.hf
        if e:
            label = self.hf['targets/label'][index]
        
            if self.which:
                data_obj["label"] = np.expand_dims(label[self.which, :], axis=0)
            else:
                data_obj["label"] = label

        return data_obj

    def __getitem__(self, index): #to enable indexing
        if not self.preprocess:
            return self.getitem_fromraw(index)
        else:
            return self.data[index]

    def __len__(self):
        return self.length

 
# get file list
def get_filepaths(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex)) 
    
# get file list
def get_filepaths(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))


# augmentation
def seq_revc(seq):
    """reverse complement a one hot encoded DNA sequence."""
    seq_shape = seq.shape
    index = torch.tensor([3, 2, 1, 0])
    indices = torch.transpose(index.repeat(seq_shape[0], seq_shape[2], 1), 1, 2)
    
    rcseq = torch.gather(seq, dim=1, index=indices)
    rcseq = torch.flip(rcseq, dims=[2])
    return rcseq

def seq_shift(seq, shift_size, pad_value=0.25):
    
    """Shift a sequence left or right by shift_amount.
    Args:
      seq: a [batch_size, sequence_code, sequence_length] sequence to shift
      shift_size: the number of nucleotide to shift (tf.int32 or int), 
                  negative value will shift to left,
                  positve value will shift to right
      pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    
    pad = pad_value * torch.ones(seq.shape[0], seq.shape[1], np.abs(shift_size))

    def _shift_right(_seq):
        sliced_seq = _seq[:, :, shift_size:]
        return torch.cat((sliced_seq, pad), axis=2)

    def _shift_left(_seq):
        sliced_seq = _seq[:, :, :shift_size]
        return torch.cat((pad, sliced_seq), axis=2)
    
    if shift_size > 0:
        output = _shift_right(seq)
    else:
        output = _shift_left(seq)

    return output


def trg_rev(trg):
    """Reverse the targets, signals or labels.
    Args:
      trg: a [batch_size, target_depth, target_length] target to reverse
    """
    output = torch.flip(trg, dims=[2])
    return output

def augment_stochastic(data_obj, augment_rc=False, shift_size=[]):
    """Apply stochastic augmentations,
    Args:
      data_obj: dict with keys 'seq', 'seqcons', 'signal', and 'label'
      augment_rc: Boolean for whether to apply reverse complement augmentation.
      shift_size: list of int offsets to sample shift augmentations:
                  negative value will shift to left,
                  positve value will shift to right.
    Returns:
      data_obj_aug: augmented data
    """
    data_obj_aug = {}
    
    if augment_rc:
        data_obj_aug['seq'] = seq_revc(data_obj['seq'])
        data_obj_aug['seqcons'] = trg_rev(data_obj['seqcons'])
        data_obj_aug['signal'] = trg_rev(data_obj['signal'])
        #data_obj_aug['label'] = trg_rev(data_obj['label'])
    else:
        data_obj_aug = data_obj
    
    if shift_size:
        data_obj_aug['seq'] = seq_shift(data_obj_aug['seq'], 
                                        shift_size, 
                                        pad_value=0.25)

    return data_obj_aug


####################
#### training
class CosineRestartsWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup, max_iters, cycles=1.):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.cycles = cycles
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup:
            return float(epoch) / float(max(1, self.warmup))
        # progress after warmup
        progress = float(epoch - self.warmup) / float(max(1, self.max_num_iters - self.warmup))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
    
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1. over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = max(0.0, float(self.max_iters - epoch) / float(max(1, self.max_iters - self.warmup)))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ConstantWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1. over `warmup_steps` training steps.
        Keeps learning rate schedule equal to target lr after warmup_steps.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 1.
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def cal_roc(test_y, predict_y, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def cal_prc(test_y, predict_y, pos_label=1):
    prec, recall, thresholds = precision_recall_curve(test_y, predict_y, pos_label=pos_label)
    prc_auc = auc(recall, prec)
    return prec, recall, prc_auc

def save_checkpoint(state, filename):
    print("=> saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


#####################
#### plotting

def plot_pred_log2signal(trgs, preds, labels, width=18, height=9, dpi=100):
    num_y = len(labels)
    num_x = trgs.shape[1]
    
    x = [*range(0, num_x, 1)]
    
    coordinate = []
    signal = []
    group = []
    track = []
    
    for i in range(num_y):
        coordinate.append(x) # for trg
        coordinate.append(x) # for pred
        signal.append(np.log2(trgs[i,:]+1))
        signal.append(np.log2(preds[i,:]+1))
        group.append(["target"] * num_x)
        group.append(["predict"] * num_x)
        track.append([labels[i]] * num_x * 2)

    d = {'coordinate': np.concatenate(coordinate), 
         'signal': np.concatenate(signal), 
         'group' : np.concatenate(group), 
         'track' : np.concatenate(track)}
    df = pd.DataFrame(data=d)
    
    plt.figure(figsize=(width, height))
    sns.set_theme(style="ticks", font_scale=2)
    sns.relplot(x="coordinate", y="signal", hue="group",
            row="track", height=2, aspect=5, 
            kind="line", estimator=None, data=df);    
    plt.show()


def plot_pred_normsignal(trgs, preds, types, width=12, height=6, dpi=100):
    num_y = len(types)
    num_x = trgs.shape[1]
    
    x = [*range(0, num_x, 1)]
    
    coordinate = []
    signal = []
    group = []
    track = []
    
    for i in range(num_y):
        coordinate.append(x) # for trg
        coordinate.append(x) # for pred
        #signal.append(np.log2(trgs[i,:]+1)) # log2
        #signal.append(np.log2(preds[i,:]+1)) # log2
        signal.append(trgs[i,:]/np.max(trgs[i,:])) # norm
        signal.append(preds[i,:]/np.max(preds[i,:])) # norm
        #signal.append(trgs[i,:])
        #signal.append(preds[i,:])
        group.append(["target"] * num_x)
        group.append(["predict"] * num_x)
        track.append([types[i]] * num_x * 2)

    d = {'coordinate': np.concatenate(coordinate), 
         'signal': np.concatenate(signal), 
         'group' : np.concatenate(group), 
         'track' : np.concatenate(track)}
    df = pd.DataFrame(data=d)
    
    plt.figure(figsize=(width, height))
    sns.set_theme(style="ticks", font_scale=2)
    sns.relplot(x="coordinate", y="signal", hue="group",
            row="track", height=2, aspect=5, 
            kind="line", estimator=None, data=df);    
    plt.show()

def plot_pred_rawsignal(trgs, preds, types, width=12, height=6, dpi=100):
    num_y = len(types)
    num_x = trgs.shape[1]
    
    x = [*range(0, num_x, 1)]
    
    coordinate = []
    signal = []
    group = []
    track = []
    
    for i in range(num_y):
        coordinate.append(x) # for trg
        coordinate.append(x) # for pred
        #signal.append(np.log2(trgs[i,:]+1)) # log2
        #signal.append(np.log2(preds[i,:]+1)) # log2
        #signal.append(trgs[i,:]/np.max(trgs[i,:])) # norm
        #signal.append(preds[i,:]/np.max(preds[i,:])) # norm
        signal.append(trgs[i,:])
        signal.append(preds[i,:])
        group.append(["target"] * num_x)
        group.append(["predict"] * num_x)
        track.append([types[i]] * num_x * 2)

    d = {'coordinate': np.concatenate(coordinate), 
         'signal': np.concatenate(signal), 
         'group' : np.concatenate(group), 
         'track' : np.concatenate(track)}
    df = pd.DataFrame(data=d)
    
    plt.figure(figsize=(width, height))
    sns.set_theme(style="ticks", font_scale=2)
    sns.relplot(x="coordinate", y="signal", hue="group",
            row="track", height=2, aspect=5, 
            kind="line", estimator=None, data=df);    
    plt.show()


def plot_pred_label(trgs, preds, types, cutoff=0.9, width=8, height=12, dpi=100):
    
    fig, axs = plt.subplots(nrows=3, figsize=(width, height))
    
    sns.set_theme(style="ticks", font_scale=1)
    
    sns.heatmap(trgs, xticklabels=False, 
                yticklabels=types, 
                ax=axs[0]).set_title('true label')
    
    sns.heatmap(preds, xticklabels=False, 
                yticklabels=types, ax=axs[1], 
                ).set_title('pred probability')
    
    sns.heatmap(preds>cutoff, yticklabels=types, 
                ax=axs[2]).set_title('pred label')
    
    plt.show()



def plotone_pred_rawsignal(trgs, preds, labels, width=18, height=2, dpi=100):
    num_y = len(labels)
    num_x = trgs.shape[1]
    
    x = [*range(0, num_x, 1)]
    
    coordinate = []
    signal = []
    group = []
    track = []
    
    coordinate.append(x) # for trg
    coordinate.append(x) # for pred
    signal.append(trgs[0,])
    signal.append(preds[0,])
    group.append(["target"] * num_x)
    group.append(["predict"] * num_x)
    track.append([labels] * num_x * 2)

    d = {'coordinate': np.concatenate(coordinate), 
         'signal': np.concatenate(signal), 
         'group' : np.concatenate(group), 
         'track' : np.concatenate(track)}
    df = pd.DataFrame(data=d)
    
    plt.figure(figsize=(width, height))
    #sns.set_theme(style="ticks", font_scale=2)
    sns.relplot(x="coordinate", y="signal", hue="group",
            row="track", height=2, aspect=5, 
            kind="line", estimator=None, data=df);    
    plt.show()


def plotone_pred_log2signal(trgs, preds, labels, width=18, height=2, dpi=100):
    num_y = len(labels)
    num_x = trgs.shape[1]
    
    x = [*range(0, num_x, 1)]
    
    coordinate = []
    signal = []
    group = []
    track = []
    
    coordinate.append(x) # for trg
    coordinate.append(x) # for pred
    signal.append(np.log2(trgs[0,:]+1))
    signal.append(np.log2(preds[0,:]+1))
    group.append(["target"] * num_x)
    group.append(["predict"] * num_x)
    track.append([labels] * num_x * 2)

    d = {'coordinate': np.concatenate(coordinate), 
         'signal': np.concatenate(signal), 
         'group' : np.concatenate(group), 
         'track' : np.concatenate(track)}
    df = pd.DataFrame(data=d)
    
    plt.figure(figsize=(width, height))
    #sns.set_theme(style="ticks", font_scale=2)
    sns.relplot(x="coordinate", y="signal", hue="group",
            row="track", height=2, aspect=5, 
            kind="line", estimator=None, data=df);    
    plt.show()


##########################
#### predit in region

def seq2onehot(seq):
    acgt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3} 
    seq = seq.upper() # uppercase
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=int)  # True or false in mat

    for i in range(w):
        if seq[i] != 'N':
            mat[acgt2num[seq[i]], i] = 1.
    return mat
    

def sig2sum(sig, bin_size = 128, sum_method = 'sum', norm = False, logtrans = False):
    """normalize signals"""
    if norm:
        sig = sig/np.sum(sig)*1000
        sig = sig.tolist()
    if logtrans:
        sig = [log(i+1) for i in sig]
    """ summarize signal to bins"""
    sig_length = len(sig)
    bin_num = sig_length // bin_size
    
    chunk_begin, chunk_end = 0, bin_size
    splitted_vals = []
    for _ in range(bin_num):
        b = sig[chunk_begin:chunk_end]
        if sum_method == 'mean':
            val = np.mean(b)
        elif sum_method == 'median':
            val = np.median(b)
        elif sum_method == 'sum':
            val = np.sum(b)
        else:
            print("please change to supported method: mean or median")
        splitted_vals.append(val)
        chunk_begin, chunk_end = chunk_begin + bin_size, chunk_end + bin_size
    if chunk_begin < sig_length:
        b = sig[chunk_begin:sig_length]
        if sum_method == 'mean':
            val = np.mean(b)
        elif sum_method == 'median':
            val = np.median(b)
        elif sum_method == 'sum':
            val = np.sum(b)
        else:
            print("please change to supported method: mean or median")
        splitted_vals.append(val)
        
    #splitted_vals = torch.FloatTensor(splitted_vals)
    return splitted_vals

def get_bin_index(start, bin_size, peak_start, peak_end):
    start_bin = (peak_start - start) // bin_size
    #if peak_start - ( start + (start_bin * bin_size) ) > bin_size // 2: # 50% overlap
    #    start_bin = start_bin + 1
    
    end_bin = (peak_end - start) // bin_size 
    #if peak_end - ( start + (end_bin * bin_size) ) > bin_size // 2: # 50% overlap
    #    end_bin = end_bin - 1
    end_bin = end_bin + 1
    if start_bin == end_bin:
        start_bin = start_bin - 1
    return(start_bin, end_bin)


def extract_data(region4pred, 
                 path2trg, 
                 path2fasta,
                 path2seqcons,
                 seq_len,
                 bin_size
                ):
    # get region for prediction
    # extend from center 
    region4pred_str = region4pred.strip().replace(':', ' ').replace('-', ' ').split()
    chrom = region4pred_str[0]
    region_len = int(region4pred_str[2]) - int(region4pred_str[1])
    region_diff = seq_len - region_len
    region_flank = int(region_diff/2)
    if region_diff % 2 == 1:
        start = int(region4pred_str[1]) - region_flank - 1
        end = int(region4pred_str[2]) + region_flank
    else:
        start = int(region4pred_str[1]) - region_flank
        end = int(region4pred_str[2]) + region_flank
    region4pred_str = ' '.join([chrom, str(start), str(end)])
    print(region4pred_str)
    
    '''read input files...'''
    trgfiles = pd.read_csv(path2trg, delimiter = "\t")
    trg_num = trgfiles.shape[0]
    identifier_list = trgfiles['identifier']
    bw_list = [pyBigWig.open(path2bw) for path2bw in trgfiles['file']]
    clip_list = trgfiles['clip']
    stat_list = trgfiles['sum_stat']
    
    if 'peak' in trgfiles.columns:
        peak_list = [BedTool(path2peak) for path2peak in trgfiles['peak']]
    if path2seqcons is not None:
        consbw = pyBigWig.open(path2seqcons) 
    
    '''extract sequence...'''
    bed = BedTool(region4pred_str, from_string=True)
    fasta = BedTool(path2fasta)
    seqs = bed.sequence(fi=fasta, tab=True, name=True)
    seq = open(seqs.seqfn, 'r').readlines()[0].rstrip().split(None, 1)[1]
    inputs = seq2onehot(seq)
    
    '''extract values from seqcons file'''
    if path2seqcons is not None: 
        incons = consbw.values(chrom, start, end)
        incons = np.nan_to_num(incons)
    else:
        incons = np.zeros(len(seq))
    incons = np.expand_dims(incons, axis=0)
    
    if 'peak' in trgfiles.columns:
        '''intersect with peaks'''
        path2itsc = "tmp.itsc.bed"
        cmd = "intersectBed -loj -a "+ bed.fn + " -b " + ' '.join(trgfiles['peak']) + " -nonamecheck " + " > " + path2itsc # with index in column 4
        subprocess.call(cmd, shell=True)
        
        '''extract peak overlapped with bins'''
        nbin = seq_len // bin_size
        nrow = len(identifier_list)
        ncol = nbin
        lmx = csr_matrix((nrow, ncol), dtype = np.float32).toarray()
        with open(path2itsc, "r") as infile:
            for line in infile:
                line = line.rstrip().split()
                sig_index = line[3]
                if sig_index != '.':
                    sig_index = int(sig_index) - 1 
                    peak_start = int(line[5])
                    peak_end = int(line[6])
                    start_bin, end_bin = get_bin_index(start, bin_size, peak_start, peak_end)
                    lmx[sig_index, start_bin:end_bin] = 1
                else:
                    pass
    
    '''extract values from each bw file'''
    vals_list = [ bw_list[i].values(chrom, start, end) for i in range(trg_num) ]
    outs_list = [ sig2sum(vals_list[i], bin_size = bin_size, sum_method = stat_list[i]) for i in range(trg_num) ]
    clip_outs_list = [ np.clip(outs_list[i], 0, clip_list[i]) for i in range(trg_num) ]
    out_mx = np.matrix(clip_outs_list)
    
    incons
    data_obj = {}
    data_obj["seq"] = inputs
    data_obj["seqcons"] = incons
    data_obj["signal"] = out_mx
    if 'peak' in trgfiles.columns:
        data_obj["label"] = lmx
    
    return data_obj


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)

    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])

