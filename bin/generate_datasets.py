#!/bin/python

import argparse

parser = argparse.ArgumentParser(description='parsing region & signal to HDF5 dataset')
parser.add_argument('--bed', type=str, dest="bed", default=None, help='input regions in bed format')
parser.add_argument('--target', type=str, dest="target", help='input targets: list of bigwig and peaks')
parser.add_argument('--seqcons', type=str, dest="seqcons", default=None, help='input sequence cons: phastCons or PhyloP')
parser.add_argument('--fa', type=str, dest="fa", help='input genome sequence in fasta format')
parser.add_argument('--gsize', type=str, dest="gsize", help='genome chrom size')
parser.add_argument('--blacklist', type=str, dest="blacklist", default=None, help='path to genome blacklist')
#parser.add_argument('--aug', action='store_true', default=False, dest="aug", help='whether do augmentation or not: shift 3 bp and flip')
parser.add_argument('--seqlen', type=int, default=98304, dest="seqlen", help='length of the input sequence')
parser.add_argument('--nchunk', type=int, default=5000, dest="nchunk", help='# of input per chunk')
parser.add_argument('--binsize', type=int, default=128, dest="binsize", help='bin size')
#parser.add_argument('--method', type=str, default='sum', dest="method", help='methods used for signal: mean, median, sum')
parser.add_argument('-o', '--outprfx', type=str, dest="outprfx", help='output directory')

args = parser.parse_args()

import os
import glob
import re

import pyBigWig
import matplotlib.pyplot as plt
from pybedtools import BedTool
import pysam
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import subprocess
from operator import attrgetter

from math import log
import h5py
import hdf5plugin

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset, ConcatDataset
import torch.nn.functional as F

import time

from time import perf_counter as pc

def run():
    """ Parsing sequence based on bed """
    start_time = pc()
    """ init input files """
    path2bed = args.bed
    path2trg = args.target
    path2seqcons = args.seqcons
    path2fasta = args.fa
    path2gsize = args.gsize
    path2blacklist = args.blacklist
    #augment = args.aug
    seq_len = args.seqlen 
    chunk_size = args.nchunk
    bin_size = args.binsize
    #method = args.method
    outprfx = args.outprfx
    peak_list = None

    #pattern_string = '|'.join(['N', 'n'])
    #pattern = re.compile(pattern_string)

    print('data generation parameters...')
    #print("augmentation:", augment)

    print('read input files...')    
    trgfiles = pd.read_csv(path2trg, delimiter = "\t")
    trg_num = trgfiles.shape[0]
    print("number of target files: ", trg_num)   
    identifier_list = trgfiles['identifier']
    bw_list = [pyBigWig.open(path2bw) for path2bw in trgfiles['file']]
    clip_list = trgfiles['clip']
    stat_list = trgfiles['sum_stat']
    if 'scale' in trgfiles.columns:
        scale_list = trgfiles['scale']
    else:
        scale_list = [1] * trg_num
    peak_list = None
    if 'peak' in trgfiles.columns:
        peak_list = [BedTool(path2peak) for path2peak in trgfiles['peak']]

    if path2seqcons is not None:
        print('read seqcons file...')
        consbw = pyBigWig.open(path2seqcons) 
    
    if path2bed is not None:
        print("using pre-defined regions...")
        bed = BedTool(path2bed)
    else:
        print("make windows...")
        bed = BedTool.window_maker(BedTool(), g=path2gsize, w = seq_len)

    print("1) filtering region shorter than desired length")
    bed = bed.filter(lambda x: len(x) == seq_len ) # 1) filter region shorter than desired length
    bed = bed.saveas() # save generator to tmp file
    if path2blacklist is not None:
        print("2) filtering for blacklist regions")
        print('read genome blacklist...')
        black_list = BedTool(path2blacklist)
        bed = bed.intersect(b=black_list, wa=True, v=True) # 2) filter for blacklist regions
    else:
        print('2) skipping for filtering blacklist regions')

    print("3) filtering genomic region with N/n in sequence...")
    fasta = BedTool(path2fasta)
    seqs = bed.sequence(fi=fasta, tab=True, name=True)

    indexN = []
    pattern_string = '|'.join(['N', 'n'])
    pattern = re.compile(pattern_string)

    with open(seqs.seqfn, 'r') as f:   
        for index, line in enumerate(f):
            m=pattern.search(line)
            if m:
                indexN.append(index)
    
    bed = [j for i, j in enumerate(bed) if i not in indexN] # 3) filter regions with N/n
    bed = BedTool(bed)
    #print(len(bed))

    print("extract sequence...")
    seqs = bed.sequence(fi=fasta, tab=True, name=True)

    path2itsc = outprfx + ".itsc.bed"
    if peak_list is not None: 
        print("intersect with peaks...")
        cmd = "intersectBed -loj -a "+ bed.fn + " -b " + ' '.join(trgfiles['peak']) + " -nonamecheck " + " > " + path2itsc # with index in column 4
        subprocess.call(cmd, shell=True)
    else:
        cmd = "awk 'BEGIN{FS=OFS=\"\t\"}{print $0, \".\"}' "+ bed.fn + " > " + path2itsc # with index in column 4
        subprocess.call(cmd, shell=True)
    
    print("generate peak dict for indexing...")
    get_interval_chrom = attrgetter("chrom")
    get_interval_start = attrgetter("start")
    get_interval_end = attrgetter("end")

    chrom_list = [get_interval_chrom(interval) for interval in bed]
    start_list = [get_interval_start(interval) for interval in bed]
    end_list = [get_interval_end(interval) for interval in bed]
    start_list = list(map(str, start_list))
    end_list = list(map(str, end_list))
    lst = [chrom_list,start_list,end_list]
    interval_list = list(map(':'.join, zip(*lst)))

    peak_dict = dict(zip(interval_list, range(len(interval_list))))

    # get inputs & output
    print("writing to files...")
    '''init'''
    data_size = len(bed)
    saved_name = ""
    lmx = None
    
    names = []
    inputs = []
    #if path2seqcons is not None:
    seqcons = []
    #else:
    #    seqcons = None
    targets = []
    labels = []    
    count = 0
    nth = 0
    totcount = 0
    
    seq_len = len(bed[0])
    nbin = seq_len // bin_size
    nrow = len(identifier_list)
    ncol = nbin
    
    totline = int(subprocess.check_output("cat " + path2itsc + " | wc -l", shell=True).strip())
    seqfile = open(seqs.seqfn, 'r').readlines()
    
    with open(path2itsc, "r") as infile:
        for line in infile:
            line = line.rstrip().split()
            
            '''extract coordinate'''
            chrom = line[0]
            start = int(line[1])
            end = int(line[2])
            name = line[0]+":"+line[1]+"-"+line[2]
            count += 1

            if name != saved_name:
                
                if lmx is not None:
                    labels.append(lmx)
                
                '''write to files'''
                if totcount>0 and totcount % chunk_size == 0:
                    print("generating dataset...")
                    print("total number of data: ", str(totcount))
                    if np.sum(labels) > 0:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx, labels=labels)
                    else:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx)
                    print("save output of chunk " + str(nth) + " to HDF5 ...")
                    ''' init '''
                    inputs = []
                    targets = []
                    #if path2seqcons is not None:
                    seqcons = []
                    #else:
                    #    seqcons = None
                    names = []
                    labels = []
                    nth += 1
                else:
                    pass
                
                '''generate label matrix'''
                lmx = csr_matrix((nrow, ncol), 
                              dtype = np.float32).toarray()
                sig_index = line[3]
                if sig_index != '.':
                    sig_index = int(sig_index) - 1 
                    peak_start = int(line[5])
                    peak_end = int(line[6])
                    start_bin, end_bin = get_bin_index(start, bin_size, peak_start, peak_end)
                    #print(start_bin)
                    #print(end_bin)
                    lmx[sig_index, start_bin:end_bin] = 1
                else:
                    pass
                
                '''get sequence'''
                index = peak_dict[ ':'.join([line[0],line[1],line[2]]) ]
                seq = seqfile[index].rstrip().split(None, 1)[0]
            
                '''extract values from each bw file'''
                vals_list = [ bw_list[i].values(chrom, start, end) for i in range(trg_num) ]

                '''extract values from seqcons file'''
                if path2seqcons is not None: 
                    incons = consbw.values(chrom, start, end)
                    incons = np.nan_to_num(incons)
                else:
                    incons = np.zeros(len(seq))
                
                '''sum and clip signals'''
                outs_list = [ sig2sum(vals_list[i], bin_size = bin_size, sum_method = stat_list[i], scale = scale_list[i]) for i in range(trg_num) ]
                clip_outs_list = [ np.clip(outs_list[i], 0, clip_list[i]) for i in range(trg_num) ]
    
                '''concat to signal matrix'''
                out_mx = np.matrix(clip_outs_list)
                #if np.sum(out_mx) == 0:
                #    print("found 0 in matrix")
                #    continue
                
                inputs.append(seq)
                #if path2seqcons is not None:
                seqcons.append(incons)
                targets.append(out_mx)
                names.append(name)
                
                saved_name = name
                totcount += 1
                
                '''write to files'''
                if totcount == data_size and count == totline:
                    labels.append(lmx)
                    print("generating dataset...")
                    print("total number of data: ", str(totcount))
                    if np.sum(labels) > 0:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx, labels=labels)
                    else:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx)
                    print("save output of chunk " + str(nth) + " to HDF5 ...")
                    print("end")
                else:
                    pass
                
            else:
                '''modify label matrix'''
                sig_index = int(line[3]) - 1
                peak_start = int(line[5])
                peak_end = int(line[6])
                start_bin, end_bin = get_bin_index(start, bin_size, peak_start, peak_end)
                #print(start_bin)
                #print(end_bin)
                lmx[sig_index, start_bin:end_bin] = 1
                
                if totcount == data_size and count == totline:
                    labels.append(lmx)
                    print("generating dataset...")
                    print("total number of data: ", str(totcount))
                    if np.sum(labels) > 0:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx, labels=labels)
                    else:
                        save2h5(inputs=inputs, seqcons=seqcons, targets=targets, names=names, index=str(nth), outprfx=outprfx)
                    print("save output of chunk " + str(nth) + " to HDF5 ...")
                    print("end") 


    end_time = pc()
    time_elapsed = end_time - start_time
    print('-' * 10)
    print('running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# convert to one-hot
acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}

def checkseq(seq):
    legal = ('n' not in seq) and ('N' not in seq)
    return legal

def seq2onehot(seq):
    legal = checkseq(seq)
    seq = seq.upper() # uppercase
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=int)  # True or false in mat
    if legal:
        for i in range(w):
            mat[acgt2num[seq[i]], i] = 1.
    else:
        for i in range(w):
            if seq[i] != 'N':
                mat[acgt2num[seq[i]], i] = 1.
            else:
                continue
    #mat = torch.from_numpy(mat)
    return mat


def sig2sum(sig, bin_size = 128, sum_method = 'sum', scale = 1, norm = False, logtrans = False):
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
        b = sig[chunk_begin:chunk_end] * scale
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
        b = sig[chunk_begin:sig_length] * scale
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

def save2h5(inputs, seqcons, targets, names, index, outprfx, labels=None):
    hf = h5py.File(outprfx + '.' + index + '.h5', 'w')
    '''save to h5 format'''
    hf.create_dataset('inputs/seq', data=inputs, **hdf5plugin.LZ4(nbytes=0)) #**hdf5plugin.Blosc(cname='blosclz', clevel=9))
    #if seqcons is not None:
    hf.create_dataset('inputs/seqcons', data=seqcons, **hdf5plugin.LZ4(nbytes=0)) #, **hdf5plugin.Blosc(cname='blosclz', clevel=9))
    hf.create_dataset('targets/value', data=targets, **hdf5plugin.LZ4(nbytes=0)) # , **hdf5plugin.Blosc(cname='blosclz', clevel=9))
    if labels is not None:
        hf.create_dataset('targets/label', data=labels, **hdf5plugin.LZ4(nbytes=0)) #, **hdf5plugin.Blosc(cname='blosclz', clevel=9))
    hf.create_dataset('segments/name', data=names, **hdf5plugin.LZ4(nbytes=0)) #, **hdf5plugin.Blosc(cname='blosclz', clevel=9))
    hf.close()

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


if __name__ == "__main__":
    """generate region & signal to HDF5 dataset"""
    run()
