#!/bin/python

import argparse

parser = argparse.ArgumentParser(description='selftrans training')
parser.add_argument('-m', '--model', type=str, dest="model", help='path to model')
parser.add_argument('-s', '--species', type=str, dest="species", help='species: human, macaque, marmoset, mouse')
parser.add_argument('-r', '--region', type=str, dest="region", help='genomic region: chr:xxxx-xxxx')
parser.add_argument('-d', '--path2dir', type=str, dest="path2dir", help='path to working directory')
#parser.add_argument('--target', type=str, dest="target", help='input targets: list of bigwig and peaks')
#parser.add_argument('--seqcons', type=str, dest="seqcons", default=None, help='input sequence cons: phastCons or PhyloP')
#parser.add_argument('--fa', type=str, dest="fa", help='input genome sequence in fasta format')
parser.add_argument('--index', type=int, default=12, dest="index", help='celltype index')
parser.add_argument('--seqlen', type=int, default=98304, dest="seqlen", help='sequence length')
parser.add_argument('--binsize', type=int, default=256, dest="binsize", help='bin size')
parser.add_argument('-o', '--outprfx', type=str, dest="outprfx", help='output prefix')

args = parser.parse_args()

from model import *
from func import *

import time
from time import perf_counter as pc

# Setting the seed
pl.seed_everything(2022)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# universal para
celltype_list = ["ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6b", "L6_CT", "L6_IT_CAR3", "L6_IT", "LAMP5", "MGC", "OGC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC"]
#celltype_labels = "OPC"
#index = 14

def run():
    start_time = pc()
    """ init input files """
    path2model = args.model
    region4pred = args.region
    species = args.species
    path2dir = args.path2dir
    index = args.index
    celltype_labels = celltype_list[index]
    path2trg = ''.join([path2dir, "/datasets/", species, ".targets.txt"])
    path2seqcons = ''.join([path2dir, "/genome/", species, ".phastCons.bw"])
    if os.path.isfile(path2seqcons) is not True:
        path2seqcons = None
    path2fasta = ''.join([path2dir, "/genome/", species, ".fa"])
    seq_len = args.seqlen 
    bin_size = args.binsize
    outprfx = args.outprfx

    """ load checkpoint """
    checkpoint = torch.load(path2model)
    print("current model corr: ", checkpoint['current_model_corr'])
    print("current valid corr: ", checkpoint['current_valid_corr'])
    print("current model loss: ", checkpoint['current_model_loss'])
    print("current valid loss: ", checkpoint['current_valid_loss'])
#    print("param: ", checkpoint['param']) 

    """ extract para """
    # Model hyperparameters
    seq_len = checkpoint['param']['seq_len']
    bin_size = checkpoint['param']['bin_size']
    num_cnn_layer = checkpoint['param']['num_cnn_layer']
    max_len = checkpoint['param']['max_len']
    add_positional_encoding = checkpoint['param']['add_positional_encoding']
    embed_size = checkpoint['param']['embed_size']
    num_heads = checkpoint['param']['num_heads']
    att_dropout = checkpoint['param']['att_dropout']
    dim_feedforward = checkpoint['param']['dim_feedforward']
    enc_dropout = checkpoint['param']['enc_dropout']
    batch_first = checkpoint['param']['batch_first']
    num_encoder_layers = checkpoint['param']['num_encoder_layers']
    crop_size = checkpoint['param']['crop_size']
    ptw_dropout = checkpoint['param']['ptw_dropout']
    multiout_dim = checkpoint['param']['multiout_dim']
    if_cons = checkpoint['param']['if_cons']
    if_augment = checkpoint['param']['if_augment']
    penalty = checkpoint['param']['penalty']

    """ build model """
    model = SelfTrans_signalOUT(
                  seq_len,
                  bin_size,
                  num_cnn_layer,
                  max_len,
                  add_positional_encoding,
                  embed_size,
                  num_heads,
                  att_dropout,
                  dim_feedforward,
                  enc_dropout,
                  batch_first,
                  num_encoder_layers,
                  crop_size,
                  ptw_dropout,
                  multiout_dim
                 ) 
    load_checkpoint(checkpoint, model) # load state_dict to model
    model = model.cpu()    

    """ pred labels """
    data = extract_data(region4pred, 
                 path2trg, 
                 path2fasta,
                 path2seqcons,
                 seq_len,
                 bin_size
                )
    inputs, seqcons, targets = data['seq'], data['seqcons'], data['signal']
    inputs, seqcons, targets = torch.tensor(inputs), torch.tensor(seqcons), torch.tensor(targets)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.float()
    seqcons = seqcons.unsqueeze(0)
    seqcons = seqcons.float()
    targets = targets.unsqueeze(0)

    targets = targets[:, index, :].unsqueeze(1)

    """ prediction """
    if if_cons:
        inputs = inputs * torch.exp(seqcons)
    
    if crop_size > 0:
        crop_seq_size = targets.shape[2] - (crop_size * 2)
        targets = center_crop(targets, targets.shape[1], crop_seq_size)
        
    # testing
    outs = model.forward(inputs)

    # calculate statistics
    #corr = metric(torch.flatten(outs), start_dim=1), 
    #              torch.flatten(targets, start_dim=1)) # pcc flatten
    corr = metric(torch.flatten(torch.log2(outs+1), start_dim=1), 
                  torch.flatten(torch.log2(targets+1), start_dim=1)) # log2 pcc flatten
    # corr = metric(outs, targets)

    # output
    corr = float(np.mean(corr.cpu().detach().numpy()))

    print('Predicting...')
    print('Model_Corr: {:.4f}'.format(corr))

    """ plotting """
    plot_targets = targets.squeeze(0).detach().numpy()
    plot_outs = outs.squeeze(0).detach().numpy()

    fig = plt.figure(figsize=(6, 2))
    plotone_pred_rawsignal(plot_targets, plot_outs, celltype_labels)
    plt.savefig(outprfx + '.' + species + '.' + region4pred + '.' + 'pred_rawsignals.pdf')

    fig = plt.figure(figsize=(6, 2))
    plotone_pred_log2signal(plot_targets, plot_outs, celltype_labels)
    plt.savefig(outprfx + '.' + species + '.' + region4pred + '.' + 'pred_log2signals.pdf')

    """ output """
    outfile = '.'.join([outprfx, species, region4pred, "true_signals.tsv"])
    np.savetxt(outfile, plot_targets)

    outfile = '.'.join([outprfx, species, region4pred, "pred_signals.tsv"])
    np.savetxt(outfile, plot_outs)

    end_time = pc()
    time_elapsed = end_time - start_time
    print('-' * 10)
    print('running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    """plot predicted signals in region"""
    run()

