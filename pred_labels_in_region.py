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
parser.add_argument('--seqlen', type=int, default=98304, dest="seqlen", help='sequence length')
parser.add_argument('--binsize', type=int, default=256, dest="binsize", help='bin size')
parser.add_argument('--cutoff', type=float, default=0.7, dest="cutoff", help='cutoff')
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
celltype_labels = ["ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6b", "L6_CT", "L6_IT_CAR3", "L6_IT", "LAMP5", "MGC", "OGC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC"]
#bce_weights = torch.FloatTensor([1.0, 10.0]) 

def run():
    start_time = pc()
    """ init input files """
    path2model = args.model
    region4pred = args.region
    species = args.species
    path2dir = args.path2dir
    path2trg = ''.join([path2dir, "/datasets/", species, ".targets.txt"])
    path2seqcons = ''.join([path2dir, "/genome/", species, ".phastCons.bw"])
    if os.path.isfile(path2seqcons) is not True:
        path2seqcons = None
    path2fasta = ''.join([path2dir, "/genome/", species, ".fa"])
    seq_len = args.seqlen 
    bin_size = args.binsize
    cutoff = args.cutoff
    outprfx = args.outprfx

    """ load checkpoint """
    checkpoint = torch.load(path2model)
    print("current model auroc: ", checkpoint['current_model_auroc'])
    print("current valid auroc: ", checkpoint['current_valid_auroc'])
    print("current model auprc: ", checkpoint['current_model_auprc'])
    print("current valid auprc: ", checkpoint['current_valid_auprc'])
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
    crop_size = 0 #checkpoint['param']['crop_size']
    cls_dropout = checkpoint['param']['cls_dropout']
    multiout_dim = 20 #checkpoint['param']['multiout_dim']
    if_cons = True # checkpoint['param']['if_cons']

    """ build model """
    model = SelfTrans_labelOUT(
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
                  cls_dropout,
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
    inputs, seqcons, labels = data['seq'], data['seqcons'], data['label']
    inputs, seqcons, labels = torch.tensor(inputs), torch.tensor(seqcons), torch.tensor(labels)
    inputs = inputs.unsqueeze(0)
    seqcons = seqcons.unsqueeze(0)
    seqcons = seqcons.float()
    labels = labels.unsqueeze(1)

    """ prediction """
    if if_cons:
        inputs = inputs * torch.exp(seqcons)
    
    if crop_size > 0:
        crop_seq_size = labels.shape[2] - ( crop_size * 2 )
        labels = center_crop(labels, labels.shape[1], crop_seq_size)
        
    classes = model.forward(inputs)

    acc = metrics.accuracy_score(torch.flatten(labels).detach().cpu().numpy(), 
                             torch.flatten(classes).detach().cpu().numpy() > 0.5)

    epoch_y = torch.flatten(labels).detach().cpu().numpy()
    epoch_pred = torch.flatten(classes).detach().cpu().numpy()
    fpr, tpr, roc_auc = cal_roc(epoch_y, epoch_pred, pos_label=1)
    prec, recall, prc_auc = cal_prc(epoch_y, epoch_pred, pos_label=1)

    print('Predicting on regions')
    print('Model_Acc: {:.4f} Model_AUROC: {:.4f} Model_AUPRC: {:.4f}'.format(
            acc, roc_auc, prc_auc))

    """ plotting """
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig(outprfx + '.' + species + '.' + region4pred + '.' + 'pred_roc.pdf')
    PrecisionRecallDisplay(recall=recall, precision=prec).plot()
    plt.savefig(outprfx + '.' + species + '.' + region4pred + '.' + 'pred_prc.pdf')

    plot_labels = labels.squeeze(1).detach().numpy()
    plot_classes = classes.squeeze(0).detach().numpy()
    fig = plt.figure(figsize=(6, 12))
    plot_pred_label(plot_labels, plot_classes, celltype_labels, cutoff=cutoff)
    plt.savefig(outprfx + '.' + species + '.' + region4pred + '.' + 'pred_labels.pdf')

    """ output """
    outfile = '.'.join([outprfx, species, region4pred, "true_labels.tsv"])
    np.savetxt(outfile, plot_labels)

    outfile = '.'.join([outprfx, species, region4pred, "pred_prob.tsv"])
    np.savetxt(outfile, plot_classes)

    pred_labels = np.zeros(plot_labels.shape)
    pred_labels[plot_classes > cutoff] = 1
    outfile = '.'.join([outprfx, species, region4pred, "pred_labels.tsv"])
    np.savetxt(outfile, pred_labels)


    end_time = pc()
    time_elapsed = end_time - start_time
    print('-' * 10)
    print('running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    """plot predicted labels in region"""
    run()

