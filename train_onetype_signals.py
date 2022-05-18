#!/bin/python

import argparse

parser = argparse.ArgumentParser(description='selftrans training')
parser.add_argument('-i', '--indir', type=str, dest="indir", help='path to dataset')
parser.add_argument('-o', '--outprfx', type=str, dest="outprfx", help='output prefix')

args = parser.parse_args()

from model import *
from func import *

import time
from time import perf_counter as pc

# Training hyperparameters
num_epoches = 50
learning_rate = 1e-5
min_lr = 1e-6

batch_size = 4 # using 4 GPUs
warmup_epoches = 10
max_iter_epoches = 50

# Model hyperparameters
seq_len = 98304
bin_size = 128
num_cnn_layer = 6
max_len = 384 * 2
add_positional_encoding = True
embed_size = 384 * 2
num_heads = 4
att_dropout = 0.4
dim_feedforward = 2048 
enc_dropout = 0.4 
batch_first = True
num_encoder_layers = 8
crop_size = 32
ptw_dropout = 0.1
multiout_dim = 1 # 20
recycle_count = 1

if_augment = True
if_cons = True
penalty = 0

celltype_list = ["ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6b", "L6_CT", "L6_IT_CAR3", "L6_IT", "LAMP5", "MGC", "OGC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC"]
celltype_index = 12

#device_ids = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device_ids = [0, 1, 2, 3]

TRAIN_PCT = 0.8
VALID_PCT = 0.2
ncpu = 8


# Setting the seed
pl.seed_everything(2022)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

    
start_time = pc()
""" init input files """
path2indir = args.indir 
path2outprfx = args.outprfx

# load dataset
regex = "*.h5"
infile_list = sorted(get_filepaths(root_path=path2indir, file_regex=regex))
print("# of h5 file loaded: " + str(len(infile_list)))

dataset_list = [HDF5Dataset(f_path, preprocess=True, if_log2=False, which = celltype_index) for f_path in infile_list]
dataset = ConcatDataset(dataset_list)
print("# of inputs and targets lazy loaded: " + str(len(dataset)))

dataset_size = len(dataset)
train_num = int(dataset_size * TRAIN_PCT)
print("training set size: ",train_num)
val_num = dataset_size - train_num
print("validation set size: ", val_num)

train_X, val_X = random_split(dataset, [train_num, val_num])
train_loader = DataLoader(train_X, batch_size=batch_size, shuffle=True, num_workers=ncpu) #, pin_memory=True)
val_loader = DataLoader(val_X, batch_size=batch_size, shuffle=True, num_workers=ncpu) #, pin_memory=True)


#######################
### Train model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name())
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

#del(model)
torch.cuda.empty_cache()

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("available_gpus number: ", len(available_gpus))

MODEL_NAME = f"model-" + f"{int(time.time())}"
print("Initializing model: ", MODEL_NAME)

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
                 ).to(device)

# parallelized over multiple GPUs in the batch dimension.
model = nn.DataParallel(model, device_ids=device_ids)

# loss and optimizer
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

# criterion
#criterion = nn.MSELoss()
#criterion = nn.MSELoss(reduction='none')
#criterion = nn.SmoothL1Loss()
criterion = nn.PoissonNLLLoss(log_input=False)
#criterion2 = nn.BCELoss()

metric = PearsonR(reduction='mean', batch_first=True)
#metric = nn.CosineSimilarity(dim=1, eps=1e-6) # Cosine Similarity

#scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=warmup_epoches, max_iters=max_iter_epoches)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-5, verbose=True) # for corr
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=min_lr, verbose=True) # for loss

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
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


########################
# train + valid

since = time.time()

epoch_corr_list=[]
epoch_loss_list=[]

test_corr_list=[]
test_loss_list=[]

best_loss = 0.0
best_corr = 0.1

for epoch in range(num_epoches):
    print('-' * 10)
    print("Target Learning Rate: {}".format(str(learning_rate)))
    print('Epoch {}/{}'.format(epoch + 1, num_epoches))

    running_corr = 0.0
    running_loss = 0.0
    
    test_running_corr = 0.0
    test_running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        
        if i % 2 == 0 and if_augment:
            random_shift_size = random.choice(range(-3,3,1))
            data = augment_stochastic(data, augment_rc=True, shift_size=random_shift_size)
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, seqcons, targets = data['seq'], data['seqcons'], data['signal']
        inputs, seqcons, targets= inputs.to(device), seqcons.to(device), targets.to(device)
        if if_cons:
            inputs = inputs * torch.exp(seqcons)
        
        outs = model(inputs)
        
        if crop_size > 0:
            crop_seq_size = targets.shape[2] - (crop_size * 2)
            targets = center_crop(targets, targets.shape[1], crop_seq_size)

        loss = criterion(outs, targets)
        
        if penalty > 0:
            idxy = targets < 1
            weights = 1 + idxy * (penalty - 1)
            loss = loss * weights
        loss = loss.mean()
            
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss.backward() # multi output loss
        
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        # print statistics
        #corr = metric(torch.flatten(torch.transpose(outs, 0, 1), start_dim=1), 
        #              torch.flatten(torch.transpose(targets, 0, 1), start_dim=1)) # pcc flatten
        corr = metric(torch.flatten(torch.transpose(torch.log2(outs+1), 0, 1), start_dim=1), 
                      torch.flatten(torch.transpose(torch.log2(targets+1), 0, 1), start_dim=1)) # log2 pcc flatten
        corr = float(np.mean(corr.cpu().detach().numpy()))
        #print(corr)
        running_corr += corr        
        running_loss += loss.item()
        
    epoch_corr = running_corr / len(train_loader)
    epoch_loss = running_loss / len(train_loader)

    epoch_corr_list.append(epoch_corr)
    epoch_loss_list.append(epoch_loss)
    
    for test_data in val_loader:
        
        if i % 2 == 0 and if_augment:
            test_random_shift_size = random.choice(range(-3,3,1))
            test_data = augment_stochastic(test_data, augment_rc=True, shift_size=test_random_shift_size)
        
        sequences, incons, signals = test_data['seq'], test_data['seqcons'], test_data['signal']
        sequences, incons, signals = sequences.to(device), incons.to(device), signals.to(device)
        if if_cons:
            sequences = sequences * torch.exp(incons)
        
        # testing
        outputs= model.forward(sequences)
       
        if crop_size > 0:
            crop_seq_size = signals.shape[2] - (crop_size * 2)
            signals = center_crop(signals, signals.shape[1], crop_seq_size) 
        
        test_loss = criterion(outputs, signals)
        
        if penalty > 0:
            test_idxy = signals < 1
            test_weights = 1 + test_idxy * (penalty - 1)
            test_loss = test_loss * test_weights
        
        test_loss = test_loss.mean()
        
        # calculate statistics
        #test_corr = metric(torch.flatten(torch.transpose(outputs, 0, 1), start_dim=1), 
        #                   torch.flatten(torch.transpose(signals, 0, 1), start_dim=1)) # pcc flatten
        test_corr = metric(torch.flatten(torch.transpose(torch.log2(outputs+1), 0, 1), start_dim=1), 
                           torch.flatten(torch.transpose(torch.log2(signals+1), 0, 1), start_dim=1)) # log2 pcc flatten
        test_corr = float(np.mean(test_corr.cpu().detach().numpy()))
        #print(test_corr)
        test_running_corr += test_corr
        test_running_loss += test_loss.item()
    
    test_corr = test_running_corr / len(val_loader)
    test_loss = test_running_loss / len(val_loader)
    
    test_corr_list.append(test_corr)
    test_loss_list.append(test_loss)

    
    scheduler.step(epoch_loss)
    curr_lr = scheduler._last_lr
    #scheduler.step() # for warmup
    #curr_lr = scheduler.get_lr() # for warmup
    print("Current Learning Rate: {}".format(str(curr_lr)))
    
    if(test_corr > best_corr):
        best_corr = test_corr
        checkpoint = {'state_dict' : model.module.state_dict(), # Saving torch.nn.DataParallel Models
                      'optimizer': optimizer.state_dict(),
                      'current_model_loss': epoch_loss,
                      'current_model_corr': epoch_corr,
                      'current_valid_loss': test_loss,
                      'current_valid_corr': test_corr,
                      'param' : {'num_epoches' : num_epoches, 'learning_rate' : learning_rate, 'batch_size' : batch_size,
                                 'seq_len' : seq_len, 'bin_size' : bin_size, 'num_cnn_layer' : num_cnn_layer,
                                 'max_len' : max_len, 'add_positional_encoding' : add_positional_encoding, 
                                 'embed_size' : embed_size, 'num_heads' : num_heads, 'att_dropout' : att_dropout, 
                                 'dim_feedforward' : dim_feedforward, 'enc_dropout' : enc_dropout, 
                                 'batch_first' : batch_first, 'num_encoder_layers' : num_encoder_layers, 
                                 'crop_size': crop_size, 'ptw_dropout' : ptw_dropout, 'multiout_dim': multiout_dim,
                                 'if_augment' : if_augment, 'if_cons' : if_cons, 'penalty' : penalty,
                                 'recycle_count' : recycle_count
                                }
                     }
        save_checkpoint(checkpoint, path2outprfx + '.' + MODEL_NAME + '.pth.tar')
    print('Model_Loss: {:.4f} Valid_Loss: {:.4f} Model_Corr: {:.4f} Valid_Corr: {:.4f}'.format(
            epoch_loss, test_loss, epoch_corr, test_corr))

time_elapsed = time.time() - since

print('-' * 10)
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best Valid Corr: {:4f}'.format(best_corr))

###########################
### plot corr and loss

import matplotlib.pyplot as plt
x = [*range(0, len(epoch_corr_list), 1)]

fig = plt.figure(figsize=(8,4))

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1), sharex=ax1)

ax1.plot(x, epoch_corr_list, x, test_corr_list)
ax1.axis([0, len(x), 0, 1])
ax1.legend(['Model', 'Valid'])
ax1.set_title('corr: model vs. valid')
ax1.set(xlabel='epoches')

ax2.plot(x, epoch_loss_list, x, test_loss_list)
ax2.axis([0, len(x), -10, 10])
ax2.legend(['Model', 'Valid'])
ax2.set_title('loss: model vs. valid')
ax2.set(xlabel='epoches')

fig.tight_layout()
#plt.show()
plt.savefig(path2outprfx + '.' + MODEL_NAME + '.' + 'running_res.pdf')


########################
### generate output

import pandas as pd

x = [*range(0, len(epoch_corr_list), 1)]
res = pd.DataFrame(dict(epoches=x,
                        model_corr_list=epoch_corr_list,
                        model_loss_list=epoch_loss_list,
                        
                        valid_corr_list=test_corr_list,
                        valid_loss_list=test_loss_list,
                       )
                  )

OUT_NAME = path2outprfx + '.' + MODEL_NAME + ".running_res.txt"
res.to_csv(OUT_NAME, sep="\t", header=True, index=False)




