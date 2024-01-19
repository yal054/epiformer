# Epiformer

11/11/2022 - This package is still actively updating!
For more details, please check out our recent paper ***A comparative atlas of single-cell chromatin accessibility in the human brain*** (2023) [Science] (https://www.science.org/doi/full/10.1126/science.adf7044) 13;382(6667):eadf7044

### Introduction

To further understand how risk variants contribute to the function of regulatory elements, we used deep learning (DL) models to predict chromatin accessibility from DNA sequences. Our model was inspired by [Enformer](https://www.nature.com/articles/s41592-021-01252-x), which adapts the attention-based Transformer to better capture syntactic forms (e.g., order and combination of words in a sentence) in order to predict gene expression. 

Here, our model utilizes a similar framework to predict epigenic information such as chromatin accessibility. Called ***Epiformer***, our model takes (1) one-hot-encoded DNA sequences (A = [1, 0, 0, 0], C = [0, 1, 0, 0], G = [0, 0, 1, 0], T = [0, 0, 0, 1]) and (2) conservation scores (phastCons, range from 0 to 1) calculated from multiple species alignment as inputs, and predicts chromatin accessibility on ~100 kb genomic region at 128 bp resolution as output. 

### Model archiecture

***Epiformer*** rests on open-source machine learning framework, [PyTorch](https://pytorch.org), and contains three major parts: 
- In total there are 6 residual convolutional blocks with max pooling (Conv Tower). We used a kernel size (4,15) for the 1st residual convolutional block, and a kernel size (4, 5) for the rest of residual convolutional block with padding. These two different kernels allow us to extract informative sequence features at lower and higher resolution, respectively. Batch normalization and Gaussian Error Linear Unit (GELU) activation function interleave each convolutional layer. The convolutional blocks with pooling reduce the spatial dimension from 98,304 bp to 768 so that each sequence position vector represents a bin of 128 bp.
- Transformer blocks including 4 multi-head attention layers is followed by the Conv Tower, which captures long-range combinations and orders of sequence features. To inject positional information, we add relative positional encodings. We used a dropout rate of 0.4 for the attention layer to avoid potential overfitting. In particular, we used [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf), a GELU variant, as activation function. 
- A pointwise convolution block is included to aggregate the weights from the last transformer layer and eventually output the predicted signals. Here we used GELU and SwiGLU activation functions with dropout rate of 0.1.

![ model architecture ](/img/epiformer.png)

### Installation and Configuration

```
conda create -n epiformer python==3.9
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c bioconda bedtools
pip install -r requirements.txt
```

### Usage

#### Step 0: Prepare targets and download
- We create few directories to keep various datasets, models, and predictions.
- Download human genome (hg38) in fasta format, chromosome size, ENCODE blacklist regions, conservation score tracks from multiple species alignment (e.g. phastCons). These files are kept under directory `./genome`.
- Generate ATAC-seq signal tracks using softwares like [deepTools](https://deeptools.readthedocs.io/en/develop/), put files under directory `./targets`
- Provide peak calls (optional) under directory `./targets`. 
- Generate one tab delimited text file ***targets.txt*** listing the path to every target file, such as:
```
index   identifier      file    clip    sum_stat        description     peak
0       HIP     path2targets/human.HIP.norm_cov.bw      128     mean    HIP     path2targets/human.HIP.bed
1       ASCT    path2targets/human.ASCT.norm_cov.bw     128     mean    ASCT    path2targets/human.ASCT.bed
2       OPC     path2targets/human.OPC.norm_cov.bw      128     mean    OPC     path2targets/human.OPC.bed
3       OGC     path2targets/human.OGC.norm_cov.bw      128     mean    OGC     path2targets/human.OGC.bed
4       MGC     path2targets/human.MGC.norm_cov.bw      128     mean    MGC     path2targets/human.MGC.bed
5       VIP     path2targets/human.VIP.norm_cov.bw      128     mean    VIP     path2targets/human.VIP.bed
6       ITL23   path2targets/human.ITL23.norm_cov.bw    128     mean    ITL23   path2targets/human.ITL23.bed
7       ET      path2targets/human.ET.norm_cov.bw       128     mean    ET      path2targets/human.ET.bed
```

#### Step 1: Generate datasets

```
python bin/generate_datasets.py \
       --target targets/human.targets.txt \
       --seqcons genome/human.phastCons.bw \
       --fa genome/human.fa \
       --gsize genome/human.chrom.sizes.lite \
       --blacklist genome/human.blacklist.bed.gz \
       --seqlen 98304 \
       --nchunk 5000 \
       --binsize 128 \
       -o datasets/human.mean100kSeg128.multitype.phastCons
```

#### Step 2: Train model

Before training, please modify the hyperparameters in script ***train_signals.py***. The current models were trained on 4 NVIDIA GeForce RTX 3090 (24GB) GPUs.

```
#--------------------------#
# Training hyperparameters #

num_epoches = 100
learning_rate = 1e-4
min_lr = 1e-6

batch_size = 8 # using 4 NVIDIA GeForce RTX 3090 GPUs
warmup_epoches = 10
max_iter_epoches = 100

accum_iter = 8 # gradient accumulation

#-----------------------#
# Model hyperparameters #
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
crop_size = 64
ptw_dropout = 0.1
multiout_dim = 1 # 20
#recycle_count = 3

if_augment = True # data augmentation
if_cons = True # to use conservation score or not
penalty = 0 # give penalty to loss function or not
```

Here, we demonstrate the model training for microglia (MGC).

```
celltype_list=("HIP" "ASCT" "OPC" "OGC" "MGC" "VIP" "ITL23" "ET")
i=4
typename=${celltype_list[${i}]}

path2dataset=datasets/
outprfx=model/${typename}

CUDA_VISIBLE_DEVICES=0,1,2,3 python bin/train_signals.py \
                                    -i  $path2dataset \
                                    --index ${i} \
                                    -o $outprfx &> $outprfx.train.log

```

The current models were trained on 4 NVIDIA GeForce RTX 3090 (24GB) GPUs. Three model will be saved from checkpoints with best correlation, best loss and from the last epoch.

#### Step 3: Make predictions

Here, we demonstrate the prediction centered at TSS of gene ***BIN1*** in microglia (MGC).

```
celltype_list=("HIP" "ASCT" "OPC" "OGC" "MGC" "VIP" "ITL23" "ET")
i=4
typename=${celltype_list[${i}]}

seq_len=98304 
bin_size=128
cell_index=${i} # MGC
species="human"
region="chr2:127068747-127150006" # centered at TSS of gene ***BIN1***

path2model=model/MGC.model-1663637197.best_loss.pth.tar
path2pred=pred/MGC

python epiformer/pred_signals.py -m $path2model \
                                 -s $species -r $region \
                                 -d ./ --index $cell_index --seqlen $seq_len --binsize ${bin_size} \
                                 -o $path2pred.${species} &> $path2pred.${species}.pred_signals.log
```

### Download link to pre-trained models

The models pre-trained in this study: [http://catlas.org/catlas_downloads/humanbrain/epiformer_data/](http://catlas.org/catlas_downloads/humanbrain/epiformer_data/).

We will keep updating these cell-type specific models.


