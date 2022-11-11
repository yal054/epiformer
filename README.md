# Epiformer

This package is under actively updating

### Introduction

To further understand how risk variants contribute to the function of regulatory elements, we used deep learning (DL) models to predict chromatin accessibility from DNA sequences. The deep learning model architecture was inspired by [Enformer](https://www.nature.com/articles/s41592-021-01252-x), which adapts attention-based architecture, Transformer, that could better capture syntactic forms (e.g., order and combination of words in a sentence) and outperforms most existing models in natural language processing tasks. 

The deep learning model, called ***Epiformer***, takes both one-hot-encoded DNA sequences (A = [1, 0, 0, 0], C = [0, 1, 0, 0], G = [0, 0, 1, 0], T = [0, 0, 0, 1]) and conservation scores (phastCons, range from 0 to 1) calculated from multiple species alignment as inputs, and predicts the chromatin accessibility on ~100 kb genomic region at 128 bp resolution. 

### Model archiecture

***Epiformer*** is written based on one open-source machine learning framework, [PyTorch](https://pytorch.org), and contains three major parts: 
- In total of 6 residual convolutional blocks with max pooling (Conv Tower). The kernel size (4,15) is used for the 1st residual convolutional block, and kernel size (4, 5) is used for the rest of residual convolutional block with padding, in order to extract informative sequence features at lower and higher resolution, respectively; The batch normalization and Gaussian Error Linear Unit (GELU) activation function are inserted between each convolutional layer. The convolutional blocks with pooling reduce the spatial dimension from 98,304 bp to 768 so that each sequence position vector represents a bin of 128  bp.
- Transformer blocks including 4 multi-head attention layers is followed by the Conv Tower, to capture the long-range combinations and orders of sequence features. To inject positional information, we add relative positional encodings. Dropout rate of 0.4 is used for attention layer to avoid potential overfitting. The [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf), a GELU variant, is used as activation function. 
- A pointwise convolution block is included to aggregate the weights from the last transformer layer and eventually output the predicted signals. The GELU and SwiGLU activation functions are used with dropout rate of 0.1.

![ model architecture ](/img/epiformer.png)

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
j=mean
k=phastCons
l=100kSeg
seqlen=98304 # 49152 # 196608
b=128 # 64 # 256

for i in human mouse; # filter blacklist
do echo $i;
mkdir datasets/${i}.${j}${l}${b}.multitype.${k}
python bin/selftrans.generate_multi_input.py \
                                     --target datasets/${i}.targets.txt \
                                     --seqcons genome/${i}.${k}.bw \
                                     --fa genome/${i}.fa \
                                     --gsize genome/${i}.chrom.sizes.lite \
                                     --blacklist genome/${i}.blacklist.bed.gz \
                                     --seqlen ${seqlen} \
                                     --nchunk 5000 \
                                     --binsize ${b} \
                                     -o datasets/${i}.${j}${l}${b}.multitype.${k}/${i}.${j}${l}${b}.multitype.${k}
done;
```

#### Step 2: Train model

Before training, please modify the hyperparameters in script ***train_signals.py***. The current models were trained on 4 NVIDIA GeForce RTX 3090 (24GB) GPUs.

```
celltype_list=("HIP" "ASCT" "OPC" "OGC" "MGC" "VIP" "LAMP5" "PVALB" "SST" "MSN" "FOXP2" "ITL23" "ITL4" "ITL5" "ITL6" "ITL6_2" "CT" "L6B" "NP" "PIR" "ET")

i=4
typename=${celltype_list[${i}]}

human overall corr

for i in `seq 1 19`;
do echo $i;
typename=${celltype_list[${i}]}
echo $typename

path2dataset=datasets/human.mean100kSeg128.multitype.phastCons/
outprfx=model/human.input10k.mean100kSeg128.${typename}.4batch12accum.lr1e-4.phastCons.crop64.overallCorr.ConstantWarmup.RAdam

CUDA_VISIBLE_DEVICES=0,1,2,3 python epiformer/train_signals.py \
                                    -i  $path2dataset \
                                    --index ${i} \
                                    -o $outprfx &> $outprfx.train.log

done
```


#### Step 3: Make predictions



### Download link to pre-trained models



