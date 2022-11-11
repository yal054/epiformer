# epiformer

This package will be actively updated

### Introduction

To further understand how risk variants contribute to the function of regulatory elements, we used deep learning (DL) models to predict chromatin accessibility from DNA sequences. The deep learning model architecture was inspired by [Enformer](https://www.nature.com/articles/s41592-021-01252-x), which adapts attention-based architecture, Transformer, that could better capture syntactic forms (e.g., order and combination of words in a sentence) and outperforms most existing models in natural language processing tasks. 

The deep learning model, called ***Epiformer***, takes both one-hot-encoded DNA sequences (A = [1, 0, 0, 0], C = [0, 1, 0, 0], G = [0, 0, 1, 0], T = [0, 0, 0, 1]) and conservation scores (phastCons, range from 0 to 1) calculated from multiple species alignment as inputs, and predicts the chromatin accessibility on ~100 kb genomic region at 128 bp resolution. 

### Model archiecture

***Epiformer*** is written based on one open-source machine learning framework, [PyTorch](https://pytorch.org), and contains three major parts: 
- 6 residual convolutional blocks with max pooling (Conv Tower). The kernel size (4,15) is used for the 1st residual convolutional block, and kernel size (4, 5) is used for the rest of residual convolutional block with padding, in order to extract informative sequence features at lower and higher resolution, respectively; The batch normalization and Gaussian Error Linear Unit (GELU) activation function are inserted between each convolutional layer. The convolutional blocks with pooling reduce the spatial dimension from 98,304 bp to 768 so that each sequence position vector represents a bin of 128  bp.
- Transformer blocks including 4 multi-head attention layers is followed by the Conv Tower, to capture the long-range combinations and orders of sequence features. To inject positional information, we add relative positional encodings. Dropout rate of 0.4 is used for attention layer to avoid potential overfitting. The [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf), a GELU variant, is used as activation function. 
- A pointwise convolution block is included to aggregate the weights from the last transformer layer and eventually output the predicted signals. The GELU and SwiGLU activation functions are used with dropout rate of 0.1.

![ model architecture ](/img/epiformer.png)

### Usage


### Download link to pre-trained models



