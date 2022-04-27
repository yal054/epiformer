#!/bin/python

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
# Setting the seed
# PyTorch Lightning
import pytorch_lightning as pl

# hdf5 
import h5py
import hdf5plugin

# computing
import math
import numpy as np
from audtorch.metrics.functional import pearsonr
from audtorch.metrics import PearsonR
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


#########################
# criterion

#criterion = nn.MSELoss()
#criterion = nn.MSELoss(reduction='none')

#criterion = nn.PoissonNLLLoss(log_input=False)
criterion = nn.PoissonNLLLoss(log_input=False, reduction='none')

#criterion = nn.L1Loss(reduction='none')

#criterion = nn.BCELoss()

def BCELoss_class_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)
    return loss

#criterion = BCELoss_class_weighted(weights=bce_weights) # add weight

######################
## metric

metric = PearsonR(reduction='mean', batch_first=True)
#metric = nn.CosineSimilarity(dim=1, eps=1e-6) # Cosine Similarity


######################
## optimizer

#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.RAdam(model.parameters(), lr=learning_rate)


##################
### build model

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

class Residual(nn.Module):
    '''Residual Block'''
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class ConvBlock(nn.Module):
    def __init__(self, 
                 seq_len=98304, 
                 bin_size=256, 
                 embed_size=384*2,
                 num_cnn_layer = 4
                ):
        super(ConvBlock, self).__init__()
        self._model_name = "ConvBlock"
        
        C = embed_size # # of channels
        
        def conv_block(channels_init, channels, width=5, **kwargs):
            return nn.Sequential(
                nn.BatchNorm1d(num_features=channels_init),
                nn.GELU(),
                nn.Conv1d(in_channels=channels_init, out_channels=channels, kernel_size=width, **kwargs)
            )
        
        # stem layer
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=int(C // 2), kernel_size=15, dilation=1, padding=7),
            Residual(conv_block(channels_init=int(C // 2), channels=int(C // 2), width=1)),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        
        # get constant factor and channel nums
        channel_list = exponential_linspace_int(start=int(C // 2), end=C,
                                           num=num_cnn_layer+1, divisible_by=1)

        # build conv tower
        init_channel_list = channel_list[:-1]  
        trg_channel_list = channel_list[1:]
        
        module = []
        for channels_init, channels in zip(init_channel_list, trg_channel_list): 
            module.append(
                nn.Sequential(
                    conv_block(channels_init, channels, width=5, dilation=1, padding=2),
                    Residual(conv_block(channels_init=channels, channels=channels, width=1)),
                    nn.MaxPool1d(kernel_size=2, stride=None)
                )
            )

        self.tower = nn.Sequential(*module)

    def forward(self, x):
        '''stem'''
        out = self.stem(x)
        #print(out.shape)
        '''tower'''
        out = self.tower(out) # (batch, channel, seq)
        #print(out.shape)
        # switch shape: (batch, seq, feature)
        #out = out.view(out.shape[0], 312, 312).transpose(1,2)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=312):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# with relative positioning
class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 attention_probs_dropout_prob=0.1,
                 position_embedding_type=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs)

        return outputs


# build from MultiheadAttention
class EncoderBlock(nn.Module):
    def __init__(self,
                 embed_size = 312,
                 num_heads = 4,
                 att_dropout = 0.05,
                 dim_feedforward = 2048,
                 enc_dropout = 0.4,
                 batch_first = True):
        super(EncoderBlock, self).__init__()

        # self.mha = nn.MultiheadAttention(embed_size,
        #                                  num_heads,
        #                                  dropout=att_dropout,
        #                                  batch_first=batch_first)
        self.mha = SelfAttention(
            hidden_size=embed_size,
            num_attention_heads=num_heads,
            max_position_embeddings=embed_size,
            attention_probs_dropout_prob=att_dropout,
            position_embedding_type="relative_key"
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.Dropout(enc_dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_size)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(enc_dropout)

    def forward(self, x, return_attention=False):
        # Attention part
        # attn_out, attn_weights = self.mha(x, x, x)
        attn_out, attn_weights = self.mha(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        if return_attention:
            return x, attn_weights
        else:
            return x
        
class TransformerEncoder(nn.Module):

    def __init__(self,
                 embed_size = 384*2,
                 num_heads = 4,
                 att_dropout = 0.05,
                 dim_feedforward = 2048,
                 enc_dropout = 0.05,
                 batch_first = True,
                 num_encoder_layers = 6
                ):
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock(embed_size,
                                                  num_heads,
                                                  att_dropout,
                                                  dim_feedforward,
                                                  enc_dropout,
                                                  batch_first
        ) for _ in range(num_encoder_layers)])

    def forward(self, x):
        # take input shape: (batch, seq, feature)
        for l in self.layers:
            x = l(x)
        return x

    def get_attention_maps(self, x):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.mha(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps




class PointwiseSignalBlock_lnheads(nn.Module):
    def __init__(self, 
                 max_len = 384,
                 crop_size = 32,
                 ptw_in_dim = 384*2,
                 ptw_dropout = 0.05,
                 multiout_dim = 20
                ):
        super(PointwiseSignalBlock, self).__init__()
        self._model_name = "PointwiseSignalBlock"
        self.crop_size = crop_size
        self.multiout_dim = multiout_dim
        
        # conv layer
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=ptw_in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
            nn.Dropout(ptw_dropout)
        )
        
        # build output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
            nn.Linear(in_features=ptw_in_dim * 2, out_features=1),
            nn.Dropout(ptw_dropout),
            nn.Softplus()
            )
            for _ in range(multiout_dim)])
        
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch size, dimensions, seq length)
        x = self.center_crop(x) # crop
        
        # conv & transposing
        x = self.conv(x) 
        x = x.transpose(1,2) # x: (batch size, seq length, 2 * dimensions)
        
        # multiple linear heads
        outs = []
        for h in self.heads:
            o = h(x)
            o = o.squeeze(2) 
            outs.append(o)
        out = torch.stack(outs, dim=1) # concat to (batch size, multiout_dim, seq_length) 
        
        return out


class PointwiseSignalBlock_heads(nn.Module):
    def __init__(self, 
                 crop_size = 32,
                 ptw_in_dim = 384*2,
                 ptw_dropout = 0.05,
                 multiout_dim = 20
                ):
        super(PointwiseSignalBlock, self).__init__()
        self._model_name = "PointwiseSignalBlock"
        self.crop_size = crop_size
        self.multiout_dim = multiout_dim
        
        # conv heads
        self.convheads = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(num_features=ptw_in_dim), 
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
                nn.Dropout(ptw_dropout)
            )
            for _ in range(multiout_dim)])
        
        # linear heads
        self.lnheads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=ptw_in_dim * 2, out_features=1),
                nn.Dropout(ptw_dropout),
                nn.Softplus()
            )
            for _ in range(multiout_dim)])
        
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch size, dimensions, seq length)
        x = self.center_crop(x) # crop
        
        # multiple linear heads
        outs = []
        for i in range(self.multiout_dim):
            conv = self.convheads[i]
            linear = self.lnheads[i]
            o0 = conv(x)
            o0 = o0.transpose(1,2) # x: (batch size, seq length, 2 * dimensions)
            o1 = linear(o0)
            o1 = o1.squeeze(2)
            outs.append(o1)
        out = torch.stack(outs, dim=1) # concat to (batch size, multiout_dim, seq_length) 
        
        return out


class PointwiseSignalBlock_seqexp(nn.Module):
    def __init__(self, 
                 max_len = 384,
                 crop_size = 32,
                 ptw_in_dim = 384*2,
                 ptw_dropout = 0.05,
                 multiout_dim = 20
                ):
        super(PointwiseSignalBlock, self).__init__()
        self._model_name = "PointwiseSignalBlock"
        self.crop_size = crop_size
        self.multiout_dim = multiout_dim
        
        # conv layer
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=ptw_in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
            nn.Dropout(ptw_dropout)
        )
        
        # init linear & expand dim for cell type
        self.init_linear = nn.Linear(in_features=2 * ptw_in_dim, out_features=multiout_dim * 2 * ptw_in_dim)
        self.expand_linear_weights = nn.init.kaiming_uniform_(
            nn.Parameter(torch.randn(max_len - 2 * crop_size, multiout_dim, 2 * ptw_in_dim))
        )
        self.expand_linear_bias = nn.Parameter(torch.zeros(max_len - 2 * crop_size, multiout_dim))
        
        # dropout & active 
        self.active = nn.Sequential(
            nn.Dropout(ptw_dropout),
            nn.Softplus()
        )
    
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch size, dimensions, seq length)
        x = self.center_crop(x) # crop
        
        # init shape
        batch_size = x.shape[0]
        dimensions = x.shape[1]
        seq_length = x.shape[2]
        
        # conv & transposing
        x = self.conv(x) 
        x = x.transpose(1,2) # x: (batch size, seq length, 2 * dimensions)
        
        # init expanded linear 
        x = self.init_linear(x) # x: (batch size, seq length, multiout_dim * 2 * dimensions)
        x = x.view(batch_size, seq_length, self.multiout_dim, 2 * dimensions) # x: (batch size, seq length, multiout_dim, 2 * dimensions)
        # matrix multiplications 
        x = torch.einsum('ijkl,jkl->ijk', x, self.expand_linear_weights)  # x: (batch size, seq length, multiout_dim)
        x = x + self.expand_linear_bias # x: (batch size, seq length, multiout_dim)
        x = x.transpose(1,2) # x: (batch size, multiout_dim, seq length)
        
        # active 
        out = self.active(x)
        return out


class PointwiseSignalBlock_weight(nn.Module):
    def __init__(self, 
                 crop_size = 32,
                 ptw_in_dim = 384*2,
                 ptw_dropout = 0.05,
                 multiout_dim = 20
                ):
        super(PointwiseSignalBlock, self).__init__()
        self._model_name = "PointwiseSignalBlock"
        self.crop_size = crop_size
        self.multiout_dim = multiout_dim
        
        # conv layer
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=ptw_in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
            nn.Dropout(ptw_dropout)
        )
        
        # init linear & expand dim for cell type
        self.init_linear = nn.Linear(in_features=2 * ptw_in_dim, out_features=multiout_dim * 2 * ptw_in_dim)
        self.expand_linear_weights = nn.init.kaiming_uniform_(
            nn.Parameter(torch.randn(multiout_dim, 2 * ptw_in_dim))
        )
        self.expand_linear_bias = nn.Parameter(torch.zeros(multiout_dim))
        
        # dropout & active 
        self.active = nn.Sequential(
            nn.Dropout(ptw_dropout),
            nn.Softplus()
        )
    
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch size, dimensions, seq length)
        x = self.center_crop(x) # crop
        
        # init shape
        batch_size = x.shape[0]
        dimensions = x.shape[1]
        seq_length = x.shape[2]
        
        # conv & transposing
        x = self.conv(x) 
        x = x.transpose(1,2) # x: (batch size, seq length, 2 * dimensions)
        
        # init expanded linear 
        x = self.init_linear(x) # x: (batch size, seq length, multiout_dim * 2 * dimensions)
        x = x.view(batch_size, seq_length, self.multiout_dim, 2 * dimensions) # x: (batch size, seq length, multiout_dim, 2 * dimensions)
        # matrix multiplications 
        x = torch.einsum('ijkl,kl->ijk', x, self.expand_linear_weights)  # x: (batch size, seq length, multiout_dim)
        x = x + self.expand_linear_bias # x: (batch size, seq length, multiout_dim)
        x = x.transpose(1,2) # x: (batch size, multiout_dim, seq length)
        
        # active 
        out = self.active(x)
        return out



class PointwiseSignalBlock(nn.Module):
    def __init__(self, 
                 crop_size = 64,
                 ptw_in_dim=384*2,
                 ptw_dropout=0.05,
                 multiout_dim = 20
                ):
        super(PointwiseSignalBlock, self).__init__()
        self._model_name = "PointwiseSignalBlock"
        self.crop_size = crop_size
        
        # conv layer
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=ptw_in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
            nn.Dropout(ptw_dropout)
        )

        # linear        
        self.linear = nn.Sequential(
            nn.Linear(in_features=ptw_in_dim * 2, out_features=multiout_dim),
            nn.Dropout(ptw_dropout),
            nn.Softplus()
        )
    
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch, channel, seq)
        x = self.center_crop(x) # crop
        # conv
        x = self.conv(x)
        # linear
        x = x.transpose(1,2) # convert to (batch, seq, channel)
        out = self.linear(x)
        out = out.transpose(1,2)
        return out



class PointwiseLabelBlock(nn.Module):
    def __init__(self,
                 crop_size = 64,
                 ptw_in_dim=384*2,
                 cls_dropout=0.05,
                 multiout_dim = 20
                ):
        super(PointwiseLabelBlock, self).__init__()
        self._model_name = "PointwiseLabelBlock"
        self.crop_size = crop_size

        # conv layer
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=ptw_in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=ptw_in_dim, out_channels=ptw_in_dim * 2, kernel_size=1, dilation=1),
            nn.Dropout(cls_dropout)
        )
        
        # linear layer
        self.linear = nn.Sequential(
            nn.Linear(in_features=ptw_in_dim * 2, out_features=multiout_dim),
            nn.Dropout(cls_dropout),
            nn.Sigmoid()
        )

    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])

    def forward(self, x):
        # take input shape: (batch, channel, seq)
        x = self.center_crop(x) # crop
        # conv
        x = self.conv(x)
        # linear
        x = x.transpose(1,2) # convert to (batch, seq, channel)
        label = self.linear(x)
        label = label.transpose(1,2)
        #print(out.shape)
        return label



class PointwiseLabelBlock_BK(nn.Module):
    def __init__(self, 
                 crop_size = 64,
                 ptw_in_dim=384*2,
                 cls_dropout=0.05,
                 multiout_dim = 20
                ):
        super(PointwiseLabelBlock, self).__init__()
        self._model_name = "PointwiseLabelBlock"
        self.crop_size = crop_size
        
        # class layer
        self.cls = nn.Sequential(
            nn.Linear(in_features=ptw_in_dim, out_features=multiout_dim * 2),
            nn.Dropout(cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=multiout_dim * 2, out_features=multiout_dim),
            nn.Dropout(cls_dropout),
            nn.Sigmoid()
        )
        
    # crop layer
    def center_crop(self, x):
        height = x.shape[1]
        width = x.shape[2] - (self.crop_size * 2)
        crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])
        
    def forward(self, x):
        # take input shape: (batch, channel, seq)
        # crop input
        x = self.center_crop(x)
        # convert to (batch, seq, channel)
        x = x.transpose(1,2)
        label = self.cls(x)
        label = label.transpose(1,2)
        #print(out.shape)
        return label


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
    
    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])


class SelfTrans_labelOUT(nn.Module):
    def __init__(
        self,
        seq_len=98304,
        bin_size=256,
        num_cnn_layer = 7,
        max_len = 384,
        add_positional_encoding = True,
        embed_size = 384*2,
        num_heads = 4,
        att_dropout = 0.4,
        dim_feedforward = 1024,
        enc_dropout = 0.2,
        batch_first = True,
        num_encoder_layers = 8,
        crop_size = 64,
        cls_dropout = 0.2,
        multiout_dim = 20
    ):

        super(SelfTrans_labelOUT, self).__init__()
        self.embedding_size = embed_size
        self.ptw_in_dim = self.embedding_size
        self.cls_in_dim = self.embedding_size
        self.add_positional_encoding = add_positional_encoding

        # Conv
        self.conv = ConvBlock(
                 seq_len,
                 bin_size,
                 embed_size,
                 num_cnn_layer
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        # Transformer encoder
        self.trans = TransformerEncoder(
                 embed_size,
                 num_heads,
                 att_dropout,
                 dim_feedforward,
                 enc_dropout,
                 #activation,
                 batch_first,
                 num_encoder_layers
        )
        
        # Pointwise
        self.ptw = PointwiseLabelBlock(
                 crop_size,
                 self.ptw_in_dim,
                 cls_dropout,
                 multiout_dim
        )

    def forward(self, src):
        conv_out = self.conv(src)
        conv_out = conv_out.transpose(1,2) # reshape input to: (batch, seq, feature)
        #print(conv_out.shape)
        
        # potional encoding
        if self.add_positional_encoding:
            conv_out = self.positional_encoding(conv_out)
            #print(conv_out.shape)
            
        # transformer
        trans_out = self.trans(conv_out)
        trans_out = trans_out.transpose(1,2) # reshape input to: (batch, channel, seq)
        #print(trans_out.shape)

        # pred signal & class
        label = self.ptw(trans_out)
        #print(out.shape)
        #print(label.shape)
        
        return label



class SelfTrans_signalOUT(nn.Module):
    def __init__(
        self,
        seq_len=98304,
        bin_size=256,
        num_cnn_layer = 7,
        max_len = 384,
        add_positional_encoding = True,
        embed_size = 384*2,
        num_heads = 4,
        att_dropout = 0.4,
        dim_feedforward = 1024,
        enc_dropout = 0.2,
        batch_first = True,
        num_encoder_layers = 8,
        crop_size = 64,
        ptw_dropout = 0.2,
        multiout_dim = 13
    ):

        super(SelfTrans_signalOUT, self).__init__()
        self.embedding_size = embed_size
        self.ptw_in_dim = self.embedding_size
        self.cls_in_dim = self.embedding_size
        self.add_positional_encoding = add_positional_encoding

        # Conv
        self.conv = ConvBlock(
                 seq_len,
                 bin_size,
                 embed_size,
                 num_cnn_layer
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        # Transformer encoder
        self.trans = TransformerEncoder(
                 embed_size,
                 num_heads,
                 att_dropout,
                 dim_feedforward,
                 enc_dropout,
                 #activation,
                 batch_first,
                 num_encoder_layers
        )
        
        # Pointwise
        self.ptw = PointwiseSignalBlock(
                 #max_len,
                 crop_size,
                 self.ptw_in_dim,
                 ptw_dropout,
                 multiout_dim
        )

    def forward(self, src):
        conv_out = self.conv(src)
        #print("conv_out", conv_out.shape)
        conv_out = conv_out.transpose(1,2) # reshape input to: (batch, seq, feature)
        #print("conv_out_transpose", conv_out.shape)
        
        # potional encoding
        if self.add_positional_encoding:
            conv_out = self.positional_encoding(conv_out)
            #print("positional", conv_out.shape)
            
        # transformer
        trans_out = self.trans(conv_out)
        #print("trans", trans_out.shape)
        trans_out = trans_out.transpose(1,2) # reshape input to: (batch, channel, seq)
        #print("trans_transpose", trans_out.shape)

        # pred signal & class
        out = self.ptw(trans_out)
        #print("out", out.shape)

        return out

