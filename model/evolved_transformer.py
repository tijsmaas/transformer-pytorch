import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

# pylint: disable=arguments-differ


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class GLU(nn.Module):
    def __init__(self, in_features, dropout_rate):
        super(GLU, self).__init__()
        self.sigm = nn.Sigmoid()
        self.W = nn.Linear(in_features, out_features=512, bias=True)
        self.V = nn.Linear(in_features, out_features=512, bias=True)
        initialize_weight(self.W)
        initialize_weight(self.V)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.W(x) * self.sigm(self.V(x))
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        # FIXME: is this correct? Why are we not counting in attention heads
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

# See https://discuss.pytorch.org/t/using-optimised-depthwise-convolutions/11819/14
# Visual explanation https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
#  this impl considers depthwise multiplier K=1
# FIXME: Please optimize this to use less memory
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        initialize_weight(self.conv1)
        initialize_weight(self.pointwise)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = torch.tensor(beta)

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()
        # dropout applied uniformly after each layer
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.glu = GLU(hidden_size, dropout_rate)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.conv1x1_2048 = nn.Linear(hidden_size, 2048)
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.dropout_b = nn.Dropout(dropout_rate)

        self.conv3x1_256 = nn.Conv2d(in_channels=512, out_channels=256,
                                     kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.layer_norm = nn.LayerNorm(2048, eps=1e-6)

        self.sep_conv_9x1 = SeparableConv2d(in_channels=2048, out_channels=256,
                                            kernel_size=(9, 1), padding=(4, 0))

        initialize_weight(self.conv1x1_2048)
        initialize_weight(self.conv3x1_256)

    # 20 x 46 x 512
    # 24 x 36 x 512
    # batch x words? x emb
    def forward(self, x, mask):  # pylint: disable=arguments-differ
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        # batch * emb * 512
        y = self.self_attention_norm(x) # emb
        y = self.glu(y) # -> 512
        y = self.dropout_1(x) # dropout, residual, norm
        x = x + y  # residual connection (can only add if shapes are the same)
        # -----
        y = self.ffn_norm(x)
        # L branch
        ya = self.conv1x1_2048(y)
        ya = self.relu_a(ya)
        # R branch
        yb = self.conv3x1_256(y.view(1, 512, sentence_len, batch_size)) #
        yb = yb.view(batch_size, sentence_len, 256)
        yb = self.relu_b(yb)
        # Merge, note that channels of yb fit 8 times in ya
        yb = yb.repeat(1, 1, 8)
        y = ya + yb
        y = self.dropout_2(y)
        y = self.layer_norm(y)
        y = self.sep_conv_9x1(y.view(1, 2048, sentence_len, batch_size)) # -> 256 channels
        y = y.view(batch_size, sentence_len, 256)
        y = y.repeat(1, 1, 2)
        # Careful! dropout on something with 'duplicated memory'
        y = self.dropout_3(y)
        x = x + y
        # Transformer self-att
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_1 = MultiHeadAttention(hidden_size, dropout_rate, head_size=16)
        self.enc_dec_attention_1 = MultiHeadAttention(hidden_size, dropout_rate)

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)


        self.conv1x1_2048 = nn.Linear(hidden_size, 2048)
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.dropout_b = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(1024, eps=1e-6)

        self.conv3x1_256 = nn.Conv2d(in_channels=512, out_channels=256,
                                     kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.layer_norm = nn.LayerNorm(2048, eps=1e-6)

        self.sep_conv11x1_1024 = SeparableConv2d(in_channels=512, out_channels=1024,
                                            kernel_size=(11, 1), padding=(5, 0))
        self.sep_conv_7x1_256 = SeparableConv2d(in_channels=512, out_channels=256,
                                            kernel_size=(7, 1), padding=(3, 0))

        self.sep_conv_7x1_512 = SeparableConv2d(in_channels=1024, out_channels=512,
                                            kernel_size=(7, 1), padding=(3, 0))

        initialize_weight(self.conv1x1_2048)
        initialize_weight(self.conv3x1_256)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        assert enc_output is not None
        batch_size = x.shape[0]
        sentence_len = x.shape[1]

        y = self.layer_norm_1(x)
        # TODO: change from 8 -> 16 self attention heads
        ya = self.self_attention_1(y, y, y, self_mask)
        if enc_output is not None:
            yb = self.enc_dec_attention_1(y, enc_output, enc_output, i_mask, cache)
            y = ya + yb
        else:
            y = ya
        # -----
        x = x + y
        x = self.dropout_1(x)
        y = self.layer_norm_2(x)
        # L branch
        ya = self.sep_conv11x1_1024(y.view(1, 512, sentence_len, batch_size))
        ya = ya.view(batch_size, sentence_len, 1024)
        ya = self.relu_a(ya)
        # R branch
        yb = self.sep_conv_7x1_256(y.view(1, 512, sentence_len, batch_size))  #
        yb = yb.view(batch_size, sentence_len, 256)
        # Merge, note that channels of yb fit 8 times in ya
        yb = yb.repeat(1, 1, 4)
        y = ya + yb
        y = self.dropout_2(y)
        y = self.layer_norm_3(y)
        y = self.sep_conv_7x1_512(y.view(1, 1024, sentence_len, batch_size))  # -> 256 channels
        y = y.view(batch_size, sentence_len, 512)
        y = self.dropout_3(y)
        x = x + y

        # --- original decoder start
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        # --- Inserted swish activation function
        x = self.layer1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.layer2(x)
        # ---
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)


class EvolvedTransformer(nn.Module):
    """
    Big/Deep models have dropout 0.3, all others with input emb 768 have 0.2

    Decoding beam size: 4, see So et al. 2019 for more details.
    """
    def __init__(self, i_vocab_size, t_vocab_size,
                 n_layers=6,
                 hidden_size=512,
                 filter_size=2048,
                 dropout_rate=0.2,
                 share_target_embedding=True,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None):
        super(EvolvedTransformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
                        std=hidden_size**-0.5)
        self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers)

        if has_inputs:
            if not share_target_embedding:
                self.i_vocab_embedding = nn.Embedding(i_vocab_size,
                                                      hidden_size)
                nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
                                std=hidden_size**-0.5)
            else:
                self.i_vocab_embedding = self.t_vocab_embedding

            self.i_emb_dropout = nn.Dropout(dropout_rate)

            self.encoder = Encoder(hidden_size, filter_size,
                                   dropout_rate, n_layers)

        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, targets):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            enc_output = self.encode(inputs, i_mask)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.size()[1]
        t_self_mask = utils.create_trg_self_mask(target_size,
                                                 device=targets.device)
        return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)

    def encode(self, inputs, i_mask):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.emb_scale
        input_embedded += self.get_position_encoding(inputs)
        input_embedded = self.i_emb_dropout(input_embedded)

        return self.encoder(input_embedded, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask,
               cache=None):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask,
                                      t_self_mask, cache)
        # linear
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))

        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal
