import math

import jittor as jt
import jittor.nn as nn

class NewGELUActivation(jt.Module):

    def execute(self, input):

        output = (input + 0.044715 * jt.pow(input.float(), 3))
        if  jt.flags.amp_level >= 1:
            output = output.half()

        return 0.5 * input * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * output))
    
def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (jt.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        jt.einsum("i , j -> i j", jt.arange(seq_len, dtype=jt.float), inv_freq).float()
    )
    if jt.flags.use_tensorcore:
        sinusoid_inp = sinusoid_inp.half()
    return jt.sin(sinusoid_inp), jt.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = jt.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def duplicate_interleave(m):
    """
    A simple version of `jt.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

def _init_weights(module, config):
    if isinstance(module, (nn.Linear,)):
        # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def _convert_head_mask_to_5d(head_mask, num_hidden_layers, dtype):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=dtype)  # switch to float if need + fp16 compatibility
    return head_mask

def get_head_mask(
        head_mask, num_hidden_layers: int,
        is_attention_chunked: bool = False
    ):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers, 'float16')
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask