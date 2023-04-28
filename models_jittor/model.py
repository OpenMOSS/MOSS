from functools import partial
from typing import Optional, Tuple, Union

import jittor as jt
import jittor.nn as nn
from jittor import Module

from .utils import NewGELUActivation
from .utils import (fixed_pos_embedding, apply_rotary_pos_emb, _init_weights,
                    get_head_mask)

class MossAttention(Module):
    def __init__(self, config):
        super(MossAttention, self).__init__()

        max_positions = config.n_positions
        self.register_buffer(
            "causal_mask",
            jt.tril(jt.ones((max_positions, max_positions), dtype=jt.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.n_embd
        self.num_attention_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = jt.sqrt(jt.float32(self.head_dim))
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        jt.float16

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to('float32')
        key = key.to('float32')

        attn_weights = jt.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = -3.4e38 # torch.finfo(attn_weights.dtype).min)

        mask_value = jt.Var(mask_value).type_as(attn_weights)
        attn_weights = jt.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = jt.matmul(attn_weights, value.float())
        if jt.flags.amp_level >= 1:
            attn_output = attn_output.half()

        return attn_output, attn_weights

    def execute(
        self,
        hidden_states: Optional[jt.Var],
        attention_mask: Optional[jt.Var] = None,
        layer_past: Optional[Tuple[jt.Var]] = None,
        head_mask: Optional[jt.Var] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[jt.Var, Tuple[jt.Var]],
        Optional[Tuple[jt.Var, Tuple[jt.Var], Tuple[jt.Var, ...]]],
    ]:
        qkv = self.qkv_proj(hidden_states)
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = jt.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = jt.cat([k_rot, k_pass], dim=-1)
            query = jt.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = jt.cat((past_key, key), dim=-2)
            value = jt.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present


class MossMLP(Module):
    def __init__(self, intermediate_size, config):
        # in MLP: intermediate_size= 4 * embed_dim
        super(MossMLP, self).__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def execute(self, hidden_states: Optional[jt.Var]) -> jt.Var:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MossBlock(Module):
    def __init__(self, config):
        super(MossBlock, self).__init__()
        self.config = config
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MossAttention(config)
        self.mlp = MossMLP(inner_dim, config)

    def execute(
        self,
        hidden_states: Optional[jt.Var],
        layer_past: Optional[Tuple[jt.Var]] = None,
        attention_mask: Optional[jt.Var] = None,
        head_mask: Optional[jt.Var] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[jt.Var], Optional[Tuple[jt.Var, Tuple[jt.Var, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache
        )
        attn_output = attn_outputs[0]  # output_attn: a, present
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present


class MossModel(Module):
    def __init__(self, config):
        super(MossModel, self).__init__()

        self.config = config
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MossBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.n_head)

        self.gradient_checkpointing = False

        self.apply(partial(_init_weights, config))

    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None,
        attention_mask: Optional[jt.Var] = None,
        token_type_ids: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        head_mask: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = jt.arange(past_length, input_shape[-1] + past_length, dtype='int64')
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # [batch_size, 1, 1, to_seq_length]
            attention_mask = attention_mask[:, None, None, :]

            if jt.flags.amp_level >= 3:
                attention_mask = attention_mask.half()  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -65504.0
            else:
                # finfo.min
                attention_mask = (1.0 - attention_mask) * -3.402e38

        # n_layer x batch x num_attention_heads x N x N
        head_mask = get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        presents = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return hidden_states, presents


class MossForCausalLM(Module):

    def __init__(self, config):
        super(MossForCausalLM, self).__init__()
        self.config = config
        self.transformer = MossModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, config))

    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None,
        attention_mask: Optional[jt.Var] = None,
        token_type_ids: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        head_mask: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
        labels: Optional[jt.Var] = None,
        use_cache: Optional[bool] = None,
    ):

        hidden_states, presents = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        lm_logits = self.lm_head(hidden_states).to('float32')

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        return dict(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents
        )
