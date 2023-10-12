"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""


from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from transformers.models.gptj.modeling_gptj import get_embed_positions, rotate_every_two
from transformers.utils import is_torch_fx_proxy

__all__ = ["gptj_pos_shift_attention_forward"]


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, None, :, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, None, :, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


def gptj_pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Tuple[torch.Tensor]],
    Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
]:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_attention_heads, self.head_dim, False)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, False)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

    if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
        # The logic to conditionally copy to GPU could not be traced, so we do this
        # every time in the torch.fx case
        embed_positions = get_embed_positions(self.embed_positions, position_ids)
    else:
        embed_positions = self._get_embed_positions(position_ids)

    repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

    if self.rotary_dim is not None:
        # [bsz, num_attn_heads, seq_len, head_dim]
        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]
        q_rot = apply_rotary_pos_emb(q_rot, sin, cos)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        query = apply_rotary_pos_emb(query, sin, cos)

    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    key_position_ids = torch.arange(key.shape[-2], device=position_ids.device).unsqueeze(0)
    repeated_key_position_ids = key_position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    sincos = torch.gather(embed_positions, 1, repeated_key_position_ids)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    if self.rotary_dim is not None:
        # [bsz, num_attn_heads, seq_len, head_dim]
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]
        k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
        key = torch.cat([k_rot, k_pass], dim=-1)
    else:
        key = apply_rotary_pos_emb(key, sin, cos)

    # compute self-attention: V x Softmax(QK^T)
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)
