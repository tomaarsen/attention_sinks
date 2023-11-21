# adapted from https://huggingface.co/01-ai/Yi-6B/blob/main/modeling_yi.py

from typing import Optional, Tuple
import math
from einops import repeat
import torch
import torch.nn as nn

__all__ = ["yi_pos_shift_attention_forward"]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

is_flash_attn_available = False

def flash_attn_func(*args, **kwargs) : 
    pass

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, flash_attn_available):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    expand_dim = 2 if flash_attn_available else 1
    cos = cos[position_ids].unsqueeze(expand_dim)  # [bs, seq_len, dim]
    sin = sin[position_ids].unsqueeze(expand_dim)  # [bs, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# For now we disable flash attention

def yi_pos_shift_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )

        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        if not is_flash_attn_available:
            if self.num_key_value_groups > 1:
                key_states = repeat(
                    key_states, f"b n h d -> b n (h {self.num_key_value_groups}) d"
                )
                value_states = repeat(
                    value_states, f"b n h d -> b n (h {self.num_key_value_groups}) d"
                )

            # b n h d -> b h n d
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        seq_dim = 1 if is_flash_attn_available else 2
        kv_seq_len = key_states.shape[seq_dim]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[seq_dim]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, _ = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, is_flash_attn_available
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=seq_dim)
            value_states = torch.cat([past_key_value[1], value_states], dim=seq_dim)

        past_key_value = (key_states, value_states) if use_cache else None
        
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        _, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, key_position_ids, is_flash_attn_available
        )

        if is_flash_attn_available:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout_p=0.0, causal=True
            )
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                        f"{attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                dtype_min = torch.tensor(
                    torch.finfo(attn_weights.dtype).min,
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
                attn_weights = torch.max(attn_weights, dtype_min)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        if not is_flash_attn_available:
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value