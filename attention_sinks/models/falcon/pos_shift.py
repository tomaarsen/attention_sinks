import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

__all__ = ["falcon_pos_shift_attention_forward"]


def falcon_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads,
        query_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    # CHANGED:
    # query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length, position_ids)
    query_layer_copy = query_layer.clone()
    query_layer, _ = self.maybe_rotary(query_layer, query_layer_copy, past_kv_length, position_ids)
    # ########

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, kv_length, head_dim]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    _, kv_length, _ = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    # CHANGED:
    key_layer_copy = key_layer.clone()
    key_position_ids = torch.arange(kv_length, device=position_ids.device).unsqueeze(0)
    _, key_layer = self.maybe_rotary(key_layer_copy, key_layer, 0, key_position_ids)
    # ########
    float_min = torch.finfo(query_layer.dtype).min
    attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float_min).to(query_layer.dtype)

    query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
    key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
    value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

    if alibi is None:
        if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
            # TODO: deprecate this once we add FA2 support in Falcon
            # query_layer_.shape, key_layer_.shape, value_layer_.shape
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
            )
            attention_scores = None
        else:
            attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
            attention_scores /= math.sqrt(self.head_dim)

            attention_scores = F.softmax(attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            attn_output = attention_scores @ value_layer_

        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        if output_attentions:
            return output_tensor, present, attention_scores
        else:
            return output_tensor, present

    else:
        matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
        # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
        # equivalent and more performant, but there might be a numerical difference. If you're reading this
        # and you'd like to experiment and maybe file a PR, feel free!
        attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
        attention_logits *= self.inv_norm_factor
        attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size, num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        if output_attentions:
            return output_tensor, present, attention_probs
        else:
            return output_tensor, present
