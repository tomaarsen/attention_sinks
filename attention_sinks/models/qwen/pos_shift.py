from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.utils import logging

__all__ = ["qwen_pos_shift_attention_forward"]
logger = logging.get_logger(__name__)

SUPPORT_TORCH2 = hasattr(torch, "__version__") and int(torch.__version__.split(".")[0]) >= 2
_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """\
We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
"""

ATTEMPTED_IMPORT = False
apply_rotary_emb_func = None
flash_attn_unpadded_func = None


def _import_flash_attn():
    global apply_rotary_emb_func, flash_attn_unpadded_func
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func as __apply_rotary_emb_func

        apply_rotary_emb_func = __apply_rotary_emb_func
    except ImportError:
        logger.warn(
            "Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary"
        )

    try:
        import flash_attn

        if not hasattr(flash_attn, "__version__"):
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        else:
            if int(flash_attn.__version__.split(".")[0]) >= 2:
                from flash_attn.flash_attn_interface import flash_attn_varlen_func as __flash_attn_unpadded_func
            else:
                from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        flash_attn_unpadded_func = __flash_attn_unpadded_func
    except ImportError as ex:
        logger.warn(
            "Warning: import flash_attn fail, please install FlashAttention to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention"
        )


def quantize_cache_v(fdata, bits, qmax, qmin):
    # b, s, head, h-dim->b, head, s, h-dim
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape

    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    # Compute params
    if qmax.device != fmax.device:
        qmax = qmax.to(device)
        qmin = qmin.to(device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    # Quantize
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    return qdata.contiguous(), scale, zero


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    if apply_rotary_emb_func is not None and t.is_cuda:
        t_ = t.float()
        cos = cos.squeeze(0).squeeze(1)[:, : cos.shape[-1] // 2]
        sin = sin.squeeze(0).squeeze(1)[:, : sin.shape[-1] // 2]
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs[0].shape[-1]
        cos, sin = freqs
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * cos) + (_rotate_half(t_) * sin)
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def qwen_pos_shift_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    # Custom import attempt
    global ATTEMPTED_IMPORT
    if self.use_flash_attn and not ATTEMPTED_IMPORT:
        ATTEMPTED_IMPORT = True
        _import_flash_attn()

    mixed_x_layer = self.c_attn(hidden_states)

    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    if rotary_pos_emb_list is not None:
        cur_len = query.shape[1]
        if len(rotary_pos_emb_list) == 1:
            rotary_pos_emb = rotary_pos_emb_list[0]
            rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            query = apply_rotary_pos_emb(query, q_pos_emb)
            # key = apply_rotary_pos_emb(key, k_pos_emb)
        else:
            # TODO: modify batch infer
            query_list = []
            key_list = []
            for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # Slice the pos emb for current inference
                query_list += [apply_rotary_pos_emb(query[i : i + 1, :, :], q_pos_emb)]
                key_list += [apply_rotary_pos_emb(key[i : i + 1, :, :], k_pos_emb)]
            query = torch.cat(query_list, dim=0)
            key = torch.cat(key_list, dim=0)

    if self.use_cache_quantization:
        key = quantize_cache_v(key.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)
        value = quantize_cache_v(value.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)

    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        if self.use_cache_quantization:
            # use_cache_quantization:
            # present=((q_key,key_scale,key_zero_point),
            #          (q_value,value_scale,value_zero_point))
            key = (
                torch.cat((past_key[0], key[0]), dim=2),
                torch.cat((past_key[1], key[1]), dim=2),
                torch.cat((past_key[2], key[2]), dim=2),
            )
            value = (
                torch.cat((past_value[0], value[0]), dim=2),
                torch.cat((past_value[1], value[1]), dim=2),
                torch.cat((past_value[2], value[2]), dim=2),
            )
        else:
            # not use_cache_quantization:
            # present=(key,value)
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

    if use_cache:
        present = (key, value)
    else:
        present = None

    ### Shift pos ###
    kv_seq_len = key.size(1)
    key_shifted_position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=key.device)
    key_rotary_pos_emb = [it[:, key_shifted_position_ids, :, :] for it in rotary_pos_emb_list[0]]
    key = apply_rotary_pos_emb(key, key_rotary_pos_emb)
    #######

    if self.use_logn_attn and not self.training:
        if self.use_cache_quantization:
            seq_start = key[0].size(2) - query.size(1)
            seq_end = key[0].size(2)
        else:
            seq_start = key.size(1) - query.size(1)
            seq_end = key.size(1)
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
        query = query * logn_tensor.expand_as(query)

    if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32 and query.is_cuda:
        q, k, v = query, key, value
        attn_output = self.core_attention_flash(q, k, v, attention_mask=attention_mask)
    else:
        query = query.permute(0, 2, 1, 3)
        if not self.use_cache_quantization:
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
        if (
            registered_causal_mask is None
            and self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
            and not query.is_cuda
        ):
            raise Exception(_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED)

        if not self.use_cache_quantization and SUPPORT_TORCH2:
            causal_mask = registered_causal_mask[:, :, key.size(-2) - query.size(-2) : key.size(-2), : key.size(-2)]
            if attention_mask is not None:
                attention_mask = attention_mask.expand(-1, -1, causal_mask.size(2), -1).masked_fill(
                    ~causal_mask, torch.finfo(query.dtype).min
                )
            else:
                attention_mask = causal_mask
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask).transpose(1, 2)
            attn_weight = None
        else:
            attn_output, attn_weight = self._attn(query, key, value, registered_causal_mask, attention_mask, head_mask)
    context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)

    attn_output = self.c_proj(context_layer)

    outputs = (attn_output, present)
    if output_attentions:
        if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32:
            raise ValueError("Cannot output attentions while using flash-attn")
        else:
            outputs += (attn_weight,)

    return outputs
