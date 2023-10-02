import os
from typing import List, Optional, Tuple, Union
import torch
from transformers import (
    LlamaModel as TLlamaModel,
    LlamaPreTrainedModel as TLlamaPreTrainedModel,
    LlamaForCausalLM as TLlamaForCausalLM,
    LlamaForSequenceClassification as TLlamaForSequenceClassification,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from attention_sinks.attention_sink_kv_cache import AttentionSinkKVCache

from attention_sinks.models.llama.pos_shift import enable_llama_pos_shift_attention


class LlamaPreTrainedModel(TLlamaPreTrainedModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> None:
        # Separate Attention Sink kwargs from regular kwargs
        attention_sink_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key.startswith("attention_sink")
        }
        for key in attention_sink_kwargs:
            kwargs.pop(key)

        model = super(LlamaPreTrainedModel, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        # Enable position shifting attention for Llama
        enable_llama_pos_shift_attention(model)

        # Hackishly attach the Attention Sink KV Cache to the model
        model.attention_sink_kv_cache = AttentionSinkKVCache(
            **attention_sink_kwargs,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        return model


class LlamaModel(LlamaPreTrainedModel, TLlamaModel):
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        outputs.past_key_values = self.attention_sink_kv_cache(past_key_values)
        breakpoint()
        return outputs

class LlamaForCausalLM(LlamaPreTrainedModel, TLlamaForCausalLM):
    pass


class LlamaForSequenceClassification(
    LlamaPreTrainedModel, TLlamaForSequenceClassification
):
    pass
