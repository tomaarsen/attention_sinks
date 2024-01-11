import types
from typing import Callable, Optional

from transformers import PreTrainedModel
from transformers.utils import logging

from attention_sinks.attention_sink_kv_cache import AttentionSinkKVCache
from attention_sinks.generation.utils import _update_model_kwargs_for_generation

logger = logging.get_logger(__name__)

MODEL_NAME_MAPPING = {
    "llama": "LlamaModel",
    "falcon": "FalconModel",
    "mpt": "MptModel",
    "gpt_neox": "GPTNeoXModel",
    "gptj": "GPTJModel",
    "mistral": "MistralModel",
    "qwen": "QWenModel",
    "stablelm_epoch": "StableLMEpochModel",
    "btlm": "BTLMModel",
    "Yi": "YiModel",
}
ATTENTION_NAME_MAPPING = {
    "llama": "LlamaAttention",
    "falcon": "FalconAttention",
    "mpt": "MptAttention",
    "gpt_neox": "GPTNeoXAttention",
    "gptj": "GPTJAttention",
    "mistral": "MistralAttention",
    "qwen": "QWenAttention",
    "stablelm_epoch": "Attention",
    "btlm": "BTLMAttention",
    "Yi": "YiAttention",
}
KV_DIM_MAPPING = {
    "llama": (2, 2),
    "falcon": (2, 2),
    "mpt": (2, 2),
    "gpt_neox": (2, 2),
    "gptj": (2, 2),
    "mistral": (2, 2),
    "qwen": (1, 1),
    "stablelm_epoch": (2, 2),
    "btlm": (2, 2),
    "Yi": (2, 2),
}


class InjectAttentionSinksMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Separate Attention Sink kwargs from regular kwargs
        attention_sink_kwargs = {key: value for key, value in kwargs.items() if key.startswith("attention_sink")}
        for key in attention_sink_kwargs:
            kwargs.pop(key)

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model_type = model.config.model_type
        if model_type not in MODEL_NAME_MAPPING:
            raise NotImplementedError(
                f"`attention_sinks` does not support models with the `{model_type}` architecture at this time."
            )

        # Enable position shifting attention
        call_count = cls._inject_pos_shift_attention(model)
        if call_count is not None:
            logger.warn(
                f"[Attention Sinks] Injected Position Shifting into {call_count} attention class{'es' if call_count != 1 else ''}."
            )

        # Inject the Attention Sink KV Cache to the model
        call_count = cls._inject_attention_sink_kv_cache(model, **attention_sink_kwargs)
        logger.warn(
            f"[Attention Sinks] Injected Attention Sink KV Cache into {call_count} model class{'es' if call_count != 1 else ''}."
        )

        # Overwrite broken model kwargs, prevents indexing error when generating
        # The default _update_model_kwargs_for_generation expects the seq_length to keep growing
        # as generation occurs, but that isn't the case
        model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, model)

        return model
    @classmethod
    def from_quantized(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Separate Attention Sink kwargs from regular kwargs
        attention_sink_kwargs = {key: value for key, value in kwargs.items() if key.startswith("attention_sink")}
        for key in attention_sink_kwargs:
            kwargs.pop(key)

        model = super().from_quantized(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model_type = model.config.model_type
        if model_type not in MODEL_NAME_MAPPING:
            raise NotImplementedError(
                f"`attention_sinks` does not support models with the `{model_type}` architecture at this time."
            )

        # Enable position shifting attention
        call_count = cls._inject_pos_shift_attention(model)
        if call_count is not None:
            logger.warn(
                f"[Attention Sinks] Injected Position Shifting into {call_count} attention class{'es' if call_count != 1 else ''}."
            )

        # Inject the Attention Sink KV Cache to the model
        call_count = cls._inject_attention_sink_kv_cache(model, **attention_sink_kwargs)
        logger.warn(
            f"[Attention Sinks] Injected Attention Sink KV Cache into {call_count} model class{'es' if call_count != 1 else ''}."
        )

        # Overwrite broken model kwargs, prevents indexing error when generating
        # The default _update_model_kwargs_for_generation expects the seq_length to keep growing
        # as generation occurs, but that isn't the case
        model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, model)

        return model
    @classmethod
    def _inject_pos_shift_attention(cls, model: PreTrainedModel) -> Optional[int]:
        model_type = model.config.model_type

        from attention_sinks.models import (
            falcon_pos_shift_attention_forward,
            gpt_neox_pos_shift_attention_forward,
            gptj_pos_shift_attention_forward,
            llama_pos_shift_attention_forward,
            mistral_pos_shift_attention_forward,
            qwen_pos_shift_attention_forward,
            stablelm_epoch_pos_shift_attention_forward,
            yi_pos_shift_attention_forward,
        )

        ATTENTION_FORWARD_MAPPING = {
            "llama": llama_pos_shift_attention_forward,
            "falcon": falcon_pos_shift_attention_forward,
            "mpt": None,
            "gpt_neox": gpt_neox_pos_shift_attention_forward,
            "gptj": gptj_pos_shift_attention_forward,
            "mistral": mistral_pos_shift_attention_forward,
            "qwen": qwen_pos_shift_attention_forward,
            "stablelm_epoch": stablelm_epoch_pos_shift_attention_forward,
            "btlm": None,
            "Yi": yi_pos_shift_attention_forward,
        }

        # Not all models require updated attention forwards
        if ATTENTION_FORWARD_MAPPING[model_type] is None:
            return

        def overwrite_forward(module) -> None:
            module.forward = types.MethodType(ATTENTION_FORWARD_MAPPING[model_type], module)

        return cls._call_modules_by_name(model, ATTENTION_NAME_MAPPING[model_type], overwrite_forward)

    @classmethod
    def _inject_attention_sink_kv_cache(cls, model, **attention_sink_kwargs) -> int:
        model_type = model.config.model_type
        attention_sink_kwargs["k_seq_dim"], attention_sink_kwargs["v_seq_dim"] = KV_DIM_MAPPING[model_type]

        def overwrite_forward(module):
            # Create the new cache
            module.attention_sink_kv_cache = AttentionSinkKVCache(**attention_sink_kwargs)

            # Keep track of the old forward method, we need it in the wrapped one
            old_forward = module.forward

            # Wrap the forward by overriding the past_key_values using the cache
            def wrapped_forward(self, *args, **kwargs):
                outputs = old_forward(*args, **kwargs)
                outputs.past_key_values = self.attention_sink_kv_cache(outputs.past_key_values)
                return outputs

            module.forward = types.MethodType(wrapped_forward, module)

        return cls._call_modules_by_name(model, MODEL_NAME_MAPPING[model_type], overwrite_forward)

    @classmethod
    def _call_modules_by_name(cls, module, target_name: str, func: Callable) -> int:
        if module.__class__.__name__ == target_name:
            func(module)
            return 1

        return sum(cls._call_modules_by_name(module, target_name, func) for module in module.children())
