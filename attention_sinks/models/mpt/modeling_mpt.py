import os
import types
from typing import Optional, Tuple, Union

from transformers import MptForCausalLM as TMptForCausalLM
from transformers import MptForQuestionAnswering as TMptForQuestionAnswering
from transformers import MptForSequenceClassification as TMptForSequenceClassification
from transformers import MptForTokenClassification as TMptForTokenClassification
from transformers import MptModel as TMptModel
from transformers import MptPreTrainedModel as TMptPreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.mpt.modeling_mpt import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    MPT_INPUTS_DOCSTRING,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

from attention_sinks.attention_sink_kv_cache import AttentionSinkKVCache
from attention_sinks.generation.utils import GenerationMixin


class MptPreTrainedModel(GenerationMixin, TMptPreTrainedModel):
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
        attention_sink_kwargs = {key: value for key, value in kwargs.items() if key.startswith("attention_sink")}
        for key in attention_sink_kwargs:
            kwargs.pop(key)

        model = super(MptPreTrainedModel, cls).from_pretrained(
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
        # MPT already has windowed attention

        # Hackishly attach the Attention Sink KV Cache to the model
        cls._attach_attention_sink_kv_cache(model, **attention_sink_kwargs)

        return model

    @classmethod
    def _attach_attention_sink_kv_cache(cls, module, **attention_sink_kwargs):
        if isinstance(module, TMptModel):
            # Create the new cache
            module.attention_sink_kv_cache = AttentionSinkKVCache(
                **attention_sink_kwargs,
                k_seq_dim=2,
                v_seq_dim=2,
            )

            # Keep track of the old forward method, we need it in the wrapped one
            old_forward = module.forward

            # Wrap the forward by overriding the past_key_values using the cache
            @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
            @add_code_sample_docstrings(
                checkpoint=_CHECKPOINT_FOR_DOC,
                output_type=BaseModelOutputWithPastAndCrossAttentions,
                config_class=_CONFIG_FOR_DOC,
            )
            def wrapped_forward(self, *args, **kwargs) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                outputs = old_forward(*args, **kwargs)
                outputs.past_key_values = self.attention_sink_kv_cache(outputs.past_key_values)
                return outputs

            module.forward = types.MethodType(wrapped_forward, module)

        # Recursively call this to find all MptModels
        for module in reversed(module._modules.values()):
            if len(list(module.children())) > 0:
                cls._attach_attention_sink_kv_cache(module, **attention_sink_kwargs)


class MptModel(MptPreTrainedModel, TMptModel):
    pass


class MptForCausalLM(MptPreTrainedModel, TMptForCausalLM):
    pass


class MptForSequenceClassification(MptPreTrainedModel, TMptForSequenceClassification):
    pass


class MptForTokenClassification(MptPreTrainedModel, TMptForTokenClassification):
    pass


class MptForQuestionAnswering(MptPreTrainedModel, TMptForQuestionAnswering):
    pass
