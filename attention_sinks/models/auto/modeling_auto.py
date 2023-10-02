
from transformers import LlamaConfig

from transformers import (
    AutoModel as TAutoModel,
    AutoModelForCausalLM as TAutoModelForCausalLM
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
# , MODEL_MAPPING_NAMES
# from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
# from transformers.models.auto.auto_factory import _LazyAutoMapping
# from .auto_factory import _LazyAutoMapping

from ..llama import LlamaModel, LlamaForCausalLM

MODEL_MAPPING._extra_content = {
    LlamaConfig: LlamaModel
}
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content = {
    LlamaConfig: LlamaForCausalLM
}

class AutoModel(TAutoModel):
    _model_mapping = MODEL_MAPPING

class AutoModelForCausalLM(TAutoModelForCausalLM):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
