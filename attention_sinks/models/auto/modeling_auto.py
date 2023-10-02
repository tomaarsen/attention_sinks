from transformers import (
    AutoModel as TAutoModel,
)
from transformers import (
    AutoModelForCausalLM as TAutoModelForCausalLM,
)
from transformers import (
    AutoModelForSequenceClassification as TAutoModelForSequenceClassification,
)
from transformers import LlamaConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
)

from ..llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel

MODEL_MAPPING._extra_content = {LlamaConfig: LlamaModel}
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content = {LlamaConfig: LlamaForCausalLM}
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content = {LlamaConfig: LlamaForSequenceClassification}


class AutoModel(TAutoModel):
    _model_mapping = MODEL_MAPPING


class AutoModelForCausalLM(TAutoModelForCausalLM):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


class AutoModelForSequenceClassification(TAutoModelForSequenceClassification):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
