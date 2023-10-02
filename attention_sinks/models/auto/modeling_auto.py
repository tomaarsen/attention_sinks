
from transformers import LlamaConfig

from transformers import (
    AutoModel as TAutoModel,
    AutoModelForCausalLM as TAutoModelForCausalLM,
    AutoModelForSequenceClassification as TAutoModelForSequenceClassification
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from ..llama import LlamaModel, LlamaForCausalLM, LlamaForSequenceClassification

MODEL_MAPPING._extra_content = {
    LlamaConfig: LlamaModel
}
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content = {
    LlamaConfig: LlamaForCausalLM
}
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content = {
    LlamaConfig: LlamaForSequenceClassification
}

class AutoModel(TAutoModel):
    _model_mapping = MODEL_MAPPING

class AutoModelForCausalLM(TAutoModelForCausalLM):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

class AutoModelForSequenceClassification(TAutoModelForSequenceClassification):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
