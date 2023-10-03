from transformers import (
    AutoModel as TAutoModel,
)
from transformers import (
    AutoModelForCausalLM as TAutoModelForCausalLM,
)
from transformers import (
    AutoModelForQuestionAnswering as TAutoModelForQuestionAnswering,
)
from transformers import (
    AutoModelForSequenceClassification as TAutoModelForSequenceClassification,
)
from transformers import (
    AutoModelForTokenClassification as TAutoModelForTokenClassification,
)
from transformers import FalconConfig, LlamaConfig, MptConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
)

from ..falcon import (
    FalconForCausalLM,
    FalconForQuestionAnswering,
    FalconForSequenceClassification,
    FalconForTokenClassification,
    FalconModel,
)
from ..llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel
from ..mpt import (
    MptForCausalLM,
    MptForQuestionAnswering,
    MptForSequenceClassification,
    MptForTokenClassification,
    MptModel,
)

MODEL_MAPPING._extra_content = {LlamaConfig: LlamaModel, FalconConfig: FalconModel, MptConfig: MptModel}
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content = {
    LlamaConfig: LlamaForCausalLM,
    FalconConfig: FalconForCausalLM,
    MptConfig: MptForCausalLM,
}
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content = {
    LlamaConfig: LlamaForSequenceClassification,
    FalconConfig: FalconForSequenceClassification,
    MptConfig: MptForSequenceClassification,
}
MODEL_FOR_QUESTION_ANSWERING_MAPPING._extra_content = {
    FalconConfig: FalconForQuestionAnswering,
    MptConfig: MptForQuestionAnswering,
}
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING._extra_content = {
    FalconConfig: FalconForTokenClassification,
    MptConfig: MptForTokenClassification,
}


class AutoModel(TAutoModel):
    _model_mapping = MODEL_MAPPING


class AutoModelForCausalLM(TAutoModelForCausalLM):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


class AutoModelForSequenceClassification(TAutoModelForSequenceClassification):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


class AutoModelForQuestionAnswering(TAutoModelForQuestionAnswering):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


class AutoModelForTokenClassification(TAutoModelForTokenClassification):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
