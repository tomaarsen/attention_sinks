from transformers import AutoModel as TAutoModel
from transformers import AutoModelForCausalLM as TAutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering as TAutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification as TAutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification as TAutoModelForTokenClassification
from transformers import FalconConfig, GPTJConfig, GPTNeoXConfig, LlamaConfig, MistralConfig, MptConfig
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
from ..gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXForQuestionAnswering,
    GPTNeoXForSequenceClassification,
    GPTNeoXForTokenClassification,
    GPTNeoXModel,
)
from ..gptj import (
    GPTJForCausalLM,
    GPTJForQuestionAnswering,
    GPTJForSequenceClassification,
    GPTJModel,
)
from ..llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel
from ..mistral import MistralForCausalLM, MistralForSequenceClassification, MistralModel
from ..mpt import (
    MptForCausalLM,
    MptForQuestionAnswering,
    MptForSequenceClassification,
    MptForTokenClassification,
    MptModel,
)
from ..qwen import QWenModel, QWenLMHeadModel, QWenConfig

MODEL_MAPPING._extra_content = {
    LlamaConfig: LlamaModel,
    FalconConfig: FalconModel,
    MptConfig: MptModel,
    GPTNeoXConfig: GPTNeoXModel,
    GPTJConfig: GPTJModel,
    MistralConfig: MistralModel,
    QWenConfig: QWenModel,
}
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content = {
    LlamaConfig: LlamaForCausalLM,
    FalconConfig: FalconForCausalLM,
    MptConfig: MptForCausalLM,
    GPTNeoXConfig: GPTNeoXForCausalLM,
    GPTJConfig: GPTJForCausalLM,
    MistralConfig: MistralForCausalLM,
    QWenConfig: QWenLMHeadModel,
}
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content = {
    LlamaConfig: LlamaForSequenceClassification,
    FalconConfig: FalconForSequenceClassification,
    MptConfig: MptForSequenceClassification,
    GPTNeoXConfig: GPTNeoXForSequenceClassification,
    GPTJConfig: GPTJForSequenceClassification,
    MistralConfig: MistralForSequenceClassification,
}
MODEL_FOR_QUESTION_ANSWERING_MAPPING._extra_content = {
    FalconConfig: FalconForQuestionAnswering,
    MptConfig: MptForQuestionAnswering,
    GPTNeoXConfig: GPTNeoXForQuestionAnswering,
    GPTJConfig: GPTJForQuestionAnswering,
}
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING._extra_content = {
    FalconConfig: FalconForTokenClassification,
    MptConfig: MptForTokenClassification,
    GPTNeoXConfig: GPTNeoXForTokenClassification,
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
