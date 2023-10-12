from transformers import AutoModel as TAutoModel
from transformers import AutoModelForCausalLM as TAutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering as TAutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification as TAutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification as TAutoModelForTokenClassification

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class AutoModel(InjectAttentionSinksMixin, TAutoModel):
    pass


class AutoModelForCausalLM(InjectAttentionSinksMixin, TAutoModelForCausalLM):
    pass


class AutoModelForSequenceClassification(InjectAttentionSinksMixin, TAutoModelForSequenceClassification):
    pass


class AutoModelForQuestionAnswering(InjectAttentionSinksMixin, TAutoModelForQuestionAnswering):
    pass


class AutoModelForTokenClassification(InjectAttentionSinksMixin, TAutoModelForTokenClassification):
    pass
