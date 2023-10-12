from transformers import GPTNeoXForCausalLM as TGPTNeoXForCausalLM
from transformers import GPTNeoXForQuestionAnswering as TGPTNeoXForQuestionAnswering
from transformers import GPTNeoXForSequenceClassification as TGPTNeoXForSequenceClassification
from transformers import GPTNeoXForTokenClassification as TGPTNeoXForTokenClassification
from transformers import GPTNeoXModel as TGPTNeoXModel
from transformers import GPTNeoXPreTrainedModel as TGPTNeoXPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class GPTNeoXPreTrainedModel(InjectAttentionSinksMixin, TGPTNeoXPreTrainedModel):
    pass


class GPTNeoXModel(GPTNeoXPreTrainedModel, TGPTNeoXModel):
    pass


class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel, TGPTNeoXForCausalLM):
    pass


class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel, TGPTNeoXForSequenceClassification):
    pass


class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel, TGPTNeoXForTokenClassification):
    pass


class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel, TGPTNeoXForQuestionAnswering):
    pass
