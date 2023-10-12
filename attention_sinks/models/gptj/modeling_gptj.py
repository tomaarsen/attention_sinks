from transformers import GPTJForCausalLM as TGPTJForCausalLM
from transformers import GPTJForQuestionAnswering as TGPTJForQuestionAnswering
from transformers import GPTJForSequenceClassification as TGPTJForSequenceClassification
from transformers import GPTJModel as TGPTJModel
from transformers import GPTJPreTrainedModel as TGPTJPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class GPTJPreTrainedModel(InjectAttentionSinksMixin, TGPTJPreTrainedModel):
    pass


class GPTJModel(GPTJPreTrainedModel, TGPTJModel):
    pass


class GPTJForCausalLM(GPTJPreTrainedModel, TGPTJForCausalLM):
    pass


class GPTJForSequenceClassification(GPTJPreTrainedModel, TGPTJForSequenceClassification):
    pass


class GPTJForQuestionAnswering(GPTJPreTrainedModel, TGPTJForQuestionAnswering):
    pass
