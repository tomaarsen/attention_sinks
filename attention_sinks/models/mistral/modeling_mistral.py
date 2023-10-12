from transformers import MistralForCausalLM as TMistralForCausalLM
from transformers import MistralForSequenceClassification as TMistralForSequenceClassification
from transformers import MistralModel as TMistralModel
from transformers import MistralPreTrainedModel as TMistralPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class MistralPreTrainedModel(InjectAttentionSinksMixin, TMistralPreTrainedModel):
    pass


class MistralModel(MistralPreTrainedModel, TMistralModel):
    pass


class MistralForCausalLM(MistralPreTrainedModel, TMistralForCausalLM):
    pass


class MistralForSequenceClassification(MistralPreTrainedModel, TMistralForSequenceClassification):
    pass
