from transformers import LlamaForCausalLM as TLlamaForCausalLM
from transformers import LlamaForSequenceClassification as TLlamaForSequenceClassification
from transformers import LlamaModel as TLlamaModel
from transformers import LlamaPreTrainedModel as TLlamaPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class LlamaPreTrainedModel(InjectAttentionSinksMixin, TLlamaPreTrainedModel):
    pass


class LlamaModel(LlamaPreTrainedModel, TLlamaModel):
    pass


class LlamaForCausalLM(LlamaPreTrainedModel, TLlamaForCausalLM):
    pass


class LlamaForSequenceClassification(LlamaPreTrainedModel, TLlamaForSequenceClassification):
    pass
