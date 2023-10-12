from transformers import MptForCausalLM as TMptForCausalLM
from transformers import MptForQuestionAnswering as TMptForQuestionAnswering
from transformers import MptForSequenceClassification as TMptForSequenceClassification
from transformers import MptForTokenClassification as TMptForTokenClassification
from transformers import MptModel as TMptModel
from transformers import MptPreTrainedModel as TMptPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class MptPreTrainedModel(InjectAttentionSinksMixin, TMptPreTrainedModel):
    pass


class MptModel(MptPreTrainedModel, TMptModel):
    pass


class MptForCausalLM(MptPreTrainedModel, TMptForCausalLM):
    pass


class MptForSequenceClassification(MptPreTrainedModel, TMptForSequenceClassification):
    pass


class MptForTokenClassification(MptPreTrainedModel, TMptForTokenClassification):
    pass


class MptForQuestionAnswering(MptPreTrainedModel, TMptForQuestionAnswering):
    pass
