from transformers import FalconForCausalLM as TFalconForCausalLM
from transformers import FalconForQuestionAnswering as TFalconForQuestionAnswering
from transformers import FalconForSequenceClassification as TFalconForSequenceClassification
from transformers import FalconForTokenClassification as TFalconForTokenClassification
from transformers import FalconModel as TFalconModel
from transformers import FalconPreTrainedModel as TFalconPreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class FalconPreTrainedModel(InjectAttentionSinksMixin, TFalconPreTrainedModel):
    pass


class FalconModel(FalconPreTrainedModel, TFalconModel):
    pass


class FalconForCausalLM(FalconPreTrainedModel, TFalconForCausalLM):
    pass


class FalconForSequenceClassification(FalconPreTrainedModel, TFalconForSequenceClassification):
    pass


class FalconForTokenClassification(FalconPreTrainedModel, TFalconForTokenClassification):
    pass


class FalconForQuestionAnswering(FalconPreTrainedModel, TFalconForQuestionAnswering):
    pass
