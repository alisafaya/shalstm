
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.generation_utils import GenerationMixin
from transformers.configuration_utils import PretrainedConfig

from ..model import SHALSTM

from collections import namedtuple

_output = namedtuple('SHALSTMOutput', ['logits'])

class SHALSTMforCausalGeneration(SHALSTM, GenerationMixin):
    
    def __init__(self, config, device=torch.device('cpu')):
        super().__init__(config, device=device)
        self.config = PretrainedConfig()

    def forward(self, **kwargs):
        output, h, new_hidden, new_mems = super().forward(kwargs['input_ids'].t(), attention_mask=kwargs.pop('attention_mask', None))
        return _output(output.permute(1, 0, 2))
