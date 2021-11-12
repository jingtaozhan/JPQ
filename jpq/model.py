import torch
from torch import nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class RobertaDot(RobertaPreTrainedModel):
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(self._init_weights)        
    
    def encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0]
        embeds = self.norm(self.embeddingHead(full_emb))
        return embeds
    
    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)