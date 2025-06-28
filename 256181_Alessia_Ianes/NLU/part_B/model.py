import torch.nn as nn
from transformers import BertModel

class BertModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, dropout=0.1):
        super(BertModelIAS, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

    def forward(self, utterance, attention_mask):
        outputs = self.bert(input_ids=utterance, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slots = self.slot_out(sequence_output)
        intent = self.intent_out(pooled_output)

        slots = slots.permute(0, 2, 1)
        return slots, intent