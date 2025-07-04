import torch.nn as nn
from transformers import BertModel

class BertModelIAS(nn.Module):
    def __init__(self, out_slot, out_int, dropout=0.1):
        super(BertModelIAS, self).__init__()
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer for slot classification (applied to each token's representation)
        # Input size is BERT's hidden size
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        
        # Linear layer for intent classification (applied to the pooled output, e.g., [CLS] token representation)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

    def forward(self, utterance, attention_mask):
        # Forward pass through BERT
        # `outputs` contains last_hidden_state, pooler_output, etc.
        outputs = self.bert(input_ids=utterance, attention_mask=attention_mask)
        
        # Sequence output: Hidden states for each token in the sequence
        # Shape: (batch_size, seq_len, bert_hidden_size)
        sequence_output = outputs.last_hidden_state 
        
        # Pooled output: Typically the representation of the [CLS] token, processed by a linear layer + activation
        # Shape: (batch_size, bert_hidden_size)
        pooled_output = outputs.pooler_output 

        # Apply dropout to the BERT outputs
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Predict slots for each token
        # Output shape: (batch_size, seq_len, num_slot_classes)
        slots = self.slot_out(sequence_output)
        
        # Predict intent based on pooled output
        # Output shape: (batch_size, num_intent_classes)
        intent = self.intent_out(pooled_output)

        # Permute slots output for CrossEntropyLoss compatibility
        # Original shape: (batch_size, seq_len, num_slot_classes)
        # Permuted shape: (batch_size, num_slot_classes, seq_len)
        # This matches the expected (N, C, L) format for CrossEntropyLoss on sequences.
        slots = slots.permute(0, 2, 1) 

        return slots, intent