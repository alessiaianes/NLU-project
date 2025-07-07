import torch
import torch.nn as nn
from transformers import BertModel

# Define the joint Intent and Slot filling model using BERT
class BertModelIAS(nn.Module):
    """
    A PyTorch module that integrates a pre-trained BERT model for joint Intent Classification and Slot Filling

    Args:
        out_slot (int): The number of possible slot tags (output classes for slot filling)
        out_int (int): The number of possible intents (output classes for intent classification)
        dropout (float): The dropout rate to apply after BERT layers. Defaults to 0.1
    """
    def __init__(self, out_slot, out_int, dropout=0.1):
        super(BertModelIAS, self).__init__()
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Initialize a dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer for slot classification
        # Input size is BERT's hidden size
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        
        # Linear layer for intent classification
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

    def forward(self, utterance, attention_mask):
        """
        Defines the forward pass of the model

        Args:
            utterance (torch.Tensor): Input tensor containing token IDs for the utterances
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor indicating which tokens are real (1) and which are padding (0)
                                           Shape: (batch_size, sequence_length)

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted slot logits for each token. Shape: (batch_size, num_slot_classes, sequence_length)
                - torch.Tensor: Predicted intent logits for the sequence. Shape: (batch_size, num_intent_classes)
        """
        # Forward pass through BERT
        # `outputs` is a dictionary-like object containing various outputs from BERT,
        # including hidden states and the pooled output.
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
        slots = slots.permute(0, 2, 1) 

        return slots, intent