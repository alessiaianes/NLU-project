import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class ModelIAS(nn.Module):
    """
    Implements a neural network model for joint Intent Classification and Slot Filling (IAS).
    This model uses a bidirectional LSTM to encode utterances and predicts slots for each token
    and an overall intent for the utterance
    """

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.1):
        """
        Initializes the ModelIAS.

        Args:
            hid_size (int): The number of hidden units in the LSTM layer
            out_slot (int): The number of possible slot tags (output dimension for slot classification)
            out_int (int): The number of possible intents (output dimension for intent classification)
            emb_size (int): The dimensionality of the word embeddings
            vocab_len (int): The total size of the vocabulary (number of unique words)
            n_layer (int, optional): The number of recurrent layers in the LSTM. Defaults to 1
            pad_index (int, optional): The index used for padding tokens in the input sequences. Defaults to 0
            dropout (float, optional): The dropout probability applied to embeddings, LSTM output, and final hidden state. Defaults to 0.1
        """
        super(ModelIAS, self).__init__()
        
        # 1. Embedding Layer: Converts input word indices into dense vector representations
        #    - vocab_len: Size of the vocabulary
        #    - emb_size: Dimension of each word embedding vector
        #    - padding_idx=pad_index: Ensures that the embedding vector for the padding index remains zero and is not updated during training
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # 2. Bidirectional LSTM Encoder: Processes the sequence of word embeddings
        #    - emb_size: Input size to the LSTM (embedding dimension)
        #    - hid_size: Number of hidden units in each LSTM layer
        #    - n_layer: Number of stacked LSTM layers
        #    - bidirectional=True: Allows the LSTM to process the sequence in both forward and backward directions
        #    - batch_first=True: Specifies that the input tensors are expected in the format (batch_size, seq_len, input_size)        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)   

        # 3. Slot Output Layer: A linear layer to predict slot tags for each token in the sequence
        #    - Input size is hid_size * 2 because the bidirectional LSTM outputs concatenated forward and backward hidden states
        #    - out_slot: Number of possible slot tags 
        self.slot_out = nn.Linear(hid_size * 2, out_slot)

        # 4. Intent Output Layer: A linear layer to predict the intent of the entire utterance
        #    - Input size is hid_size * 2, taking the final concatenated hidden state from the LSTM
        #    - out_int: Number of possible intents
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
        # 5. Dropout Layer: Applies dropout regularization to prevent overfitting
        #    - dropout: The probability of dropping units during training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, utterance, seq_lengths):
        """
        Defines the forward pass of the model

        Args:
            utterance (torch.Tensor): A tensor containing the input utterance sequences (word indices)
                                      Shape: (batch_size, seq_len)
            seq_lengths (torch.Tensor): A tensor containing the actual lengths of each sequence in the batch
                                        Shape: (batch_size)

        Returns:
            tuple: A tuple containing:
                - slots (torch.Tensor): Logits for slot tagging. Shape: (batch_size, out_slot, seq_len)
                - intent (torch.Tensor): Logits for intent classification. Shape: (batch_size, out_int)
        """
        # Convert word indices to embedding vectors
        utt_emb = self.embedding(utterance) 
        # Apply dropout to the embeddings for regularization
        utt_emb = self.dropout(utt_emb)
        
        # Pack the padded sequence embeddings. This is an optimization that prevents the LSTM from
        # processing the padding tokens, saving computation.
        # seq_lengths needs to be a numpy array for pack_padded_sequence
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Pass the packed sequence through the bidirectional LSTM encoder
        # packed_output: Contains the packed hidden states for all time steps
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the LSTM output sequence back into a padded tensor
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to the LSTM output sequence for regularization before feeding to the slot classifier
        utt_encoded = self.dropout(utt_encoded)

        # For a bidirectional LSTM, the final hidden state is composed of the last forward pass state
        # and the first backward pass state (which corresponds to the last state in the backward sequence).
        # These are located at indices -1 (last backward) and -2 (last forward) in the `last_hidden` tensor
        # when n_layer=1 and bidirectional=True.
        # Concatenate them to get a single vector representation of the entire utterance.
        # last_hidden[-2,:,:] -> Last forward hidden state. Shape: (batch_size, hid_size)
        # last_hidden[-1,:,:] -> Last backward hidden state. Shape: (batch_size, hid_size)
        last_hidden =  torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)

        # Apply dropout to the final utterance representation for regularization before the intent classifier
        last_hidden = self.dropout(last_hidden)

        
        
        # Compute slot logits by passing the encoded sequence through the slot output linear layer
        slots = self.slot_out(utt_encoded)
        # Compute intent logits by passing the final utterance representation through the intent output linear layer
        intent = self.intent_out(last_hidden)
        
        # Permute the slots tensor to match the expected format for CrossEntropyLoss,
        # which requires the class dimension to be the second dimension: (batch_size, num_classes, seq_len)
        slots = slots.permute(0,2,1) # We need this for computing the loss
        
        return slots, intent
    
