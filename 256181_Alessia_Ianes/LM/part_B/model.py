import torch
import torch.nn as nn

# Class for an LSTM Language Model with Weight Tying
class LM_LSTM_weight_tying(nn.Module):
    """
    An LSTM-based Language Model where the output layer's weights are tied
    (shared) with the input embedding layer's weights

    Weight tying is a technique that can reduce the number of parameters and
    potentially improve generalization by enforcing a relationship between
    input representations and output prediction distributions
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        """
        Initializes the LM_LSTM_weight_tying model

        Args:
            emb_size (int): The dimension of the word embeddings
            hidden_size (int): The number of features in the LSTM's hidden state
            output_size (int): The size of the output vocabulary (number of classes)
            pad_index (int, optional): The index of the padding token. Defaults to 0
            n_layers (int, optional): The number of stacked LSTM layers. Defaults to 1
        """
        super(LM_LSTM_weight_tying, self).__init__()
        # Embedding layer: Maps input token IDs to dense vectors of size `emb_size`
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM layer: Processes the sequence of embeddings.
        # `batch_first=True` means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # Output layer: Linearly transforms LSTM hidden states to vocabulary logits
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: Share the weights of the output layer with the embedding layer
        # This means the matrix used for projecting LSTM output to vocabulary logits
        # is the same as the matrix used for looking up word embeddings.
        # Note: This requires `emb_size` to be equal to `hidden_size` if `n_layers=1` and bidirectional=False.
        # Ensure compatibility.
        if emb_size != hidden_size:
             print(f"Warning: Weight tying might be problematic as emb_size ({emb_size}) != hidden_size ({hidden_size}). "
                   f"The output layer will use hidden_size, but tying requires matching dimensions.")     
        self.output.weight = self.embedding.weight 

        # Store the padding token index for potential use 
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        """
        Defines the forward pass of the LSTM model with weight tying

        Args:
            input_sequence (torch.Tensor): Input tensor of token IDs
                                           Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Output logits for each token position
                          Shape: (batch_size, output_size, sequence_length)
        """

        # 1. Embed the input sequence tokens
        # Shape: (batch_size, sequence_length, emb_size)
        emb = self.embedding(input_sequence)
         # 2. Pass the embedded sequence through the LSTM layer
        # `lstm_out` contains hidden states for each time step. Shape: (batch_size, sequence_length, hidden_size)
        # `_` captures the final hidden and cell states, which are ignored in this forward pass
        lstm_out, _ = self.lstm(emb)
        # 3. Project the LSTM output sequence to vocabulary logits using the tied output layer
        # Shape: (batch_size, sequence_length, output_size)
        # 4. Permute dimensions to (N, C, L) format, often required by loss functions like CrossEntropyLoss
        # Shape: (batch_size, output_size, sequence_length)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    

# Custom Dropout Class implementing Variational Dropout
class VariationalDropout(nn.Module):
    """
    Implements Variational Dropout 

    This dropout technique applies the *same* dropout mask across all time steps (or sequence positions)
    for a given batch item's feature vector. This is often preferred for recurrent networks (RNNs, LSTMs)
    as it helps maintain temporal consistency compared to standard dropout applied independently at each time step
    """
    def __init__(self, dropout_prob):
        """
        Initializes the VariationalDropout layer

        Args:
            dropout_prob (float): The probability of setting an element to zero (the dropout rate)
        """
        super(VariationalDropout, self).__init__()
        # Check if dropout_prob is valid
        if not 0 <= dropout_prob <= 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, but got {dropout_prob}")
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        Applies the variational dropout mask

        Args:
            x (torch.Tensor): The input tensor. Expected shape is typically (batch, seq_len, features)
                              or (batch, features) if applied to embeddings before LSTM

        Returns:
            torch.Tensor: The tensor after applying dropout
        """
        # Dropout is only applied during training and if the dropout probability is non-zero
        if not self.training or self.dropout_prob == 0:
            return x
        
        # Create a dropout mask.
        # 1. Generate a random tensor with the same shape as the features dimension of the input,
        #    broadcastable across the sequence length.
        #    Shape: (batch_size, num_features) 
        # `x.new_empty` creates a tensor on the same device and with the same dtype as x
        # `bernoulli_` fills the tensor with values drawn from a Bernoulli distribution (0 or 1)
        # The probability of drawing a 1 is `1 - self.dropout_prob`
        mask = x.new_empty(x.size(0), x.size(2)).bernoulli_(1 - self.dropout_prob)
        # Scale the mask during training
        # To ensure the expected value of the output remains the same as the input,
        # we scale the kept values (1s in the mask) by `1 / (1 - dropout_prob)`.
        mask = mask.div_(1 - self.dropout_prob)
        # Expand the mask's dimensions to allow broadcasting across the sequence length
        # Input `x` shape: (batch, seq_len, features). Mask shape: (batch, features).
        # We need mask shape: (batch, 1, features) for broadcasting.
        mask = mask.unsqueeze(1) 

        # Apply the mask element-wise by multiplication.
        return x * mask


# LSTM Language Model with Weight Tying and Variational Dropout
class LM_LSTM_wt_vd(nn.Module):
    """
    An LSTM-based Language Model incorporating both Weight Tying and Variational Dropout

    Combines the benefits of parameter sharing (weight tying) with the regularization
    effects of variational dropout, potentially leading to better generalization and
    reduced overfitting
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, 
                 emb_dropout=0.1, out_dropout=0.4):
        """
        Initializes the LM_LSTM_wt_vd model

        Args:
            emb_size (int): The dimension of the word embeddings
            hidden_size (int): The number of features in the LSTM's hidden state
            output_size (int): The size of the output vocabulary
            pad_index (int, optional): The index of the padding token. Defaults to 0
            n_layers (int, optional): The number of stacked LSTM layers. Defaults to 1
            emb_dropout (float, optional): Dropout rate for the embedding layer using Variational Dropout. Defaults to 0.1
            out_dropout (float, optional): Dropout rate for the LSTM output layer using Variational Dropout. Defaults to 0.4
        """
        super(LM_LSTM_wt_vd, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Variational Dropout layers applied after embedding
        self.embedding_dropout = VariationalDropout(emb_dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        # Variational Dropout applied after LSTM output
        self.output_dropout = VariationalDropout(out_dropout)

        # Output layer (with shared weights)
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: Shared weights between embedding and output layer
        # Requires emb_size == hidden_size for this specific implementation.
        if emb_size != hidden_size:
             print(f"Warning: Weight tying requires emb_size ({emb_size}) == hidden_size ({hidden_size}). "
                   f"Tying may not work as expected.")
        self.output.weight = self.embedding.weight

        # Store padding token index
        self.pad_token = pad_index


        
    def forward(self, input_sequence):
        """
        Defines the forward pass of the LSTM model with weight tying and variational dropout

        Args:
            input_sequence (torch.Tensor): Input tensor of token IDs
                                           Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Output logits for each token position
                          Shape: (batch_size, output_size, sequence_length)
        """
        # 1. Embed the input sequence
        # Shape: (batch_size, sequence_length, emb_size)
        emb = self.embedding(input_sequence)
        # 2. Apply variational dropout to the embeddings
        emb = self.embedding_dropout(emb)
        
        # 3. Pass the dropout-applied embeddings through the LSTM layer
        # `lstm_out` contains hidden states for each time step. Shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(emb)
        
        # 4. Apply variational dropout to the LSTM output sequence
        lstm_out = self.output_dropout(lstm_out)
        
        # 5. Project the dropout-applied LSTM output to vocabulary logits using the tied output layer
        # Shape: (batch_size, sequence_length, output_size)
        # 6. Permute dimensions to (N, C, L) for compatibility with loss functions
        # Shape: (batch_size, output_size, sequence_length)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output