import torch
import torch.nn as nn

# Define a basic Recurrent Neural Network (RNN) language model class
class LM_RNN(nn.Module):
    """
    A simple RNN-based Language Model

    This model uses a basic RNN layer to process sequences and predict the next token
    It includes an embedding layer, an RNN layer, and a final linear layer for output
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        """
        Initializes the LM_RNN model

        Args:
            emb_size (int): The size of the word embeddings
            hidden_size (int): The number of features in the hidden state of the RNN
            output_size (int): The size of the output vocabulary (number of classes)
            pad_index (int, optional): The index used for padding tokens. Defaults to 0
            n_layers (int, optional): Number of recurrent layers. Defaults to 1
        """
        super(LM_RNN, self).__init__()
        # Embedding layer: Converts input token IDs into dense vectors of size `emb_size`
        # `output_size` is the vocabulary size. `padding_idx` ensures the embedding for padding is zero
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # RNN layer: Processes the sequence of embeddings
        # - `emb_size`: Input feature size
        # - `hidden_size`: Hidden state size
        # - `n_layers`: Stacked RNN layers
        # - `bidirectional=False`: Only a forward RNN
        # - `batch_first=True`: Input/output tensors are (batch, seq, feature)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    

        # Store the padding token index for potential use
        self.pad_token = pad_index
        # Output layer: A linear transformation from the RNN's hidden state (`hidden_size`)
        # to the vocabulary size (`output_size`), predicting the probability distribution
        # for the next token
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        """
        Defines the forward pass of the RNN language model

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
        # 2. Pass the embeddings through the RNN layer
        # `rnn_out` contains the hidden state output for each time step. Shape: (batch_size, sequence_length, hidden_size)
        # `_` captures the final hidden state (and cell state if LSTM), which is not used in this forward pass
        rnn_out, _  = self.rnn(emb)
        # 3. Project the RNN output sequence to the vocabulary size using the linear layer
        # Shape: (batch_size, sequence_length, output_size)
        # 4. Permute the dimensions to match the expected format for CrossEntropyLoss (N, C, L)
        # Shape: (batch_size, output_size, sequence_length)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    

# Define an LSTM-based Language Model with Dropout
class LM_LSTM_dropout(nn.Module):
    """
    An LSTM-based Language Model incorporating dropout for regularization

    This model uses an LSTM layer, which is generally more effective than simple RNNs
    at capturing long-range dependencies. Dropout is applied to both the embeddings
    and the LSTM output to prevent overfitting
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        """
        Initializes the LM_LSTM_dropout model

        Args:
            emb_size (int): The size of the word embeddings
            hidden_size (int): The number of features in the hidden and cell states of the LSTM
            output_size (int): The size of the output vocabulary (number of classes)
            pad_index (int, optional): The index used for padding tokens. Defaults to 0
            out_dropout (float, optional): Dropout probability applied after the LSTM layer. Defaults to 0.1
            emb_dropout (float, optional): Dropout probability applied after the embedding layer. Defaults to 0.1
            n_layers (int, optional): Number of LSTM layers stacked. Defaults to 1
        """
        super(LM_LSTM_dropout, self).__init__()

        # Embedding layer: Converts token IDs to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Dropout layer applied to embeddings
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        # LSTM layer: Processes sequences, capturing dependencies with gates
        # - `emb_size`: Input feature size (from embeddings)
        # - `hidden_size`: Hidden and cell state size
        # - `n_layers`: Stacked LSTM layers
        # - `bidirectional=False`: Standard unidirectional LSTM
        # - `batch_first=True`: Input/output format is (batch, seq, feature)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)   
        # Dropout layer applied to the output of the LSTM layer
        self.out_dropout_layer = nn.Dropout(out_dropout)
        # Output layer: Linear transformation from LSTM hidden state to vocabulary size
        self.output = nn.Linear(hidden_size, output_size)
        # Store the padding token index
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        """
        Defines the forward pass of the LSTM language model with dropout

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
        # 2. Apply dropout to the embeddings
        emb = self.emb_dropout_layer(emb)
        # 3. Pass the embedded sequence through the LSTM layer
        # `lstm_out` contains the hidden states for each time step. Shape: (batch_size, sequence_length, hidden_size)
        # `_` captures the final hidden and cell states, ignored here
        lstm_out, _ = self.lstm(emb)
        # 4. Apply dropout to the LSTM output
        lstm_out = self.out_dropout_layer(lstm_out)
        # 5. Project the LSTM output to the vocabulary size
        # Shape: (batch_size, sequence_length, output_size)
        # 6. Permute dimensions for the loss function (N, C, L)
        # Shape: (batch_size, output_size, sequence_length)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    

# Define a basic LSTM Language Model without explicit dropout layers
class LM_LSTM(nn.Module):
    """
    A standard LSTM-based Language Model without dropout layers

    This model provides the core LSTM functionality for sequence modeling
    It consists of an embedding layer, an LSTM layer, and a linear output layer
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        """
        Initializes the LM_LSTM model

        Args:
            emb_size (int): The size of the word embeddings
            hidden_size (int): The number of features in the hidden and cell states of the LSTM
            output_size (int): The size of the output vocabulary (number of classes)
            pad_index (int, optional): The index used for padding tokens. Defaults to 0
            n_layers (int, optional): Number of LSTM layers stacked. Defaults to 1
        """
        super(LM_LSTM, self).__init__()
        # Embedding layer: Maps token IDs to dense vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM layer: Processes the sequence of embeddings
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # Output layer: Transforms LSTM hidden states to vocabulary logits
        self.output = nn.Linear(hidden_size, output_size)
        # Store the padding token index
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        """
        Defines the forward pass of the basic LSTM language model

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
        # 2. Pass the embeddings through the LSTM layer
        # `lstm_out` contains hidden states for each time step. Shape: (batch_size, sequence_length, hidden_size)
        # `_` captures the final hidden and cell states, which are ignored here
        lstm_out, _ = self.lstm(emb)
        # 3. Project the LSTM output sequence to the vocabulary size
        # Shape: (batch_size, sequence_length, output_size)
        # 4. Permute dimensions to (N, C, L) for compatibility with loss functions
        # Shape: (batch_size, output_size, sequence_length)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output