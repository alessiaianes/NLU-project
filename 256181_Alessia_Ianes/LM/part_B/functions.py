import math
import torch.nn as nn
import torch

# Function to perform one training epoch
def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Performs a single training epoch.

    Args:
        data (DataLoader): DataLoader for the training dataset
        optimizer (optim.Optimizer): The optimizer used for training
        criterion (nn.Module): The loss function
        model (nn.Module): The language model
        clip (float): The threshold for gradient clipping

    Returns:
        float: The average loss per token for the epoch
    """
    model.train() # Set the model to training mode (enables dropout, etc.)
    loss_array = [] # List to store loss values for the epoch
    number_of_tokens = [] # List to store the number of tokens processed in each batch
    
    # Iterate through batches provided by the DataLoader    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source']) # Perform the forward pass: predict output tokens based on input sequence
        loss = criterion(output, sample['target']) # Compute the loss
        loss_array.append(loss.item() * sample["number_tokens"]) # Store the loss scaled by the number of tokens
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # Clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    # Compute and return the average loss per token over the entire epoch
    return sum(loss_array)/sum(number_of_tokens)


# Function to evaluate the model on a dataset (validation or test)
def eval_loop(data, eval_criterion, model):
    """
    Evaluates the model on a given dataset.

    Args:
        data (DataLoader): DataLoader for the evaluation dataset
        eval_criterion (nn.Module): The loss function used for evaluation (often with reduction='sum')
        model (nn.Module): The language model

    Returns:
        tuple: A tuple containing:
            - float: The perplexity (PPL) on the dataset
            - float: The average loss per token on the dataset
    """
    model.eval() # Set the model to evaluation mode
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source']) # Perform the forward pass
            loss = eval_criterion(output, sample['target']) # Compute the loss using the evaluation criterion
            loss_array.append(loss.item()) # Store the loss value
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens)) # Calculate Perplexity (PPL) as the exponential of the average loss

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


# Function to initialize model weights
def init_weights(mat):
    """
    Initializes the weights of the model layers using specific strategies

    Args:
        mat (nn.Module): The model (or a part of it) whose weights need initialization
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]: # Check if the module is a recurrent layer 
            for name, param in m.named_parameters():
                if 'weight_ih' in name: # Initialize input-hidden weights
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name: # Initialize hidden-hidden weights
                    for idx in range(4): 
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name: # Initialize biases
                    param.data.fill_(0)
        else: # Handle other types of layers, like Linear
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


# Helper function to update the averaged model state dictionary
def update_avg_model(avg_model_sd, current_model_sd, num_averaged_steps):
    """ 
    Updates the running average of the model's state dictionary based on the current model's weights

    Args:
        avg_model_sd (dict): The current averaged state dictionary (weights)
        current_model_sd (dict): The state dictionary of the model at the current step
        num_averaged_steps (int): The total number of models included in the average so far (including the current one)

    Returns:
        dict: The updated averaged state dictionary
    """
    # Iterate through each parameter (key) in the state dictionaries
    for key in avg_model_sd:
        # Compute the new average for the current parameter:
        # Formula: avg_new = avg_old * (N-1)/N + current_param / N
        # where N is `num_averaged_steps`.
        # Ensure both tensors are on the CPU for this calculation
        avg_model_sd[key] = avg_model_sd[key] * (num_averaged_steps - 1.0) / num_averaged_steps + \
                            current_model_sd[key].cpu() / num_averaged_steps
    return avg_model_sd
