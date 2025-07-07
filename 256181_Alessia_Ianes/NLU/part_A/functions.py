import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report


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


# Function to perform one training epoch
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Performs a single training epoch over the provided data

    Args:
        data (DataLoader): DataLoader providing batches of training data
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights
        criterion_slots (nn.Module): Loss function for slot tagging
        criterion_intents (nn.Module): Loss function for intent classification
        model (nn.Module): The model to be trained.
        clip (float): The maximum norm for gradient clipping to prevent exploding gradients

    Returns:
        list: A list containing the average loss for each batch processed during the epoch
    """
    # Set the model to training mode
    model.train()
    # Initialize a list to store the loss values for each batch
    loss_array = []
    # Iterate through each batch
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        # Perform the forward pass: get predicted slots and intents from the model
        slots, intent = model(sample['utterances'], sample['slots_len'])
        # Calculate the loss for intent classification
        loss_intent = criterion_intents(intent, sample['intents'])
        # Calculate the loss for slot tagging
        loss_slot = criterion_slots(slots, sample['y_slots'])
        # Combine the losses
        loss = loss_intent + loss_slot 
        # Append the loss value to the array
        loss_array.append(loss.item())
        loss.backward() # Compute gradients of the loss with respect to model parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # Clip gradients to prevent exploding gradients, ensuring stable training
        optimizer.step() # Update the weights
    return loss_array

# Function to evaluate the model on a given dataset
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluates the model's performance on a given dataset (validation or test)

    Args:
        data (DataLoader): DataLoader providing batches of evaluation data
        criterion_slots (nn.Module): Loss function for slot tagging
        criterion_intents (nn.Module): Loss function for intent classification
        model (nn.Module): The model to be evaluated
        lang (Lang): Language object containing mappings (id2intent, id2slot, id2word)

    Returns:
        tuple: A tuple containing:
            - dict: Slot tagging evaluation results (e.g., F1 score)
            - dict: Intent classification evaluation report (e.g., accuracy, precision, recall)
            - list: A list containing the average loss for each batch processed
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store losses and prediction results for aggregation
    loss_array = []
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    # Disable gradient calculations to reduce memory consumption and speed up computation
    with torch.no_grad():
        # Iterate through each batch
        for sample in data:
            # Perform the forward pass to get predicted slots and intents
            slots, intents = model(sample['utterances'], sample['slots_len'])
            # Compute the loss
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            

            # Get the predicted intent label IDs by finding the index of the highest probability
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            # Get the ground truth intent label IDs and map them to labels
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            # Extend the lists for overall intent evaluation
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Get the predicted slot label IDs for each sequence in the batch
            output_slots = torch.argmax(slots, dim=1)
            # Process each sequence within the batch
            for id_seq, seq in enumerate(output_slots):
                # Get the actual length of the sequence
                length = sample['slots_len'].tolist()[id_seq]
                # Get the utterance words and ground truth slot labels for the current sequence
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                # Map ground truth IDs to slot labels
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                # Map utterance word IDs to words
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                # Format ground truth slots for CoNLL evaluation
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode): # Iterate only up to the sequence length
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem])) # Map predicted ID to slot label
                hyp_slots.append(tmp_seq)

    try:
        # Evaluate slot tagging performance using the CoNLL evaluation script            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Handle potential errors during slot evaluation 
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
    
    # Generate a detailed classification report for intent classification using sklearn
    # zero_division=False prevents warnings/errors if a class has no predicted or true samples
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    
    # Return the slot evaluation results, the intent classification report, and the batch losses
    return results, report_intent, loss_array
