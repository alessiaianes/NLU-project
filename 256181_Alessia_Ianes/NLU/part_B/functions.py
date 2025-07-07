import torch
from conll import evaluate
from sklearn.metrics import classification_report
from utils import * 

# Function to perform one training epoch
def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Trains the model for one epoch

    Args:
        data_loader (DataLoader): DataLoader for the training dataset
        optimizer (Optimizer): PyTorch optimizer instance
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot tagging
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent classification
        model (nn.Module): The neural network model to train
        clip (float): Gradient clipping value to prevent exploding gradients

    Returns:
        float: The average training loss over the epoch
    """
    model.train() # Set the model to training mode
    total_loss = 0 # Initialize total loss accumulator
    num_batches = 0 # Initialize batch counter

    # Iterate over each batch of data provided by the data_loader
    for sample in data_loader:
        optimizer.zero_grad() # Clear previous gradients before computing new ones
        
        # Get attention mask from the sample batch
        attention_mask = sample['attention_mask']
        
        # --- Forward Pass ---
        # Pass the utterance (token IDs) and attention mask through the model
        # The model returns predicted slot logits and intent logits
        slots, intents = model(sample['utterance'], attention_mask)
        
        # --- Calculate Losses ---
        # Calculate the intent classification loss using the ground truth intent labels
        loss_intent = criterion_intents(intents, sample['intent'])
        
        # Calculate the slot tagging loss
        loss_slot = criterion_slots(slots, sample['slots'])
        # Combine the intent and slot losses
        loss = loss_intent + loss_slot
        
        # Backward pass and optimization
        loss.backward() # Compute gradients of the loss with respect to model parameters
        # Clip gradients to prevent them from becoming too large, which helps stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update model parameters using the optimizer
        
        # Accumulate the loss for this batch
        total_loss += loss.item()
        num_batches += 1
        
    # Compute and return the average loss per batch for the epoch
    return total_loss / num_batches if num_batches > 0 else 0


# This function evaluates the model on a given dataset
def eval_loop(data_loader, criterion_slots, criterion_intents, model, lang):
    """
    Evaluates the model on the provided data loader

    Args:
        data_loader (DataLoader): DataLoader for the evaluation dataset
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot tagging
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent classification
        model (nn.Module): The neural network model to evaluate
        lang (Lang): The language object containing mappings (id2intent, id2slot, tokenizer)

    Returns:
        tuple: A tuple containing:
            - dict: Slot filling evaluation results (e.g., F1 score from conll.evaluate)
            - dict: Intent classification evaluation results (e.g., accuracy, precision, recall from classification_report)
            - float: The average loss over the evaluation dataset
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0 # Initialize total loss accumulator
    
    # Lists to store ground truth and predicted intents and slots for evaluation metrics
    ref_intents = [] # Ground truth intents.
    hyp_intents = [] # Predicted intents.
    ref_slots = [] # List of lists of (token, slot_tag) tuples for conll format
    hyp_slots = [] # List of lists of (token, slot_tag) tuples for conll format

    # Disable gradient calculations during evaluation to save memory and computation
    with torch.no_grad():
        # Iterate over each batch of data
        for sample in data_loader:
            # Get the attention mask for the current batch
            attention_mask = sample['attention_mask']
            
            # Get model predictions
            slots, intents = model(sample['utterance'], attention_mask)
            
            # Compute losses for reference
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            total_loss += loss.item()

            # --- Intent Evaluation ---
            # Get predicted intent IDs (index of max logit)
            predicted_intent_ids = torch.argmax(intents, dim=1).tolist()
            # Get ground truth intent IDs
            true_intent_ids = sample['intent'].tolist()
            
            # Convert IDs to labels for comparison
            out_intents = [lang.id2intent.get(idx, 'UNK') for idx in predicted_intent_ids]
            gt_intents = [lang.id2intent.get(idx, 'UNK') for idx in true_intent_ids]
            
            # Add the predicted and ground truth intents for this batch to the overall lists
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # --- Slot Evaluation ---
            # Get predicted slot IDs per token 
            # Model output shape is assumed (batch_size, num_slot_classes, seq_len)
            predicted_slot_ids = torch.argmax(slots, dim=1) # Shape: (batch_size, seq_len)

            # Process each sequence in the batch
            for i in range(sample['utterance'].shape[0]):
                # Determine the actual length of the sequence by summing the attention mask for the current sample
                seq_len = int(torch.sum(attention_mask[i])) 
                
                if seq_len == 0: continue # Skip empty sequences

                # Get ground truth and predicted slots for this sequence
                current_gt_slots_ids = sample['slots'][i][:seq_len].tolist()
                current_pred_slots_ids = predicted_slot_ids[i][:seq_len].tolist()

                # Convert IDs back to slot labels
                # Use .get() with a default value for safety
                gt_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_gt_slots_ids]
                pred_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_pred_slots_ids]

                # Get corresponding BERT tokens IDs for this sequence
                current_utt_ids = sample['utterance'][i][:seq_len].tolist()
                # Convert token IDs back to token strings
                current_tokens = lang.tokenizer.convert_ids_to_tokens(current_utt_ids)

                # Prepare lists for the conll evaluation function, which expects (token, slot_label) pairs
                ref_slots_sequence = []
                hyp_slots_sequence = []
                
                # Iterate through tokens and their corresponding slots in the sequence
                for token_idx in range(seq_len):
                     token = current_tokens[token_idx]
                     gt_slot = gt_slots_labels[token_idx]
                     pred_slot = pred_slots_labels[token_idx]

                     # Skip CLS token if present and aligned
                     if token == lang.tokenizer.cls_token and gt_slot == lang.id2slot.get(lang.slot2id.get('O', PAD_TOKEN), 'O'):
                          continue
                     # Skip SEP token if present and aligned
                     if token == lang.tokenizer.sep_token and gt_slot == lang.id2slot.get(lang.slot2id.get('O', PAD_TOKEN), 'O'):
                          continue
                          
                     # Add word-slot pair if token is not a special token part like ##
                     ref_slots_sequence.append((token, gt_slot))
                     hyp_slots_sequence.append((token, pred_slot))
                
                # Append the formatted sentence slots to the overall lists
                ref_slots.append(ref_slots_sequence)
                hyp_slots.append(hyp_slots_sequence)

            

    # Calculate average loss over all samples/batches processed
    avg_loss = total_loss / len(data_loader) # Average loss per batch

    # Initialize results dictionary with default values
    results = {"total":{"f":0}}
    try:
        # Ensure ref_slots and hyp_slots are not empty before calling evaluate
        if ref_slots and hyp_slots:
             results = evaluate(ref_slots, hyp_slots) # Compute precision, recall, F1 for slots
        else:
             print("Warning: No slots data to evaluate.") 
    except Exception as ex:
        # Handle potential errors during evaluation
        print(f"Error during slot evaluation: {ex}")

    # Evaluate intents using classification report
    if not ref_intents or not hyp_intents:
        print("Warning: No intent data to evaluate.")
        report_intent = {'accuracy': 0} # Default values
    else:
        # Compute precision, recall, F1-score, and accuracy for intent classification
        report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)

    # Return slot evaluation results (dict), intent evaluation results (dict), average loss
    return results, report_intent, avg_loss
