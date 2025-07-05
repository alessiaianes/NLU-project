import torch
import torch.nn as nn
# Pack/Pad sequences might not be needed if using Hugging Face models directly
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
from conll import evaluate # Assuming this library is available and works as expected
from sklearn.metrics import classification_report
from utils import * # Import PAD_TOKEN, device, Lang etc.

def init_weights(mat):
    # Placeholder: BERT model weights are loaded from pretrained, no manual init needed usually
    pass

def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    total_loss = 0
    num_batches = 0
    # Use tqdm for progress bar if available and desired
    # from tqdm import tqdm
    # data_loader = tqdm(data_loader, desc="Training") 

    for sample in data_loader:
        optimizer.zero_grad()
        
        # Get attention mask from the sample batch
        attention_mask = sample['attention_mask']
        
        # Forward pass
        slots, intents = model(sample['utterance'], attention_mask)
        
        # Calculate losses
        # Ensure targets have the correct shape and dtype
        loss_intent = criterion_intents(intents, sample['intent'])
        
        # For CrossEntropyLoss with sequence data (slots):
        # Model output: (batch_size, num_classes, seq_len) -> permute in model did this
        # Target: (batch_size, seq_len)
        # Ensure sample['slots'] is LongTensor and has shape (batch_size, seq_len)
        loss_slot = criterion_slots(slots, sample['slots'])
        
        loss = loss_intent + loss_slot
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    # Return average loss per batch
    return total_loss / num_batches if num_batches > 0 else 0


def eval_loop(data_loader, criterion_slots, criterion_intents, model, lang):
    model.eval()
    total_loss = 0
    num_samples = 0 # Count samples processed
    
    ref_intents = []
    hyp_intents = []
    ref_slots = [] # List of lists of (token, slot_tag) tuples for conll format
    hyp_slots = [] # List of lists of (token, slot_tag) tuples for conll format

    # Use tqdm for progress bar if available
    # data_loader = tqdm(data_loader, desc="Evaluation")

    with torch.no_grad():
        for sample in data_loader:
            attention_mask = sample['attention_mask']
            
            # Forward pass
            slots, intents = model(sample['utterance'], attention_mask)
            
            # Calculate loss (optional, but useful for monitoring)
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
            
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # --- Slot Evaluation ---
            # Get predicted slot IDs per token (index of max logit for each token)
            # Model output shape is assumed (batch_size, num_slot_classes, seq_len)
            predicted_slot_ids = torch.argmax(slots, dim=1) # Shape: (batch_size, seq_len)

            # Process each sequence in the batch
            for i in range(sample['utterance'].shape[0]): # Iterate through batch dimension
                # Get the actual length of the sequence (excluding padding)
                seq_len = int(torch.sum(attention_mask[i])) 
                
                if seq_len == 0: continue # Skip empty sequences

                # Get ground truth and predicted slots for this sequence, truncated to seq_len
                current_gt_slots_ids = sample['slots'][i][:seq_len].tolist()
                current_pred_slots_ids = predicted_slot_ids[i][:seq_len].tolist()

                # Convert IDs back to slot labels
                # Use .get() with a default value (e.g., 'O' or PAD_TOKEN mapped label) for safety
                gt_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_gt_slots_ids]
                pred_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_pred_slots_ids]

                # Get corresponding BERT tokens for this sequence
                current_utt_ids = sample['utterance'][i][:seq_len].tolist()
                current_tokens = lang.tokenizer.convert_ids_to_tokens(current_utt_ids)

                # Format for conll.evaluate: list of (token, slot_label) tuples
                # Need to handle potential CLS/SEP tokens if they are included in seq_len calculation
                
                # Filter out special tokens if evaluate function doesn't expect them
                # or adjust indices accordingly. Let's assume evaluate handles standard BIO format.
                
                ref_slots_sequence = []
                hyp_slots_sequence = []
                
                # Iterate through tokens and their corresponding slots
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
                     # The conll evaluator usually expects alignment with original words, 
                     # this token-level alignment might need adjustment based on its expectations.
                     # If evaluate expects word-level, we'd need to aggregate subword slots.
                     # Assuming token-level evaluation for now:
                     ref_slots_sequence.append((token, gt_slot))
                     hyp_slots_sequence.append((token, pred_slot))
                
                # Append the formatted sentence slots to the overall lists
                ref_slots.append(ref_slots_sequence)
                hyp_slots.append(hyp_slots_sequence)

            # Update total loss and count based on batch size
            # Use batch size for loss averaging if needed, but iterating samples is safer
            # num_samples += sample['utterance'].shape[0] 

    # Calculate average loss over all samples/batches processed
    avg_loss = total_loss / len(data_loader) # Average loss per batch

    # Evaluate slots using conll format
    results = {"total":{"f":0}} # Default value
    try:
        # Ensure ref_slots and hyp_slots are not empty before calling evaluate
        if ref_slots and hyp_slots:
             results = evaluate(ref_slots, hyp_slots)
        else:
             print("Warning: No slots data to evaluate.")
    except Exception as ex:
        print(f"Error during slot evaluation: {ex}")
        # Fallback: calculate simple accuracy if conll fails
        # results = {"total":{"f":0}} # Keep default

    # Evaluate intents using classification report
    # Ensure lists are not empty
    if not ref_intents or not hyp_intents:
        print("Warning: No intent data to evaluate.")
        report_intent = {'accuracy': 0} # Default values
    else:
        report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)

    # Return slot evaluation results (dict), intent evaluation results (dict), average loss
    return results, report_intent, avg_loss
