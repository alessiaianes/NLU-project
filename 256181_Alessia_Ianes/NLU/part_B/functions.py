# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from conll import evaluate
# from sklearn.metrics import classification_report
# from utils import *

# def init_weights(mat):
#     pass

# def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
#     model.train()
#     loss_array = []
#     for sample in data:
#         optimizer.zero_grad()
#         attention_mask = (sample['utterance'] != PAD_TOKEN).float()
#         slots, intent = model(sample['utterance'], attention_mask)
#         loss_intent = criterion_intents(intent, sample['intent'])
#         loss_slot = criterion_slots(slots, sample['slots'])
#         loss = loss_intent + loss_slot
#         loss_array.append(loss.item())
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#     return loss_array

# def eval_loop(data, criterion_slots, criterion_intents, model, lang):
#     model.eval()
#     loss_array = []
#     ref_intents = []
#     hyp_intents = []
#     ref_slots = []
#     hyp_slots = []

#     with torch.no_grad():
#         for sample in data:
#             attention_mask = (sample['utterance'] != PAD_TOKEN).float()
#             slots, intents = model(sample['utterance'], attention_mask)
#             loss_intent = criterion_intents(intents, sample['intent'])
#             loss_slot = criterion_slots(slots, sample['slots'])
#             loss = loss_intent + loss_slot
#             loss_array.append(loss.item())

#             out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
#             gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()]
#             ref_intents.extend(gt_intents)
#             hyp_intents.extend(out_intents)

#             output_slots = torch.argmax(slots, dim=1)
#             for id_seq, seq in enumerate(output_slots):
#                 length = len(seq)
#                 utt_ids = sample['utterance'][id_seq][:length].tolist()
#                 gt_ids = sample['slots'][id_seq].tolist()
#                 gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
#                 utterance = [lang.tokenizer.convert_ids_to_tokens(elem) for elem in utt_ids]
#                 to_decode = seq[:length].tolist()
#                 ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
#                 tmp_seq = []
#                 for id_el, elem in enumerate(to_decode):
#                     tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
#                 hyp_slots.append(tmp_seq)

#     try:
#         results = evaluate(ref_slots, hyp_slots)
#     except Exception as ex:
#         print("Warning:", ex)
#         ref_s = set([x[1] for x in ref_slots])
#         hyp_s = set([x[1] for x in hyp_slots])
#         print(hyp_s.difference(ref_s))
#         results = {"total":{"f":0}}

#     report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
#     return results, report_intent, loss_array


import torch
import torch.nn as nn
# Pack/Pad sequences might not be needed if using Hugging Face models directly
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
from conll import evaluate # Assuming this library is available and works as expected
from sklearn.metrics import classification_report
from utils import * # Import PAD_TOKEN, device, Lang etc.

import torch
import torch.nn as nn
# Removed unused imports like pack_padded_sequence
from conll import evaluate # Assuming this works correctly
from sklearn.metrics import classification_report
from utils import * # Import necessary variables and classes like PAD_TOKEN, device, Lang

def init_weights(mat):
    """Placeholder for weight initialization if needed (usually handled by pre-trained models)."""
    pass

def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """Performs one epoch of training."""
    model.train()
    total_loss = 0
    num_batches = 0

    for sample in data_loader:
        optimizer.zero_grad()
        
        attention_mask = sample['attention_mask']
        
        # Forward pass: Get slot and intent predictions
        slots, intents = model(sample['utterance'], attention_mask)
        
        # Calculate losses
        loss_intent = criterion_intents(intents, sample['intent'])
        # Ensure slots output shape is (batch_size, num_classes, seq_len) for CrossEntropyLoss
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
    """Performs evaluation on the given data loader."""
    model.eval()
    total_loss = 0
    
    ref_intents = [] # Ground truth intents
    hyp_intents = [] # Predicted intents
    ref_slots = []   # Ground truth slots (for conll format)
    hyp_slots = []   # Predicted slots (for conll format)

    with torch.no_grad():
        for sample in data_loader:
            attention_mask = sample['attention_mask']
            
            # Forward pass
            slots, intents = model(sample['utterance'], attention_mask)
            
            # Calculate loss (optional, for monitoring)
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            total_loss += loss.item()

            # --- Intent Evaluation ---
            predicted_intent_ids = torch.argmax(intents, dim=1).tolist()
            true_intent_ids = sample['intent'].tolist()
            
            # Convert IDs to labels
            out_intents = [lang.id2intent.get(idx, 'UNK') for idx in predicted_intent_ids]
            gt_intents = [lang.id2intent.get(idx, 'UNK') for idx in true_intent_ids]
            
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # --- Slot Evaluation ---
            # Predicted slot IDs per token: shape (batch_size, seq_len)
            predicted_slot_ids = torch.argmax(slots, dim=1) 

            # Process each sequence in the batch
            for i in range(sample['utterance'].shape[0]): # Iterate through batch dimension
                # Get the actual sequence length using the attention mask
                seq_len = int(torch.sum(attention_mask[i])) 
                
                # Skip if sequence length is 0
                if seq_len == 0: continue 

                # Get ground truth and predicted slots, truncated to seq_len
                current_gt_slots_ids = sample['slots'][i][:seq_len].tolist()
                current_pred_slots_ids = predicted_slot_ids[i][:seq_len].tolist()

                # Convert IDs back to slot labels, using 'O' as fallback
                gt_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_gt_slots_ids]
                pred_slots_labels = [lang.id2slot.get(sid, 'O') for sid in current_pred_slots_ids]

                # Get corresponding BERT tokens for this sequence
                current_utt_ids = sample['utterance'][i][:seq_len].tolist()
                # Convert token IDs back to strings
                current_tokens = lang.tokenizer.convert_ids_to_tokens(current_utt_ids)

                # Format for conll.evaluate: list of (token, slot_label) tuples
                ref_slots_sequence = []
                hyp_slots_sequence = []
                
                for token_idx in range(seq_len):
                    token = current_tokens[token_idx]
                    gt_slot = gt_slots_labels[token_idx]
                    pred_slot = pred_slots_labels[token_idx]

                    # Skip CLS/SEP tokens if they align with 'O' slot tag, assuming evaluator handles them
                    # Or filter them out if the evaluator expects only content words/slots
                    # Basic check: skip if token is special and slot is 'O'/'pad'
                    if token == lang.tokenizer.cls_token and gt_slot == 'O': continue
                    if token == lang.tokenizer.sep_token and gt_slot == 'O': continue
                    
                    # Append (token, slot) pair
                    ref_slots_sequence.append((token, gt_slot))
                    hyp_slots_sequence.append((token, pred_slot))
                
                # Append the formatted sentence slots to the overall lists
                ref_slots.append(ref_slots_sequence)
                hyp_slots.append(hyp_slots_sequence)

    # Calculate average loss per batch
    avg_loss = total_loss / len(data_loader) if data_loader else 0

    # Evaluate slots using conll format
    results = {"total":{"f":0}} # Default value if evaluation fails or data is empty
    if ref_slots and hyp_slots: # Ensure there's data to evaluate
        try:
            results = evaluate(ref_slots, hyp_slots)
        except Exception as ex:
            print(f"Error during slot evaluation: {ex}. Returning default results.")
            # results remains {"total":{"f":0}}
    else:
        print("Warning: No slot data available for evaluation.")

    # Evaluate intents using classification report
    report_intent = {'accuracy': 0} # Default value
    if ref_intents and hyp_intents:
        report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)
    else:
        print("Warning: No intent data available for evaluation.")

    # Return slot evaluation results, intent evaluation results, and average loss
    return results, report_intent, avg_loss