# import json
# from pprint import pprint
# from collections import Counter
# import os
# import torch
# import torch.utils.data as data
# from transformers import BertTokenizer

# PAD_TOKEN = 0
# device = 'cuda:0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# def load_data(path):
#     dataset = []
#     with open(path) as f:
#         dataset = json.loads(f.read())
#     return dataset

# tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
# test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
# print('Train samples:', len(tmp_train_raw))
# print('Test samples:', len(test_raw))
# pprint(tmp_train_raw[0])

# class Lang():
#     def __init__(self, words, intents, slots, cutoff=0):
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.slot2id = self.lab2id(slots)
#         self.intent2id = self.lab2id(intents, pad=False)
#         self.id2slot = {v:k for k, v in self.slot2id.items()}
#         self.id2intent = {v:k for k, v in self.intent2id.items()}
        

#     def lab2id(self, elements, pad=True):
#         vocab = {}
#         if pad:
#             vocab['pad'] = PAD_TOKEN
#         for elem in elements:
#             vocab[elem] = len(vocab)
#         return vocab

# class IntentsAndSlotsBERT(data.Dataset):
#     def __init__(self, dataset, lang):
#         self.utterances = []
#         self.intents = []
#         self.slots = []
#         self.lang = lang

#         for x in dataset:
#             self.utterances.append(x['utterance'])
#             self.slots.append(x['slots'])
#             self.intents.append(x['intent'])

#     def __len__(self):
#         return len(self.utterances)

#     def __getitem__(self, idx):
#         utt = self.utterances[idx]
#         slots = self.slots[idx]
#         intent = self.intents[idx]
#         utt_ids = self.lang.tokenizer.encode(utt, add_special_tokens=True)
#         slot_ids = self.align_slot_labels_with_tokens(slots, utt_ids, self.lang.tokenizer)
#         intent_id = self.lang.intent2id[intent]
#         sample = {'utterance': utt_ids, 'slots': slot_ids, 'intent': intent_id}
#         return sample

#     def align_slot_labels_with_tokens(self, slots, utt_ids, tokenizer):
#         orig_tokens = slots.split()
#         orig_slots = slots.split()
#         bert_tokens = []
#         bert_slot_labels = []

#         for orig_token, orig_slot in zip(orig_tokens, orig_slots):
#             sub_tokens = tokenizer.tokenize(orig_token)
#             bert_tokens.extend(sub_tokens)
#             bert_slot_labels.extend([orig_slot] + ['X'] * (len(sub_tokens) - 1))

#         bert_slot_ids = [self.lang.slot2id.get(slot, PAD_TOKEN) for slot in bert_slot_labels]
#         return bert_slot_ids
    
# def collate_fn_bert(data):
#     def merge(sequences):
#         '''
#         merge from batch * sent_len to batch * max_len 
#         '''
#         lengths = [len(seq) for seq in sequences]
#         max_len = 1 if max(lengths)==0 else max(lengths)
#         # Pad token is zero in our case
#         # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
#         # batch_size X maximum length of a sequence
#         padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             if end > 0:
#                 padded_seqs[i, :end] = torch.tensor(seq, dtype=torch.long)  # Converti la lista in tensore
#         # print(padded_seqs)
#         padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
#         return padded_seqs, lengths

#     # Sort data by seq lengths
#     data.sort(key=lambda x: len(x['utterance']), reverse=True) 
#     new_item = {}
#     for key in data[0].keys():
#         new_item[key] = [d[key] for d in data]
#     # We just need one length for packed pad seq, since len(utt) == len(slots)
#     src_utt, _ = merge(new_item['utterance'])
#     y_slots, _ = merge(new_item["slots"])
#     intent = torch.LongTensor(new_item["intent"])
#     src_utt = src_utt.to(device) # We load the Tensor on our selected device
#     y_slots = y_slots.to(device)
#     intent = intent.to(device)
#     attention_mask = (src_utt != PAD_TOKEN).float().to(device)  # Creazione dell'attention_mask
#     new_item["utterance"] = src_utt
#     new_item["intent"] = intent
#     new_item["slots"] = y_slots
#     new_item["attention_mask"] = attention_mask  # Aggiunta dell'attention_mask al nuovo item
#     return new_item

import json
from pprint import pprint
from collections import Counter
import os
import torch
import torch.utils.data as data
from transformers import AutoTokenizer # Keep using AutoTokenizer


# Ensure PAD_TOKEN and device are defined globally or passed appropriately
PAD_TOKEN = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_data(path):
    """Loads data from a JSON file."""
    try:
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {path}")
        exit()

# Load data globally (or ensure it's loaded before Lang/Dataset are instantiated)
try:
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


class Lang():
    """Manages vocabulary, tokenization, and label mappings."""
    def __init__(self, words, intents, slots, cutoff=0):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create slot and intent mappings
        self.slot2id = self.lab2id(slots, pad=True) 
        self.intent2id = self.lab2id(intents, pad=False)
        
        # Create reverse mappings
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
        # Store PAD_TOKEN ID for potential use elsewhere
        self.pad_token_id = PAD_TOKEN

    def lab2id(self, elements, pad=True):
        """Creates a mapping from labels (slots/intents) to IDs."""
        vocab = {}
        # Assign PAD_TOKEN (0) to 'pad' and 'O' if padding is enabled
        if pad:
            vocab['pad'] = PAD_TOKEN
            # Ensure 'O' (often the default non-slot tag) maps to PAD_TOKEN
            if 'O' not in vocab:
                 vocab['O'] = PAD_TOKEN 
            
        current_id = 1 # Start assigning IDs from 1 (since 0 is PAD_TOKEN)
        
        # Sort elements for consistent ID assignment, excluding 'pad' and 'O'
        sorted_elements = sorted([elem for elem in elements if elem not in ['pad', 'O']])
        
        for elem in sorted_elements:
            # Assign the next available ID if the element isn't already mapped
            if elem not in vocab: 
                vocab[elem] = current_id
                current_id += 1
        return vocab

class IntentsAndSlotsBERT(data.Dataset):
    """Custom Dataset for BERT-based intent and slot tagging."""
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.lang = lang

        # Store original data
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots']) # String like "O B-flight-from O ..."
            self.intents.append(x['intent'])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.utterances)

    def __getitem__(self, idx):
        """Generates one sample (tokenized utterance, aligned slots, intent)."""
        utt_string = self.utterances[idx]
        slots_str = self.slots[idx]
        intent_str = self.intents[idx]

        # Tokenize utterance using BERT tokenizer
        # IMPORTANT: No truncation is applied as per requirement. Handle potential long sequences.
        encoded_input = self.lang.tokenizer.encode_plus(
            utt_string,
            add_special_tokens=True,          # Add [CLS] and [SEP]
            return_attention_mask=False,      # We create our own attention mask later if needed
            return_offsets_mapping=True,      # Crucial for aligning slots
            truncation=False                  # Explicitly disable truncation
        )
        
        utt_ids = encoded_input['input_ids']
        offset_mapping = encoded_input['offset_mapping'] # List of (start, end) byte offsets

        # Align slot labels to the tokenized sequence
        slot_ids = self.align_slot_labels_with_tokens(slots_str, utt_string, utt_ids, offset_mapping, self.lang)
        
        # Get intent ID
        intent_id = self.lang.intent2id.get(intent_str, self.lang.intent2id.get('pad', PAD_TOKEN)) # Use 'pad' as fallback
        
        sample = {'utterance': utt_ids, 'slots': slot_ids, 'intent': intent_id}
        return sample

    def align_slot_labels_with_tokens(self, slots_str, utt_string, utt_ids, offset_mapping, lang):
        """Aligns original slot tags to BERT subword tokens."""
        
        original_slots = slots_str.split()
        
        # Initialize slot IDs list with PAD_TOKEN for all BERT tokens
        bert_slot_ids = [lang.slot2id.get('pad', PAD_TOKEN)] * len(utt_ids) 
        
        original_word_idx = 0 # Index for tracking the current original word/slot
        
        # Iterate through the tokens and their offset mappings
        for token_idx, token_info in enumerate(offset_mapping):
            token_start, token_end = token_info
            
            # Skip if token has zero length (can happen with special tokens sometimes)
            if token_start == token_end and token_idx != 0 and token_idx != len(utt_ids) - 1: 
                continue

            # Handle Special Tokens ([CLS], [SEP]) - assign 'pad' slot ID ('O')
            if token_idx == 0 or token_idx == len(utt_ids) - 1:
                # Ensure the slot ID for CLS/SEP is the PAD_TOKEN ID (mapped from 'pad'/'O')
                bert_slot_ids[token_idx] = lang.slot2id.get('pad', PAD_TOKEN) 
                continue # Move to the next token
                 
            # Check if we have processed all original slots
            if original_word_idx >= len(original_slots):
                 # Assign 'pad' slot ID if we run out of original slots prematurely
                 bert_slot_ids[token_idx] = lang.slot2id.get('pad', PAD_TOKEN)
                 continue

            # Get the current original slot tag
            current_original_slot = original_slots[original_word_idx]
            
            # Assign the slot ID to the current BERT token
            bert_slot_ids[token_idx] = lang.slot2id.get(current_original_slot, lang.slot2id.get('pad', PAD_TOKEN))
            
            # Determine if we should advance to the next original slot
            # Advance if the current token is the end of its original word span
            is_last_subword = True # Assume it's the last subword by default
            if token_idx + 1 < len(offset_mapping):
                next_token_start, _ = offset_mapping[token_idx + 1]
                # If the next token starts exactly where the current one ends, it's part of the same word
                if next_token_start == token_end: 
                    is_last_subword = False
            
            # If this token marks the end of an original word, advance the original word index
            if is_last_subword:
                original_word_idx += 1

        # Final sanity check: Ensure lengths match
        if len(bert_slot_ids) != len(utt_ids):
             # Pad or truncate bert_slot_ids to match utt_ids length
             if len(bert_slot_ids) > len(utt_ids):
                 bert_slot_ids = bert_slot_ids[:len(utt_ids)]
             else:
                 bert_slot_ids.extend([lang.slot2id.get('pad', PAD_TOKEN)] * (len(utt_ids) - len(bert_slot_ids)))

        return bert_slot_ids


def collate_fn_bert(data):
    """
    Collate function for creating batches of BERT-formatted data.
    Pads sequences to the maximum length in the batch.
    """
    # Handle empty data case
    if not data:
        return {}
        
    # Sort data by sequence length (utterance length) in descending order
    # Useful for potential packed sequences, though not strictly required here with attention masks
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    
    new_item = {}
    # Use keys from the first item (assuming all items have the same structure)
    keys = data[0].keys()
    for key in keys:
        new_item[key] = [d[key] for d in data]

    # Helper function to merge sequences into padded tensors
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        
        # Return empty tensors if max_len is 0
        if max_len == 0: 
             return torch.empty((len(sequences), 0), dtype=torch.long), lengths
             
        # Create padded tensor initialized with PAD_TOKEN (0)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end > 0:
                # Fill tensor with actual sequence data
                padded_seqs[i, :end] = torch.LongTensor(seq) 
        return padded_seqs, lengths

    # Merge utterance IDs, slot IDs, and intent IDs
    src_utt, _ = merge(new_item['utterance'])
    y_slots, _ = merge(new_item["slots"]) 
    intent = torch.LongTensor(new_item["intent"])

    # Move tensors to the specified device
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (src_utt != PAD_TOKEN).float().to(device)

    # Update dictionary with processed tensors
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = y_slots
    new_item["attention_mask"] = attention_mask

    return new_item