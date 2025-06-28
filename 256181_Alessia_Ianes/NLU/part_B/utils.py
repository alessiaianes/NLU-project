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
from transformers import BertTokenizer

PAD_TOKEN = 0
# Consider using 'cuda' if CUDA is available, otherwise 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

# Load data globally (or pass it around)
try:
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))
    # print(tmp_train_raw[0]) # Uncomment to see structure
except FileNotFoundError:
    print("Error: Dataset files not found. Make sure 'dataset/ATIS/train.json' and 'dataset/ATIS/test.json' exist.")
    # Handle error appropriately, maybe exit or use dummy data
    exit()


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Add special tokens to padding index if not already done by tokenizer
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        # Ensure PAD_TOKEN is mapped to 0 for slot2id if not present
        if 'O' in self.slot2id: # Assuming 'O' is the default non-slot tag
             self.slot2id['O'] = PAD_TOKEN # Ensure 'O' maps to 0 if it exists
        elif PAD_TOKEN not in self.slot2id.values():
             # If 'O' isn't there and 0 isn't used, map PAD_TOKEN explicitly
             # This part might need adjustment based on actual slot names
             pass 


    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            # Ensure PAD_TOKEN is 0 if padding is enabled
            vocab['pad'] = PAD_TOKEN 
            # Also map the 'O' tag (often used for non-slot tokens) to PAD_TOKEN
            # Make sure 'O' is processed correctly, might need to add it explicitly if not in elements
            if 'O' not in elements:
                 elements.append('O') # Ensure 'O' is considered
            
        # Sort elements to ensure consistent ID assignment if needed, though order doesn't strictly matter here
        sorted_elements = sorted(list(elements))

        current_id = PAD_TOKEN + 1 if pad else 0
        for elem in sorted_elements:
             if elem == 'pad': continue # Skip the padding entry itself
             # Map 'O' tag to PAD_TOKEN if padding is enabled
             if pad and elem == 'O':
                  if 'O' not in vocab: # Assign 'O' to PAD_TOKEN if not already done
                       vocab[elem] = PAD_TOKEN
             elif elem not in vocab: # Assign next available ID
                  vocab[elem] = current_id
                  current_id += 1
        return vocab

class IntentsAndSlotsBERT(data.Dataset):
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.lang = lang

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots']) # This is the string like "O B-flight-from O ..."
            self.intents.append(x['intent'])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt_string = self.utterances[idx]
        slots_str = self.slots[idx]
        intent_str = self.intents[idx]

        # Tokenize utterance WITH special tokens (like [CLS], [SEP])
        encoded_input = self.lang.tokenizer.encode_plus(
            utt_string,
            add_special_tokens=True,
            return_attention_mask=False, # We create it later if needed
            return_offsets_mapping=True, # Get token character offsets for alignment
            truncation=True, # Truncate sequences longer than model max length
            max_length=512 # Default BERT max length
        )
        
        utt_ids = encoded_input['input_ids']
        offset_mapping = encoded_input['offset_mapping'] # List of (start, end) tuples

        # Align slots to the tokenized sequence
        slot_ids = self.align_slot_labels_with_tokens(slots_str, utt_string, utt_ids, offset_mapping, self.lang)
        
        intent_id = self.lang.intent2id[intent_str]
        
        sample = {'utterance': utt_ids, 'slots': slot_ids, 'intent': intent_id}
        return sample

    def align_slot_labels_with_tokens(self, slots_str, utt_string, utt_ids, offset_mapping, lang):
        """Aligns slot labels with BERT subword tokens."""
        original_words = utt_string.split()
        original_slots = slots_str.split()
        
        # Initialize slot IDs list matching the length of BERT tokens (utt_ids)
        bert_slot_ids = [lang.slot2id.get('O', PAD_TOKEN)] * len(utt_ids) # Default to 'O'/'PAD'
        
        current_word_idx = 0
        slot_idx_in_original = 0

        for token_idx, token_info in enumerate(offset_mapping):
            token_start, token_end = token_info
            
            # Skip CLS token if present
            if token_idx == 0 and lang.tokenizer.cls_token_id in utt_ids:
                 bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) # Assign 'O' slot to CLS
                 continue

            # Skip SEP token if present
            if token_idx == len(utt_ids) - 1 and lang.tokenizer.sep_token_id in utt_ids:
                 bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) # Assign 'O' slot to SEP
                 continue
            
            # Find which original word this token belongs to
            # This assumes tokens derived from the same word are contiguous
            # Find the word that this token's start offset falls into
            word_found = False
            for i in range(current_word_idx, len(original_words)):
                 # Rough check: does the token start within the character range of the word?
                 # This is simplified; a precise check would compare token_start against word start/end chars
                 
                 # A better approach: Check if the token is NOT a subword continuation
                 token_str = lang.tokenizer.convert_ids_to_tokens([utt_ids[token_idx]])[0]
                 is_subword_continuation = token_str.startswith(lang.tokenizer.prefix_tokens_map.get('subword', '##'))

                 if not is_subword_continuation:
                     # This token corresponds to the start of original_words[i]
                     if i < len(original_slots):
                          bert_slot_ids[token_idx] = lang.slot2id.get(original_slots[i], PAD_TOKEN)
                     else:
                          bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) # Default if out of bounds
                     
                     current_word_idx = i + 1 # Move to the next word for subsequent tokens
                     word_found = True
                     break # Found the word for this token
                 else:
                     # This token is a subword continuation of the previous word
                     # Assign the *same* slot ID as the start of the word
                     # Use the slot from the *previous* word index
                     prev_word_slot_idx = i - 1 # The word this subword belongs to
                     if prev_word_slot_idx >= 0 and prev_word_slot_idx < len(original_slots):
                          # Use the actual slot tag for the word, or 'X' if preferred for subwords
                          bert_slot_ids[token_idx] = lang.slot2id.get(original_slots[prev_word_slot_idx], PAD_TOKEN) 
                          # Alternative: Use a specific ID for subwords, e.g., map 'X' to PAD_TOKEN
                          # bert_slot_ids[token_idx] = lang.slot2id.get('X', PAD_TOKEN) 
                     else:
                          bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN)
                     # Don't increment current_word_idx here, stay on the same word

            # If no word was found (e.g., empty input or unexpected tokenization)
            if not word_found:
                 bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN)
                 
        # Final check for length consistency (should ideally match due to offset mapping)
        if len(bert_slot_ids) != len(utt_ids):
            print(f"Warning: Slot ID length mismatch! Slots: {len(bert_slot_ids)}, Utt: {len(utt_ids)}")
            # Truncate or pad if necessary, though the logic above should prevent this
            if len(bert_slot_ids) > len(utt_ids):
                bert_slot_ids = bert_slot_ids[:len(utt_ids)]
            else:
                bert_slot_ids.extend([lang.slot2id.get('O', PAD_TOKEN)] * (len(utt_ids) - len(bert_slot_ids)))

        return bert_slot_ids


def collate_fn_bert(data):
    # data is a list of dictionaries, e.g., [{'utterance': [...], 'slots': [...], 'intent': ...}]
    
    # Sort data by sequence length (utterance length) in descending order
    # This is useful for potential packing later, though not strictly required here
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    
    # Prepare dictionaries for different fields
    new_item = {}
    # Use the keys from the first item to iterate, assuming all items have the same keys
    if not data: # Handle empty data case
        return {}
        
    keys = data[0].keys()
    for key in keys:
        new_item[key] = [d[key] for d in data]

    # Pad sequences
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        if max_len == 0: # Handle sequences that are empty after tokenization
             return torch.empty((len(sequences), 0), dtype=torch.long), lengths
             
        # Create a tensor filled with PAD_TOKEN
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end > 0:
                # Use LongTensor for the sequence data
                padded_seqs[i, :end] = torch.LongTensor(seq) 
        # Detach is generally not needed here as tensors are created newly
        return padded_seqs, lengths

    # Merge utterance IDs (input sequence)
    src_utt, lengths = merge(new_item['utterance'])
    # Merge slot IDs (target sequence)
    y_slots, _ = merge(new_item["slots"]) # Lengths are the same as src_utt due to alignment
    # Merge intent IDs (target scalar)
    intent = torch.LongTensor(new_item["intent"])

    # Move tensors to the specified device
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)

    # Create attention mask: 1 for real tokens, 0 for padding tokens
    attention_mask = (src_utt != PAD_TOKEN).float().to(device)

    # Update new_item with padded tensors and attention mask
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = y_slots
    new_item["attention_mask"] = attention_mask
    # Optionally add lengths if needed by the model/loss, but attention_mask is usually sufficient
    # new_item["lengths"] = lengths 

    return new_item