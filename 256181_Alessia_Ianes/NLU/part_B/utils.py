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

PAD_TOKEN = 0
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
except FileNotFoundError:
    print("Error: Dataset files not found. Make sure 'dataset/ATIS/train.json' and 'dataset/ATIS/test.json' exist.")
    exit()


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Call lab2id with default pad=True for slots
        self.slot2id = self.lab2id(slots, pad=True) 
        # Call lab2id with pad=False for intents
        self.intent2id = self.lab2id(intents, pad=False)
        
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
        # --- REMOVED THE ERRONEOUS 'if pad' BLOCK ---
        # The logic for mapping PAD_TOKEN is handled correctly within lab2id.
        # The variable 'pad' was not defined in this scope.
        # --- END REMOVAL ---

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN 
            # Ensure 'O' is considered for slot mapping, often it's the default non-slot tag
            if 'O' not in elements:
                 elements.append('O') 
            
        # Sort elements for consistent ID assignment, though not strictly necessary here
        sorted_elements = sorted(list(elements))

        current_id = PAD_TOKEN + 1 if pad else 0 # Start assigning IDs after PAD_TOKEN (0)
        for elem in sorted_elements:
             if elem == 'pad': continue # Skip the special 'pad' entry itself
             
             # Map 'O' tag to PAD_TOKEN if padding is enabled and 'O' is encountered
             if pad and elem == 'O':
                  if 'O' not in vocab: # Assign 'O' to PAD_TOKEN if not already done
                       vocab[elem] = PAD_TOKEN
             elif elem not in vocab: # Assign next available ID to other elements
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
        # WARNING: Without truncation=True, sequences longer than the model's max input length 
        # (e.g., 512 for BERT) will NOT be automatically truncated. This could lead to errors
        # or incorrect alignment if not handled elsewhere.
        encoded_input = self.lang.tokenizer.encode_plus(
            utt_string,
            add_special_tokens=True,
            return_attention_mask=False, 
            return_offsets_mapping=True, 
            # --- REMOVED AS PER YOUR REQUEST ---
            # truncation=True, 
            # max_length=512 
            # --- END REMOVAL ---
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
        bert_slot_ids = [lang.slot2id.get('O', PAD_TOKEN)] * len(utt_ids) 
        
        current_word_idx = 0 # Tracks the index of the current word in the original utterance
        
        # BERT's subword prefix is '##'
        subword_prefix = '##' 

        for token_idx, token_info in enumerate(offset_mapping):
            token_start, token_end = token_info

            # Skip CLS token if it's the first token and matches the CLS ID
            if token_idx == 0 and lang.tokenizer.cls_token_id == utt_ids[token_idx]:
                 bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) # Assign 'O' slot to CLS
                 continue

            # Skip SEP token if it's the last token and matches the SEP ID
            if token_idx == len(utt_ids) - 1 and lang.tokenizer.sep_token_id == utt_ids[token_idx]:
                 bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) # Assign 'O' slot to SEP
                 continue
            
            # Get the actual token string from its ID
            token_str = lang.tokenizer.convert_ids_to_tokens([utt_ids[token_idx]])[0]
            
            # Check if the token starts with the BERT subword prefix '##'
            is_subword_continuation = token_str.startswith(subword_prefix)

            if not is_subword_continuation:
                 # This token represents the start of a new original word.
                 # Assign the slot label corresponding to this word.
                 if current_word_idx < len(original_slots):
                      bert_slot_ids[token_idx] = lang.slot2id.get(original_slots[current_word_idx], PAD_TOKEN)
                 else:
                      # If we run out of original slots (shouldn't happen with correct alignment)
                      bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN) 
                 
                 current_word_idx += 1 # Move to the next original word for subsequent tokens
            else:
                 # This token is a subword continuation of the *previous* original word.
                 # Assign the same slot ID as the beginning of that word.
                 prev_word_slot_idx = current_word_idx - 1 # Index of the word this subword belongs to
                 if prev_word_slot_idx >= 0 and prev_word_slot_idx < len(original_slots):
                      bert_slot_ids[token_idx] = lang.slot2id.get(original_slots[prev_word_slot_idx], PAD_TOKEN) 
                 else:
                      # Fallback if prev_word_slot_idx is invalid
                      bert_slot_ids[token_idx] = lang.slot2id.get('O', PAD_TOKEN)
        
        # Final check for length consistency. This is crucial.
        if len(bert_slot_ids) != len(utt_ids):
            print(f"Warning: Slot ID length mismatch! Slots: {len(bert_slot_ids)}, Utt: {len(utt_ids)}")
            # Truncate or pad the slot IDs to match the utterance IDs length.
            # This ensures the tensors have compatible shapes for batching.
            if len(bert_slot_ids) > len(utt_ids):
                bert_slot_ids = bert_slot_ids[:len(utt_ids)]
            else:
                # Pad with 'O' slot if utterance is longer than slots (shouldn't happen with correct logic)
                bert_slot_ids.extend([lang.slot2id.get('O', PAD_TOKEN)] * (len(utt_ids) - len(bert_slot_ids)))

        return bert_slot_ids


def collate_fn_bert(data):
    # Sort data by sequence length (utterance length) in descending order
    # This is useful for potential packing/padding strategies, though not strictly required here if attention mask is used.
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    
    new_item = {}
    # Handle the case where data might be empty
    if not data: 
        return {}
        
    # Use the keys from the first item to iterate, assuming all items have the same keys
    keys = data[0].keys()
    for key in keys:
        new_item[key] = [d[key] for d in data]

    # Helper function to merge sequences into padded tensors
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        
        # Handle cases where sequences might be empty after processing
        if max_len == 0: 
             return torch.empty((len(sequences), 0), dtype=torch.long), lengths
             
        # Create a tensor filled with PAD_TOKEN (0)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end > 0:
                # Fill the tensor with sequence data
                padded_seqs[i, :end] = torch.LongTensor(seq) 
        return padded_seqs, lengths

    # Merge utterance IDs (input sequence)
    src_utt, lengths = merge(new_item['utterance'])
    # Merge slot IDs (target sequence)
    y_slots, _ = merge(new_item["slots"]) # Lengths are the same as src_utt due to alignment
    # Merge intent IDs (target scalar)
    intent = torch.LongTensor(new_item["intent"])

    # Move tensors to the specified device (CPU or GPU)
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)

    # Create attention mask: 1 for real tokens, 0 for padding tokens
    attention_mask = (src_utt != PAD_TOKEN).float().to(device)

    # Update new_item dictionary with the processed tensors
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = y_slots
    new_item["attention_mask"] = attention_mask

    return new_item