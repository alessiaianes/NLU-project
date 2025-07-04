import json
from pprint import pprint
from collections import Counter
import os
import torch
import torch.utils.data as data
from transformers import AutoTokenizer # Keep using AutoTokenizer



# Define PAD_TOKEN globally (usually 0)
PAD_TOKEN = 0
# Determine the device (GPU if available, otherwise CPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
# Set CUDA_LAUNCH_BLOCKING for debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_data(path):
    """Loads data from a JSON file."""
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        # Handle the error appropriately, maybe exit or return empty list
        # For this context, exiting might be suitable if data is critical
        exit() 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file at {path}")
        exit()
    return dataset

# Load dataset globally (or ensure it's loaded before use)
# These paths assume a specific directory structure. Adjust if necessary.
try:
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    print(f'Successfully loaded {len(tmp_train_raw)} training samples and {len(test_raw)} test samples.')
    # Optional: Print a sample to verify structure
    # if tmp_train_raw:
    #     pprint(tmp_train_raw[0])
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


class Lang():
    """
    Manages vocabulary and mappings for intents and slots, 
    integrating with BERT tokenization.
    """
    def __init__(self, words, intents, slots, cutoff=0):
        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Map slots to IDs. PAD_TOKEN (0) is reserved for padding. 'O' tag gets the next ID.
        self.slot2id = self.lab2id(slots, pad=True) 
        # Map intents to IDs. Padding is not typically needed for intents.
        self.intent2id = self.lab2id(intents, pad=False)
        
        # Create inverse mappings (ID to label)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        """
        Maps a list of labels (elements) to integer IDs.
        Handles padding and the 'O' (Outside) tag specifically.
        
        Args:
            elements (list): A list of unique labels (strings).
            pad (bool): If True, reserves ID 0 for padding and assigns 'O' tag to ID 1.
                        If False, assigns IDs starting from 0.
        
        Returns:
            dict: A dictionary mapping labels to their assigned integer IDs.
        """
        vocab = {}
        idx_counter = 0
        
        # 1. Handle Padding Token
        if pad:
            vocab['pad'] = PAD_TOKEN  # Assign PAD_TOKEN (0) to the special 'pad' key
            idx_counter = PAD_TOKEN + 1 # Start subsequent IDs from 1

        # 2. Handle the 'O' (Outside) Tag
        o_tag = 'O'
        elements_set = set(elements) # Use a set for efficient lookup and uniqueness
        
        # Check if 'O' tag is present in the provided elements
        has_o = o_tag in elements_set
        
        # Assign ID to 'O' tag:
        if pad and has_o:
            # If padding is enabled and 'O' exists, assign it the next available ID (e.g., 1)
            vocab[o_tag] = idx_counter 
            idx_counter += 1
        elif has_o and not pad: 
             # If no padding, 'O' gets the first ID (0), unless 'pad' was already assigned 0.
             # This branch ensures 'O' gets ID 0 if padding is disabled.
             vocab[o_tag] = idx_counter # Assigns 0 if pad=False and no 'pad' key was added.
             idx_counter += 1
            
        # 3. Assign IDs to Remaining Labels
        # Filter out reserved keys ('pad', 'O') and sort the rest for consistent ID assignment.
        sorted_remaining_elements = sorted([el for el in elements_set if el != 'pad' and el != o_tag])
        
        for element in sorted_remaining_elements:
            # Assign the next available sequential ID
            vocab[element] = idx_counter
            idx_counter += 1
             
        return vocab

    def tokenize_utterance(self, utterance):
        """Tokenizes an utterance using the BERT tokenizer and returns IDs and offset mapping."""
        # `encode_plus` is preferred for getting multiple outputs like input_ids, token_type_ids, etc.
        # `return_offsets_mapping=True` is crucial for aligning labels with subword tokens.
        encoded_input = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,      # Add [CLS] and [SEP] tokens
            return_attention_mask=False,  # We generate this manually later if needed
            return_offsets_mapping=True,  # Returns start/end indices for each token in the original string
            truncation=False              # Do not truncate here; handle sequence length in batching/model if needed
        )
        return encoded_input['input_ids'], encoded_input['offset_mapping']

    def align_slot_labels_with_tokens(self, slots_str, utterance_str, utt_ids, offset_mapping):
        """
        Aligns slot labels (from original words) with BERT's subword tokens.
        
        Args:
            slots_str (str): The string of space-separated slot labels (e.g., "O B-LOC I-LOC").
            utterance_str (str): The original utterance string.
            utt_ids (list): The list of token IDs output by the BERT tokenizer.
            offset_mapping (list): List of (start, end) tuples for each token ID.
            
        Returns:
            list: A list of slot IDs corresponding to each BERT token ID.
        """
        original_words = utterance_str.split()
        original_slots = slots_str.split()
        
        # Initialize slot IDs list with the length of BERT tokens, using 'O' slot ID as default
        # Use .get() with a default value (e.g., PAD_TOKEN or the mapped 'O' ID) for safety
        o_slot_id = self.slot2id.get('O', PAD_TOKEN) # Get the ID for 'O', default to PAD_TOKEN if 'O' somehow missing
        bert_slot_ids = [o_slot_id] * len(utt_ids) 
        
        current_word_idx = 0 # Index for iterating through original words and slots
        
        # BERT's subword prefix is typically '##'
        subword_prefix = '##' 

        for token_idx, token_info in enumerate(offset_mapping):
            token_start, token_end = token_info

            # Skip CLS token ([CLS] usually maps to index 0)
            # Check if it's the first token and corresponds to the CLS token ID
            if token_idx == 0 and self.tokenizer.cls_token_id == utt_ids[token_idx]:
                 bert_slot_ids[token_idx] = o_slot_id # Assign 'O' slot to CLS token
                 continue

            # Skip SEP token ([SEP] usually maps to the last token)
            # Check if it's the last token and corresponds to the SEP token ID
            if token_idx == len(utt_ids) - 1 and self.tokenizer.sep_token_id == utt_ids[token_idx]:
                 bert_slot_ids[token_idx] = o_slot_id # Assign 'O' slot to SEP token
                 continue
            
            # Get the actual token string from its ID
            # convert_ids_to_tokens expects a list
            token_str = self.tokenizer.convert_ids_to_tokens([utt_ids[token_idx]])[0]
            
            # Check if the token is a continuation of a previous word (i.e., a subword)
            # This check assumes subwords start with '##'. Adjust if using a different BERT variant.
            is_subword_continuation = token_str.startswith(subword_prefix)

            if not is_subword_continuation:
                 # This token likely represents the start of a new original word.
                 # Assign the slot label corresponding to this original word.
                 if current_word_idx < len(original_slots):
                      slot_label = original_slots[current_word_idx]
                      bert_slot_ids[token_idx] = self.slot2id.get(slot_label, o_slot_id) # Use .get for safety
                 else:
                      # If we run out of original slots (shouldn't happen with correct alignment)
                      bert_slot_ids[token_idx] = o_slot_id 
                 
                 current_word_idx += 1 # Move to the next original word for subsequent tokens
            else:
                 # This token is a subword continuation of the *previous* original word.
                 # Assign the same slot ID as the beginning of that word.
                 prev_word_slot_idx = current_word_idx - 1 # Index of the word this subword belongs to
                 
                 # Check if the previous word index is valid
                 if prev_word_slot_idx >= 0 and prev_word_slot_idx < len(original_slots):
                      slot_label = original_slots[prev_word_slot_idx]
                      bert_slot_ids[token_idx] = self.slot2id.get(slot_label, o_slot_id)
                 else:
                      # Fallback if prev_word_slot_idx is invalid (e.g., subword belongs to CLS/SEP)
                      bert_slot_ids[token_idx] = o_slot_id
        
        # Final sanity check: Ensure the length of slot IDs matches the length of token IDs.
        # This is crucial for batching. If mismatch occurs, it indicates an alignment issue.
        if len(bert_slot_ids) != len(utt_ids):
            print(f"Warning: Slot ID length mismatch! Slots len: {len(bert_slot_ids)}, Utt IDs len: {len(utt_ids)}")
            # Truncate or pad the slot IDs to match the utterance IDs length.
            if len(bert_slot_ids) > len(utt_ids):
                bert_slot_ids = bert_slot_ids[:len(utt_ids)]
            else:
                # Pad with 'O' slot ID if utterance sequence is longer than slots (should ideally not happen)
                bert_slot_ids.extend([o_slot_id] * (len(utt_ids) - len(bert_slot_ids)))

        return bert_slot_ids


class IntentsAndSlotsBERT(data.Dataset):
    """
    PyTorch Dataset for Intent and Slot classification using BERT.
    Processes utterances and aligns slot labels to BERT tokens.
    """
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.lang = lang

        # Populate internal lists from the raw dataset
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots']) # Raw slot string like "O B-LOC I-LOC"
            self.intents.append(x['intent']) # Raw intent string like "flight-booking"

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single sample from the dataset.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing processed 'utterance' (token IDs), 
                  'slots' (aligned slot IDs), and 'intent' (intent ID).
        """
        utterance_str = self.utterances[idx]
        slots_str = self.slots[idx]
        intent_str = self.intents[idx]

        # Tokenize the utterance using the Lang object's tokenizer
        utt_ids, offset_mapping = self.lang.tokenize_utterance(utterance_str)
        
        # Align the raw slot labels with the tokenized sequence
        slot_ids = self.lang.align_slot_labels_with_tokens(slots_str, utterance_str, utt_ids, offset_mapping)
        
        # Get the integer ID for the intent
        intent_id = self.lang.intent2id.get(intent_str, self.lang.intent2id.get('UNK', PAD_TOKEN)) # Use fallback
        
        # Construct the sample dictionary
        sample = {'utterance': utt_ids, 'slots': slot_ids, 'intent': intent_id}
        return sample

def collate_fn_bert(data):
    """
    Custom collate function for creating batches of BERT data.
    Pads sequences to the same length and creates attention masks.
    
    Args:
        data (list): A list of samples (dictionaries) from the Dataset.
        
    Returns:
        dict: A dictionary containing batched tensors for 'utterance', 'intent', 
              'slots', and 'attention_mask'. Returns an empty dict if data is empty.
    """
    # Sort data by sequence length (utterance length) in descending order.
    # This is standard practice for sequence models, especially if using packing.
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    
    new_item = {}
    # Handle the edge case where the input data list might be empty
    if not data: 
        return {} # Return empty dictionary if no data
        
    # Use the keys from the first item to iterate, assuming all items have the same keys
    keys = data[0].keys()
    for key in keys:
        # Collect all values for a given key across the batch
        new_item[key] = [d[key] for d in data]

    # Helper function to merge sequences into padded tensors
    def merge(sequences):
        """Pads sequences to the maximum length in the batch."""
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        
        # Handle cases where sequences might be empty after processing
        if max_len == 0: 
             # Return empty tensor with shape (batch_size, 0) if max_len is 0
             return torch.empty((len(sequences), 0), dtype=torch.long), lengths
             
        # Create a tensor filled with PAD_TOKEN (0)
        # Shape: (batch_size, max_sequence_length)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        
        # Fill the tensor with the actual sequence data
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end > 0:
                # Convert list sequence to tensor before assigning
                padded_seqs[i, :end] = torch.LongTensor(seq) 
        return padded_seqs, lengths

    # Merge utterance IDs (input sequence)
    src_utt, lengths = merge(new_item['utterance'])
    # Merge slot IDs (target sequence)
    # Slot sequence length should match utterance sequence length after alignment
    y_slots, _ = merge(new_item["slots"]) 
    # Merge intent IDs (target scalar value per sample)
    intent = torch.LongTensor(new_item["intent"])

    # Move tensors to the specified device (GPU or CPU)
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)

    # Create attention mask: 1 indicates a real token, 0 indicates padding.
    # This mask is crucial for BERT to ignore padding tokens.
    attention_mask = (src_utt != PAD_TOKEN).float().to(device)

    # Update the batch dictionary with the processed tensors
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = y_slots
    new_item["attention_mask"] = attention_mask

    return new_item