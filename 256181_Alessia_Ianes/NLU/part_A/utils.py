import json
from pprint import pprint
from collections import Counter
import os
import torch
import torch.utils.data as data

PAD_TOKEN = 0
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

def load_data(path):
    """
    Loads a dataset from a JSON file

    Args:
        path (str): The file path to the JSON dataset

    Returns:
        list: A list of dictionaries, where each dictionary represents a data sample
    """
    dataset = [] # Initialize an empty list to store the dataset
    # Open the file in read mode
    with open(path) as f:
        # Read the entire file content and parse it as JSON
        dataset = json.loads(f.read())
    # Return the loaded dataset
    return dataset

# Load the training data
tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
# Load the test data
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

# --- Language Processing Class ---
class Lang():
    """
    Manages vocabulary creation and mappings between words, intents, and slots,
    and their corresponding numerical IDs
    """
    def __init__(self, words, intents, slots, cutoff=0):
        """
        Initializes the Lang object

        Args:
            words (list): A list of all words from the dataset
            intents (list): A list of all unique intent labels
            slots (list): A list of all unique slot tags
            cutoff (int): Minimum frequency threshold for a word to be included in the vocabulary. Defaults to 0 (include all words)
        """
        # Create word-to-ID mapping. Includes 'pad' and potentially 'unk' tokens
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        # Create slot-to-ID mapping
        self.slot2id = self.lab2id(slots)
        # Create intent-to-ID mapping
        self.intent2id = self.lab2id(intents, pad=False)
        # Create inverse mappings (ID-to-label/word) for easy decoding later
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        """
        Creates a word-to-ID mapping

        Args:
            elements (list): List of words
            cutoff (int): Minimum frequency threshold
            unk (bool): Whether to include an '<UNK>' token for unknown words

        Returns:
            dict: A dictionary mapping words to their unique IDs
        """
        vocab = {'pad': PAD_TOKEN} # Initialize vocabulary with padding token
        if unk:
             # Add 'unk' token if specified
            vocab['unk'] = len(vocab)
        # Count word frequencies
        count = Counter(elements)
        # Add words to vocabulary based on frequency cutoff
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab) # Assign the next available ID
        return vocab
    
    def lab2id(self, elements, pad=True):
        """
        Creates a label-to-ID mapping (for intents or slots)

        Args:
            elements (list): List of unique labels
            pad (bool): Whether to include a 'pad' token ID

        Returns:
            dict: A dictionary mapping labels to their unique IDs
        """
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        # Assign IDs to each unique label
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

# --- PyTorch Dataset Class ---
class IntentsAndSlots (data.Dataset):
    """
    PyTorch Dataset class to handle utterances and their corresponding intents and slots
    Prepares the data for batching and model input
    """
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        """
        Initializes the IntentsAndSlots dataset

        Args:
            dataset (list): The raw dataset (list of dictionaries)
            lang (Lang): The Lang object containing word/slot/intent mappings
            unk (str): The key representing the unknown token in the Lang vocabulary. Defaults to 'unk'
            """
        self.utterances = [] # List to store utterances (text)
        self.intents = [] # List to store intents (text)
        self.slots = [] # List to store slots (text sequences)
        self.unk = unk # Store the unknown token key
        
        # Populate the lists from the input dataset
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # Convert text sequences/labels into numerical IDs using the Lang object mapping
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        """ Returns the total number of samples in the dataset """
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample by index

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            dict: A dictionary containing the processed sample:
                  {'utterance': tensor, 'slots': tensor, 'intent': tensor}
        """
        # Retrieve the numerical IDs for utterance, slots, and intent for the given index
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        # Package them into a dictionary
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        """
        Maps a list of labels (like intents) to their IDs using a mapper dictionary
        Handles cases where a label might not be in the mapper by assigning the UNK token ID

        Args:
            data (list): List of labels (strings)
            mapper (dict): Dictionary mapping labels to IDs (e.g., lang.intent2id)

        Returns:
            list: List of corresponding numerical IDs
        """
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        """
        Maps sequences of tokens (like utterances or slots) to sequences of IDs
        Handles unknown words/slots by mapping them to the UNK token ID

        Args:
            data (list): List of sequences (strings where tokens are space-separated)
            mapper (dict): Dictionary mapping tokens to IDs (e.g., lang.word2id)

        Returns:
            list: A list of lists, where each inner list contains the numerical IDs for a sequence
        """
        res = [] # Initialize list to store sequences of IDs
        # Iterate through each sequence string in the input data
        for seq in data: 
            tmp_seq = [] # Temporary list for the current sequence's IDs
            # Split the sequence string into individual tokens
            for x in seq.split():
                # Map the token to its ID, or use the UNK token ID if not found
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            
            # Add the sequence of IDs to the result list
            res.append(tmp_seq)
        return res
    

def collate_fn(data):
    """
    Function used by PyTorch DataLoader to merge a list of samples into a batch
    It handles padding sequences to the same length and sorting the batch by sequence length
    for efficient processing with RNNs (especially when using pack_padded_sequence)

    Args:
        data (list): A list of samples fetched from the Dataset (each sample is a dict)

    Returns:
        dict: A dictionary containing the batch tensors (utterances, intents, slots, lengths),
              moved to the specified device
    """
    def merge(sequences):
        """
        Helper function to pad sequences to the maximum length within the batch

        Args:
            sequences (list): A list of sequences (tensors or lists)

        Returns:
            tuple: (padded_seqs, lengths)
                   - padded_seqs (torch.Tensor): Tensor with sequences padded to max_len
                   - lengths (list): List of the original lengths of each sequence
        """
        # Compute the lengths of all sequences in the batch
        lengths = [len(seq) for seq in sequences]
        # Find the maximum length in the batch
        max_len = 1 if max(lengths)==0 else max(lengths)
        
        # Create a tensor filled with PAD_TOKEN (0) with shape [batch_size, max_len]
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        # Copy each sequence into the padded tensor
        for i, seq in enumerate(sequences):
            end = lengths[i] # Get the length of the current sequence
            padded_seqs[i, :end] = seq # Copy the sequence data into the correct position in the padded tensor

        # Detach the tensor from the computation graph (as it's just padding)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    
    # Sort the data samples by utterance length in descending order
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    # Create a dictionary to hold the batch data, merging lists of samples
    new_item = {}
    # Iterate through the keys ('utterance', 'slots', 'intent') of the first sample
    # and create lists of corresponding items for the entire batch
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        

    # Merge and pad the utterances and slots sequences
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"]) # Get padded slots and their original lengths
    intent = torch.LongTensor(new_item["intent"]) # Convert the list of intent IDs into a LongTensor
    
    # Move the tensors and the lengths tensor to the pre-defined device
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    # Update the new_item dictionary with the processed tensors    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item