import torch
import torch.utils.data as data

# File Reading Function
def read_file(path, eos_token="<eos>"):
    """
    Reads a text file line by line, strips leading/trailing whitespace from each line,
    and appends a specified end-of-sentence (EOS) token

    Args:
        path (str): The file path to the text corpus
        eos_token (str, optional): The token to append to the end of each line. Defaults to "<eos>"

    Returns:
        list[str]: A list where each element is a processed line (sentence) from the file
    """
    output = []
    with open(path, "r") as f: # Open the file in read mode
        for line in f.readlines():
            # Strip leading/trailing whitespace and append EOS token
            output.append(line.strip() + " " + eos_token)
    return output



# Vocabulary Creation Function
def get_vocab(corpus, special_tokens=[]):
    """
    Builds a vocabulary mapping words to unique integer IDs

    Args:
        corpus (list[str]): A list of sentences (strings).
        special_tokens (list[str], optional): A list of special tokens (e.g., padding, unknown) 
                                              to be included in the vocabulary with initial IDs
                                              Defaults to an empty list

    Returns:
        dict: A dictionary mapping each unique word to its assigned integer ID
    """
    output = {}
    i = 0 
    # Assign IDs to any specified special tokens first
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus: # Split the sentence into words
        for w in sentence.split(): # Split the sentence into words
            # If the word hasn't been seen before, assign it the next available ID
            if w not in output:
                output[w] = i
                i += 1
    return output



# This class computes and stores our vocab 
# Word to ids and ids to word
class Lang():
    """
    Represents the vocabulary of the language corpus
    Provides mappings between words and their corresponding integer IDs
    """
    def __init__(self, corpus, special_tokens=[]):
        """
        Initializes the Lang object by building the vocabulary

        Args:
            corpus (list[str]): The list of sentences from the corpus
            special_tokens (list[str], optional): List of special tokens to include. Defaults to []
        """
        # Build the word-to-ID mapping using the internal get_vocab method
        self.word2id = self.get_vocab(corpus, special_tokens)
        # Create the reverse mapping (ID-to-word) from the word-to-ID mapping
        self.id2word = {v:k for k, v in self.word2id.items()}

    
    def get_vocab(self, corpus, special_tokens=[]):
        """
        Builds the word-to-ID mapping. (Internal helper method, identical to the standalone get_vocab function)

        Args:
            corpus (list[str]): The list of sentences
            special_tokens (list[str], optional): List of special tokens. Defaults to []

        Returns:
            dict: The word-to-ID mapping dictionary
        """
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    


class PennTreeBank (data.Dataset):
    """
    PyTorch Dataset class for the Penn Treebank dataset
    Prepares source (input) and target (output) sequences for language modeling
    Input sequences are fed into the model, and target sequences are the ground truth labels
    """
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        """
        Initializes the dataset

        Args:
            corpus (list[str]): The raw sentences from the corpus
            lang (Lang): The Lang object containing vocabulary mappings
        """
        self.source = []
        self.target = []
        
        # Process each sentence in the corpus
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
        
        # Convert the token sequences into sequences of integer IDs using the provided Lang object
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        """
        Returns the total number of sequences (sentences) in the dataset
        """
        return len(self.source)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample (source and target sequences) by index

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            dict: A dictionary containing:
                  - 'source': A LongTensor of source sequence IDs
                  - 'target': A LongTensor of target sequence IDs
        """
        # Convert the list of IDs for source and target to PyTorch Tensors
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        # Create a dictionary containing the source and target tensors
        sample = {'source': src, 'target': trg}
        return sample
    

    # Auxiliary methods
    def mapping_seq(self, data, lang): 
        """
        Maps a list of word sequences to a list of ID sequences using the Lang object

        Args:
            data (list[list[str]]): A list where each element is a sequence of words
            lang (Lang): The Lang object containing word-to-ID mapping

        Returns:
            list[list[int]]: A list where each element is a sequence of integer IDs
                             Prints a warning if an Out-Of-Vocabulary (OOV) word is encountered
        """
        res = []
        # Iterate through each word sequence in the input data
        for seq in data:
            tmp_seq = []
            # Iterate through each word in the current sequence
            for x in seq:
                # If word is in vocabulary, append its ID
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    # Handle Out-Of-Vocabulary (OOV) words
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break # Exit the inner loop for this sequence
            res.append(tmp_seq)
        return res
    

def collate_fn(data, pad_token, DEVICE):
    """
    Custom collate function for PyTorch DataLoader
    It pads sequences within a batch to the same length and moves data to the specified device
    It also sorts the sequences by length in descending order, which is useful for
    packed sequences in RNNs/LSTMs

    Args:
        data (list): A list of samples (dictionaries from PennTreeBank.__getitem__)
        pad_token (int): The integer ID used for padding sequences
        DEVICE (str): The target device ('cpu' or 'cuda:0', etc.)

    Returns:
        dict: A dictionary containing padded 'source' and 'target' tensors moved to the
              specified DEVICE, and the total number of tokens in the batch ('number_tokens')
    """
    def merge(sequences):
        """
        Helper function to pad a list of sequences to the maximum length in the list
        
        Args:
            sequences (list[torch.Tensor or list]): A list of sequences (can be lists or tensors)

        Returns:
            tuple: A tuple containing:
                   - torch.LongTensor: The padded tensor of sequences
                   - list[int]: The original lengths of the sequences before padding
        """
        # Calculate the length of each sequence
        lengths = [len(seq) for seq in sequences]
        # Determine the maximum sequence length in the batch (handle empty batches)
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)

        # Copy each sequence into the padded tensor
        for i, seq in enumerate(sequences):
            end = lengths[i] # Get the original length of the current sequence
            padded_seqs[i, :end] = seq # Copy the sequence elements
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys(): # Assumes all samples have the same keys ('source', 'target')
        new_item[key] = [d[key] for d in data]

    # Merge (pad) the source sequences and the target sequences and get their original lengths
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    # Move the padded tensors to the specified device
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)

    # Compute the total number of non-padding tokens in the batch, useful for averaging loss
    new_item["number_tokens"] = sum(lengths)
    return new_item