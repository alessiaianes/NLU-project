import json
from pprint import pprint
from collections import Counter
import os
import torch
import torch.utils.data as data
from transformers import BertTokenizer

PAD_TOKEN = 0
device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))
pprint(tmp_train_raw[0])

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlotsBERT(data.Dataset):
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.lang = lang
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]
        slots = self.slots[idx]
        intent = self.intents[idx]
        utt_ids = self.lang.tokenizer.encode(utt, add_special_tokens=True)
        slot_ids = self.align_slot_labels_with_tokens(slots, utt_ids, self.lang.tokenizer)
        intent_id = self.lang.intent2id[intent]
        sample = {'utterance': utt_ids, 'slots': slot_ids, 'intent': intent_id}
        return sample

    def align_slot_labels_with_tokens(self, slots, utt_ids, tokenizer):
        orig_tokens = slots.split()
        orig_slots = slots.split()
        bert_tokens = []
        bert_slot_labels = []

        for orig_token, orig_slot in zip(orig_tokens, orig_slots):
            sub_tokens = tokenizer.tokenize(orig_token)
            bert_tokens.extend(sub_tokens)
            bert_slot_labels.extend([orig_slot] + ['X'] * (len(sub_tokens) - 1))

        bert_slot_ids = [self.lang.slot2id.get(slot, PAD_TOKEN) for slot in bert_slot_labels]
        return bert_slot_ids
    
def collate_fn_bert(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, _ = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    attention_mask = (src_utt != PAD_TOKEN).long().to(device)
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = y_slots
    new_item["attention_mask"] = attention_mask
    return new_item