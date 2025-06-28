import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from conll import evaluate
from sklearn.metrics import classification_report
from utils import *

def init_weights(mat):
    pass

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        attention_mask = (sample['utterance'] != PAD_TOKEN).float()
        slots, intent = model(sample['utterance'], attention_mask)
        loss_intent = criterion_intents(intent, sample['intent'])
        loss_slot = criterion_slots(slots, sample['slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            attention_mask = (sample['utterance'] != PAD_TOKEN).float()
            slots, intents = model(sample['utterance'], attention_mask)
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = len(seq)
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.tokenizer.convert_ids_to_tokens(elem) for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array