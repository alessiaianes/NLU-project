from functions import *
from utils import *
from model import *
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    labels = []
    inputs = []
    mini_train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlotsBERT(train_raw, lang)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang)

    hid_size_values = [300, 200]
    emb_size = 300
    batch_size_values = [128, 64, 32]
    dropout_values = [0.1, 0.2, 0.3, 0.4]
    lr_values = [0.0005, 0.0007, 0.0009,]
    clip = 5
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    #vocab_len = len(lang.word2id)

    os.makedirs('results/BERT_drop/plots', exist_ok=True)
    all_results = []
    total_configurations = len(lr_values) * len(hid_size_values) * len(batch_size_values) * len(dropout_values)
    current_configuration = 0

    for hid_size in hid_size_values:
        for bs in batch_size_values:
            for d in dropout_values:
                for lr in lr_values:
                    print("=" * 89)
                    print(f"Starting run #{current_configuration + 1} of {total_configurations}")
                    print("=" * 89)
                    print(f"Running configuration: lr={lr}, Hid Size={hid_size}, Batch Size={bs}, Dropout={d}")

                    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
                    dev_loader = DataLoader(dev_dataset, batch_size=bs//2, collate_fn=collate_fn_bert)
                    test_loader = DataLoader(test_dataset, batch_size=bs//2, collate_fn=collate_fn_bert)

                    model = BertModelIAS(hid_size, out_slot, out_int, dropout=d).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                    criterion_intents = nn.CrossEntropyLoss()

                    n_epochs = 200
                    patience = 3
                    losses_train = []
                    losses_dev = []
                    sampled_epochs = []
                    f1 = []
                    accuracy = []
                    best_f1 = 0

                    for x in tqdm(range(1, n_epochs)):
                        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                        if x % 5 == 0:  # We check the performance every 5 epochs
                            sampled_epochs.append(x)
                            losses_train.append(np.asarray(loss).mean())
                            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                            losses_dev.append(np.asarray(loss_dev).mean())
                            f1.append(results_dev['total']['f'])
                            accuracy.append(intent_res['accuracy'])

                            if f1[-1] > best_f1:
                                best_f1 = f1[-1]
                                patience = 3
                            else:
                                patience -= 1
                            if patience <= 0:
                                break

                    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
                    print('Slot F1: ', results_test['total']['f'])
                    print('Intent Accuracy:', intent_test['accuracy'])

                    all_results.append({
                            'Batch Size': bs,
                            'Learning Rate': lr,
                            'Hid size': hid_size,
                            'Dropout': d,
                            'F1 score dev': max(f1),
                            'Accuracy dev': max(accuracy)
                        })

                    results_df = pd.DataFrame({
                        'Epoch': sampled_epochs,
                        'F1 dev': f1,
                        'Acc dev': accuracy,
                        'F1 test': [results_test['total']['f']] * len(sampled_epochs),
                        'Acc test': [intent_test['accuracy']] * len(sampled_epochs)
                    })
                    csv_filename = f'results/BERT_drop/BERT_drop_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.csv'
                    results_df.to_csv(csv_filename, index=False)
                    print(f'CSV file successfully saved in {csv_filename}')

                    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
                    plt.title('Accuracy and F1 Score on Dev Set')
                    plt.ylabel('Accuracy / F1 Score')
                    plt.xlabel('Epochs')
                    plt.plot(sampled_epochs, f1, label='F1 Score on Dev Set')
                    plt.plot(sampled_epochs, accuracy, label='Accuracy on Dev Set')
                    plt.legend()
                    plt.xlim(min(sampled_epochs), max(sampled_epochs))
                    plt.ylim(0.0, 1.0)
                    res_plot_filename = f'results/BERT_drop/plots/BERT_res_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png'
                    plt.savefig(res_plot_filename)
                    print(f"F1 and Accuracy plot saved: '{res_plot_filename}'")
                    plt.close()

                    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
                    plt.title('Train and Dev Losses')
                    plt.ylabel('Loss')
                    plt.xlabel('Epochs')
                    plt.plot(sampled_epochs, losses_train, label='Train loss')
                    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
                    plt.legend()
                    plt.xlim(min(sampled_epochs), max(sampled_epochs))
                    plt.ylim(0.0, 2.5)
                    loss_plot_filename = f'results/BERT_drop/plots/BERT_loss_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png'
                    plt.savefig(loss_plot_filename)
                    print(f"Loss plot saved: '{loss_plot_filename}'")
                    plt.close()

                    print(f"Ending run #{current_configuration + 1} of {total_configurations}")
                    print("=" * 89)
                    current_configuration += 1
                    print("=" * 89)

    best_result_f1 = max(all_results, key=lambda x: x['F1 score dev'])
    best_result_acc = max(all_results, key=lambda x: x['Accuracy dev'])
    print(f"Best configuration f1: {best_result_f1}")
    print(f"Best configuration acc: {best_result_acc}")
    best_result_df_f1 = pd.DataFrame([best_result_f1])
    best_result_df_acc = pd.DataFrame([best_result_acc])
    best_result_df_f1.to_csv('results/BERT_drop/best_configuration_f1.csv', index=False)
    best_result_df_acc.to_csv('results/BERT_drop/best_configuration_acc.csv', index=False)
    print(f'Best configuration successfully saved')