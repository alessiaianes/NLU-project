# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
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
import pandas as pd  # for DataFrame



if __name__ == "__main__":
    # First we get the 10% of the training set, then we compute the percentage of these examples 

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    # Intent distributions
    # print('Train:')
    # pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    # print('Dev:'), 
    # pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    # print('Test:') 
    # pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    # print('='*89)
    # Dataset size
    # print('TRAIN size:', len(train_raw))
    # print('DEV size:', len(dev_raw))
    # print('TEST size:', len(test_raw))


    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    hid_size_values = [100, 200, 300] # Hidden size
    emb_size = 300
    batch_size_values = [32, 64, 128, 256] # Batch size
    dropout_values = [0.1, 0.2, 0.3, 0.4] # Dropout values

    lr_values = [0.0001, 0.0005, 0.0007, 0.0009, 0.001] # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Create a directory to save the results, if it doesn't exist
    os.makedirs('results/LSTM_drop/plots', exist_ok=True)
    #os.makedirs('bin/LSTM_dropout_ADAM', exist_ok=True)
    all_results = [] # To store the results of each configuration
    total_configurations = len(lr_values) * len(hid_size_values) * len(batch_size_values)
    current_configuration = 0

    for lr in lr_values:
        for hid_size in hid_size_values:
            for bs in batch_size_values:
                for d in dropout_values:
                    print("=" * 89)
                    print(f"Starting run #{current_configuration + 1} of {total_configurations}")
                    print("=" * 89)
                    print(f"Running configuration: LR={lr}, Hid Size={hid_size}, Batch Size={bs}")

                    # Dataloader instantiations
                    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn,  shuffle=True)
                    dev_loader = DataLoader(dev_dataset, batch_size=bs//2, collate_fn=collate_fn)
                    test_loader = DataLoader(test_dataset, batch_size=bs//2, collate_fn=collate_fn)

                    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, dropout=d).to(device)
                    model.apply(init_weights)

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


                    n_epochs = 200
                    patience = 3
                    losses_train = []
                    losses_dev = []
                    sampled_epochs = []
                    f1 = []
                    accuracy = []
                    best_f1 = 0
                    for x in tqdm(range(1,n_epochs)):
                        loss = train_loop(train_loader, optimizer, criterion_slots, 
                                        criterion_intents, model, clip=clip)
                        if x % 5 == 0: # We check the performance every 5 epochs
                            sampled_epochs.append(x)
                            losses_train.append(np.asarray(loss).mean())
                            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                                        criterion_intents, model, lang)
                            losses_dev.append(np.asarray(loss_dev).mean())
                            
                            
                            f1.append(results_dev['total']['f'])
                            accuracy.append(intent_res['accuracy'])
                            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                            if f1[-1] > best_f1:
                                best_f1 = f1[-1]
                                # Here you should save the model
                                patience = 3
                            else:
                                patience -= 1
                            if patience <= 0: # Early stopping with patience
                                break # Not nice but it keeps the code clean

                    print(f"Ending run #{current_configuration + 1} of {total_configurations}")
                    print("=" * 89)
                    current_configuration += 1
                    print("=" * 89)

                    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                            criterion_intents, model, lang)    
                    print('Slot F1: ', results_test['total']['f'])
                    print('Intent Accuracy:', intent_test['accuracy'])

                    all_results.append({
                            'Batch Size': bs,
                            'Learning Rate': lr,
                            'Hid size': hid_size,
                            'F1 score dev': max(f1),
                            'Accuracy dev': max(accuracy)
                        })


                    # PATH = os.path.join("bin", model_name)
                    # saving_object = {"epoch": x, 
                    #                  "model": model.state_dict(), 
                    #                  "optimizer": optimizer.state_dict(), 
                    #                  "w2id": w2id, 
                    #                  "slot2id": slot2id, 
                    #                  "intent2id": intent2id}
                    # torch.save(saving_object, PATH)

                    # Save the results in a CSV file
                    results_df = pd.DataFrame({
                        'Epoch': sampled_epochs,
                        'F1 dev': f1,
                        'Acc dev': accuracy,
                        'F1 test': [results_test['total']['f']] * len(sampled_epochs),
                        'Acc test': [intent_test['accuracy']] * len(sampled_epochs)
                    })
                    csv_filename = f'results/LSTM_drop/LSTM_drop_lr_{lr}_bs_{bs}_hid_{hid_size}.csv'
                    results_df.to_csv(csv_filename, index=False)
                    print(f'CSV file successfully saved in {csv_filename}')


                    # Create ppl_dev plot
                    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
                    plt.title('Accuracy and F1 Score on Dev Set')
                    plt.ylabel('Accuracy / F1 Score')
                    plt.xlabel('Epochs')
                    plt.plot(sampled_epochs, f1, label='F1 Score on Dev Set')
                    plt.plot(sampled_epochs, accuracy, label='Accuracy on Dev Set')
                    plt.legend()
                    plt.xlim(min(sampled_epochs), max(sampled_epochs))
                    plt.ylim(0.0, 1.0)  #lim for accuracy/F1
                    #plt.show()

                    # Save ppl_dev plot
                    res_plot_filename = f'results/LSTM_drop/plots/LSTM_res_plot_lr_{lr}_bs_{bs}_hid_{hid_size}.png'
                    plt.savefig(res_plot_filename)
                    print(f"F1 and Accuracy plot saved: '{res_plot_filename}'")
                    plt.close()

    


                    #PLOT OF THE TRAIN AND VALID LOSSES
                    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
                    plt.title('Train and Dev Losses')
                    plt.ylabel('Loss')
                    plt.xlabel('Epochs')
                    plt.plot(sampled_epochs, losses_train, label='Train loss')
                    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
                    plt.legend()
                    plt.xlim(min(sampled_epochs), max(sampled_epochs))
                    plt.ylim(0.0, 2.5)  #lim for loss
                    #plt.show()

                    # Save ppl_dev plot
                    loss_plot_filename = f'results/LSTM_drop/plots/LSTM_loss_plot_lr_{lr}_bs_{bs}_hid_{hid_size}.png'
                    plt.savefig(loss_plot_filename)
                    print(f"Loss plot saved: '{loss_plot_filename}'")
                    plt.close()

    # After the loops, find the best configuration:
    best_result_f1 = max(all_results, key=lambda x: x['F1 score dev'])
    best_result_acc = max(all_results, key=lambda x: x['Accuracy dev'])
    print(f"Best configuration f1: {best_result_f1}")
    print(f"Best configuration acc: {best_result_acc}")
    best_result_df_f1 = pd.DataFrame([best_result_f1])
    best_result_df_acc = pd.DataFrame([best_result_acc])
    best_result_df_f1.to_csv('results/LSTM_drop/best_configuration_f1.csv', index=False)
    best_result_df_acc.to_csv('results/LSTM_drop/best_configuration_acc.csv', index=False)
    print(f'Best configuration successfully saved')


    #REMEMBER TO RUN THE BEST MODEL MULTIPLE TIMES
