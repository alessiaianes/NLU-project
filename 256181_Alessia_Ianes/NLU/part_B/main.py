# from functions import *
# from utils import *
# from model import *
# import os
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import pandas as pd

# if __name__ == "__main__":
#     portion = 0.10
#     intents = [x['intent'] for x in tmp_train_raw]
#     count_y = Counter(intents)
#     labels = []
#     inputs = []
#     mini_train = []
#     for id_y, y in enumerate(intents):
#         if count_y[y] > 1:
#             inputs.append(tmp_train_raw[id_y])
#             labels.append(y)
#         else:
#             mini_train.append(tmp_train_raw[id_y])

#     X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
#                                                         random_state=42, 
#                                                         shuffle=True,
#                                                         stratify=labels)
#     X_train.extend(mini_train)
#     train_raw = X_train
#     dev_raw = X_dev
#     y_test = [x['intent'] for x in test_raw]

#     words = sum([x['utterance'].split() for x in train_raw], [])
#     corpus = train_raw + dev_raw + test_raw
#     slots = set(sum([line['slots'].split() for line in corpus],[]))
#     intents = set([line['intent'] for line in corpus])
#     lang = Lang(words, intents, slots, cutoff=0)

#     train_dataset = IntentsAndSlotsBERT(train_raw, lang)
#     dev_dataset = IntentsAndSlotsBERT(dev_raw, lang)
#     test_dataset = IntentsAndSlotsBERT(test_raw, lang)

#     hid_size_values = [300, 200]
#     emb_size = 300
#     batch_size_values = [128, 64, 32]
#     dropout_values = [0.1, 0.2, 0.3, 0.4]
#     lr_values = [0.0005, 0.0007, 0.0009,]
#     clip = 5
#     out_slot = len(lang.slot2id)
#     out_int = len(lang.intent2id)
#     #vocab_len = len(lang.word2id)

#     os.makedirs('results/BERT_drop/plots', exist_ok=True)
#     all_results = []
#     total_configurations = len(lr_values) * len(hid_size_values) * len(batch_size_values) * len(dropout_values)
#     current_configuration = 0

#     for hid_size in hid_size_values:
#         for bs in batch_size_values:
#             for d in dropout_values:
#                 for lr in lr_values:
#                     print("=" * 89)
#                     print(f"Starting run #{current_configuration + 1} of {total_configurations}")
#                     print("=" * 89)
#                     print(f"Running configuration: lr={lr}, Hid Size={hid_size}, Batch Size={bs}, Dropout={d}")

#                     train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
#                     dev_loader = DataLoader(dev_dataset, batch_size=bs//2, collate_fn=collate_fn_bert)
#                     test_loader = DataLoader(test_dataset, batch_size=bs//2, collate_fn=collate_fn_bert)

#                     model = BertModelIAS(hid_size, out_slot, out_int, dropout=d).to(device)
#                     optimizer = optim.Adam(model.parameters(), lr=lr)
#                     criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
#                     criterion_intents = nn.CrossEntropyLoss()

#                     n_epochs = 200
#                     patience = 3
#                     losses_train = []
#                     losses_dev = []
#                     sampled_epochs = []
#                     f1 = []
#                     accuracy = []
#                     best_f1 = 0

#                     for x in tqdm(range(1, n_epochs)):
#                         loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
#                         if x % 5 == 0:  # We check the performance every 5 epochs
#                             sampled_epochs.append(x)
#                             losses_train.append(np.asarray(loss).mean())
#                             results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
#                             losses_dev.append(np.asarray(loss_dev).mean())
#                             f1.append(results_dev['total']['f'])
#                             accuracy.append(intent_res['accuracy'])

#                             if f1[-1] > best_f1:
#                                 best_f1 = f1[-1]
#                                 patience = 3
#                             else:
#                                 patience -= 1
#                             if patience <= 0:
#                                 break

#                     results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
#                     print('Slot F1: ', results_test['total']['f'])
#                     print('Intent Accuracy:', intent_test['accuracy'])

#                     all_results.append({
#                             'Batch Size': bs,
#                             'Learning Rate': lr,
#                             'Hid size': hid_size,
#                             'Dropout': d,
#                             'F1 score dev': max(f1),
#                             'Accuracy dev': max(accuracy)
#                         })

#                     results_df = pd.DataFrame({
#                         'Epoch': sampled_epochs,
#                         'F1 dev': f1,
#                         'Acc dev': accuracy,
#                         'F1 test': [results_test['total']['f']] * len(sampled_epochs),
#                         'Acc test': [intent_test['accuracy']] * len(sampled_epochs)
#                     })
#                     csv_filename = f'results/BERT_drop/BERT_drop_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.csv'
#                     results_df.to_csv(csv_filename, index=False)
#                     print(f'CSV file successfully saved in {csv_filename}')

#                     plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
#                     plt.title('Accuracy and F1 Score on Dev Set')
#                     plt.ylabel('Accuracy / F1 Score')
#                     plt.xlabel('Epochs')
#                     plt.plot(sampled_epochs, f1, label='F1 Score on Dev Set')
#                     plt.plot(sampled_epochs, accuracy, label='Accuracy on Dev Set')
#                     plt.legend()
#                     plt.xlim(min(sampled_epochs), max(sampled_epochs))
#                     plt.ylim(0.0, 1.0)
#                     res_plot_filename = f'results/BERT_drop/plots/BERT_res_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png'
#                     plt.savefig(res_plot_filename)
#                     print(f"F1 and Accuracy plot saved: '{res_plot_filename}'")
#                     plt.close()

#                     plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
#                     plt.title('Train and Dev Losses')
#                     plt.ylabel('Loss')
#                     plt.xlabel('Epochs')
#                     plt.plot(sampled_epochs, losses_train, label='Train loss')
#                     plt.plot(sampled_epochs, losses_dev, label='Dev loss')
#                     plt.legend()
#                     plt.xlim(min(sampled_epochs), max(sampled_epochs))
#                     plt.ylim(0.0, 2.5)
#                     loss_plot_filename = f'results/BERT_drop/plots/BERT_loss_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png'
#                     plt.savefig(loss_plot_filename)
#                     print(f"Loss plot saved: '{loss_plot_filename}'")
#                     plt.close()

#                     print(f"Ending run #{current_configuration + 1} of {total_configurations}")
#                     print("=" * 89)
#                     current_configuration += 1
#                     print("=" * 89)

#     best_result_f1 = max(all_results, key=lambda x: x['F1 score dev'])
#     best_result_acc = max(all_results, key=lambda x: x['Accuracy dev'])
#     print(f"Best configuration f1: {best_result_f1}")
#     print(f"Best configuration acc: {best_result_acc}")
#     best_result_df_f1 = pd.DataFrame([best_result_f1])
#     best_result_df_acc = pd.DataFrame([best_result_acc])
#     best_result_df_f1.to_csv('results/BERT_drop/best_configuration_f1.csv', index=False)
#     best_result_df_acc.to_csv('results/BERT_drop/best_configuration_acc.csv', index=False)
#     print(f'Best configuration successfully saved')


# Imports at the top
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn # Import nn module
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom modules (assuming they are in the same directory or accessible)
from utils import * # Imports load_data, Lang, IntentsAndSlotsBERT, PAD_TOKEN, device
from model import BertModelIAS
from functions import train_loop, eval_loop


if __name__ == "__main__":
    
    # --- Data Preparation ---
    # Handle potential errors if data loading failed in utils.py
    if 'tmp_train_raw' not in globals() or 'test_raw' not in globals():
         print("Error: Dataset not loaded properly. Exiting.")
         exit()

    portion = 0.10 # Portion for the development set from the training data
    
    # Separate intents that appear only once for potential different handling
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    inputs_multi = [] # Data with intents appearing more than once
    mini_train = []   # Data with intents appearing only once

    for data_point in tmp_train_raw:
        if count_y[data_point['intent']] > 1:
            inputs_multi.append(data_point)
        else:
            mini_train.append(data_point)

    # Split the data with multiple samples per intent
    # Ensure there are enough samples to split (at least 2)
    if len(inputs_multi) > 1:
        X_train_multi, X_dev, y_train_multi, y_dev = train_test_split(
            inputs_multi, 
            [x['intent'] for x in inputs_multi], # Pass intents for stratification
            test_size=portion, 
            random_state=42, 
            shuffle=True,
            stratify= [x['intent'] for x in inputs_multi] # Stratify based on intents
        )
        # Combine training data (multi-intent samples + single-intent samples)
        train_raw = X_train_multi + mini_train
        dev_raw = X_dev
    else:
         print("Warning: Not enough samples with multiple intents to perform stratified split. Using all data for training.")
         train_raw = tmp_train_raw # Use all original training data
         dev_raw = [] # No separate dev set from split

    # Prepare test set intents (needed for Lang class)
    y_test = [x['intent'] for x in test_raw]

    # Build vocabulary and language structures using ALL data (train + dev + test)
    # This ensures the Lang object knows about all possible intents and slots.
    corpus = train_raw + dev_raw + test_raw
    all_words = []
    all_slots = set()
    all_intents = set()
    for line in corpus:
        all_words.extend(line['utterance'].split())
        all_slots.update(line['slots'].split())
        all_intents.add(line['intent'])
        
    # Initialize Lang object - cutoff=0 means keep all words
    lang = Lang(all_words, list(all_intents), list(all_slots), cutoff=0)

    # Create Datasets
    train_dataset = IntentsAndSlotsBERT(train_raw, lang)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang)

    # --- Slot Distribution Analysis ---
    # Function to analyze and print slot distribution
    def analyze_slot_distribution(dataset_list, name="Dataset"):
        """Analyzes and prints the distribution of slot tags in the dataset."""
        slot_counts = Counter()
        total_slots = 0
        if not dataset_list: 
            print(f"\n--- Slot Distribution Analysis for {name} ---")
            print("Dataset is empty. No slots to analyze.")
            print("-------------------------------------------\n")
            return

        for item in dataset_list:
            slots_str = item['slots']
            slots = slots_str.split()
            slot_counts.update(slots)
            total_slots += len(slots)
        
        print(f"\n--- Slot Distribution Analysis for {name} ---")
        print(f"Total slots counted: {total_slots}")
        sorted_slots = sorted(slot_counts.items(), key=lambda item: item[1], reverse=True)
        
        # Print top N slots and overall counts for clarity
        max_slots_to_print = 10 # Limit output for clarity
        for i, (slot, count) in enumerate(sorted_slots):
            if i >= max_slots_to_print:
                print(f"... and {len(sorted_slots) - max_slots_to_print} more unique slots.")
                break
            percentage = (count / total_slots) * 100 if total_slots > 0 else 0
            print(f"'{slot}': {count} ({percentage:.2f}%)")
        print("-------------------------------------------\n")

    # Call the analysis functions on the raw data lists to confirm imbalance
    analyze_slot_distribution(train_raw, name="Train Raw")
    analyze_slot_distribution(dev_raw, name="Dev Raw") 
    analyze_slot_distribution(test_raw, name="Test Raw")
    # --- End Slot Distribution Analysis ---


    # --- Calculate Slot Class Weights for Weighted Loss ---
    slot_name_counts = Counter()
    total_slots_train = 0
    for item in train_raw:
        slots_str = item['slots']
        slots = slots_str.split()
        slot_name_counts.update(slots)
        total_slots_train += len(slots)

    # `out_slot` is the total number of unique slot IDs (including PAD_TOKEN)
    num_classes = out_slot 
    
    # Initialize weights list with zeros. Size must be `num_classes`.
    weights_tensor_list = [0.0] * num_classes

    # Calculate weights for each slot ID, excluding the PAD_TOKEN index.
    epsilon = 1e-6 # Small value to prevent division by zero
    
    for slot_name, slot_id in lang.slot2id.items():
        # Skip the padding index (usually 0) as it's handled by `ignore_index` in the loss
        if slot_id == PAD_TOKEN: 
            continue 
            
        count = slot_name_counts.get(slot_name, 0)
        
        if count > 0:
            # Calculate weight: total_samples / (num_classes * class_count)
            # This formula assigns higher weights to rarer classes.
            weight = total_slots_train / (num_classes * (count + epsilon))
        else:
            # Default weight if a slot somehow has 0 count but is in slot2id (shouldn't happen with proper corpus processing)
            weight = 1.0 
        
        # Ensure the slot_id is within the bounds of the list
        if slot_id < num_classes:
            weights_tensor_list[slot_id] = weight
        else:
            print(f"Warning: Slot ID {slot_id} for slot '{slot_name}' is out of bounds (num_classes={num_classes}). Skipping weight assignment.")

    # Convert the list to a PyTorch tensor and move it to the correct device
    class_weights = torch.Tensor(weights_tensor_list).to(device)
    # --- End of Class Weight Calculation ---


    # --- Hyperparameter Search Setup ---
    hid_size_values = [300, 200] 
    batch_size_values = [128, 64, 32] 
    dropout_values = [0.1, 0.2, 0.3, 0.4] 
    # Adjusted LR values based on common BERT practices; original values might have been too high.
    lr_values = [0.00005, 0.00007, 0.00009] 
    clip = 5 
    
    results_dir = 'results/BERT_drop'
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    all_results = [] # To store results for finding the best configuration
    
    total_configurations = len(hid_size_values) * len(batch_size_values) * len(dropout_values) * len(lr_values)
    current_configuration = 0

    # --- Hyperparameter Tuning Loop ---
    for hid_size in hid_size_values:
        for bs in batch_size_values:
            for d in dropout_values:
                for lr in lr_values:
                    current_configuration += 1
                    
                    # --- Setup for the current hyperparameter configuration ---
                    # Use a descriptive prefix for the tqdm bar's description
                    config_desc = f"Config {current_configuration}/{total_configurations} (lr={lr:.5f}, bs={bs}, hid={hid_size}, drop={d:.1f})"
                    
                    # Create DataLoaders
                    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
                    # Use a reasonable batch size for evaluation, ensuring it's not larger than training batch size if possible.
                    eval_batch_size = min(bs // 2, 32) if bs > 16 else 16 
                    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=collate_fn_bert)
                    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn_bert)

                    # Initialize Model, Optimizer, Loss Functions
                    model = BertModelIAS(hid_size, out_slot, out_int, dropout=d).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    # --- APPLY WEIGHTED LOSS for slot predictions ---
                    # The weights help the model focus on minority slot classes.
                    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, weight=class_weights) 
                    # --- END WEIGHTED LOSS ---
                    
                    criterion_intents = nn.CrossEntropyLoss()

                    # Training Parameters
                    n_epochs = 50 # Max epochs; early stopping will determine actual epochs.
                    patience = 3 # Number of epochs to wait for improvement before stopping.
                    patience_counter = 0
                    
                    # Lists to store metrics for plotting and analysis
                    losses_train_avg = [] 
                    losses_dev_avg = []   
                    sampled_epochs = []   
                    f1_scores_dev = []    
                    accuracies_dev = []   
                    best_f1_dev = 0.0     # Track the best F1 score achieved on the dev set.
                    
                    # --- Create tqdm object for the epoch loop ---
                    # `leave=False` ensures the bar disappears after completion, keeping output clean.
                    # `position=0` ensures it's at the top if multiple bars were ever shown.
                    epoch_progress_bar = tqdm(range(1, n_epochs + 1), desc=config_desc, leave=False, position=0) 

                    # Loop through epochs for training and evaluation
                    for epoch in epoch_progress_bar:
                        # Train the model for one epoch
                        avg_loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                        
                        # Evaluate the model on the development set every epoch
                        sampled_epochs.append(epoch)
                        losses_train_avg.append(avg_loss_train)
                        
                        results_dev, intent_res_dev, avg_loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                        
                        current_f1_dev = results_dev.get('total', {}).get('f', 0.0) 
                        current_acc_dev = intent_res_dev.get('accuracy', 0.0) 
                        
                        losses_dev_avg.append(avg_loss_dev)
                        f1_scores_dev.append(current_f1_dev)
                        accuracies_dev.append(current_acc_dev)

                        # --- Update the tqdm postfix with the current metrics ---
                        # This creates the single, updating progress line.
                        epoch_progress_bar.set_postfix({
                            'Epoch': epoch, 
                            'T_Loss': f'{avg_loss_train:.4f}', 
                            'D_Loss': f'{avg_loss_dev:.4f}', 
                            'D_F1': f'{current_f1_dev:.4f}', 
                            'D_Acc': f'{current_acc_dev:.4f}'
                        })
                        # --- End of postfix update ---

                        # Early stopping check: Stop if dev F1 score hasn't improved for `patience` epochs.
                        if current_f1_dev > best_f1_dev:
                            best_f1_dev = current_f1_dev
                            patience_counter = 0
                            # Optionally save the best model state here if needed.
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            # Print early stopping message on a new line for clarity before the next config starts.
                            print(f"\nEarly stopping triggered after {epoch} epochs.") 
                            break # Exit the epoch loop for this configuration

                    # --- Evaluation on Test Set ---
                    # Perform final evaluation on the test set after training (or early stopping)
                    print("Evaluating on Test set...")
                    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
                    print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
                    print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

                    # --- Store Results ---
                    # Store the best dev scores and the final test scores for this configuration
                    all_results.append({
                        'Batch Size': bs,
                        'Learning Rate': lr,
                        'Hid size': hid_size,
                        'Dropout': d,
                        'F1 score dev': best_f1_dev, # Use the best dev F1 found during training
                        'Accuracy dev': max(accuracies_dev) if accuracies_dev else 0.0, # Use the best dev accuracy
                        'Test F1': results_test.get('total', {}).get('f', 0.0),
                        'Test Acc': intent_test.get('accuracy', 0.0)
                    })

                    # Save detailed results per configuration to CSV for later analysis
                    results_df = pd.DataFrame({
                        'Epoch': sampled_epochs,
                        'F1 dev': f1_scores_dev,
                        'Acc dev': accuracies_dev,
                        'Loss Train': losses_train_avg,
                        'Loss Dev': losses_dev_avg,
                        # Repeat test scores for each epoch line for consistency in CSV structure
                        'F1 test': [results_test.get('total', {}).get('f', 0.0)] * len(sampled_epochs),
                        'Acc test': [intent_test.get('accuracy', 0.0)] * len(sampled_epochs)
                    })
                    csv_filename = os.path.join(results_dir, f'BERT_drop_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.csv')
                    results_df.to_csv(csv_filename, index=False)
                    print(f"Detailed results saved to {csv_filename}")

                    # --- Plotting ---
                    # Plot F1 and Accuracy on Dev Set
                    plt.figure(figsize=(10, 6)) 
                    plt.title(f'Dev Set Performance (lr={lr}, bs={bs}, hid={hid_size}, drop={d})')
                    plt.ylabel('Score')
                    plt.xlabel('Epoch')
                    plt.plot(sampled_epochs, f1_scores_dev, label='F1 Score (Dev)')
                    plt.plot(sampled_epochs, accuracies_dev, label='Accuracy (Dev)')
                    plt.legend()
                    plt.ylim(0.0, 1.05) # Set y-axis limits for scores
                    plt.grid(True)
                    res_plot_filename = os.path.join(plot_dir, f'BERT_res_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png')
                    plt.savefig(res_plot_filename)
                    print(f"Dev performance plot saved: '{res_plot_filename}'")
                    plt.close() # Close the plot figure to free memory

                    # Plot Losses
                    plt.figure(figsize=(10, 6)) 
                    plt.title(f'Training and Dev Losses (lr={lr}, bs={bs}, hid={hid_size}, drop={d})')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.plot(sampled_epochs, losses_train_avg, label='Train Loss')
                    plt.plot(sampled_epochs, losses_dev_avg, label='Dev Loss')
                    plt.legend()
                    # Dynamically adjust y-axis limit for losses based on observed values
                    max_loss = 0
                    if losses_train_avg: max_loss = max(max_loss, max(losses_train_avg))
                    if losses_dev_avg: max_loss = max(max_loss, max(losses_dev_avg))
                    plt.ylim(0.0, (max_loss * 1.2) if max_loss > 0 else 2.5) # Use 2.5 as fallback if no losses observed
                    plt.grid(True)
                    loss_plot_filename = os.path.join(plot_dir, f'BERT_loss_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png')
                    plt.savefig(loss_plot_filename)
                    print(f"Loss plot saved: '{loss_plot_filename}'")
                    plt.close() # Close the plot figure to free memory

                    print(f"Finished run #{current_configuration} of {total_configurations}")
                    print("-" * 89)


    # --- Find and Save Best Configuration ---
    if all_results:
        # Find the best configuration based on the highest F1 score achieved on the development set
        best_result_f1 = max(all_results, key=lambda x: x['F1 score dev'])
        # Find the best configuration based on the highest Accuracy achieved on the development set
        best_result_acc = max(all_results, key=lambda x: x['Accuracy dev'])

        print("\n" + "=" * 89)
        print("Hyperparameter Search Complete")
        print(f"Best configuration (by Dev F1): {best_result_f1}")
        print(f"Best configuration (by Dev Acc): {best_result_acc}")
        print("=" * 89)

        # Save the best configurations to CSV files
        best_result_df_f1 = pd.DataFrame([best_result_f1])
        best_result_df_acc = pd.DataFrame([best_result_acc])
        
        best_f1_csv = os.path.join(results_dir, 'best_configuration_f1.csv')
        best_acc_csv = os.path.join(results_dir, 'best_configuration_acc.csv')
        
        best_result_df_f1.to_csv(best_f1_csv, index=False)
        best_result_df_acc.to_csv(best_acc_csv, index=False)
        
        print(f"Best configurations saved to '{best_f1_csv}' and '{best_acc_csv}'")
    else:
        print("No results were generated. Check for errors during training.")

if __name__ == "__main__":
    
    # --- Data Preparation ---
    # Handle potential errors if data loading failed in utils.py
    if 'tmp_train_raw' not in globals() or 'test_raw' not in globals():
         print("Error: Dataset not loaded properly. Exiting.")
         exit()

    portion = 0.10 # Portion for the development set from the training data
    
    # Separate intents that appear only once for potential different handling
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    inputs_multi = [] # Data with intents appearing more than once
    mini_train = []   # Data with intents appearing only once

    for data_point in tmp_train_raw:
        if count_y[data_point['intent']] > 1:
            inputs_multi.append(data_point)
        else:
            mini_train.append(data_point)

    # Split the data with multiple samples per intent
    # Ensure there are enough samples to split (at least 2)
    if len(inputs_multi) > 1:
        X_train_multi, X_dev, y_train_multi, y_dev = train_test_split(
            inputs_multi, 
            [x['intent'] for x in inputs_multi], # Pass intents for stratification
            test_size=portion, 
            random_state=42, 
            shuffle=True,
            stratify= [x['intent'] for x in inputs_multi] # Stratify based on intents
        )
        # Combine training data (multi-intent samples + single-intent samples)
        train_raw = X_train_multi + mini_train
        dev_raw = X_dev
    else:
         print("Warning: Not enough samples with multiple intents to perform stratified split. Using all data for training.")
         train_raw = tmp_train_raw # Use all original training data
         dev_raw = [] # No separate dev set from split

    # Prepare test set intents (needed for Lang class)
    y_test = [x['intent'] for x in test_raw]

    # Build vocabulary and language structures using ALL data (train + dev + test)
    # This ensures the Lang object knows about all possible intents and slots.
    corpus = train_raw + dev_raw + test_raw
    all_words = []
    all_slots = set()
    all_intents = set()
    for line in corpus:
        all_words.extend(line['utterance'].split())
        all_slots.update(line['slots'].split())
        all_intents.add(line['intent'])
        
    # Initialize Lang object - cutoff=0 means keep all words
    lang = Lang(all_words, list(all_intents), list(all_slots), cutoff=0)

    # Create Datasets
    train_dataset = IntentsAndSlotsBERT(train_raw, lang)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang)

    # --- Calculate Slot Class Weights for Weighted Loss ---
    slot_name_counts = Counter()
    total_slots_train = 0
    # Iterate through the raw training data to count slot occurrences
    for item in train_raw:
        slots_str = item['slots']
        slots = slots_str.split()
        slot_name_counts.update(slots)
        total_slots_train += len(slots)

    # `out_slot` is the total number of unique slot IDs (including PAD_TOKEN)
    num_classes = out_slot 
    
    # Initialize weights list with zeros. Size must be `num_classes`.
    weights_tensor_list = [0.0] * num_classes

    # Calculate weights for each slot ID, excluding the PAD_TOKEN index.
    # The formula `total_samples / (num_classes * class_frequency)` helps balance the loss.
    epsilon = 1e-6 # Small value to prevent division by zero if a class count is 0
    
    for slot_name, slot_id in lang.slot2id.items():
        # Skip the padding index (usually 0) as its targets are ignored by `ignore_index`
        if slot_id == PAD_TOKEN: 
            continue 
            
        count = slot_name_counts.get(slot_name, 0)
        
        if count > 0:
            # Calculate weight: total_samples / (num_classes * class_count)
            weight = total_slots_train / (num_classes * (count + epsilon))
        else:
            # Default weight if a slot somehow has 0 count but is in slot2id (shouldn't typically happen)
            weight = 1.0 
        
        # Assign the calculated weight to the correct slot ID index
        weights_tensor_list[slot_id] = weight

    # Convert the list to a PyTorch tensor and move it to the correct device
    class_weights = torch.Tensor(weights_tensor_list).to(device)
    # --- End of Class Weight Calculation ---

    # --- Hyperparameter Search Setup ---
    hid_size_values = [300, 200] # Example hidden sizes
    batch_size_values = [128, 64, 32] # Batch sizes to test
    dropout_values = [0.1, 0.2, 0.3, 0.4] # Dropout rates to test
    # Using lower learning rates which are often more suitable for BERT fine-tuning
    lr_values = [0.00005, 0.00007, 0.00009] 
    clip = 5 # Gradient clipping value
    
    # `out_slot` and `out_int` are already defined based on lang
    
    results_dir = 'results/BERT_drop'
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    all_results = [] # To store results for finding the best configuration
    
    # Calculate total configurations for progress tracking
    total_configurations = len(hid_size_values) * len(batch_size_values) * len(dropout_values) * len(lr_values)
    current_configuration = 0

    # --- Hyperparameter Tuning Loop ---
    for hid_size in hid_size_values:
        for bs in batch_size_values:
            for d in dropout_values:
                for lr in lr_values:
                    current_configuration += 1
                    
                    # --- Configuration Description for Progress Bar ---
                    config_desc = f"Config {current_configuration}/{total_configurations} (lr={lr:.5f}, bs={bs}, hid={hid_size}, drop={d:.1f})"
                    
                    # Create DataLoaders
                    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
                    # Use a smaller batch size for dev/test loaders for efficiency, capped at 32
                    eval_batch_size = min(bs // 2, 32) if bs > 16 else 16 
                    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=collate_fn_bert)
                    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn_bert)

                    # Initialize Model, Optimizer, Loss Functions
                    model = BertModelIAS(hid_size, out_slot, out_int, dropout=d).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    # --- Apply class weights to the slot loss function ---
                    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, weight=class_weights) 
                    
                    criterion_intents = nn.CrossEntropyLoss()

                    # Training Parameters
                    n_epochs = 50 # Number of epochs to train for (early stopping can stop it earlier)
                    patience = 3  # Number of epochs to wait for improvement before stopping
                    patience_counter = 0 # Counter for early stopping
                    
                    # Lists to store metrics for plotting and analysis
                    losses_train_avg = [] 
                    losses_dev_avg = []   
                    sampled_epochs = []   
                    f1_scores_dev = []    
                    accuracies_dev = []   
                    best_f1_dev = 0.0     # Track the best F1 score achieved on the dev set
                    
                    # --- Create tqdm progress bar for the epoch loop ---
                    # `desc` shows configuration details, `leave=False` cleans up the bar after completion.
                    epoch_progress_bar = tqdm(range(1, n_epochs + 1), desc=config_desc, leave=False, position=0) 

                    # --- Epoch Training Loop ---
                    for epoch in epoch_progress_bar:
                        # Train the model for one epoch
                        avg_loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                        
                        # Evaluate on Dev set every epoch
                        sampled_epochs.append(epoch)
                        losses_train_avg.append(avg_loss_train)
                        
                        results_dev, intent_res_dev, avg_loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                        
                        current_f1_dev = results_dev.get('total', {}).get('f', 0.0) # Safely get F1 score
                        current_acc_dev = intent_res_dev.get('accuracy', 0.0) # Safely get accuracy
                        
                        losses_dev_avg.append(avg_loss_dev)
                        f1_scores_dev.append(current_f1_dev)
                        accuracies_dev.append(current_acc_dev)

                        # Update the tqdm progress bar's postfix with current epoch metrics
                        epoch_progress_bar.set_postfix({
                            'Epoch': epoch, 
                            'T_Loss': f'{avg_loss_train:.4f}', 
                            'D_Loss': f'{avg_loss_dev:.4f}', 
                            'D_F1': f'{current_f1_dev:.4f}', 
                            'D_Acc': f'{current_acc_dev:.4f}'
                        })

                        # --- Early Stopping Check ---
                        if current_f1_dev > best_f1_dev:
                            best_f1_dev = current_f1_dev
                            patience_counter = 0 # Reset patience counter if improvement is found
                        else:
                            patience_counter += 1 # Increment counter if no improvement
                        
                        if patience_counter >= patience:
                            print(f"\nEarly stopping triggered after {epoch} epochs.") # Print on new line for clarity
                            break # Exit the epoch loop

                    # --- Evaluation on Test Set ---
                    print("Evaluating on Test set...")
                    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
                    print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
                    print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

                    # --- Store Results ---
                    # Store the best dev scores and the final test scores for this configuration
                    all_results.append({
                        'Batch Size': bs,
                        'Learning Rate': lr,
                        'Hid size': hid_size,
                        'Dropout': d,
                        'F1 score dev': best_f1_dev, # Use the best dev score found
                        'Accuracy dev': max(accuracies_dev) if accuracies_dev else 0.0, # Use the max dev accuracy
                        'Test F1': results_test.get('total', {}).get('f', 0.0),
                        'Test Acc': intent_test.get('accuracy', 0.0)
                    })

                    # Save detailed results per configuration to CSV
                    results_df = pd.DataFrame({
                        'Epoch': sampled_epochs,
                        'F1 dev': f1_scores_dev,
                        'Acc dev': accuracies_dev,
                        'Loss Train': losses_train_avg,
                        'Loss Dev': losses_dev_avg,
                        # Repeat test scores for each epoch line for consistency in the CSV
                        'F1 test': [results_test.get('total', {}).get('f', 0.0)] * len(sampled_epochs),
                        'Acc test': [intent_test.get('accuracy', 0.0)] * len(sampled_epochs)
                    })
                    csv_filename = os.path.join(results_dir, f'BERT_drop_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.csv')
                    results_df.to_csv(csv_filename, index=False)
                    print(f"Detailed results saved to {csv_filename}")

                    # --- Plotting ---
                    # Plot F1 and Accuracy on Dev Set
                    plt.figure(figsize=(10, 6)) 
                    plt.title(f'Dev Set Performance (lr={lr}, bs={bs}, hid={hid_size}, drop={d})')
                    plt.ylabel('Score')
                    plt.xlabel('Epoch')
                    plt.plot(sampled_epochs, f1_scores_dev, label='F1 Score (Dev)')
                    plt.plot(sampled_epochs, accuracies_dev, label='Accuracy (Dev)')
                    plt.legend()
                    plt.ylim(0.0, 1.05) # Set y-axis limits for scores
                    plt.grid(True)
                    res_plot_filename = os.path.join(plot_dir, f'BERT_res_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png')
                    plt.savefig(res_plot_filename)
                    print(f"Dev performance plot saved: '{res_plot_filename}'")
                    plt.close()

                    # Plot Losses
                    plt.figure(figsize=(10, 6)) 
                    plt.title(f'Training and Dev Losses (lr={lr}, bs={bs}, hid={hid_size}, drop={d})')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.plot(sampled_epochs, losses_train_avg, label='Train Loss')
                    plt.plot(sampled_epochs, losses_dev_avg, label='Dev Loss')
                    plt.legend()
                    # Dynamically adjust y-axis limit for losses
                    max_loss = 0
                    if losses_train_avg: max_loss = max(max_loss, max(losses_train_avg))
                    if losses_dev_avg: max_loss = max(max_loss, max(losses_dev_avg))
                    # Set ylim, using 2.5 as a fallback if no losses are recorded or max_loss is 0
                    plt.ylim(0.0, (max_loss * 1.2) if max_loss > 0 else 2.5) 
                    plt.grid(True)
                    loss_plot_filename = os.path.join(plot_dir, f'BERT_loss_plot_lr_{lr}_bs_{bs}_hid_{hid_size}_dropout_{d}.png')
                    plt.savefig(loss_plot_filename)
                    print(f"Loss plot saved: '{loss_plot_filename}'")
                    plt.close()

                    print(f"Finished run #{current_configuration} of {total_configurations}")
                    print("-" * 89)


    # --- Find and Save Best Configuration ---
    if all_results:
        # Find the best configuration based on the highest F1 score on the Dev set
        best_result_f1 = max(all_results, key=lambda x: x['F1 score dev'])
        # Find the best configuration based on the highest Accuracy on the Dev set
        best_result_acc = max(all_results, key=lambda x: x['Accuracy dev'])

        print("\n" + "=" * 89)
        print("Hyperparameter Search Complete")
        print(f"Best configuration (by Dev F1): {best_result_f1}")
        print(f"Best configuration (by Dev Acc): {best_result_acc}")
        print("=" * 89)

        # Save the best configurations to CSV files
        best_result_df_f1 = pd.DataFrame([best_result_f1])
        best_result_df_acc = pd.DataFrame([best_result_acc])
        
        best_f1_csv = os.path.join(results_dir, 'best_configuration_f1.csv')
        best_acc_csv = os.path.join(results_dir, 'best_configuration_acc.csv')
        
        best_result_df_f1.to_csv(best_f1_csv, index=False)
        best_result_df_acc.to_csv(best_acc_csv, index=False)
        
        print(f"Best configurations saved to '{best_f1_csv}' and '{best_acc_csv}'")
    else:
        print("No results were generated. Check for errors during training.")