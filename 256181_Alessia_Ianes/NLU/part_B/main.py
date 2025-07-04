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
    # Assuming tmp_train_raw and test_raw are loaded globally in utils.py or passed here
    
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

    # --- Hyperparameter Search Setup ---
    #hid_size_values = [200, 300] # Example hidden sizes
    emb_size = 300 # Embedding size (might be fixed by BERT model, e.g., 768)
    batch_size_values = [128] # Reduced batch sizes for potentially faster testing
    dropout_values = [0.1] # Example dropout values
    lr_values = [0.00009] # Lower learning rates often work better with Adam/BERT
    clip = 5 # Gradient clipping value
    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    
    # Create results directory
    results_dir = 'results/BERT_large'
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    model_save_dir = "bin/BERT_large"
    os.makedirs(model_save_dir, exist_ok=True)

    

    all_results = [] # To store results for finding the best configuration
    
    # Calculate total configurations for progress tracking
    total_configurations = len(batch_size_values) * len(dropout_values) * len(lr_values)
    current_configuration = 0


    # --- Hyperparameter Tuning Loop ---
  
    for bs in batch_size_values:
        for d in dropout_values:
            for lr in lr_values:
                current_configuration += 1
                print("\n" + "=" * 89)
                print(f"Starting run #{current_configuration} of {total_configurations}")
                print(f"Running configuration: lr={lr}, Batch Size={bs}, Dropout={d}")
                print("=" * 89)

                # Create DataLoaders
                # Use smaller batch size for dev/test if needed, but ensure collate_fn handles it
                train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
                # Use bs//2 or a fixed smaller size like 32 for dev/test loaders
                dev_loader = DataLoader(dev_dataset, batch_size=bs // 2, collate_fn=collate_fn_bert)
                test_loader = DataLoader(test_dataset, batch_size= bs // 2, collate_fn=collate_fn_bert)

                # Initialize Model, Optimizer, Loss Functions
                model = BertModelIAS(out_slot, out_int, dropout=d).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # Ignore index PAD_TOKEN (0) in slot loss calculation
                criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
                criterion_intents = nn.CrossEntropyLoss()

                # Training Parameters
                n_epochs = 50 # Reduced epochs for quicker testing; increase for full run (e.g., 50 or 100)
                patience = 3 # Early stopping patience
                patience_counter = 0
                
                losses_train_avg = [] # Store average training loss per epoch (or per N steps)
                losses_dev_avg = []   # Store average dev loss per epoch
                sampled_epochs = []   # Store epochs where evaluation was done
                f1_scores_dev = []    # Store F1 scores on dev set
                accuracies_dev = []   # Store intent accuracies on dev set
                best_f1_dev = 0.0     # Track best F1 score on dev set

                print(f"Starting training for {n_epochs} epochs...")
                epoch_progress_bar = tqdm(range(1, n_epochs), desc="Training process") 
                for epoch in epoch_progress_bar:
                    # Train for one epoch
                    avg_loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                    
                    # Evaluate every N epochs (e.g., every epoch or every 5 epochs)
                    #if epoch % 1 == 0: # Check performance every epoch for more detail
                    sampled_epochs.append(epoch)
                    losses_train_avg.append(avg_loss_train)
                    
                    # Evaluate on Dev set
                    results_dev, intent_res_dev, avg_loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                    
                    current_f1_dev = results_dev.get('total', {}).get('f', 0.0) # Safely get F1 score
                    current_acc_dev = intent_res_dev.get('accuracy', 0.0) # Safely get accuracy
                    
                    losses_dev_avg.append(avg_loss_dev)
                    f1_scores_dev.append(current_f1_dev)
                    accuracies_dev.append(current_acc_dev)

                    epoch_progress_bar.set_postfix({
                        'Epoch': epoch, 
                        'Dev_F1': f'{current_f1_dev:.4f}', 
                        'Dev_Acc': f'{current_acc_dev:.4f}'
                    })

                    # Early stopping check
                    if current_f1_dev > best_f1_dev:
                        best_f1_dev = current_f1_dev
                        patience_counter = 0
                        # Optionally save the best model based on dev F1 score
                        # torch.save(model.state_dict(), f"{results_dir}/best_model_f1_config_{current_configuration}.pt")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch} epochs.")
                        break # Exit epoch loop

                # --- Evaluation on Test Set ---
                print("Evaluating on Test set...")
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
                print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
                print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

                config_params = {
                    'Model': 'BERT-large-uncased', # Explicitly state the model used
                    'Batch Size': bs,
                    'Learning Rate': lr,
                    'Dropout': d
                }

                model_filename = f"BERT_large_model.pt"
                model_save_path = os.path.join(model_save_dir, model_filename)
                saving_object = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "slot2id": lang.slot2id,
                    "intent2id": lang.intent2id,
                    "config": config_params, # Save the config that led to this best score
                }
                torch.save(saving_object, model_save_path)


                # --- Store Results ---
                # Store configuration parameters and best dev scores achieved
                # all_results.append({
                #     'Batch Size': bs,
                #     'Learning Rate': lr,
                #     'Dropout': d,
                #     'F1 score dev': best_f1_dev, # Use best dev score found
                #     'Accuracy dev': max(accuracies_dev) if accuracies_dev else 0.0,
                #     'Test F1': results_test.get('total', {}).get('f', 0.0),
                #     'Test Acc': intent_test.get('accuracy', 0.0)
                # })

                # Save detailed results per configuration to CSV
                results_df = pd.DataFrame({
                    'Epoch': sampled_epochs,
                    'F1 dev': f1_scores_dev,
                    'Acc dev': accuracies_dev,
                    'Loss Train': losses_train_avg,
                    'Loss Dev': losses_dev_avg,
                    # Add test scores repeated for each epoch line
                    'F1 test': [results_test.get('total', {}).get('f', 0.0)] * len(sampled_epochs),
                    'Acc test': [intent_test.get('accuracy', 0.0)] * len(sampled_epochs)
                })
                csv_filename = os.path.join(results_dir, f'BERT_large_lr_{lr}_bs_{bs}_dropout_{d}.csv')
                results_df.to_csv(csv_filename, index=False)
                print(f"Detailed results saved to {csv_filename}")

                # --- Plotting ---
                # Plot F1 and Accuracy on Dev Set
                plt.figure(figsize=(10, 6)) # Use different figure number or clear figure
                plt.title(f'Dev Set Performance (lr={lr}, bs={bs}, drop={d})')
                plt.ylabel('Score')
                plt.xlabel('Epoch')
                plt.plot(sampled_epochs, f1_scores_dev, label='F1 Score (Dev)')
                plt.plot(sampled_epochs, accuracies_dev, label='Accuracy (Dev)')
                plt.legend()
                plt.xlim(1, n_epochs) # Ensure x-axis covers all epochs
                plt.ylim(0.0, 1.05) # Adjust ylim slightly
                plt.grid(True)
                res_plot_filename = os.path.join(plot_dir, f'BERT_large_res_plot_lr_{lr}_bs_{bs}_dropout_{d}.png')
                plt.savefig(res_plot_filename)
                print(f"Dev performance plot saved: '{res_plot_filename}'")
                plt.close()

                # Plot Losses
                plt.figure(figsize=(10, 6)) # Use different figure number or clear figure
                plt.title(f'Training and Dev Losses (lr={lr}, bs={bs}, drop={d})')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.plot(sampled_epochs, losses_train_avg, label='Train Loss')
                plt.plot(sampled_epochs, losses_dev_avg, label='Dev Loss')
                plt.legend()
                plt.xlim(1, n_epochs) # Ensure x-axis covers all epochs
                plt.ylim(0.0, 3.5) # Auto adjust ylim
                plt.grid(True)
                loss_plot_filename = os.path.join(plot_dir, f'BERT_large_loss_plot_lr_{lr}_bs_{bs}_dropout_{d}.png')
                plt.savefig(loss_plot_filename)
                print(f"Loss plot saved: '{loss_plot_filename}'")
                plt.close()

                print(f"Finished run #{current_configuration} of {total_configurations}")
                print("-" * 89)