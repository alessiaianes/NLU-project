# Imports at the top
import os
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
from model import *
from functions import *

# --- Control Variable ---
# Set to True to load the saved model, False to train and test a new one.
TEST = True 


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
    #y_test = [x['intent'] for x in test_raw]

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

    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    model_save_dir = "bin/BERT_base"
    os.makedirs(model_save_dir, exist_ok=True)

    # Ignore index PAD_TOKEN (0) in slot loss calculation
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
    criterion_intents = nn.CrossEntropyLoss()

    #     # --- Hyperparameter Search Setup ---
    #hid_size_values = [200, 300] # Example hidden sizes
    emb_size = 300 # Embedding size (might be fixed by BERT model, e.g., 768)
    bs = 128 # Reduced batch sizes for potentially faster testing
    d = 0.2 # Example dropout values
    lr = 0.00003 # Lower learning rates often work better with Adam/BERT
    clip = 5 # Gradient clipping value
    runs = 5
    # Training Parameters
    n_epochs = 50 # Reduced epochs for quicker testing; increase for full run (e.g., 50 or 100)
    patience = 3 # Early stopping patience

    if TEST:
        print("--- Loading pre-trained model and Evaluating ---")
        # Define path to the saved model
        model_filename = f"BERT_base_model.pt" # File saved by the training process
        model_save_path = os.path.join(model_save_dir, model_filename)

        if not os.path.exists(model_save_path):
            print(f"Error: Model file '{model_save_path}' does not exist. Exiting.")
            exit()

        try:
            # Load the saved object (state_dict, config, mappings)
            # Ensure loading happens on the correct device ('cpu' if GPU not available/needed)
            saving_object = torch.load(model_save_path, map_location=device) 
            loaded_config = saving_object.get('config', {})
            loaded_dropout = loaded_config.get('Dropout', d) # Get dropout used during training, or default

            # Instantiate the model with parameters matching the saved model
            model = BertModelIAS(out_slot, out_int, dropout=loaded_dropout).to(device)
            model.load_state_dict(saving_object['model'])
            model.eval() # Set model to evaluation mode (disables dropout)

            print(f"Model loaded successfully from {model_save_path}")
            print(f"Loaded configuration: {loaded_config}")
            print(f'Learning rate: {loaded_config.get('Learning Rate', 0.0)}')

            # --- Evaluation on Test Set ---
            print("Evaluating loaded model on Test set...")
            
            # Create DataLoader for the test set
            # Use batch size from loaded config if available, otherwise default
            train_batch_size = loaded_config.get('Batch Size', bs) 
            test_loader = DataLoader(test_dataset, batch_size=train_batch_size // 2, collate_fn=collate_fn_bert)

            # Perform evaluation
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
            
            print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
            print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

        except Exception as e:
            print(f"Error loading model or during evaluation: {e}")
            exit()

    else:
        # test is False, proceed with training and testing
        print("--- Training and Testing Model ---")

        # Create DataLoaders for training and development
        train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=bs // 2, collate_fn=collate_fn_bert)
        # test_loader created earlier, it's needed after the training loop too.
        test_loader = DataLoader(test_dataset, batch_size=bs // 2, collate_fn=collate_fn_bert)

        # Variables to store results across multiple runs
        # all_runs_results = [] # Store dicts containing results for each run

        # --- Training Loop (Multiple runs) ---
        slot_f1s, intent_acc = [], []
        global_model = None # To store the best model state across runs
        global_epoch = -1 # To store the epoch of the best model across runs
    
        for x in tqdm(range(0, runs), desc="Running training iterations"):
          
            # Initialize Model, Optimizer, Loss Functions
            model = BertModelIAS(out_slot, out_int, dropout=d).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            patience_counter = 0
            best_model_state = None # To store the state_dict of the best model in this run
            best_epoch_this_run = -1 # To store the epoch number of the best model in this run
            losses_train_avg = [] # Store average training loss per epoch (or per N steps)
            losses_dev_avg = []   # Store average dev loss per epoch
            sampled_epochs = []   # Store epochs where evaluation was done
            f1_scores_dev = []    # Store F1 scores on dev set
            accuracies_dev = []   # Store intent accuracies on dev set
            best_f1_dev = 0.0     # Track best F1 score on dev set
            best_f1 = 0
            

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

                    best_model_state = model.state_dict()
                    best_epoch_this_run = epoch # Store the epoch
                     
                    # torch.save(model.state_dict(), f"{results_dir}/best_model_f1_config_{current_configuration}.pt")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break # Exit epoch loop

            
            model.load_state_dict(best_model_state)
            model.eval() # Ensure eval mode

            # --- Evaluation on Test Set ---
            print("Evaluating on Test set...")
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
            print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
            print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')
            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])

            if slot_f1s[-1] > best_f1:
                global_model = best_model_state
                global_epoch = best_epoch_this_run
        
        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)
        print('Slot F1', round(slot_f1s.mean(),3))
        print('Intent Acc', round(intent_acc.mean(), 3))

       
        # --- TO SAVE THE MODEL ---

        # config_params = {
        #     'Model': 'BERT-base-uncased', # Explicitly state the model used
        #     'Batch Size': bs,
        #     'Learning Rate': lr,
        #     'Dropout': d
        # }

        # model_filename = f"BERT_base_model.pt"
        # model_save_path = os.path.join(model_save_dir, model_filename)
        # saving_object = {
        #     "epoch": global_epoch,
        #     "model": global_model,
        #     "optimizer": optimizer.state_dict(),
        #     "slot2id": lang.slot2id,
        #     "intent2id": lang.intent2id,
        #     "config": config_params, # Save the config that led to this best score
        # }
        # torch.save(saving_object, model_save_path)
        # print("Saving model for best intent accuracy with configuration:", config_params)


  

   