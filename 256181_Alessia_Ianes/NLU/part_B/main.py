# Import necessary libraries for machine learning and data handling
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn # Import nn module
from tqdm import tqdm
from utils import *
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
    intents = [x['intent'] for x in tmp_train_raw] # Extract all intents from the raw training data
    count_y = Counter(intents) # Count the occurrences of each intent
    inputs_multi = [] # Data with intents appearing more than once
    mini_train = []   # Data with intents appearing only once

    # Iterate through the raw training data to categorize data points
    for data_point in tmp_train_raw:
        if count_y[data_point['intent']] > 1:
            # If intent occurs more than once, add to the list for splitting
            inputs_multi.append(data_point)
        else:
            # If intent occurs only once, add to a separate list
            mini_train.append(data_point)

    # Split the data with multiple samples per intent
    # Ensure there are enough samples to split (at least 2)
    if len(inputs_multi) > 1:
        # Use train_test_split to divide 'inputs_multi' into training and development sets
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


    # Build vocabulary and language structures using ALL data (train + dev + test)
    # This ensures the Lang object knows about all possible intents and slots.
    corpus = train_raw + dev_raw + test_raw
    all_words = [] # List to store all words from all utterances
    all_slots = set() # Set to store all unique slot names
    all_intents = set() # Set to store all unique intent names

    # Populate the lists/sets with data from the corpus
    for line in corpus:
        all_words.extend(line['utterance'].split()) # Add words from the utterance
        all_slots.update(line['slots'].split()) # Add slots
        all_intents.add(line['intent']) # Add the intent
        
    # Initialize Lang object - cutoff=0 means keep all words
    lang = Lang(all_words, list(all_intents), list(all_slots), cutoff=0)

    # Create Datasets for training, development, and testing
    train_dataset = IntentsAndSlotsBERT(train_raw, lang)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang)

    # Determine the output sizes for slots and intents based on the vocabulary built by the Lang object
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # Define the directory where the model checkpoints will be saved and the filename
    model_save_dir = "bin/BERT_base"
    model_filename = f"BERT_base_model_Acc.pt"
    os.makedirs(model_save_dir, exist_ok=True)

    # Define the loss functions.
    # CrossEntropyLoss is used for classification tasks (intent classification and slot tagging).
    # `ignore_index=PAD_TOKEN` tells the slot loss function to ignore predictions made for padding tokens
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
    criterion_intents = nn.CrossEntropyLoss()

    # --- Hyperparameter Setup ---
    bs = 128 # Batch size to test
    d = 0.2 # Dropout value to test
    lr = 0.00003 # Learning rate to test
    clip = 5 # Gradient clipping value to prevent exploding gradients
    runs = 5 # Number of times to run the training process
    n_epochs = 50 # Maximum number of training epochs
    patience = 3 # Early stopping patience

    if TEST:
        # ========== TESTING MODE ==========
        print("--- Loading pre-trained model and Evaluating ---")
        # Define path to the saved model
        model_save_path = os.path.join(model_save_dir, model_filename)

        if not os.path.exists(model_save_path):
            print(f"Error: Model file '{model_save_path}' does not exist. Exiting.")
            exit()

        try:
            # Load the saved object (state_dict, config, mappings) ensuring loading happens on the correct device
            saving_object = torch.load(model_save_path, map_location=device) 
            loaded_config = saving_object.get('config', {}) # Get the config dictionary, default to empty if not found.
            loaded_dropout = loaded_config.get('Dropout', d) # Get dropout used during training, or default
            loaded_lr = loaded_config.get('Learning Rate', d) # Get dropout used during training, or default
            loaded_slot2id = saving_object.get('slot2id', lang.slot2id) # Get slot to ID mapping
            loaded_intent2id = saving_object.get('intent2id', lang.intent2id) # Get intent to ID mapping

            # Create a temporary Lang object using the loaded mappings. This is crucial for correctly processing the test data
            # with the vocabulary and tag sets the model was trained on
            temp_lang = Lang([], [], []) # Initialize with empty lists
            temp_lang.slot2id = loaded_slot2id
            temp_lang.intent2id = loaded_intent2id

            # Recreate inverse mappings
            temp_lang.id2slot = {v: k for k, v in loaded_slot2id.items()}
            temp_lang.id2intent = {v: k for k, v in loaded_intent2id.items()}

            # Create the test dataset using the loaded vocabulary and tag mappings
            # This ensures consistency between the data processing and the model's learned representations
            test_dataset_eval = IntentsAndSlotsBERT(test_raw, temp_lang) 

            # Instantiate the model with parameters matching the saved model
            model = BertModelIAS(len(loaded_slot2id), len(loaded_intent2id), dropout=loaded_dropout).to(device)
             # Load the saved model weights (state_dict) into the instantiated model
            model.load_state_dict(saving_object['model'])
            model.eval() # Set model to evaluation mode 

            print(f"Model loaded successfully from {model_save_path}")

            # --- Evaluation on Test Set ---
            print("Evaluating loaded model on Test set...")
            
            # Create DataLoader for the test set
            # Use batch size from loaded config if available, otherwise default
            train_batch_size = loaded_config.get('Batch Size', bs) 
            test_loader = DataLoader(test_dataset_eval, batch_size=train_batch_size // 2, collate_fn=collate_fn_bert)

            # Perform evaluation
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
            
            print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
            print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

        except Exception as e:
            # Catch potential errors during model loading or evaluation
            print(f"Error loading model or during evaluation: {e}")
            exit()

    else:
        # ========== TRAINING MODE ==========
        print("--- Training and Testing Model ---")

        # Create DataLoaders for training, development and testing
        train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn_bert, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=bs // 2, collate_fn=collate_fn_bert)
        test_loader = DataLoader(test_dataset, batch_size=bs // 2, collate_fn=collate_fn_bert)


        # Initialize lists to store metrics across multiple training runs
        slot_f1s, intent_accs = [], [] # Stores the test slot F1 scores and the test intent accuracies for each run
        best_f1 = 0.0         # Track best F1 score on test set across all runs
        global_model = None # To store the best model state across runs
        global_epoch = -1 # To store the epoch of the best model across runs
    
        for x in tqdm(range(0, runs), desc="Running training iterations"):
          
            # Initialize Model, Optimizer
            model = BertModelIAS(out_slot, out_int, dropout=d).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_model_state = None # To store the state_dict of the best model in this run
            best_epoch_this_run = -1 # To store the epoch number of the best model in this run
            losses_train_avg = [] # Store average training loss per epoch for potential plotting
            losses_dev_avg = []   # Store average dev loss per epoch for potential plotting
            sampled_epochs = []   # Store epochs where evaluation was done
            f1_scores_dev = []    # Store F1 scores on dev set for potential plotting
            accuracies_dev = []   # Store intent accuracies on dev set for potential plotting
            best_f1_dev = 0.0     # Track best F1 score on dev set
            
            

            print(f"Starting training for {n_epochs} epochs...")
            epoch_progress_bar = tqdm(range(1, n_epochs), desc="Training process") 
            for epoch in epoch_progress_bar:
                # Train for one epoch
                avg_loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                # Store epochs and average loss for potential plotting
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

                # Check if the current F1 score on the dev set is better than the best dev F1 score
                if current_f1_dev > best_f1_dev:
                    best_f1_dev = current_f1_dev # Update the best F1 score
                    patience = 3 # Reset patience counter

                    best_model_state = model.state_dict() # Save the current model state as the best so far
                    best_epoch_this_run = epoch # Store the epoch
                     
                else:
                    patience -= 1 # If not improved, decrease patience
                
                if patience <= 0: # Early stopping
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break # Exit epoch loop

            # Load the best model state saved during training
            model.load_state_dict(best_model_state) 
            model.eval() # Ensure eval mode

            # --- Evaluation on Test Set ---
            print("Evaluating on Test set...")
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
            print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
            print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

            # Store the test results for each run
            intent_accs.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])

            if slot_f1s[-1] > best_f1:
                global_model = best_model_state # Save the best model state overall
                global_epoch = best_epoch_this_run # Save the corresponding epoch
        
        # Convert lists of results to NumPy arrays for easier calculations
        slot_f1s = np.asarray(slot_f1s)
        intent_accs = np.asarray(intent_accs)
        # Print the average of the metrics across all runs
        print('Slot F1', round(slot_f1s.mean(),3))
        print('Intent Acc', round(intent_accs.mean(), 3))

       
        # --- UNCOMMENT TO SAVE THE MODEL ---

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
        # print(f'Model correctly saved to {model_save_path}')


  

   