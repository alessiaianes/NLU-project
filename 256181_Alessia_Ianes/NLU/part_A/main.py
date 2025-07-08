# Import necessary libraries for machine learning and data handling
from functions import *
from utils import *
from model import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Set the TEST flag to True to evaluate the saved model, False to train a new one
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TEST = True

if __name__ == "__main__":
    # This section prepares the data, splitting the training set for validation
    # Define the portion of the training data to be used for the development (validation) set
    portion = 0.10

    # Extract intents from the raw training data (tmp_train_raw)
    intents = [x['intent'] for x in tmp_train_raw] 
    # Count the occurrences of each intent to facilitate stratified splitting
    count_y = Counter(intents)

    # Lists to hold data for splitting
    labels = [] # Stores intents
    inputs = [] # Stores utterances and slots data
    mini_train = [] # Stores examples whose intents appear only once

    # Iterate through the raw training data to separate examples
    for id_y, y in enumerate(intents):
        # Intents that appear more than once are candidates for splitting
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else: # Intents that appear only once are kept directly in the training set to avoid losing information
            mini_train.append(tmp_train_raw[id_y])
    
    
    # Stratified split: Divide the 'inputs' and 'labels' into training and development sets.
    # Stratification ensures that the proportion of intents is maintained in both sets.
    # 'portion' determines the size of the development set
    # 'random_state' ensures reproducibility of the split
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    
    # Add the examples with single-occurrence intents back to the training set
    X_train.extend(mini_train)
    # Update the train_raw and dev_raw variables with the prepared data
    train_raw = X_train
    dev_raw = X_dev


    # Build the vocabulary, intents, and slots from the entire dataset (train, dev, test)
    # This ensures the model knows about all possible words, intents, and slots
    # 'cutoff=0' in Lang likely means no minimum frequency cutoff for words, including all words
    words = sum([x['utterance'].split() for x in train_raw], []) # Concatenate all words from train_raw
    corpus = train_raw + dev_raw + test_raw # Combine all data splits for vocabulary building
    slots = set(sum([line['slots'].split() for line in corpus],[])) # Extract unique slot tags
    intents = set([line['intent'] for line in corpus]) # Extract unique intent labels

    # Initialize the Lang class
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets for training, development, and testing
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Define the loss function for slot tagging and intent classification
    # nn.CrossEntropyLoss computes cross-entropy loss, suitable for classification
    # 'ignore_index=PAD_TOKEN' ensures that padding tokens in the target sequences are ignored during loss calculation
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- Hyperparameter Configuration ---
    hid_size = 200 # Hidden size to test
    emb_size = 300 # Embedding size to test
    bs = 64 # Batch size to test
    d = 0.4 # Dropout value to test
    lr = 0.0007 # Learning rate to test
    clip = 5 # Gradient clipping threshold to prevent exploding gradients

    # Determine the output dimensions based on the vocabulary and language mappings
    out_slot = len(lang.slot2id)     # Number of possible slot tags + 1 (for potential UNK or padding)
    out_int = len(lang.intent2id)    # Number of possible intents
    vocab_len = len(lang.word2id)    # Size of the vocabulary (number of unique words)

    # Define the directory path for saving model checkpoints and the model filename (to test or to save)
    model_save_dir = "bin/LSTM_drop"
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = f"LSTM_drop_model_f1.pt"

    

   
    if TEST:
        # ========== EVALUATION MODE ==========
        print("--- Loading pre-trained model and Evaluating ---")
        model_save_path = os.path.join(model_save_dir, model_filename)

        if not os.path.exists(model_save_path):
            print(f"Error: Model file '{model_save_path}' does not exist. Exiting.")
            exit()

        try:
            # Load the saved object containing the model state_dict, configuration, and mappings
            # 'map_location=device' ensures the model is loaded onto the correct device
            saving_object = torch.load(model_save_path, map_location=device) 

            # Extract necessary information from the saved object
            # Use .get() with defaults to handle potentially missing keys
            loaded_config = saving_object.get('config', {}) # Dictionary of training parameters
            loaded_dropout = loaded_config.get('Dropout', d) # Get dropout used during training
            loaded_hid = loaded_config.get('Hidden Size', hid_size) # Get hidden size used during training
            loaded_w2id = saving_object.get('w2id', lang.word2id) # Get word to ID mapping
            loaded_slot2id = saving_object.get('slot2id', lang.slot2id) # Get slot to ID mapping
            loaded_intent2id = saving_object.get('intent2id', lang.intent2id) # Get intent to ID mapping


            # Create a temporary Lang object using the loaded mappings. This is crucial for correctly processing the test data
            # with the vocabulary and tag sets the model was trained on
            temp_lang = Lang([], [], []) # Initialize with empty lists
            temp_lang.word2id = loaded_w2id
            temp_lang.slot2id = loaded_slot2id
            temp_lang.intent2id = loaded_intent2id
            
            # Recreate inverse mappings
            temp_lang.id2word = {v: k for k, v in loaded_w2id.items()}
            temp_lang.id2slot = {v: k for k, v in loaded_slot2id.items()}
            temp_lang.id2intent = {v: k for k, v in loaded_intent2id.items()}

            # Create the test dataset using the loaded vocabulary and tag mappings
            # This ensures consistency between the data processing and the model's learned representations
            test_dataset_eval = IntentsAndSlots(test_raw, temp_lang) 
            

            # Instantiate the model with parameters from the saved model
            model = ModelIAS(loaded_hid, len(loaded_slot2id), len(loaded_intent2id), emb_size, len(loaded_w2id), pad_index=PAD_TOKEN, dropout=loaded_dropout).to(device)
            model.load_state_dict(saving_object['model'])
            model.eval() # Set model to evaluation mode 

            print(f"Model loaded successfully from {model_save_path}")

            # --- Evaluation on Test Set ---
            print("Evaluating loaded model on Test set...")
            
            # Create DataLoader for the test set
            # Use batch size from loaded config if available, otherwise default
            train_batch_size = loaded_config.get('Batch Size', bs) 
            test_loader = DataLoader(test_dataset_eval, batch_size=train_batch_size//2, collate_fn=collate_fn)

            # Perform evaluation
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
            
            print(f'Test Slot F1: {results_test.get("total", {}).get("f", "N/A")}')
            print(f'Test Intent Accuracy: {intent_test.get("accuracy", "N/A")}')

        except Exception as e:
            # Catch potential errors during loading or evaluation
            print(f"Error loading model or during evaluation: {e}")
            exit()

    else:
            # ================ TRAINING MODE ==========
            print("--- Training and Testing Model ---")
            runs = 5 # Number of independent training runs to average results
            f1s, accs = [], [] # Lists to store the final test metrics for each run
            best_f1_test = 0 # Initialize the best F1 score on the test set across all runs
            # Variables to keep track of the best model across all runs
            global_model = None
            global_epoch = -1
            
            # Dataloader instantiations
            train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn,  shuffle=True)
            dev_loader = DataLoader(dev_dataset, batch_size=bs//2, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=bs//2, collate_fn=collate_fn)

            

            for run in range(0, runs):
                # Instantiate the model with the specified hyperparameters for this run
                # Move the model to the target device 
                model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, dropout=d).to(device)
                 # Initialize model weights using a specific initialization function
                model.apply(init_weights)
                # Initialize the optimizer
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                
                # --- Training Loop Configuration ---
                n_epochs = 200
                patience = 3
                losses_train = []
                losses_dev = []
                sampled_epochs = []
                f1 = [] # List to store slot tagging F1 scores on the development set
                accuracy = [] # List to store intent classification accuracies on the development set
                best_f1 = 0 # Initialize the best F1 score achieved on the development set in this run
                best_model_state = None # Store the state dictionary of the best model found so far
                best_epoch_this_run = -1 # Store the epoch number corresponding to the best model stat

                print(f"Starting training for {n_epochs} epochs...")
                for x in tqdm(range(1,n_epochs)):
                    loss = train_loop(train_loader, optimizer, criterion_slots, 
                                    criterion_intents, model, clip=clip)
                    if x % 5 == 0: # We check the performance every 5 epochs
                        # Append the current epoch, training loss, and development loss for potential plotting
                        sampled_epochs.append(x)
                        losses_train.append(np.asarray(loss).mean())
                        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                                    criterion_intents, model, lang)
                        losses_dev.append(np.asarray(loss_dev).mean())
                        
                        # Record the performance metrics
                        f1.append(results_dev['total']['f'])
                        accuracy.append(intent_res['accuracy'])

                        if f1[-1] > best_f1:
                            best_f1 = f1[-1]
                            patience = 3 # Reset patience counter
                            best_model_state = model.state_dict() # Save the current model state as the best so far
                            best_epoch_this_run = x 
                        else:
                            patience -= 1 # If no improvement, decrement patience
                        if patience <= 0: # Early stopping
                            break # Exit the epoch loop

                # Load the best model state saved during training
                model.load_state_dict(best_model_state)
                model.eval() # Ensure eval mode

                # --- Evaluation on Test Set ---
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
                # Store the test results for each run
                accs.append(intent_test['accuracy'])
                f1s.append(results_test['total']['f'])

                # Track the best test performance across runs
                if f1s[-1] > best_f1_test:
                    global_model = best_model_state # Save the best model state overall
                    global_epoch = best_epoch_this_run # Save the corresponding epoch

            # Convert lists of results to NumPy arrays for easier calculations
            f1s = np.asarray(f1s)
            accs = np.asarray(accs)
            # Print the average of the metrics across all runs
            print('Slot F1', round(f1s.mean(),3))
            print('Intent Acc', round(accs.mean(), 3))

            
            # --- UNCOMMENT TO SAVE THE MODEL ---

            # config_params = {
            #                 'Model': 'LSTM dropout', # Explicitly state the model used
            #                 'Embedding Size': emb_size,
            #                 'Hidden Size': hid_size,
            #                 'Batch Size': bs,
            #                 'Learning Rate': lr,
            #                 'Dropout': d
            #             }

            
            # model_save_path = os.path.join(model_save_dir, model_filename)
            # saving_object = {
            #     "epoch": x,
            #     "model": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "w2id": lang.word2id,
            #     "slot2id": lang.slot2id,
            #     "intent2id": lang.intent2id,
            #     "config": config_params, # Save the config that led to this best score
            # }
            # torch.save(saving_object, model_save_path)
            # print(f"Model correctly saved in {model_save_path}")