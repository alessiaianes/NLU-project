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

# --- Control Variable ---
# Set to True to load the saved model, False to train and test a new one.
TEST = True

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

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    hid_size = 200 # Hidden size
    emb_size = 300
    bs = 32 # Batch size
    d = 0.4 # Dropout values
    lr = 0.0007 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Create a directory to save the results, if it doesn't exist
    # os.makedirs('results/LSTM_drop/plots', exist_ok=True)
    model_save_dir = "bin/LSTM_drop"
    os.makedirs(model_save_dir, exist_ok=True)
    #all_results = [] # To store the results of each configuration

    

   
    if TEST:
        print("--- Loading pre-trained model and Evaluating ---")
        model_filename = f"LSTM_drop_model_acc.pt"
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
            loaded_hid = loaded_config.get('Hidden Size', hid_size) # Get hidden size used during training, or default
            loaded_w2id = saving_object.get('w2id', lang.word2id) # Word to ID mapping
            loaded_slot2id = saving_object.get('slot2id', lang.slot2id) # Slot to ID mapping
            loaded_intent2id = saving_object.get('intent2id', lang.intent2id) # Intent to ID mapping

            temp_lang = Lang([], [], []) # Inizializzazione base
            temp_lang.word2id = loaded_w2id
            temp_lang.slot2id = loaded_slot2id
            temp_lang.intent2id = loaded_intent2id
            
            # Potrebbe essere necessario aggiungere anche le mappature inverse se usate
            temp_lang.id2word = {v: k for k, v in loaded_w2id.items()}
            temp_lang.id2slot = {v: k for k, v in loaded_slot2id.items()}
            temp_lang.id2intent = {v: k for k, v in loaded_intent2id.items()}

            # Gestisci il PAD_TOKEN: assicurati che sia presente nelle mappature caricate
            # Se PAD_TOKEN non è una parola reale, potresti doverlo aggiungere manualmente
            # o assicurarti che sia stato aggiunto correttamente durante il salvataggio.
            # Ad esempio, se il token stringa per PAD è '<PAD>':
            # if '<PAD>' not in temp_lang.word2id:
            #    temp_lang.word2id['<PAD>'] = PAD_TOKEN # O l'ID che usi per il padding

            # 2. Crea il Dataset di Test usando le mappature CARICATE (tramite temp_lang)
            #    Assicurati che test_raw sia disponibile qui.
            test_dataset_eval = IntentsAndSlots(test_raw, temp_lang) 
            

            # Instantiate the model with parameters matching the saved model
            model = ModelIAS(loaded_hid, len(loaded_slot2id), len(loaded_intent2id), emb_size, len(loaded_w2id), pad_index=PAD_TOKEN, dropout=loaded_dropout).to(device)
            model.load_state_dict(saving_object['model'])
            model.eval() # Set model to evaluation mode (disables dropout)

            print(f"Model loaded successfully from {model_save_path}")
            print(f"Loaded configuration: {loaded_config}")
            # print(f'Learning rate: {loaded_config.get('Learning Rate', 0.0)}')

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
            print(f"Error loading model or during evaluation: {e}")
            exit()

    else:
            # TEST is False, proceed with training and testing
            print("--- Training and Testing Model ---")
            runs = 5
            f1s, accs = [], []
            global_model = None
            global_epoch = -1
            
            # Dataloader instantiations
            train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn,  shuffle=True)
            dev_loader = DataLoader(dev_dataset, batch_size=bs//2, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=bs//2, collate_fn=collate_fn)


            for run in range(0, runs):

                model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, dropout=d).to(device)
                model.apply(init_weights)

                optimizer = optim.Adam(model.parameters(), lr=lr)
                

                n_epochs = 200
                patience = 3
                losses_train = []
                losses_dev = []
                sampled_epochs = []
                f1 = []
                accuracy = []
                best_f1 = 0
                best_f1_test = 0
                best_model_state = None
                best_epoch_this_run = -1
                print(f"Starting training for {n_epochs} epochs...")
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
                            best_model_state = model.state_dict()
                            best_epoch_this_run = x 
                        else:
                            patience -= 1
                        if patience <= 0: # Early stopping with patience
                            break # Not nice but it keeps the code clean

                
                model.load_state_dict(best_model_state)
                model.eval() # Ensure eval mode

                # --- Evaluation on Test Set ---
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
                accs.append(intent_test['accuracy'])
                f1s.append(results_test['total']['f'])

                if f1s[-1] > best_f1_test:
                    global_model = best_model_state
                    global_epoch = best_epoch_this_run

            f1s = np.asarray(f1s)
            accs = np.asarray(accs)
            print('Slot F1', round(f1s.mean(),3), '+-', round(f1s.std(),3))
            print('Intent Acc', round(accs.mean(), 3), '+-', round(accs.std(), 3))

            
            # --- UNCOMMENT TO SAVE THE MODEL ---

            config_params = {
                            'Model': 'LSTM dropout', # Explicitly state the model used
                            'Embedding Size': emb_size,
                            'Hidden Size': hid_size,
                            'Batch Size': bs,
                            'Learning Rate': lr,
                            'Dropout': d
                        }

            model_filename = f"LSTM_drop_model_f1.pt"
            model_save_path = os.path.join(model_save_dir, model_filename)
            saving_object = {
                "epoch": x,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "w2id": lang.word2id,
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id,
                "config": config_params, # Save the config that led to this best score
            }
            torch.save(saving_object, model_save_path)
            print("model correctly saved")