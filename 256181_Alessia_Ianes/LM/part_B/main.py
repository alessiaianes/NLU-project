# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pandas as pd  # for DataFrame
import os
import math
import time

TEST = True


if __name__ == "__main__":
    DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])


    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)


    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 300 # Hidden size to test
    emb_size = 300 # Embedding size to test
    vocab_len = len(lang.word2id)
    clip = 5 # Clip the gradient
    lr = 3.7 # Learning rates to test
    bs = 32
    ed =  0.15
    od = 0.4

    # Create a directory to save the results, if it doesn't exist
    # os.makedirs('results/LSTM_wt_vd_avsgd/plots', exist_ok=True)
    models_dir = os.makedirs('bin/LSTM_wt_vd_avsgd', exist_ok=True)
    model_filename = f'LSTM_wt_vd_avsgd_model.pt'
    print("prova:", os.path.join(models_dir, model_filename) )

   
    # NT-AvSGD parameters
    non_monotonic_trigger_window = 5 # 'n' in the paper's algorithm
    # asgd_trigger_patience = non_monotonic_trigger_window # How many epochs to wait after condition met
    # asgd_patience_reset = 7 # Patience reset value when ASGD is triggered

    
    
    # Define the collate function
    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    if TEST:
        print("--- Loading pre-trained model and Evaluating ---")
        model = LM_LSTM_wt_vd(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], n_layers=1, emb_dropout=ed, out_dropout=od).to(DEVICE)
        model_path = os.path.join(models_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' does not exist. Exiting.")
            exit()
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model or during evaluation: {e}")
            exit()
        model.eval()
        with torch.no_grad():
            final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"Final PPL on test set: {final_ppl:.2f}")
    else:
        print("--- Training LSTM model with weight tying and variational dropout ---")
        # Initialize the model
        model = LM_LSTM_wt_vd(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], n_layers=1, emb_dropout=ed, out_dropout=od).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.SGD(model.parameters(), lr=lr)
        

        n_epochs = 100
        x_min, x_max = 0, n_epochs  # Limiti per l'asse x (epoche)
        ppl_min, ppl_max = 0, 500  # Limiti per l'asse y (PPL)
        loss_min, loss_max = 0, 10  # Limiti per l'asse y (Loss)
        patience = 7
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs))
        
        ppl_values = []
        avg_model_sd = None # State dict for the averaged model
        trigger = 0 # Averaging trigger epoch
        t = 0 # Validation check counter
        num_avg_steps = 0


        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:
                epoch_start_time = time.time()
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                ppl_values.append(ppl_dev) # add PPL to list

                pbar.set_description(
                    f"Epoch: {epoch} PPL: {ppl_dev:.2f}"
                )

                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 7
                    # asgd_trigger_counter = 0
                else:
                    patience -= 1

                if patience <= 0:
                    break

                if trigger == 0 and t >= non_monotonic_trigger_window:
                    start_idx = t - non_monotonic_trigger_window
                    min_ppl = min(ppl_values[start_idx:t])

                    if ppl_dev > min_ppl:
                        print(f"Triggering ASGD at epoch {epoch}")
                        trigger = epoch
                
                t += 1

            if trigger > 0 and epoch > trigger:
                num_avg_steps += 1
                current_sd = model.state_dict()
                if avg_model_sd is None:
                    print(f"Starting averaging with model from epoch {epoch}")
                    # Initialize avg_model_sd on CPU
                    avg_model_sd = {k: v.cpu().clone() for k, v in current_sd.items()}
                else:
                        # Update running average (current_sd needs to be moved to CPU within the function)
                    avg_model_sd = update_avg_model(avg_model_sd, current_sd, num_avg_steps)



        if avg_model_sd is not None:
            print("\nUsing averaged model for final evaluation.")
            # Create a new model instance and load the averaged weights
            final_model = LM_LSTM_wt_vd(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], n_layers=1, emb_dropout=ed, out_dropout=od)
            final_model.load_state_dict(avg_model_sd) # Load CPU state dict
            final_model = final_model.to(DEVICE) # Move to evaluation device
        elif best_model is not None:
            final_model = LM_LSTM_wt_vd(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], n_layers=1, emb_dropout=ed, out_dropout=od)
            final_model.load_state_dict(best_model) # Load CPU state dict
            final_model = final_model.to(DEVICE) # Move to evaluation device
        else:
            print("\nAveraging never triggered. Using last model state for final evaluation.")
            # The current 'model' is the final model (already on DEVICE)
            final_model = model
            

        # best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, final_model)

        # --- UNCOMMENT TO SAVE THE MODEL ---
        # path = os.path.join(models_dir, model_filename)
        # torch.save(model.state_dict(), path)
   