# Import necessary libraries and custom modules
from functions import *
from utils import *
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import copy
import os 
import math
import numpy as np 

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Set the TEST flag to True to evaluate the saved model, False to train a new one
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TEST = True

if __name__ == "__main__":
    DEVICE = 'cuda:0' # Computation device, change to 'cpu' if you don't have a GPU

    # Load the raw text data for training, validation, and testing sets
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Create a vocabulary mapping words to IDs using the training data
    # Include special tokens "<pad>" (for padding) and "<eos>" (end of sentence)
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    # Initialize the Lang object, which stores vocabulary and provides word-to-ID/ID-to-word mappings
    lang = Lang(train_raw, ["<pad>", "<eos>"])


    # Create dataset objects for each split (train, dev, test) using the Lang object
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # Define hyperparameters for the LSTM model and training process
    hid = 200 # Hidden size to test
    emb = 300 # Embedding size to test
    vocab_len = len(lang.word2id)
    clip = 5 # Gradient clipping threshold to prevent exploding gradients
    lr = 0.0001 # Learning rates to test
    bs = 32 # Batch size to test

    # Directory to save the trained model and the model filename
    models_dir = f'bin/LSTM_dropout_ADAM'
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f'LSTM_dropout_ADAM_model.pt'

    # Create DataLoaders to iterate over the datasets in batches
    # The collate_fn handles padding sequences within each batch to the maximum length in that batch
    # `partial` is used to pass fixed arguments (pad_token ID, DEVICE) to the collate_fn
    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    
    # Define the loss function for training and evaluation
    # Ignore the padding token index when calculating loss
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # --- Conditional Execution: Evaluation or Training ---
    if TEST:
        # ========== EVALUATION MODE ==========
        print("--- Loading pre-trained model and Evaluating ---")
        # Initialize the model architecture
        model = LM_LSTM_dropout(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        # Define the path to the saved model file
        model_path = os.path.join(models_dir, model_filename) 

        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' does not exist. Exiting.")
            exit()

        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            # Catch potential errors during model loading
            print(f"Error loading model or during evaluation: {e}")
            exit()

        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            # Evaluate the model on the test set and print the final PPL
            final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"Final PPL on test set: {final_ppl:.2f}")

        
    else:
        # ========== TRAINING MODE ==========
        print("--- Training LSTM model with dropout and AdamW optimizer ---")

        # Initialize the model
        model = LM_LSTM_dropout(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        # Apply weight initialization to the model's parameters
        model.apply(init_weights)
        # Initialize the AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        

        # Setting training parameters
        n_epochs = 100
        patience = 3 # Number of epochs to wait for improvement before early stopping
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        ppl_values = []

        best_ppl = math.inf # Initialize best validation perplexity to infinity
        best_model = None # Initialize the variable to store the best performing model state

        pbar = tqdm(range(1,n_epochs))
        
        # --- Training Loop ---
        for epoch in pbar:
            # Train the model for one epoch and get the average loss
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:
                # Record the current epoch number and losses for potential plotting
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                # Evaluate the model on the validation data and store PPL and loss values for potential plotting
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                ppl_values.append(ppl_dev)

                # Update the progress bar description with current epoch PPL
                pbar.set_description(f"Epoch: {epoch} PPL: {ppl_dev:.2f}")
               
                # Check if the current validation PPL is better than the best recorded PPL
                if  ppl_dev < best_ppl:
                    best_ppl = ppl_dev # Update the best PPL
                    best_model = copy.deepcopy(model).to('cpu') # Save the best model state
                    patience = 3 # Reset patience 
                else:
                    patience -= 1 # Decrease patience if no improvement
                
                if patience <= 0: # Early stopping with patience
                    print(f"Early stopping triggered at epoch {epoch} for lr={lr}, bs={bs}, emb={emb}, hid={hid}")
                    break # Exit the training loop

        # Move the best saved model back to the target device
        best_model.to(DEVICE)
        # Evaluate the best model on the test set
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f"Final PPL on test set after training: {final_ppl:.2f}")


        # ========== UNCOMMENT TO SAVE THE MODEL ==========
        # path = os.path.join(models_dir, model_filename)  # Ensure the directory exists
        # print(f"Saving best model to {path}")
        # torch.save(model.state_dict(), path)