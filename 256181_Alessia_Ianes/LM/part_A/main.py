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
import seaborn as sns

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
    hid = 200 # Hidden size to test
    emb = 300 # Embedding size to test
    vocab_len = len(lang.word2id)
    clip = 5 # Clip the gradient
    lr = 0.0001 # Learning rates to test with SGD
    #lr_values = [0.0001, 0.0005, 0.001, 0.005, 0.01] # Learning rates to test with AdamW
    bs = 32

    # Create a directory to save the results, if it doesn't exist
    # os.makedirs('results/LSTM_dropout_ADAM/plots', exist_ok=True)
    models_dir = f'bin/LSTM_dropout_ADAM'
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f'LSTM_dropout_ADAM_model.pt'

    # print(f"Training with batch size: {bs} emb size: {emb} hid size: {hid} and lr {lr}")
    # Define the collate function
    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if TEST:
        print("--- Loading pre-trained model and Evaluating ---")
        model = LM_LSTM_dropout(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model_path = os.path.join(models_dir, model_filename) #'bin/LSTM_dropout_ADAM/LSTM_dropout_ADAM_model.pt' #

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
        print("--- Training LSTM model with dropout and AdamW optimizer ---")


        # Initialize the model
        model = LM_LSTM_dropout(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        # model = LM_LSTM(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        # model = LM_RNN(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        


        n_epochs = 100
        x_min, x_max = 0, n_epochs  # Limiti per l'asse x (epoche)
        ppl_min, ppl_max = 0, 500  # Limiti per l'asse y (PPL)
        loss_min, loss_max = 0, 10  # Limiti per l'asse y (Loss)
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs))
        
        ppl_values = []
        

        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                ppl_values.append(ppl_dev) # add PPL to list
                pbar.set_description(f"Epoch: {epoch} PPL: {ppl_dev:.2f}")
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                
            if patience <= 0: # Early stopping with patience
                print(f"Early stopping triggered at epoch {epoch} for lr={lr}, bs={bs}, emb={emb}, hid={hid}")
                break # Not nice but it keeps the code clean

        best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

        # --- UNCOMMENT TO SAVE THE MODEL ---
        # path = os.path.join(models_dir, model_filename)  # Ensure the directory exists
        # torch.save(model.state_dict(), path)
   
   
    # Inside the loop, append results:

    # Save the results in a CSV file
    # results_df = pd.DataFrame({
    #     'Epoch': sampled_epochs,
    #     'PPL': ppl_values,
    #     'Test PPL': [final_ppl] * len(sampled_epochs)
    # })
    # csv_filename = f'results/LSTM_dropout_ADAM/LSTM_ppl_results_lr_{lr}_bs_{bs}_emb_{emb}_hid_{hid}.csv'
    # results_df.to_csv(csv_filename, index=False)
    # print(f'CSV file successfully saved in {csv_filename}')
    

    # # Create ppl_dev plot
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    # ax1.plot(sampled_epochs, ppl_values, label='PPL Dev', color='red')
    # ax1.set_title(f'PPL Dev for lr={lr}, bs={bs}, emb={emb}, hid={hid}')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('PPL')
    # ax1.set_xlim(x_min, x_max)
    # ax1.set_ylim(ppl_min, ppl_max)
    # ax1.legend()
    # ax1.grid()

    # # Save ppl_dev plot
    # ppl_plot_filename = f'results/LSTM_dropout_ADAM/plots/LSTM_ppl_plot_lr_{lr}_bs_{bs}_emb_{emb}_hid_{hid}.png'
    # plt.savefig(ppl_plot_filename)
    # plt.close(fig)
    # print(f"PPL plot saved: '{ppl_plot_filename}'")

    # # Create the loss plot
    # fig, ax2 = plt.subplots(figsize=(10, 5))
    # ax2.plot(sampled_epochs, losses_train, label='Train Loss', color='orange')
    # ax2.plot(sampled_epochs, losses_dev, label='Dev Loss', color='blue')
    # ax2.set_title(f'Train and Dev Loss for lr={lr}, bs={bs}, emb={emb}, hid={hid}')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Loss')
    # ax2.set_xlim(x_min, x_max)
    # ax2.set_ylim(loss_min, loss_max)
    # ax2.legend()
    # ax2.grid()

    # # Save loss plot
    # loss_plot_filename = f'results/LSTM_dropout_ADAM/plots/LSTM_loss_plot_lr_{lr}_bs_{bs}_emb_{emb}_hid_{hid}.png'
    # plt.savefig(loss_plot_filename)
    # plt.close(fig)

    
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
