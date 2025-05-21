import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory contenente i file CSV
results_dir = 'results/LSTM_dropout_ADAM/'

# Funzione per estrarre i parametri dal nome del file
def extract_params_from_filename(filename):
    # Esempio di nome file: LSTM_ppl_results_lr_0.1_bs_64_emb_300_hid_300.csv
    # match = re.search(r'lr_([\d.]+)_bs_(\d+)_ed_([\d.]+)_od_\d+(\.\d*)?|\.\d+', filename)
    match = re.search(r'lr_([\d.]+)_bs_([\d]+)', filename)
    if not match:
        raise ValueError(f"Nome file non valido: {filename}. Assicurati che contenga 'lr_X_bs_Y'.")
    lr = float(match.group(1))  # Learning Rate
    bs = int(match.group(2))    # Batch Size
    # ed = float(match.group(3))  # Embedding Dropout
    # od = float(match.group(4))  # Output Dropout
    return lr, bs, #ed, od

# Leggi tutti i file CSV nella directory
# Regex to match the expected filename pattern
filename_pattern = re.compile(r'lr_([\d.]+)_bs_([\d]+)')
csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and filename_pattern.search(f)]

# Estrai i dati da ogni file e aggiungi i parametri estratti
all_results = []
for csv_file in csv_files:
    file_path = os.path.join(results_dir, csv_file)
    
    try:
        # Estrai Batch Size e Learning Rate dal nome del file
        # lr, bs, ed, od = extract_params_from_filename(csv_file)
        lr, bs = extract_params_from_filename(csv_file)
        
        # Carica solo la colonna "Test PPL" e prendi il primo valore
        # Assumiamo che tutti i valori in questa colonna siano uguali (il PPL finale del test)
        df_ppl = pd.read_csv(file_path, usecols=['PPL'])
        if df_ppl.empty:
            print(f"Warning: Skipping empty file {csv_file}")
            continue
            
        dev_ppl = df_ppl['PPL'].min()
        
        # Aggiungi i risultati come dizionario alla lista
        all_results.append({
            'Batch Size': bs,
            'Learning Rate': lr,
            # 'Embedding Dropout': ed,
            # 'Output Dropout': od,
            'PPL': dev_ppl
        })
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        continue # Skip this file if there's an error

# Crea il DataFrame finale direttamente dalla lista di dizionari
results_df = pd.DataFrame(all_results)

# Assicurati che i dati abbiano le colonne necessarie
required_columns = {'Batch Size', 'Learning Rate', 'PPL'}#'Embedding Dropout', 'Output Dropout', 'Test PPL'}
if not required_columns.issubset(results_df.columns):
    raise ValueError(f"Il DataFrame deve contenere le seguenti colonne: {required_columns}")



# Creiamo una pivot table per la heatmap
pivot_table = results_df.pivot_table(
    values='PPL',
    index='Batch Size',  # Righe: Batch Size
    columns='Learning Rate'  # Colonne: Learning Rate
)

# Visualizziamo la heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Final PPL'})
plt.title("Heatmap of Dev PPL for Different Configurations")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.tight_layout()

# Salviamo la heatmap come immagine
heatmap_filename = os.path.join(results_dir, 'plots/heatmap_dev_ppl.png')
os.makedirs(os.path.dirname(heatmap_filename), exist_ok=True)  # Crea la cartella "plots" se non esiste
plt.savefig(heatmap_filename)
plt.close()
print(f"Heatmap saved: '{heatmap_filename}'")

# Troviamo la migliore configurazione
pd.DataFrame(all_results).to_csv('results/LSTM_dropout_ADAM/all_results_dev.csv', index=False)
print(f'All results successfully saved in results/LSTM_dropout_ADAM/all_results_dev.csv')

best_result = min(all_results, key=lambda x: x['PPL'])
print(f"Best configuration: {best_result}")

# Salviamo la migliore configurazione in un file CSV
best_result_df = pd.DataFrame([best_result])
best_config_filename = os.path.join(results_dir, 'best_configuration_dev.csv')
best_result_df.to_csv(best_config_filename, index=False)
print(f'Best configuration successfully saved in {best_config_filename}')