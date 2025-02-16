import os
import pandas as pd

# Percorso della directory contenente i file TXT
directory = r"C:\Users\paolo\Desktop\progetto ict\dati base\taxi_log_2008_by_id"

# Percorso del file in output
output_file = r"C:\Users\paolo\Desktop\input.txt"

all_data = [] #qui tengo tutti i dati

# Ottieni e ordina i file numericamente; lambda estrae i numeri dai nomi dei file e li converte in interi per l'ordinamento numerico.
file_list = sorted(os.listdir(directory), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

#crea i vari dataframe
for filename in file_list:
    if filename.endswith(".txt"):  # considera solo i file TXT
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path, sep=",", header=None, names=["Col1", "Col2", "Col3", "Col4"])
        all_data.append(df)

# Combina tutti i DataFrame in un unico DataFrame
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False, sep=",", header=False)  # Usa , per separare i valori
    print("File TXT creato con successo:", output_file)
else:
    print("ERRORE")