import pandas as pd
import numpy as np

# Carico i dati, non voglio l'header senno potrei avere problemi dopo
file_path = r'C:\Users\paolo\Desktop\progetto ict\input.txt'
df = pd.read_csv(file_path, sep=",", header=None, names=["ID", "DataOra", "Lat", "Lon"])

# Converto la colonna DataOra per evitare errori
df["DataOra"] = pd.to_datetime(df["DataOra"])

# scelgo la dimensione della cella, 0.03 sembra un buon compromesso!
cell_size = 0.03

# Crea colonne per assegnare le celle
df["Cell_X"] = (df["Lat"] // cell_size).astype(int)
df["Cell_Y"] = (df["Lon"] // cell_size).astype(int)

# Estraggo l'ora dalla data dato che la data non mi interessa essendo solo pochi giorni (no stagionalità)
df["Ora"] = df["DataOra"].dt.hour

# Conto i taxi per cella e ora
heatmap = df.groupby(["Ora", "Cell_X", "Cell_Y"])["ID"].nunique().reset_index()

# Rinomino la colonna con il numero di taxi per cella
heatmap.rename(columns={"ID": "Taxi_Count"}, inplace=True)

# Trovo le celle presenti nel dataset
celle_presenti = df[["Cell_X", "Cell_Y"]].drop_duplicates()

# Creo un dataframe con tutte le combinazioni di Ora e celle presenti
all_combinations = pd.MultiIndex.from_product([range(24), celle_presenti["Cell_X"].unique(), celle_presenti["Cell_Y"].unique()],names=["Ora", "Cell_X", "Cell_Y"]).to_frame(index=False)

# Unisco la heatmap originale con la griglia completa, riempiendo i valori mancanti con 0
heatmap_complete = all_combinations.merge(heatmap, on=["Ora", "Cell_X", "Cell_Y"], how="left").fillna(0)

# Converto Taxi_Count in intero
heatmap_complete["Taxi_Count"] = heatmap_complete["Taxi_Count"].astype(int)

# Salvo il file aggiornato
output_file = r"C:\Users\paolo\Desktop\heatmap_taxi.txt"
heatmap_complete.to_csv(output_file, index=False, sep=",")

print("Mappa delle celle creata con successo:", output_file)
