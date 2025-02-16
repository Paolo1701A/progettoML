import pandas as pd
import matplotlib.pyplot as plt

# Carico i dati della mappa generata prima
file_path = r"C:\Users\paolo\Desktop\heatmap_taxi.txt"
df = pd.read_csv(file_path, sep=",")

# Analizzo la distribuzione del numero di taxi per cella
print(df["Taxi_Count"].describe())

# Istogramma del numero di taxi per cella
plt.figure(figsize=(10, 5))
plt.hist(df["Taxi_Count"], bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Numero di taxi per cella")
plt.ylabel("Numero di celle")
plt.title("Distribuzione del numero di taxi per cella")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()