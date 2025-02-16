import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

start_time = time.perf_counter()

# Carica i dati della heatmap generata prima
file_path = r"C:\Users\paolo\Desktop\heatmap_taxi.txt"
df = pd.read_csv(file_path, sep=",")

#splitto il df in trainig set e testing set
training_set, testing_set = train_test_split(df, train_size=0.7 ,test_size=0.3, random_state=42)

# Funzione per trasformare i dati con trasformazione logaritmica
def trasf_log(df):
    df = df.sort_values(by=['Cell_X', 'Cell_Y', 'Ora'])
    # Applico la trasformazione logaritmica a Taxi_Count
    df['Taxi_Count_Log'] = np.log1p(df['Taxi_Count'])  # log(1 + x) per evitare 0
    # Creo la variabile target con Taxi_Count_Log shiftato (a seconda delle ore che voglio pred)
    df['Target_Log'] = df.groupby(['Cell_X', 'Cell_Y'])['Taxi_Count_Log'].shift(-3)
    # Rimuoviamo righe con valori NaN
    df = df.dropna()
    return df

# Preparo i dati
training_set = trasf_log(training_set)
testing_set = trasf_log(testing_set)

# Separazione delle variabili indipendenti (X) e della variabile target (y)
X_train = training_set[['Ora', 'Cell_X', 'Cell_Y', 'Taxi_Count_Log']]
y_train = training_set['Target_Log']

X_test = testing_set[['Ora', 'Cell_X', 'Cell_Y', 'Taxi_Count_Log']]
y_test = testing_set['Target_Log']

# Creazione e addestramento del modello
model = LinearRegression()
model.fit(X_train, y_train)

# Previsioni
y_pred_log = model.predict(X_test)

# Inverto la trasformazione logaritmica per confrontare i dati originali
y_pred = np.expm1(y_pred_log)  # exp(x) - 1
testing_set['Predicted_Taxi_Count'] = y_pred

# Valutazione del modello
mse = mean_squared_error(np.expm1(y_test), y_pred)  # Confronto con i valori reali
print(f"Mean Squared Error (MSE): {mse}")
print(f"Varianza dei dati reali: {np.var(np.expm1(y_test))}")
r2 = r2_score(np.expm1(y_test), y_pred)
print(f"R² sul Testing set: {r2}")
y_train_pred = model.predict(X_train)
r2_train = r2_score(np.expm1(y_train), np.expm1(y_train_pred))
print(f"R² sul Training Set: {r2_train}")

# Identificazione delle celle con pochi taxi (meno di 3 taxi previsti)
low_taxi_cells = testing_set[testing_set['Predicted_Taxi_Count'] < 3].copy()
# Identificazione delle celle con molti taxi (più di 300 taxi previsti)
high_taxi_cells = testing_set[testing_set['Predicted_Taxi_Count'] >= 300].copy()
# Stampo le celle con pochi taxi
print("\nCelle con pochi taxi previste nelle prossime 3 ore:")
print(low_taxi_cells[['Cell_X', 'Cell_Y', 'Predicted_Taxi_Count']])
# Stampo le celle con molti taxi
print("\nCelle con molti taxi previste nelle prossime 3 ore:")
print(high_taxi_cells[['Cell_X', 'Cell_Y', 'Predicted_Taxi_Count']])

#Grafico a dispersione per confrontare valori reali e predetti, la linea rossa son i valori reali
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_test), y_pred, alpha=0.5)
plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], [min(np.expm1(y_test)), max(np.expm1(y_test))], color='red', linestyle='--')  # Linea di riferimento
plt.xlabel('Valori Reali (Taxi Count)')
plt.ylabel('Valori Predetti (Taxi Count)')
plt.title('Confronto tra Valori Reali e Predetti')
plt.grid(True)
plt.show()

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time:} secondi")