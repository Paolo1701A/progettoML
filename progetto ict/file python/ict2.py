import matplotlib.pyplot as plt
import pandas as pd

# do nomi alle colonne
nomi_col = ['taxi_id', 'date_time', 'long', 'lat']

# Carico il file .txt input
df = pd.read_csv(r'C:\Users\paolo\Desktop\progetto ict\input.txt',
                 header=None,  # Indica che il file non ha intestazioni
                 names=nomi_col,  # Assegna manualmente i nomi delle colonne
                 sep=',')  # Specifica il separatore

# creo Statistiche utili per presentazione
print(df.describe())

# Raggruppo per ora
df['hour'] = pd.to_datetime(df['date_time']).dt.hour
distribuz_ora = df.groupby('hour').size()

# Grafico a barre, ho messo un colore al bordo per migliorare la visibilità
distribuz_ora.plot(kind='bar', color='blue', edgecolor='black', grid=True)
plt.title('Distribuzione dei taxi per ora del giorno')
plt.xlabel('Ora del giorno')
plt.ylabel('Numero di taxi')
plt.show()

#grafico con punti, ho impostato alpha 0.5 per evitare troppa trasparenza e s=5 perchè così la dimensione dei punti
#è migliore e si vede bene
plt.scatter(df['long'], df['lat'], alpha=0.5, s=5, c='blue')
plt.xlim(df['long'].min(), df['long'].max())
plt.ylim(df['lat'].min(), df['lat'].max())
plt.title('Distribuzione spaziale dei taxi')
plt.xlabel('Longitudine')
plt.ylabel('Latitudine')
plt.grid(True)
plt.show()
#il grafico si carica lentamente ma non saprei come migliorare perchè ci sono molti dati...