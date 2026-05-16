import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# 1. Incarcarea datelor
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)

# 2. Crearea seriei logaritmice pentru pret (stabilizeaza varianta)
df['Pret_Log'] = np.log(df['Pret_Imobiliare'])

# 3. Trasarea graficelor la nivel (seriile originale)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
axes[0].plot(df.index, df['Pret_Imobiliare'], color='blue')
axes[0].set_title('Evolutia Pretului Imobiliarelor (Indice)')
axes[1].plot(df.index, df['Dobanda_Ipotecara'], color='red')
axes[1].set_title('Evolutia Dobandii Ipotecare (%)')
axes[2].plot(df.index, df['Santiere_Noi'], color='green')
axes[2].set_title('Santiere Noi (Evolutie lunara evidentiind sezonalitatea)')
plt.tight_layout()
plt.savefig('grafic_nivel.png') # Salveaza imaginea pentru a o pune in proiect
plt.show()

# 4. Descompunerea seriei principale (Log) pentru a vedea trendul si sezonalitatea
descompunere = seasonal_decompose(df['Pret_Log'], model='additive', period=12)
fig_decomp = descompunere.plot()
fig_decomp.set_size_inches(12, 8)
plt.savefig('descompunere_stl.png')
plt.show()

# 5. Functie pentru Testul Augmented Dickey-Fuller (ADF)
def test_stationaritate(serie, nume_serie):
    rezultat = adfuller(serie.dropna())
    print(f"--- Test ADF pentru {nume_serie} ---")
    print(f"Statistica ADF: {rezultat[0]:.4f}")
    print(f"P-value: {rezultat[1]:.4f}")
    if rezultat[1] < 0.05:
        print("Concluzie: Seria ESTE stationara (respingem H0).\n")
    else:
        print("Concluzie: Seria NU este stationara (are radacina unitara).\n")

# Aplicam testul pe serii
test_stationaritate(df['Pret_Log'], "Pret Imobiliare (Log)")
test_stationaritate(df['Dobanda_Ipotecara'], "Dobanda Ipotecara")
test_stationaritate(df['Santiere_Noi'], "Santiere Noi")