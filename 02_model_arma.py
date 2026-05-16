import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Incarcam datele
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
df['Pret_Log'] = np.log(df['Pret_Imobiliare'])

# 2. Diferentierea seriei (ordinul 1) pentru a elimina trendul si a o face stationara
df['Pret_Log_Diff'] = df['Pret_Log'].diff()

# Stergem prima valoare care devine NaN (lipsa) in urma diferentierii
serie_stationara = df['Pret_Log_Diff'].dropna()

# 3. Testam din nou cu ADF pentru a demonstra academic ca seria a devenit stationara
print("--- Test ADF pe seria DIFERENTIATA (Pret_Log_Diff) ---")
rezultat_diff = adfuller(serie_stationara)
print(f"Statistica ADF: {rezultat_diff[0]:.4f}")
print(f"P-value: {rezultat_diff[1]:.4f}")

if rezultat_diff[1] < 0.05:
    print("Succes! Seria este acum stationara.")
else:
    print("Seria inca nu este stationara. Ar putea necesita diferentiere de ordin 2.")

# 4. Generam corelogramele ACF si PACF pentru a citi parametrii p si q
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(serie_stationara, lags=36, ax=axes[0], title="Autocorelare (ACF) -> Ajuta la alegerea 'q' (MA)")
plot_pacf(serie_stationara, lags=36, ax=axes[1], title="Autocorelare Partiala (PACF) -> Ajuta la alegerea 'p' (AR)")
plt.tight_layout()
plt.savefig('corelograme_arma.png')
plt.show()