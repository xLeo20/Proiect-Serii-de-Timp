import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore") # Ignoram avertismentele

# Incarcam datele din fisierul master
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
serie_log = np.log(df['Pret_Imobiliare']) # Folosim logaritmul pentru stabilitate

# Impartim datele: 80% pentru antrenare (train), 20% pentru testare
train_size = int(len(serie_log) * 0.8)
train, test = serie_log.iloc[:train_size], serie_log.iloc[train_size:]

# 1. Metoda Holt-Winters (Netezire exponentiala)
hw_model = ExponentialSmoothing(train, trend='add', seasonal=None, initialization_method="estimated").fit()
hw_pred = hw_model.forecast(len(test)) # Generam predictia pe lungimea setului de test

# 2. Modelul SARIMA
# Antrenam modelul cu parametrii alesi
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)
sarima_pred = sarima_result.get_forecast(steps=len(test)).predicted_mean

# Readucem valorile din logaritm in preturi reale aplicand functia exponentiala
train_real = np.exp(train)
test_real = np.exp(test)
hw_pred_real = np.exp(hw_pred)
sarima_pred_real = np.exp(sarima_pred)

# Comparam acuratetea folosind RMSE si MAE
print("--- Acuratete Holt-Winters ---")
print(f"RMSE: {np.sqrt(mean_squared_error(test_real, hw_pred_real)):.4f}")
print(f"MAE: {mean_absolute_error(test_real, hw_pred_real):.4f}")

print("\n--- Acuratete SARIMA ---")
print(f"RMSE: {np.sqrt(mean_squared_error(test_real, sarima_pred_real)):.4f}")
print(f"MAE: {mean_absolute_error(test_real, sarima_pred_real):.4f}")

# Generam graficul comparativ pentru a vizualiza performanta
plt.figure(figsize=(12, 6))
plt.plot(train_real.index, train_real, label='Train (Istoric)')
plt.plot(test_real.index, test_real, label='Test (Realitate)')
plt.plot(test_real.index, hw_pred_real, label='Predictie Holt-Winters', linestyle='--')
plt.plot(test_real.index, sarima_pred_real, label='Predictie SARIMA', linestyle='-.')
plt.title('Comparatie Train/Test: Holt-Winters vs SARIMA')
plt.legend()
plt.savefig('06_comparatie_modele.png')
plt.show()