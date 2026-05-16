import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# 1. Incarcam datele (folosim seria logaritmatica, nediferentiata)
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
serie_log = np.log(df['Pret_Imobiliare'])

# 2. Definim si antrenam modelul SARIMA
# order=(1,1,1) -> partea non-sezoniera (AR=1, Diferentiere=1, MA=1)
# seasonal_order=(1,0,1,12) -> partea sezoniera (frecventa 12 luni)
print("Se estimeaza modelul SARIMA... Acest proces poate dura cateva secunde.")
model_sarima = SARIMAX(serie_log, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 0, 1, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
rezultat_sarima = model_sarima.fit(disp=False)

# Afisam sumarul pentru proiect
print(rezultat_sarima.summary())

# 3. Generam previziuni (Forecast) pentru urmatoarele 12 luni
forecast = rezultat_sarima.get_forecast(steps=12)
forecast_index = pd.date_range(start=serie_log.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
forecast_mean = np.exp(forecast.predicted_mean) # readucem din logaritm in preturi normale
interval_incredere = np.exp(forecast.conf_int())

# 4. Graficul previziunii
plt.figure(figsize=(12, 6))
# Afisam doar ultimii 5 ani din seria reala pentru a vedea mai bine previziunea
istoric_recent = df['Pret_Imobiliare']['2019-01-01':] 
plt.plot(istoric_recent.index, istoric_recent, label='Date Istorice (Pret Real)', color='blue')
plt.plot(forecast_index, forecast_mean, label='Previziune SARIMA (12 luni)', color='red', linestyle='--')
plt.fill_between(forecast_index, interval_incredere.iloc[:, 0], interval_incredere.iloc[:, 1], color='pink', alpha=0.3, label='Interval de incredere (95%)')

plt.title('Previziunea Preturilor Imobiliare folosind modelul SARIMA')
plt.xlabel('Data')
plt.ylabel('Indice Pret Imobiliare')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('forecast_sarima.png')
plt.show()