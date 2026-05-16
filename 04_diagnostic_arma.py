import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import warnings
warnings.filterwarnings("ignore")

# 1. Incarcam datele si refacem seria
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
serie_stationara = np.log(df['Pret_Imobiliare']).diff().dropna()

# 2. Rulam modelul castigator ARMA(1,1)
model_arma = ARIMA(serie_stationara, order=(1, 0, 1)).fit()
reziduuri = model_arma.resid

# 3. Graficul reziduurilor
plt.figure(figsize=(10, 4))
plt.plot(reziduuri, color='purple')
plt.title("Evolutia reziduurilor pentru modelul ARMA(1,1)")
plt.axhline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('reziduuri_arma.png')
plt.show()

# 4. Testul Ljung-Box pentru autocorelare (Ipoteza nula: erorile nu sunt corelate)
lb_test = acorr_ljungbox(reziduuri, lags=[10], return_df=True)
print("=== Testul Ljung-Box (Autocorelarea erorilor) ===")
print(lb_test)

# 5. Testul ARCH pentru heteroschedasticitate (Ipoteza nula: varianta este constanta)
# Folosim dropna() pe reziduuri pentru siguranta
arch_test = het_arch(reziduuri.dropna())
print("\n=== Testul ARCH (Heteroschedasticitate) ===")
print(f"Statistica Lagrange Multiplier (LM): {arch_test[0]:.4f}")
print(f"P-value: {arch_test[1]:.4f}")