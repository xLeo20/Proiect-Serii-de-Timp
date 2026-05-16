import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

# 1. Incarcam datele
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
df['Pret_Log'] = np.log(df['Pret_Imobiliare'])

# 2. Cream baza pentru modelul VAR 
# VAR necesita date stationare, deci vom folosi diferenta de ordin 1 (cresterea/scaderea lunara)
data_var = pd.DataFrame({
    'Pret': df['Pret_Log'].diff(),
    'Dobanda': df['Dobanda_Ipotecara'].diff(),
    'Santiere': df['Santiere_Noi'].diff()
}).dropna()

# 3. Testul de Cauzalitate Granger (Ipoteza: Dobanda cauzeaza Pretul?)
print("=== Test Cauzalitate Granger: Dobanda influenteaza Pretul? ===")
# Functia cere formatul [Y (Ce este influentat), X (Ce influenteaza)]
granger_test = grangercausalitytests(data_var[['Pret', 'Dobanda']], maxlag=3, verbose=False)
for lag in range(1, 4):
    p_val = granger_test[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_val:.4f}")

# 4. Estimam modelul VAR
# Lasam algoritmul sa aleaga automat cel mai bun numar de lag-uri bazat pe scorul AIC
model = VAR(data_var)
rezultat_var = model.fit(maxlags=5, ic='aic')
print("\nNumarul optim de lag-uri ales de model:", rezultat_var.k_ar)

# 5. Functia de Impuls-Raspuns (IRF) - GRAFICUL SUPREM
# Simulam un "soc" pozitiv de 1 deviatie standard in Dobanda si vedem ce pateste Pretul pe 12 luni
irf = rezultat_var.irf(12)
fig = irf.plot(impulse='Dobanda', response='Pret', figsize=(10, 5))
plt.title("Functia Impuls-Raspuns: Efectul cresterii Dobanzii asupra Pretului Imobiliar")
plt.ylabel("Raspunsul Pretului")
plt.xlabel("Luni de la momentul socului")
plt.tight_layout()
plt.savefig('irf_multivariat.png')
plt.show()