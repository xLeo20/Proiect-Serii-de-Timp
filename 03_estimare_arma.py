import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore") # Ascundem avertismentele de convergenta

# 1. Incarcam datele si pregatim seria stationara
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
df['Pret_Log'] = np.log(df['Pret_Imobiliare'])
serie_stationara = df['Pret_Log'].diff().dropna()

# 2. Functie pentru a testa mai multe modele ARMA
def testeaza_arma(p, q):
    # Folosim ordinul de integrare d=0 deoarece aplicam pe seria deja diferentiata
    model = ARIMA(serie_stationara, order=(p, 0, q))
    rezultat = model.fit()
    print(f"--- Model ARMA({p},{q}) ---")
    print(f"AIC: {rezultat.aic:.2f} | BIC: {rezultat.bic:.2f}")
    return rezultat

# 3. Testam configuratiile sugerate de corelograme
model_11 = testeaza_arma(1, 1)
model_21 = testeaza_arma(2, 1)
model_12 = testeaza_arma(1, 2)

# 4. Afisam sumarul detaliat pentru cel care de obicei iese cel mai bine (ex. ARMA 1,1)
# Vom analiza P>|z| pentru a vedea daca coeficientii sunt valizi
print("\n=== SUMAR DETALIAT PENTRU ARMA(1,1) ===")
print(model_11.summary())