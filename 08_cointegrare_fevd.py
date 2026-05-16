import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings("ignore") # Ascundem eventualele mesaje de atentionare

# Incarcam datele 
df = pd.read_csv('master_dataset.csv', index_col='observation_date', parse_dates=True)
df['Pret_Log'] = np.log(df['Pret_Imobiliare'])

# Selectam variabilele pentru testul de cointegrare si stergem valorile lipsa
data_coint = df[['Pret_Log', 'Dobanda_Ipotecara', 'Santiere_Noi']].dropna()

# Efectuam testul Johansen pentru a gasi relatiile de echilibru pe termen lung
print("=== Testul de Cointegrare Johansen ===")
johansen_test = coint_johansen(data_coint, det_order=0, k_ar_diff=1)
print("Statistici Trace:\n", johansen_test.lr1)
print("\nValori critice (90%, 95%, 99%):\n", johansen_test.cvt)

# Pregatim datele stationare pentru modelul VAR (folosind diferenta)
data_var = pd.DataFrame({
    'Pret': df['Pret_Log'].diff(),
    'Dobanda': df['Dobanda_Ipotecara'].diff(),
    'Santiere': df['Santiere_Noi'].diff()
}).dropna()

# Antrenam modelul VAR pentru analiza multivariata
model = VAR(data_var)
rezultat_var = model.fit(maxlags=5, ic='aic')

# Analiza de descompunere a variantei (FEVD) pe un orizont de 12 luni
fevd = rezultat_var.fevd(12)
print("\n=== Descompunerea Variantei (FEVD) ===")
print(fevd.summary())

# Generam graficul pentru vizualizarea descompunerii variantei
fevd.plot(figsize=(10, 6))
plt.suptitle("Descompunerea Variantei pe 12 luni")
plt.tight_layout()
plt.savefig('08_descompunere_varianta.png')
plt.show()