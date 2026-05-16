import pandas as pd

# 1. Citim cele 3 fișiere și setăm coloana de dată ca index
df_price = pd.read_csv('CSUSHPINSA.csv', parse_dates=['observation_date'], index_col='observation_date')
df_mortgage = pd.read_csv('MORTGAGE30US.csv', parse_dates=['observation_date'], index_col='observation_date')
df_starts = pd.read_csv('HOUSTNSA.csv', parse_dates=['observation_date'], index_col='observation_date')

# 2. Transformăm dobânda din date săptămânale în medie lunară (MS = Month Start)
df_mortgage_monthly = df_mortgage.resample('MS').mean()

# Asigurăm și pentru celelalte serii alinierea la început de lună
df_price = df_price.resample('MS').first()
df_starts = df_starts.resample('MS').first()

# 3. Unim totul într-un singur DataFrame
master_df = pd.concat([df_price, df_mortgage_monthly, df_starts], axis=1, join='inner')

# 4. Redenumim coloanele pentru a fi ușor de folosit în modele
master_df.columns = ['Pret_Imobiliare', 'Dobanda_Ipotecara', 'Santiere_Noi']

# 5. Salvăm setul de date final
master_df.to_csv('master_dataset.csv')

print("Fișierul master_dataset.csv a fost creat cu succes!")
print(master_df.head())