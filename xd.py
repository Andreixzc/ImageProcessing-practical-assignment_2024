import pandas as pd

csv = 'novo_arquivo.csv'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(csv)

# Exibir as colunas do DataFrame
print("Colunas:")
print(df.columns)

# Exibir a primeira linha do DataFrame
print("\nPrimeira linha:")
print(df.head(1))
