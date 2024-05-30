import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('apended.csv')

# Trocar a posição das colunas
coluna_ultima = df.columns[-1]
coluna_penultima = df.columns[-2]

df[coluna_ultima], df[coluna_penultima] = df[coluna_penultima].copy(), df[coluna_ultima].copy()

# Salvar o DataFrame modificado em um novo arquivo CSV
df.to_csv('novo_arquivo.csv', index=False)
