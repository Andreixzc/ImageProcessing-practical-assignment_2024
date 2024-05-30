import pandas as pd


file2 = 'labelPath.csv'  # Substitua pelo caminho do seu segundo arquivo CSV

df2 = pd.read_csv(file2)

print("\nColunas do segundo CSV:")
print(df2.columns)
