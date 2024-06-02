import pandas as pd

# Carregar o arquivo CSV
file_path = 'hu_moments.csv'
data = pd.read_csv(file_path)

# Função para mapear os valores da coluna 'label' para 0 ou 1
def map_label(label):
    if label == 'Negative for intraepithelial lesion':
        return 0
    else:
        return 1

# Aplicar a função à coluna 'label'
data['label'] = data['label'].apply(map_label)

# Contar a quantidade de classes com 0 e 1
class_counts = data['label'].value_counts()

# Definir o novo caminho do arquivo
new_file_path = 'hu_moments_modified.csv'

# Salvar as alterações no novo arquivo CSV
data.to_csv(new_file_path, index=False)

print("Valores da coluna 'label' foram alterados e o novo arquivo foi salvo com sucesso.")
print("Contagem das classes:")
print("Classe 0:", class_counts[0])
print("Classe 1:", class_counts[1])
