import os
import csv

# Definir o caminho base
basePath = '28-05-2024/'

# Definir os rótulos
labels = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Negative for intraepithelial lesion', 'SCC']

# Nome do arquivo CSV de saída
output_csv = 'image_paths_labels.csv'

# Função para coletar as informações das imagens e escrever no CSV
def generate_csv(basePath, labels, output_csv):
    # Abrir o arquivo CSV para escrita
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escrever o cabeçalho
        writer.writerow(['Path', 'Label'])
        
        # Percorrer cada rótulo
        for label in labels:
            # Construir o caminho completo da pasta do rótulo
            label_path = os.path.join(basePath, label)
            print(label_path)
            
            # Verificar se o diretório existe
            if os.path.isdir(label_path):
                # Percorrer cada arquivo na pasta do rótulo
                for filename in os.listdir(label_path):
                    # Verificar se o arquivo é uma imagem (você pode ajustar a verificação conforme necessário)
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                        # Construir o caminho completo da imagem
                        image_path = os.path.join(label_path, filename)
                        # Escrever a linha no CSV
                        writer.writerow([image_path, label])

# Executar a função para gerar o CSV
generate_csv(basePath, labels, output_csv)

print(f"CSV gerado com sucesso: {output_csv}")
