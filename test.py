import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Carregar o arquivo CSV
df = pd.read_csv('novo_arquivo.csv')

# Loop pelos caminhos das células e mostrar as imagens
for index, row in df.iterrows():
    cell_path = row['cell_path']
    cell_label = row['cell_label']
    
    # Carregar a imagem usando o caminho da célula
    img = Image.open(cell_path)
    
    # Mostrar a imagem na tela
    plt.imshow(img)
    plt.title(f"Label: {cell_label}")
    plt.axis('off')  # Desativar os eixos
    plt.show()
