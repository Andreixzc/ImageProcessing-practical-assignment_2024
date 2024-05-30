import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
import onnx

# Carregar os dados dos momentos de Hu do arquivo CSV
hu_moments_data = pd.read_csv('hu_moments.csv')

# Separar os caminhos das imagens e os rótulos das células
X_paths = hu_moments_data['cell_path']
y_labels = hu_moments_data['cell_label']

# Extrair as características (momentos de Hu) como um DataFrame
X = hu_moments_data.drop(['cell_path', 'cell_label'], axis=1)

# Converter os rótulos das células para valores numéricos usando LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 ,stratify=y)

# Inicializar o classificador XGBoost
xgb_classifier = XGBClassifier()

# Definir a grade de hiperparâmetros a serem testados
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Inicializar o GridSearchCV
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Executar a busca em grade nos dados de treinamento
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print("Melhores hiperparâmetros encontrados:", best_params)

# Treinar o modelo com os melhores hiperparâmetros
best_xgb_classifier = XGBClassifier(**best_params)
best_xgb_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = best_xgb_classifier.predict(X_test)

# Calcular a acurácia do classificador
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do classificador XGBoost com melhores hiperparâmetros:", accuracy)

# Mostrar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão com melhores hiperparâmetros:")
print(conf_matrix)

# Salvar o modelo treinado em formato pkl
with open('modelo_xgboost_multiclass.pkl', 'wb') as f:
    pickle.dump(best_xgb_classifier, f)

# Convertendo o modelo para o formato ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(best_xgb_classifier, initial_types=initial_type)

# Salvar o modelo ONNX em um arquivo
with open('modelo_xgboost_multiclass.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Modelo salvo em formato ONNX")
