import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import update_registered_converter
import pickle

# Atualizar o conversor registrado para o XGBClassifier
update_registered_converter(
    XGBClassifier, 
    'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, 
    convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

# Parâmetro para controlar o uso do GridSearchCV
use_grid_search = False

# Carregar os dados dos momentos de Hu do arquivo CSV
hu_moments_data = pd.read_csv('hu_moments_all_channels.csv')

# Separar os caminhos das imagens e os rótulos das células
X_paths = hu_moments_data['cell_path']
y_labels = hu_moments_data['cell_label']

# Extrair as características (momentos de Hu) como um DataFrame
X = hu_moments_data.drop(['cell_path', 'cell_label'], axis=1)

# Renomear as características para seguir o padrão esperado pelo conversor
X.columns = [f'f{idx}' for idx in range(X.shape[1])]

# Converter os rótulos das células para valores numéricos usando LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inicializar o classificador XGBoost
xgb_classifier = XGBClassifier()

# Treinar o modelo sem GridSearchCV
xgb_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = xgb_classifier.predict(X_test)

# Calcular a acurácia do classificador
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do classificador XGBoost:", accuracy)

# Mostrar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Convertendo o modelo para o formato ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(xgb_classifier, initial_types=initial_type, target_opset={'': 12, 'ai.onnx.ml': 2})

# Salvar o modelo ONNX em um arquivo
with open('modelo_xgboost_multiclass.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Modelo salvo em formato ONNX")
