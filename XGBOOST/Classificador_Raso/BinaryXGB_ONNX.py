import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import update_registered_converter


# Atualizar o conversor registrado para o XGBClassifier
update_registered_converter(
    XGBClassifier, 
    'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, 
    convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)


# Load Hu moments from CSV
def load_hu_moments_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return features, labels

# Load features and labels from CSV
csv_file = 'hu_moments_modified.csv'
features, labels = load_hu_moments_from_csv(csv_file)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy='auto')
features_resampled, labels_resampled = smote.fit_resample(features, labels)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42, stratify=labels_resampled)

# Print the number of positive and negative classes after oversampling
print("Number of positive classes after oversampling:", Counter(y_train)[1])
print("Number of negative classes after oversampling:", Counter(y_train)[0])

# Instantiate the XGBoost classifier
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    verbosity=1
)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


  # Convertendo o modelo para o formato ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 12, 'ai.onnx.ml': 2})

# Salvar o modelo ONNX em um arquivo
with open('modelo_xgboost_binary.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())