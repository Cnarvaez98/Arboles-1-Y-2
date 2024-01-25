import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Preparación del dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento de un Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluación del rendimiento del modelo
y_pred = rf_classifier.predict(X_test)

# Calcular el accuracy del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy}')

# Generar una matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Métricas para analizar el desbalance
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

print("Precision por clase:")
print(precision)
print("Recall por clase:")
print(recall)
print("F1-Score por clase:")
print(f1)
