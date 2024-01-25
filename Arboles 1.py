import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carga de los datos
digits = load_digits()
X, y = digits.data, digits.target

# Exploración del dataset
rng = np.random.default_rng(42)
random_indices = rng.choice(len(X), size=10, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices, 1):
    plt.subplot(2, 5, i)
    plt.imshow(X[idx].reshape(8, 8), cmap='gray')
    plt.title(f'Digito: {y[idx]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Entrenamiento de un árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Aplanar las imágenes
X_flat = X.reshape(len(X), -1)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Evaluación del rendimiento del modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Exactitud del modelo: {accuracy}')
