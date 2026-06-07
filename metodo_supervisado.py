import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. PREPARACIÓN DE LOS DATOS
# ==========================================
# Cargar el set de datos (asegúrate de que el archivo esté en la misma carpeta)
# Nota: Si tu archivo se llama diferente, cambia 'diabetes.csv' por tu archivo de corazón
data = pd.read_csv('diabetes.csv') 

# Separar características (X) y la respuesta/target (y)
X = data.drop('Outcome', axis=1) # Cambia 'Outcome' por la columna objetivo de tu dataset
y = data['Outcome']

# Dividir en datos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. PROCESAMIENTO: LA CAJA CUADRADA (Escalar datos para KNN)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODELO 1: K-NEAREST NEIGHBORS (KNN)
# ==========================================
print("====== EJECUTANDO MODELO KNN ======")
modelo_knn = KNeighborsClassifier(n_neighbors=5) # K=5 vecinos
modelo_knn.fit(X_train_scaled, y_train) # Entrenamos con datos ESCALADOS

predicciones_knn = modelo_knn.predict(X_test_scaled)
print(f"Exactitud de KNN: {accuracy_score(y_test, predicciones_knn) * 100:.2f}%")

# ==========================================
# 4. MODELO 2: ÁRBOL DE DECISIÓN
# ==========================================
print("\n====== EJECUTANDO ÁRBOL DE DECISIÓN ======")
# Usamos entropía y max_depth=4 para evitar que el árbol crezca infinito (Pre-podado)
modelo_arbol = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
modelo_arbol.fit(X_train, y_train) # Los árboles no sufren por las escalas, usamos X_train directo

predicciones_arbol = modelo_arbol.predict(X_test)
print(f"Exactitud del Árbol: {accuracy_score(y_test, predicciones_arbol) * 100:.2f}%")

# ==========================================
# 5. COMPARACIÓN CLÍNICA (Métricas)
# ==========================================
print("\n====== REPORTE CLÍNICO (Árbol de Decisión) ======")
print(classification_report(y_test, predicciones_arbol))

print("Matriz de Confusión del Árbol:")
print(confusion_matrix(y_test, predicciones_arbol))