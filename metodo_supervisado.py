import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. PREPARACIÓN DE LOS DATOS
# Cargar datos
data = pd.read_csv('diabetes.csv') 

# Separar características (X) y (y)
X = data.drop('Outcome', axis=1) 
y = data['Outcome']

# Dividir en datos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. PROCESAMIENTO: LA CAJA CUADRADA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODELO 1: K-NEAREST NEIGHBORS (KNN)
print("====== EJECUTANDO MODELO KNN ======")
modelo_knn = KNeighborsClassifier(n_neighbors=5) 
modelo_knn.fit(X_train_scaled, y_train) 

predicciones_knn = modelo_knn.predict(X_test_scaled)
print(f"Exactitud de KNN: {accuracy_score(y_test, predicciones_knn) * 100:.2f}%")


# 4. MODELO 2: ÁRBOL DE DECISIÓN
print("\n====== EJECUTANDO ÁRBOL DE DECISIÓN ======")
# Uso entropía y max_depth=4 para evitar que el árbol crezca infinito (Pre-podado)
modelo_arbol = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
modelo_arbol.fit(X_train, y_train) 

predicciones_arbol = modelo_arbol.predict(X_test)
print(f"Exactitud del Árbol: {accuracy_score(y_test, predicciones_arbol) * 100:.2f}%")

# 5. COMPARACIÓN CLÍNICA (Métricas)
print("\n====== REPORTE CLÍNICO (Árbol de Decisión) ======")
print(classification_report(y_test, predicciones_arbol))

print("Matriz de Confusión del Árbol:")
print(confusion_matrix(y_test, predicciones_arbol))