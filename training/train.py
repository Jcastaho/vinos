import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
from mord import LogisticIT
import joblib
import os
import time

# ----------------------------------------
# Esperar a PostgreSQL
# ----------------------------------------

time.sleep(10)

# ----------------------------------------
# Conexión a la base de datos
# ----------------------------------------

DATABASE_URL = "postgresql://mluser:mlpass@postgres:5432/mldb"
engine = create_engine(DATABASE_URL)

# ----------------------------------------
# Leer dataset desde PostgreSQL
# ----------------------------------------

df = pd.read_sql("SELECT * FROM wine_dataset", engine)
print("Dataset cargado desde PostgreSQL")
print(df.head())

# ----------------------------------------
# Preprocesamiento (igual que en el notebook)
# ----------------------------------------

# 1. Eliminar 'density' por multicolinealidad
if 'density' in df.columns:
    df = df.drop(columns=['density'])

# 2. Transformaciones logarítmicas
df['residual sugar'] = np.log1p(df['residual sugar'])
df['chlorides']      = np.log1p(df['chlorides'])

# 3. Codificar tipo de vino: red=0, white=1
df['wine_type'] = df['wine_type'].map({'red': 0, 'white': 1})

# ----------------------------------------
# Separar features y target
# ----------------------------------------

X         = df.drop(columns=['quality'])
y_ordinal = df['quality']                        # 3-9 para regresión ordinal
y_binario = (y_ordinal >= 7).astype(int)         # 1=Bueno (7-9), 0=Regular/Malo (3-6)

print(f"\nFeatures: {X.columns.tolist()}")
print(f"\nDistribución quality original:\n{y_ordinal.value_counts().sort_index()}")
print(f"\nDistribución binaria:\n{y_binario.value_counts()}")

# ----------------------------------------
# Split estratificado (mismo para los dos
# modelos, mismos índices de test)
# ----------------------------------------

X_train, X_test, y_train_ord, y_test_ord = train_test_split(
    X, y_ordinal,
    test_size=0.2,
    random_state=42,
    stratify=y_ordinal
)
y_train_bin = (y_train_ord >= 7).astype(int)
y_test_bin  = (y_test_ord  >= 7).astype(int)

# ----------------------------------------
# MODELO 1: Clasificación Binaria
# RandomForest — Bueno (>=7) vs Regular/Malo (<7)
# ----------------------------------------

scaler_bin = StandardScaler()
X_train_bin = scaler_bin.fit_transform(X_train)
X_test_bin  = scaler_bin.transform(X_test)

modelo_clasificacion = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)
modelo_clasificacion.fit(X_train_bin, y_train_bin)

y_pred_bin = modelo_clasificacion.predict(X_test_bin)
acc_bin = accuracy_score(y_test_bin, y_pred_bin)
f1_bin  = f1_score(y_test_bin, y_pred_bin, average='weighted')

print(f"\n=== MODELO 1: Clasificación Binaria (Random Forest) ===")
print(f"Accuracy      : {acc_bin:.4f}")
print(f"F1 (weighted) : {f1_bin:.4f}")
print(classification_report(y_test_bin, y_pred_bin,
      target_names=['Regular/Malo (3-6)', 'Bueno (7-9)'], zero_division=0))

# ----------------------------------------
# MODELO 2: Regresión Ordinal
# LogisticIT — predice nota exacta 3-9
# ----------------------------------------

scaler_ord = StandardScaler()
X_train_ord = scaler_ord.fit_transform(X_train)
X_test_ord  = scaler_ord.transform(X_test)

modelo_ordinal = LogisticIT()
modelo_ordinal.fit(X_train_ord, y_train_ord)

y_pred_ord = modelo_ordinal.predict(X_test_ord)
acc_ord = accuracy_score(y_test_ord, y_pred_ord)
f1_ord  = f1_score(y_test_ord, y_pred_ord, average='weighted')
mae_ord = mean_absolute_error(y_test_ord, y_pred_ord)

print(f"\n=== MODELO 2: Regresión Ordinal (LogisticIT) ===")
print(f"Accuracy      : {acc_ord:.4f}")
print(f"F1 (weighted) : {f1_ord:.4f}")
print(f"MAE           : {mae_ord:.4f}")

# ----------------------------------------
# Guardar modelos y escaladores
# ----------------------------------------

os.makedirs("/models", exist_ok=True)

joblib.dump(modelo_clasificacion, "/models/modelo_clasificacion.pkl")
joblib.dump(modelo_ordinal,       "/models/modelo_ordinal.pkl")
joblib.dump(scaler_bin,           "/models/scaler_bin.pkl")
joblib.dump(scaler_ord,           "/models/scaler_ord.pkl")

print("\nModelos y escaladores guardados en /models/")

# ----------------------------------------
# Guardar metadata en PostgreSQL
# ----------------------------------------

metadata = pd.DataFrame([
    {
        "model_name"  : "RandomForest_Binario",
        "accuracy"    : round(acc_bin, 4),
        "f1_weighted" : round(f1_bin,  4),
        "mae"         : None
    },
    {
        "model_name"  : "LogisticIT_Ordinal",
        "accuracy"    : round(acc_ord, 4),
        "f1_weighted" : round(f1_ord,  4),
        "mae"         : round(mae_ord, 4)
    }
])

metadata.to_sql("models_metadata", engine, if_exists="append", index=False)
print("Metadata almacenada en PostgreSQL")
