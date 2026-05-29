from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time

# ----------------------------------------
# Esperar a que los modelos existan
# ----------------------------------------

time.sleep(15)

# ----------------------------------------
# Cargar modelos y escaladores
# ----------------------------------------

modelo_clasificacion = joblib.load("/models/modelo_clasificacion.pkl")
modelo_ordinal       = joblib.load("/models/modelo_ordinal.pkl")
scaler_bin           = joblib.load("/models/scaler_bin.pkl")
scaler_ord           = joblib.load("/models/scaler_ord.pkl")

print("Modelos cargados correctamente")

# ----------------------------------------
# Conexión a PostgreSQL
# ----------------------------------------

DATABASE_URL = "postgresql://mluser:mlpass@postgres:5432/mldb"
engine = create_engine(DATABASE_URL)

# ----------------------------------------
# App FastAPI
# ----------------------------------------

app = FastAPI(title="Wine Quality API", version="1.0")

# ----------------------------------------
# Esquema de entrada
# ----------------------------------------

class WineRequest(BaseModel):
    fixed_acidity:        float
    volatile_acidity:     float
    citric_acid:          float
    residual_sugar:       float
    chlorides:            float
    free_sulfur_dioxide:  float
    total_sulfur_dioxide: float
    density:              float   # se recibe pero no entra al modelo
    pH:                   float
    sulphates:            float
    alcohol:              float
    wine_type:            str     # "red" o "white"

# ----------------------------------------
# Endpoint raíz
# ----------------------------------------

@app.get("/")
def root():
    return {"message": "API de Calidad del Vino en marcha"}

# ----------------------------------------
# Endpoint de predicción
# ----------------------------------------

@app.post("/predict")
def predict(data: WineRequest):

    # Mismas transformaciones que en train.py
    tipo_vino_num = 0 if data.wine_type.lower() == "red" else 1

    features_dict = {
        "fixed acidity":        data.fixed_acidity,
        "volatile acidity":     data.volatile_acidity,
        "citric acid":          data.citric_acid,
        "residual sugar":       np.log1p(data.residual_sugar),
        "chlorides":            np.log1p(data.chlorides),
        "free sulfur dioxide":  data.free_sulfur_dioxide,
        "total sulfur dioxide": data.total_sulfur_dioxide,
        "pH":                   data.pH,
        "sulphates":            data.sulphates,
        "alcohol":              data.alcohol,
        "wine_type":            tipo_vino_num
        # density NO entra: fue eliminada en preprocesamiento
    }

    features_df = pd.DataFrame([features_dict])

    # Escalar con el scaler de cada modelo
    features_bin = scaler_bin.transform(features_df)
    features_ord = scaler_ord.transform(features_df)

    # Predicciones
    pred_bin = int(modelo_clasificacion.predict(features_bin)[0])
    pred_ord = int(modelo_ordinal.predict(features_ord)[0])

    # Probabilidades del modelo binario
    proba     = modelo_clasificacion.predict_proba(features_bin)[0]
    prob_malo = round(float(proba[0]), 4)
    prob_bueno= round(float(proba[1]), 4)

    # Texto descriptivo
    categoria = "Bueno (7-9)"         if pred_bin == 1 else "Regular / Malo (3-6)"
    emoji     = "excelente"           if pred_ord >= 8 else \
                "muy bueno"           if pred_ord == 7 else \
                "aceptable"           if pred_ord == 6 else \
                "regular"             if pred_ord == 5 else "bajo"

    result = {
        # Modelo 1 — Clasificación binaria
        "prediccion_binaria" : pred_bin,
        "categoria_binaria"  : categoria,
        "prob_bueno"         : prob_bueno,
        "prob_malo"          : prob_malo,
        # Modelo 2 — Regresión ordinal
        "calidad_ordinal"    : pred_ord,
        "descripcion_ordinal": emoji
    }

    # Guardar en PostgreSQL
    log_df = pd.DataFrame([{
        "fixed_acidity":        data.fixed_acidity,
        "volatile_acidity":     data.volatile_acidity,
        "citric_acid":          data.citric_acid,
        "residual_sugar":       data.residual_sugar,
        "chlorides":            data.chlorides,
        "free_sulfur_dioxide":  data.free_sulfur_dioxide,
        "total_sulfur_dioxide": data.total_sulfur_dioxide,
        "density":              data.density,
        "pH":                   data.pH,
        "sulphates":            data.sulphates,
        "alcohol":              data.alcohol,
        "wine_type":            data.wine_type,
        "pred_binario":         pred_bin,
        "pred_ordinal":         pred_ord
    }])
    log_df.to_sql("wine_predictions", engine, if_exists="append", index=False)

    return result
