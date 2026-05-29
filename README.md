# Wine ML Pipeline — Docker + FastAPI + Streamlit

## Descripción

Pipeline completo de Machine Learning para predicción de calidad de vinos,
implementado con microservicios Docker.

---

## Modelos

| #   | Modelo                   | Tipo                  | Output                           |
| --- | ------------------------ | --------------------- | -------------------------------- |
| 1   | Random Forest Classifier | Clasificación binaria | Bueno (>=7) vs Regular/Malo (<7) |
| 2   | LogisticIT (mord)        | Regresión ordinal     | Nota exacta del 3 al 9           |

---

## Estructura del proyecto

```
wine-ml-pipeline/
├── docker-compose.yml
├── README.md
├── datasets/
│   └── wine_dataset.csv
├── models/
├── init-db/
│   ├── Dockerfile
│   ├── load_data.py
│   └── requirements.txt
├── training/
│   ├── Dockerfile
│   ├── train.py
│   └── requirements.txt
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── frontend/
    ├── Dockerfile
    ├── app.py
    └── requirements.txt
```

---
## Flujo del sistema

```
CSV Dataset
    ↓
Init-DB (carga el CSV en PostgreSQL)
    ↓
PostgreSQL (almacena dataset, metadata y predicciones)
    ↓
Training Service (entrena los dos modelos)
    ↓
modelo_clasificacion.pkl + modelo_ordinal.pkl
    ↓
FastAPI (sirve predicciones via REST)
    ↓
Streamlit (interfaz de usuario)
    ↓
Usuario
```


---
## Ejecución paso a paso

### Paso 1 — Construir los contenedores

```bash
docker compose build
```

---

### Paso 2 — Levantar PostgreSQL

```bash
docker compose up -d postgres
```

---

### Paso 3 — Cargar el dataset en PostgreSQL

```bash
docker compose run init-db
```

---

### Paso 4 — Entrenar los modelos

```bash
docker compose run training
```

Entrena Random Forest (binario) + LogisticIT (ordinal) y guarda:

- `/models/modelo_clasificacion.pkl`
- `/models/modelo_ordinal.pkl`
- `/models/scaler_bin.pkl`
- `/models/scaler_ord.pkl`

---

### Paso 5 — Levantar API y frontend

```bash
docker compose up api frontend
```

---

## Servicios

| Servicio     | URL                        |
| ------------ | -------------------------- |
| FastAPI      | http://localhost:8000      |
| Swagger Docs | http://localhost:8000/docs |
| Streamlit    | http://localhost:8501      |

---


## Preprocesamiento aplicado en train.py

1. Eliminación de `density` por multicolinealidad
2. `log1p` en `residual sugar` y `chlorides`
3. `wine_type` → red=0, white=1
4. `StandardScaler` independiente por modelo
5. `stratify=y` en el split para preservar proporciones
