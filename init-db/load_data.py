import pandas as pd
from sqlalchemy import create_engine
import time

# ----------------------------------------
# Esperar a que PostgreSQL esté listo
# ----------------------------------------

time.sleep(5)

# ----------------------------------------
# Conexión a PostgreSQL
# ----------------------------------------

DATABASE_URL = "postgresql://mluser:mlpass@postgres:5432/mldb"
engine = create_engine(DATABASE_URL)

# ----------------------------------------
# Leer CSV
# ----------------------------------------

df = pd.read_csv("/datasets/wine_dataset.csv")

print("Dataset de vinos cargado desde CSV")
print(df.head())
print(f"Shape: {df.shape}")

# ----------------------------------------
# Cargar en PostgreSQL
# ----------------------------------------

df.to_sql(
    "wine_dataset",
    engine,
    if_exists="replace",
    index=False
)

print("Tabla 'wine_dataset' creada en PostgreSQL")
print(f"Registros insertados: {len(df)}")
