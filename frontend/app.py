import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Predictor de Calidad del Vino",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Predictor de Calidad del Vino")
st.write("Introduce los parámetros químicos del vino para obtener las predicciones")

# ----------------------------------------
# Inicializar historial en session_state
# ----------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []

# ----------------------------------------
# Inputs en 2 columnas
# ----------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Propiedades generales")
    wine_type = st.selectbox(
        "Tipo de vino", ["red", "white"],
        help="Vino tinto (red) o blanco (white)"
    )
    fixed_acidity = st.number_input(
        "Acidez fija (Fixed Acidity)", min_value=0.0, max_value=25.0, value=7.4, step=0.1
    )
    volatile_acidity = st.number_input(
        "Acidez volátil (Volatile Acidity)", min_value=0.0, max_value=3.0, value=0.52, step=0.01
    )
    citric_acid = st.number_input(
        "Ácido cítrico (Citric Acid)", min_value=0.0, max_value=2.0, value=0.27, step=0.01
    )
    residual_sugar = st.number_input(
        "Azúcar residual (Residual Sugar)", min_value=0.0, max_value=100.0, value=2.2, step=0.1
    )
    chlorides = st.number_input(
        "Cloruros (Chlorides)", min_value=0.0, max_value=1.0,
        value=0.076, step=0.001, format="%.3f"
    )

with col2:
    st.markdown("### Mediciones y composición")
    free_sulfur_dioxide = st.number_input(
        "Dióxido de azufre libre (Free SO₂)", min_value=0.0, max_value=300.0, value=11.0, step=1.0
    )
    total_sulfur_dioxide = st.number_input(
        "Dióxido de azufre total (Total SO₂)", min_value=0.0, max_value=400.0, value=34.0, step=1.0
    )
    density = st.number_input(
        "Densidad (Density)", min_value=0.8, max_value=1.2,
        value=0.9968, step=0.0001, format="%.4f"
    )
    pH = st.number_input(
        "Acidez total (pH)", min_value=0.0, max_value=14.0, value=3.20, step=0.01
    )
    sulphates = st.number_input(
        "Sulfatos (Sulphates)", min_value=0.0, max_value=3.0, value=0.56, step=0.01
    )
    alcohol = st.number_input(
        "Contenido de alcohol (Alcohol % vol)", min_value=0.0, max_value=25.0, value=10.5, step=0.1
    )

st.markdown("---")

# ----------------------------------------
# Nombre del vino (opcional, para historial)
# ----------------------------------------

nombre_vino = st.text_input(
    "Nombre del vino (opcional)",
    placeholder="Ej: Rioja Reserva 2020",
    help="Se usará en el historial de predicciones"
)

# ----------------------------------------
# Botón de predicción
# ----------------------------------------

if st.button("🚀 Evaluar calidad del vino", type="primary", use_container_width=True):

    payload = {
        "fixed_acidity":        fixed_acidity,
        "volatile_acidity":     volatile_acidity,
        "citric_acid":          citric_acid,
        "residual_sugar":       residual_sugar,
        "chlorides":            chlorides,
        "free_sulfur_dioxide":  free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density":              density,
        "pH":                   pH,
        "sulphates":            sulphates,
        "alcohol":              alcohol,
        "wine_type":            wine_type
    }

    try:
        response = requests.post(
            "http://api:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            res = response.json()

            st.success("¡Análisis completado!")

            # ----------------------------------------
            # Resultados principales — tarjetas grandes
            # ----------------------------------------

            st.markdown("### Resultados")

            card1, card2 = st.columns(2)

            with card1:
                es_bueno = res["prediccion_binaria"] == 1
                icono    = "🟢" if es_bueno else "🔴"
                color_bg = "#1a3a1a" if es_bueno else "#3a1a1a"
                texto    = "Bueno (7–9)" if es_bueno else "Regular / Malo (3–6)"
                confianza = res["prob_bueno"] if es_bueno else res["prob_malo"]

                st.markdown(
                    f"""
                    <div style="
                        background:{color_bg};
                        border-radius:12px;
                        padding:20px 24px;
                        text-align:center;
                        min-height:130px;
                        display:flex;
                        flex-direction:column;
                        justify-content:center;
                    ">
                        <p style="margin:0 0 6px;font-size:13px;color:#aaa;letter-spacing:0.05em;">
                            📊 CLASIFICACIÓN DE CALIDAD
                        </p>
                        <p style="margin:0 0 4px;font-size:28px;font-weight:600;color:#fff;">
                            {icono} {texto}
                        </p>
                        <p style="margin:0;font-size:13px;color:#aaa;">
                            Confianza: {round(confianza * 100, 1)}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with card2:
                nota = res["calidad_ordinal"]
                desc = res["descripcion_ordinal"].capitalize()
                pct  = round((nota - 3) / (9 - 3) * 100)

                st.markdown(
                    f"""
                    <div style="
                        background:#1a2a3a;
                        border-radius:12px;
                        padding:20px 24px;
                        text-align:center;
                        min-height:130px;
                        display:flex;
                        flex-direction:column;
                        justify-content:center;
                    ">
                        <p style="margin:0 0 6px;font-size:13px;color:#aaa;letter-spacing:0.05em;">
                            📈 REGRESIÓN ORDINAL
                        </p>
                        <p style="margin:0 0 4px;font-size:48px;font-weight:700;color:#fff;">
                            {nota}<span style="font-size:20px;color:#aaa;"> / 9</span>
                        </p>
                        <p style="margin:0;font-size:13px;color:#aaa;">
                            {desc}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ----------------------------------------
            # Barra de probabilidades
            # ----------------------------------------

            st.markdown("#### Confianza del modelo de clasificación")

            col_b, col_m = st.columns(2)
            with col_b:
                st.markdown(f"🟢 **Bueno (7–9)**")
                st.progress(res["prob_bueno"])
                st.caption(f"{round(res['prob_bueno']*100, 1)}%")
            with col_m:
                st.markdown(f"🔴 **Regular / Malo (3–6)**")
                st.progress(res["prob_malo"])
                st.caption(f"{round(res['prob_malo']*100, 1)}%")

            # ----------------------------------------
            # Guardar en historial
            # ----------------------------------------

            etiqueta = nombre_vino.strip() if nombre_vino.strip() else f"Vino #{len(st.session_state.historial) + 1}"
            st.session_state.historial.append({
                "🍷 Nombre":        etiqueta,
                "Tipo":             "Tinto" if wine_type == "red" else "Blanco",
                "Clasificación":    "Bueno (7–9)" if res["prediccion_binaria"] == 1 else "Regular/Malo (3–6)",
                "Nota ordinal":     f"{res['calidad_ordinal']} / 9",
                "Confianza":        f"{round((res['prob_bueno'] if res['prediccion_binaria']==1 else res['prob_malo'])*100, 1)}%",
            })

        else:
            st.error(f"Error de la API: código {response.status_code}")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API. "
            "Asegúrate de haber levantado los contenedores con `docker compose up`."
        )
    except Exception as e:
        st.error(f"Error inesperado: {e}")

# ----------------------------------------
# Historial de predicciones
# ----------------------------------------

if st.session_state.historial:
    st.markdown("---")
    st.markdown("### 📋 Historial de predicciones")

    df_hist = pd.DataFrame(st.session_state.historial[::-1])  # más reciente primero

    # Colorear filas según clasificación
    def colorear_fila(row):
        if "Bueno" in row["Clasificación"]:
            return ["background-color: #1a3a1a"] * len(row)
        else:
            return ["background-color: #3a1a1a"] * len(row)

    st.dataframe(
        df_hist.style.apply(colorear_fila, axis=1),
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Limpiar historial"):
        st.session_state.historial = []
        st.rerun()
