
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Diccionario con rutas a archivos CSV simulados
productos = {
    "Arroz": "arroz.csv",
    "Azúcar": "azucar.csv",
    "Aceite": "aceite.csv"
}

st.set_page_config(page_title="Sistema de Precios", layout="wide")
st.title("📊 Sistema de Análisis y Predicción de Precios")

# Selector de Producto
producto_seleccionado = st.selectbox("Selecciona un producto:", list(productos.keys()))

# Cargar datos del producto seleccionado
df = pd.read_csv(productos[producto_seleccionado])
df['fecha'] = pd.to_datetime(df['fecha'])
df = df[['fecha', 'precio']].rename(columns={'fecha': 'ds', 'precio': 'y'})

# Entrada manual de nuevos datos
st.subheader("📝 Agregar un nuevo precio manualmente")
with st.form("formulario_precio"):
    nueva_fecha = st.date_input("Fecha del nuevo dato", value=datetime.today())
    nuevo_precio = st.number_input("Precio registrado (Bs)", min_value=0.0, step=0.1)
    agregar = st.form_submit_button("Agregar precio")

if agregar:
    nuevo_dato = pd.DataFrame({"ds": [pd.to_datetime(nueva_fecha)], "y": [nuevo_precio]})
    df = pd.concat([df, nuevo_dato], ignore_index=True)
    df = df.sort_values("ds")
    st.success(f"Se agregó el nuevo precio de {nuevo_precio} Bs para el {nueva_fecha.strftime('%Y-%m-%d')}.")

# Crear modelo y predicción
modelo = Prophet()
modelo.fit(df)
futuro = modelo.make_future_dataframe(periods=90)
prediccion = modelo.predict(futuro)

# Indicadores clave
precio_promedio = round(df['y'].mean(), 2)
precio_actual = round(df['y'].iloc[-1], 2)
precio_predicho = round(prediccion['yhat'].iloc[-1], 2)
variacion = round(((precio_predicho - precio_actual) / precio_actual) * 100, 2)

# Visualización en columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Evolución Histórica del Precio")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['ds'], df['y'], marker='o')
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precio")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 Indicadores Clave")
    st.markdown(f"""
    **Precio Promedio:** {precio_promedio} Bs  
    **Precio Actual:** {precio_actual} Bs  
    **Precio Predicho (3 meses):** {precio_predicho} Bs  
    **Variación Esperada:** {variacion}%  
    """)

    st.subheader("⚠️ Alerta")
    if variacion > 5:
        st.error("¡Atención! Se espera un aumento significativo del precio.")
    elif variacion < -5:
        st.success("Buena noticia: se espera una baja importante en el precio.")
    else:
        st.info("El precio se mantendría estable en los próximos meses.")

# Gráfico de predicción con etiquetas claras
st.subheader("🔮 Predicción de Precios (3 meses)")
fig2 = modelo.plot(prediccion)
ax2 = fig2.gca()
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Precio")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Descarga del archivo de predicción
st.subheader("📥 Descargar Predicción")
csv = prediccion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
    'ds': 'Fecha',
    'yhat': 'Predicción',
    'yhat_lower': 'Límite Inferior',
    'yhat_upper': 'Límite Superior'
}).to_csv(index=False)

st.download_button(
    label="Descargar como CSV",
    data=csv,
    file_name=f"prediccion_{producto_seleccionado.lower()}.csv",
    mime='text/csv'
)
