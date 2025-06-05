
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import io

st.set_page_config(page_title="Predicción de Precios", layout="wide")
st.title("📊 Sistema de Análisis y Predicción de Precios")

# Selección de producto
producto = st.selectbox("Selecciona un producto:", ["arroz", "azucar", "aceite"])
archivo_csv = f"{producto}.csv"

# Cargar datos
df = pd.read_csv(archivo_csv)
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha')

# Entrada manual de nuevos datos
st.markdown("### ➕ Agregar nuevo dato")
with st.form("formulario"):
    nueva_fecha = st.date_input("Fecha del precio:")
    nuevo_precio = st.number_input("Precio (Bs):", min_value=0.0, format="%.2f")
    enviar = st.form_submit_button("Agregar")
    if enviar:
        nuevo_df = pd.DataFrame({"fecha": [nueva_fecha], "precio": [nuevo_precio]})
        df = pd.concat([df, nuevo_df], ignore_index=True)
        df.to_csv(archivo_csv, index=False)
        st.success("✅ Nuevo dato agregado exitosamente.")

# Mostrar datos
st.markdown("### 📈 Evolución histórica del precio")
fig1, ax1 = plt.subplots()
ax1.plot(df['fecha'], df['precio'], marker='o')
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio (Bs)")
ax1.set_title(f"Precio histórico de {producto.capitalize()}")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Predicción con Prophet
df_modelo = df.rename(columns={"fecha": "ds", "precio": "y"})
modelo = Prophet()
modelo.fit(df_modelo)
futuro = modelo.make_future_dataframe(periods=90)
forecast = modelo.predict(futuro)

# Mostrar predicción
st.markdown("### 🔮 Predicción para los próximos 3 meses")
fig2 = modelo.plot(forecast)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Descargar predicción como CSV
st.download_button(
    label="⬇️ Descargar predicción como CSV",
    data=forecast[['ds', 'yhat']].to_csv(index=False).encode('utf-8'),
    file_name=f"prediccion_{producto}.csv",
    mime='text/csv'
)

# Generar PDF
st.markdown("---")
st.markdown("## 📄 Generar y Descargar Reporte PDF")

if st.button("🖨️ Generar PDF de predicción"):
    with st.spinner("Generando PDF..."):
        resultado = subprocess.run(["python", "generar_pdf.py"])
        if resultado.returncode == 0:
            st.success("✅ PDF generado exitosamente.")
        else:
            st.error("❌ Ocurrió un error al generar el PDF.")

# Mostrar botón si existe el PDF
pdf_path = Path("reporte_final.pdf")
if pdf_path.exists():
    with open(pdf_path, "rb") as file:
        st.download_button(
            label="⬇️ Descargar PDF",
            data=file,
            file_name="reporte_final.pdf",
            mime="application/pdf"
        )
