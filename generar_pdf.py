
import pandas as pd
from prophet import Prophet
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xhtml2pdf import pisa

# Leer datos
df = pd.read_csv("arroz.csv")
df['fecha'] = pd.to_datetime(df['fecha'])
df = df[['fecha', 'precio']].rename(columns={'fecha': 'ds', 'precio': 'y'})

# Modelo y predicción
modelo = Prophet()
modelo.fit(df)
futuro = modelo.make_future_dataframe(periods=90)
prediccion = modelo.predict(futuro)

# Generar gráficos
fig1, ax1 = plt.subplots()
ax1.plot(df['ds'], df['y'], marker='o')
ax1.set_title("Gráfico de Precios Históricos")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
fig1.tight_layout()
fig1.savefig("grafico1.png")

fig2 = modelo.plot(prediccion)
ax2 = fig2.gca()
ax2.set_title("Gráfico de Predicción Futura")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Precio")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
fig2.tight_layout()
fig2.savefig("grafico2.png")

# Datos para el HTML
precio_promedio = round(df['y'].mean(), 2)
precio_actual = round(df['y'].iloc[-1], 2)
precio_predicho = round(prediccion['yhat'].iloc[-1], 2)
variacion = round(((precio_predicho - precio_actual) / precio_actual) * 100, 2)

if variacion > 5:
    mensaje = "¡Atención! Se espera un aumento significativo del precio."
    estilo = "sube"
elif variacion < -5:
    mensaje = "Buena noticia: se espera una baja importante en el precio."
    estilo = "baja"
else:
    mensaje = "El precio se mantendría estable en los próximos meses."
    estilo = "estable"

contexto = {
    "producto": "Arroz",
    "fecha": date.today().strftime("%Y-%m-%d"),
    "promedio": precio_promedio,
    "actual": precio_actual,
    "predicho": precio_predicho,
    "variacion": variacion,
    "mensaje": mensaje,
    "estilo": estilo,
    "grafico1": Path("grafico1.png").absolute().as_posix(),
    "grafico2": Path("grafico2.png").absolute().as_posix()
}

# Reemplazar plantilla
with open("reporte.html", "r", encoding="utf-8") as plantilla:
    html = plantilla.read()

for clave, valor in contexto.items():
    html = html.replace(f"{{{{ {clave} }}}}", str(valor))

with open("reporte_final.html", "w", encoding="utf-8") as archivo:
    archivo.write(html)

# Crear PDF con xhtml2pdf
with open("reporte_final.html", "r", encoding="utf-8") as fuente_html, open("reporte_final.pdf", "wb") as salida_pdf:
    pisa_status = pisa.CreatePDF(fuente_html.read(), dest=salida_pdf)

if pisa_status.err:
    print("❌ Error al generar el PDF.")
else:
    print("✅ PDF generado exitosamente con xhtml2pdf.")
