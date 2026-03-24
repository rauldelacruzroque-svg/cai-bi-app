import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ==========================================
# CONFIGURACION DE PAGINA
# ==========================================
st.set_page_config(page_title="CAI Inventory Intelligence", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.sub-title {
    font-size: 1rem;
    color: #A9B4C2;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}
.small-note {
    font-size: 0.9rem;
    color: #A9B4C2;
}
.metric-card {
    padding: 0.8rem;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📦 CAI Inventory Intelligence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Diagnóstico inteligente de inventario | Activo, Exceso, Obsoleto, Valor y Recomendaciones Ejecutivas</div>",
    unsafe_allow_html=True
)

# ==========================================
# FORMATO REQUERIDO Y PLANTILLA
# ==========================================
st.markdown("<div class='section-title'>📥 Formato requerido del archivo</div>", unsafe_allow_html=True)

st.markdown("""
El archivo debe contener estas columnas exactamente:

- Producto
- Categoria
- Sucursal
- Inventario_Actual
- Costo_Unitario
- Consumo_Mes_1
- Consumo_Mes_2
- Consumo_Mes_3
- Consumo_Mes_4
- Consumo_Mes_5
- Consumo_Mes_6
- Ultima_Fecha_Venta
- Ultima_Fecha_Compra
- Fecha_Creacion_Producto
""")

plantilla_ejemplo = pd.DataFrame({
    "Producto": ["Producto A", "Producto B", "Producto C"],
    "Categoria": ["Bebidas", "Snacks", "Limpieza"],
    "Sucursal": ["Santiago", "Santo Domingo", "La Vega"],
    "Inventario_Actual": [120, 40, 300],
    "Costo_Unitario": [50, 120, 30],
    "Consumo_Mes_1": [20, 8, 5],
    "Consumo_Mes_2": [18, 7, 6],
    "Consumo_Mes_3": [22, 6, 4],
    "Consumo_Mes_4": [19, 9, 5],
    "Consumo_Mes_5": [21, 5, 3],
    "Consumo_Mes_6": [20, 8, 4],
    "Ultima_Fecha_Venta": ["2026-03-01", "2026-02-20", "2025-10-15"],
    "Ultima_Fecha_Compra": ["2026-02-15", "2026-01-30", "2025-09-10"],
    "Fecha_Creacion_Producto": ["2025-06-10", "2026-01-10", "2024-05-01"]
})

st.markdown("### Ejemplo de plantilla")
st.dataframe(plantilla_ejemplo, use_container_width=True)

csv_plantilla = plantilla_ejemplo.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Descargar plantilla CSV",
    data=csv_plantilla,
    file_name="plantilla_inventario_cai.csv",
    mime="text/csv"
)

st.divider()

# ==========================================
# CARGA DE ARCHIVO
# ==========================================
uploaded_file = st.file_uploader(
    "Sube tu archivo de inventario",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info("Carga un archivo para iniciar el análisis.")
    st.stop()

# ==========================================
# LECTURA DEL ARCHIVO
# ==========================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

# ==========================================
# VALIDACION DE COLUMNAS
# ==========================================
columnas_requeridas = [
    "Producto",
    "Categoria",
    "Sucursal",
    "Inventario_Actual",
    "Costo_Unitario",
    "Consumo_Mes_1",
    "Consumo_Mes_2",
    "Consumo_Mes_3",
    "Consumo_Mes_4",
    "Consumo_Mes_5",
    "Consumo_Mes_6",
    "Ultima_Fecha_Venta",
    "Ultima_Fecha_Compra",
    "Fecha_Creacion_Producto"
]

faltantes = [col for col in columnas_requeridas if col not in df.columns]

if faltantes:
    st.error("Faltan estas columnas en el archivo: " + ", ".join(faltantes))
    st.stop()

# ==========================================
# LIMPIEZA Y CONVERSIONES
# ==========================================
columnas_numericas = [
    "Inventario_Actual",
    "Costo_Unitario",
    "Consumo_Mes_1",
    "Consumo_Mes_2",
    "Consumo_Mes_3",
    "Consumo_Mes_4",
    "Consumo_Mes_5",
    "Consumo_Mes_6"
]

for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

columnas_fechas = [
    "Ultima_Fecha_Venta",
    "Ultima_Fecha_Compra",
    "Fecha_Creacion_Producto"
]

for col in columnas_fechas:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ==========================================
# PARAMETROS DE ANALISIS
# ==========================================
st.sidebar.header("⚙️ Parámetros de análisis")

meses_analisis = st.sidebar.slider(
    "Meses a considerar en el promedio",
    min_value=1,
    max_value=6,
    value=6
)

umbral_activo = st.sidebar.number_input(
    "Límite máximo para Activo (meses)",
    min_value=0.1,
    max_value=12.0,
    value=1.5,
    step=0.1
)

umbral_exceso = st.sidebar.number_input(
    "Límite máximo para Exceso (meses)",
    min_value=0.1,
    max_value=24.0,
    value=6.0,
    step=0.1
)

dias_producto_nuevo = st.sidebar.number_input(
    "Días para considerar producto nuevo",
    min_value=1,
    max_value=365,
    value=90,
    step=1
)

# ==========================================
# CALCULOS PRINCIPALES
# ==========================================
columnas_consumo = [f"Consumo_Mes_{i}" for i in range(1, meses_analisis + 1)]

df["Promedio_Consumo_Mensual"] = df[columnas_consumo].mean(axis=1)

df["Meses_Inventario"] = np.where(
    df["Promedio_Consumo_Mensual"] > 0,
    df["Inventario_Actual"] / df["Promedio_Consumo_Mensual"],
    np.nan
)

hoy = pd.Timestamp.today().normalize()

df["Dias_Desde_Creacion"] = (hoy - df["Fecha_Creacion_Producto"]).dt.days
df["Producto_Nuevo"] = df["Dias_Desde_Creacion"].fillna(99999) <= dias_producto_nuevo

df["Dias_Sin_Venta"] = (hoy - df["Ultima_Fecha_Venta"]).dt.days
df["Dias_Sin_Compra"] = (hoy - df["Ultima_Fecha_Compra"]).dt.days

df["Valor_Inventario"] = df["Inventario_Actual"] * df["Costo_Unitario"]

def clasificar_estado(row):
    meses = row["Meses_Inventario"]
    producto_nuevo = row["Producto_Nuevo"]
    inventario = row["Inventario_Actual"]

    if inventario <= 0:
        return "Sin Inventario"

    if producto_nuevo:
        return "Nuevo"

    if pd.isna(meses):
        return "Sin Rotación"

    if meses < umbral_activo:
        return "Activo"
    elif meses <= umbral_exceso:
        return "Exceso"
    else:
        return "Obsoleto"

df["Estado"] = df.apply(clasificar_estado, axis=1)

# ==========================================
# RECOMENDACIONES POR PRODUCTO
# ==========================================
def generar_recomendacion(row):
    estado = row["Estado"]
    dias_sin_venta = row["Dias_Sin_Venta"] if pd.notna(row["Dias_Sin_Venta"]) else None

    if estado == "Activo":
        return "Mantener nivel actual y monitorear reposición."
    elif estado == "Exceso":
        return "Reducir compras y revisar política de reabastecimiento."
    elif estado == "Obsoleto":
        return "Aplicar liquidación, transferencia o plan de salida."
    elif estado == "Nuevo":
        return "Dar seguimiento antes de clasificar su rotación."
    elif estado == "Sin Rotación":
        if dias_sin_venta is not None and dias_sin_venta > 90:
            return "Revisar producto sin movimiento y validar continuidad."
        return "Validar consumo, histórico y parametrización."
    elif estado == "Sin Inventario":
        return "Sin existencia actual. Revisar disponibilidad y demanda."
    return "Revisar manualmente."

df["Recomendacion"] = df.apply(generar_recomendacion, axis=1)

# ==========================================
# FILTROS
# ==========================================
st.sidebar.header("🔎 Filtros")

sucursales = sorted(df["Sucursal"].dropna().astype(str).unique().tolist())
categorias = sorted(df["Categoria"].dropna().astype(str).unique().tolist())
estados = sorted(df["Estado"].dropna().astype(str).unique().tolist())

sucursales_sel = st.sidebar.multiselect(
    "Sucursal",
    options=sucursales,
    default=sucursales
)

categorias_sel = st.sidebar.multiselect(
    "Categoría",
    options=categorias,
    default=categorias
)

estados_sel = st.sidebar.multiselect(
    "Estado",
    options=estados,
    default=estados
)

df_filtrado = df[
    df["Sucursal"].astype(str).isin(sucursales_sel) &
    df["Categoria"].astype(str).isin(categorias_sel) &
    df["Estado"].astype(str).isin(estados_sel)
].copy()

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# ==========================================
# KPIS
# ==========================================
total_productos = len(df_filtrado)
inventario_total = df_filtrado["Inventario_Actual"].sum()
valor_total_inventario = df_filtrado["Valor_Inventario"].sum()
promedio_meses = df_filtrado["Meses_Inventario"].replace([np.inf, -np.inf], np.nan).mean()

activos = (df_filtrado["Estado"] == "Activo").sum()
excesos = (df_filtrado["Estado"] == "Exceso").sum()
obsoletos = (df_filtrado["Estado"] == "Obsoleto").sum()

valor_exceso = df_filtrado.loc[df_filtrado["Estado"] == "Exceso", "Valor_Inventario"].sum()
valor_obsoleto = df_filtrado.loc[df_filtrado["Estado"] == "Obsoleto", "Valor_Inventario"].sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Productos", f"{total_productos:,.0f}")
col2.metric("Inventario total", f"{inventario_total:,.0f}")
col3.metric("Valor inventario", f"{valor_total_inventario:,.2f}")
col4.metric("Meses prom.", f"{0 if pd.isna(promedio_meses) else promedio_meses:,.2f}")
col5.metric("Valor exceso", f"{valor_exceso:,.2f}")
col6.metric("Valor obsoleto", f"{valor_obsoleto:,.2f}")

st.divider()

# ==========================================
# GRAFICOS
# ==========================================
st.markdown("<div class='section-title'>📊 Dashboard de inventario</div>", unsafe_allow_html=True)

resumen_estado = df_filtrado["Estado"].value_counts().reset_index()
resumen_estado.columns = ["Estado", "Cantidad"]

resumen_categoria = (
    df_filtrado.groupby("Categoria", as_index=False)["Valor_Inventario"]
    .sum()
    .sort_values("Valor_Inventario", ascending=False)
)

top_exceso = (
    df_filtrado[df_filtrado["Estado"] == "Exceso"]
    .sort_values("Meses_Inventario", ascending=False)
    .head(10)
)

top_obsoleto = (
    df_filtrado[df_filtrado["Estado"] == "Obsoleto"]
    .sort_values("Meses_Inventario", ascending=False)
    .head(10)
)

top_sucursal = (
    df_filtrado.groupby("Sucursal", as_index=False)["Valor_Inventario"]
    .sum()
    .sort_values("Valor_Inventario", ascending=False)
)

g1, g2 = st.columns(2)

with g1:
    fig1 = px.pie(
        resumen_estado,
        names="Estado",
        values="Cantidad",
        title="Distribución del inventario por estado"
    )
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    fig2 = px.bar(
        resumen_categoria,
        x="Categoria",
        y="Valor_Inventario",
        title="Valor del inventario por categoría"
    )
    st.plotly_chart(fig2, use_container_width=True)

g3, g4 = st.columns(2)

with g3:
    if not top_exceso.empty:
        fig3 = px.bar(
            top_exceso,
            x="Meses_Inventario",
            y="Producto",
            orientation="h",
            title="Top 10 productos en exceso"
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No hay productos en exceso con los filtros actuales.")

with g4:
    if not top_obsoleto.empty:
        fig4 = px.bar(
            top_obsoleto,
            x="Meses_Inventario",
            y="Producto",
            orientation="h",
            title="Top 10 productos obsoletos"
        )
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No hay productos obsoletos con los filtros actuales.")

fig5 = px.bar(
    top_sucursal,
    x="Sucursal",
    y="Valor_Inventario",
    title="Valor de inventario por sucursal"
)
st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# INSIGHTS AUTOMATICOS
# ==========================================
st.markdown("<div class='section-title'>💡 Insights automáticos</div>", unsafe_allow_html=True)

porc_exceso = (excesos / total_productos * 100) if total_productos > 0 else 0
porc_obsoleto = (obsoletos / total_productos * 100) if total_productos > 0 else 0
porc_activo = (activos / total_productos * 100) if total_productos > 0 else 0

top_categoria_obsoleta = None
if not df_filtrado[df_filtrado["Estado"] == "Obsoleto"].empty:
    tmp = (
        df_filtrado[df_filtrado["Estado"] == "Obsoleto"]
        .groupby("Categoria", as_index=False)["Valor_Inventario"]
        .sum()
        .sort_values("Valor_Inventario", ascending=False)
    )
    top_categoria_obsoleta = tmp.iloc[0]["Categoria"]

top_sucursal_nombre = top_sucursal.iloc[0]["Sucursal"] if not top_sucursal.empty else "N/D"

insights = [
    f"El {porc_activo:.1f}% de los productos analizados está clasificado como Activo.",
    f"El {porc_exceso:.1f}% de los productos analizados está en Exceso.",
    f"El {porc_obsoleto:.1f}% de los productos analizados está en condición de Obsoleto.",
    f"El inventario promedio representa {0 if pd.isna(promedio_meses) else promedio_meses:.2f} meses de cobertura.",
    f"El valor total del inventario analizado es {valor_total_inventario:,.2f}.",
]

if top_categoria_obsoleta:
    insights.append(f"La categoría con mayor valor obsoleto es {top_categoria_obsoleta}.")

insights.append(f"La sucursal con mayor valor de inventario es {top_sucursal_nombre}.")

for i, insight in enumerate(insights[:6], start=1):
    st.write(f"{i}. {insight}")

# ==========================================
# ALERTAS EJECUTIVAS
# ==========================================
st.markdown("<div class='section-title'>🚨 Alertas ejecutivas</div>", unsafe_allow_html=True)

if porc_obsoleto > 20:
    st.error(f"⚠️ ALERTA: Más del {porc_obsoleto:.1f}% de los productos está obsoleto. Se requiere acción inmediata.")
else:
    st.success(f"Nivel de obsolescencia controlado: {porc_obsoleto:.1f}%.")

if porc_exceso > 30:
    st.warning(f"⚠️ Alto nivel de exceso ({porc_exceso:.1f}%). Posible sobrecompra o baja rotación.")
else:
    st.info(f"Nivel de exceso dentro de rango esperado: {porc_exceso:.1f}%.")

if valor_obsoleto > 0:
    st.warning(f"Valor comprometido en inventario obsoleto: {valor_obsoleto:,.2f}")

# ==========================================
# RECOMENDACIONES EJECUTIVAS
# ==========================================
st.markdown("<div class='section-title'>🧭 Recomendaciones ejecutivas</div>", unsafe_allow_html=True)

recomendaciones_ejecutivas = []

if porc_obsoleto > 20:
    recomendaciones_ejecutivas.append("Ejecutar un plan inmediato de liquidación, transferencia o depuración del inventario obsoleto.")
if porc_exceso > 30:
    recomendaciones_ejecutivas.append("Reducir compras en categorías con exceso y revisar parámetros de reabastecimiento.")
if valor_obsoleto > 0:
    recomendaciones_ejecutivas.append("Cuantificar el impacto financiero del obsoleto y presentarlo a gerencia para toma de decisiones.")
if total_productos > 0 and porc_activo < 50:
    recomendaciones_ejecutivas.append("Revisar el balance del inventario, ya que menos del 50% de los productos está en condición activa.")
if not recomendaciones_ejecutivas:
    recomendaciones_ejecutivas.append("Mantener la política actual y continuar monitoreando la rotación y el valor del inventario.")

for i, rec in enumerate(recomendaciones_ejecutivas, start=1):
    st.write(f"{i}. {rec}")

st.divider()

# ==========================================
# TABLA DETALLADA
# ==========================================
st.markdown("<div class='section-title'>📋 Detalle del análisis</div>", unsafe_allow_html=True)

columnas_mostrar = [
    "Producto",
    "Categoria",
    "Sucursal",
    "Inventario_Actual",
    "Costo_Unitario",
    "Valor_Inventario",
    "Promedio_Consumo_Mensual",
    "Meses_Inventario",
    "Estado",
    "Producto_Nuevo",
    "Dias_Sin_Venta",
    "Dias_Sin_Compra",
    "Ultima_Fecha_Venta",
    "Ultima_Fecha_Compra",
    "Fecha_Creacion_Producto",
    "Recomendacion"
]

st.dataframe(
    df_filtrado[columnas_mostrar].sort_values(
        by=["Estado", "Meses_Inventario"],
        ascending=[True, False]
    ),
    use_container_width=True
)

# ==========================================
# FUNCION PDF
# ==========================================
def generar_pdf_ejecutivo():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 0.7 * inch

    def write_line(text, size=10, bold=False, gap=0.22):
        nonlocal y
        if y < 0.8 * inch:
            c.showPage()
            y = height - 0.7 * inch
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawString(0.7 * inch, y, str(text))
        y -= gap * inch

    c.setTitle("Reporte Ejecutivo de Inventario CAI")

    write_line("CAI Inventory Intelligence", size=16, bold=True, gap=0.28)
    write_line("Reporte Ejecutivo de Inventario", size=12, bold=True, gap=0.25)
    write_line(f"Productos analizados: {total_productos}", size=10)
    write_line(f"Inventario total: {inventario_total:,.2f}", size=10)
    write_line(f"Valor total inventario: {valor_total_inventario:,.2f}", size=10)
    write_line(f"Valor en exceso: {valor_exceso:,.2f}", size=10)
    write_line(f"Valor obsoleto: {valor_obsoleto:,.2f}", size=10)
    write_line(f"Meses promedio de cobertura: {0 if pd.isna(promedio_meses) else promedio_meses:.2f}", size=10, gap=0.3)

    write_line("Insights automáticos", size=12, bold=True, gap=0.25)
    for i, item in enumerate(insights[:6], start=1):
        write_line(f"{i}. {item}", size=10)

    y -= 0.1 * inch
    write_line("Alertas ejecutivas", size=12, bold=True, gap=0.25)
    alertas_pdf = []
    if porc_obsoleto > 20:
        alertas_pdf.append(f"Más del {porc_obsoleto:.1f}% de los productos está obsoleto.")
    if porc_exceso > 30:
        alertas_pdf.append(f"Alto nivel de exceso: {porc_exceso:.1f}%.")
    if valor_obsoleto > 0:
        alertas_pdf.append(f"Valor comprometido en obsoleto: {valor_obsoleto:,.2f}.")
    if not alertas_pdf:
        alertas_pdf.append("No se detectan alertas críticas relevantes con los parámetros actuales.")

    for i, item in enumerate(alertas_pdf, start=1):
        write_line(f"{i}. {item}", size=10)

    y -= 0.1 * inch
    write_line("Recomendaciones ejecutivas", size=12, bold=True, gap=0.25)
    for i, item in enumerate(recomendaciones_ejecutivas, start=1):
        write_line(f"{i}. {item}", size=10)

    y -= 0.1 * inch
    write_line("Top productos obsoletos", size=12, bold=True, gap=0.25)
    if top_obsoleto.empty:
        write_line("No hay productos obsoletos para mostrar.", size=10)
    else:
        for _, row in top_obsoleto.head(5).iterrows():
            write_line(
                f"- {row['Producto']} | Meses Inv.: {row['Meses_Inventario']:.2f} | Valor: {row['Valor_Inventario']:,.2f}",
                size=10
            )

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
# DESCARGAS
# ==========================================
resultado_csv = df_filtrado.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Descargar resultado analizado",
    data=resultado_csv,
    file_name="resultado_inventario_analizado.csv",
    mime="text/csv"
)

pdf_buffer = generar_pdf_ejecutivo()

st.download_button(
    label="📄 Descargar PDF ejecutivo",
    data=pdf_buffer,
    file_name="reporte_ejecutivo_inventario_cai.pdf",
    mime="application/pdf"
)
