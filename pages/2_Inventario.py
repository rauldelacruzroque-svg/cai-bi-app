import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# CONFIGURACION DE PAGINA
# ==========================================
st.set_page_config(page_title="CAI Inventory Intelligence", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 CAI Inventory Intelligence")
st.caption("Análisis automático de inventario: Activo, Exceso y Obsoleto")

# ==========================================
# FORMATO REQUERIDO Y PLANTILLA
# ==========================================
st.markdown("## 📥 Formato requerido del archivo")

st.markdown("""
El archivo debe contener estas columnas exactamente:

- Producto
- Categoria
- Sucursal
- Inventario_Actual
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

def clasificar_estado(row):
    meses = row["Meses_Inventario"]
    producto_nuevo = row["Producto_Nuevo"]
    promedio = row["Promedio_Consumo_Mensual"]
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
# KPIs
# ==========================================
total_productos = len(df_filtrado)
inventario_total = df_filtrado["Inventario_Actual"].sum()
promedio_meses = df_filtrado["Meses_Inventario"].replace([np.inf, -np.inf], np.nan).mean()

activos = (df_filtrado["Estado"] == "Activo").sum()
excesos = (df_filtrado["Estado"] == "Exceso").sum()
obsoletos = (df_filtrado["Estado"] == "Obsoleto").sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Productos", f"{total_productos:,.0f}")
col2.metric("Inventario total", f"{inventario_total:,.0f}")
col3.metric("Meses prom.", f"{0 if pd.isna(promedio_meses) else promedio_meses:,.2f}")
col4.metric("Activos", f"{activos:,.0f}")
col5.metric("Excesos", f"{excesos:,.0f}")
col6.metric("Obsoletos", f"{obsoletos:,.0f}")

st.divider()

# ==========================================
# GRAFICOS
# ==========================================
st.markdown("## 📊 Dashboard de inventario")

resumen_estado = df_filtrado["Estado"].value_counts().reset_index()
resumen_estado.columns = ["Estado", "Cantidad"]

resumen_categoria = (
    df_filtrado.groupby("Categoria", as_index=False)["Inventario_Actual"]
    .sum()
    .sort_values("Inventario_Actual", ascending=False)
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
    df_filtrado.groupby("Sucursal", as_index=False)["Inventario_Actual"]
    .sum()
    .sort_values("Inventario_Actual", ascending=False)
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
        y="Inventario_Actual",
        title="Inventario por categoría"
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
    y="Inventario_Actual",
    title="Inventario total por sucursal"
)
st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# INSIGHTS AUTOMATICOS
# ==========================================
st.markdown("## 💡 Insights automáticos")

porc_exceso = (excesos / total_productos * 100) if total_productos > 0 else 0
porc_obsoleto = (obsoletos / total_productos * 100) if total_productos > 0 else 0
porc_activo = (activos / total_productos * 100) if total_productos > 0 else 0

top_categoria_obsoleta = None
if not df_filtrado[df_filtrado["Estado"] == "Obsoleto"].empty:
    tmp = (
        df_filtrado[df_filtrado["Estado"] == "Obsoleto"]
        .groupby("Categoria", as_index=False)["Inventario_Actual"]
        .sum()
        .sort_values("Inventario_Actual", ascending=False)
    )
    top_categoria_obsoleta = tmp.iloc[0]["Categoria"]

insights = [
    f"El {porc_activo:.1f}% de los productos analizados está clasificado como Activo.",
    f"El {porc_exceso:.1f}% de los productos analizados está en Exceso.",
    f"El {porc_obsoleto:.1f}% de los productos analizados está en condición de Obsoleto.",
    f"El inventario promedio representa {0 if pd.isna(promedio_meses) else promedio_meses:.2f} meses de cobertura.",
    f"La sucursal con mayor inventario es {top_sucursal.iloc[0]['Sucursal']}." if not top_sucursal.empty else "No hay información suficiente de sucursales.",
]

if top_categoria_obsoleta:
    insights.append(f"La categoría con mayor inventario obsoleto es {top_categoria_obsoleta}.")

for i, insight in enumerate(insights[:5], start=1):
    st.write(f"{i}. {insight}")

st.divider()

# ==========================================
# TABLA DETALLADA
# ==========================================
st.markdown("## 📋 Detalle del análisis")

columnas_mostrar = [
    "Producto",
    "Categoria",
    "Sucursal",
    "Inventario_Actual",
    "Promedio_Consumo_Mensual",
    "Meses_Inventario",
    "Estado",
    "Producto_Nuevo",
    "Dias_Sin_Venta",
    "Dias_Sin_Compra",
    "Ultima_Fecha_Venta",
    "Ultima_Fecha_Compra",
    "Fecha_Creacion_Producto"
]

st.dataframe(
    df_filtrado[columnas_mostrar].sort_values(
        by=["Estado", "Meses_Inventario"],
        ascending=[True, False]
    ),
    use_container_width=True
)

# ==========================================
# DESCARGA DE RESULTADOS
# ==========================================
resultado_csv = df_filtrado.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Descargar resultado analizado",
    data=resultado_csv,
    file_name="resultado_inventario_analizado.csv",
    mime="text/csv"
)
