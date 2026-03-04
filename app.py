# app.py — CAI Business Intelligence (MVP)
# Upload (Excel/CSV) -> Dashboard 1 pantalla + Panel derecho (Resumen/Insights) + PDF con logo
# Campos esperados (como tu PRUEBA APP.xlsx):
# Fecha, ID_Producto, Nombre_Producto, Categoría, Cantidad_Vendida, Precio_Unitario,
# Costo_Unitario, ID_Cliente, Canal_Venta

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CAI Business Intelligence", layout="wide")
st.title("CAI Business Intelligence — Dashboard Interactivo")

# Sticky right panel + small styling
st.markdown(
    """
<style>
/* Panel derecho sticky */
div[data-testid="column"]:nth-of-type(2) {
  position: sticky;
  top: 0.75rem;
  align-self: flex-start;
  height: fit-content;
}
.block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Template config (YOUR schema)
# ---------------------------
REQUIRED_COLS = [
    "Fecha",
    "ID_Producto",
    "Nombre_Producto",
    "Categoría",
    "Cantidad_Vendida",
    "Precio_Unitario",
    "Costo_Unitario",
    "ID_Cliente",
    "Canal_Venta",
]

BRAND_NAME = "CAI Business Intelligence"
LOGO_PATH = "assets/logo_cai.png"


# ---------------------------
# Helpers
# ---------------------------
def safe_div(a, b):
    return np.nan if b in [0, 0.0, None] or pd.isna(b) else a / b


def abbr_number(n: float) -> str:
    if n is None or pd.isna(n):
        return "N/D"
    n = float(n)
    absn = abs(n)
    if absn >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if absn >= 1_000_000:
        return f"{n/1_000_000:.2f}MM"
    if absn >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,.0f}"


def money(n: float, currency: str = "RD$") -> str:
    if n is None or pd.isna(n):
        return "N/D"
    return f"{currency} {abbr_number(n)}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Sube un .csv o .xlsx")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for col in ["Cantidad_Vendida", "Precio_Unitario", "Costo_Unitario"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ID_Cliente"] = pd.to_numeric(df["ID_Cliente"], errors="coerce")
    return df


def validate_df(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Faltan columnas obligatorias: {missing}")

    if "Fecha" in df.columns:
        bad_dates = df["Fecha"].isna().mean()
        if bad_dates > 0.10:
            errors.append(f"Muchas fechas inválidas en 'Fecha' (≈{bad_dates:.0%}).")

    for col in ["Cantidad_Vendida", "Precio_Unitario", "Costo_Unitario"]:
        if col in df.columns and df[col].isna().mean() > 0.10:
            errors.append(f"Muchos valores inválidos en '{col}' (>10%).")

    return errors


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ventas"] = df["Cantidad_Vendida"] * df["Precio_Unitario"]
    df["Costos"] = df["Cantidad_Vendida"] * df["Costo_Unitario"]
    df["Margen"] = df["Ventas"] - df["Costos"]
    return df


def generate_template_csv() -> bytes:
    sample = pd.DataFrame(
        {
            "Fecha": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "ID_Producto": ["P-001", "P-002", "P-003"],
            "Nombre_Producto": ["Producto A", "Producto B", "Producto C"],
            "Categoría": ["Pantalones", "Camisetas", "Leggings"],
            "Cantidad_Vendida": [2, 1, 3],
            "Precio_Unitario": [40, 50, 36],
            "Costo_Unitario": [28, 35, 25.2],
            "ID_Cliente": [101, 102, 101],
            "Canal_Venta": ["Website", "Tienda", "Website"],
        }
    )
    return sample.to_csv(index=False).encode("utf-8")


def executive_summary_and_insights(df: pd.DataFrame) -> tuple[str, list[str]]:
    ventas_total = df["Ventas"].sum()
    margen_total = df["Margen"].sum()
    margen_pct = safe_div(margen_total, ventas_total)

    clientes = df["ID_Cliente"].nunique()
    unidades = df["Cantidad_Vendida"].sum()
    rango = (df["Fecha"].min(), df["Fecha"].max())

    # MoM por mes
    df_m = df.copy()
    df_m["Mes"] = df_m["Fecha"].dt.to_period("M").dt.to_timestamp()
    ventas_mes = df_m.groupby("Mes", as_index=False)["Ventas"].sum().sort_values("Mes")
    mom_text = ""
    if len(ventas_mes) >= 2:
        last = ventas_mes.iloc[-1]["Ventas"]
        prev = ventas_mes.iloc[-2]["Ventas"]
        mom = safe_div(last - prev, prev)
        if mom is not np.nan and not pd.isna(mom):
            mom_text = f"Último mes vs anterior: {mom:+.1%}."

    # Tops
    top_cat = (
        df.groupby("Categoría", as_index=False)["Ventas"]
        .sum()
        .sort_values("Ventas", ascending=False)
    )
    top_canal = (
        df.groupby("Canal_Venta", as_index=False)["Ventas"]
        .sum()
        .sort_values("Ventas", ascending=False)
    )
    top_prod = (
        df.groupby("Nombre_Producto", as_index=False)["Ventas"]
        .sum()
        .sort_values("Ventas", ascending=False)
    )

    resumen = (
        f"Del {rango[0].date()} al {rango[1].date()}, las ventas suman {ventas_total:,.0f} "
        f"con {unidades:,.0f} unidades y {clientes:,} clientes únicos. "
        f"El margen bruto estimado es {margen_total:,.0f} ({margen_pct:.1%}). "
    )
    if mom_text:
        resumen += mom_text

    insights = []
    if len(top_cat):
        insights.append(
            f"Categoría líder: **{top_cat.iloc[0]['Categoría']}** con {top_cat.iloc[0]['Ventas']:,.0f} "
            f"({safe_div(top_cat.iloc[0]['Ventas'], ventas_total):.1%} del total)."
        )
    if len(top_canal):
        insights.append(
            f"Canal líder: **{top_canal.iloc[0]['Canal_Venta']}** con {top_canal.iloc[0]['Ventas']:,.0f} "
            f"({safe_div(top_canal.iloc[0]['Ventas'], ventas_total):.1%} del total)."
        )
    if len(top_prod):
        insights.append(
            f"Producto top: **{top_prod.iloc[0]['Nombre_Producto']}** con {top_prod.iloc[0]['Ventas']:,.0f} "
            f"({safe_div(top_prod.iloc[0]['Ventas'], ventas_total):.1%} del total)."
        )

    # Peor margen % por categoría
    by_cat = df.groupby("Categoría", as_index=False)[["Ventas", "Margen"]].sum()
    by_cat["Margen_%"] = by_cat["Margen"] / by_cat["Ventas"].replace(0, np.nan)
    worst = by_cat.sort_values("Margen_%").iloc[0]
    insights.append(
        f"Categoría con menor margen %: **{worst['Categoría']}** ({worst['Margen_%']:.1%})."
    )

    # Concentración portafolio (Pareto)
    tp = top_prod["Ventas"].values
    cum = np.cumsum(tp) / (tp.sum() if tp.sum() else 1)
    k80 = int(np.argmax(cum >= 0.80)) + 1 if len(tp) else 0
    insights.append(f"Concentración: ~80% de ventas se logra con **{k80}** productos.")

    return resumen, insights[:5]


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = str(text).split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) + len(w) <= max_chars:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def strip_md(s: str) -> str:
    return str(s).replace("**", "")


def build_pdf_report(
    currency: str,
    kpis: dict,
    resumen: str,
    insights: list[str],
    brand_name: str = BRAND_NAME,
    logo_path: str = LOGO_PATH,
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x = 0.75 * inch
    y = height - 0.85 * inch

    # Header: logo + marca + fecha
    logo_w = 1.05 * inch
    logo_h = 1.05 * inch

    try:
        c.drawImage(
            logo_path,
            x,
            y - logo_h + 0.15 * inch,
            width=logo_w,
            height=logo_h,
            mask="auto",
        )
        title_x = x + logo_w + 0.35 * inch
    except Exception:
        title_x = x

    c.setFont("Helvetica-Bold", 16)
    c.drawString(title_x, y, brand_name)

    c.setFont("Helvetica", 10)
    fecha_reporte = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.drawString(title_x, y - 0.22 * inch, f"Reporte ejecutivo | {fecha_reporte} | Moneda: {currency}")

    c.line(x, y - 0.38 * inch, width - 0.75 * inch, y - 0.38 * inch)
    y = y - 0.65 * inch

    # KPIs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "KPIs")
    y -= 0.22 * inch

    c.setFont("Helvetica", 10)
    for k, v in kpis.items():
        c.drawString(x, y, f"- {k}: {v}")
        y -= 0.18 * inch
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.85 * inch
            c.setFont("Helvetica", 10)

    y -= 0.10 * inch

    # Resumen
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Resumen ejecutivo")
    y -= 0.22 * inch

    c.setFont("Helvetica", 10)
    for line in wrap_text(resumen, 95):
        c.drawString(x, y, line)
        y -= 0.18 * inch
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.85 * inch
            c.setFont("Helvetica", 10)

    y -= 0.10 * inch

    # Insights
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Top 5 insights")
    y -= 0.22 * inch

    c.setFont("Helvetica", 10)
    for i, ins in enumerate(insights, start=1):
        text = strip_md(ins)
        for line in wrap_text(f"{i}. {text}", 95):
            c.drawString(x, y, line)
            y -= 0.18 * inch
            if y < 1.2 * inch:
                c.showPage()
                y = height - 0.85 * inch
                c.setFont("Helvetica", 10)

    c.setFont("Helvetica", 9)
    c.drawString(x, 0.65 * inch, "Generado automáticamente por CAI Business Intelligence")

    c.save()
    buffer.seek(0)
    return buffer.read()


# ---------------------------
# Sidebar: plantilla + upload + settings
# ---------------------------
with st.sidebar:
    st.header("1) Plantilla")
    st.download_button(
        "Descargar plantilla CSV",
        data=generate_template_csv(),
        file_name="plantilla_CAI_BI.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Pega tus datos en esta plantilla y súbela de vuelta (CSV o Excel).")

    st.header("2) Subir archivo")
    uploaded = st.file_uploader("Sube tu CSV o Excel", type=["csv", "xlsx", "xls"])

    st.divider()
    currency = st.selectbox("Moneda", ["RD$", "US$"], index=0)
    show_raw = st.toggle("Mostrar datos filtrados", value=False)

if not uploaded:
    st.info("Descarga la plantilla, pega los datos y súbela para generar el dashboard.")
    st.stop()

# ---------------------------
# Load + validate
# ---------------------------
try:
    df = read_uploaded_file(uploaded)
    df = normalize_columns(df)
    df = coerce_types(df)
except Exception as e:
    st.error(f"No pude leer el archivo: {e}")
    st.stop()

errs = validate_df(df)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

df = df.dropna(subset=["Fecha"]).copy()
df = build_metrics(df)

# ---------------------------
# Filters
# ---------------------------
min_date = df["Fecha"].min()
max_date = df["Fecha"].max()

c1, c2, c3 = st.columns(3)
with c1:
    date_range = st.date_input("Rango de fechas", value=(min_date.date(), max_date.date()))
with c2:
    canales = sorted(df["Canal_Venta"].dropna().unique().tolist())
    sel_canal = st.multiselect("Canal", options=canales, default=canales)
with c3:
    cats = sorted(df["Categoría"].dropna().unique().tolist())
    sel_cat = st.multiselect("Categoría", options=cats, default=cats)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

filtered = df[
    (df["Fecha"] >= start_date)
    & (df["Fecha"] <= end_date)
    & (df["Canal_Venta"].isin(sel_canal))
    & (df["Categoría"].isin(sel_cat))
].copy()

if filtered.empty:
    st.warning("No hay datos con esos filtros.")
    st.stop()

# ---------------------------
# KPIs (para líneas de venta)
# ---------------------------
ventas_total = filtered["Ventas"].sum()
costos_total = filtered["Costos"].sum()
margen_total = filtered["Margen"].sum()
margen_pct = safe_div(margen_total, ventas_total)

lineas = filtered.shape[0]
unidades_total = filtered["Cantidad_Vendida"].sum()
clientes_unicos = filtered["ID_Cliente"].nunique()

venta_prom_linea = safe_div(ventas_total, lineas)
venta_prom_cliente = safe_div(ventas_total, clientes_unicos)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ventas", money(ventas_total, currency))
k2.metric("Margen", money(margen_total, currency), f"{margen_pct:.1%}")
k3.metric("Líneas", f"{lineas:,}")
k4.metric("Unidades", f"{unidades_total:,.0f}")
k5.metric("Clientes únicos", f"{clientes_unicos:,}")

a1, a2 = st.columns(2)
a1.caption(f"Venta prom. por línea: {money(venta_prom_linea, currency)}")
a2.caption(f"Venta prom. por cliente: {money(venta_prom_cliente, currency)}")

st.divider()

# ---------------------------
# Layout: izquierda (gráficos) + derecha (resumen/insights/pdf)
# ---------------------------
left, right = st.columns([3.2, 1.2], gap="large")

with right:
    st.subheader("Resumen ejecutivo")
    resumen, insights = executive_summary_and_insights(filtered)
    st.write(resumen)

    st.subheader("Top 5 insights")
    for i, ins in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {ins}")

    # PDF button
    kpis_pdf = {
        "Ventas": money(ventas_total, currency),
        "Costos": money(costos_total, currency),
        "Margen": f"{money(margen_total, currency)} ({margen_pct:.1%})",
        "Líneas": f"{lineas:,}",
        "Unidades": f"{unidades_total:,.0f}",
        "Clientes únicos": f"{clientes_unicos:,}",
        "Venta prom. por línea": money(venta_prom_linea, currency),
        "Venta prom. por cliente": money(venta_prom_cliente, currency),
    }

    pdf_bytes = build_pdf_report(
        currency=currency,
        kpis=kpis_pdf,
        resumen=resumen,
        insights=insights,
        brand_name=BRAND_NAME,
        logo_path=LOGO_PATH,
    )

    st.download_button(
        "📄 Descargar reporte PDF",
        data=pdf_bytes,
        file_name="CAI_Reporte_Dashboard.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with left:
    # ---------- fila 1 ----------
    r1c1, r1c2 = st.columns(2)

    # 1) Ventas por día
    tmp = filtered.copy()
    tmp["Dia"] = tmp["Fecha"].dt.date
    by_day = tmp.groupby("Dia", as_index=False)["Ventas"].sum().sort_values("Dia")
    fig1 = px.line(by_day, x="Dia", y="Ventas", title="Ventas por día")
    r1c1.plotly_chart(fig1, use_container_width=True)

    # 2) Ventas por canal
    by_canal = filtered.groupby("Canal_Venta", as_index=False)["Ventas"].sum().sort_values("Ventas", ascending=False)
    fig2 = px.bar(by_canal, x="Canal_Venta", y="Ventas", title="Ventas por canal")
    r1c2.plotly_chart(fig2, use_container_width=True)

    # ---------- fila 2 ----------
    r2c1, r2c2 = st.columns(2)

    # 3) Mix por categoría
    by_cat = filtered.groupby("Categoría", as_index=False)["Ventas"].sum().sort_values("Ventas", ascending=False)
    fig3 = px.pie(by_cat, names="Categoría", values="Ventas", hole=0.45, title="Mix de ventas por categoría")
    r2c1.plotly_chart(fig3, use_container_width=True)

    # 4) Top 10 productos
    by_prod = (
        filtered.groupby("Nombre_Producto", as_index=False)["Ventas"]
        .sum()
        .sort_values("Ventas", ascending=False)
        .head(10)
    )
    fig4 = px.bar(
        by_prod.sort_values("Ventas"),
        x="Ventas",
        y="Nombre_Producto",
        orientation="h",
        title="Top 10 productos por ventas",
    )
    r2c2.plotly_chart(fig4, use_container_width=True)

    # ---------- fila 3: selector avanzado ----------
    st.subheader("Análisis avanzado")
    modo = st.selectbox("Selecciona vista", ["Margen % por categoría", "Pareto de productos (80/20)"], index=0)

    if modo == "Margen % por categoría":
        by_cat2 = filtered.groupby("Categoría", as_index=False)[["Ventas", "Margen"]].sum()
        by_cat2["Margen_%"] = by_cat2["Margen"] / by_cat2["Ventas"].replace(0, np.nan)
        by_cat2 = by_cat2.sort_values("Margen_%")
        fig5 = px.bar(by_cat2, x="Margen_%", y="Categoría", orientation="h", title="Margen % por categoría")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        # Pareto 80/20 por producto (ventas)
        by_prod_all = (
            filtered.groupby("Nombre_Producto", as_index=False)["Ventas"]
            .sum()
            .sort_values("Ventas", ascending=False)
        )

        by_prod_all["Ventas_Acum"] = by_prod_all["Ventas"].cumsum()
        total = by_prod_all["Ventas"].sum() if by_prod_all["Ventas"].sum() else 1
        by_prod_all["Pct_Acum"] = by_prod_all["Ventas_Acum"] / total

        k80 = int((by_prod_all["Pct_Acum"] <= 0.80).sum())
        k80 = max(k80, 1)

        st.caption(f"Productos necesarios para cubrir ~80% de ventas: **{k80}** de {by_prod_all.shape[0]}")

        top_n = st.slider(
            "Top N productos",
            min_value=10,
            max_value=max(10, min(50, by_prod_all.shape[0])),
            value=min(20, by_prod_all.shape[0]),
        )
        pareto_top = by_prod_all.head(top_n).copy()

        fig_bar = px.bar(
            pareto_top.sort_values("Ventas"),
            x="Ventas",
            y="Nombre_Producto",
            orientation="h",
            title=f"Pareto - Top {top_n} productos por ventas",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = px.line(
            pareto_top,
            x="Nombre_Producto",
            y="Pct_Acum",
            title="Acumulado % (sobre Top N)",
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------
# Optional data preview
# ---------------------------
if show_raw:
    st.subheader("Datos (filtrados)")
    st.dataframe(filtered, use_container_width=True)
