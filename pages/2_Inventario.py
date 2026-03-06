import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="CAI Inventory Intelligence", layout="wide")

st.title("📦 CAI Inventory Intelligence")
st.caption("Análisis automático de inventario: Activo, Exceso y Obsoleto")

st.markdown("### Subir archivo Excel")

uploaded_file = st.file_uploader("Suba su archivo de inventario (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Suba un archivo para comenzar el análisis.")
    st.stop()

df = pd.read_excel(uploaded_file)

required_columns = [
"ItemCode",
"ItemName",
"Category",
"Warehouse",
"OnHandQty",
"UnitCost",
"SalesQty_6M",
"LastMovementDate",
"FirstPurchaseDate"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Faltan columnas: {missing}")
    st.stop()

df["OnHandQty"] = pd.to_numeric(df["OnHandQty"], errors="coerce").fillna(0)
df["UnitCost"] = pd.to_numeric(df["UnitCost"], errors="coerce").fillna(0)
df["SalesQty_6M"] = pd.to_numeric(df["SalesQty_6M"], errors="coerce").fillna(0)

df["LastMovementDate"] = pd.to_datetime(df["LastMovementDate"], errors="coerce")
df["FirstPurchaseDate"] = pd.to_datetime(df["FirstPurchaseDate"], errors="coerce")

df["AvgMonthlySales"] = df["SalesQty_6M"] / 6
df["MonthsOnHand"] = np.where(df["AvgMonthlySales"] > 0, df["OnHandQty"] / df["AvgMonthlySales"], np.nan)
df["InventoryValue"] = df["OnHandQty"] * df["UnitCost"]

def classify(row):

    if pd.isna(row["MonthsOnHand"]):
        return "Obsoleto"

    if row["MonthsOnHand"] <= 1.5:
        return "Activo"

    if row["MonthsOnHand"] <= 6:
        return "Exceso"

    return "Obsoleto"

df["Status"] = df.apply(classify, axis=1)

total_inventory = df["InventoryValue"].sum()
excess_inventory = df[df["Status"]=="Exceso"]["InventoryValue"].sum()
obsolete_inventory = df[df["Status"]=="Obsoleto"]["InventoryValue"].sum()
active_inventory = df[df["Status"]=="Activo"]["InventoryValue"].sum()

st.markdown("## Indicadores principales")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Inventario Total", f"${total_inventory:,.0f}")
c2.metric("Inventario Activo", f"${active_inventory:,.0f}")
c3.metric("Inventario en Exceso", f"${excess_inventory:,.0f}")
c4.metric("Inventario Obsoleto", f"${obsolete_inventory:,.0f}")

st.markdown("## Distribución del Inventario")

fig = px.pie(
df,
names="Status",
values="InventoryValue",
hole=0.5
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("## Capital por Categoría")

category_df = df.groupby("Category")["InventoryValue"].sum().reset_index()

fig2 = px.bar(
category_df,
x="Category",
y="InventoryValue"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("## Top 10 productos obsoletos")

top_obsolete = df[df["Status"]=="Obsoleto"].sort_values("InventoryValue", ascending=False).head(10)

st.dataframe(top_obsolete)

st.markdown("## Tabla completa")

st.dataframe(df)
