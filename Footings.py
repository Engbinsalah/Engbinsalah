import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(layout="wide", page_title="Foundation Sizing Tool")

st.title("🏗️ Foundation Sizing & Review Tool")
st.markdown("Inspired by MAT 3D & STAAD RCDC logic. Review bearing, stability, and sizing.")

# --- Sidebar Inputs ---
st.sidebar.header("Design Parameters")
unit_system = st.sidebar.selectbox("Units", ["Imperial (kip, ft)", "Metric (kN, m)"])

if unit_system == "Imperial (kip, ft)":
    gamma_conc = st.sidebar.number_input("Concrete Density (pcf)", value=150.0) / 1000 
    gamma_soil = st.sidebar.number_input("Soil Density (pcf)", value=110.0) / 1000 
    q_allow = st.sidebar.number_input("Allowable Bearing Pressure (ksf)", value=3.0)
    mu = st.sidebar.number_input("Friction Coefficient (μ)", value=0.4)
    fos_ot_limit = st.sidebar.number_input("Min FOS Overturning", value=1.5)
    fos_sl_limit = st.sidebar.number_input("Min FOS Sliding", value=1.5)
else:
    gamma_conc = st.sidebar.number_input("Concrete Density (kN/m³)", value=24.0)
    gamma_soil = st.sidebar.number_input("Soil Density (kN/m³)", value=18.0)
    q_allow = st.sidebar.number_input("Allowable Bearing Pressure (kPa)", value=150.0)
    mu = st.sidebar.number_input("Friction Coefficient (μ)", value=0.4)
    fos_ot_limit = st.sidebar.number_input("Min FOS Overturning", value=1.5)
    fos_sl_limit = st.sidebar.number_input("Min FOS Sliding", value=1.5)

st.sidebar.subheader("Footing Dimensions")
L = st.sidebar.number_input("Length (X-dir) [L]", value=8.0)
W = st.sidebar.number_input("Width (Z-dir) [W]", value=8.0)
H = st.sidebar.number_input("Thickness [H]", value=2.0)
D_soil = st.sidebar.number_input("Soil Depth above Footing", value=1.0)

# --- Load Input ---
st.subheader("1. Input Loads")
st.info("Paste your load table below. Columns: LC, FX, FY, FZ, MX, MY, MZ")

default_data = """LC,FX,FY,FZ,MX,MY,MZ
[4-1.1:DS+DO+TSEXP],-0.01,0.86,-0.09,-0.93,0.02,0.41
[4-1.2:DS+DO+TSCON+TT],2.91,-0.65,-0.62,-5.33,0.1,-37.75
[4-3.1:DS+DO+TSCON+S],0.01,0.88,0.11,1.01,-0.02,0.19"""

load_input = st.text_area("Paste CSV content here", value=default_data, height=150)

if load_input:
    df_loads = pd.read_csv(StringIO(load_input))
    
    # --- Calculations ---
    Area = L * W
    Sx = (L * W**2) / 6
    Sz = (W * L**2) / 6
    W_ftg = Area * H * gamma_conc
    W_soil = (Area) * D_soil * gamma_soil
    Total_Dead = W_ftg + W_soil

    results = []
    for index, row in df_loads.iterrows():
        P_applied = row['FY']
        Fx, Fz = row['FX'], row['FZ']
        Mx_app, Mz_app = row['MX'], row['MZ']
        
        P_total = P_applied + Total_Dead
        Mx_base = Mx_app + Fz * H
        Mz_base = Mz_app + Fx * H
        
        # Pressures
        q1 = (P_total/Area) + (Mx_base/Sx) + (Mz_base/Sz)
        q2 = (P_total/Area) + (Mx_base/Sx) - (Mz_base/Sz)
        q3 = (P_total/Area) - (Mx_base/Sx) + (Mz_base/Sz)
        q4 = (P_total/Area) - (Mx_base/Sx) - (Mz_base/Sz)
        q_max = max(q1, q2, q3, q4)
        q_min = min(q1, q2, q3, q4)
        
        # Stability
        fos_sl = (abs(P_total) * mu) / np.sqrt(Fx**2 + Fz**2) if np.sqrt(Fx**2 + Fz**2) > 0 else 99
        fos_ot_x = (P_total * W/2) / abs(Mx_base) if Mx_base != 0 else 99
        fos_ot_z = (P_total * L/2) / abs(Mz_base) if Mz_base != 0 else 99
        
        results.append({
            "LC": row['LC'], "P_Total": round(P_total, 2), "q_max": round(q_max, 3),
            "FOS_Sliding": round(fos_sl, 2), "FOS_OT": round(min(fos_ot_x, fos_ot_z), 2),
            "Status": "✅ OK" if (q_max <= q_allow and q_min >= 0 and min(fos_ot_x, fos_ot_z) >= fos_ot_limit) else "❌ FAIL"
        })

    st.subheader("2. Sizing Results")
    st.dataframe(pd.DataFrame(results))

    # --- 3D Visualization ---
    st.subheader("3. 3D Load Application")
    fig = go.Figure()
    # Draw Footing
    fig.add_trace(go.Mesh3d(x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2], 
                           y=[0, 0, 0, 0, -H, -H, -H, -H], 
                           z=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2], 
                           color='lightgray', opacity=0.5))
    # Draw Load Vector
    fig.add_trace(go.Scatter3d(x=[0, df_loads.iloc[0]['FX']], y=[0, -df_loads.iloc[0]['FY']], z=[0, df_loads.iloc[0]['FZ']],
                               mode='lines+markers', line=dict(color='red', width=6), name='Applied Force'))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Vertical', zaxis_title='Z'))
    st.plotly_chart(fig)
