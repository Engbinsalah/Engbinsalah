import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(layout="wide", page_title="Foundation Sizing Tool")

st.title("🏗️ Foundation Sizing & Review (ASD vs LRFD)")
st.markdown("Map Load Cases to ASD (Sizing) or LRFD (Design) logic. Sizing considers self-weight.")

# --- Sidebar Inputs ---
st.sidebar.header("1. Load Case Mapping")
asd_filter = st.sidebar.text_input("ASD LC Identifier (contains)", value="[4-")
lrfd_filter = st.sidebar.text_input("LRFD LC Identifier (contains)", value="[5-")

st.sidebar.header("2. Design Parameters")
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

st.sidebar.subheader("3. Footing Dimensions")
L = st.sidebar.number_input("Length (X-dir) [L]", value=10.0)
W = st.sidebar.number_input("Width (Z-dir) [W]", value=10.0)
H = st.sidebar.number_input("Thickness [H]", value=2.0)
D_soil = st.sidebar.number_input("Soil Depth above Footing", value=1.0)

# --- Load Input ---
st.subheader("1. Input Loads")
default_data = """LC,FX,FY,FZ,MX,MY,MZ
[4-1.2:DS+DO+TSCON+TT],2.91,-0.65,-0.62,-5.33,0.1,-37.75
[5-2.1:1.2(DS+DO)+1.2TSCON+1.0TT+1.6L+0.5S],2.91,-0.47,-0.6,-5.12,0.1,-37.71"""

load_input = st.text_area("Paste CSV content here (LC, FX, FY, FZ, MX, MY, MZ)", value=default_data, height=150)

if load_input:
    df_loads = pd.read_csv(StringIO(load_input))
    
    # Geometry calcs
    Area = L * W
    Sx = (L * W**2) / 6
    Sz = (W * L**2) / 6
    Wt_service = (Area * H * gamma_conc) + (Area * D_soil * gamma_soil)
    Wt_factored = (Area * H * gamma_conc * 1.2) + (Area * D_soil * gamma_soil * 1.2)

    sizing_results = []
    design_results = []

    for index, row in df_loads.iterrows():
        lc_name = str(row['LC'])
        P_app = row['FY']
        Fx, Fz = row['FX'], row['FZ']
        Mx_app, Mz_app = row['MX'], row['MZ']

        # --- ASD SIZING LOGIC ---
        if asd_filter in lc_name:
            P_total = P_app + Wt_service
            Mx_base = Mx_app + Fz * H
            Mz_base = Mz_app + Fx * H
            
            # Pressures
            q_corners = [
                (P_total/Area) + (Mx_base/Sx) + (Mz_base/Sz),
                (P_total/Area) + (Mx_base/Sx) - (Mz_base/Sz),
                (P_total/Area) - (Mx_base/Sx) + (Mz_base/Sz),
                (P_total/Area) - (Mx_base/Sx) - (Mz_base/Sz)
            ]
            q_max, q_min = max(q_corners), min(q_corners)
            
            # Stability
            fos_sl = (abs(P_total) * mu) / np.sqrt(Fx**2 + Fz**2) if np.sqrt(Fx**2 + Fz**2) > 0 else 99
            fos_ot = min((P_total * W/2)/abs(Mx_base) if Mx_base != 0 else 99, 
                         (P_total * L/2)/abs(Mz_base) if Mz_base != 0 else 99)

            sizing_results.append({
                "LC": lc_name, "Type": "ASD", "P_Total": round(P_total, 2), 
                "q_max": round(q_max, 3), "q_min": round(q_min, 3),
                "FOS_Sliding": round(fos_sl, 2), "FOS_OT": round(fos_ot, 2),
                "Status": "✅ OK" if (q_max <= q_allow and q_min >= 0 and fos_ot >= fos_ot_limit) else "❌ FAIL"
            })

        # --- LRFD DESIGN LOGIC ---
        elif lrfd_filter in lc_name:
            P_ult = P_app + Wt_factored
            Mx_ult = Mx_app + Fz * H
            Mz_ult = Mz_app + Fx * H
            
            q_ult = (P_ult/Area) + abs(Mx_ult/Sx) + abs(Mz_ult/Sz)
            
            design_results.append({
                "LC": lc_name, "Type": "LRFD", "Factored P": round(P_ult, 2),
                "Factored Pressure (qu)": round(q_ult, 3),
                "Note": "Use for Reinforcement Design"
            })

    # Display Tables
    st.subheader("2. Sizing Results (ASD)")
    if sizing_results:
        st.dataframe(pd.DataFrame(sizing_results), use_container_width=True)
    else:
        st.warning(f"No ASD cases found matching '{asd_filter}'")

    st.subheader("3. Design Forces (LRFD)")
    if design_results:
        st.dataframe(pd.DataFrame(design_results), use_container_width=True)
    else:
        st.warning(f"No LRFD cases found matching '{lrfd_filter}'")

    # --- 3D Visualization ---
    st.subheader("4. 3D Load View (First Case)")
    if not df_loads.empty:
        fig = go.Figure()
        # Footing
        fig.add_trace(go.Mesh3d(x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2], 
                               y=[0, 0, 0, 0, -H, -H, -H, -H], 
                               z=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2], 
                               color='blue', opacity=0.2, name='Footing'))
        # Load Vector
        fig.add_trace(go.Scatter3d(x=[0, df_loads.iloc[0]['FX']], y=[0, -df_loads.iloc[0]['FY']], z=[0, df_loads.iloc[0]['FZ']],
                                   mode='lines+markers', line=dict(color='red', width=6), name='Force Vector'))
        fig.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig)
