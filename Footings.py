import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re

st.set_page_config(layout="wide", page_title="MAT 3D Style Foundation Tool")

st.title("🏗️ Foundation Sizing & Review (MAT 3D Style)")
st.markdown("Automated sizing with STAAD reaction sign handling and 3D Vector visualization.")

# --- Sidebar: Design Parameters ---
st.sidebar.header("1. Load Mapping & Convention")
asd_filter = st.sidebar.text_input("ASD LC Identifier (contains)", value="[4-")
lrfd_filter = st.sidebar.text_input("LRFD LC Identifier (contains)", value="[5-")
flip_axial = st.sidebar.checkbox("Flip Axial Sign (Reaction to Load)", value=True)

st.sidebar.header("2. Soil & Material")
q_allow = st.sidebar.number_input("Allowable Bearing Pressure (ksf)", value=3.0)
gamma_conc = st.sidebar.number_input("Concrete Density (pcf)", value=150.0) / 1000
gamma_soil = st.sidebar.number_input("Soil Density (pcf)", value=110.0) / 1000
mu = st.sidebar.number_input("Friction Coefficient (μ)", value=0.45)
fos_ot_limit = st.sidebar.number_input("Min FOS Overturning", value=1.5)
fos_sl_limit = st.sidebar.number_input("Min FOS Sliding", value=1.5)

st.sidebar.header("3. Footing Geometry")
L = st.sidebar.number_input("Length (X-dir) [L]", value=12.0)
W = st.sidebar.number_input("Width (Z-dir) [W]", value=12.0)
H = st.sidebar.number_input("Thickness [H]", value=2.5)
D_soil = st.sidebar.number_input("Soil Depth on top", value=2.0)

# --- Load Input ---
st.subheader("1. Input Loads")
load_input = st.text_area("Paste Load Table (Tabs/Spaces/Commas supported)", height=150)

if load_input:
    try:
        clean_input = re.sub(r'[ \t]+', ',', load_input.strip())
        df_loads = pd.read_csv(StringIO(clean_input))
        
        # Geometry & Self-Weight
        Area = L * W
        Sx, Sz = (L * W**2)/6, (W * L**2)/6
        Weight_Service = (Area * H * gamma_conc) + (Area * D_soil * gamma_soil)
        Weight_Factored = Weight_Service * 1.2

        asd_results = []
        lrfd_results = []

        for _, row in df_loads.iterrows():
            lc = str(row['LC'])
            P_app = -row['FY'] if flip_axial else row['FY']
            Fx, Fz = row['FX'], row['FZ']
            Mx_app, Mz_app = row['MX'], row['MZ']

            if asd_filter in lc:
                P_total = P_app + Weight_Service
                Mx_base = Mx_app + (Fz * H)
                Mz_base = Mz_app + (Fx * H)
                q = [(P_total/Area) + (i*Mx_base/Sx) + (j*Mz_base/Sz) for i in [1, -1] for j in [1, -1]]
                q_max, q_min = max(q), min(q)
                sf_sl = (abs(P_total) * mu) / np.sqrt(Fx**2 + Fz**2) if np.sqrt(Fx**2 + Fz**2) > 0 else 99
                sf_ot = min((P_total*W/2)/abs(Mx_base) if Mx_base != 0 else 99, (P_total*L/2)/abs(Mz_base) if Mz_base != 0 else 99)
                
                asd_results.append({"LC": lc, "P_Total": round(P_total, 2), "q_max": round(q_max, 3), "q_min": round(q_min, 3), "SF_Sliding": round(sf_sl, 2), "SF_OT": round(sf_ot, 2), "Ratio": round(q_max/q_allow, 2)})

            elif lrfd_filter in lc:
                P_ult = P_app + Weight_Factored
                q_u = (P_ult/Area) + abs((Mx_app + Fz*H)/Sx) + abs((Mz_app + Fx*H)/Sz)
                lrfd_results.append({"LC": lc, "P_ult": round(P_ult, 2), "q_ult": round(q_u, 3)})

        # --- Summary Tables ---
        if asd_results:
            df_asd = pd.DataFrame(asd_results)
            st.subheader("2. Controlling Case & SF Ratios")
            c_bearing = df_asd.loc[df_asd['q_max'].idxmax()]
            c_sliding = df_asd.loc[df_asd['SF_Sliding'].idxmin()]
            c_ot = df_asd.loc[df_asd['SF_OT'].idxmin()]
            
            st.table(pd.DataFrame([
                {"Check": "Bearing", "LC": c_bearing['LC'], "Value": c_bearing['q_max'], "Limit": q_allow, "Ratio": c_bearing['Ratio']},
                {"Check": "Sliding", "LC": c_sliding['LC'], "Value": c_sliding['SF_Sliding'], "Limit": fos_sl_limit, "SF": c_sliding['SF_Sliding']},
                {"Check": "Overturning", "LC": c_ot['LC'], "Value": c_ot['SF_OT'], "Limit": fos_ot_limit, "SF": c_ot['SF_OT']}
            ]))

        # --- 3D Visualization (MAT 3D Style) ---
        st.subheader("3. 3D Load Visualization")
        sel_lc = st.selectbox("Select LC", df_loads['LC'])
        r = df_loads[df_loads['LC'] == sel_lc].iloc[0]
        f_y = -r['FY'] if flip_axial else r['FY']
        
        fig = go.Figure()

        # Footing Geometry (Box)
        fig.add_trace(go.Mesh3d(
            x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2],
            y=[0, 0, 0, 0, -H, -H, -H, -H],
            z=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2],
            color='lightgray', opacity=0.6, flatshading=True, name='Footing'
        ))

        # Resultant Vector Arrow (From Top Center)
        scale = 5.0 # Scale for visibility
        fig.add_trace(go.Scatter3d(
            x=[0, r['FX']*scale], y=[0, -f_y*scale], z=[0, r['FZ']*scale],
            mode='lines+markers+text',
            line=dict(color='red', width=10),
            marker=dict(size=5, symbol='cone', color='red'),
            text=["", f"Resultant: {round(np.sqrt(r['FX']**2 + f_y**2 + r['FZ']**2),2)}"],
            name='Applied Resultant'
        ))

        # Axis Indicators (MAT 3D style)
        fig.add_trace(go.Scatter3d(x=[L/2+2, L/2+4], y=[0,0], z=[0,0], mode='lines', line=dict(color='blue', width=4), name='X-Axis'))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[W/2+2, W/2+4], mode='lines', line=dict(color='green', width=4), name='Z-Axis'))

        fig.update_layout(scene=dict(
            aspectmode='data',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, title="Vertical (H)"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True),
        ), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
