import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re

st.set_page_config(layout="wide", page_title="STAAD Foundation Sizing Tool")

st.title("🏗️ Foundation Sizing & Review (STAAD/RCDC Style)")
st.markdown("Automated sizing with STAAD sign handling. Note: Sizing includes footing & soil weight.")

# --- Sidebar: Design Parameters ---
st.sidebar.header("1. Load Mapping & Convention")
asd_filter = st.sidebar.text_input("ASD LC Identifier (contains)", value="[4-")
lrfd_filter = st.sidebar.text_input("LRFD LC Identifier (contains)", value="[5-")
# STAAD Logic: Often FY (+) is uplift in reactions, or vice versa. 
# Switching sign ensures FY (+) = Compression for the bearing formula.
flip_axial = st.sidebar.checkbox("Flip Axial Sign (Reaction to Load)", value=True, 
                                 help="STAAD reactions are often opposite to the load applied to the foundation.")

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
st.info("Paste your data from Excel or STAAD. The tool handles Tabs, Spaces, or Commas.")
default_data = """LC	FX	FY	FZ	MX	MY	MZ
[4-1.2:DS+DO+TSCON+TT]	2.91	-0.65	-0.62	-5.33	0.1	-37.75
[5-8.1:1.2(DS+DO)+1.2TSCON+1.2TT+0.5L+0.5S]	3.49	-0.78	-0.75	-6.39	0.13	-45.3"""

load_input = st.text_area("Paste Load Table", value=default_data, height=200)

if load_input:
    try:
        # Robust parsing for Tabs/Spaces/Commas
        # Replace multiple spaces/tabs with a comma to standardize
        clean_input = re.sub(r'[ \t]+', ',', load_input.strip())
        df_loads = pd.read_csv(StringIO(clean_input))
        
        # Geometry & Self-Weight
        Area = L * W
        Sx, Sz = (L * W**2)/6, (W * L**2)/6
        Weight_Service = (Area * H * gamma_conc) + (Area * D_soil * gamma_soil)
        Weight_Factored = Weight_Service * 1.2 # LRFD DL Factor

        asd_results = []
        lrfd_results = []

        for _, row in df_loads.iterrows():
            lc = str(row['LC'])
            # STAAD Sign Handling: Convert Reaction to Load
            p_val = row['FY']
            P_app = -p_val if flip_axial else p_val
            
            Fx, Fz = row['FX'], row['FZ']
            Mx_app, Mz_app = row['MX'], row['MZ']

            # --- ASD (Sizing) ---
            if asd_filter in lc:
                P_total = P_app + Weight_Service
                Mx_base = Mx_app + (Fz * H)
                Mz_base = Mz_app + (Fx * H)
                
                # Pressures at 4 corners
                q = []
                for i in [1, -1]:
                    for j in [1, -1]:
                        q.append((P_total/Area) + (i*Mx_base/Sx) + (j*Mz_base/Sz))
                q_max, q_min = max(q), min(q)

                # Stability SF
                sf_sl = (abs(P_total) * mu) / np.sqrt(Fx**2 + Fz**2) if np.sqrt(Fx**2 + Fz**2) > 0 else 99
                sf_ot = min((P_total*W/2)/abs(Mx_base) if Mx_base != 0 else 99, 
                            (P_total*L/2)/abs(Mz_base) if Mz_base != 0 else 99)

                asd_results.append({
                    "LC": lc, "P_Total": round(P_total, 2), "q_max": round(q_max, 3), "q_min": round(q_min, 3),
                    "SF_Sliding": round(sf_sl, 2), "SF_Overturning": round(sf_ot, 2),
                    "Bearing_Ratio": round(q_max/q_allow, 2)
                })

            # --- LRFD (Design) ---
            elif lrfd_filter in lc:
                P_ult = P_app + Weight_Factored
                Mx_u = Mx_app + (Fz * H)
                Mz_u = Mz_app + (Fx * H)
                q_u = (P_ult/Area) + abs(Mx_u/Sx) + abs(Mz_u/Sz)
                lrfd_results.append({"LC": lc, "P_ult": round(P_ult, 2), "q_ult": round(q_u, 3)})

        # --- Display Summary (Controlling LC & SF Ratios) ---
        if asd_results:
            df_asd = pd.DataFrame(asd_results)
            st.subheader("2. Controlling Case & SF Ratios")
            
            c_bearing = df_asd.loc[df_asd['q_max'].idxmax()]
            c_sliding = df_asd.loc[df_asd['SF_Sliding'].idxmin()]
            c_ot = df_asd.loc[df_asd['SF_Overturning'].idxmin()]
            
            summary_table = pd.DataFrame([
                {"Check": "Bearing Pressure", "Controlling LC": c_bearing['LC'], "Value": c_bearing['q_max'], "Limit": q_allow, "SF / Ratio": c_bearing['Bearing_Ratio']},
                {"Check": "Sliding Stability", "Controlling LC": c_sliding['LC'], "Value": c_sliding['SF_Sliding'], "Limit": fos_sl_limit, "SF / Ratio": c_sliding['SF_Sliding']},
                {"Check": "Overturning Stability", "Controlling LC": c_ot['LC'], "Value": c_ot['SF_Overturning'], "Limit": fos_ot_limit, "SF / Ratio": c_ot['SF_Overturning']}
            ])
            st.table(summary_table)

            with st.expander("View All ASD Results"):
                st.dataframe(df_asd, use_container_width=True)
        else:
            st.warning("No ASD load cases detected. Check LC names or sidebar filters.")

        if lrfd_results:
            st.subheader("3. LRFD Design Forces (Strength)")
            st.dataframe(pd.DataFrame(lrfd_results), use_container_width=True)

        # --- 3D Visualization ---
        st.subheader("4. 3D Load Visualization")
        sel_lc = st.selectbox("Select Case for 3D", df_loads['LC'])
        row_3d = df_loads[df_loads['LC'] == sel_lc].iloc[0]
        vert_load = -row_3d['FY'] if flip_axial else row_3d['FY']

        fig = go.Figure()
        # Footing
        fig.add_trace(go.Mesh3d(x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2], 
                               y=[0, 0, 0, 0, -H, -H, -H, -H], 
                               z=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2], 
                               color='royalblue', opacity=0.3, name='Footing'))
        # Load Arrow
        fig.add_trace(go.Scatter3d(x=[0, row_3d['FX']], y=[0, -vert_load], z=[0, row_3d['FZ']], 
                                   mode='lines+markers', line=dict(color='red', width=7), name='Resultant Force'))
        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X (L)', yaxis_title='Vertical', zaxis_title='Z (W)'))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Processing Error: {e}. Please ensure data is in columns: LC, FX, FY, FZ, MX, MY, MZ")
