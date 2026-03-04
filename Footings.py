import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(layout="wide", page_title="Foundation Sizing Tool")

st.title("🏗️ Foundation Sizing & Review (STAAD RCDC Style)")
st.markdown("Automated sizing with STAAD reaction sign handling and stability review.")

# --- Sidebar: Design Parameters ---
st.sidebar.header("1. Load Mapping & Convention")
asd_filter = st.sidebar.text_input("ASD LC Identifier (contains)", value="[4-")
lrfd_filter = st.sidebar.text_input("LRFD LC Identifier (contains)", value="[5-")
flip_axial = st.sidebar.checkbox("Flip Axial Sign (STAAD Reaction -> Load)", value=False, 
                                 help="Check this if your input FY is negative for compression.")

st.sidebar.header("2. Soil & Material")
q_allow = st.sidebar.number_input("Allowable Bearing Pressure (ksf/kPa)", value=3.0)
gamma_conc = st.sidebar.number_input("Concrete Density", value=0.150 if "ksf" in str(q_allow) else 24.0)
gamma_soil = st.sidebar.number_input("Soil Density", value=0.110 if "ksf" in str(q_allow) else 18.0)
mu = st.sidebar.number_input("Friction Coefficient (μ)", value=0.45)
fos_ot_limit = st.sidebar.number_input("Min FOS Overturning", value=1.5)
fos_sl_limit = st.sidebar.number_input("Min FOS Sliding", value=1.5)

st.sidebar.header("3. Footing Geometry")
L = st.sidebar.number_input("Length (X-dir)", value=10.0)
W = st.sidebar.number_input("Width (Z-dir)", value=10.0)
H = st.sidebar.number_input("Thickness (T)", value=2.0)
D_soil = st.sidebar.number_input("Soil depth on top", value=1.0)

# --- Load Input ---
st.subheader("1. Input Loads")
default_data = """LC,FX,FY,FZ,MX,MY,MZ
[4-1.2:DS+DO+TSCON+TT],2.91,-0.65,-0.62,-5.33,0.1,-37.75
[5-2.1:1.2(DS+DO)+1.2TSCON+1.0TT+1.6L+0.5S],2.91,-0.47,-0.6,-5.12,0.1,-37.71"""

load_input = st.text_area("Paste CSV (LC, FX, FY, FZ, MX, MY, MZ)", value=default_data, height=150)

if load_input:
    try:
        df_loads = pd.read_csv(StringIO(load_input))
        
        # Geometry / Weight Calcs
        Area = L * W
        Sx, Sz = (L * W**2)/6, (W * L**2)/6
        Wt_service = (Area * H * gamma_conc) + (Area * D_soil * gamma_soil)
        Wt_factored = Wt_service * 1.2  # Typical LRFD factor for Dead

        sizing_results = []
        design_results = []

        for _, row in df_loads.iterrows():
            lc = str(row['LC'])
            # Sign correction: P should be positive for compression
            P_raw = row['FY']
            P_app = -P_raw if flip_axial else P_raw
            
            Fx, Fz = row['FX'], row['FZ']
            Mx_app, Mz_app = row['MX'], row['MZ']

            # --- ASD Logic (Sizing) ---
            if asd_filter in lc:
                P_total = P_app + Wt_service
                Mx_base = Mx_app + (Fz * H)
                Mz_base = Mz_app + (Fx * H)
                
                # Corner Pressures
                q = []
                for i in [1, -1]:
                    for j in [1, -1]:
                        q.append((P_total/Area) + (i*Mx_base/Sx) + (j*Mz_base/Sz))
                q_max, q_min = max(q), min(q)

                # Stability SF
                sf_sliding = (abs(P_total) * mu) / np.sqrt(Fx**2 + Fz**2) if np.sqrt(Fx**2 + Fz**2) > 0 else 99
                sf_ot = min((P_total*W/2)/abs(Mx_base) if Mx_base != 0 else 99, 
                            (P_total*L/2)/abs(Mz_base) if Mz_base != 0 else 99)

                sizing_results.append({
                    "LC": lc, "P_Total": round(P_total, 2), "q_max": round(q_max, 3), "q_min": round(q_min, 3),
                    "SF_Sliding": round(sf_sliding, 2), "SF_Overturning": round(sf_ot, 2),
                    "Util_Bearing": round(q_max/q_allow, 2),
                    "Pass": q_max <= q_allow and q_min >= 0 and sf_sliding >= fos_sl_limit and sf_ot >= fos_ot_limit
                })

            # --- LRFD Logic (Design) ---
            elif lrfd_filter in lc:
                P_ult = P_app + Wt_factored
                q_u = (P_ult/Area) + abs((Mx_app + Fz*H)/Sx) + abs((Mz_app + Fx*H)/Sz)
                design_results.append({"LC": lc, "Factored P": round(P_ult, 2), "q_ult": round(q_u, 3)})

        # --- Display Results ---
        if sizing_results:
            df_asd = pd.DataFrame(sizing_results)
            
            st.subheader("2. Controlling Case Summary")
            c1 = df_asd.loc[df_asd['q_max'].idxmax()]
            c2 = df_asd.loc[df_asd['SF_Sliding'].idxmin()]
            c3 = df_asd.loc[df_asd['SF_Overturning'].idxmin()]
            
            summary = pd.DataFrame([
                {"Check": "Max Bearing", "Controlling LC": c1['LC'], "Value": c1['q_max'], "Limit": q_allow, "Ratio": c1['Util_Bearing']},
                {"Check": "Min Sliding SF", "Controlling LC": c2['LC'], "Value": c2['SF_Sliding'], "Limit": fos_sl_limit, "Ratio": round(fos_sl_limit/c2['SF_Sliding'], 2)},
                {"Check": "Min Overturning SF", "Controlling LC": c3['LC'], "Value": c3['SF_Overturning'], "Limit": fos_ot_limit, "Ratio": round(fos_ot_limit/c3['SF_Overturning'], 2)}
            ])
            st.table(summary)

            st.subheader("3. All ASD Load Cases (Sizing)")
            st.dataframe(df_asd, use_container_width=True)
        else:
            st.warning(f"No ASD cases found matching '{asd_filter}'. Check the mapping filter in sidebar.")

        if design_results:
            st.subheader("4. LRFD Load Cases (Structural Design)")
            st.dataframe(pd.DataFrame(design_results), use_container_width=True)

        # --- 3D Load Plot ---
        st.subheader("5. 3D Load Visualization")
        sel = st.selectbox("Select LC", df_loads['LC'])
        r = df_loads[df_loads['LC'] == sel].iloc[0]
        f_y = -r['FY'] if flip_axial else r['FY']
        
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2], 
                               y=[0, 0, 0, 0, -H, -H, -H, -H], 
                               z=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2], color='gray', opacity=0.3))
        fig.add_trace(go.Scatter3d(x=[0, r['FX']], y=[0, -f_y], z=[0, r['FZ']], mode='lines+markers', line=dict(color='red', width=6)))
        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X (L)', yaxis_title='Vertical', zaxis_title='Z (W)'))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
