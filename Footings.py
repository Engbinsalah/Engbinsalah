import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import re

# --- Page Setup & Professional Styling ---
st.set_page_config(layout="wide", page_title="Foundation Design Calculation Sheet")

st.markdown("""
<style>
    .calc-sheet {
        background-color: #fcfcfc;
        border: 1px solid #d1d5db;
        padding: 40px;
        border-radius: 4px;
        color: #111827;
        font-family: 'Segoe UI', serif;
    }
    .sec-header {
        border-bottom: 2px solid #1e3a8a;
        color: #1e3a8a;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 10px;
        text-transform: uppercase;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Constants & SF Limits ---
with st.sidebar:
    st.header("📋 Design Criteria & Geometry")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("Safety Factor Limits")
    sf_sliding_limit = st.number_input("Min SF Sliding", value=1.50)
    sf_ot_limit = st.number_input("Min SF Overturning", value=1.50)
    
    st.subheader("Foundation Geometry")
    L = st.number_input("Footing Length (Lx)", value=7.0)
    W = st.number_input("Footing Width (Lz)", value=8.0)
    H = st.number_input("Footing Thickness (T)", value=1.0)
    D = st.number_input("Total Depth GL to Bottom (D)", value=3.0)
    
    st.subheader("Pedestal / Column")
    cx = st.number_input("Column Dim X (cx)", value=2.0)
    cz = st.number_input("Column Dim Z (cz)", value=2.0)

    st.subheader("Densities")
    if "Imp" in unit_sys:
        gamma_c = st.number_input("Concrete Density (pcf)", value=150.0) / 1000
        gamma_s = st.number_input("Soil Density (pcf)", value=100.0) / 1000
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_u, l_u, p_u = "kip", "ft", "ksf"
    else:
        gamma_c = st.number_input("Concrete Density (kN/m³)", value=24.0)
        gamma_s = st.number_input("Soil Density (kN/m³)", value=18.0)
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_u, l_u, p_u = "kN", "m", "kPa"
    mu = st.number_input("Friction Coeff (μ)", value=0.45)

# --- Main Interface ---
st.title("🏗️ Professional Foundation Calculation Sheet")

# Load Input Section
st.markdown('<div class="sec-header">1. Applied Loads (at Top of Pedestal)</div>', unsafe_allow_html=True)
default_load = "LC-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75"
load_raw = st.text_area("Paste Load Case (LC | FX | FY | FZ | MX | MY | MZ)", value=default_load)

def parse_line(text):
    try:
        parts = re.split(r'[ \t,]+', text.strip())
        return {"LC": parts[0], "Fx": float(parts[1]), "Fy": float(parts[2]), "Fz": float(parts[3]), 
                "Mx": float(parts[4]), "My": float(parts[5]), "Mz": float(parts[6])}
    except: return None

load = parse_line(load_raw)

if load:
    # --- Calculations ---
    Area_ftg = L * W
    Area_col = cx * cz
    Hp = D - H # Pedestal Height
    
    # Weight breakdown
    Wt_ftg = Area_ftg * H * gamma_c
    Wt_ped = Area_col * Hp * gamma_c
    Wt_soil = (Area_ftg - Area_col) * Hp * gamma_s
    Wt_total = Wt_ftg + Wt_ped + Wt_soil

    # Base Forces (Transferred to base)
    P_total = load['Fy'] + Wt_total
    Mx_base = load['Mx'] + abs(load['Fz'] * D)
    Mz_base = load['Mz'] + abs(load['Fx'] * D)

    # Section Properties
    Sx, Sz = (L * W**2)/6, (W * L**2)/6
    Ix, Iz = (L * W**3)/12, (W * L**3)/12

    tab1, tab2, tab3 = st.tabs(["📜 Calculation Sheet", "🌈 Stress Contour", "🧊 3D View"])

    with tab1:
        st.markdown('<div class="calc-sheet">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">Section A: Vertical Load Breakdown</div>', unsafe_allow_html=True)
        st.latex(rf"W_f = L \times W \times H \times \gamma_c = {L} \times {W} \times {H} \times {gamma_c:.3f} = {Wt_ftg:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_p = c_x \times c_z \times (D - H) \times \gamma_c = {cx} \times {cz} \times {Hp:.2f} \times {gamma_c:.3f} = {Wt_ped:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_s = (A_{{ftg}} - A_{{col}}) \times (D - H) \times \gamma_s = {Area_ftg - Area_col:.2f} \times {Hp:.2f} \times {gamma_s:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        st.latex(rf"P_{{tot}} = {load['Fy']} + {Wt_ftg:.2f} + {Wt_ped:.2f} + {Wt_soil:.2f} = {P_total:.2f} \text{{ {f_u}}}")

        st.markdown('<div class="sec-header">Section B: Soil Bearing & Summary</div>', unsafe_allow_html=True)
        q_max = (P_total/Area_ftg) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
        st.latex(rf"q_{{max}} = \frac{{{P_total:.2f}}}{{{Area_ftg:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_u}}}")
        
        # Stability
        sf_sl = (abs(P_total)*mu)/math.sqrt(load['Fx']**2+load['Fz']**2)
        
        summary = [
            {"Check": "Bearing", "Actual": f"{q_max:.3f}", "Limit": f"{qa}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Actual": f"{sf_sl:.2f}", "Limit": f"{sf_sliding_limit}", "Ratio": round(sf_sliding_limit/sf_sl, 2), "Status": "PASS" if sf_sl >= sf_sliding_limit else "FAIL"}
        ]
        st.table(pd.DataFrame(summary))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # ACTUAL SIZE CONTOUR
        x_g, z_g = np.linspace(-L/2, L/2, 50), np.linspace(-W/2, W/2, 50)
        X, Z = np.meshgrid(x_g, z_g)
        Q_dist = (P_total/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        
        fig_q = go.Figure(data=go.Heatmap(z=Q_dist, x=x_g, y=z_g, colorscale='RdYlGn_r', zmin=0))
        
        # Corner Annotations
        cx_c, cz_c = [L/2, -L/2, L/2, -L/2], [W/2, W/2, -W/2, -W/2]
        cq_c = (P_total/Area_ftg) + (Mx_base * np.array(cz_c) / Ix) + (Mz_base * np.array(cx_c) / Iz)
        fig_q.add_trace(go.Scatter(x=cx_c, y=cz_c, mode='text+markers', text=[f"{v:.2f}" for v in cq_c], 
                                   textfont=dict(size=14, color="black", family="Arial Black"), name="Corners"))
        
        fig_q.update_layout(title="Soil Contact Pressure Distribution", 
                            xaxis_title=f"Lx ({l_u})", yaxis_title=f"Lz ({l_u})",
                            yaxis=dict(scaleanchor="x", scaleratio=1), # FORCE ACTUAL SIZE ASPECT RATIO
                            width=800, height=800)
        st.plotly_chart(fig_q)

    with tab3:
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Mesh3d(x=[-L/2,L/2,L/2,-L/2,-L/2,L/2,L/2,-L/2], y=[-W/2,-W/2,W/2,W/2,-W/2,-W/2,W/2,W/2], z=[-H,-H,-H,-H,0,0,0,0], color='royalblue', opacity=0.3))
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.8))
        v_sc = max(L,W) / (abs(load['Fy'])+1) * 0.4
        fig_3d.add_trace(go.Scatter3d(x=[0, load['Fx']*v_sc], y=[0, load['Fz']*v_sc], z=[Hp, Hp - abs(load['Fy'])*v_sc], mode='lines+markers', line=dict(color='red', width=10), name='Force'))
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig_3d, use_container_width=True)
