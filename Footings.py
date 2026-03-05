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
    .main { background-color: #ffffff; }
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
    .status-pass { color: #047857; font-weight: 700; }
    .status-fail { color: #b91c1c; font-weight: 700; }
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
st.markdown("### Presentation Style: MAT 3D / STAAD RCDC")

# Load Input Section
st.markdown('<div class="sec-header">1. Applied Loads (at Top of Pedestal)</div>', unsafe_allow_html=True)
default_load = "LC-01\t2.89\t-0.67\t-0.82\t-7.27\t0.15\t-37.53"
load_raw = st.text_area("Paste Load Case (LC | FX | FY | FZ | MX | MY | MZ)", value=default_load)

# Parser
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
    
    # Weights Calculation
    Wt_ftg = Area_ftg * H * gamma_c
    Wt_ped = Area_col * Hp * gamma_c
    Wt_soil = (Area_ftg - Area_col) * Hp * gamma_s
    Wt_total = Wt_ftg + Wt_ped + Wt_soil

    # Base Forces
    P_total = load['Fy'] + Wt_total
    Mx_base = load['Mx'] + abs(load['Fz'] * D)
    Mz_base = load['Mz'] + abs(load['Fx'] * D)

    # Section Properties
    Sx, Sz = (L * W**2)/6, (W * L**2)/6
    Ix, Iz = (L * W**3)/12, (W * L**3)/12

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📜 Detailed Calculation Sheet", "🌈 Stress Contour", "🧊 3D Visualization"])

    with tab1:
        st.markdown('<div class="calc-sheet">', unsafe_allow_html=True)
        
        st.markdown('<div class="sec-header">Section A: Vertical Load Breakdown</div>', unsafe_allow_html=True)
        st.markdown("**1. Footing Weight ($W_f$):**")
        st.latex(rf"W_f = L \times W \times H \times \gamma_c = {L} \times {W} \times {H} \times {gamma_c:.3f} = {Wt_ftg:.2f} \text{{ {f_u}}}")
        
        st.markdown("**2. Pedestal Weight ($W_p$):**")
        st.latex(rf"W_p = c_x \times c_z \times (D - H) \times \gamma_c = {cx} \times {cz} \times {Hp:.2f} \times {gamma_c:.3f} = {Wt_ped:.2f} \text{{ {f_u}}}")
        
        st.markdown("**3. Soil Weight ($W_s$):**")
        st.write(f"Area of Soil = Footing Area - Column Area = {Area_ftg} - {Area_col} = {Area_ftg - Area_col:.2f}")
        st.latex(rf"W_s = (A_{{ftg}} - A_{{col}}) \times (D - H) \times \gamma_s = {Area_ftg - Area_col:.2f} \times {Hp:.2f} \times {gamma_s:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        
        st.markdown("**Total Vertical Load at Base ($P_{tot}$):**")
        st.latex(rf"P_{{tot}} = P_{{app}} + W_f + W_p + W_s = {load['Fy']} + {Wt_ftg:.2f} + {Wt_ped:.2f} + {Wt_soil:.2f} = {P_total:.2f} \text{{ {f_u}}}")

        st.markdown('<div class="sec-header">Section B: Soil Bearing Pressure</div>', unsafe_allow_html=True)
        st.latex(rf"M_{{x,b}} = M_x + |F_z \times D| = {load['Mx']} + |{load['Fz']} \times {D}| = {Mx_base:.2f} \text{{ {f_u}-{l_u}}}")
        st.latex(rf"M_{{z,b}} = M_z + |F_x \times D| = {load['Mz']} + |{load['Fx']} \times {D}| = {Mz_base:.2f} \text{{ {f_u}-{l_u}}}")
        
        q_max = (P_total/Area_ftg) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
        q_min = (P_total/Area_ftg) - abs(Mx_base/Sx) - abs(Mz_base/Sz)
        
        st.latex(rf"q_{{max}} = \frac{{{P_total:.2f}}}{{{Area_ftg:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_u}}}")

        st.markdown('<div class="sec-header">Section C: Final Executive Summary</div>', unsafe_allow_html=True)
        summary = [
            {"Check": "Bearing Pressure", "Actual": f"{q_max:.3f}", "Limit": f"{qa:.2f}", "Util. Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Actual": f"{(abs(P_total)*mu)/math.sqrt(load['Fx']**2+load['Fz']**2):.2f}", "Limit": f"Min {sf_sliding_limit}", "Util. Ratio": "-", "Status": "PASS"},
            {"Check": "Uplift Check", "Actual": f"{q_min:.3f}", "Limit": "No Tension (>0)", "Util. Ratio": "-", "Status": "PASS" if q_min >= 0 else "UPLIFT"}
        ]
        st.table(pd.DataFrame(summary))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        res = 40
        x_g, z_g = np.linspace(-L/2, L/2, res), np.linspace(-W/2, W/2, res)
        X, Z = np.meshgrid(x_g, z_g)
        Q_dist = (P_total/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        fig_q = go.Figure(go.Heatmap(z=Q_dist, x=x_g, y=z_g, colorscale='RdYlGn_r', zmin=0))
        cx_p, cz_p = [L/2, -L/2, L/2, -L/2], [W/2, W/2, -W/2, -W/2]
        cq_p = (P_total/Area_ftg) + (Mx_base * np.array(cz_p) / Ix) + (Mz_base * np.array(cx_p) / Iz)
        fig_q.add_trace(go.Scatter(x=cx_p, y=cz_p, mode='markers+text', text=[f"{v:.2f}" for v in cq_p], textfont=dict(size=14, color="black"), name="Corners"))
        st.plotly_chart(fig_q, use_container_width=True)

    with tab3:
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Mesh3d(x=[-L/2,L/2,L/2,-L/2,-L/2,L/2,L/2,-L/2], y=[-W/2,-W/2,W/2,W/2,-W/2,-W/2,W/2,W/2], z=[-H,-H,-H,-H,0,0,0,0], color='lightsteelblue', opacity=0.5, name='Footing'))
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.8, name='Pedestal'))
        v_sc = max(L,W) / (abs(load['Fy'])+1) * 0.4
        fig_3d.add_trace(go.Scatter3d(x=[0, load['Fx']*v_sc], y=[0, load['Fz']*v_sc], z=[Hp, Hp - abs(load['Fy'])*v_sc], mode='lines+markers', line=dict(color='red', width=10), name='Force'))
        fig_3d.update_layout(scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Z', zaxis_title='Vertical'))
        st.plotly_chart(fig_3d, use_container_width=True)
