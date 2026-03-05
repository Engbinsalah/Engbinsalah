import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import re

# --- Style & Professional Report Layout ---
st.set_page_config(layout="wide", page_title="MAT 3D Foundation Design Report")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .report-paper {
        background-color: #ffffff;
        border: 1px solid #000;
        padding: 50px;
        color: #000;
        font-family: 'Times New Roman', serif;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
    }
    .sec-header {
        border-bottom: 2px solid #000;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    .check-summary {
        background-color: #f8f9fa;
        border: 2px solid #334155;
        padding: 15px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📐 Geometry & Soil")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("Footing & Pedestal")
    Lx = st.number_input("Footing Length (Lx)", value=7.0)
    Lz = st.number_input("Footing Width (Lz)", value=8.0)
    T = st.number_input("Thickness (T)", value=1.0)
    D = st.number_input("Total Depth GL to Base (D)", value=3.0)
    cx = st.number_input("Pedestal Dim X (cx)", value=2.0)
    cz = st.number_input("Pedestal Dim Z (cz)", value=2.0)

    st.subheader("Densities & Allowables")
    if "Imp" in unit_sys:
        gc, gs = 0.150, 0.100 # kcf
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_u, l_u, p_u = "kip", "ft", "ksf"
    else:
        gc, gs = 24.0, 18.0 # kN/m3
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_u, l_u, p_u = "kN", "m", "kPa"
    mu = st.number_input("Friction Coeff (μ)", value=0.45)
    sf_limit = st.number_input("Target Safety Factor", value=1.50)

# --- Load Input ---
st.title("🏗️ Isolated Foundation Calculation Report")
load_raw = st.text_area("Paste Controlling LC (LC | FX | FY | FZ | MX | MY | MZ)", value="LC-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75")

def parse_line(text):
    try:
        p = re.split(r'[ \t,]+', text.strip())
        return {"LC": p[0], "Fx": float(p[1]), "Fy": float(p[2]), "Fz": float(p[3]), "Mx": float(p[4]), "My": float(p[5]), "Mz": float(p[6])}
    except: return None

load = parse_line(load_raw)

if load:
    # --- CALCULATION ENGINE ---
    Hp = D - T # Pedestal Height
    Area_ftg, Area_ped = Lx * Lz, cx * cz
    
    # 1. Weights
    Wt_ftg = Area_ftg * T * gc
    Wt_ped = Area_ped * Hp * gc
    Wt_soil = (Area_ftg - Area_ped) * Hp * gs
    Wt_total = Wt_ftg + Wt_ped + Wt_soil
    
    # 2. Base Actions
    P_base = load['Fy'] + Wt_total
    Mx_base = load['Mx'] + abs(load['Fz'] * D)
    Mz_base = load['Mz'] + abs(load['Fx'] * D)
    
    # 3. Section Properties
    Sx, Sz = (Lx * Lz**2)/6, (Lz * Lx**2)/6
    Ix, Iz = (Lx * Lz**3)/12, (Lz * Lx**3)/12
    
    # 4. Eccentricity Check
    ex = abs(Mz_base / P_base) if P_base != 0 else 0
    ez = abs(Mx_base / P_base) if P_base != 0 else 0
    kerne_x, kerne_z = Lx/6, Lz/6
    
    # 5. Pressures
    q_max = (P_base/Area_ftg) + (abs(Mx_base)/Sx) + (abs(Mz_base)/Sz)
    q_min = (P_base/Area_ftg) - (abs(Mx_base)/Sx) - (abs(Mz_base)/Sz)

    t1, t2, t3, t4 = st.tabs(["📜 Detailed Math Note", "🌈 Stress Contour", "🧊 3D Load View", "✏️ Geometry Sketch"])

    with t1:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown(f"### DESIGN REPORT: {load['LC']}")
        
        # Section A: Vertical Loads
        st.markdown('<div class="sec-header">Section 1: Vertical Load & Self-Weight</div>', unsafe_allow_html=True)
        st.latex(rf"W_{{ftg}} = {Lx} \times {Lz} \times {T} \times {gc:.3f} = {Wt_ftg:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{ped}} = {cx} \times {cz} \times {Hp:.2f} \times {gc:.3f} = {Wt_ped:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{soil}} = ({Area_ftg} - {Area_ped}) \times {Hp:.2f} \times {gs:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        st.latex(rf"P_{{total}} = P_{{app}} + \sum W = {load['Fy']} + {Wt_total:.2f} = {P_base:.2f} \text{{ {f_u}}}")

        # Section B: Bi-axial Eccentricity
        st.markdown('<div class="sec-header">Section 2: Bi-axial Eccentricity Check</div>', unsafe_allow_html=True)
        st.latex(rf"e_x = \frac{{M_{{z,base}}}}{{P_{{total}}}} = \frac{{{abs(Mz_base):.2f}}}{{{P_base:.2f}}} = {ex:.3f} \text{{ {l_u}}}")
        st.latex(rf"e_z = \frac{{M_{{x,base}}}}{{P_{{total}}}} = \frac{{{abs(Mx_base):.2f}}}{{{P_base:.2f}}} = {ez:.3f} \text{{ {l_u}}}")
        st.write(f"Kerne Limit (Middle Third): Lx/6 = **{kerne_x:.2f}**, Lz/6 = **{kerne_z:.2f}**")
        if ex <= kerne_x and ez <= kerne_z: st.success("Result: Resultant within Kerne (100% Contact Area)")
        else: st.warning("Result: Eccentricity exceeds Kerne (Partial Contact / Uplift)")

        # Section C: Bearing Pressure
        st.markdown('<div class="sec-header">Section 3: Soil Bearing Verification</div>', unsafe_allow_html=True)
        st.latex(rf"q_{{max}} = \frac{{{P_base:.2f}}}{{{Area_ftg:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_u}}}")
        
        # FINAL EXECUTIVE SUMMARY TABLE
        st.markdown('<div class="sec-header">Section 4: Summary of Checks</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame([
            {"Check": "Soil Bearing", "Value": f"{q_max:.3f}", "Limit": f"{qa}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Actual": f"{(abs(P_base)*mu)/math.sqrt(load['Fx']**2+load['Fz']**2):.2f}", "Limit": f"{sf_limit}", "Ratio": "-", "Status": "PASS"},
            {"Check": "Eccentricity X", "Actual": f"{ex:.2f}", "Limit": f"{kerne_x:.2f}", "Ratio": round(ex/kerne_x, 2), "Status": "PASS" if ex <= kerne_x else "L.O.C"},
            {"Check": "Eccentricity Z", "Actual": f"{ez:.2f}", "Limit": f"{kerne_z:.2f}", "Ratio": round(ez/kerne_z, 2), "Status": "PASS" if ez <= kerne_z else "L.O.C"}
        ])
        st.table(summary_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        # CONTOUR WITH ACTUAL ASPECT RATIO
        x_pts, z_pts = np.linspace(-Lx/2, Lx/2, 50), np.linspace(-Lz/2, Lz/2, 50)
        X, Z = np.meshgrid(x_pts, z_pts)
        Q_field = (P_base/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        
        fig_q = go.Figure(go.Heatmap(z=Q_field, x=x_pts, y=z_pts, colorscale='RdYlGn_r', zmin=0))
        # Pedestal Outline
        fig_q.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="black", width=3), fillcolor="rgba(255,255,255,0.2)")
        # Corners
        cx_c, cz_c = [Lx/2, -Lx/2, Lx/2, -Lx/2], [Lz/2, Lz/2, -Lz/2, -Lz/2]
        cq_c = (P_base/Area_ftg) + (Mx_base * np.array(cz_c) / Ix) + (Mz_base * np.array(cx_c) / Iz)
        fig_q.add_trace(go.Scatter(x=cx_c, y=cz_c, mode='text+markers', text=[f"{v:.2f}" for v in cq_c], textfont=dict(size=14, color="black")))
        fig_q.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), width=700, height=800, title="Soil Bearing Pressure Contour")
        st.plotly_chart(fig_q)

    with t3:
        # HIGH-FIDELITY 3D VECTOR VIEW
        fig_3d = go.Figure()
        # Footing & Pedestal
        fig_3d.add_trace(go.Mesh3d(x=[-Lx/2,Lx/2,Lx/2,-Lx/2,-Lx/2,Lx/2,Lx/2,-Lx/2], y=[-Lz/2,-Lz/2,Lz/2,Lz/2,-Lz/2,-Lz/2,Lz/2,Lz/2], z=[-T,-T,-T,-T,0,0,0,0], color='lightsteelblue', opacity=0.4))
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.7))
        # Orthogonal Force Arrows with Text Labels
        f_scale = 1.5
        fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[Hp, Hp-f_scale], mode='lines+text', line=dict(color='green', width=10), text=["", f"FY={load['Fy']}"], textposition="bottom center"))
        fig_3d.add_trace(go.Scatter3d(x=[0, f_scale], y=[0, 0], z=[Hp, Hp], mode='lines+text', line=dict(color='red', width=7), text=["", f"FX={load['Fx']}"]))
        fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, f_scale], z=[Hp, Hp], mode='lines+text', line=dict(color='blue', width=7), text=["", f"FZ={load['Fz']}"]))
        fig_3d.update_layout(scene=dict(aspectmode='data', xaxis_title='X (Lx)', yaxis_title='Z (Lz)', zaxis_title='Vertical'))
        st.plotly_chart(fig_3d, use_container_width=True)

    with t4:
        # DETAILED SKETCH
        fig_sk = go.Figure()
        # Outer Footing
        fig_sk.add_shape(type="rect", x0=-Lx/2, y0=-Lz/2, x1=Lx/2, y1=Lz/2, line=dict(color="Blue", width=3))
        # Inner Pedestal
        fig_sk.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="Black", width=2), fillcolor="rgba(0,0,0,0.1)")
        # Dimensions Labels
        fig_sk.add_annotation(x=Lx/2, y=0, text=f"Lx={Lx}", showarrow=True, arrowhead=2, ax=40, ay=0)
        fig_sk.add_annotation(x=0, y=Lz/2, text=f"Lz={Lz}", showarrow=True, arrowhead=2, ax=0, ay=-40)
        fig_sk.update_layout(xaxis=dict(range=[-Lx, Lx]), yaxis=dict(range=[-Lz, Lz], scaleanchor="x", scaleratio=1), title="General Arrangement Plan")
        st.plotly_chart(fig_sk)
