import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import re

# --- Style & Layout ---
st.set_page_config(layout="wide", page_title="Foundation Calculation Report")
st.markdown("""
<style>
    .calc-sheet { background-color: #ffffff; border: 1px solid #d1d5db; padding: 40px; border-radius: 4px; color: #111827; }
    .sec-header { border-bottom: 2px solid #1e3a8a; color: #1e40af; font-weight: 800; margin-top: 25px; text-transform: uppercase; }
    .status-pass { color: #15803d; font-weight: bold; }
    .status-fail { color: #b91c1c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Geometry & Design Inputs ---
with st.sidebar:
    st.header("📐 Geometry & Materials")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("Footing & Depth")
    L = st.number_input("Footing Lx", value=7.0)
    W = st.number_input("Footing Lz", value=8.0)
    H = st.number_input("Thickness T", value=1.0)
    D = st.number_input("Total Depth (D)", value=3.0)
    
    st.subheader("Pedestal (Column)")
    cx = st.number_input("Column bx", value=2.0)
    cz = st.number_input("Column bz", value=2.0)

    st.subheader("Densities & Limits")
    if "Imp" in unit_sys:
        gc, gs = 0.150, 0.100  # kcf
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_u, l_u, p_u = "kip", "ft", "ksf"
    else:
        gc, gs = 24.0, 18.0  # kN/m³
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_u, l_u, p_u = "kN", "m", "kPa"
    mu = st.number_input("Friction Coeff (μ)", value=0.45)
    sf_limit = st.number_input("Min Safety Factor", value=1.50)

# --- Load Input ---
st.title("🏗️ Isolated Foundation Calculation Sheet")
st.markdown('<div class="sec-header">1. Load Case Entry</div>', unsafe_allow_html=True)
load_raw = st.text_area("Paste Load Case (LC | FX | FY | FZ | MX | MY | MZ)", value="LC-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75")

def parse_line(text):
    try:
        p = re.split(r'[ \t,]+', text.strip())
        return {"LC": p[0], "Fx": float(p[1]), "Fy": float(p[2]), "Fz": float(p[3]), "Mx": float(p[4]), "My": float(p[5]), "Mz": float(p[6])}
    except: return None

load = parse_line(load_raw)

if load:
    # --- Detailed Weight & Property Logic ---
    Area_ftg = L * W
    Area_col = cx * cz
    Hp = D - H # Pedestal height above footing
    
    Wt_ftg = Area_ftg * H * gc
    Wt_ped = Area_col * Hp * gc
    Wt_soil = (Area_ftg - Area_col) * Hp * gs # Deduct column volume from soil
    Wt_total = Wt_ftg + Wt_ped + Wt_soil

    # Base Forces (Transferred to base)
    P_total = load['Fy'] + Wt_total
    Mx_base = load['Mx'] + abs(load['Fz'] * D)
    Mz_base = load['Mz'] + abs(load['Fx'] * D)

    # Section Properties
    Sx, Sz = (L * W**2)/6, (W * L**2)/6
    Ix, Iz = (L * W**3)/12, (W * L**3)/12

    # Calculations for Checks
    q_max = (P_total/Area_ftg) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
    sf_sl = (abs(P_total)*mu)/math.sqrt(load['Fx']**2+load['Fz']**2)
    sf_ot = min((P_total*W/2)/abs(Mx_base), (P_total*L/2)/abs(Mz_base))

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📜 Calculation Sheet", "🌈 Stress Contour", "🧊 3D Load View", "✏️ GA Sketch"])

    with t1:
        st.markdown('<div class="calc-sheet">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">Section A: Vertical Load Breakdown</div>', unsafe_allow_html=True)
        st.latex(rf"W_{{footing}} = {L} \times {W} \times {H} \times {gc:.3f} = {Wt_ftg:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{pedestal}} = {cx} \times {cz} \times {Hp:.2f} \times {gc:.3f} = {Wt_ped:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{soil}} = ({Area_ftg} - {Area_col}) \times {Hp:.2f} \times {gs:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        st.latex(rf"P_{{total}} = {load['Fy']} + {Wt_ftg:.2f} + {Wt_ped:.2f} + {Wt_soil:.2f} = {P_total:.2f} \text{{ {f_u}}}")

        st.markdown('<div class="sec-header">Section B: Soil Bearing & Stability Summary</div>', unsafe_allow_html=True)
        summary = [
            {"Check": "Soil Bearing", "Actual": f"{q_max:.3f}", "Limit": f"{qa}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Actual": f"{sf_sl:.2f}", "Limit": f"{sf_limit}", "Ratio": round(sf_limit/sf_sl, 2), "Status": "PASS" if sf_sl >= sf_limit else "FAIL"},
            {"Check": "Overturning SF", "Actual": f"{sf_ot:.2f}", "Limit": f"{sf_limit}", "Ratio": round(sf_limit/sf_ot, 2), "Status": "PASS" if sf_ot >= sf_limit else "FAIL"}
        ]
        st.table(pd.DataFrame(summary))
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        # ACTUAL SIZE CONTOUR WITH COLUMN INDICATOR
        x_g, z_g = np.linspace(-L/2, L/2, 50), np.linspace(-W/2, W/2, 50)
        X, Z = np.meshgrid(x_g, z_g)
        Q = (P_total/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        fig_q = go.Figure(data=go.Heatmap(z=Q, x=x_g, y=z_g, colorscale='RdYlGn_r', zmin=0))
        # Add Column Outline
        fig_q.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="black", width=3), fillcolor="rgba(0,0,0,0.1)")
        # Corner Values
        c_x, c_z = [L/2, -L/2, L/2, -L/2], [W/2, W/2, -W/2, -W/2]
        c_q = (P_total/Area_ftg) + (Mx_base * np.array(c_z) / Ix) + (Mz_base * np.array(c_x) / Iz)
        fig_q.add_trace(go.Scatter(x=c_x, y=c_z, mode='text+markers', text=[f"{v:.2f}" for v in c_q], textfont=dict(size=14, color="black")))
        fig_q.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), width=700, height=700, title="Soil Contact Pressure")
        st.plotly_chart(fig_q)

    with t3:
        # MAT 3D STYLE VECTOR VIEW
        fig_3d = go.Figure()
        # Foundation slab
        fig_3d.add_trace(go.Mesh3d(x=[-L/2,L/2,L/2,-L/2,-L/2,L/2,L/2,-L/2], y=[-W/2,-W/2,W/2,W/2,-W/2,-W/2,W/2,W/2], z=[-H,-H,-H,-H,0,0,0,0], color='lightsteelblue', opacity=0.4))
        # Column
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.8))
        # ARROWS WITH VALUES
        # Vertical Arrow (Fy)
        fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[Hp, Hp-1], mode='lines+text', line=dict(color='green', width=10), text=["", f"Fy={load['Fy']}"], textposition="bottom center"))
        # Lateral Arrows
        fig_3d.add_trace(go.Scatter3d(x=[0, 1.5], y=[0,0], z=[Hp, Hp], mode='lines+text', line=dict(color='red', width=5), text=["", f"Fx={load['Fx']}"]))
        fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0, 1.5], z=[Hp, Hp], mode='lines+text', line=dict(color='blue', width=5), text=["", f"Fz={load['Fz']}"]))
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig_3d, use_container_width=True)

    with t4:
        # GA SKETCH (2D Plan View)
        fig_ga = go.Figure()
        # Footing
        fig_ga.add_shape(type="rect", x0=-L/2, y0=-W/2, x1=L/2, y1=W/2, line=dict(color="Blue", width=2))
        # Column
        fig_ga.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="Black", width=2), fillcolor="LightGray")
        # Annotations
        fig_ga.add_annotation(x=L/2, y=0, text=f"Lx={L}", showarrow=True, arrowhead=2)
        fig_ga.add_annotation(x=0, y=W/2, text=f"Lz={W}", showarrow=True, arrowhead=2)
        fig_ga.update_layout(xaxis=dict(range=[-L, L]), yaxis=dict(range=[-W, W], scaleanchor="x", scaleratio=1), title="General Arrangement Plan")
        st.plotly_chart(fig_ga)
