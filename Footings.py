import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

# --- Style & Theme ---
st.set_page_config(layout="wide", page_title="Foundation Design Report")
st.markdown("""
<style>
    .report-box { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 30px; border-radius: 8px; }
    .header-blue { border-bottom: 3px solid #1e40af; color: #1e40af; font-weight: 800; text-transform: uppercase; margin-top: 25px; }
    .status-pass { color: #15803d; font-weight: bold; }
    .status-fail { color: #b91c1c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Design Parameters")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("Safety Factor Limits")
    sf_sliding_limit = st.number_input("Min SF Sliding", value=1.50)
    sf_ot_limit = st.number_input("Min SF Overturning", value=1.50)
    
    st.subheader("Geometry & Materials")
    L = st.number_input("Footing Length (L)", value=12.0 if "Imp" in unit_sys else 4.0)
    W = st.number_input("Footing Width (W)", value=12.0 if "Imp" in unit_sys else 4.0)
    H = st.number_input("Thickness (T)", value=2.5 if "Imp" in unit_sys else 0.8)
    Df = st.number_input("Soil Surcharge (Df)", value=2.0 if "Imp" in unit_sys else 0.6)
    
    if "Imp" in unit_sys:
        gc, gs = 0.150, 0.110 # kcf
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_u, l_u, p_u = "kip", "ft", "ksf"
    else:
        gc, gs = 24.0, 18.0 # kN/m3
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_u, l_u, p_u = "kN", "m", "kPa"
    mu = st.number_input("Friction Coeff (μ)", value=0.45)

# --- Main Logic ---
st.title("🏗️ Isolated Foundation Calculation Sheet")
st.info("Paste your controlling ASD load case. Visuals and math follow MAT 3D logic.")

# Input load case
default_load = "LC-CONTROLLING\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75"
load_raw = st.text_area("Load Input (LC | FX | FY | FZ | MX | MY | MZ)", value=default_load)

def parse(text):
    try:
        p = re.split(r'[ \t,]+', text.strip())
        return {"LC": p[0], "Fx": float(p[1]), "Fy": float(p[2]), "Fz": float(p[3]), 
                "Mx": float(p[4]), "My": float(p[5]), "Mz": float(p[6])}
    except: return None

load = parse(load_raw)

if load:
    # 1. Properties & Weights
    Area = L * W
    Sx, Sz = (L * W**2)/6, (W * L**2)/6
    Ix, Iz = (L * W**3)/12, (W * L**3)/12
    Wt_conc, Wt_soil = Area * H * gc, Area * Df * gs
    Wt_total = Wt_conc + Wt_soil
    P_total = load['Fy'] + Wt_total
    
    # Base Moments
    Mx_base = load['Mx'] + abs(load['Fz'] * H)
    Mz_base = load['Mz'] + abs(load['Fx'] * H)

    tab1, tab2, tab3 = st.tabs(["📜 Calculation Sheet", "🌈 Stress Contour", "🧊 3D Load View"])

    with tab1:
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        
        # --- Weight Calc ---
        st.markdown('<div class="header-blue">1. Gravity Load Breakdown</div>', unsafe_allow_html=True)
        st.latex(rf"W_{{conc}} = {L} \times {W} \times {H} \times {gc:.3f} = {Wt_conc:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{soil}} = {L} \times {W} \times {Df} \times {gs:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        st.latex(rf"P_{{total}} = P_{{app}} + W_{{tot}} = {load['Fy']} + {Wt_total:.2f} = {P_total:.2f} \text{{ {f_u}}}")

        # --- Bearing Calc ---
        st.markdown('<div class="header-blue">2. Bearing Pressure (ASD)</div>', unsafe_allow_html=True)
        st.latex(rf"M_{{x,base}} = {load['Mx']} + |{load['Fz']} \times {H}| = {Mx_base:.2f}")
        st.latex(rf"M_{{z,base}} = {load['Mz']} + |{load['Fx']} \times {H}| = {Mz_base:.2f}")
        q_max = (P_total/Area) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
        st.latex(rf"q_{{max}} = \frac{{{P_total:.2f}}}{{{Area:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_u}}}")

        # --- Stability ---
        st.markdown('<div class="header-blue">3. Stability Checks</div>', unsafe_allow_html=True)
        sf_sl = (abs(P_total) * mu) / math.sqrt(load['Fx']**2 + load['Fz']**2)
        sf_ot = min((P_total*W/2)/abs(Mx_base), (P_total*L/2)/abs(Mz_base))
        st.latex(rf"SF_{{sliding}} = \frac{{{abs(P_total):.2f} \times {mu}}}{{\sqrt{{{load['Fx']}^2 + {load['Fz']}^2}}}} = {sf_sl:.2f}")
        st.latex(rf"SF_{{overturn}} = \min(SF_x, SF_z) = {sf_ot:.2f}")

        # --- FINAL SUMMARY TABLE ---
        st.markdown('<div class="header-blue">4. Executive Summary</div>', unsafe_allow_html=True)
        summary = [
            {"Check": "Soil Bearing", "Actual": f"{q_max:.3f}", "Limit": f"{qa:.2f}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Actual": f"{sf_sl:.2f}", "Limit": f"{sf_sliding_limit:.2f}", "Ratio": round(sf_sliding_limit/sf_sl, 2), "Status": "PASS" if sf_sl >= sf_sliding_limit else "FAIL"},
            {"Check": "Overturning SF", "Actual": f"{sf_ot:.2f}", "Limit": f"{sf_ot_limit:.2f}", "Ratio": round(sf_ot_limit/sf_ot, 2), "Status": "PASS" if sf_ot >= sf_ot_limit else "FAIL"}
        ]
        st.table(pd.DataFrame(summary))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # Heatmap with Corner Values
        x, z = np.linspace(-L/2, L/2, 40), np.linspace(-W/2, W/2, 40)
        X, Z = np.meshgrid(x, z)
        Q = (P_total/Area) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        fig_q = go.Figure(go.Heatmap(z=Q, x=x, y=z, colorscale='RdYlGn_r', zmin=0))
        # Corner Annotations
        cx, cz = [L/2, -L/2, L/2, -L/2], [W/2, W/2, -W/2, -W/2]
        cq = (P_total/Area) + (Mx_base * np.array(cz) / Ix) + (Mz_base * np.array(cx) / Iz)
        fig_q.add_trace(go.Scatter(x=cx, y=cz, mode='text+markers', text=[f"{v:.2f}" for v in cq], textfont=dict(color="black", size=14), name="Corner Pressures"))
        st.plotly_chart(fig_q, use_container_width=True)

    with tab3:
        # Corrected 3D Vector (Pointing correctly from top-center)
        fig_3d = go.Figure()
        # Footing Box
        fig_3d.add_trace(go.Mesh3d(x=[-L/2,L/2,L/2,-L/2,-L/2,L/2,L/2,-L/2], y=[-W/2,-W/2,W/2,W/2,-W/2,-W/2,W/2,W/2], z=[-H,-H,-H,-H,0,0,0,0], 
                                  color='lightsteelblue', opacity=0.5, name='Footing'))
        # Pedestal indicator (1x1 area)
        fig_3d.add_trace(go.Mesh3d(x=[-0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5], y=[-0.5,-0.5,0.5,0.5,-0.5,-0.5,0.5,0.5], z=[0,0,0,0,1,1,1,1], 
                                  color='gray', opacity=0.8, name='Pedestal'))
        # Corrected Vector Logic
        # Resultant Vector scales for visibility
        v_scale = max(L,W) / (abs(load['Fy']) + 1) * 0.5
        # Vector points FROM the top of pedestal (0,0,1) TOWARDS the resultant direction
        fig_3d.add_trace(go.Scatter3d(x=[0, load['Fx']*v_scale], y=[0, load['Fz']*v_scale], z=[1, 1 - abs(load['Fy'])*v_scale], 
                                   mode='lines+markers', line=dict(color='red', width=10), marker=dict(size=4, color='red'), name='Resultant Force'))
        fig_3d.update_layout(scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Z', zaxis_title='Y (Vert)'))
        st.plotly_chart(fig_3d, use_container_width=True)
