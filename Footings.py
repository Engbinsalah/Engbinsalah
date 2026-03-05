import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

# --- Page Setup & Professional Styling ---
st.set_page_config(layout="wide", page_title="MAT 3D Foundation Calc Sheet")

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
    .math-expr { font-size: 1.1rem; margin: 15px 0; }
    .status-pass { color: #047857; font-weight: 700; }
    .status-fail { color: #b91c1c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Constants & SF Limits ---
with st.sidebar:
    st.header("📋 Design Criteria")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("Safety Factor Limits")
    sf_sliding_limit = st.number_input("Min SF Sliding", value=1.50)
    sf_ot_limit = st.number_input("Min SF Overturning", value=1.50)
    
    st.subheader("Geometry")
    L = st.number_input("Footing Length (L)", value=12.0 if "Imp" in unit_sys else 4.0)
    W = st.number_input("Footing Width (W)", value=12.0 if "Imp" in unit_sys else 4.0)
    H = st.number_input("Footing Thickness (T)", value=2.5 if "Imp" in unit_sys else 0.8)
    Df = st.number_input("Soil Surcharge Depth", value=2.0 if "Imp" in unit_sys else 0.6)

    st.subheader("Materials & Soil")
    if "Imp" in unit_sys:
        gc = st.number_input("Concrete Density (pcf)", value=150.0) / 1000
        gs = st.number_input("Soil Density (pcf)", value=110.0) / 1000
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_u, l_u, p_u = "kip", "ft", "ksf"
    else:
        gc = st.number_input("Concrete Density (kN/m³)", value=24.0)
        gs = st.number_input("Soil Density (kN/m³)", value=18.0)
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_u, l_u, p_u = "kN", "m", "kPa"
    mu = st.number_input("Friction Coeff (μ)", value=0.45)

# --- Main Interface ---
st.title("🏗️ Isolated Footing Sizing & Review")
st.markdown("### Logic & Presentation Logic: MAT 3D")

# Load Input Section
st.markdown('<div class="sec-header">1. Applied Loads (at Top of Foundation)</div>', unsafe_allow_html=True)
default_load = "LC-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75"
load_raw = st.text_area("Paste Load Case (LC | FX | FY | FZ | MX | MY | MZ)", value=default_load, help="Use Tab or Space between numbers")

# Parser
def parse(text):
    try:
        parts = re.split(r'[ \t]+', text.strip())
        return {"LC": parts[0], "Fx": float(parts[1]), "Fy": float(parts[2]), "Fz": float(parts[3]), 
                "Mx": float(parts[4]), "My": float(parts[5]), "Mz": float(parts[6])}
    except: return None

load = parse(load_raw)

if load:
    # 1. Total Weight Calculation
    Area = L * W
    Wt_conc = Area * H * gc
    Wt_soil = Area * Df * gs
    Wt_total = Wt_conc + Wt_soil

    # 2. Base Load Calculation (assuming Fy is reaction, flip for load if needed)
    # We use user logic: +Fy is compression.
    P_total = load['Fy'] + Wt_total
    Mx_base = load['Mx'] + abs(load['Fz'] * H)
    Mz_base = load['Mz'] + abs(load['Fx'] * H)

    # 3. Section Properties
    Sx, Sz = (L * W**2)/6, (W * L**2)/6
    Ix, Iz = (L * W**3)/12, (W * L**3)/12

    # Tabs for Display
    tab1, tab2, tab3 = st.tabs(["📜 Detailed Calc Note", "🌈 Stress Contour", "🧊 3D Load View"])

    with tab1:
        st.markdown('<div class="calc-sheet">', unsafe_allow_html=True)
        
        # Section 1: Gravity
        st.markdown('<div class="sec-header">Section A: Vertical Load Breakdown</div>', unsafe_allow_html=True)
        st.latex(rf"W_{{concrete}} = L \times W \times T \times \gamma_c = {L} \times {W} \times {H} \times {gc:.3f} = {Wt_conc:.2f} \text{{ {f_u}}}")
        st.latex(rf"W_{{soil}} = L \times W \times D_f \times \gamma_s = {L} \times {W} \times {Df} \times {gs:.3f} = {Wt_soil:.2f} \text{{ {f_u}}}")
        st.latex(rf"P_{{total}} = P_{{applied}} + W_{{conc}} + W_{{soil}} = {load['Fy']} + {Wt_conc:.2f} + {Wt_soil:.2f} = {P_total:.2f} \text{{ {f_u}}}")

        # Section 2: Bearing
        st.markdown('<div class="sec-header">Section B: Soil Bearing Pressure</div>', unsafe_allow_html=True)
        st.write("Base Moments including eccentricity from lateral loads:")
        st.latex(rf"M_{{x,base}} = M_x + (F_z \times T) = {load['Mx']} + ({load['Fz']} \times {H}) = {Mx_base:.2f} \text{{ {f_u}-{l_u}}}")
        st.latex(rf"M_{{z,base}} = M_z + (F_x \times T) = {load['Mz']} + ({load['Fx']} \times {H}) = {Mz_base:.2f} \text{{ {f_u}-{l_u}}}")
        
        q_max = (P_total/Area) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
        ratio_b = q_max / qa
        st.latex(rf"q_{{max}} = \frac{{P_{{tot}}}}{{A}} + \frac{{|M_{{x,b}}|}}{{S_x}} + \frac{{|M_{{z,b}}|}}{{S_z}} = \frac{{{P_total:.2f}}}{{{Area:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_u}}}")
        
        status_b = "PASS" if ratio_b <= 1.0 else "FAIL"
        st.markdown(f"**Bearing Check:** {q_max:.3f} / {qa} = **Ratio: {ratio_b:.2f}** → <span class='status-{'pass' if status_b=='PASS' else 'fail'}'>{status_b}</span>", unsafe_allow_html=True)

        # Section 3: Sliding
        st.markdown('<div class="sec-header">Section C: Sliding Stability</div>', unsafe_allow_html=True)
        F_res = abs(P_total) * mu
        F_act = math.sqrt(load['Fx']**2 + load['Fz']**2)
        sf_sl = F_res / F_act if F_act > 0 else 99
        st.latex(rf"F_{{resisting}} = P_{{total}} \times \mu = {P_total:.2f} \times {mu} = {F_res:.2f} \text{{ {f_u}}}")
        st.latex(rf"F_{{acting}} = \sqrt{{F_x^2 + F_z^2}} = \sqrt{{{load['Fx']}^2 + {load['Fz']}^2}} = {F_act:.2f} \text{{ {f_u}}}")
        
        ratio_sl = sf_sliding_limit / sf_sl
        status_sl = "PASS" if sf_sl >= sf_sliding_limit else "FAIL"
        st.markdown(f"**Sliding SF:** {sf_sl:.2f} (Limit: {sf_sliding_limit}) = **Ratio: {ratio_sl:.2f}** → <span class='status-{'pass' if status_sl=='PASS' else 'fail'}'>{status_sl}</span>", unsafe_allow_html=True)

        # Section 4: Overturning
        st.markdown('<div class="sec-header">Section D: Overturning Stability</div>', unsafe_allow_html=True)
        M_res_x, M_res_z = P_total * (W/2), P_total * (L/2)
        sf_ot_x = M_res_x / abs(Mx_base) if Mx_base != 0 else 99
        sf_ot_z = M_res_z / abs(Mz_base) if Mz_base != 0 else 99
        sf_ot_min = min(sf_ot_x, sf_ot_z)
        
        st.latex(rf"M_{{res,x}} = P \times \frac{{W}}{{2}} = {P_total:.2f} \times {W/2} = {M_res_x:.2f}")
        st.latex(rf"SF_{{ot,x}} = \frac{{M_{{res,x}}}}{{M_{{over,x}}}} = \frac{{{M_res_x:.2f}}}{{{abs(Mx_base):.2f}}} = {sf_ot_x:.2f}")
        
        ratio_ot = sf_ot_limit / sf_ot_min
        status_ot = "PASS" if sf_ot_min >= sf_ot_limit else "FAIL"
        st.markdown(f"**Overturning SF:** {sf_ot_min:.2f} (Limit: {sf_ot_limit}) = **Ratio: {ratio_ot:.2f}** → <span class='status-{'pass' if status_ot=='PASS' else 'fail'}'>{status_ot}</span>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # Contour Logic
        res = 40
        x_g = np.linspace(-L/2, L/2, res)
        z_g = np.linspace(-W/2, W/2, res)
        X, Z = np.meshgrid(x_g, z_g)
        Q_dist = (P_total/Area) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        
        fig_q = go.Figure(data=go.Heatmap(z=Q_dist, x=x_g, y=z_g, colorscale='RdYlGn_r', zmin=0, zmax=qa*1.1))
        
        # Corner Labels
        cx, cz = [L/2, -L/2, L/2, -L/2], [W/2, W/2, -W/2, -W/2]
        cq = (P_total/Area) + (Mx_base * np.array(cz) / Ix) + (Mz_base * np.array(cx) / Iz)
        fig_q.add_trace(go.Scatter(x=cx, y=cz, mode='text+markers', text=[f"{v:.2f}" for v in cq], textfont=dict(color="black", size=14), name="Corner Pressures"))
        
        fig_q.update_layout(title="Base Pressure Contour", xaxis_title="X (m/ft)", yaxis_title="Z (m/ft)", width=700, height=600)
        st.plotly_chart(fig_q)

    with tab3:
        # 3D Load logic
        fig_3d = go.Figure()
        # Footing
        fig_3d.add_trace(go.Mesh3d(x=[-L/2, L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2], y=[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2, W/2], z=[-H, -H, -H, -H, 0, 0, 0, 0], 
                                  color='royalblue', opacity=0.4, name='Foundation'))
        # Arrows
        sc = max(L,W)*0.4 / (abs(load['Fy'])+1)
        fig_3d.add_trace(go.Scatter3d(x=[0, load['Fx']*sc], y=[0, load['Fz']*sc], z=[1, 1-abs(load['Fy'])*sc], mode='lines+markers', line=dict(color='red', width=8), name='Load Vector'))
        fig_3d.update_layout(scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Z', zaxis_title='Vert'), margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_3d)

else:
    st.warning("Invalid load format. Please ensure you have 7 columns (LC + 6 values).")
