import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import re

# --- Professional Styling ---
st.set_page_config(layout="wide", page_title="Foundation Engineering Report")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .report-paper {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        padding: 50px;
        color: #111827;
        font-family: 'Segoe UI', 'Times New Roman', serif;
    }
    .sec-header {
        border-bottom: 3px solid #1e3a8a;
        color: #1e3a8a;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
        text-transform: uppercase;
        font-size: 1.2rem;
    }
    .status-pass { color: #15803d; font-weight: bold; border: 1px solid #15803d; padding: 2px 8px; border-radius: 4px; }
    .status-fail { color: #b91c1c; font-weight: bold; border: 1px solid #b91c1c; padding: 2px 8px; border-radius: 4px; }
    .math-box { background-color: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Constants & SF Limits ---
with st.sidebar:
    st.header("📋 Design Constants")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    st.subheader("1. Geometry")
    Lx = st.number_input("Footing Length (Lx)", value=7.0)
    Lz = st.number_input("Footing Width (Lz)", value=8.0)
    T = st.number_input("Footing Thickness (T)", value=1.0)
    D = st.number_input("Total Depth (GL to Base)", value=3.0)
    cx = st.number_input("Column Dim X (cx)", value=2.0)
    cz = st.number_input("Column Dim Z (cz)", value=2.0)

    st.subheader("2. Material Densities")
    if "Imp" in unit_sys:
        gc = st.number_input("Concrete (pcf)", value=150.0) / 1000
        gs = st.number_input("Soil (pcf)", value=100.0) / 1000
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        f_unit, l_unit, p_unit = "kip", "ft", "ksf"
    else:
        gc = st.number_input("Concrete (kN/m³)", value=24.0)
        gs = st.number_input("Soil (kN/m³)", value=18.0)
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        f_unit, l_unit, p_unit = "kN", "m", "kPa"
        
    st.subheader("3. Safety Limits")
    mu = st.number_input("Friction Coeff (μ)", value=0.45)
    sf_sliding_min = st.number_input("Min SF Sliding", value=1.50)
    sf_ot_min = st.number_input("Min SF Overturning", value=1.50)

# --- Load Input Parser ---
st.title("🏗️ Isolated Foundation Calculation Report")
st.markdown("### Structural Engineering Verification Sheet")

default_load = "LC-ASD-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75"
raw_load = st.text_area("Paste Load Case (LC | FX | FY | FZ | MX | MY | MZ)", value=default_load)

def parse_load(text):
    try:
        p = re.split(r'[ \t,]+', text.strip())
        return {"LC": p[0], "Fx": float(p[1]), "Fy": float(p[2]), "Fz": float(p[3]), 
                "Mx": float(p[4]), "My": float(p[5]), "Mz": float(p[6])}
    except: return None

L_DATA = parse_load(raw_load)

if L_DATA:
    # --- CORE CALCULATIONS ---
    Area_ftg = Lx * Lz
    Area_col = cx * cz
    Hp = D - T  # Pedestal/Column height
    
    # Weight Calculation
    Wt_ftg = Area_ftg * T * gc
    Wt_ped = Area_col * Hp * gc
    Wt_soil = (Area_ftg - Area_col) * Hp * gs
    Wt_total = Wt_ftg + Wt_ped + Wt_soil
    
    # Total Base Actions
    P_total = L_DATA['Fy'] + Wt_total
    Mx_base = L_DATA['Mx'] + abs(L_DATA['Fz'] * D)
    Mz_base = L_DATA['Mz'] + abs(L_DATA['Fx'] * D)
    
    # Properties
    Sx, Sz = (Lx * Lz**2)/6, (Lz * Lx**2)/6
    Ix, Iz = (Lx * Lz**3)/12, (Lz * Lx**3)/12
    
    # --- TABS FOR REPORT ---
    tabs = st.tabs(["⚖️ Weight & Area", "🗠 Bearing Pressure", "📉 Stability Check", "🌈 Stress Contour", "🧊 3D Load View"])

    # TAB 1: WEIGHT & AREA
    with tabs[0]:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">1. Gravity Load Breakdown</div>', unsafe_allow_html=True)
        
        st.write("**A. Foundation Components**")
        st.latex(rf"W_{{footing}} = {Lx} \times {Lz} \times {T} \times {gc:.3f} = {Wt_ftg:.2f} \text{{ {f_unit}}}")
        st.latex(rf"W_{{pedestal}} = {cx} \times {cz} \times {Hp:.2f} \times {gc:.3f} = {Wt_ped:.2f} \text{{ {f_u if 'f_u' in locals() else f_unit}}}")
        
        st.write("**B. Soil Overburden (Deducting Column Area)**")
        st.latex(rf"A_{{soil}} = A_{{ftg}} - A_{{col}} = {Area_ftg} - {Area_col} = {Area_ftg - Area_col:.2f} \text{{ {l_unit}}}^2")
        st.latex(rf"W_{{soil}} = A_{{soil}} \times (D - T) \times \gamma_s = {Area_ftg - Area_col:.2f} \times {Hp:.2f} \times {gs:.3f} = {Wt_soil:.2f} \text{{ {f_unit}}}")
        
        st.write("**C. Total Vertical Load at Base**")
        st.latex(rf"P_{{total}} = P_{{applied}} + \sum W = {L_DATA['Fy']} + {Wt_total:.2f} = {P_total:.2f} \text{{ {f_unit}}}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: BEARING PRESSURE
    with tabs[1]:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">2. Soil Bearing Pressure Verification</div>', unsafe_allow_html=True)
        
        st.write("**A. Base Moments (including lever arm of Depth D)**")
        st.latex(rf"M_{{x,base}} = M_{{x,app}} + |F_z \times D| = {L_DATA['Mx']} + |{L_DATA['Fz']} \times {D}| = {Mx_base:.2f} \text{{ {f_unit}-{l_unit}}}")
        st.latex(rf"M_{{z,base}} = M_{{z,app}} + |F_x \times D| = {L_DATA['Mz']} + |{L_DATA['Fx']} \times {D}| = {Mz_base:.2f} \text{{ {f_unit}-{l_unit}}}")
        
        st.write("**B. Maximum Pressure Calculation**")
        q_max = (P_total/Area_ftg) + abs(Mx_base/Sx) + abs(Mz_base/Sz)
        st.latex(rf"q_{{max}} = \frac{{P_{{total}}}}{{A}} + \frac{{|M_{{x,base}}|}}{{S_x}} + \frac{{|M_{{z,base}}|}}{{S_z}}")
        st.latex(rf"q_{{max}} = \frac{{{P_total:.2f}}}{{{Area_ftg:.2f}}} + \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} + \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \text{{ {p_unit}}}")
        
        ratio = q_max / qa
        st.markdown(f"**Verification:** {q_max:.3f} / {qa} = **Ratio: {ratio:.2f}**")
        st.markdown(f"<span class='status-{'pass' if ratio <= 1.0 else 'fail'}'>{'PASS' if ratio <= 1.0 else 'FAIL'}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: STABILITY
    with tabs[2]:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">3. Stability & Safety Factors</div>', unsafe_allow_html=True)
        
        # Sliding
        st.write("**A. Sliding Stability**")
        F_res = abs(P_total) * mu
        F_act = math.sqrt(L_DATA['Fx']**2 + L_DATA['Fz']**2)
        sf_sl = F_res / F_act if F_act > 0 else 99
        st.latex(rf"SF_{{sliding}} = \frac{{P_{{total}} \times \mu}}{{\sqrt{{F_x^2 + F_z^2}}}} = \frac{{{abs(P_total):.2f} \times {mu}}}{{{F_act:.2f}}} = {sf_sl:.2f}")
        st.markdown(f"**Status:** {'PASS' if sf_sl >= sf_sliding_min else 'FAIL'} (Min: {sf_sliding_min})")
        
        # Overturning
        st.write("**B. Overturning Stability**")
        sf_ot_x = (P_total * (Lz/2)) / abs(Mx_base) if Mx_base != 0 else 99
        sf_ot_z = (P_total * (Lx/2)) / abs(Mz_base) if Mz_base != 0 else 99
        st.latex(rf"SF_{{ot,min}} = \min(SF_x, SF_z) = \min({sf_ot_x:.2f}, {sf_ot_z:.2f}) = {min(sf_ot_x, sf_ot_z):.2f}")
        st.markdown(f"**Status:** {'PASS' if min(sf_ot_x, sf_ot_z) >= sf_ot_min else 'FAIL'} (Min: {sf_ot_min})")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 4: CONTOUR
    with tabs[3]:
        st.markdown('<div class="sec-header">4. Soil Pressure Distribution</div>', unsafe_allow_html=True)
        x_lin, z_lin = np.linspace(-Lx/2, Lx/2, 50), np.linspace(-Lz/2, Lz/2, 50)
        X, Z = np.meshgrid(x_lin, z_lin)
        Q_field = (P_total/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        
        fig_q = go.Figure(data=go.Heatmap(z=Q_field, x=x_lin, y=z_lin, colorscale='RdYlGn_r', zmin=0))
        # Pedestal Indicator
        fig_q.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0.1)")
        # Corners
        cx_c, cz_c = [Lx/2, -Lx/2, Lx/2, -Lx/2], [Lz/2, Lz/2, -Lz/2, -Lz/2]
        cq_c = (P_total/Area_ftg) + (Mx_base * np.array(cz_c) / Ix) + (Mz_base * np.array(cx_c) / Iz)
        fig_q.add_trace(go.Scatter(x=cx_c, y=cz_c, mode='text+markers', text=[f"{v:.2f}" for v in cq_c], textfont=dict(size=14, color="black")))
        fig_q.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), width=700, height=800, title="Pressure Contour (Actual Proportions)")
        st.plotly_chart(fig_q)

    # TAB 5: 3D VIEW
    with tabs[4]:
        st.markdown('<div class="sec-header">5. 3D Applied Force Visualization</div>', unsafe_allow_html=True)
        fig_3d = go.Figure()
        # Footing
        fig_3d.add_trace(go.Mesh3d(x=[-Lx/2,Lx/2,Lx/2,-Lx/2,-Lx/2,Lx/2,Lx/2,-Lx/2], y=[-Lz/2,-Lz/2,Lz/2,Lz/2,-Lz/2,-Lz/2,Lz/2,Lz/2], z=[-T,-T,-T,-T,0,0,0,0], color='royalblue', opacity=0.3))
        # Pedestal
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.8))
        # Arrows (Simplified High-Vis Arrows)
        sc = 1.5
        fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[Hp, Hp-sc], mode='lines+text', line=dict(color='green', width=10), text=["", f"FY={L_DATA['Fy']}"], name='Vertical'))
        fig_3d.add_trace(go.Scatter3d(x=[0, sc], y=[0,0], z=[Hp, Hp], mode='lines+text', line=dict(color='red', width=7), text=["", f"FX={L_DATA['Fx']}"], name='X-Force'))
        fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, sc], z=[Hp, Hp], mode='lines+text', line=dict(color='blue', width=7), text=["", f"FZ={L_DATA['Fz']}"], name='Z-Force'))
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- FINAL EXECUTIVE SUMMARY ---
    st.divider()
    st.subheader("🏁 Executive Summary")
    final_summary = pd.DataFrame([
        {"Check": "Soil Bearing", "Actual": f"{q_max:.3f}", "Limit": f"{qa}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
        {"Check": "Sliding SF", "Actual": f"{sf_sl:.2f}", "Limit": f"{sf_sliding_min}", "Ratio": round(sf_sliding_min/sf_sl, 2), "Status": "PASS" if sf_sl >= sf_sliding_min else "FAIL"},
        {"Check": "Overturning SF", "Actual": f"{min(sf_ot_x, sf_ot_z):.2f}", "Limit": f"{sf_ot_min}", "Ratio": round(sf_ot_min/min(sf_ot_x, sf_ot_z), 2), "Status": "PASS" if min(sf_ot_x, sf_ot_z) >= sf_ot_min else "FAIL"}
    ])
    st.table(final_summary)

else:
    st.warning("Please paste a valid load case row above to begin calculations.")
