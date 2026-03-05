import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Foundation Calculation Sheet")

st.markdown("""
<style>
    .reportview-container { background: #f5f5f5; }
    .main { background: #ffffff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
    .metric-box { background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; }
    .status-pass { color: #059669; font-weight: bold; }
    .status-fail { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Isolated Foundation Calculation Sheet")
st.caption("Based on ACI 318-19 & ASCE 7-22 Methodology")

# --- Sidebar: Geometry & Materials ---
st.sidebar.header("1. Geometry & Soil")
unit_sys = st.sidebar.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])

if unit_sys == "Imperial (kip, ft)":
    L, W = st.sidebar.number_input("Footing Length (L)", 8.0), st.sidebar.number_input("Footing Width (W)", 8.0)
    H = st.sidebar.number_input("Footing Thickness (H)", 2.0)
    Df = st.sidebar.number_input("Soil Depth on Top (Df)", 1.0)
    gamma_c, gamma_s = 150.0/1000, 110.0/1000  # to kcf
    qa = st.sidebar.number_input("Allowable Bearing (ksf)", 3.0)
    mu = st.sidebar.number_input("Friction Coefficient (μ)", 0.45)
    force_unit, len_unit, press_unit = "kip", "ft", "ksf"
else:
    L, W = st.sidebar.number_input("Footing Length (L)", 3.0), st.sidebar.number_input("Footing Width (W)", 3.0)
    H = st.sidebar.number_input("Footing Thickness (H)", 0.6)
    Df = st.sidebar.number_input("Soil Depth on Top (Df)", 0.6)
    gamma_c, gamma_s = 24.0, 18.0
    qa = st.sidebar.number_input("Allowable Bearing (kPa)", 150.0)
    mu = st.sidebar.number_input("Friction Coefficient (μ)", 0.45)
    force_unit, len_unit, press_unit = "kN", "m", "kPa"

# --- Main Area: Input Two Controlling Cases ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔹 Controlling ASD Case (Sizing)")
    asd_fy = st.number_input("ASD Vertical (Fy)", value=0.88, key="asd_fy", help="+ is Compression")
    asd_fx = st.number_input("ASD Lateral (Fx)", value=2.91, key="asd_fx")
    asd_fz = st.number_input("ASD Lateral (Fz)", value=0.62, key="asd_fz")
    asd_mx = st.number_input("ASD Moment (Mx)", value=5.33, key="asd_mx")
    asd_mz = st.number_input("ASD Moment (Mz)", value=37.75, key="asd_mz")

with col2:
    st.subheader("🔸 Controlling LRFD Case (Strength)")
    ult_fy = st.number_input("LRFD Vertical (Fy)", value=1.23, key="ult_fy")
    ult_fx = st.number_input("LRFD Lateral (Fx)", value=3.49, key="ult_fx")
    ult_fz = st.number_input("LRFD Lateral (Fz)", value=0.75, key="ult_fz")
    ult_mx = st.number_input("LRFD Moment (Mx)", value=6.39, key="ult_mx")
    ult_mz = st.number_input("LRFD Moment (Mz)", value=45.3, key="ult_mz")

# --- Calculations Engine ---
area = L * W
sx, sz = (L * W**2)/6, (W * L**2)/6
ix, iz = (L * W**3)/12, (W * L**3)/12

# Weights
w_conc = area * H * gamma_c
w_soil = area * Df * gamma_s
w_total = w_conc + w_soil

# Base Moments (ASD)
asd_mx_base = asd_mx + abs(asd_fz * H)
asd_mz_base = asd_mz + abs(asd_fx * H)
asd_p_total = asd_fy + w_total

# Bearing Pressure (ASD)
q_corners = []
for i in [1, -1]:
    for j in [1, -1]:
        q_corners.append((asd_p_total/area) + (i*asd_mx_base/sx) + (j*asd_mz_base/sz))
q_max, q_min = max(q_corners), min(q_corners)

# Stability (ASD)
sf_sliding = (abs(asd_p_total) * mu) / math.sqrt(asd_fx**2 + asd_fz**2) if (asd_fx**2 + asd_fz**2) > 0 else 99
sf_ot_x = (asd_p_total * W/2) / abs(asd_mx_base) if asd_mx_base != 0 else 99
sf_ot_z = (asd_p_total * L/2) / abs(asd_mz_base) if asd_mz_base != 0 else 99
sf_ot = min(sf_ot_x, sf_ot_z)

# --- RESULTS REPORT ---
st.header("📋 Foundation Calculation Report")

tab_res, tab_calc, tab_stress = st.tabs(["✅ Summary & SF Ratios", "📜 Detailed Math", "🌈 Stress Contour"])

with tab_res:
    st.subheader("Design Verification Summary")
    
    m1, m2, m3 = st.columns(3)
    
    # Bearing Check
    ratio_bearing = q_max / qa
    status_b = "PASS" if ratio_bearing <= 1.0 and q_min >= 0 else "FAIL"
    m1.metric("Bearing Utilization", f"{ratio_bearing:.2%}", delta=status_b, delta_color="inverse")
    
    # Sliding Check
    ratio_sl = 1.5 / sf_sliding
    status_sl = "PASS" if sf_sliding >= 1.5 else "FAIL"
    m2.metric("Sliding FOS", f"{sf_sliding:.2f}", delta=status_sl, delta_color="normal")
    
    # Overturning Check
    ratio_ot = 1.5 / sf_ot
    status_ot = "PASS" if sf_ot >= 1.5 else "FAIL"
    m3.metric("Overturning FOS", f"{sf_ot:.2f}", delta=status_ot, delta_color="normal")

    results_table = pd.DataFrame({
        "Check": ["Max Bearing Pressure", "Minimum Pressure (Uplift)", "Safety Factor Sliding", "Safety Factor Overturning"],
        "Actual Value": [f"{q_max:.3f} {press_unit}", f"{q_min:.3f} {press_unit}", f"{sf_sliding:.2f}", f"{sf_ot:.2f}"],
        "Limit / Allowable": [f"{qa} {press_unit}", f"≥ 0", "≥ 1.50", "≥ 1.50"],
        "Ratio (D/C)": [f"{ratio_bearing:.2f}", "-" if q_min >= 0 else "UPLIFT", f"{ratio_sl:.2f}", f"{ratio_ot:.2f}"],
        "Status": [status_b, "PASS" if q_min >=0 else "FAIL", status_sl, status_ot]
    })
    st.table(results_table)

with tab_calc:
    st.subheader("Step-by-Step Mathematical Report")
    
    st.markdown("### 1. Foundation Self-Weight")
    st.latex(f"W_{{concrete}} = L \\times W \\times H \\times \\gamma_c = {L} \\times {W} \\times {H} \\times {gamma_c:.3f} = {w_conc:.2f} \text{{ {force_unit}}}")
    st.latex(f"W_{{soil}} = L \\times W \\times D_f \\times \\gamma_s = {L} \\times {W} \\times {Df} \\times {gamma_s:.3f} = {w_soil:.2f} \text{{ {force_unit}}}")
    st.latex(f"W_{{total}} = {w_conc:.2f} + {w_soil:.2f} = {w_total:.2f} \text{{ {force_unit}}}")

    st.markdown("### 2. Properties of Section")
    st.latex(f"Area = {L} \\times {W} = {area:.2f} \text{{ {len_unit}}}^2")
    st.latex(f"S_{{xx}} = \\frac{{{L} \\times {W}^2}}{{6}} = {sx:.2f} \text{{ {len_unit}}}^3")
    st.latex(f"S_{{zz}} = \\frac{{{W} \\times {L}^2}}{{6}} = {sz:.2f} \text{{ {len_unit}}}^3")

    st.markdown("### 3. Stability Checks (ASD)")
    st.write("**Moments at the Footing Base:**")
    st.latex(f"M_{{x,base}} = M_{{applied}} + (F_z \\times H) = {asd_mx} + ({asd_fz} \\times {H}) = {asd_mx_base:.2f} \text{{ {force_unit}-{len_unit}}}")
    st.latex(f"M_{{z,base}} = M_{{applied}} + (F_x \\times H) = {asd_mz} + ({asd_fx} \\times {H}) = {asd_mz_base:.2f} \text{{ {force_unit}-{len_unit}}}")

    st.markdown("**Stability Factors of Safety:**")
    st.latex(f"FOS_{{sliding}} = \\frac{{P_{{total}} \\times \\mu}}{{\\sqrt{{F_x^2 + F_z^2}}}} = \\frac{{{asd_p_total:.2f} \\times {mu}}}{{\\sqrt{{{asd_fx}^2 + {asd_fz}^2}}}} = {sf_sliding:.2f}")
    st.latex(f"FOS_{{overturning}} = \\min \\left( \\frac{{P_{{total}} \\times W/2}}{{M_{{x,base}}}}, \\frac{{P_{{total}} \\times L/2}}{{M_{{z,base}}}} \\right) = {sf_ot:.2f}")

with tab_stress:
    st.subheader("Soil Bearing Pressure Heatmap")
    
    # Generate 50x50 grid for contour
    res = 50
    x_lin = np.linspace(-L/2, L/2, res)
    z_lin = np.linspace(-W/2, W/2, res)
    X, Z = np.meshgrid(x_lin, z_lin)
    
    # Formula: q = P/A + Mx*z/Ix + Mz*x/Iz
    Q = (asd_p_total/area) + (asd_mx_base * Z / ix) + (asd_mz_base * X / iz)
    
    fig = go.Figure(data=go.Heatmap(
        z=Q, x=x_lin, y=z_lin,
        colorscale='RdYlGn_r',
        colorbar=dict(title=f"q ({press_unit})"),
        zmin=0, zmax=qa * 1.1
    ))
    
    # Values at corners
    c_x = [L/2, -L/2, L/2, -L/2]
    c_z = [W/2, W/2, -W/2, -W/2]
    c_q = (asd_p_total/area) + (asd_mx_base * np.array(c_z) / ix) + (asd_mz_base * np.array(c_x) / iz)
    
    fig.add_trace(go.Scatter(
        x=c_x, y=c_z, mode='text+markers',
        text=[f"{v:.2f}" for v in c_q],
        textposition="top center",
        marker=dict(color='black', size=10),
        name="Corner Values"
    ))

    fig.update_layout(
        xaxis_title="Length (X)", yaxis_title="Width (Z)",
        width=800, height=700, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    if q_min < 0:
        st.error(f"⚠️ Warning: Negative pressure ({q_min:.3f}) detected. Loss of contact occurs.")

st.markdown("---")
st.caption("Simplified Calculation Sheet Tool | Design based on provided controlling cases.")
