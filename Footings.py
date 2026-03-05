import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Foundation Calculation Sheet")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .report-box { 
        background-color: #f8f9fa; 
        border: 1px solid #dee2e6; 
        padding: 25px; 
        border-radius: 5px; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .math-header { 
        color: #1a365d; 
        border-bottom: 2px solid #1a365d; 
        margin-top: 20px; 
        padding-bottom: 5px;
        font-weight: bold;
    }
    .check-pass { color: #28a745; font-weight: bold; }
    .check-fail { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Foundation Design Calculation Sheet")
st.caption("Detailed Step-by-Step Structural Verification")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Geometry & Materials")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])
    
    if unit_sys == "Imperial (kip, ft)":
        L = st.number_input("Footing Length (L)", value=12.0)
        W = st.number_input("Footing Width (W)", value=12.0)
        H = st.number_input("Footing Thickness (T)", value=2.5)
        Df = st.number_input("Soil Depth on Top", value=2.0)
        gamma_c = st.number_input("Concrete Density (pcf)", value=150.0) / 1000
        gamma_s = st.number_input("Soil Density (pcf)", value=110.0) / 1000
        qa = st.number_input("Allowable Bearing (ksf)", value=3.0)
        mu = st.number_input("Friction Coefficient (μ)", value=0.45)
        f_unit, l_unit, p_unit = "kip", "ft", "ksf"
    else:
        L = st.number_input("Footing Length (L)", value=4.0)
        W = st.number_input("Footing Width (W)", value=4.0)
        H = st.number_input("Footing Thickness (T)", value=0.8)
        Df = st.number_input("Soil Depth on Top", value=0.6)
        gamma_c = st.number_input("Concrete Density (kN/m³)", value=24.0)
        gamma_s = st.number_input("Soil Density (kN/m³)", value=18.0)
        qa = st.number_input("Allowable Bearing (kPa)", value=150.0)
        mu = st.number_input("Friction Coefficient (μ)", value=0.45)
        f_unit, l_unit, p_unit = "kN", "m", "kPa"

# --- Load Inputs ---
st.markdown('<div class="math-header">1. Controlling Applied Loads (at Pedestal Top)</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**ASD Case (Sizing/Stability)**")
    asd_fy = st.number_input("ASD Vertical (Fy)", value=0.88, key="afy")
    asd_fx = st.number_input("ASD Lateral (Fx)", value=2.91, key="afx")
    asd_fz = st.number_input("ASD Lateral (Fz)", value=0.62, key="afz")
    asd_mx = st.number_input("ASD Moment (Mx)", value=5.33, key="amx")
    asd_mz = st.number_input("ASD Moment (Mz)", value=37.75, key="amz")

with c2:
    st.markdown("**LRFD Case (Concrete Strength)**")
    ult_fy = st.number_input("LRFD Vertical (Fy)", value=1.23, key="ufy")
    ult_fx = st.number_input("LRFD Lateral (Fx)", value=3.49, key="ufx")
    ult_fz = st.number_input("LRFD Lateral (Fz)", value=0.75, key="ufz")
    ult_mx = st.number_input("LRFD Moment (Mx)", value=6.39, key="umx")
    ult_mz = st.number_input("LRFD Moment (Mz)", value=45.30, key="umz")

# --- Calculation Logic ---
# Properties
Area = L * W
Sx = (L * W**2) / 6
Sz = (W * L**2) / 6
Ix = (L * W**3) / 12
Iz = (W * L**3) / 12

# Weights
Wt_c = Area * H * gamma_c
Wt_s = Area * Df * gamma_s
Wt_total = Wt_c + Wt_s

# ASD Calculations
P_tot_asd = asd_fy + Wt_total
Mx_base_asd = asd_mx + abs(asd_fz * H)
Mz_base_asd = asd_mz + abs(asd_fx * H)

# Bearing
q_calc = []
for i in [1, -1]:
    for j in [1, -1]:
        q_calc.append((P_tot_asd/Area) + (i*Mx_base_asd/Sx) + (j*Mz_base_asd/Sz))
q_max, q_min = max(q_calc), min(q_calc)

# Stability
resisting_force = abs(P_tot_asd) * mu
acting_force = math.sqrt(asd_fx**2 + asd_fz**2)
fos_sliding = resisting_force / acting_force if acting_force > 0 else 99

res_mx = P_tot_asd * (W/2)
res_mz = P_tot_asd * (L/2)
fos_ot_x = res_mx / abs(Mx_base_asd) if Mx_base_asd != 0 else 99
fos_ot_z = res_mz / abs(Mz_base_asd) if Mz_base_asd != 0 else 99

# --- Report Tabs ---
tab1, tab2 = st.tabs(["📜 Detailed Calculation Sheet", "🌈 Stress Contour"])

with tab1:
    st.markdown('<div class="math-header">2. Foundation Self-Weight & Section Properties</div>', unsafe_allow_html=True)
    st.latex(f"W_{{concrete}} = L \\times W \\times T \\times \\gamma_c = {L} \\times {W} \\times {H} \\times {gamma_c:.3f} = {Wt_c:.2f} \\text{{ {f_unit}}}")
    st.latex(f"W_{{soil}} = L \\times W \\times D_f \\times \\gamma_s = {L} \\times {W} \\times {Df} \\times {gamma_s:.3f} = {Wt_s:.2f} \\text{{ {f_unit}}}")
    st.latex(f"W_{{total}} = {Wt_c:.2f} + {Wt_s:.2f} = {Wt_total:.2f} \\text{{ {f_unit}}}")
    
    st.markdown('<div class="math-header">3. Stability & Bearing Check (ASD)</div>', unsafe_allow_html=True)
    st.write("**Total Vertical Load at Base:**")
    st.latex(f"P_{{total}} = P_{{applied}} + W_{{total}} = {asd_fy} + {Wt_total:.2f} = {P_tot_asd:.2f} \\text{{ {f_unit}}}")
    
    st.write("**Overturning Moments at Base:**")
    st.latex(f"M_{{x,base}} = M_x + (F_z \\times T) = {asd_mx} + ({asd_fz} \\times {H}) = {Mx_base_asd:.2f} \\text{{ {f_unit}-{l_unit}}}")
    st.latex(f"M_{{z,base}} = M_z + (F_x \\times T) = {asd_mz} + ({asd_fx} \\times {H}) = {Mz_base_asd:.2f} \\text{{ {f_unit}-{l_unit}}}")
    
    st.write("**Maximum Bearing Pressure:**")
    st.latex(f"q_{{max}} = \\frac{{P_{{total}}}}{{A}} + \\frac{{M_{{x,base}}}}{{S_x}} + \\frac{{M_{{z,base}}}}{{S_z}}")
    st.latex(f"q_{{max}} = \\frac{{{P_tot_asd:.2f}}}{{{Area:.2f}}} + \\frac{{{abs(Mx_base_asd):.2f}}}{{{Sx:.2f}}} + \\frac{{{abs(Mz_base_asd):.2f}}}{{{Sz:.2f}}} = {q_max:.3f} \\text{{ {p_unit}}}")
    
    # SF Summary Table
    st.write("**Summary of Safety Factors:**")
    summary_data = {
        "Check": ["Bearing Pressure", "Sliding Stability", "Overturning (X)", "Overturning (Z)"],
        "Actual / FOS": [f"{q_max:.3f} {p_unit}", f"{fos_sliding:.2f}", f"{fos_ot_x:.2f}", f"{fos_ot_z:.2f}"],
        "Limit": [f"Allowable: {qa} {p_unit}", "Min FOS: 1.50", "Min FOS: 1.50", "Min FOS: 1.50"],
        "Ratio": [f"{q_max/qa:.2f}", f"{1.5/fos_sliding:.2f}", f"{1.5/fos_ot_x:.2f}", f"{1.5/fos_ot_z:.2f}"],
        "Status": ["PASS" if q_max <= qa else "FAIL", "PASS" if fos_sliding >= 1.5 else "FAIL", "PASS" if fos_ot_x >= 1.5 else "FAIL", "PASS" if fos_ot_z >= 1.5 else "FAIL"]
    }
    st.table(pd.DataFrame(summary_data))

with tab2:
    st.subheader("Soil Contact Stress Contour")
    
    # 50x50 Grid for heatmap
    res = 50
    x_grid = np.linspace(-L/2, L/2, res)
    z_grid = np.linspace(-W/2, W/2, res)
    X, Z = np.meshgrid(x_grid, z_grid)
    
    # Stress distribution: q = P/A + (Mx*z)/Ix + (Mz*x)/Iz
    Q = (P_tot_asd/Area) + (Mx_base_asd * Z / Ix) + (Mz_base_asd * X / Iz)
    
    fig = go.Figure(data=go.Heatmap(
        z=Q, x=x_grid, y=z_grid,
        colorscale='RdYlGn_r',
        colorbar=dict(title=f"q ({p_unit})"),
        zmin=0, zmax=qa * 1.1
    ))
    
    # Corner Value Annotations
    cx = [L/2, -L/2, L/2, -L/2]
    cz = [W/2, W/2, -W/2, -W/2]
    cq = (P_tot_asd/Area) + (Mx_base_asd * np.array(cz) / Ix) + (Mz_base_asd * np.array(cx) / Iz)
    
    fig.add_trace(go.Scatter(
        x=cx, y=cz, mode='text+markers',
        text=[f"{v:.2f}" for v in cq],
        textposition="top center",
        marker=dict(color='black', size=12, symbol='square'),
        name="Corner Pressures"
    ))

    fig.update_layout(
        xaxis_title="Length (X)", yaxis_title="Width (Z)",
        width=800, height=700, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    if q_min < 0:
        st.error(f"⚠️ Loss of Contact: Minimum pressure is {q_min:.3f}. Part of footing is in tension.")
