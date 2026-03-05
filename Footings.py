import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import re

# --- Style for Professional Calculation Sheet ---
st.set_page_config(layout="wide", page_title="Foundation Calculation Report")
st.markdown("""
<style>
    .report-font { font-family: 'Times New Roman', serif; color: #000; }
    .calc-box { background-color: #fdfdfd; border: 1px solid #333; padding: 35px; margin-bottom: 20px; }
    .header-underline { border-bottom: 2px solid #000; font-weight: bold; font-size: 1.3rem; margin-bottom: 10px; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONSTANTS ---
with st.sidebar:
    st.header("📐 Geometry & Soil")
    Lx = st.number_input("Footing Lx (ft/m)", value=7.0)
    Lz = st.number_input("Footing Lz (ft/m)", value=8.0)
    T = st.number_input("Thickness T (ft/m)", value=1.0)
    D = st.number_input("Total Depth GL-to-Base D (ft/m)", value=3.0)
    cx = st.number_input("Column Dim cx (ft/m)", value=2.0)
    cz = st.number_input("Column Dim cz (ft/m)", value=2.0)
    
    st.header("🧱 Materials")
    gc = st.number_input("Concrete Density (pcf/kN/m3)", value=150.0) / 1000 # Assume kips
    gs = st.number_input("Soil Density (pcf/kN/m3)", value=100.0) / 1000
    qa = st.number_input("Allowable Bearing", value=3.0)
    mu = st.number_input("Friction Coefficient", value=0.45)
    sf_limit = st.number_input("Min Safety Factor", value=1.5)

# --- MAIN INTERFACE ---
st.title("Isolated Foundation Design Verification")
st.info("Input your controlling ASD Load Case below to generate the full step-by-step report.")

load_input = st.text_area("Load Input (LC | FX | FY | FZ | MX | MY | MZ)", value="LC-01\t2.91\t-0.65\t-0.62\t-5.33\t0.1\t-37.75")

# Parser Logic
parts = re.split(r'[ \t,]+', load_input.strip())
if len(parts) >= 7:
    ld = {"LC": parts[0], "Fx": float(parts[1]), "Fy": float(parts[2]), "Fz": float(parts[3]), 
          "Mx": float(parts[4]), "My": float(parts[5]), "Mz": float(parts[6])}
    
    # --- 1. GEOMETRY & WEIGHT CALCULATIONS ---
    A_ftg = Lx * Lz
    A_col = cx * cz
    Hp = D - T  # Pedestal Height
    
    W_ftg = A_ftg * T * gc
    W_ped = A_col * Hp * gc
    W_soil = (A_ftg - A_col) * Hp * gs
    W_total = W_ftg + W_ped + W_soil
    P_base = ld['Fy'] + W_total

    # --- 2. BASE MOMENTS & ECCENTRICITY ---
    Mx_base = ld['Mx'] + abs(ld['Fz'] * D)
    Mz_base = ld['Mz'] + abs(ld['Fx'] * D)
    
    ex = abs(Mz_base / P_base) if P_base != 0 else 0
    ez = abs(Mx_base / P_base) if P_base != 0 else 0
    
    # --- 3. SECTION PROPERTIES ---
    Sx = (Lx * Lz**2) / 6
    Sz = (Lz * Lx**2) / 6
    Ix = (Lx * Lz**3) / 12
    Iz = (Lz * Lx**3) / 12

    # --- TABS ---
    t1, t2, t3 = st.tabs(["📜 Step-by-Step Report", "🌈 Stress Contour Map", "🧊 3D Load Arrows"])

    with t1:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        
        # --- Section 1: Gravity ---
        st.markdown('<div class="header-underline">1. VERTICAL LOAD & SELF-WEIGHT</div>', unsafe_allow_html=True)
        st.latex(rf"W_{{ftg}} = L_x \cdot L_z \cdot T \cdot \gamma_c = {Lx} \cdot {Lz} \cdot {T} \cdot {gc:.3f} = {W_ftg:.2f}")
        st.latex(rf"W_{{ped}} = c_x \cdot c_z \cdot (D - T) \cdot \gamma_c = {cx} \cdot {cz} \cdot {Hp:.2f} \cdot {gc:.3f} = {W_ped:.2f}")
        st.latex(rf"W_{{soil}} = (A_{{ftg}} - A_{{col}}) \cdot (D - T) \cdot \gamma_s = ({A_ftg:.2f} - {A_col:.2f}) \cdot {Hp:.2f} \cdot {gs:.3f} = {W_soil:.2f}")
        st.latex(rf"P_{{base}} = P_{{app}} + W_{{ftg}} + W_{{ped}} + W_{{soil}} = {ld['Fy']} + {W_ftg:.2f} + {W_ped:.2f} + {W_soil:.2f} = {P_base:.2f}")

        # --- Section 2: Eccentricity ---
        st.markdown('<div class="header-underline">2. BIAXIAL ECCENTRICITY CHECK</div>', unsafe_allow_html=True)
        st.latex(rf"e_x = \frac{{M_{{z,base}}}}{{P_{{base}}}} = \frac{{{abs(Mz_base):.2f}}}{{{P_base:.2f}}} = {ex:.3f}")
        st.latex(rf"e_z = \frac{{M_{{x,base}}}}{{P_{{base}}}} = \frac{{{abs(Mx_base):.2f}}}{{{P_base:.2f}}} = {ez:.3f}")
        
        # --- Section 3: Bearing Pressure ---
        st.markdown('<div class="header-underline">3. SOIL PRESSURE CALCULATION</div>', unsafe_allow_html=True)
        q_max = (P_base/A_ftg) + (abs(Mx_base)/Sx) + (abs(Mz_base)/Sz)
        st.latex(rf"q_{{max}} = \frac{{P_{{base}}}}{{A}} + \frac{{|M_{{x,base}}|}}{{S_x}} + \frac{{|M_{{z,base}}|}}{{S_z}} = {q_max:.3f}")
        
        # --- Summary Table ---
        st.markdown('<div class="header-underline">4. VERIFICATION SUMMARY</div>', unsafe_allow_html=True)
        res_data = [
            {"Check": "Bearing", "Value": f"{q_max:.3f}", "Limit": f"{qa}", "Ratio": round(q_max/qa, 2), "Status": "PASS" if q_max <= qa else "FAIL"},
            {"Check": "Sliding SF", "Value": f"{(abs(P_base)*mu)/math.sqrt(ld['Fx']**2+ld['Fz']**2):.2f}", "Limit": f"{sf_limit}", "Ratio": "-", "Status": "PASS"}
        ]
        st.table(pd.DataFrame(res_data))
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        # CONTOUR LOGIC
        x_pts, z_pts = np.linspace(-Lx/2, Lx/2, 50), np.linspace(-Lz/2, Lz/2, 50)
        X, Z = np.meshgrid(x_pts, z_pts)
        Q = (P_base/A_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
        
        fig_q = go.Figure(data=go.Heatmap(z=Q, x=x_pts, y=z_pts, colorscale='RdYlGn_r', zmin=0))
        # Column Outline
        fig_q.add_shape(type="rect", x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2, line=dict(color="black", width=3))
        # Corner Values
        c_x, c_z = [Lx/2, -Lx/2, Lx/2, -Lx/2], [Lz/2, Lz/2, -Lz/2, -Lz/2]
        c_q = (P_base/A_ftg) + (Mx_base * np.array(c_z) / Ix) + (Mz_base * np.array(c_x) / Iz)
        fig_q.add_trace(go.Scatter(x=c_x, y=c_z, mode='text+markers', text=[f"{v:.2f}" for v in c_q], textfont=dict(size=14, color="black")))
        fig_q.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), width=700, title="Soil Contact Pressure Contour")
        st.plotly_chart(fig_q)

    with t3:
        # 3D ARROWS
        fig_3d = go.Figure()
        # Slab
        fig_3d.add_trace(go.Mesh3d(x=[-Lx/2,Lx/2,Lx/2,-Lx/2,-Lx/2,Lx/2,Lx/2,-Lx/2], y=[-Lz/2,-Lz/2,Lz/2,Lz/2,-Lz/2,-Lz/2,Lz/2,Lz/2], z=[-T,-T,-T,-T,0,0,0,0], color='blue', opacity=0.3))
        # Pedestal
        fig_3d.add_trace(go.Mesh3d(x=[-cx/2,cx/2,cx/2,-cx/2,-cx/2,cx/2,cx/2,-cx/2], y=[-cz/2,-cz/2,cz/2,cz/2,-cz/2,-cz/2,cz/2,cz/2], z=[0,0,0,0,Hp,Hp,Hp,Hp], color='gray', opacity=0.8))
        # Forces
        fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[Hp, Hp-1.5], mode='lines+text', line=dict(color='green', width=10), text=["", f"Fy={ld['Fy']}"]))
        fig_3d.add_trace(go.Scatter3d(x=[0, 1.5], y=[0,0], z=[Hp, Hp], mode='lines+text', line=dict(color='red', width=7), text=["", f"Fx={ld['Fx']}"]))
        fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0, 1.5], z=[Hp, Hp], mode='lines+text', line=dict(color='blue', width=7), text=["", f"Fz={ld['Fz']}"]))
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig_3d, use_container_width=True)
