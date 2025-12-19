import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="RC Design P-M", layout="wide")

st.title("🏗️ Design P-M Interaction Diagram (ACI 318)")

# Sidebar for Inputs
with st.sidebar:
    st.header("Section Parameters")
    b = st.number_input("Width (b) [mm]", value=300)
    h = st.number_input("Depth (h) [mm]", value=500)
    fc = st.number_input("f'c [MPa]", value=30)
    fy = st.number_input("fy [MPa]", value=420)
    cover = st.number_input("Clear Cover [mm]", value=40)
    bar_dia = st.number_input("Bar Diameter [mm]", value=20)
    n_bars = st.number_input("Bars per face", value=3)

# Constants
Es = 200000 
ecu = 0.003
beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
As = n_bars * (np.pi * (bar_dia**2) / 4)
d = h - cover - bar_dia/2
d_prime = cover + bar_dia/2

def get_phi(eps_t):
    # ACI 318-19 Table 21.2.2 for Tied Columns
    if eps_t <= 0.002: return 0.65
    if eps_t >= 0.005: return 0.90
    return 0.65 + (eps_t - 0.002) * (0.25 / 0.003)

def calculate_pm():
    results = []
    c_values = np.linspace(0.05 * h, 1.2 * h, 200)
    
    for c in c_values:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        
        eps_s_bot = ecu * (d - c) / c
        eps_s_top = ecu * (c - d_prime) / c
        
        fs_bot = np.clip(eps_s_bot * Es, -fy, fy)
        fs_top = np.clip(eps_s_top * Es, -fy, fy)
        
        Pn = (Cc + fs_top * As - fs_bot * As) / 1000
        Mn = (Cc*(h/2 - a/2) + fs_top*As*(h/2 - d_prime) + fs_bot*As*(d - h/2)) / 1e6
        
        phi = get_phi(eps_s_bot)
        results.append({"Pn": Pn, "Mn": Mn, "phiPn": Pn * phi, "phiMn": Mn * phi})

    return pd.DataFrame(results)

df = calculate_pm()

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name='Nominal (Pn, Mn)', line=dict(color='gray', dash='dash')))
fig.add_trace(go.Scatter(x=df['phiMn'], y=df['phiPn'], name='Design (phiPn, phiMn)', fill='tozerox', line=dict(color='blue')))

# Add ACI Max Axial Limit (0.80 * phi * Pn_max)
p_max_nom = (0.85 * fc * (b * h - 2 * As) + fy * 2 * As) / 1000
p_max_design = 0.80 * 0.65 * p_max_nom
fig.add_hline(y=p_max_design, line_dash="dot", line_color="red", annotation_text="Max Allowable P_u")

fig.update_layout(xaxis_title="Moment (kNm)", yaxis_title="Axial Load (kN)", height=600)
st.plotly_chart(fig, use_container_width=True)
