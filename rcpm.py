import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="RC P-M Diagram Generator", layout="wide")

st.title("🏗️ RC Section P-M Interaction Diagram")
st.sidebar.header("Section Parameters")

# Sidebar Inputs
b = st.sidebar.number_input("Width (b) [mm]", value=300)
h = st.sidebar.number_input("Depth (h) [mm]", value=500)
fc = st.sidebar.number_input("f'c [MPa]", value=30)
fy = st.sidebar.number_input("fy [MPa]", value=400)
cover = st.sidebar.number_input("Clear Cover [mm]", value=40)
bar_dia = st.sidebar.selectbox("Bar Diameter [mm]", [12, 16, 20, 25, 32], index=2)
n_bars = st.sidebar.number_input("Bars per face (Top/Bot)", value=3)

# Constants
Es = 200000 # MPa
ecu = 0.003
beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
As = n_bars * (np.pi * (bar_dia**2) / 4)
d = h - cover - bar_dia/2
d_prime = cover + bar_dia/2

def calculate_pm():
    P_list = []
    M_list = []
    
    # Range of Neutral Axis depth 'c' from very small to larger than h
    c_values = np.linspace(0.05 * h, 1.5 * h, 100)
    
    for c in c_values:
        a = beta1 * c
        if a > h: a = h
        
        # Concrete Force (Whitney Block)
        Cc = 0.85 * fc * a * b
        
        # Steel Strains
        eps_s = ecu * (d - c) / c
        eps_s_prime = ecu * (c - d_prime) / c
        
        # Steel Stresses
        fs = np.clip(eps_s * Es, -fy, fy)
        fs_prime = np.clip(eps_s_prime * Es, -fy, fy)
        
        # Forces
        Ts = fs * As
        Cs = fs_prime * As
        
        # Nominal Axial Strength (Pn)
        Pn = (Cc + Cs - Ts) / 1000 # kN
        
        # Nominal Moment (Mn) about Centroid
        Mn = (Cc * (h/2 - a/2) + Cs * (h/2 - d_prime) + Ts * (d - h/2)) / 1e6 # kNm
        
        P_list.append(Pn)
        M_list.append(Mn)

    # Add Pure Tension Point
    P_list.append(-2 * As * fy / 1000)
    M_list.append(0)
    
    return M_list, P_list

M_vals, P_vals = calculate_pm()

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=M_vals, y=P_vals, mode='lines+markers', name='Nominal Capacity'))
fig.update_layout(
    title="P-M Interaction Diagram (Nominal)",
    xaxis_title="Moment Mn (kNm)",
    yaxis_title="Axial Load Pn (kN)",
    template="plotly_white"
)

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Max axial (Pure Comp)", f"{round(max(P_vals))} kN")
    st.metric("Max Moment", f"{round(max(M_vals))} kNm")
    st.write(f"**Beta 1:** {round(beta1, 2)}")

with col2:
    st.plotly_chart(fig, use_container_width=True)
