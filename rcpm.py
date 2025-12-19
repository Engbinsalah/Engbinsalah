import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="RC Section Designer", layout="wide")

# --- UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_all_with_experimental_markdown=True)

st.title("🏗️ Professional RC Section P-M Designer")
st.write("Calculations based on **ACI 318-19** principles.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Section Geometry")
    b = st.number_input("Width (b) [mm]", value=300, step=10)
    h = st.number_input("Depth (h) [mm]", value=500, step=10)
    
    st.header("💪 Materials")
    fc = st.number_input("f'c [MPa]", value=30)
    fy = st.number_input("fy [MPa]", value=420)
    
    st.header("🔩 Reinforcement")
    cover = st.number_input("Clear Cover [mm]", value=40)
    bar_dia = st.selectbox("Bar Diameter [mm]", [12, 16, 20, 25, 32], index=2)
    n_bars = st.number_input("Bars per face (Top/Bot)", value=3, min_value=2)

# --- Constants & Geometry ---
Es = 200000 
ecu = 0.003
beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
As_face = n_bars * (np.pi * (bar_dia**2) / 4)
d = h - cover - bar_dia/2
d_prime = cover + bar_dia/2

def get_phi(eps_t):
    if eps_t <= 0.002: return 0.65
    if eps_t >= 0.005: return 0.90
    return 0.65 + (eps_t - 0.002) * (0.25 / 0.003)

# --- Calculations ---
def generate_data():
    results = []
    # Full range from pure compression to pure tension
    # c_values from very large (compression) to very small (tension)
    c_values = np.logspace(np.log10(0.01*h), np.log10(5*h), 200)[::-1]
    
    for c in c_values:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        eps_s_bot = ecu * (d - c) / c
        eps_s_top = ecu * (c - d_prime) / c
        fs_bot = np.clip(eps_s_bot * Es, -fy, fy)
        fs_top = np.clip(eps_s_top * Es, -fy, fy)
        
        # Pn = Cc + Cs - Ts
        Pn = (Cc + fs_top * As_face - fs_bot * As_face) / 1000
        # Mn about plastic centroid (h/2)
        Mn = (Cc*(h/2 - a/2) + fs_top*As_face*(h/2 - d_prime) + fs_bot*As_face*(d - h/2)) / 1e6
        
        phi = get_phi(eps_s_bot)
        results.append({"c": c, "Pn": Pn, "Mn": Mn, "phiPn": Pn*phi, "phiMn": Mn*phi, "eps_t": eps_s_bot})
    
    # Add Pure Tension Point (No concrete contribution)
    Pn_tension = - (2 * As_face * fy) / 1000
    results.append({"c": 0, "Pn": Pn_tension, "Mn": 0, "phiPn": Pn_tension * 0.9, "phiMn": 0, "eps_t": 1.0})
    
    return pd.DataFrame(results)

df = generate_data()

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Interaction Diagram", "📐 Detailed Calculations", "🖼️ Section View"])

with tab1:
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Max Design Compression (φPn_max)", f"{round(0.8 * 0.65 * df['Pn'].max())} kN")
    col_m2.metric("Pure Design Tension (φTn)", f"{round(df['phiPn'].min())} kN")

    fig_pm = go.Figure()
    # Nominal Curve
    fig_pm.add_trace(go.Scatter(x=df['Mn'], y=df['Pn'], name='Nominal (Pn, Mn)', line=dict(color='rgba(150,150,150,0.5)', dash='dash')))
    # Design Curve
    fig_pm.add_trace(go.Scatter(x=df['phiMn'], y=df['phiPn'], name='Design (φPn, φMn)', fill='tozeroy', fillcolor='rgba(0,100,255,0.1)', line=dict(color='#1f77b4', width=3)))
    
    fig_pm.update_layout(xaxis_title="Moment Mn, φMn (kNm)", yaxis_title="Axial Load Pn, φPn (kN)", height=600, template="plotly_white", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_pm, use_container_width=True)

with tab2:
    st.header("Step-by-Step Methodology")
    st.write("Example calculation for the **Balanced Point** (where steel begins to yield):")
    
    c_bal = (ecu / (ecu + (fy/Es))) * d
    a_bal = beta1 * c_bal
    
    st.latex(rf"c_{{bal}} = \frac{{\epsilon_{{cu}}}}{{\epsilon_{{cu}} + \epsilon_y}} \cdot d = {round(c_bal, 2)} \text{{ mm}}")
    st.latex(rf"a = \beta_1 \cdot c = {round(a_bal, 2)} \text{{ mm}}")
    st.latex(rf"C_c = 0.85 \cdot f'_c \cdot a \cdot b = {round(0.85*fc*a_bal*b/1000, 2)} \text{{ kN}}")
    
    st.subheader("Interactive Data Table")
    st.dataframe(df[['c', 'Pn', 'Mn', 'phiPn', 'phiMn', 'eps_t']].style.format("{:.2f}"))

with tab3:
    st.header("Cross-Section Visualization")
    
    fig_sec = go.Figure()
    # Concrete Outline
    fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="Black"), fillcolor="LightGrey")
    
    # Reinforcement
    x_pos = np.linspace(cover + bar_dia/2, b - cover - bar_dia/2, n_bars)
    # Bottom Bars
    fig_sec.add_trace(go.Scatter(x=x_pos, y=[cover + bar_dia/2]*n_bars, mode='markers', marker=dict(size=bar_dia, color='red'), name='Bottom Steel'))
    # Top Bars
    fig_sec.add_trace(go.Scatter(x=x_pos, y=[h - (cover + bar_dia/2)]*n_bars, mode='markers', marker=dict(size=bar_dia, color='blue'), name='Top Steel'))
    
    fig_sec.update_layout(xaxis=dict(range=[-50, b+50], title="Width (mm)"), yaxis=dict(range=[-50, h+50], title="Depth (mm)"), width=400, height=500, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_sec)
