import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="RC Section Designer", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Professional RC Section P-M Designer")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📐 Section Geometry")
    b = st.number_input("Width (b) [mm]", value=300, step=10)
    h = st.number_input("Depth (h) [mm]", value=500, step=10)
    fc = st.number_input("f'c [MPa]", value=30)
    fy = st.number_input("fy [MPa]", value=420)
    cover = st.number_input("Clear Cover [mm]", value=40)

    st.header("🔩 Reinforcement Assignment")
    # Corner Bars
    c_dia = st.number_input("Corner Bar Dia [mm]", value=20)
    
    # Face Assignments
    col_t, col_b = st.columns(2)
    with col_t:
        n_top = st.number_input("Top Face No.", value=4, min_value=2)
        d_top = st.number_input("Top Dia [mm]", value=20)
    with col_b:
        n_bot = st.number_input("Bot Face No.", value=4, min_value=2)
        d_bot = st.number_input("Bot Dia [mm]", value=20)
        
    col_l, col_r = st.columns(2)
    with col_l:
        n_left = st.number_input("Left Face No.", value=4, min_value=2)
        d_left = st.number_input("Left Dia [mm]", value=20)
    with col_r:
        n_right = st.number_input("Right Face No.", value=4, min_value=2)
        d_right = st.number_input("Right Dia [mm]", value=20)

# --- Reinforcement Coordinate Logic ---
def get_rebar_coords():
    bars = [] 
    corner_area = np.pi * (c_dia**2) / 4
    coords = [(cover, cover), (b-cover, cover), (cover, h-cover), (b-cover, h-cover)]
    for x, y in coords:
        bars.append({'x': x, 'y': y, 'area': corner_area, 'dia': c_dia, 'type': 'Corner'})
    
    if n_top > 2:
        x_top = np.linspace(cover, b-cover, n_top)[1:-1]
        area = np.pi * (d_top**2) / 4
        for x in x_top:
            bars.append({'x': x, 'y': h-cover, 'area': area, 'dia': d_top, 'type': 'Top'})
    if n_bot > 2:
        x_bot = np.linspace(cover, b-cover, n_bot)[1:-1]
        area = np.pi * (d_bot**2) / 4
        for x in x_bot:
            bars.append({'x': x, 'y': cover, 'area': area, 'dia': d_bot, 'type': 'Bottom'})
    if n_left > 2:
        y_left = np.linspace(cover, h-cover, n_left)[1:-1]
        area = np.pi * (d_left**2) / 4
        for y in y_left:
            bars.append({'x': cover, 'y': y, 'area': area, 'dia': d_left, 'type': 'Left'})
    if n_right > 2:
        y_right = np.linspace(cover, h-cover, n_right)[1:-1]
        area = np.pi * (d_right**2) / 4
        for y in y_right:
            bars.append({'x': b-cover, 'y': y, 'area': area, 'dia': d_right, 'type': 'Right'})
    return pd.DataFrame(bars)

rebar_df = get_rebar_coords()
total_as = rebar_df['area'].sum()

# --- Calculation Engine ---
Es = 200000
ecu = 0.003
beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85

def calculate_pm():
    results = []
    c_vals = np.concatenate([np.linspace(h*3, h, 30), np.linspace(h, 1, 150)])
    for c in c_vals:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        Pn = Cc / 1000
        Mn = (Cc * (h/2 - a/2)) / 1e6
        for _, bar in rebar_df.iterrows():
            eps_s = ecu * (c - (h - bar['y'])) / c 
            fs = np.clip(eps_s * Es, -fy, fy)
            force = (fs * bar['area']) / 1000
            Pn += force
            Mn += (force * (bar['y'] - h/2)) / 1e6
        results.append({'c': c, 'Pn': Pn, 'Mn': abs(Mn)})
    results.append({'c': 0, 'Pn': -(total_as * fy) / 1000, 'Mn': 0})
    return pd.DataFrame(results).sort_values('Pn', ascending=False)

pm_df = calculate_pm()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Interaction Diagram & Calcs", "🖼️ Section Detail"])

with tab1:
    # 1. The Plot
    fig_pm = go.Figure()
    fig_pm.add_trace(go.Scatter(x=pm_df['Mn'], y=pm_df['Pn'], fill='toself', 
                                fillcolor='rgba(31,119,180,0.1)', 
                                line=dict(color='#1f77b4', width=3), 
                                name='Nominal Capacity'))
    fig_pm.update_layout(xaxis_title="Moment Mn (kNm)", yaxis_title="Axial Pn (kN)", 
                         template="none", height=600)
    st.plotly_chart(fig_pm, use_container_width=True)

    # 2. The Equations Section
    st.divider()
    st.header("📐 Governing Design Equations")
    st.info("The following equations are solved iteratively for varying Neutral Axis depths (c).")

    col_eq1, col_eq2 = st.columns(2)
    
    with col_eq1:
        st.subheader("Force Equilibrium")
        st.write("Concrete Compression Force ($C_c$):")
        st.latex(r"C_c = 0.85 \cdot f'_c \cdot a \cdot b \quad \text{where } a = \beta_1 c")
        
        st.write("Steel Stress ($f_{si}$) for each layer $i$:")
        st.latex(r"\epsilon_{si} = 0.003 \frac{c - d_i}{c}")
        st.latex(r"f_{si} = \text{sign}(\epsilon_{si}) \cdot \min(|E_s \epsilon_{si}|, f_y)")
        
        st.write("Total Axial Nominal Strength ($P_n$):")
        st.latex(r"P_n = C_c + \sum_{i=1}^{n} (A_{si} \cdot f_{si})")

    with col_eq2:
        st.subheader("Moment Equilibrium")
        st.write("Moment about Plastic Centroid ($M_n$):")
        st.latex(r"M_n = C_c \left( \frac{h}{2} - \frac{a}{2} \right) + \sum_{i=1}^{n} \left[ A_{si} f_{si} \left( \frac{h}{2} - d_i \right) \right]")
        
        st.divider()
        st.subheader("Limits")
        st.write("Pure Tension Point ($T_n$):")
        st.latex(r"P_{nt} = - \sum (A_{si} \cdot f_y), \quad M_{nt} = 0")
with tab2:
    fig_sec = go.Figure()
    fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="rgba(200,200,200,0.3)")
    for b_type in rebar_df['type'].unique():
        subset = rebar_df[rebar_df['type'] == b_type]
        fig_sec.add_trace(go.Scatter(x=subset['x'], y=subset['y'], mode='markers', 
                                    marker=dict(size=subset['dia'], line=dict(width=1, color='black')), name=b_type))
    fig_sec.update_layout(xaxis=dict(title="Width (mm)"), yaxis=dict(title="Depth (mm)", scaleanchor="x", scaleratio=1), height=700)
    st.plotly_chart(fig_sec)
