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

st.title("🏗️ Enhanced RC Section P-M Designer")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📐 Section Geometry")
    b = st.number_input("Width (b) [mm]", value=300, step=10)
    h = st.number_input("Depth (h) [mm]", value=500, step=10)
    fc = st.number_input("f'c [MPa]", value=30)
    fy = st.number_input("fy [MPa]", value=420)
    cover = st.number_input("Clear Cover [mm]", value=40)

    st.header("🔩 Reinforcement Assignment")
    st.info("Corners are always active (4 bars). Adjust faces below.")
    
    # Corner Bars (Fixed per your request)
    c_dia = st.number_input("Corner Bar Dia [mm]", value=20)
    
    # Face Assignments
    col_t, col_b = st.columns(2)
    with col_t:
        n_top = st.number_input("Top Face No.", value=4)
        d_top = st.number_input("Top Dia", value=20)
    with col_b:
        n_bot = st.number_input("Bot Face No.", value=4)
        d_bot = st.number_input("Bot Dia", value=20)
        
    col_l, col_r = st.columns(2)
    with col_l:
        n_left = st.number_input("Left Face No.", value=4)
        d_left = st.number_input("Left Dia", value=20)
    with col_r:
        n_right = st.number_input("Right Face No.", value=4)
        d_right = st.number_input("Right Dia", value=20)

# --- Reinforcement Coordinate Logic ---
def get_rebar_coords():
    bars = [] # List of (x, y, area, dia, label)
    
    # 1. Corners
    corner_area = np.pi * (c_dia**2) / 4
    coords = [
        (cover, cover), (b-cover, cover), 
        (cover, h-cover), (b-cover, h-cover)
    ]
    for x, y in coords:
        bars.append({'x': x, 'y': y, 'area': corner_area, 'dia': c_dia, 'type': 'Corner'})
    
    # 2. Top/Bottom Faces (excluding corners)
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

    # 3. Left/Right Faces (excluding corners)
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

# --- Interaction Diagram Engine ---
def calculate_pm():
    results = []
    Es = 200000
    ecu = 0.003
    beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
    
    # Iterate Neutral Axis depth 'c'
    c_vals = np.concatenate([np.linspace(h*2, h, 20), np.linspace(h, 1, 150)])
    
    for c in c_vals:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        
        Pn = Cc / 1000
        Mn = (Cc * (h/2 - a/2)) / 1e6
        
        for _, bar in rebar_df.iterrows():
            eps_s = ecu * (c - (h - bar['y'])) / c # Strain relative to top fiber
            fs = np.clip(eps_s * Es, -fy, fy)
            force = (fs * bar['area']) / 1000
            Pn += force
            Mn += (force * (bar['y'] - h/2)) / 1e6
            
        results.append({'Pn': Pn, 'Mn': abs(Mn)})
        
    # Add pure tension point
    results.append({'Pn': -(total_as * fy) / 1000, 'Mn': 0})
    return pd.DataFrame(results).sort_values('Pn', ascending=False)

pm_df = calculate_pm()

# --- Visualization Tabs ---
tab1, tab2 = st.tabs(["📊 Interaction Diagram", "🖼️ Section Detail"])

with tab1:
    fig_pm = go.Figure()
    fig_pm.add_trace(go.Scatter(x=pm_df['Mn'], y=pm_df['Pn'], fill='toself', name='Nominal Capacity'))
    fig_pm.update_layout(xaxis_title="Moment (kNm)", yaxis_title="Axial (kN)", template="none")
    st.plotly_chart(fig_pm, use_container_width=True)

with tab2:
    fig_sec = go.Figure()
    # Draw Section
    fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="rgba(200,200,200,0.3)")
    # Draw Bars
    for b_type in rebar_df['type'].unique():
        subset = rebar_df[rebar_df['type'] == b_type]
        fig_sec.add_trace(go.Scatter(
            x=subset['x'], y=subset['y'], mode='markers',
            marker=dict(size=subset['dia'], line=dict(width=1, color='black')),
            name=b_type
        ))
    
    fig_sec.update_layout(
        xaxis=dict(range=[-50, b+50], title="Width (mm)"),
        yaxis=dict(range=[-50, h+50], title="Depth (mm)", scaleanchor="x", scaleratio=1),
        width=600, height=700, template="none"
    )
    st.plotly_chart(fig_sec)

st.metric("Total Steel Area (As)", f"{round(total_as, 2)} mm²")
