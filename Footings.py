"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Complete Version: Sizing, Detailed Calcs, and Stress Contours
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="Foundation Sizing & Stress", page_icon="📐", layout="wide")

st.markdown("""
<style>
.stApp { background:#0f1117; color:#e0e0e0; }
.sec-hdr { 
    background: linear-gradient(90deg, #1565c0, #0d47a1); 
    color: #fff; 
    padding: 10px 16px; 
    border-radius: 6px; 
    font-weight: 700; 
    margin: 15px 0 10px; 
}
.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #16213e);
    border: 1px solid #2d3561;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #4fc3f7; }
.metric-label { font-size: 0.75rem; color: #90a4ae; text-transform: uppercase; }
</style>""", unsafe_allow_html=True)

# 2. HELPER FUNCTIONS (To prevent NameError)
def sec(t): 
    st.markdown(f'<div class="sec-hdr">{t}</div>', unsafe_allow_html=True)

def info(t): 
    st.info(t)

def mcard(lbl, val, col):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

def parse_staad_text(text, flip_axial):
    """Handles Tabs, Spaces, and Commas from STAAD/Excel."""
    try:
        # Standardize delimiters
        clean_text = re.sub(r'[ \t]+', ',', text.strip())
        df = pd.read_csv(io.StringIO(clean_text))
        
        # Column Name Mapping
        mapping = {
            'LC': 'Case', 'L/C': 'Case', 'LOAD': 'Case', 'CASE': 'Case',
            'FX': 'Fx', 'FY': 'Fy', 'FZ': 'Fz',
            'MX': 'Mx', 'MY': 'My', 'MZ': 'Mz'
        }
        df.columns = [str(c).upper().strip() for c in df.columns]
        df.rename(columns=mapping, inplace=True)
        
        # Fallback if 'Case' is missing
        if 'Case' not in df.columns:
            df.rename(columns={df.columns[0]: 'Case'}, inplace=True)
            
        # Numeric conversion
        for c in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
        if flip_axial:
            df['Fy'] = -df['Fy']
            
        return df
    except Exception as e:
        st.error(f"Data Parser Error: {e}")
        return None

# 3. MAIN APPLICATION
def main():
    st.title("🏗️ Foundation Sizing & Stress Analysis")
    
    # SIDEBAR CONTROLS
    with st.sidebar:
        sec("1. Configuration")
        unit_sys = st.selectbox("Units", ["Imperial (kip, ft)", "Metric (kN, m)"])
        flip_ax = st.checkbox("Flip Axial Sign (STAAD Reaction)", value=True)
        asd_tag = st.text_input("Sizing Case Prefix", value="[4-")
        
        sec("2. Design Parameters")
        qa = st.number_input("Allowable Bearing (ksf/kPa)", value=3.0 if "Imp" in unit_sys else 150.0)
        mu = st.number_input("Friction Coefficient (μ)", value=0.45)
        
        sec("3. Densities")
        gc = st.number_input("Concrete (pcf/kN/m³)", value=150.0 if "Imp" in unit_sys else 24.0)
        gs = st.number_input("Soil (pcf/kN/m³)", value=110.0 if "Imp" in unit_sys else 18.0)
        
        sec("4. Geometry")
        Lx = st.number_input("Length Lx", value=12.0 if "Imp" in unit_sys else 4.0)
        Lz = st.number_input("Width Lz", value=12.0 if "Imp" in unit_sys else 4.0)
        H = st.number_input("Thickness H", value=2.0 if "Imp" in unit_sys else 0.6)
        Df = st.number_input("Soil Depth Df", value=2.0 if "Imp" in unit_sys else 0.6)

    t1, t2, t3, t4 = st.tabs(["📂 Load Input", "📐 Sizing Results", "📜 Calc Report", "🌈 Stress Contour"])

    with t1:
        sec("① Load Case Entry")
        default_data = "LC	FX	FY	FZ	MX	MY	MZ\n[4-1.2]	2.91	-0.65	-0.62	-5.33	0.1	-37.75"
        raw = st.text_area("Paste STAAD/Excel Rows Here", value=default_data, height=150)
        if raw:
            df = parse_staad_text(raw, flip_ax)
            if df is not None:
                st.session_state['ldf'] = df
                st.dataframe(df, use_container_width=True)

    if 'ldf' in st.session_state:
        ldf = st.session_state['ldf']
        
        # Calculation Constants
        Area = Lx * Lz
        Sx, Sz = (Lx * Lz**2)/6, (Lz * Lx**2)/6
        Ix, Iz = (Lx * Lz**3)/12, (Lz * Lx**3)/12
        
        W_conc = Area * H * (gc/1000 if "Imp" in unit_sys else gc)
        W_soil = Area * Df * (gs/1000 if "Imp" in unit_sys else gs)
        W_total = W_conc + W_soil

        with t2:
            sec("② Bearing & Sizing Summary")
            asd_df = ldf[ldf['Case'].str.contains(asd_tag, na=False, regex=False)]
            if asd_df.empty: asd_df = ldf # Fallback if no tag matches
            
            res_rows = []
            for _, r in asd_df.iterrows():
                P_tot = r['Fy'] + W_total
                Mx_b = r['Mx'] + r['Fz']*H
                Mz_b = r['Mz'] + r['Fx']*H
                q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
                res_rows.append({"Case": r['Case'], "P_Total": round(P_tot,2), "q_max": round(q_max,3), "Status": "OK" if q_max <= qa else "FAIL"})
            st.table(pd.DataFrame(res_rows))

        with t3:
            sec("③ Step-by-Step Mathematical Report")
            sel = st.selectbox("Select Case", ldf['Case'])
            r = ldf[ldf['Case'] == sel].iloc[0]
            
            Mx_b, Mz_b = r['Mx'] + r['Fz']*H, r['Mz'] + r['Fx']*H
            P_tot = r['Fy'] + W_total
            q_val = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)

            st.markdown("#### Weight Calculations")
            st.latex(f"W_{{total}} = (Area \\times H \\times \\gamma_c) + (Area \\times D_f \\times \\gamma_s) = {W_total:.2f}")
            st.markdown("#### Bearing Pressure Calculation")
            st.latex(f"q_{{max}} = \\frac{{{P_tot:.2f}}}{{{Area:.2f}}} + \\frac{{{abs(Mx_b):.2f}}}{{{Sx:.2f}}} + \\frac{{{abs(Mz_b):.2f}}}{{{Sz:.2f}}} = {q_val:.3f}")

        with t4:
            sec("④ Bearing Pressure Heatmap")
            sel_p = st.selectbox("Select Case for Contour", ldf['Case'], key="p_case")
            rp = ldf[ldf['Case'] == sel_p].iloc[0]
            
            # Grid Generation
            x_pts = np.linspace(-Lx/2, Lx/2, 40)
            z_pts = np.linspace(-Lz/2, Lz/2, 40)
            X, Z = np.meshgrid(x_pts, z_pts)
            
            P_p, Mx_p, Mz_p = rp['Fy'] + W_total, rp['Mx'] + rp['Fz']*H, rp['Mz'] + rp['Fx']*H
            Q = (P_p/Area) + (Mx_p * Z / Ix) + (Mz_p * X / Iz)
            
            fig = go.Figure(data=go.Heatmap(z=Q, x=x_pts, y=z_pts, colorscale='RdYlGn_r', zmin=0, zmax=qa*1.2))
            fig.update_layout(title=f"Pressure Contour: {sel_p}", xaxis_title="X (Length)", yaxis_title="Z (Width)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
