"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Optimized for STAAD Copy-Paste with 'LC' Column Support
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foundation Design Tool", page_icon="🏗️",
                    layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e0e0e0}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#16213e);border:1px solid #2d3561;
  border-radius:10px;padding:16px;text-align:center;margin:4px}
.metric-value{font-size:1.6rem;font-weight:700;color:#4fc3f7}
.metric-label{font-size:.75rem;color:#90a4ae;text-transform:uppercase;letter-spacing:1px}
.sec-hdr{background:linear-gradient(90deg,#1565c0,#0d47a1);color:#fff;
  padding:8px 16px;border-radius:6px;font-weight:700;font-size:1.05rem;margin:12px 0 8px}
.info-box{background:#0d2137;border-left:4px solid #1565c0;
  padding:10px 14px;border-radius:4px;margin:6px 0;font-size:.88rem}
div[data-testid="stSidebar"]{background:#13192b}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-hdr">{t}</div>', unsafe_allow_html=True)
def info(t): st.markdown(f'<div class="info-box">{t}</div>', unsafe_allow_html=True)
def mcard(lbl, val, col):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                  f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_staad_text(text, flip_axial):
    try:
        # First try to read as tab-separated (standard for Excel/STAAD copy-paste)
        df = pd.read_csv(io.StringIO(text), sep='\t')
        
        # If it failed to find columns, try standardizing spaces/tabs to commas
        if len(df.columns) <= 1:
            clean_text = re.sub(r'[ \t]+', ',', text.strip())
            df = pd.read_csv(io.StringIO(clean_text))
        
        # Mapping common column names
        mapping = {
            'LC': 'Case', 'L/C': 'Case', 'LOAD': 'Case', 'CASE': 'Case',
            'FX': 'Fx', 'FY': 'Fy', 'FZ': 'Fz',
            'MX': 'Mx', 'MY': 'My', 'MZ': 'Mz'
        }
        
        # Standardize names
        df.columns = [str(c).upper().strip() for c in df.columns]
        df.rename(columns=mapping, inplace=True)
        
        # Ensure 'Case' exists - if not, use the first column
        if 'Case' not in df.columns:
            df.rename(columns={df.columns[0]: 'Case'}, inplace=True)
            
        # Clean numeric data
        num_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
        if flip_axial:
            df['Fy'] = -df['Fy']
            
        return df
    except Exception as e:
        st.error(f"Parsing error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3D VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d(row, Lx, Lz, H, units):
    fig = go.Figure()
    hw, hd = Lx/2, Lz/2
    # Footing Mesh
    fig.add_trace(go.Mesh3d(
        x=[-hw,hw,hw,-hw,-hw,hw,hw,-hw], y=[-hd,-hd,hd,hd,-hd,-hd,hd,hd], z=[-H,-H,-H,-H,0,0,0,0],
        i=[0,0,0,4,4,4,0,2,2,0,6,4], j=[1,2,3,5,6,7,1,3,6,4,7,5], k=[2,3,0,6,7,4,5,7,5,6,2,1],
        color="#37474f", opacity=0.6, name="Footing"
    ))
    # Load Vector
    sc = max(Lx, Lz) * 0.4 / (abs(row['Fy']) + 1)
    fig.add_trace(go.Scatter3d(
        x=[0, row['Fx']*sc], y=[0, row['Fz']*sc], z=[1, 1-abs(row['Fy'])*sc],
        mode='lines+markers', line=dict(color='red', width=8), name='Applied Load'
    ))
    fig.update_layout(scene=dict(aspectmode='data', bgcolor="#0d1117"), margin=dict(l=0,r=0,t=0,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("<h2 style='color:#90caf9'>🏗️ Foundation Design & Sizing Tool</h2>", unsafe_allow_html=True)
    
    with st.sidebar:
        sec("1. Configuration")
        unit_sys = st.selectbox("Units", ["Imperial (kip, ft)", "Metric (kN, m)"])
        flip_ax = st.checkbox("Flip Axial Sign (STAAD Reaction)", value=True)
        asd_tag = st.text_input("Sizing LC Prefix", value="[4-")
        uls_tag = st.text_input("Strength LC Prefix", value="[5-")
        
        sec("2. Soil & Material")
        qa = st.number_input("Allowable Bearing (ksf/kPa)", value=3.0 if "Imp" in unit_sys else 150.0)
        gc = 0.150 if "Imp" in unit_sys else 24.0
        
        sec("3. Trial Geometry")
        Lx = st.number_input("Length Lx", value=12.0 if "Imp" in unit_sys else 4.0)
        Lz = st.number_input("Width Lz", value=12.0 if "Imp" in unit_sys else 4.0)
        H = st.number_input("Thickness H", value=2.5 if "Imp" in unit_sys else 0.8)

    t1, t2 = st.tabs(["📂 Load Input", "📐 Sizing Results"])

    with t1:
        sec("① Load Entry")
        default_data = """LC	FX	FY	FZ	MX	MY	MZ
[5-1.1:1.4(DS+DO)+1.2TSEXP]	-0.01	1.2	-0.1	-1.11	0.02	0.55
[5-8.1:1.2(DS+DO)+1.2TSCON+1.2TT+0.5L+0.5S]	3.49	-0.78	-0.75	-6.39	0.13	-45.3
[4-1.2:DS+DO+TSCON+TT]	2.91	-0.65	-0.62	-5.33	0.1	-37.75
[4-7.2:0.6(DS+DO)+TSCON+0.6WZ]	0.01	0.51	-0.27	-1.51	-0.02	0.09"""
        
        raw = st.text_area("Paste STAAD Reactions Here", value=default_data, height=200)
        if raw:
            df = parse_staad_text(raw, flip_ax)
            if df is not None:
                st.dataframe(df, use_container_width=True)
                st.session_state['ldf'] = df

    with t2:
        if 'ldf' in st.session_state:
            ldf = st.session_state['ldf']
            asd_df = ldf[ldf['Case'].str.contains(asd_tag, na=False, regex=False)]
            
            if not asd_df.empty:
                sec("② Sizing Summary (ASD)")
                Area = Lx * Lz
                Sx, Sz = (Lx * Lz**2)/6, (Lz * Lx**2)/6
                Wt = Area * H * gc
                
                res_rows = []
                for _, r in asd_df.iterrows():
                    P_tot = r['Fy'] + Wt
                    Mx_b = r['Mx'] + r['Fz']*H
                    Mz_b = r['Mz'] + r['Fx']*H
                    q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
                    res_rows.append({"Case": r['Case'], "P_Total": round(P_tot, 2), "q_max": round(q_max, 3), "Status": "✅ OK" if q_max <= qa else "❌ FAIL"})
                
                st.table(pd.DataFrame(res_rows))
                
                # 3D View of controlling case
                st.divider()
                sec("③ 3D Critical Load View")
                crit_row = asd_df.iloc[0]
                st.plotly_chart(fig_3d(crit_row, Lx, Lz, H, unit_sys), use_container_width=True)
            else:
                st.warning(f"No sizing cases found starting with '{asd_tag}'")

if __name__ == "__main__":
    main()
