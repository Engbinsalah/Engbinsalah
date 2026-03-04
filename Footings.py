"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Enhanced with Robust STAAD Copy-Paste Parser
ACI 318-19 | ASCE 7-22 Methodology
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io, math, re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foundation Design Tool", page_icon="🏗️",
                    layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e0e0e0}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#16213e);border:1px solid #2d3561;
  border-radius:10px;padding:16px;text-align:center;margin:4px}
.metric-value{font-size:1.8rem;font-weight:700;color:#4fc3f7}
.metric-label{font-size:.8rem;color:#90a4ae;text-transform:uppercase;letter-spacing:1px}
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
# LOAD ENGINE & PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_staad_text(text, flip_axial):
    """Parses raw text from STAAD (tabs/spaces/commas) into a clean DataFrame."""
    try:
        # Standardize delimiters: replace tabs and multiple spaces with a single comma
        clean_text = re.sub(r'[ \t]+', ',', text.strip())
        df = pd.read_csv(io.StringIO(clean_text))
        
        # Mapping common STAAD column names to our standard
        mapping = {
            'L/C': 'Case', 'Node': 'Node',
            'FX': 'Fx', 'FY': 'Fy', 'FZ': 'Fz',
            'MX': 'Mx', 'MY': 'My', 'MZ': 'Mz'
        }
        df.rename(columns=lambda x: mapping.get(x.upper(), x), inplace=True)
        
        # Sign Handling: STAAD reactions are often opposite to the load on the footing
        if flip_axial:
            df['Fy'] = -df['Fy']
            
        return df
    except Exception as e:
        st.error(f"Parsing failed: {e}")
        return None

def apply_combos(load_df, unit_system):
    """Applies ACI/ASCE load combinations based on Case name matching."""
    # This logic matches Case names like 'DL', 'LL' etc to factors
    # For a purely manual review of pasted results, we often just identify 
    # which pasted rows are SLS (Service) and which are ULS (Ultimate).
    return load_df # Returning raw for this implementation to allow direct selection

# ─────────────────────────────────────────────────────────────────────────────
# 3D VISUALISER
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d(load_df, Lx, Lz, col_w, col_d, units):
    fig = go.Figure()
    hw, hd, ht = Lx/2, Lz/2, 0.6 if units == "Metric" else 2.0
    vx=[-hw,hw,hw,-hw,-hw,hw,hw,-hw]; vy=[-hd,-hd,hd,hd,-hd,-hd,hd,hd]
    vz=[-ht,-ht,-ht,-ht,0,0,0,0]
    I=[0,0,0,4,4,4,0,2,2,0,6,4]; J=[1,2,3,5,6,7,1,3,6,4,7,5]; K=[2,3,0,6,7,4,5,7,5,6,2,1]
    fig.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=I,j=J,k=K,color="#37474f",opacity=.55,name="Footing"))
    
    # Critical load row selection
    try:
        idx = load_df['Fy'].abs().idxmax()
        row = load_df.loc[idx]
    except:
        return fig

    sc = max(Lx, Lz) * 0.5 / (abs(row['Fy']) + 1e-9)
    fig.add_trace(go.Scatter3d(x=[0, row['Fx']*sc], y=[0, row['Fz']*sc], z=[1.5, 1.5 - abs(row['Fy'])*sc],
                               mode='lines+markers', line=dict(color='red', width=6), name='Applied Load Vector'))

    fig.update_layout(scene=dict(aspectmode='data', bgcolor="#0d1117"), paper_bgcolor="#0f1117", margin=dict(l=0,r=0,t=0,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("<h1>🏗️ Foundation Design Tool <span style='font-size:15px; color:#78909c;'>STAAD Enhanced</span></h1>", unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        sec("⚙️ Configuration")
        unit_sys = st.selectbox("Unit System", ["Metric (kN, m, MPa)", "Imperial (kip, ft, ksi)"])
        
        sec("🧱 Materials")
        if unit_sys == "Metric (kN, m, MPa)":
            fc = st.number_input("f'c (MPa)", 20., 80., 28.)
            fy = st.number_input("fy (MPa)", 300., 600., 420.)
            qa = st.number_input("Allowable Bearing (kPa)", 50., 1000., 200.)
            gc = 24.0 # kN/m3
        else:
            fc = st.number_input("f'c (psi)", 3000, 6000, 4000) / 1000 # to ksi
            fy = st.number_input("fy (psi)", 40000, 75000, 60000) / 1000 # to ksi
            qa = st.number_input("Allowable Bearing (ksf)", 1.0, 10.0, 3.0)
            gc = 0.150 # kcf

        sec("🏛️ Footing & Column")
        Df = st.number_input("Total Depth Df", 0.5, 15.0, 1.5 if "Metric" in unit_sys else 5.0)
        cw = st.number_input("Column X-dim", 0.2, 5.0, 0.5)
        cd = st.number_input("Column Z-dim", 0.2, 5.0, 0.5)
        flip_axial = st.checkbox("Flip Axial Sign (STAAD Reaction)", value=True)

    t1, t2, t3 = st.tabs(["📂 Load Input", "📐 Sizing Results", "📊 3D View"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: ENHANCED LOAD INPUT
    # ════════════════════════════════════════════════════════════════════════
    with t1:
        sec("① Load Entry")
        st.info("Paste your STAAD 'Support Reactions' table below. Supports Tab or Space delimiters.")
        
        raw_text = st.text_area("Paste from STAAD / Excel here", height=250, 
                                placeholder="Node  L/C  FX  FY  FZ  MX  MY  MZ...")
        
        if raw_text:
            df = parse_staad_text(raw_text, flip_axial)
            if df is not None:
                st.success(f"Successfully parsed {len(df)} load cases.")
                st.dataframe(df, use_container_width=True)
                st.session_state['ldf'] = df
                
                # Sizing Filter Identification
                c1, c2 = st.columns(2)
                asd_tag = c1.text_input("ASD/Service Case Identifier", value="[4-")
                uls_tag = c2.text_input("LRFD/Ultimate Case Identifier", value="[5-")
                
                st.session_state['asd_tag'] = asd_tag
                st.session_state['uls_tag'] = uls_tag

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: SIZING
    # ════════════════════════════════════════════════════════════════════════
    with t2:
        if 'ldf' not in st.session_state:
            st.warning("Please paste loads in Tab 1.")
        else:
            ldf = st.session_state['ldf']
            asd_tag = st.session_state['asd_tag']
            
            # Filter ASD cases for bearing check
            asd_df = ldf[ldf['Case'].astype(str).str.contains(asd_tag, na=False, regex=False)]
            
            if asd_df.empty:
                st.error(f"No Service cases found matching '{asd_tag}'. Adjust the filter.")
            else:
                sec("② Bearing & Stability Review (Service)")
                
                # Interactive Size Adjustment
                Lx = st.number_input("Trial Length Lx", value=10.0 if "Imperial" in unit_sys else 3.0)
                Lz = st.number_input("Trial Width Lz", value=10.0 if "Imperial" in unit_sys else 3.0)
                H = st.number_input("Trial Thickness H", value=2.0 if "Imperial" in unit_sys else 0.6)
                
                # Sizing Logic
                Area = Lx * Lz
                Sx, Sz = (Lx * Lz**2)/6, (Lz * Lx**2)/6
                Wt = Area * H * gc
                
                results = []
                for _, row in asd_df.iterrows():
                    P_tot = row['Fy'] + Wt
                    Mx_b = row['Mx'] + row['Fz']*H
                    Mz_b = row['Mz'] + row['Fx']*H
                    
                    q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
                    results.append({
                        "Case": row['Case'], "P_Total": round(P_tot, 2),
                        "q_max": round(q_max, 3), "Status": "PASS" if q_max <= qa else "FAIL"
                    })
                
                st.table(pd.DataFrame(results))

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: 3D VISUALIZATION
    # ════════════════════════════════════════════════════════════════════════
    with t3:
        if 'ldf' in st.session_state:
            sec("③ 3D Applied Load Vector")
            units = "Metric" if "Metric" in unit_sys else "Imperial"
            # Using trial sizes from Tab 2 or defaults
            fig = fig_3d(st.session_state['ldf'], 3.0, 3.0, cw, cd, units)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
