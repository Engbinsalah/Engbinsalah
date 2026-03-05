"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Added: Detailed Calculation Report with Step-by-Step Math
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foundation Sizing & Calcs", page_icon="📐", layout="wide")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e0e0e0}
.sec-hdr{background:linear-gradient(90deg,#1565c0,#0d47a1);color:#fff;
  padding:8px 16px;border-radius:6px;font-weight:700;margin:12px 0 8px}
.calc-block{background:#161b22; border: 1px solid #30363d; padding: 20px; border-radius: 8px; font-family: 'Courier New', Courier, monospace;}
</style>""", unsafe_allow_html=True)

def parse_staad_text(text, flip_axial):
    try:
        clean_text = re.sub(r'[ \t]+', ',', text.strip())
        df = pd.read_csv(io.StringIO(clean_text))
        mapping = {'LC': 'Case', 'L/C': 'Case', 'LOAD': 'Case', 'CASE': 'Case', 'FX': 'Fx', 'FY': 'Fy', 'FZ': 'Fz', 'MX': 'Mx', 'MY': 'My', 'MZ': 'Mz'}
        df.columns = [str(c).upper().strip() for c in df.columns]
        df.rename(columns=mapping, inplace=True)
        if 'Case' not in df.columns: df.rename(columns={df.columns[0]: 'Case'}, inplace=True)
        for c in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        if flip_axial: df['Fy'] = -df['Fy']
        return df
    except Exception as e:
        st.error(f"Parsing error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🏗️ Foundation Sizing & Detailed Calculations")
    
    with st.sidebar:
        st.header("1. Global Settings")
        unit_sys = st.selectbox("Units", ["Imperial (kip, ft)", "Metric (kN, m)"])
        flip_ax = st.checkbox("Flip Axial Sign (STAAD Reaction)", value=True)
        
        st.header("2. Design Limits")
        qa = st.number_input("Allowable Bearing (ksf/kPa)", value=3.0 if "Imp" in unit_sys else 150.0)
        mu = st.number_input("Friction Coeff (μ)", value=0.45)
        
        st.header("3. Material Densities")
        gc = st.number_input("Concrete (pcf/kN/m³)", value=150.0 if "Imp" in unit_sys else 24.0)
        gs = st.number_input("Soil (pcf/kN/m³)", value=110.0 if "Imp" in unit_sys else 18.0)
        
        st.header("4. Footing Geometry")
        Lx = st.number_input("Length Lx", value=12.0 if "Imp" in unit_sys else 4.0)
        Lz = st.number_input("Width Lz", value=12.0 if "Imp" in unit_sys else 4.0)
        H = st.number_input("Thickness H", value=2.0 if "Imp" in unit_sys else 0.6)
        Df = st.number_input("Soil Surcharge Depth", value=2.0 if "Imp" in unit_sys else 0.6)

    t1, t2, t3 = st.tabs(["📂 Load Input", "📐 Sizing Summary", "📜 Calculation Report"])

    with t1:
        default_data = "LC	FX	FY	FZ	MX	MY	MZ\n[4-1.2]	2.91	-0.65	-0.62	-5.33	0.1	-37.75"
        raw = st.text_area("Paste Load Cases", value=default_data, height=150)
        if raw:
            df = parse_staad_text(raw, flip_ax)
            st.session_state['ldf'] = df
            st.dataframe(df)

    if 'ldf' in st.session_state:
        ldf = st.session_state['ldf']
        
        # --- Constants ---
        Area = Lx * Lz
        Sx = (Lx * Lz**2) / 6
        Sz = (Lz * Lx**2) / 6
        
        # Weight Calcs
        if "Imp" in unit_sys:
            W_conc = Area * H * (gc/1000)
            W_soil = Area * Df * (gs/1000)
        else:
            W_conc = Area * H * gc
            W_soil = Area * Df * gs
            
        W_total = W_conc + W_soil
        
        with t2:
            st.subheader("Sizing Overview")
            results = []
            for _, r in ldf.iterrows():
                P_tot = r['Fy'] + W_total
                Mx_b = r['Mx'] + r['Fz']*H
                Mz_b = r['Mz'] + r['Fx']*H
                q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
                results.append({"LC": r['Case'], "P_Total": round(P_tot,2), "q_max": round(q_max,3), "Status": "OK" if q_max <= qa else "FAIL"})
            st.table(pd.DataFrame(results))

        with t3:
            st.subheader("Step-by-Step Mathematical Breakdown")
            sel_case = st.selectbox("Select Case for Detailed Calcs", ldf['Case'])
            r = ldf[ldf['Case'] == sel_case].iloc[0]
            
            Mx_b = r['Mx'] + r['Fz']*H
            Mz_b = r['Mz'] + r['Fx']*H
            P_tot = r['Fy'] + W_total

            st.markdown("### 1. Foundation Self-Weight")
            st.latex(f"W_{{concrete}} = L_x \\times L_z \\times H \\times \\gamma_c = {Lx} \\times {Lz} \\times {H} \\times {gc/1000 if 'Imp' in unit_sys else gc} = {W_conc:.2f}")
            st.latex(f"W_{{soil}} = L_x \\times L_z \\times D_f \\times \\gamma_s = {Lx} \\times {Lz} \\times {Df} \\times {gs/1000 if 'Imp' in unit_sys else gs} = {W_soil:.2f}")
            st.latex(f"W_{{total}} = {W_conc:.2f} + {W_soil:.2f} = {W_total:.2f}")

            st.markdown("### 2. Section Properties")
            st.latex(f"Area (A) = L_x \\times L_z = {Lx} \\times {Lz} = {Area:.2f}")
            st.latex(f"S_x = \\frac{{L_x \\times L_z^2}}{{6}} = \\frac{{{Lx} \\times {Lz}^2}}{{6}} = {Sx:.2f}")
            st.latex(f"S_z = \\frac{{L_z \\times L_x^2}}{{6}} = \\frac{{{Lz} \\times {Lx}^2}}{{6}} = {Sz:.2f}")

            st.markdown("### 3. Bearing Pressure (ASD)")
            st.latex(f"M_{{x,base}} = M_x + (F_z \\times H) = {r['Mx']} + ({r['Fz']} \\times {H}) = {Mx_b:.2f}")
            st.latex(f"M_{{z,base}} = M_z + (F_x \\times H) = {r['Mz']} + ({r['Fx']} \\times {H}) = {Mz_b:.2f}")
            
            q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
            st.latex(f"q_{{max}} = \\frac{{P_{{total}}}}{{A}} + \\frac{{|M_{{x,base}}|}}{{S_x}} + \\frac{{|M_{{z,base}}|}}{{S_z}}")
            st.latex(f"q_{{max}} = \\frac{{{P_tot:.2f}}}{{{Area:.2f}}} + \\frac{{{abs(Mx_b):.2f}}}{{{Sx:.2f}}} + \\frac{{{abs(Mz_b):.2f}}}{{{Sz:.2f}}} = {q_max:.3f}")
            
            if q_max <= qa:
                st.success(f"Result: {q_max:.3f} ≤ {qa} (Allowable) → PASS")
            else:
                st.error(f"Result: {q_max:.3f} > {qa} (Allowable) → FAIL")

            st.markdown("### 4. Stability Checks")
            resisting_fx = abs(P_tot) * mu
            actual_fx = math.sqrt(r['Fx']**2 + r['Fz']**2)
            fos_sliding = resisting_fx / actual_fx if actual_fx > 0 else 999
            
            st.markdown("**Sliding Check:**")
            st.latex(f"F_{{resisting}} = P_{{total}} \\times \\mu = {P_tot:.2f} \\times {mu} = {resisting_fx:.2f}")
            st.latex(f"F_{{actual}} = \\sqrt{{F_x^2 + F_z^2}} = \\sqrt{{{r['Fx']}^2 + {r['Fz']}^2}} = {actual_fx:.2f}")
            st.latex(f"FOS_{{sliding}} = \\frac{{{resisting_fx:.2f}}}{{{actual_fx:.2f}}} = {fos_sliding:.2f}")

if __name__ == "__main__":
    main()
