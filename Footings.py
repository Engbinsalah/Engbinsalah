"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Added: Stress Contour Heatmap Tab
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
    st.title("🏗️ Foundation Sizing & Stress Analysis")
    
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

    t1, t2, t3, t4 = st.tabs(["📂 Load Input", "📐 Sizing Summary", "📜 Calculation Report", "🌈 Stress Contour"])

    with t1:
        sec("① Load Case Entry")
        default_data = "LC	FX	FY	FZ	MX	MY	MZ\n[4-1.2]	2.91	-0.65	-0.62	-5.33	0.1	-37.75"
        raw = st.text_area("Paste Load Cases", value=default_data, height=150)
        if raw:
            df = parse_staad_text(raw, flip_ax)
            st.session_state['ldf'] = df
            st.dataframe(df)

    if 'ldf' in st.session_state:
        ldf = st.session_state['ldf']
        
        # --- Constants & Properties ---
        Area = Lx * Lz
        Sx = (Lx * Lz**2) / 6
        Sz = (Lz * Lx**2) / 6
        Ix = (Lx * Lz**3) / 12
        Iz = (Lz * Lx**3) / 12
        
        # Weight Calculations
        if "Imp" in unit_sys:
            W_conc = Area * H * (gc/1000)
            W_soil = Area * Df * (gs/1000)
        else:
            W_conc = Area * H * gc
            W_soil = Area * Df * gs
        W_total = W_conc + W_soil
        
        with t2:
            sec("② Sizing Overview (ASD Checks)")
            results = []
            for _, r in ldf.iterrows():
                P_tot = r['Fy'] + W_total
                Mx_b = r['Mx'] + r['Fz']*H
                Mz_b = r['Mz'] + r['Fx']*H
                q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)
                results.append({"LC": r['Case'], "P_Total": round(P_tot,2), "q_max": round(q_max,3), "Status": "✅ OK" if q_max <= qa else "❌ FAIL"})
            st.table(pd.DataFrame(results))

        with t3:
            sec("③ Detailed Calculation (Step-by-Step)")
            sel_case = st.selectbox("Select Case for Detailed Calcs", ldf['Case'])
            r = ldf[ldf['Case'] == sel_case].iloc[0]
            
            P_tot = r['Fy'] + W_total
            Mx_b = r['Mx'] + r['Fz']*H
            Mz_b = r['Mz'] + r['Fx']*H
            q_max = (P_tot/Area) + abs(Mx_b/Sx) + abs(Mz_b/Sz)

            st.latex(f"q_{{max}} = \\frac{{{P_tot:.2f}}}{{{Area:.2f}}} + \\frac{{{abs(Mx_b):.2f}}}{{{Sx:.2f}}} + \\frac{{{abs(Mz_b):.2f}}}{{{Sz:.2f}}} = {q_max:.3f}")

        with t4:
            sec("④ Soil Bearing Pressure Contour")
            sel_case_plot = st.selectbox("Select Case for Contour", ldf['Case'], key="plot_case")
            rp = ldf[ldf['Case'] == sel_case_plot].iloc[0]
            
            P_p = rp['Fy'] + W_total
            Mx_p = rp['Mx'] + rp['Fz']*H
            Mz_p = rp['Mz'] + rp['Fx']*H
            
            # Generate Grid
            res = 50
            x = np.linspace(-Lx/2, Lx/2, res)
            z = np.linspace(-Lz/2, Lz/2, res)
            X, Z = np.meshgrid(x, z)
            
            # Linear Pressure distribution: q = P/A + Mx*z/Ix + Mz*x/Iz
            # Note: Mx creates stress along Z-axis, Mz creates stress along X-axis
            Q = (P_p/Area) + (Mx_p * Z / Ix) + (Mz_p * X / Iz)
            
            fig = go.Figure(data=go.Heatmap(
                z=Q, x=x, y=z,
                colorscale='RdYlGn_r',
                zmin=0, zmax=qa * 1.2,
                colorbar=dict(title=f"Pressure ({'ksf' if 'Imp' in unit_sys else 'kPa'})")
            ))
            
            # Add labels for corners
            corners_x = [Lx/2, -Lx/2, Lx/2, -Lx/2]
            corners_z = [Lz/2, Lz/2, -Lz/2, -Lz/2]
            corner_q = (P_p/Area) + (Mx_p * np.array(corners_z) / Ix) + (Mz_p * np.array(corners_x) / Iz)
            
            fig.add_trace(go.Scatter(
                x=corners_x, y=corners_z,
                mode='text+markers',
                text=[f"{v:.2f}" for v in corner_q],
                textposition="top center",
                marker=dict(color='black', size=10),
                name="Corner Pressure"
            ))

            fig.update_layout(
                title=f"Bearing Pressure Distribution: {sel_case_plot}",
                xaxis_title="X-Axis (Length)",
                yaxis_title="Z-Axis (Width)",
                width=800, height=600,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if np.any(Q < 0):
                st.warning("⚠️ Uplift Detected: Part of the footing base has negative pressure (loss of contact).")

if __name__ == "__main__":
    main()
