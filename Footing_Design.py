import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Foundation Engineering Report", page_icon="🏗️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background-color: #0f1117; }
.block-container { padding-top: 1rem; }

/* ── Title Bar ── */
.title-bar {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a6b 50%, #0d2244 100%);
    border-bottom: 3px solid #2563eb;
    padding: 24px 40px 20px;
    margin-bottom: 24px;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
}
.title-bar::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 300px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.15));
}
.title-bar h1 {
    color: #e0f2fe;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0 0 4px;
    text-transform: uppercase;
}
.title-bar .subtitle {
    color: #7dd3fc;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.title-bar .badge {
    display: inline-block;
    background: rgba(37,99,235,0.3);
    border: 1px solid #3b82f6;
    color: #93c5fd;
    padding: 3px 12px;
    border-radius: 2px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    margin-top: 10px;
}

/* ── Section Headers ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 4px solid #2563eb;
    padding: 10px 16px;
    background: linear-gradient(90deg, rgba(37,99,235,0.12), transparent);
    color: #bfdbfe;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 20px 0 14px;
    border-radius: 0 4px 4px 0;
}

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
.kpi-card {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-top: 3px solid #2563eb;
    padding: 16px;
    border-radius: 6px;
    text-align: center;
}
.kpi-card.pass { border-top-color: #16a34a; }
.kpi-card.fail { border-top-color: #dc2626; }
.kpi-card.warn { border-top-color: #d97706; }
.kpi-label { color: #94a3b8; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.kpi-unit  { color: #64748b; font-size: 0.75rem; margin-top: 2px; }
.kpi-status-pass { color: #4ade80; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; }
.kpi-status-fail { color: #f87171; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; }

/* ── Math Boxes ── */
.math-panel {
    background: #151c2e;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 18px 22px;
    margin: 10px 0;
    font-family: 'IBM Plex Mono', monospace;
}
.math-panel .label { color: #64748b; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.math-panel .formula { color: #a5b4fc; font-size: 0.92rem; margin: 3px 0; }
.math-panel .result  { color: #fde68a; font-size: 1.05rem; font-weight: 600; margin-top: 8px; border-top: 1px solid #2d3748; padding-top: 8px; }

/* ── Tables ── */
.eng-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin: 12px 0; }
.eng-table th {
    background: #1a2540;
    color: #7dd3fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 14px;
    border-bottom: 2px solid #2563eb;
    text-align: left;
}
.eng-table td { padding: 9px 14px; border-bottom: 1px solid #1e2535; color: #cbd5e1; }
.eng-table tr:hover td { background: rgba(37,99,235,0.05); }
.eng-table .hi { color: #fde68a; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.eng-table .pass-cell { color: #4ade80; font-weight: 700; }
.eng-table .fail-cell { color: #f87171; font-weight: 700; }

/* ── Geometry box ── */
.geom-row { display: flex; gap: 16px; margin: 12px 0; flex-wrap: wrap; }
.geom-chip {
    background: #1a2540;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    min-width: 90px;
}
.geom-chip .glabel { color: #64748b; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; }
.geom-chip .gval   { color: #e2e8f0; font-size: 1.0rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

/* ── Alert blocks ── */
.alert-pass { background: rgba(22,163,74,0.1); border: 1px solid #16a34a; border-radius: 4px; padding: 10px 16px; color: #4ade80; font-size: 0.88rem; margin: 8px 0; }
.alert-fail { background: rgba(220,38,38,0.1); border: 1px solid #dc2626; border-radius: 4px; padding: 10px 16px; color: #f87171; font-size: 0.88rem; margin: 8px 0; }
.alert-info { background: rgba(37,99,235,0.1); border: 1px solid #2563eb; border-radius: 4px; padding: 10px 16px; color: #93c5fd; font-size: 0.88rem; margin: 8px 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label { font-size: 0.78rem; color: #94a3b8 !important; letter-spacing: 0.05em; }
[data-testid="stSidebar"] hr { border-color: #1e2d4a !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #131b2e; border-radius: 6px 6px 0 0; gap: 2px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 4px; padding: 8px 18px; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.07em; }
.stTabs [aria-selected="true"] { background: #1e3a7a !important; color: #93c5fd !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Design Parameters")
    unit_sys = st.selectbox("Unit System", ["Imperial (kip, ft)", "Metric (kN, m)"])

    st.markdown("---")
    st.markdown("**Geometry**")
    col1s, col2s = st.columns(2)
    with col1s:
        Lx = st.number_input("Length Lx", value=8.0, step=0.5)
        T  = st.number_input("Thickness T", value=1.0, step=0.25)
        cx = st.number_input("Col Dim cx", value=2.0, step=0.25)
    with col2s:
        Lz = st.number_input("Width Lz", value=7.0, step=0.5)
        D  = st.number_input("Depth D", value=3.0, step=0.25)
        cz = st.number_input("Col Dim cz", value=2.0, step=0.25)

    st.markdown("---")
    st.markdown("**Materials & Soil**")
    if "Imp" in unit_sys:
        gc = st.number_input("Concrete (pcf)", value=150.0) / 1000
        gs = st.number_input("Soil (pcf)", value=100.0) / 1000
        qa = st.number_input("Allow. Bearing (ksf)", value=3.0)
        f_unit, l_unit, p_unit = "kip", "ft", "ksf"
    else:
        gc = st.number_input("Concrete (kN/m³)", value=24.0)
        gs = st.number_input("Soil (kN/m³)", value=18.0)
        qa = st.number_input("Allow. Bearing (kPa)", value=150.0)
        f_unit, l_unit, p_unit = "kN", "m", "kPa"

    st.markdown("---")
    st.markdown("**Resistance Parameters**")
    mu   = st.number_input("Friction Coeff (μ)", value=0.45, step=0.05)
    coh  = st.number_input("Cohesion (kip)", value=7.0, step=0.5)
    Ppx  = st.number_input("Passive Resist X (kip)", value=1.75, step=0.25)
    Ppz  = st.number_input("Passive Resist Z (kip)", value=2.0, step=0.25)
    sf_sliding_min = st.number_input("Min SF Sliding", value=1.5, step=0.1)
    sf_ot_min      = st.number_input("Min SF Overturning", value=1.5, step=0.1)
    sf_uplift_min  = st.number_input("Min SF Uplift", value=1.5, step=0.1)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-bar">
  <div class="subtitle">Structural Engineering</div>
  <h1>🏗️ Isolated Foundation — Geotechnical Verification</h1>
  <div class="badge">ASD METHOD · SPREAD FOOTING · SOIL SUPPORTED</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">📥 Load Case Input</div>', unsafe_allow_html=True)

default_load = "LC-ASD-01\t2.89\t-62.0\t-0.82\t-48.88\t-12.075\t-2.345"
raw_load = st.text_area(
    "Paste Load Case · Format: **LC | Fx | Fy | Fz | Mx | My | Mz**",
    value=default_load, height=68
)

def parse_load(text):
    try:
        p = re.split(r'[ \t,]+', text.strip())
        return {"LC": p[0], "Fx": float(p[1]), "Fy": float(p[2]), "Fz": float(p[3]),
                "Mx": float(p[4]), "My": float(p[5]), "Mz": float(p[6])}
    except:
        return None

L_DATA = parse_load(raw_load)

# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINEERING CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
if L_DATA:
    Area_ftg  = Lx * Lz
    Area_col  = cx * cz
    Hp        = D - T          # soil/pedestal height above footing base

    # Self-weights
    Wt_ftg    = Area_ftg * T * gc
    Wt_ped    = Area_col * Hp * gc
    Wt_soil   = (Area_ftg - Area_col) * Hp * gs
    Wt_total  = Wt_ftg + Wt_ped + Wt_soil

    # Vertical load at base
    P_applied = L_DATA['Fy']          # typically negative (compression)
    P_total   = P_applied + Wt_total  # net downward at base (negative = compression)

    # Moments transferred to base (sign convention: amplify by depth)
    Mx_base = L_DATA['Mx'] + (L_DATA['Fz'] * D)
    Mz_base = L_DATA['Mz'] + (L_DATA['Fx'] * D)

    # Section properties
    Sx = (Lx * Lz**2) / 6    # about X axis  (resists Mx)
    Sz = (Lz * Lx**2) / 6    # about Z axis  (resists Mz)
    Ix = (Lx * Lz**3) / 12
    Iz = (Lz * Lx**3) / 12

    # Eccentricity
    P_net = abs(P_total)
    ex = abs(Mx_base) / P_net if P_net > 0 else 0   # eccentricity in X-direction
    ez = abs(Mz_base) / P_net if P_net > 0 else 0   # eccentricity in Z-direction

    # ── Bearing Pressure (biaxial) ──────────────────────────────────────────
    q_avg  = P_net / Area_ftg
    dqx    = abs(Mx_base) / Sx
    dqz    = abs(Mz_base) / Sz

    # Corner pressures (four corners)
    corners = {
        "C1 (+x,+z)": q_avg + dqx + dqz,
        "C2 (-x,+z)": q_avg + dqx - dqz,
        "C3 (+x,-z)": q_avg - dqx + dqz,
        "C4 (-x,-z)": q_avg - dqx - dqz,
    }
    q_max = max(corners.values())
    q_min = min(corners.values())
    q_ratio = q_max / qa

    # Neutral axis (zero-pressure line intercepts)
    # q(x,z) = q_avg + Mx_base*z/Ix + Mz_base*x/Iz = 0
    # X-intercept (z=0): x = -q_avg * Iz / Mz_base
    # Z-intercept (x=0): z = -q_avg * Ix / Mx_base
    try:
        NA_x_intercept = -q_avg * Iz / Mx_base if abs(Mx_base) > 1e-9 else None
    except:
        NA_x_intercept = None
    try:
        NA_z_intercept = -q_avg * Ix / Mz_base if abs(Mz_base) > 1e-9 else None
    except:
        NA_z_intercept = None

    # ── Partial Contact Detection ────────────────────────────────────────────
    x_lin  = np.linspace(-Lx/2, Lx/2, 80)
    z_lin  = np.linspace(-Lz/2, Lz/2, 80)
    X, Z   = np.meshgrid(x_lin, z_lin)
    Q_raw  = (P_net/Area_ftg) + (Mx_base * Z / Ix) + (Mz_base * X / Iz)
    Q_field = np.maximum(Q_raw, 0.0)   # uplift = 0 bearing

    contact_area = np.sum(Q_field > 0) / Q_field.size * 100
    in_full_contact = q_min >= 0

    # Effective vertices (for partial contact diagram)
    v1_bearing = corners["C1 (+x,+z)"]
    v2_bearing = corners["C2 (-x,+z)"]

    # ── Stability ────────────────────────────────────────────────────────────
    # Overturning — per edge
    Mot_Xleft  = abs(Mx_base)      # overturning moment about left edge (Z+)
    Mrs_Xleft  = P_net * (Lz / 2) # resisting moment
    Mot_Xright = abs(Mx_base)
    Mrs_Xright = P_net * (Lz / 2)

    Mot_Zleft  = abs(Mz_base)
    Mrs_Zleft  = P_net * (Lx / 2)
    Mot_Zright = abs(Mz_base)
    Mrs_Zright = P_net * (Lx / 2)

    SF_ot_Xleft  = Mrs_Xleft  / Mot_Xleft  if Mot_Xleft  > 0 else 99
    SF_ot_Xright = Mrs_Xright / Mot_Xright if Mot_Xright > 0 else 99
    SF_ot_Zleft  = Mrs_Zleft  / Mot_Zleft  if Mot_Zleft  > 0 else 99
    SF_ot_Zright = Mrs_Zright / Mot_Zright if Mot_Zright > 0 else 99
    SF_ot_min_X  = min(SF_ot_Xleft, SF_ot_Xright)
    SF_ot_min_Z  = min(SF_ot_Zleft, SF_ot_Zright)
    SF_ot_overall = min(SF_ot_min_X, SF_ot_min_Z)

    # Sliding
    F_shear_x  = abs(L_DATA['Fx'])
    F_shear_z  = abs(L_DATA['Fz'])
    Adh_x = mu * P_net           # adhesive/friction resistance
    Adh_z = mu * P_net
    SR_x  = Adh_x + coh + Ppx   # total sliding resistance X
    SR_z  = Adh_z + coh + Ppz   # total sliding resistance Z
    SF_sl_x = SR_x / F_shear_x if F_shear_x > 0 else 99
    SF_sl_z = SR_z / F_shear_z if F_shear_z > 0 else 99
    SF_sl_overall = min(SF_sl_x, SF_sl_z)

    # Uplift
    SF_uplift = Wt_total / abs(P_applied) if P_applied < 0 else 99

    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRY CHIPS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="geom-row">
      <div class="geom-chip"><div class="glabel">Footing Lx</div><div class="gval">{Lx} ft</div></div>
      <div class="geom-chip"><div class="glabel">Footing Lz</div><div class="gval">{Lz} ft</div></div>
      <div class="geom-chip"><div class="glabel">Thickness T</div><div class="gval">{T} ft</div></div>
      <div class="geom-chip"><div class="glabel">Depth D</div><div class="gval">{D} ft</div></div>
      <div class="geom-chip"><div class="glabel">Col cx×cz</div><div class="gval">{cx}×{cz}</div></div>
      <div class="geom-chip"><div class="glabel">Area Ftg</div><div class="gval">{Area_ftg:.1f} ft²</div></div>
      <div class="geom-chip"><div class="glabel">Allow. q</div><div class="gval">{qa} ksf</div></div>
      <div class="geom-chip"><div class="glabel">Load Case</div><div class="gval">{L_DATA['LC']}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTIVE SUMMARY KPI ROW
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">🏁 Verification Summary</div>', unsafe_allow_html=True)

    qr  = "pass" if q_ratio <= 1.0 else "fail"
    slr = "pass" if SF_sl_overall >= sf_sliding_min else "fail"
    otr = "pass" if SF_ot_overall >= sf_ot_min else "fail"
    upr = "pass" if SF_uplift >= sf_uplift_min else "fail"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card {qr}">
        <div class="kpi-label">Max Bearing Pressure</div>
        <div class="kpi-value">{q_max:.3f}</div>
        <div class="kpi-unit">{p_unit} (Allow. {qa})</div>
        <div class="kpi-status-{'pass' if qr=='pass' else 'fail'}">{'✓ PASS' if qr=='pass' else '✗ FAIL'} · Ratio {q_ratio:.2f}</div>
      </div>
      <div class="kpi-card {slr}">
        <div class="kpi-label">Sliding Safety Factor</div>
        <div class="kpi-value">{SF_sl_overall:.2f}</div>
        <div class="kpi-unit">Min Required {sf_sliding_min}</div>
        <div class="kpi-status-{'pass' if slr=='pass' else 'fail'}">{'✓ PASS' if slr=='pass' else '✗ FAIL'}</div>
      </div>
      <div class="kpi-card {otr}">
        <div class="kpi-label">Overturning SF</div>
        <div class="kpi-value">{SF_ot_overall:.2f}</div>
        <div class="kpi-unit">Min Required {sf_ot_min}</div>
        <div class="kpi-status-{'pass' if otr=='pass' else 'fail'}">{'✓ PASS' if otr=='pass' else '✗ FAIL'}</div>
      </div>
      <div class="kpi-card {upr}">
        <div class="kpi-label">Uplift SF</div>
        <div class="kpi-value">{SF_uplift:.2f}</div>
        <div class="kpi-unit">Min Required {sf_uplift_min}</div>
        <div class="kpi-status-{'pass' if upr=='pass' else 'fail'}">{'✓ PASS' if upr=='pass' else '✗ FAIL'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TABS
    # ─────────────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "⚖️ Bearing Capacity",
        "📐 Stability",
        "➡️ Sliding",
        "⬆️ Uplift",
        "🗺️ Bearing Pressure Diagram",
        "🧊 3D Force View",
        "🏗️ Foundation Sketch"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — BEARING CAPACITY
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="sec-header">Bearing Capacity — Self-Weight & Soil Pressure</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A. Component Self-Weights**")
            st.latex(rf"W_{{ftg}} = {Lx}\times{Lz}\times{T}\times{gc:.3f} = {Wt_ftg:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"W_{{ped}} = {cx}\times{cz}\times{Hp:.2f}\times{gc:.3f} = {Wt_ped:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"A_{{soil}} = {Area_ftg:.1f} - {Area_col:.1f} = {Area_ftg-Area_col:.1f}\ \mathrm{{{l_unit}}}^2")
            st.latex(rf"W_{{soil}} = {Area_ftg-Area_col:.1f}\times{Hp:.2f}\times{gs:.3f} = {Wt_soil:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"\Sigma W = {Wt_total:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"P_{{total}} = {P_applied:.2f} + {Wt_total:.2f} = {P_total:.2f}\ \mathrm{{{f_unit}}}")

        with c2:
            st.markdown("**B. Base Moments & Eccentricity**")
            st.latex(rf"M_{{x,base}} = {L_DATA['Mx']:.2f}+({L_DATA['Fz']:.2f}\times{D}) = {Mx_base:.2f}\ \mathrm{{{f_unit}\cdot{l_unit}}}")
            st.latex(rf"M_{{z,base}} = {L_DATA['Mz']:.2f}+({L_DATA['Fx']:.2f}\times{D}) = {Mz_base:.2f}\ \mathrm{{{f_unit}\cdot{l_unit}}}")
            st.latex(rf"e_x = \frac{{|M_{{x,base}}|}}{{P_{{net}}}} = \frac{{{abs(Mx_base):.2f}}}{{{P_net:.2f}}} = {ex:.4f}\ \mathrm{{{l_unit}}}")
            st.latex(rf"e_z = \frac{{|M_{{z,base}}|}}{{P_{{net}}}} = \frac{{{abs(Mz_base):.2f}}}{{{P_net:.2f}}} = {ez:.4f}\ \mathrm{{{l_unit}}}")

        st.markdown("**C. Section Properties**")
        c1b, c2b, c3b, c4b = st.columns(4)
        with c1b: st.latex(rf"S_x=\frac{{L_x L_z^2}}{{6}}={Sx:.2f}\ \mathrm{{{l_unit}^3}}")
        with c2b: st.latex(rf"S_z=\frac{{L_z L_x^2}}{{6}}={Sz:.2f}\ \mathrm{{{l_unit}^3}}")
        with c3b: st.latex(rf"I_x=\frac{{L_x L_z^3}}{{12}}={Ix:.2f}\ \mathrm{{{l_unit}^4}}")
        with c4b: st.latex(rf"I_z=\frac{{L_z L_x^3}}{{12}}={Iz:.2f}\ \mathrm{{{l_unit}^4}}")

        st.markdown("**D. Corner Bearing Pressures**")
        st.latex(rf"q = \frac{{P_{{net}}}}{{A}} \pm \frac{{M_x}}{{S_x}} \pm \frac{{M_z}}{{S_z}} = \frac{{{P_net:.2f}}}{{{Area_ftg:.1f}}} \pm \frac{{{abs(Mx_base):.2f}}}{{{Sx:.2f}}} \pm \frac{{{abs(Mz_base):.2f}}}{{{Sz:.2f}}}")

        corner_df = pd.DataFrame([
            {"Corner": k, "Bearing Pressure (ksf)": round(v, 4),
             "Status": "✓ OK" if v <= qa else "✗ EXCEEDS",
             "Ratio": round(v/qa, 3)}
            for k, v in corners.items()
        ])
        st.markdown(corner_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

        ci = "pass" if q_min >= 0 else "fail"
        st.markdown(f"""<div class="alert-{'pass' if q_min >= 0 else 'info'}">
            {'✅ Full contact — entire footing in compression' if q_min >= 0
             else f'⚠️ Partial contact — {contact_area:.1f}% footing area effective. Neutral axis detected.'}
        </div>""", unsafe_allow_html=True)

        # Summary row
        st.markdown(f"""
        <div class="math-panel">
          <div class="label">Bearing Capacity Result</div>
          <div class="formula">q_max = {q_max:.4f} {p_unit}</div>
          <div class="formula">q_min = {q_min:.4f} {p_unit}</div>
          <div class="formula">Allowable q_a = {qa} {p_unit}</div>
          <div class="result">Utilisation Ratio = {q_ratio:.3f} → {'PASS ✓' if q_ratio <= 1.0 else 'FAIL ✗'}</div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — STABILITY
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="sec-header">Overturning Stability</div>', unsafe_allow_html=True)

        st.markdown("**Overturning & Resisting Moments — All Edges**")

        ot_df = pd.DataFrame([
            {"Direction": "X Dir – Left Edge",  "OT Moment (kip-ft)": round(Mot_Xleft,3),  "Resist Moment (kip-ft)": round(Mrs_Xleft,3),  "SF": round(SF_ot_Xleft,4),  "Status": "✓ PASS" if SF_ot_Xleft>=sf_ot_min else "✗ FAIL"},
            {"Direction": "X Dir – Right Edge", "OT Moment (kip-ft)": round(Mot_Xright,3), "Resist Moment (kip-ft)": round(Mrs_Xright,3), "SF": round(SF_ot_Xright,4), "Status": "✓ PASS" if SF_ot_Xright>=sf_ot_min else "✗ FAIL"},
            {"Direction": "Z Dir – Left Edge",  "OT Moment (kip-ft)": round(Mot_Zleft,3),  "Resist Moment (kip-ft)": round(Mrs_Zleft,3),  "SF": round(SF_ot_Zleft,4),  "Status": "✓ PASS" if SF_ot_Zleft>=sf_ot_min else "✗ FAIL"},
            {"Direction": "Z Dir – Right Edge", "OT Moment (kip-ft)": round(Mot_Zright,3), "Resist Moment (kip-ft)": round(Mrs_Zright,3), "SF": round(SF_ot_Zright,4), "Status": "✓ PASS" if SF_ot_Zright>=sf_ot_min else "✗ FAIL"},
        ])
        st.markdown(ot_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.latex(rf"SF_{{OT,X}} = \min({SF_ot_Xleft:.4f},\ {SF_ot_Xright:.4f}) = {SF_ot_min_X:.4f}")
            st.latex(rf"SF_{{OT,Z}} = \min({SF_ot_Zleft:.4f},\ {SF_ot_Zright:.4f}) = {SF_ot_min_Z:.4f}")
            st.latex(rf"SF_{{OT,governing}} = \min({SF_ot_min_X:.4f},\ {SF_ot_min_Z:.4f}) = {SF_ot_overall:.4f}")
        with c2:
            st.markdown(f"""
            <div class="math-panel">
              <div class="label">Min Stability Ratio · X Dir</div>
              <div class="result">{SF_ot_min_X:.4f} → {'PASS ✓' if SF_ot_min_X>=sf_ot_min else 'FAIL ✗'}</div>
              <div class="label" style="margin-top:10px">Min Stability Ratio · Z Dir</div>
              <div class="result">{SF_ot_min_Z:.4f} → {'PASS ✓' if SF_ot_min_Z>=sf_ot_min else 'FAIL ✗'}</div>
              <div class="label" style="margin-top:10px">Net Allowable Stability Ratio</div>
              <div class="result">{sf_ot_min}</div>
            </div>""", unsafe_allow_html=True)

        # Bar chart
        ot_fig = go.Figure()
        dirs  = ["OT X-Left", "OT X-Right", "OT Z-Left", "OT Z-Right"]
        sfs   = [SF_ot_Xleft, SF_ot_Xright, SF_ot_Zleft, SF_ot_Zright]
        colors = ["#4ade80" if v >= sf_ot_min else "#f87171" for v in sfs]
        ot_fig.add_trace(go.Bar(x=dirs, y=sfs, marker_color=colors, text=[f"{v:.2f}" for v in sfs], textposition="outside"))
        ot_fig.add_hline(y=sf_ot_min, line_dash="dash", line_color="#fbbf24", annotation_text=f"Min SF={sf_ot_min}")
        ot_fig.update_layout(template="plotly_dark", paper_bgcolor="#131b2e", plot_bgcolor="#131b2e",
                             title="Overturning Safety Factors by Edge", height=350,
                             yaxis_title="Safety Factor", showlegend=False)
        st.plotly_chart(ot_fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — SLIDING
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="sec-header">Sliding Resistance</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**X-Direction Sliding**")
            st.latex(rf"V_x = {F_shear_x:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{adhesion,x}} = \mu \cdot P_{{net}} = {mu}\times{P_net:.2f} = {Adh_x:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{cohesion,x}} = {coh:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{passive,x}} = {Ppx:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"SR_x = {Adh_x:.2f}+{coh:.2f}+{Ppx:.2f} = {SR_x:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"SF_{{sliding,x}} = \frac{{{SR_x:.2f}}}{{{F_shear_x:.2f}}} = {SF_sl_x:.4f}")

        with c2:
            st.markdown("**Z-Direction Sliding**")
            st.latex(rf"V_z = {F_shear_z:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{adhesion,z}} = \mu \cdot P_{{net}} = {mu}\times{P_net:.2f} = {Adh_z:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{cohesion,z}} = {coh:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"R_{{passive,z}} = {Ppz:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"SR_z = {Adh_z:.2f}+{coh:.2f}+{Ppz:.2f} = {SR_z:.2f}\ \mathrm{{{f_unit}}}")
            st.latex(rf"SF_{{sliding,z}} = \frac{{{SR_z:.2f}}}{{{F_shear_z:.2f}}} = {SF_sl_z:.4f}")

        sl_df = pd.DataFrame([
            {"Direction": "X", "Shear (kip)": round(F_shear_x,2), "Adhesion (kip)": round(Adh_x,2),
             "Cohesion (kip)": round(coh,2), "Passive (kip)": round(Ppx,2),
             "Total Resist (kip)": round(SR_x,2), "SF": round(SF_sl_x,4),
             "Status": "✓ PASS" if SF_sl_x>=sf_sliding_min else "✗ FAIL"},
            {"Direction": "Z", "Shear (kip)": round(F_shear_z,2), "Adhesion (kip)": round(Adh_z,2),
             "Cohesion (kip)": round(coh,2), "Passive (kip)": round(Ppz,2),
             "Total Resist (kip)": round(SR_z,2), "SF": round(SF_sl_z,4),
             "Status": "✓ PASS" if SF_sl_z>=sf_sliding_min else "✗ FAIL"},
        ])
        st.markdown(sl_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="math-panel" style="margin-top:14px">
          <div class="label">Governing Sliding SF</div>
          <div class="result">SF_sliding = min({SF_sl_x:.4f}, {SF_sl_z:.4f}) = {SF_sl_overall:.4f}
          → {'PASS ✓' if SF_sl_overall >= sf_sliding_min else 'FAIL ✗'} (Min = {sf_sliding_min})</div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — UPLIFT
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="sec-header">Uplift Verification</div>', unsafe_allow_html=True)

        uplift_df = pd.DataFrame([{
            "Description": L_DATA["LC"],
            "Uplift SF": round(SF_uplift, 4),
            "Net Allow. SF": sf_uplift_min,
            "Applied Axial (kip)": round(P_applied, 2),
            "Element Self-Wt (kip)": round(Wt_ped, 2),
            "Self Weight (kip)": round(Wt_ftg, 2),
            "Soil Self-Wt (kip)": round(Wt_soil, 2),
            "Total Resist (kip)": round(Wt_total, 2),
            "Status": "✓ PASS" if SF_uplift >= sf_uplift_min else "✗ FAIL"
        }])
        st.markdown(uplift_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

        st.latex(rf"SF_{{uplift}} = \frac{{\Sigma W}}{{|P_{{applied}}|}} = \frac{{{Wt_total:.2f}}}{{{abs(P_applied):.2f}}} = {SF_uplift:.4f}")
        st.markdown(f"""<div class="{'alert-pass' if SF_uplift >= sf_uplift_min else 'alert-fail'}">
            {'✅ Uplift PASS' if SF_uplift >= sf_uplift_min else '✗ Uplift FAIL'} — SF = {SF_uplift:.4f} vs. minimum {sf_uplift_min}</div>""",
            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 — BEARING PRESSURE DIAGRAM (matches reference image)
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="sec-header">Bearing Pressure Diagram</div>', unsafe_allow_html=True)

        # ── Full 2D bearing pressure plan diagram ──
        fig_bp = go.Figure()

        # Footing outline
        fx = [-Lx/2, Lx/2, Lx/2, -Lx/2, -Lx/2]
        fz = [-Lz/2, -Lz/2, Lz/2, Lz/2, -Lz/2]
        fig_bp.add_trace(go.Scatter(x=fx, y=fz, mode='lines',
            line=dict(color='#38bdf8', width=3), name='Footing Perimeter', fill='toself',
            fillcolor='rgba(56,189,248,0.07)'))

        # Column / pedestal
        px_c = [-cx/2, cx/2, cx/2, -cx/2, -cx/2]
        pz_c = [-cz/2, -cz/2, cz/2, cz/2, -cz/2]
        fig_bp.add_trace(go.Scatter(x=px_c, y=pz_c, mode='lines',
            line=dict(color='#fbbf24', width=2, dash='dot'), name='Column'))

        # Stress-distribution heatmap overlay
        fig_bp.add_trace(go.Contour(
            z=Q_field, x=x_lin, y=z_lin,
            colorscale=[[0,'rgba(0,0,0,0)'],[0.001,'#1e3a8a'],[0.4,'#3b82f6'],
                        [0.7,'#fbbf24'],[1.0,'#ef4444']],
            showscale=True, opacity=0.75,
            colorbar=dict(title=dict(text=f"q ({p_unit})", side='right'),
                         tickfont=dict(color='#94a3b8'), titlefont=dict(color='#94a3b8')),
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            name='Bearing Pressure'
        ))

        # Neutral axis line (zero-pressure boundary)
        if NA_x_intercept is not None and NA_z_intercept is not None:
            if abs(NA_x_intercept) <= Lx/2 and abs(NA_z_intercept) <= Lz/2:
                # clamp to footing boundary
                na_pts_x = [max(-Lx/2, min(Lx/2, NA_x_intercept)), -Lx/2]
                na_pts_z = [0.0, NA_z_intercept]
                fig_bp.add_trace(go.Scatter(x=na_pts_x, y=na_pts_z, mode='lines',
                    line=dict(color='white', width=2, dash='dash'), name='Neutral Axis'))

        # Corner pressure annotations
        cx_pts = [ Lx/2, -Lx/2,  Lx/2, -Lx/2]
        cz_pts = [ Lz/2,  Lz/2, -Lz/2, -Lz/2]
        cp_vals = [corners["C1 (+x,+z)"], corners["C2 (-x,+z)"],
                   corners["C3 (+x,-z)"], corners["C4 (-x,-z)"]]
        for xi, zi, qi in zip(cx_pts, cz_pts, cp_vals):
            color = "#ef4444" if qi == q_max else "#e2e8f0"
            fig_bp.add_annotation(x=xi, y=zi, text=f"<b>{qi:.4f}</b>",
                showarrow=False, font=dict(size=13, color=color),
                xanchor="center", yanchor="middle",
                bgcolor="rgba(0,0,0,0.6)", borderpad=3)

        # Load eccentricity dot
        # eccentricity location: if Mz pushes in +X then ex is negative offset
        ecc_x = -ez if Mz_base > 0 else ez
        ecc_z = -ex if Mx_base > 0 else ex
        fig_bp.add_trace(go.Scatter(x=[ecc_x], y=[ecc_z], mode='markers',
            marker=dict(color='#4ade80', size=14, symbol='circle',
                        line=dict(color='white', width=2)),
            name=f'Load Eccentricity ({ecc_x:.3f}, {ecc_z:.3f})'))

        # Eccentricity annotation lines
        fig_bp.add_shape(type='line', x0=ecc_x, y0=-Lz/2, x1=ecc_x, y1=Lz/2,
            line=dict(color='#4ade80', width=1, dash='dash'))
        fig_bp.add_shape(type='line', x0=-Lx/2, y0=ecc_z, x1=Lx/2, y1=ecc_z,
            line=dict(color='#4ade80', width=1, dash='dash'))

        # Dimension arrows
        fig_bp.add_annotation(x=0, y=-Lz/2-0.3, text=f"← {Lx} ft →",
            showarrow=False, font=dict(size=12, color='#7dd3fc'))
        fig_bp.add_annotation(x=-Lx/2-0.5, y=0, text=f"{Lz} ft",
            showarrow=False, font=dict(size=12, color='#7dd3fc'), textangle=-90)

        # Eccentricity dimension
        fig_bp.add_annotation(
            x=0, y=Lz/2+0.35,
            text=f"e_x = {ecc_x:.4f} ft",
            showarrow=False, font=dict(size=11, color='#4ade80'))

        fig_bp.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#131b2e",
            xaxis=dict(title=f"X ({l_unit})", scaleanchor="y", scaleratio=1,
                       range=[-Lx/2-1, Lx/2+1], showgrid=True, gridcolor="#1e2535",
                       zeroline=True, zerolinecolor="#334155"),
            yaxis=dict(title=f"Z ({l_unit})", range=[-Lz/2-1, Lz/2+1],
                       showgrid=True, gridcolor="#1e2535",
                       zeroline=True, zerolinecolor="#334155"),
            height=700, legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#e2e8f0")),
            title=dict(text=f"<b>Bearing Pressure Distribution</b>  ·  {L_DATA['LC']}  ·  "
                           f"Contact Area = {contact_area:.1f}%",
                       font=dict(color="#bfdbfe", size=15), x=0.5)
        )
        st.plotly_chart(fig_bp, use_container_width=True)

        # Tabular results
        bcap_df = pd.DataFrame([{
            "Max q (ksf)": round(q_max, 4),
            "Min q (ksf)": round(q_min, 4),
            "V1 Bearing (ksf)": round(v1_bearing, 4),
            "V2 Bearing (ksf)": round(v2_bearing, 4),
            "Allow. q (ksf)": qa,
            "Ecc X (ft)": round(ex, 4),
            "Ecc Z (ft)": round(ez, 4),
            "NA Intercept X": round(NA_x_intercept, 4) if NA_x_intercept else "∞",
            "NA Intercept Z": round(NA_z_intercept, 4) if NA_z_intercept else "∞",
            "Eff Comp Area (ft²)": round(contact_area/100*Area_ftg, 4),
            "Contact (%)": round(contact_area, 2),
        }])
        st.markdown(bcap_df.T.to_html(header=False, classes="eng-table"), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6 — 3D FORCE VIEW
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="sec-header">3D Applied Load Visualization</div>', unsafe_allow_html=True)

        fig3d = go.Figure()

        # ── Footing slab ──
        ftg_x = [-Lx/2, Lx/2, Lx/2, -Lx/2, -Lx/2, Lx/2, Lx/2, -Lx/2]
        ftg_y = [0, 0, 0, 0, -T, -T, -T, -T]
        ftg_z = [-Lz/2, -Lz/2, Lz/2, Lz/2, -Lz/2, -Lz/2, Lz/2, Lz/2]
        fig3d.add_trace(go.Mesh3d(
            x=ftg_x, y=ftg_y, z=ftg_z,
            i=[0,0,0,1,4,4,4,5], j=[1,2,4,5,5,6,7,6], k=[2,3,5,6,6,7,3,7],
            color='#1e40af', opacity=0.45, name='Footing',
            lighting=dict(ambient=0.6, diffuse=0.8)
        ))

        # ── Pedestal ──
        ped_x = [-cx/2, cx/2, cx/2, -cx/2, -cx/2, cx/2, cx/2, -cx/2]
        ped_y = [0, 0, 0, 0, Hp, Hp, Hp, Hp]
        ped_z = [-cz/2, -cz/2, cz/2, cz/2, -cz/2, -cz/2, cz/2, cz/2]
        fig3d.add_trace(go.Mesh3d(
            x=ped_x, y=ped_y, z=ped_z,
            i=[0,0,0,1,4,4,4,5], j=[1,2,4,5,5,6,7,6], k=[2,3,5,6,6,7,3,7],
            color='#475569', opacity=0.85, name='Pedestal',
            lighting=dict(ambient=0.6, diffuse=0.8)
        ))

        # ── Soil layer indicator ──
        fig3d.add_trace(go.Mesh3d(
            x=ftg_x, y=[0,0,0,0,-Hp,-Hp,-Hp,-Hp], z=ftg_z,
            i=[0,0,0,1,4,4,4,5], j=[1,2,4,5,5,6,7,6], k=[2,3,5,6,6,7,3,7],
            color='#854d0e', opacity=0.20, name='Soil', showlegend=True
        ))

        # ── Helper: draw arrow ──
        def arrow3d(x0, y0, z0, dx, dy, dz, color, label, scale=1.0):
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            if length < 1e-6: return
            tip_x = x0 + dx * scale
            tip_y = y0 + dy * scale
            tip_z = z0 + dz * scale
            fig3d.add_trace(go.Scatter3d(
                x=[x0, tip_x], y=[y0, tip_y], z=[z0, tip_z],
                mode='lines+text',
                line=dict(color=color, width=8),
                text=["", f"<b>{label}</b>"],
                textfont=dict(size=13, color=color),
                textposition="top center",
                name=label, showlegend=True
            ))
            # Cone as arrowhead
            fig3d.add_trace(go.Cone(
                x=[tip_x], y=[tip_y], z=[tip_z],
                u=[dx/length*0.3], v=[dy/length*0.3], w=[dz/length*0.3],
                colorscale=[[0, color], [1, color]],
                showscale=False, sizemode='absolute', sizeref=0.35,
                name='', showlegend=False
            ))

        sc = 1.2  # visual scale for arrows
        top_y = Hp

        # Vertical force Fy (green) — dominant
        arrow3d(0, top_y, 0,  0, 1, 0, '#4ade80', f"Fy={L_DATA['Fy']:.2f} kip", scale=sc*1.2)
        # Horizontal Fx (red)
        arrow3d(0, top_y, 0,  1, 0, 0, '#f87171', f"Fx={L_DATA['Fx']:.2f} kip", scale=sc)
        # Horizontal Fz (blue)
        arrow3d(0, top_y, 0,  0, 0, 1, '#60a5fa', f"Fz={L_DATA['Fz']:.2f} kip", scale=sc)

        # Moment arcs (represented as curved annotations using scatter3d rings)
        def moment_ring(axis, center_y, radius, color, label, n=40):
            theta = np.linspace(0, 1.5*np.pi, n)
            if axis == 'x':
                rx = np.zeros(n)
                ry = center_y + radius * np.sin(theta)
                rz = radius * np.cos(theta)
            elif axis == 'z':
                rx = radius * np.cos(theta)
                ry = center_y + radius * np.sin(theta)
                rz = np.zeros(n)
            else:
                rx = radius * np.cos(theta)
                ry = center_y * np.ones(n)
                rz = radius * np.sin(theta)
            fig3d.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz, mode='lines',
                line=dict(color=color, width=5, dash='dash'),
                name=label, showlegend=True
            ))

        moment_ring('x', top_y/2, 0.7, '#a78bfa', f"Mx={L_DATA['Mx']:.2f} kip·ft")
        moment_ring('z', top_y/2, 0.7, '#fb923c', f"Mz={L_DATA['Mz']:.2f} kip·ft")

        # Ground plane grid
        gx, gz = np.meshgrid(np.linspace(-Lx/2, Lx/2, 6), np.linspace(-Lz/2, Lz/2, 6))
        fig3d.add_trace(go.Surface(
            x=gx, z=gz, y=np.full_like(gx, -T),
            colorscale=[[0,'#1e3a5f'],[1,'#1e3a5f']],
            opacity=0.35, showscale=False, name='Soil Base'
        ))

        # Dimension labels
        fig3d.add_trace(go.Scatter3d(
            x=[0], y=[-T/2], z=[Lz/2+0.3],
            mode='text', text=[f"Lz={Lz}ft"],
            textfont=dict(size=12, color='#7dd3fc'), showlegend=False
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[Lx/2+0.3], y=[-T/2], z=[0],
            mode='text', text=[f"Lx={Lx}ft"],
            textfont=dict(size=12, color='#7dd3fc'), showlegend=False
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[-Lx/2-0.5], y=[Hp/2], z=[0],
            mode='text', text=[f"D={D}ft"],
            textfont=dict(size=12, color='#7dd3fc'), showlegend=False
        ))

        fig3d.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117',
            scene=dict(
                bgcolor='#0f1117',
                xaxis=dict(title='X (ft)', gridcolor='#1e2535', showbackground=False),
                yaxis=dict(title='Y (ft)', gridcolor='#1e2535', showbackground=False),
                zaxis=dict(title='Z (ft)', gridcolor='#1e2535', showbackground=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.8, y=1.4, z=1.5))
            ),
            legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='#e2e8f0', size=11)),
            height=680,
            title=dict(text=f"<b>3D Foundation — Applied Forces & Moments</b> · {L_DATA['LC']}",
                       font=dict(color='#bfdbfe', size=15), x=0.5)
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # Load summary table
        load_df = pd.DataFrame([{
            "LC": L_DATA['LC'],
            "Fx (kip)": L_DATA['Fx'], "Fy (kip)": L_DATA['Fy'], "Fz (kip)": L_DATA['Fz'],
            "Mx (kip-ft)": L_DATA['Mx'], "My (kip-ft)": L_DATA['My'], "Mz (kip-ft)": L_DATA['Mz']
        }])
        st.markdown(load_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 7 — FOUNDATION SKETCH (Elevation + Plan)
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="sec-header">Foundation Engineering Sketch — Elevation & Plan</div>', unsafe_allow_html=True)

        fig_sk = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Footing Elevation View", "Footing Plan View (Top)"),
            horizontal_spacing=0.12
        )
        # ── Elevation ──
        # Ground line
        fig_sk.add_shape(type='line', x0=-Lx/2-0.5, y0=0, x1=Lx/2+0.5, y1=0,
            line=dict(color='#6b7280', width=2, dash='dot'), row=1, col=1)

        # Footing rectangle (elevation)
        fig_sk.add_shape(type='rect', x0=-Lx/2, y0=-T, x1=Lx/2, y1=0,
            line=dict(color='#38bdf8', width=2), fillcolor='rgba(56,189,248,0.15)', row=1, col=1)

        # Pedestal rectangle (elevation)
        fig_sk.add_shape(type='rect', x0=-cx/2, y0=0, x1=cx/2, y1=Hp,
            line=dict(color='#fbbf24', width=2), fillcolor='rgba(251,191,36,0.2)', row=1, col=1)

        # Soil hatch (elevation)
        for yh in np.arange(-T+T/8, 0, T/5):
            fig_sk.add_shape(type='line', x0=-Lx/2, y0=yh, x1=Lx/2, y1=yh,
                line=dict(color='#854d0e', width=0.5, dash='dot'), row=1, col=1)

        # Grade elevation hatch
        for xh in np.arange(-Lx/2-0.5, Lx/2+0.6, 0.4):
            fig_sk.add_shape(type='line', x0=xh, y0=0, x1=xh-0.2, y1=-0.2,
                line=dict(color='#6b7280', width=1), row=1, col=1)

        # Dimension annotations — elevation
        # Width dimension
        fig_sk.add_annotation(x=0, y=-T-0.3, text=f"← {Lx} ft →",
            showarrow=False, font=dict(size=11, color='#7dd3fc'), row=1, col=1)
        # Depth
        fig_sk.add_annotation(x=Lx/2+0.6, y=-T/2, text=f"{T} ft",
            showarrow=False, font=dict(size=11, color='#7dd3fc'), textangle=-90, row=1, col=1)
        # Soil cover
        fig_sk.add_annotation(x=Lx/2+0.6, y=Hp/2, text=f"Soil Cover\n{Hp:.1f} ft",
            showarrow=False, font=dict(size=10, color='#a3a3a3'), row=1, col=1)
        # GL label
        fig_sk.add_annotation(x=-Lx/2-0.3, y=0.1, text="GL. EL. 0 ft",
            showarrow=False, font=dict(size=9, color='#6b7280'), row=1, col=1)

        # Force arrows on elevation
        asc = Hp * 0.4
        # Fy arrow
        fig_sk.add_shape(type='line', x0=0, y0=Hp+asc, x1=0, y1=Hp,
            line=dict(color='#4ade80', width=3), row=1, col=1)
        fig_sk.add_annotation(x=0.3, y=Hp+asc*0.7, text=f"Fy={L_DATA['Fy']:.2f}",
            showarrow=False, font=dict(size=10, color='#4ade80'), row=1, col=1)
        # Fx arrow
        fig_sk.add_shape(type='line', x0=-asc*0.8, y0=Hp, x1=0, y1=Hp,
            line=dict(color='#f87171', width=3), row=1, col=1)
        fig_sk.add_annotation(x=-asc*0.9, y=Hp+0.2, text=f"Fx={L_DATA['Fx']:.2f}",
            showarrow=False, font=dict(size=10, color='#f87171'), row=1, col=1)

        # ── Plan View ──
        # Footing outline
        fig_sk.add_shape(type='rect', x0=-Lx/2, y0=-Lz/2, x1=Lx/2, y1=Lz/2,
            line=dict(color='#38bdf8', width=2.5), fillcolor='rgba(56,189,248,0.10)', row=1, col=2)

        # Column/pedestal outline
        fig_sk.add_shape(type='rect', x0=-cx/2, y0=-cz/2, x1=cx/2, y1=cz/2,
            line=dict(color='#fbbf24', width=2), fillcolor='rgba(251,191,36,0.25)', row=1, col=2)

        # Centreline dashes
        fig_sk.add_shape(type='line', x0=0, y0=-Lz/2-0.3, x1=0, y1=Lz/2+0.3,
            line=dict(color='#60a5fa', width=1, dash='dash'), row=1, col=2)
        fig_sk.add_shape(type='line', x0=-Lx/2-0.3, y0=0, x1=Lx/2+0.3, y1=0,
            line=dict(color='#60a5fa', width=1, dash='dash'), row=1, col=2)

        # Axis arrows
        fig_sk.add_annotation(x=Lx/2+0.5, y=0, text="→ X", showarrow=False,
            font=dict(size=11, color='#60a5fa'), row=1, col=2)
        fig_sk.add_annotation(x=0, y=Lz/2+0.5, text="↑ Z", showarrow=False,
            font=dict(size=11, color='#60a5fa'), row=1, col=2)

        # Eccentricity dot (plan)
        fig_sk.add_trace(go.Scatter(x=[ecc_x], y=[ecc_z], mode='markers',
            marker=dict(color='#4ade80', size=12, symbol='circle-open',
                        line=dict(color='#4ade80', width=3)),
            name='Load Eccentricity'), row=1, col=2)

        # Neutral axis (plan)
        if NA_x_intercept is not None and abs(NA_x_intercept) <= Lx/2:
            if NA_z_intercept is not None and abs(NA_z_intercept) <= Lz/2:
                fig_sk.add_trace(go.Scatter(
                    x=[NA_x_intercept, -Lx/2], y=[0.0, NA_z_intercept],
                    mode='lines', line=dict(color='white', width=2, dash='dash'),
                    name='Neutral Axis'), row=1, col=2)

        # Plan dimensions
        fig_sk.add_annotation(x=0, y=-Lz/2-0.5, text=f"← {Lx} ft →",
            showarrow=False, font=dict(size=11, color='#7dd3fc'), row=1, col=2)
        fig_sk.add_annotation(x=-Lx/2-0.7, y=0, text=f"{Lz} ft",
            showarrow=False, font=dict(size=11, color='#7dd3fc'), textangle=-90, row=1, col=2)

        # Corner pressure labels (plan)
        plan_cx = [ Lx/2, -Lx/2,  Lx/2, -Lx/2]
        plan_cz = [ Lz/2,  Lz/2, -Lz/2, -Lz/2]
        for xi, zi, qi in zip(plan_cx, plan_cz, cp_vals):
            clr = "#ef4444" if qi == q_max else "#94a3b8"
            fig_sk.add_annotation(x=xi, y=zi,
                text=f"<b>{qi:.3f}</b>",
                showarrow=False, font=dict(size=11, color=clr),
                bgcolor="rgba(0,0,0,0.7)", row=1, col=2)

        # Dummy scatter for axes
        fig_sk.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)

        fig_sk.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#131b2e',
            height=600,
            title=dict(text=f"<b>Foundation Sketch</b> — F1 · Rectangular · Soil Supported",
                       font=dict(color='#bfdbfe', size=14), x=0.5),
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='#e2e8f0'))
        )

        # Fix axis ratios
        fig_sk.update_xaxes(scaleanchor=None, showgrid=True, gridcolor='#1e2535',
                            zeroline=True, zerolinecolor='#334155', row=1, col=1)
        fig_sk.update_yaxes(showgrid=True, gridcolor='#1e2535',
                            zeroline=True, zerolinecolor='#334155', row=1, col=1)
        fig_sk.update_xaxes(scaleanchor="y2", scaleratio=1, showgrid=True, gridcolor='#1e2535',
                            zeroline=True, zerolinecolor='#334155', row=1, col=2)
        fig_sk.update_yaxes(showgrid=True, gridcolor='#1e2535',
                            zeroline=True, zerolinecolor='#334155', row=1, col=2)

        st.plotly_chart(fig_sk, use_container_width=True)

        # Footing geometry table
        st.markdown("**Footing Geometry Parameters**")
        geom_df = pd.DataFrame([{
            "Name": "F1", "Shape": "Rectangle", "Support": "Soil Supported",
            "Size X (ft)": Lx, "Size Z (ft)": Lz, "Thickness (ft)": T,
            "Depth Below Grade (ft)": D, "Soil Cover (ft)": Hp,
            "Col cx (ft)": cx, "Col cz (ft)": cz,
            "Allow. Bearing (psf)": int(qa*1000)
        }])
        st.markdown(geom_df.to_html(index=False, classes="eng-table"), unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="alert-info">
        ℹ️ Paste a valid load case row above to begin calculations.<br>
        <b>Format:</b> LC_Name &nbsp;|&nbsp; Fx &nbsp;|&nbsp; Fy &nbsp;|&nbsp; Fz &nbsp;|&nbsp; Mx &nbsp;|&nbsp; My &nbsp;|&nbsp; Mz
    </div>
    """, unsafe_allow_html=True)
