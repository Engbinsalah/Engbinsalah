"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Paste LC table directly from STAAD output
Units: kip / kip-ft throughout
LRFD (prefix 5-) → Strength Design (ACI 318-19)
ASD  (prefix 4-) → Bearing Check
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, re

st.set_page_config(page_title="Foundation Design", page_icon="🏗️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e2e8f0}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#16213e);border:1px solid #2d3561;
  border-radius:10px;padding:14px;text-align:center;margin:3px}
.metric-value{font-size:1.6rem;font-weight:700;color:#4fc3f7}
.metric-label{font-size:.72rem;color:#90a4ae;text-transform:uppercase;letter-spacing:.8px;margin-top:2px}
.sec-hdr{background:linear-gradient(90deg,#1565c0,#0d47a1);color:#fff;
  padding:7px 14px;border-radius:6px;font-weight:700;font-size:1rem;margin:10px 0 6px}
.info-box{background:#0d2137;border-left:4px solid #1565c0;
  padding:9px 13px;border-radius:4px;margin:5px 0;font-size:.85rem}
.lrfd-tag{background:#1565c0;color:#fff;border-radius:4px;padding:2px 8px;
  font-size:.75rem;font-weight:700}
.asd-tag{background:#2e7d32;color:#fff;border-radius:4px;padding:2px 8px;
  font-size:.75rem;font-weight:700}
div[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2a3a}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;
  border:none;border-radius:6px;font-weight:700;padding:10px 28px;width:100%}
.stTextArea textarea{background:#0d1a26;color:#e2e8f0;border:1px solid #1e3a5f;
  font-family:monospace;font-size:.8rem}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-hdr">{t}</div>', unsafe_allow_html=True)
def info(t): st.markdown(f'<div class="info-box">{t}</div>', unsafe_allow_html=True)
def mcard(lbl, val, col):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PARSER
# ─────────────────────────────────────────────────────────────────────────────
def classify_lc(name: str) -> str:
    n = name.strip().lstrip("[")
    if re.match(r"5[-–]", n): return "LRFD"
    if re.match(r"4[-–]", n): return "ASD"
    if re.search(r"1\.[2-9]|1\.4|0\.9\(", n): return "LRFD"
    return "ASD"

def parse_staad_paste(text: str) -> pd.DataFrame:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line: continue
        if re.search(r"\bLC\b|\bFX\b|\bNode\b|\bkip\b|\bHorizontal\b", line, re.I): continue
        parts = re.split(r"\t|  +", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 7: continue
        offset = 1 if re.match(r"^\d+$", parts[0]) else 0
        if len(parts) < offset + 7: continue
        lc = parts[offset].strip("[]")
        try:
            vals = [float(parts[offset+i+1]) for i in range(6)]
        except ValueError:
            continue
        rows.append([lc] + vals)
    if not rows: return None
    return pd.DataFrame(rows, columns=["LC","FX","FY","FZ","MX","MY","MZ"])

def parse_excel(uploaded) -> pd.DataFrame:
    raw = pd.read_excel(uploaded, header=None)
    hrow = 0
    for i, row in raw.iterrows():
        rs = " ".join([str(v).lower() for v in row.values])
        if any(k in rs for k in ["lc","fx","fy","node"]): hrow = i; break
    df = pd.read_excel(uploaded, header=hrow)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["lc","case","combo","load","l/c"]): rename[c]="LC"
        elif cl in ("fx","fx (kip)"): rename[c]="FX"
        elif cl in ("fy","fy (kip)"): rename[c]="FY"
        elif cl in ("fz","fz (kip)"): rename[c]="FZ"
        elif cl in ("mx","mx (kip-ft)"): rename[c]="MX"
        elif cl in ("my","my (kip-ft)"): rename[c]="MY"
        elif cl in ("mz","mz (kip-ft)"): rename[c]="MZ"
    df = df.rename(columns=rename)
    for col in ["LC","FX","FY","FZ","MX","MY","MZ"]:
        if col not in df.columns: df[col] = 0.0
    return df[["LC","FX","FY","FZ","MX","MY","MZ"]].dropna(subset=["FY"])

def add_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = df["LC"].apply(classify_lc)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL COMBO
# ─────────────────────────────────────────────────────────────────────────────
def critical_combos(df: pd.DataFrame):
    def score(row):
        M = math.sqrt(float(row["MX"])**2 + float(row["MZ"])**2)
        F = math.sqrt(float(row["FX"])**2 + float(row["FZ"])**2)
        return abs(float(row["FY"])) + M + F
    lrfd = df[df["Type"]=="LRFD"].copy()
    asd  = df[df["Type"]=="ASD"].copy()
    lr = lrfd.loc[lrfd.apply(score, axis=1).idxmax()] if len(lrfd) else None
    ar = asd.loc[asd.apply(score, axis=1).idxmax()]   if len(asd)  else None
    return lr, ar

# ─────────────────────────────────────────────────────────────────────────────
# 3D FIGURE  (kip / kip-ft labels)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d(df, Lx_ft, Lz_ft, col_w_ft, col_d_ft):
    # convert ft → display units (keep as ft for geometry)
    hw,hd,ht = Lx_ft/2, Lz_ft/2, Lx_ft*0.15
    I=[0,0,0,4,4,4,0,2,2,0,6,4]; J=[1,2,3,5,6,7,1,3,6,4,7,5]; K=[2,3,0,6,7,4,5,7,5,6,2,1]
    vx=[-hw,hw,hw,-hw,-hw,hw,hw,-hw]; vy=[-hd,-hd,hd,hd,-hd,-hd,hd,hd]; vz=[-ht]*4+[0]*4
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=I,j=J,k=K,color="#37474f",opacity=.6,name="Footing"))
    cw2,cd2,ch = col_w_ft/2, col_d_ft/2, Lx_ft*0.4
    cvx=[-cw2,cw2,cw2,-cw2,-cw2,cw2,cw2,-cw2]
    cvy=[-cd2,-cd2,cd2,cd2,-cd2,-cd2,cd2,cd2]
    cvz=[0]*4+[ch]*4
    fig.add_trace(go.Mesh3d(x=cvx,y=cvy,z=cvz,i=I,j=J,k=K,color="#546e7a",opacity=.85,name="Column"))
    gx=np.linspace(-hw*2.2,hw*2.2,4); gy=np.linspace(-hd*2.2,hd*2.2,4)
    GX,GY=np.meshgrid(gx,gy); GZ=np.zeros_like(GX)-ht
    fig.add_trace(go.Surface(x=GX,y=GY,z=GZ,
        colorscale=[[0,"#1a237e"],[1,"#283593"]],
        opacity=.15,showscale=False,name="Soil"))

    arrow_scale = max(Lx_ft,Lz_ft)*0.55
    lrfd_r, asd_r = critical_combos(df)

    # thin background arrows for all combos
    for _, row in df.iterrows():
        clr = "#1e4976" if row["Type"]=="LRFD" else "#1b4d1b"
        ref = max(abs(float(row["FY"])),abs(float(row["FX"])),abs(float(row["FZ"])),0.01)
        sc  = arrow_scale/ref*0.35
        for dx,dy,dz in [
            (float(row["FX"])*sc, 0, 0),
            (0, float(row["FZ"])*sc, 0),
            (0, 0, -abs(float(row["FY"]))*sc),
        ]:
            if max(abs(dx),abs(dy),abs(dz))<0.05: continue
            fig.add_trace(go.Scatter3d(x=[0,dx],y=[0,dy],z=[ch,ch+dz],
                mode="lines",line=dict(color=clr,width=2),showlegend=False))

    # thick critical arrows
    for crit_row, clr, tag in [(lrfd_r,"#ef5350","LRFD"), (asd_r,"#ffca28","ASD")]:
        if crit_row is None: continue
        fy = float(crit_row["FY"]); fx = float(crit_row["FX"])
        fz_v = float(crit_row["FZ"]); mx = float(crit_row["MX"]); mz = float(crit_row["MZ"])
        ref = max(abs(fy),abs(fx),abs(fz_v),0.01)
        sc  = arrow_scale/ref
        for dx,dy,dz,lbl in [
            (fx*sc,0,0,         f"{tag} Fx={fx:.2f} kip"),
            (0,fz_v*sc,0,       f"{tag} Fz={fz_v:.2f} kip"),
            (0,0,-abs(fy)*sc,   f"{tag} Fy={fy:.2f} kip"),
        ]:
            if max(abs(dx),abs(dy),abs(dz))<0.05: continue
            fig.add_trace(go.Scatter3d(x=[0,dx],y=[0,dy],z=[ch,ch+dz],
                mode="lines",line=dict(color=clr,width=6),name=lbl))
            fig.add_trace(go.Cone(x=[dx],y=[dy],z=[ch+dz],
                u=[dx*.22],v=[dy*.22],w=[dz*.22],
                colorscale=[[0,clr],[1,clr]],showscale=False,
                sizemode="absolute",sizeref=max(Lx_ft,Lz_ft)*.12,showlegend=False))
        for moment,axis,mlbl in [(mx,"x",f"{tag} Mx={mx:.2f} kip-ft"),
                                  (mz,"z",f"{tag} Mz={mz:.2f} kip-ft")]:
            if abs(moment)<0.01: continue
            r=max(Lx_ft,Lz_ft)*0.32; t_=np.linspace(0,1.6*math.pi,50)
            if axis=="x":
                ax_x=np.zeros_like(t_); ax_y=r*np.cos(t_); az_=ch+Lx_ft*.12+r*np.sin(t_)*0.6
            else:
                ax_x=r*np.cos(t_); ax_y=r*np.sin(t_); az_=np.zeros_like(t_)+ch+Lx_ft*.1
            fig.add_trace(go.Scatter3d(x=ax_x,y=ax_y,z=az_,mode="lines",
                line=dict(color=clr,width=3,dash="dot"),name=mlbl))

    # dimension labels
    dz=-ht-.5
    fig.add_trace(go.Scatter3d(x=[-hw,hw],y=[-hd-.5,-hd-.5],z=[dz,dz],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lx={Lx_ft:.1f} ft"],textposition="middle right",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[hw+.5,hw+.5],y=[-hd,hd],z=[dz,dz],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lz={Lz_ft:.1f} ft"],textposition="middle right",showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (ft)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            yaxis=dict(title="Y (ft)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            zaxis=dict(title="Z (ft)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            bgcolor="#0d1117",camera=dict(eye=dict(x=1.7,y=1.7,z=1.2))),
        paper_bgcolor="#0f1117",font=dict(color="#cfd8dc"),
        legend=dict(bgcolor="#13192b",bordercolor="#263238",font=dict(size=10)),
        height=600,margin=dict(l=0,r=0,t=36,b=0),
        title=dict(
            text="3D Applied Loads — 🔴 LRFD Critical  |  🟡 ASD Critical",
            font=dict(color="#90caf9",size=13)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FOUNDATION SIZING  (all kip / kip-ft / ft / in / ksf / ksi)
# ─────────────────────────────────────────────────────────────────────────────
def size_footing(P_ult_kip, Mx_ult_kipft, Mz_ult_kipft,
                 P_svc_kip, Mx_svc_kipft, Mz_svc_kipft,
                 qa_ksf, fc_ksi, fy_ksi, cover_in,
                 col_w_ft, col_d_ft, Df_ft, gc_pcf, gs_pcf):

    gc_kcf = gc_pcf / 1000.0
    gs_kcf = gs_pcf / 1000.0

    # ── 1. Plan size from ASD bearing ────────────────────────────────────
    qa_net_ksf = qa_ksf - gs_kcf * Df_ft
    L = math.sqrt(max(abs(P_svc_kip) / max(qa_net_ksf, 0.01), 0.5))
    L = math.ceil(L * 4) / 4   # round up to nearest 3"
    for _ in range(25):
        Wf = L * L * Df_ft * gc_kcf
        L2 = math.sqrt((abs(P_svc_kip) + Wf) / max(qa_ksf, 0.01))
        L2 = math.ceil(L2 * 4) / 4
        if abs(L2 - L) < 0.05: break
        L = L2

    P_tot = abs(P_svc_kip) + Wf
    eX = abs(Mz_svc_kipft) / (P_tot + 1e-9)   # ft
    eZ = abs(Mx_svc_kipft) / (P_tot + 1e-9)   # ft
    Lx = math.ceil(max(L, 6*eX + col_w_ft + 2.0) * 4) / 4
    Lz = math.ceil(max(L, 6*eZ + col_d_ft + 2.0) * 4) / 4
    A  = Lx * Lz
    Wf = A * Df_ft * gc_kcf
    P_tot = abs(P_svc_kip) + Wf

    # ── 2. Bearing pressures (ksf) ────────────────────────────────────────
    Sx = Lx**2 * Lz / 6.0
    Sz = Lx * Lz**2 / 6.0
    q_avg = P_tot / A
    q_max = P_tot/A + abs(Mz_svc_kipft)/Sx + abs(Mx_svc_kipft)/Sz
    q_min = P_tot/A - abs(Mz_svc_kipft)/Sx - abs(Mx_svc_kipft)/Sz
    bearing_ok = q_max <= qa_ksf

    # ── 3. Net factored upward pressure (ksf) ─────────────────────────────
    qu_ksf = abs(P_ult_kip) / A

    # ── 4. Two-way punching shear  (ACI 318-19, kip / in) ─────────────────
    phi = 0.75
    fc_psi = fc_ksi * 1000.0
    fy_psi = fy_ksi * 1000.0
    # work in inches
    col_w_in = col_w_ft * 12; col_d_in = col_d_ft * 12
    Lx_in = Lx * 12; Lz_in = Lz * 12
    qu_psi  = qu_ksf / 144.0   # ksf → kip/in² = ksi
    qu_kip_in2 = qu_psi

    d_in = 10.0   # start
    Vc2 = Vu2 = 0.0
    for _ in range(80):
        b0 = 2*(col_w_in + d_in) + 2*(col_d_in + d_in)
        Vc2 = phi * 4 * math.sqrt(fc_psi) * b0 * d_in / 1000.0   # kip
        punch_area_in2 = (col_w_in + d_in) * (col_d_in + d_in)
        Vu2 = abs(P_ult_kip) - qu_kip_in2 * punch_area_in2
        if Vc2 >= Vu2: break
        d_in += 0.5
    d_punch_in = d_in

    # ── 5. One-way shear ──────────────────────────────────────────────────
    d_in = d_punch_in
    Vc1x = Vu1x = Vc1z = Vu1z = 0.0
    for _ in range(80):
        dist_x_in = max(Lx_in/2 - col_w_in/2 - d_in, 0)
        dist_z_in = max(Lz_in/2 - col_d_in/2 - d_in, 0)
        Vu1x = qu_kip_in2 * Lz_in * dist_x_in
        Vc1x = phi * 2 * math.sqrt(fc_psi) * Lz_in * d_in / 1000.0
        Vu1z = qu_kip_in2 * Lx_in * dist_z_in
        Vc1z = phi * 2 * math.sqrt(fc_psi) * Lx_in * d_in / 1000.0
        if Vc1x >= Vu1x and Vc1z >= Vu1z: break
        d_in += 0.5
    d_req_in = max(d_punch_in, d_in)
    # round total thickness to nearest 1"
    t_in = math.ceil(d_req_in + cover_in + 0.5)   # 0.5 = bar radius approx
    t_in = math.ceil(t_in / 3) * 3                 # nearest 3"
    d_eff_in = t_in - cover_in - 0.5

    # ── 6. Flexure ────────────────────────────────────────────────────────
    lx_c_in = max(Lx_in/2 - col_w_in/2, 1.0)
    lz_c_in = max(Lz_in/2 - col_d_in/2, 1.0)
    # Mu in kip-in
    Mu_X = qu_kip_in2 * Lz_in * lx_c_in**2 / 2.0
    Mu_Z = qu_kip_in2 * Lx_in * lz_c_in**2 / 2.0

    def As_in2(Mu_kipin, b_in, d_):
        Rn = Mu_kipin / (0.9 * b_in * d_**2)   # ksi
        rho = 0.85 * fc_ksi / fy_ksi * (1 - math.sqrt(max(0, 1 - 2*Rn/(0.85*fc_ksi))))
        rho_min = max(200/fy_psi, 0.0018)
        rho = max(rho, rho_min)
        return rho * b_in * d_

    As_X = As_in2(Mu_X, Lz_in, d_eff_in)
    As_Z = As_in2(Mu_Z, Lx_in, d_eff_in)
    bar_dia_in = 1.0   # #8 bar default
    bar_area   = math.pi * bar_dia_in**2 / 4
    nX = math.ceil(As_X / bar_area)
    nZ = math.ceil(As_Z / bar_area)
    sX_in = round((Lz_in - 2*cover_in) / max(nX-1, 1))
    sZ_in = round((Lx_in - 2*cover_in) / max(nZ-1, 1))

    return dict(
        Lx_ft=Lx, Lz_ft=Lz,
        t_in=t_in, d_eff_in=d_eff_in,
        A_ft2=A, Wf_kip=Wf,
        q_max=q_max, q_min=q_min, q_avg=q_avg, bearing_ok=bearing_ok,
        eX_ft=eX, eZ_ft=eZ,
        Vu2=Vu2, Vc2=Vc2, punch_ok=(Vc2>=Vu2),
        Vu1x=Vu1x, Vc1x=Vc1x, shear_x_ok=(Vc1x>=Vu1x),
        Vu1z=Vu1z, Vc1z=Vc1z, shear_z_ok=(Vc1z>=Vu1z),
        Mu_X_kipft=Mu_X/12, Mu_Z_kipft=Mu_Z/12,
        As_X=As_X, As_Z=As_Z, nX=nX, nZ=nZ,
        bar_dia_in=bar_dia_in, bar_no=8,
        sX_in=sX_in, sZ_in=sZ_in,
        qu_ksf=qu_ksf,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SOIL PRESSURE HEATMAP  (ksf)
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(res, qa_ksf):
    Lx, Lz = res["Lx_ft"], res["Lz_ft"]
    qmax, qmin = res["q_max"], res["q_min"]
    x = np.linspace(-Lx/2, Lx/2, 40)
    z = np.linspace(-Lz/2, Lz/2, 40)
    X, _ = np.meshgrid(x, z)
    Q = (qmax+qmin)/2 + (qmax-qmin)/Lx * X

    fig = go.Figure(go.Heatmap(
        z=Q, x=x, y=z,
        colorscale="RdYlGn_r",
        zmin=max(0, qmin-0.1),
        zmax=qa_ksf*1.05,
        colorbar=dict(
            title=dict(text="q (ksf)", side="right"),
            tickfont=dict(color="#cfd8dc"),
        )
    ))
    cw2 = res.get("col_w_ft_saved", 0.5)/2
    fig.add_shape(type="rect", x0=-cw2, y0=-cw2, x1=cw2, y1=cw2,
        line=dict(color="#ffca28", width=2), fillcolor="rgba(255,202,40,.15)")
    fig.add_annotation(x=0, y=Lz/2+0.5,
        text=f"q_max={qmax:.3f} ksf  |  q_allow={qa_ksf:.3f} ksf",
        showarrow=False, font=dict(color="#fff", size=11))
    fig.update_layout(
        title="ASD Soil Contact Pressure (ksf)",
        xaxis_title="X (ft)", yaxis_title="Z (ft)",
        paper_bgcolor="#0f1117", plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"), height=380,
        margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SECTION SKETCH
# ─────────────────────────────────────────────────────────────────────────────
def fig_section(res, col_w_ft, cover_in):
    Lx = res["Lx_ft"]
    t  = res["t_in"] / 12.0
    cover_ft = cover_in / 12.0
    d  = res["d_eff_in"] / 12.0

    fig = go.Figure()
    fig.add_shape(type="rect", x0=-Lx/2, y0=-t, x1=Lx/2, y1=0,
        fillcolor="#37474f", line=dict(color="#90a4ae", width=1.5))
    fig.add_shape(type="rect", x0=-col_w_ft/2, y0=0, x1=col_w_ft/2, y1=1.5,
        fillcolor="#546e7a", line=dict(color="#90a4ae", width=1.5))
    rebar_y = -(t - cover_ft)
    fig.add_shape(type="line", x0=-Lx/2+0.1, y0=rebar_y, x1=Lx/2-0.1, y1=rebar_y,
        line=dict(color="#ef5350", width=1.5, dash="dash"))
    n = min(9, res["nX"])
    xs = np.linspace(-Lx/2+0.15, Lx/2-0.15, n)
    fig.add_trace(go.Scatter(x=xs, y=[rebar_y]*n, mode="markers",
        marker=dict(symbol="circle", size=10, color="#42a5f5"),
        name=f"#{res['bar_no']} bars"))
    fig.add_annotation(x=0, y=-t-0.3, text=f"Lx = {Lx:.2f} ft", xanchor="center",
        showarrow=False, font=dict(color="#ffca28", size=12))
    fig.add_annotation(x=Lx/2+0.15, y=-t/2, text=f"t = {res['t_in']}\"",
        textangle=-90, showarrow=False, font=dict(color="#ffca28", size=11))
    fig.add_annotation(x=0, y=rebar_y+0.1, text=f"d = {res['d_eff_in']:.1f}\"",
        xanchor="center", showarrow=False, font=dict(color="#ef5350", size=10))
    fig.add_shape(type="line", x0=-Lx/2-0.3, y0=0, x1=Lx/2+0.3, y1=0,
        line=dict(color="#66bb6a", width=2, dash="dot"))
    fig.add_annotation(x=Lx/2+0.25, y=0.08, text="GL",
        showarrow=False, font=dict(color="#66bb6a", size=11))
    fig.update_layout(
        title="Section Elevation",
        xaxis=dict(showgrid=False, zeroline=False, range=[-Lx/2-0.6, Lx/2+0.6]),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
                   range=[-t-0.5, 2.0]),
        paper_bgcolor="#0f1117", plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"), height=400,
        margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PLAN VIEW
# ─────────────────────────────────────────────────────────────────────────────
def fig_plan(res, cover_in):
    Lx, Lz = res["Lx_ft"], res["Lz_ft"]
    cv = cover_in/12.0
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-Lx/2, y0=-Lz/2, x1=Lx/2, y1=Lz/2,
        fillcolor="#37474f", line=dict(color="#90a4ae", width=2))
    fig.add_shape(type="rect",
        x0=-Lx/2+cv, y0=-Lz/2+cv, x1=Lx/2-cv, y1=Lz/2-cv,
        fillcolor="rgba(0,0,0,0)", line=dict(color="#42a5f5", width=1, dash="dot"))
    # X-dir bars (run along X, spaced in Z)
    nZ_bars = min(10, res["nZ"])
    for y_ in np.linspace(-Lz/2+cv, Lz/2-cv, nZ_bars):
        fig.add_shape(type="line", x0=-Lx/2+cv, y0=y_, x1=Lx/2-cv, y1=y_,
            line=dict(color="#42a5f5", width=1.5))
    # Z-dir bars
    nX_bars = min(10, res["nX"])
    for x_ in np.linspace(-Lx/2+cv, Lx/2-cv, nX_bars):
        fig.add_shape(type="line", x0=x_, y0=-Lz/2+cv, x1=x_, y1=Lz/2-cv,
            line=dict(color="#ef5350", width=1.5))
    # column
    fig.add_shape(type="rect", x0=-0.25, y0=-0.25, x1=0.25, y1=0.25,
        fillcolor="#546e7a", line=dict(color="#ffca28", width=2))
    fig.add_annotation(x=0, y=Lz/2+0.35, text=f"Lz = {Lz:.2f} ft",
        showarrow=False, font=dict(color="#ffca28", size=12))
    fig.add_annotation(x=Lx/2+0.3, y=0, text=f"Lx = {Lx:.2f} ft",
        showarrow=False, textangle=-90, font=dict(color="#ffca28", size=12))
    fig.update_layout(
        title="Plan View — Bottom Reinforcement  (🔵 X-dir  |  🔴 Z-dir)",
        xaxis=dict(showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1,
                   range=[-Lx/2-0.6, Lx/2+0.6]),
        yaxis=dict(showgrid=False, zeroline=False, range=[-Lz/2-0.6, Lz/2+0.6]),
        paper_bgcolor="#0f1117", plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"), height=420,
        margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE = """Node\tLC\tFX\tFY\tFZ\tMX\tMY\tMZ
1\t[5-1.1:1.4(DS+DO)+1.2TSEXP]\t-0.01\t1.2\t-0.1\t-1.11\t0.02\t0.55
\t[5-1.1:1.4(DS+DO)+1.2TSCON]\t0.01\t1.23\t0.14\t1.23\t-0.02\t0.29
\t[5-2.1:1.2(DS+DO)+1.2TSEXP+1.0TT+1.6L+0.5S]\t2.89\t-0.5\t-0.84\t-7.46\t0.15\t-37.44
\t[5-2.1:1.2(DS+DO)+1.2TSCON+1.0TT+1.6L+0.5S]\t2.91\t-0.47\t-0.6\t-5.12\t0.1\t-37.71
\t[5-3.1:1.2(DS+DO)+1.2TSEXP+1.0TT+1.6S+0.5L]\t2.89\t-0.5\t-0.84\t-7.46\t0.15\t-37.44
\t[5-3.1:1.2(DS+DO)+1.2TSCON+1.0TT+1.6S+0.5L]\t2.91\t-0.47\t-0.6\t-5.12\t0.1\t-37.71
\t[5-4.1:1.2(DS+DO)+1.2TSEXP+1.0WX+0.5L+0.5S]\t-0.84\t1.03\t-0.1\t-1.12\t0.02\t7.56
\t[5-5.2:0.9(DS+DO)+1.2TSEXP+1.0WX]\t-0.84\t0.77\t-0.11\t-1.13\t0.02\t7.47
\t[5-8.1:1.2(DS+DO)+1.2TSEXP+1.2TT+0.5L+0.5S]\t3.47\t-0.81\t-0.99\t-8.72\t0.17\t-45.03
\t[5-8.1:1.2(DS+DO)+1.2TSCON+1.2TT+0.5L+0.5S]\t3.49\t-0.78\t-0.75\t-6.39\t0.13\t-45.3
\t[4-1.1:DS+DO+TSEXP]\t-0.01\t0.86\t-0.09\t-0.93\t0.02\t0.41
\t[4-1.2:DS+DO+TSEXP+TT]\t2.89\t-0.67\t-0.82\t-7.27\t0.15\t-37.53
\t[4-4.1:DS+DO+TSEXP+0.75TT+0.75L+0.75S]\t2.17\t-0.29\t-0.64\t-5.69\t0.11\t-28.04
\t[4-5.1:DS+DO+TSEXP+0.6WX]\t-0.51\t0.86\t-0.09\t-0.93\t0.02\t4.65
\t[4-7.2:0.6(DS+DO)+TSEXP+0.6WX]\t-0.51\t0.51\t-0.09\t-0.95\t0.02\t4.53"""

def main():
    st.markdown("""
    <h1 style='margin-bottom:0'>🏗️ Foundation Design Tool</h1>
    <p style='color:#78909c;margin-top:2px;font-size:.92rem'>
    Paste STAAD Reactions Directly &nbsp;|&nbsp;
    <span class="lrfd-tag">LRFD 5-</span> Strength Design (ACI 318-19) &nbsp;|&nbsp;
    <span class="asd-tag">ASD 4-</span> Bearing Check &nbsp;|&nbsp;
    Units: kip · kip-ft · ft · in · ksf · ksi
    </p>""", unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Design Parameters")
        st.markdown("### 🧱 Materials")
        fc_ksi = st.number_input("f'c (ksi)",  2.0, 12.0, 4.0, 0.5)
        fy_ksi = st.number_input("fy  (ksi)", 40.0, 80.0, 60.0, 1.0)
        gc_pcf = st.number_input("γ concrete (pcf)", 100., 160., 150., 5.)
        gs_pcf = st.number_input("γ soil     (pcf)",  80., 140., 110., 5.)
        st.markdown("### 🌍 Soil & Footing")
        qa_ksf = st.number_input("Allowable Bearing (ksf)", 0.5, 20.0, 3.0, 0.25)
        Df_ft  = st.number_input("Footing Depth Df (ft)", 1.0, 15.0, 5.0, 0.5)
        cov_in = st.number_input("Clear Cover (in)", 2.0, 6.0, 3.0, 0.5)
        st.markdown("### 🏛️ Column")
        cw_ft  = st.number_input("Column bx (ft)", 0.5, 5.0, 1.5, 0.25)
        cd_ft  = st.number_input("Column bz (ft)", 0.5, 5.0, 1.5, 0.25)
        st.divider()
        st.markdown("""
        **LC Auto-Classification:**
        - Prefix **`5-`** → LRFD (strength)
        - Prefix **`4-`** → ASD (service / bearing)
        """)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📋 Load Input + 3D View",
        "📐 Foundation Sizing",
        "📊 Results & Drawings"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        sec("① Paste STAAD Reaction Table  (kip / kip-ft)")
        info("""Copy-paste the full STAAD support reaction output directly — header row included.<br>
        Format: <code>Node &nbsp; LC &nbsp; FX &nbsp; FY &nbsp; FZ &nbsp; MX &nbsp; MY &nbsp; MZ</code><br>
        Blank Node cells are fine. Tab or multi-space separated.
        <span class="lrfd-tag">LRFD</span> auto-detected by prefix <b>5-</b> &nbsp;|&nbsp;
        <span class="asd-tag">ASD</span> by prefix <b>4-</b>""")

        paste_txt = st.text_area("Paste STAAD output:", value=SAMPLE, height=260)

        col_up, col_btn = st.columns([2,1])
        with col_up:
            uploaded = st.file_uploader("OR upload Excel", type=["xlsx","xls"])
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            parse_btn = st.button("🔄 Parse & Preview Loads")

        if parse_btn or "df" not in st.session_state:
            df_parsed = None
            if uploaded:
                try:
                    df_parsed = add_type(parse_excel(uploaded))
                except Exception as e:
                    st.error(f"Excel error: {e}")
            if df_parsed is None and paste_txt.strip():
                raw = parse_staad_paste(paste_txt)
                if raw is not None:
                    df_parsed = add_type(raw)
            if df_parsed is not None and len(df_parsed) > 0:
                st.session_state["df"] = df_parsed
            else:
                st.error("Could not parse — check format.")

        if "df" in st.session_state:
            df = st.session_state["df"]
            lrfd_n = (df["Type"]=="LRFD").sum()
            asd_n  = (df["Type"]=="ASD").sum()
            c1,c2,c3 = st.columns(3)
            mcard("Total LCs",  str(len(df)), c1)
            mcard("LRFD", str(lrfd_n), c2)
            mcard("ASD",  str(asd_n),  c3)

            sec("Parsed Load Table  (kip / kip-ft)")
            def style_type(val):
                if val=="LRFD": return "background:#0d3b6e;color:#90caf9"
                return "background:#0d3b1e;color:#a5d6a7"
            disp = df[["LC","Type","FX","FY","FZ","MX","MY","MZ"]]
            st.dataframe(
                disp.style.applymap(style_type, subset=["Type"])
                          .format({c:"{:.3f}" for c in ["FX","FY","FZ","MX","MY","MZ"]}),
                use_container_width=True, height=300)

            st.divider()
            sec("② 3D Load Visualisation  (ft geometry)")
            Lx_pre = max(Lx_pre if "Lx_pre" in dir() else 0,
                         st.session_state.get("res",{}).get("Lx_ft",10.0))
            Lz_pre = st.session_state.get("res",{}).get("Lz_ft", Lx_pre)
            info("🔴 Thick = LRFD critical &nbsp;|&nbsp; 🟡 Thick = ASD critical &nbsp;|&nbsp; "
                 "Dotted arcs = governing moments.  All arrows at column top.")
            st.plotly_chart(fig_3d(df, Lx_pre or 10.0, Lz_pre or 10.0, cw_ft, cd_ft),
                            use_container_width=True)

            # Envelope bar chart
            st.divider()
            sec("Load Envelope — All Combinations")
            fig_env = go.Figure()
            for comp, clr in [("FY","#42a5f5"),("MZ","#ef5350"),("MX","#ab47bc")]:
                for typ, op in [("LRFD",0.85),("ASD",0.45)]:
                    sub = df[df["Type"]==typ]
                    fig_env.add_trace(go.Bar(
                        x=sub["LC"], y=sub[comp].abs(),
                        name=f"{typ} |{comp}|", marker_color=clr, opacity=op,
                        visible=True if comp=="FY" else "legendonly"))
            fig_env.update_layout(barmode="group",
                paper_bgcolor="#0f1117", plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"), height=320,
                xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                legend=dict(bgcolor="#13192b"),
                margin=dict(l=0,r=0,t=20,b=80))
            st.plotly_chart(fig_env, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        sec("③ Critical Combo Selection & Foundation Sizing")
        if "df" not in st.session_state:
            st.warning("Parse loads in Tab 1 first.")
        else:
            df = st.session_state["df"]
            lrfd_r, asd_r = critical_combos(df)

            c1,c2 = st.columns(2)
            with c1:
                st.markdown('<span class="lrfd-tag">LRFD — Strength Design</span>',
                            unsafe_allow_html=True)
                if lrfd_r is not None:
                    info(f"<b>{lrfd_r['LC']}</b><br>"
                         f"Fy={float(lrfd_r['FY']):.3f} kip &nbsp;|&nbsp; "
                         f"Fx={float(lrfd_r['FX']):.3f} kip &nbsp;|&nbsp; "
                         f"Fz={float(lrfd_r['FZ']):.3f} kip<br>"
                         f"Mx={float(lrfd_r['MX']):.3f} kip-ft &nbsp;|&nbsp; "
                         f"Mz={float(lrfd_r['MZ']):.3f} kip-ft")
                else:
                    st.warning("No LRFD combos detected.")
            with c2:
                st.markdown('<span class="asd-tag">ASD — Bearing Check</span>',
                            unsafe_allow_html=True)
                if asd_r is not None:
                    info(f"<b>{asd_r['LC']}</b><br>"
                         f"Fy={float(asd_r['FY']):.3f} kip &nbsp;|&nbsp; "
                         f"Fx={float(asd_r['FX']):.3f} kip &nbsp;|&nbsp; "
                         f"Fz={float(asd_r['FZ']):.3f} kip<br>"
                         f"Mx={float(asd_r['MX']):.3f} kip-ft &nbsp;|&nbsp; "
                         f"Mz={float(asd_r['MZ']):.3f} kip-ft")
                else:
                    st.warning("No ASD combos detected.")

            info(f"Self-weight = γ_c ({gc_pcf} pcf) × Lx × Lz × Df ({Df_ft} ft) — "
                 f"automatically added to ASD vertical load for bearing check.")

            if st.button("🔩 Run Foundation Sizing — ACI 318-19"):
                if lrfd_r is None or asd_r is None:
                    st.error("Need both LRFD and ASD combos.")
                else:
                    with st.spinner("Computing..."):
                        res = size_footing(
                            P_ult_kip=float(lrfd_r["FY"]),
                            Mx_ult_kipft=float(lrfd_r["MX"]),
                            Mz_ult_kipft=float(lrfd_r["MZ"]),
                            P_svc_kip=float(asd_r["FY"]),
                            Mx_svc_kipft=float(asd_r["MX"]),
                            Mz_svc_kipft=float(asd_r["MZ"]),
                            qa_ksf=qa_ksf, fc_ksi=fc_ksi, fy_ksi=fy_ksi,
                            cover_in=cov_in, col_w_ft=cw_ft, col_d_ft=cd_ft,
                            Df_ft=Df_ft, gc_pcf=gc_pcf, gs_pcf=gs_pcf)
                        res["col_w_ft_saved"] = cw_ft
                        st.session_state["res"] = res
                        st.success("✓ Sizing complete — see Results tab")

            if "res" in st.session_state:
                res = st.session_state["res"]
                st.divider()
                sec("📏 Footing Dimensions")
                c1,c2,c3,c4 = st.columns(4)
                mcard("Length Lx",    f"{res['Lx_ft']:.2f} ft", c1)
                mcard("Width Lz",     f"{res['Lz_ft']:.2f} ft", c2)
                mcard("Thickness",    f"{res['t_in']}\"",        c3)
                mcard("d effective",  f"{res['d_eff_in']:.1f}\"", c4)
                sec("⚖️ Bearing (ASD + Self-Weight)")
                c1,c2,c3,c4 = st.columns(4)
                mcard("Footing Wt",   f"{res['Wf_kip']:.2f} kip",  c1)
                mcard("q max",        f"{res['q_max']:.3f} ksf",    c2)
                mcard("q allow",      f"{qa_ksf:.3f} ksf",          c3)
                mcard("eX / eZ",      f"{res['eX_ft']:.2f} / {res['eZ_ft']:.2f} ft", c4)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        sec("④ Detailed Results, Drawings & Export")
        if "res" not in st.session_state:
            st.warning("Run sizing in Tab 2 first.")
        else:
            res = st.session_state["res"]

            # Code checks table
            sec("✅ ACI 318-19 Code Checks")
            checks = [
                ("Bearing Pressure (ASD)",    res["q_max"],  qa_ksf,      "ksf", res["bearing_ok"]),
                ("Two-Way Punching (LRFD)",   res["Vu2"],    res["Vc2"],   "kip", res["punch_ok"]),
                ("One-Way Shear — X (LRFD)",  res["Vu1x"],   res["Vc1x"],  "kip", res["shear_x_ok"]),
                ("One-Way Shear — Z (LRFD)",  res["Vu1z"],   res["Vc1z"],  "kip", res["shear_z_ok"]),
            ]
            rows = []
            for name, D, C, unit, ok in checks:
                rows.append({"Check": name,
                    "Demand": f"{D:.3f} {unit}",
                    "Capacity φVc": f"{C:.3f} {unit}",
                    "D/C Ratio": f"{D/(C+1e-9):.3f}",
                    "Status": "PASS ✓" if ok else "FAIL ✗"})
            ck_df = pd.DataFrame(rows)
            def hl(row):
                c = "#1b5e20" if "PASS" in row["Status"] else "#b71c1c"
                return [""]*4 + [f"background:{c};color:#fff;font-weight:700"]
            st.dataframe(ck_df.style.apply(hl, axis=1),
                         use_container_width=True, hide_index=True)

            sec("🔩 Reinforcement Schedule")
            c1,c2,c3,c4 = st.columns(4)
            mcard("Bars — X dir",  f"{res['nX']} × #{res['bar_no']}",  c1)
            mcard("Spacing X",     f"@ {res['sX_in']}\"",              c2)
            mcard("Bars — Z dir",  f"{res['nZ']} × #{res['bar_no']}",  c3)
            mcard("Spacing Z",     f"@ {res['sZ_in']}\"",              c4)
            info(f"Bottom mat (LRFD flexure): "
                 f"#{res['bar_no']} @ {res['sX_in']}\" (X-dir), As={res['As_X']:.2f} in²  |  "
                 f"#{res['bar_no']} @ {res['sZ_in']}\" (Z-dir), As={res['As_Z']:.2f} in²  |  "
                 f"Cover={cov_in}\"  |  Mu_X={res['Mu_X_kipft']:.2f} kip-ft  |  "
                 f"Mu_Z={res['Mu_Z_kipft']:.2f} kip-ft")

            # Drawings
            st.divider()
            cl, cr = st.columns(2)
            with cl: st.plotly_chart(fig_heatmap(res, qa_ksf), use_container_width=True)
            with cr: st.plotly_chart(fig_section(res, cw_ft, cov_in), use_container_width=True)
            st.plotly_chart(fig_plan(res, cov_in), use_container_width=True)

            # Final 3D with correct dimensions
            sec("3D — Final Footing + All Applied Loads")
            if "df" in st.session_state:
                st.plotly_chart(
                    fig_3d(st.session_state["df"],
                           res["Lx_ft"], res["Lz_ft"], cw_ft, cd_ft),
                    use_container_width=True)

            # Export
            sec("📄 Export Design Report")
            export = {
                "Parameter": [
                    "Lx (ft)", "Lz (ft)", "Thickness (in)", "d effective (in)",
                    "Plan Area (ft²)", "Footing Self-Weight (kip)",
                    "q_max ASD (ksf)", "q_min ASD (ksf)", "q_allowable (ksf)",
                    "Bearing Check (ASD)", "Punching Shear (LRFD)",
                    "One-Way Shear X (LRFD)", "One-Way Shear Z (LRFD)",
                    "Mu_X (kip-ft)", "As_X (in²)", "nX bars", "Spacing X (in)",
                    "Mu_Z (kip-ft)", "As_Z (in²)", "nZ bars", "Spacing Z (in)",
                    "Bar Size", "Clear Cover (in)",
                ],
                "Value": [
                    res["Lx_ft"], res["Lz_ft"], res["t_in"], round(res["d_eff_in"],2),
                    round(res["A_ft2"],2), round(res["Wf_kip"],3),
                    round(res["q_max"],4), round(res["q_min"],4), qa_ksf,
                    "PASS" if res["bearing_ok"] else "FAIL",
                    "PASS" if res["punch_ok"]   else "FAIL",
                    "PASS" if res["shear_x_ok"] else "FAIL",
                    "PASS" if res["shear_z_ok"] else "FAIL",
                    round(res["Mu_X_kipft"],3), round(res["As_X"],3), res["nX"], res["sX_in"],
                    round(res["Mu_Z_kipft"],3), round(res["As_Z"],3), res["nZ"], res["sZ_in"],
                    f"#{res['bar_no']}", cov_in,
                ],
            }
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                pd.DataFrame(export).to_excel(w, index=False, sheet_name="Foundation Design")
                if "df" in st.session_state:
                    st.session_state["df"].to_excel(w, index=False, sheet_name="All Load Combos")
            buf.seek(0)
            st.download_button("⬇️ Download Full Report (.xlsx)", data=buf,
                file_name="Foundation_Design_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
