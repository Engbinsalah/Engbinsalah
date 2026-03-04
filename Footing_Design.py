"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Paste LC table directly from STAAD output (kip / kip-ft)
LRFD (prefix 5-) → Strength Design   ASD (prefix 4-) → Bearing Check
ACI 318-19 | ASCE 7-22
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io, math, re

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foundation Design", page_icon="🏗️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0f1117;color:#e2e8f0}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#16213e);border:1px solid #2d3561;
  border-radius:10px;padding:14px;text-align:center;margin:3px}
.metric-value{font-size:1.65rem;font-weight:700;color:#4fc3f7}
.metric-label{font-size:.75rem;color:#90a4ae;text-transform:uppercase;letter-spacing:.8px;margin-top:2px}
.sec-hdr{background:linear-gradient(90deg,#1565c0,#0d47a1);color:#fff;
  padding:7px 14px;border-radius:6px;font-weight:700;font-size:1rem;margin:10px 0 6px}
.info-box{background:#0d2137;border-left:4px solid #1565c0;
  padding:9px 13px;border-radius:4px;margin:5px 0;font-size:.85rem}
.lrfd-tag{background:#1565c0;color:#fff;border-radius:4px;padding:2px 8px;
  font-size:.75rem;font-weight:700;margin-right:4px}
.asd-tag{background:#2e7d32;color:#fff;border-radius:4px;padding:2px 8px;
  font-size:.75rem;font-weight:700;margin-right:4px}
div[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2a3a}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;
  border:none;border-radius:6px;font-weight:700;padding:10px 28px;width:100%}
.stTextArea textarea{background:#0d1a26;color:#e2e8f0;border:1px solid #1e3a5f;
  font-family:monospace;font-size:.82rem}
h1,h2,h3{color:#e0e0e0 !important}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-hdr">{t}</div>', unsafe_allow_html=True)
def info(t): st.markdown(f'<div class="info-box">{t}</div>', unsafe_allow_html=True)
def mcard(lbl, val, col):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION  (kip → kN,  kip-ft → kN·m)
# ─────────────────────────────────────────────────────────────────────────────
KIP2KN   = 4.44822
KIPFT2KNM = 1.35582

# ─────────────────────────────────────────────────────────────────────────────
# LOAD-COMBO PARSER  — handles STAAD paste, Excel upload, or manual table
# ─────────────────────────────────────────────────────────────────────────────
def classify_lc(name: str) -> str:
    """Return 'LRFD' if name starts with 5-, else 'ASD'."""
    n = name.strip().lstrip("[")
    if re.match(r"5[-–]", n):
        return "LRFD"
    if re.match(r"4[-–]", n):
        return "ASD"
    # fallback: presence of factored indicators
    if re.search(r"1\.[2-9]|1\.4|0\.9\(", n):
        return "LRFD"
    return "ASD"

def parse_staad_paste(text: str) -> pd.DataFrame:
    """
    Parse text pasted directly from STAAD reaction table.
    Accepts tab- or multi-space delimited.
    Skips header rows (non-numeric first token or containing LC/FX keywords).
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line: continue
        # Skip pure header lines
        if re.search(r"\bLC\b|\bFX\b|\bNode\b|\bkip\b|\bHorizontal\b", line, re.I):
            continue
        # Split on tabs first, then fallback to 2+ spaces
        parts = re.split(r"\t|  +", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 7: continue
        # Find the LC name (bracketed or first non-numeric token)
        # STAAD format: Node  LC  FX  FY  FZ  MX  MY  MZ  (Node may be blank)
        # detect if first token is a number (Node id) — skip it
        offset = 0
        if re.match(r"^\d+$", parts[0]):
            offset = 1
        if len(parts) < offset + 7: continue
        lc = parts[offset].strip("[]")
        try:
            vals = [float(parts[offset+i+1]) for i in range(6)]
        except ValueError:
            continue
        rows.append([lc] + vals)

    if not rows: return None
    df = pd.DataFrame(rows, columns=["LC","FX_kip","FY_kip","FZ_kip","MX_kipft","MY_kipft","MZ_kipft"])
    return df

def parse_manual_or_excel(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a user-edited or uploaded dataframe to kip units."""
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    # map common aliases
    rename = {}
    for c in df_raw.columns:
        cl = c.lower()
        if "lc" in cl or "case" in cl or "combo" in cl or "load" in cl: rename[c]="LC"
        elif cl in ("fx","fx (kip)","fx (kn)"): rename[c]="FX_kip"
        elif cl in ("fy","fy (kip)","fy (kn)"): rename[c]="FY_kip"
        elif cl in ("fz","fz (kip)","fz (kn)"): rename[c]="FZ_kip"
        elif cl in ("mx","mx (kip-ft)","mx (kn·m)"): rename[c]="MX_kipft"
        elif cl in ("my","my (kip-ft)","my (kn·m)"): rename[c]="MY_kipft"
        elif cl in ("mz","mz (kip-ft)","mz (kn·m)"): rename[c]="MZ_kipft"
    df_raw = df_raw.rename(columns=rename)
    needed = ["LC","FX_kip","FY_kip","FZ_kip","MX_kipft","MY_kipft","MZ_kipft"]
    for col in needed:
        if col not in df_raw.columns:
            df_raw[col] = 0.0
    return df_raw[needed].dropna(subset=["FY_kip"])

def to_kn(df: pd.DataFrame) -> pd.DataFrame:
    """Add SI columns to kip-unit dataframe."""
    df = df.copy()
    df["Type"]      = df["LC"].apply(classify_lc)
    df["FX (kN)"]   = df["FX_kip"] * KIP2KN
    df["FY (kN)"]   = df["FY_kip"] * KIP2KN
    df["FZ (kN)"]   = df["FZ_kip"] * KIP2KN
    df["MX (kN·m)"] = df["MX_kipft"] * KIPFT2KNM
    df["MY (kN·m)"] = df["MY_kipft"] * KIPFT2KNM
    df["MZ (kN·m)"] = df["MZ_kipft"] * KIPFT2KNM
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL COMBO SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
def critical_combos(df: pd.DataFrame):
    """
    LRFD: used for shear + flexure design (strength).
    ASD:  used for bearing pressure check (service).
    Returns (lrfd_row, asd_row) as Series with SI values.
    Uses max resultant moment + vertical load envelope.
    """
    lrfd = df[df["Type"]=="LRFD"].copy()
    asd  = df[df["Type"]=="ASD"].copy()

    def score(row):
        # Combined demand score: abs(FY) + resultant moment + resultant horiz
        M_res = math.sqrt(row["MX (kN·m)"]**2 + row["MZ (kN·m)"]**2)
        F_res = math.sqrt(row["FX (kN)"]**2 + row["FZ (kN)"]**2)
        return abs(row["FY (kN)"]) + M_res + F_res

    if len(lrfd):
        lrfd["_score"] = lrfd.apply(score, axis=1)
        lrfd_crit = lrfd.loc[lrfd["_score"].idxmax()]
    else:
        lrfd_crit = None

    if len(asd):
        asd["_score"] = asd.apply(score, axis=1)
        asd_crit = asd.loc[asd["_score"].idxmax()]
    else:
        asd_crit = None

    return lrfd_crit, asd_crit

# ─────────────────────────────────────────────────────────────────────────────
# 3-D LOAD FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d(df, Lx, Lz, col_w, col_d):
    fig = go.Figure()
    hw,hd,ht = Lx/2, Lz/2, 0.55
    I=[0,0,0,4,4,4,0,2,2,0,6,4]; J=[1,2,3,5,6,7,1,3,6,4,7,5]; K=[2,3,0,6,7,4,5,7,5,6,2,1]
    vx=[-hw,hw,hw,-hw,-hw,hw,hw,-hw]; vy=[-hd,-hd,hd,hd,-hd,-hd,hd,hd]; vz=[-ht]*4+[0]*4
    fig.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=I,j=J,k=K,color="#37474f",opacity=.6,name="Footing"))
    cw,cd,ch=col_w/2,col_d/2,1.4
    cvx=[-cw,cw,cw,-cw,-cw,cw,cw,-cw]; cvy=[-cd,-cd,cd,cd,-cd,-cd,cd,cd]; cvz=[0]*4+[ch]*4
    fig.add_trace(go.Mesh3d(x=cvx,y=cvy,z=cvz,i=I,j=J,k=K,color="#546e7a",opacity=.85,name="Column"))
    gx=np.linspace(-hw*2.2,hw*2.2,4); gy=np.linspace(-hd*2.2,hd*2.2,4)
    GX,GY=np.meshgrid(gx,gy); GZ=np.zeros_like(GX)-ht
    fig.add_trace(go.Surface(x=GX,y=GY,z=GZ,colorscale=[[0,"#1a237e"],[1,"#283593"]],
                             opacity=.15,showscale=False,name="Soil"))

    # Draw arrows for each TYPE with different colors
    arrow_scale = max(Lx,Lz)*0.55
    for _, row in df.iterrows():
        clr = "#42a5f5" if row["Type"]=="LRFD" else "#66bb6a"
        ref = max(abs(row["FY (kN)"]),abs(row["FX (kN)"]),abs(row["FZ (kN)"]),0.1)
        sc  = arrow_scale/ref*0.4
        for dx,dy,dz,lbl in [
            (row["FX (kN)"]*sc,0,0,f"Fx"),
            (0,row["FZ (kN)"]*sc,0,f"Fz"),
            (0,0,-abs(row["FY (kN)"])*sc,f"Fy"),
        ]:
            if max(abs(dx),abs(dy),abs(dz))<0.05: continue
            fig.add_trace(go.Scatter3d(x=[0,dx],y=[0,dy],z=[ch,ch+dz],
                mode="lines",line=dict(color=clr,width=3),
                name=f"{row['Type']} {lbl}",showlegend=False))

    # Critical LRFD & ASD arrows — thick
    lrfd_r, asd_r = critical_combos(df)
    for crit_row, clr, tag in [(lrfd_r,"#ef5350","LRFD"), (asd_r,"#ffca28","ASD")]:
        if crit_row is None: continue
        ref=max(abs(crit_row["FY (kN)"]),abs(crit_row["FX (kN)"]),abs(crit_row["FZ (kN)"]),0.1)
        sc=arrow_scale/ref
        for dx,dy,dz,lbl in [
            (crit_row["FX (kN)"]*sc,0,0,f"{tag} Fx={crit_row['FX (kN)']:.1f}kN"),
            (0,crit_row["FZ (kN)"]*sc,0,f"{tag} Fz={crit_row['FZ (kN)']:.1f}kN"),
            (0,0,-abs(crit_row["FY (kN)"])*sc,f"{tag} Fy={crit_row['FY (kN)']:.1f}kN"),
        ]:
            if max(abs(dx),abs(dy),abs(dz))<0.05: continue
            fig.add_trace(go.Scatter3d(x=[0,dx],y=[0,dy],z=[ch,ch+dz],
                mode="lines",line=dict(color=clr,width=6),name=lbl))
            fig.add_trace(go.Cone(x=[dx],y=[dy],z=[ch+dz],u=[dx*.22],v=[dy*.22],w=[dz*.22],
                colorscale=[[0,clr],[1,clr]],showscale=False,
                sizemode="absolute",sizeref=max(Lx,Lz)*.13,showlegend=False))
        # Moment arcs
        ref_m=max(abs(crit_row["MX (kN·m)"]),abs(crit_row["MZ (kN·m)"]),0.1)
        sc_m=arrow_scale/ref_m*0.4
        for moment,axis,mlbl in [(crit_row["MX (kN·m)"],"x","Mx"),(crit_row["MZ (kN·m)"],"z","Mz")]:
            if abs(moment)<0.1: continue
            r=max(Lx,Lz)*0.35; t_=np.linspace(0,1.6*math.pi,50)
            if axis=="x": ax_x=np.zeros_like(t_); ax_y=r*np.cos(t_); az_=ch+0.4+r*np.sin(t_)*0.6
            else:         ax_x=r*np.cos(t_);     ax_y=r*np.sin(t_); az_=np.zeros_like(t_)+ch+0.3
            fig.add_trace(go.Scatter3d(x=ax_x,y=ax_y,z=az_,mode="lines",
                line=dict(color=clr,width=3,dash="dot"),
                name=f"{tag} {mlbl}={moment:.1f}kN·m",showlegend=True))

    # Dimension lines
    dz=-ht-.12
    fig.add_trace(go.Scatter3d(x=[-hw,hw],y=[-hd-.4,-hd-.4],z=[dz,dz],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lx={Lx:.2f}m"],textposition="middle right",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[hw+.4,hw+.4],y=[-hd,hd],z=[dz,dz],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lz={Lz:.2f}m"],textposition="middle right",showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            yaxis=dict(title="Y (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            zaxis=dict(title="Z (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            bgcolor="#0d1117",camera=dict(eye=dict(x=1.7,y=1.7,z=1.2))),
        paper_bgcolor="#0f1117",font=dict(color="#cfd8dc"),
        legend=dict(bgcolor="#13192b",bordercolor="#263238",font=dict(size=10),
                    itemsizing="constant"),
        height=600,margin=dict(l=0,r=0,t=36,b=0),
        title=dict(text="3D Applied Loads — 🔴 LRFD Critical  |  🟡 ASD Critical  |  🔵 All LRFD  |  🟢 All ASD",
                   font=dict(color="#90caf9",size=13)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FOUNDATION SIZING  (ACI 318-19)
# ─────────────────────────────────────────────────────────────────────────────
def size_footing(P_ult, Mx_ult, Mz_ult,   # LRFD (kN, kN·m)
                 P_svc, Mx_svc, Mz_svc,   # ASD  (kN, kN·m)
                 qa, fc, fy, cover,
                 col_w, col_d, Df, gc, gs):

    # ── 1. Plan size from ASD bearing ─────────────────────────────────────
    qa_net = qa - gs*Df
    L = math.sqrt(max(P_svc/max(qa_net,1), 0.3))
    L = math.ceil(L*10)/10
    for _ in range(20):
        Wf = L*L*Df*gc
        L2 = math.sqrt((P_svc+Wf)/max(qa,1))
        L2 = math.ceil(L2*10)/10
        if abs(L2-L)<.05: break
        L = L2
    eX = abs(Mz_svc)/(P_svc+Wf+1e-9)
    eZ = abs(Mx_svc)/(P_svc+Wf+1e-9)
    Lx = math.ceil(max(L, 6*eX+col_w+.6)*20)/20
    Lz = math.ceil(max(L, 6*eZ+col_d+.6)*20)/20
    A  = Lx*Lz
    Wf = A*Df*gc
    Ptot = P_svc+Wf

    # ── 2. Bearing pressures ───────────────────────────────────────────────
    Sx=Lx**2*Lz/6; Sz=Lx*Lz**2/6
    q_avg = Ptot/A
    q_max = Ptot/A + abs(Mz_svc)/Sx + abs(Mx_svc)/Sz
    q_min = Ptot/A - abs(Mz_svc)/Sx - abs(Mx_svc)/Sz
    bearing_ok = q_max<=qa

    # ── 3. Net factored upward pressure ───────────────────────────────────
    qu = abs(P_ult)/A   # uniform, conservative

    # ── 4. Two-way punching shear (ACI 318-19 §22.6) ──────────────────────
    phi=0.75; d=250; Vc2=0; Vu2=0
    for _ in range(60):
        b0=2*(col_w*1000+d)+2*(col_d*1000+d)
        Vc2=phi*0.33*math.sqrt(fc)*b0*d/1e6
        Vu2=abs(P_ult)-qu*(col_w+d/1000)*(col_d+d/1000)
        if Vc2>=Vu2: break
        d+=10
    d_punch=d

    # ── 5. One-way shear ──────────────────────────────────────────────────
    d=d_punch; Vc1x=0; Vu1x=0; Vc1z=0; Vu1z=0
    for _ in range(60):
        dist_x=max(Lx/2-col_w/2-d/1000,0)
        dist_z=max(Lz/2-col_d/2-d/1000,0)
        Vu1x=qu*Lz*dist_x; Vc1x=phi*0.17*math.sqrt(fc)*(Lz*1000)*d/1e6
        Vu1z=qu*Lx*dist_z; Vc1z=phi*0.17*math.sqrt(fc)*(Lx*1000)*d/1e6
        if Vc1x>=Vu1x and Vc1z>=Vu1z: break
        d+=10
    d_req=max(d_punch,d)
    t_mm=math.ceil((d_req+cover+12)/50)*50
    d_eff=t_mm-cover-12

    # ── 6. Flexure ─────────────────────────────────────────────────────────
    lx_c=max(Lx/2-col_w/2,.05); lz_c=max(Lz/2-col_d/2,.05)
    Mu_X=qu*Lz*lx_c**2/2*1e6; Mu_Z=qu*Lx*lz_c**2/2*1e6
    def As_calc(Mu,b,d_):
        Rn=Mu/(0.9*b*d_**2)
        rho=.85*fc/fy*(1-math.sqrt(max(0,1-2*Rn/(.85*fc))))
        rho=max(rho,max(.0018,1.4/fy))
        return rho*b*d_
    As_X=As_calc(Mu_X,Lz*1000,d_eff)
    As_Z=As_calc(Mu_Z,Lx*1000,d_eff)
    bd=16; ba=math.pi*bd**2/4
    nX=math.ceil(As_X/ba); nZ=math.ceil(As_Z/ba)
    sX=round((Lz*1000-2*cover)/max(nX-1,1))
    sZ=round((Lx*1000-2*cover)/max(nZ-1,1))

    return dict(
        Lx=Lx,Lz=Lz,t_mm=t_mm,d_eff=d_eff,A=A,Wf=Wf,
        q_max=q_max,q_min=q_min,q_avg=q_avg,bearing_ok=bearing_ok,
        eX=eX,eZ=eZ,
        Vu2=Vu2,Vc2=Vc2,punch_ok=Vc2>=Vu2,
        Vu1x=Vu1x,Vc1x=Vc1x,shear_x_ok=Vc1x>=Vu1x,
        Vu1z=Vu1z,Vc1z=Vc1z,shear_z_ok=Vc1z>=Vu1z,
        Mu_X=Mu_X/1e6,Mu_Z=Mu_Z/1e6,
        As_X=As_X,As_Z=As_Z,nX=nX,nZ=nZ,bd=bd,sX=sX,sZ=sZ,
        qu_kPa=qu,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DRAWINGS
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(res, qa):
    Lx,Lz=res["Lx"],res["Lz"]
    qmax,qmin=res["q_max"],res["q_min"]
    x=np.linspace(-Lx/2,Lx/2,40); z=np.linspace(-Lz/2,Lz/2,40)
    X,_=np.meshgrid(x,z)
    Q=(qmax+qmin)/2+(qmax-qmin)/Lx*X
    fig=go.Figure(go.Heatmap(z=Q,x=x,y=z,colorscale="RdYlGn_r",
        zmin=max(0,qmin-5),zmax=qa*1.05,
        colorbar=dict(title="q (kPa)",tickfont=dict(color="#cfd8dc"),
                      titlefont=dict(color="#cfd8dc"))))
    fig.add_shape(type="rect",x0=-.25,y0=-.25,x1=.25,y1=.25,
        line=dict(color="#ffca28",width=2),fillcolor="rgba(255,202,40,.15)")
    fig.add_annotation(x=0,y=Lz/2+.15,text=f"q_max={qmax:.1f} kPa  q_min={qmin:.1f} kPa",
        showarrow=False,font=dict(color="#fff",size=11))
    fig.update_layout(title="ASD Soil Contact Pressure (kPa)",
        xaxis_title="X (m)",yaxis_title="Z (m)",
        paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"),height=380,margin=dict(l=0,r=0,t=40,b=0))
    return fig

def fig_section(res, col_w, cover):
    Lx=res["Lx"]; t=res["t_mm"]/1000; d=res["d_eff"]/1000
    cover_m=(res["t_mm"]-res["d_eff"]-16)/1000
    fig=go.Figure()
    fig.add_shape(type="rect",x0=-Lx/2,y0=-t,x1=Lx/2,y1=0,
        fillcolor="#37474f",line=dict(color="#90a4ae",width=1.5))
    fig.add_shape(type="rect",x0=-col_w/2,y0=0,x1=col_w/2,y1=1.2,
        fillcolor="#546e7a",line=dict(color="#90a4ae",width=1.5))
    fig.add_shape(type="line",x0=-Lx/2+.02,y0=-(t-cover_m),
        x1=Lx/2-.02,y1=-(t-cover_m),line=dict(color="#ef5350",width=1.5,dash="dash"))
    n=min(9,res["nX"])
    xs=np.linspace(-Lx/2+.1,Lx/2-.1,n)
    fig.add_trace(go.Scatter(x=xs,y=[-(t-cover_m)]*n,mode="markers",
        marker=dict(symbol="circle",size=10,color="#42a5f5"),name=f"⌀{res['bd']} bars"))
    for ann in [
        dict(x=0,y=-t-.2,text=f"Lx = {Lx:.2f} m",xanchor="center"),
        dict(x=Lx/2+.12,y=-t/2,text=f"t = {res['t_mm']} mm",textangle=-90),
        dict(x=0,y=-(t-cover_m)+.08,text=f"d = {res['d_eff']} mm",xanchor="center",
             font=dict(color="#ef5350",size=10)),
    ]:
        fig.add_annotation(showarrow=False,font=dict(color="#ffca28",size=11),**ann)
    fig.add_shape(type="line",x0=-Lx/2-.3,y0=0,x1=Lx/2+.3,y1=0,
        line=dict(color="#66bb6a",width=2,dash="dot"))
    fig.add_annotation(x=Lx/2+.22,y=.06,text="GL",showarrow=False,font=dict(color="#66bb6a",size=11))
    fig.update_layout(title="Section Elevation",
        xaxis=dict(showgrid=False,zeroline=False,range=[-Lx/2-.5,Lx/2+.5]),
        yaxis=dict(showgrid=False,zeroline=False,scaleanchor="x",scaleratio=1,range=[-t-.4,1.6]),
        paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"),height=380,margin=dict(l=0,r=0,t=40,b=0))
    return fig

def fig_plan(res):
    Lx,Lz=res["Lx"],res["Lz"]
    cover=75/1000
    fig=go.Figure()
    fig.add_shape(type="rect",x0=-Lx/2,y0=-Lz/2,x1=Lx/2,y1=Lz/2,
        fillcolor="#37474f",line=dict(color="#90a4ae",width=2))
    fig.add_shape(type="rect",x0=-Lx/2+cover,y0=-Lz/2+cover,
        x1=Lx/2-cover,y1=Lz/2-cover,
        fillcolor="rgba(0,0,0,0)",line=dict(color="#42a5f5",width=1,dash="dot"))
    # rebar in X dir
    ys=np.linspace(-Lz/2+cover,.0+cover,min(8,res["nX"]))
    for y_ in ys:
        fig.add_shape(type="line",x0=-Lx/2+cover,y0=y_,x1=Lx/2-cover,y1=y_,
            line=dict(color="#42a5f5",width=1.5))
    # rebar in Z dir
    xs=np.linspace(-Lx/2+cover,.0+cover,min(8,res["nZ"]))
    for x_ in xs:
        fig.add_shape(type="line",x0=x_,y0=-Lz/2+cover,x1=x_,y1=Lz/2-cover,
            line=dict(color="#ef5350",width=1.5))
    # column
    fig.add_shape(type="rect",x0=-.25,y0=-.25,x1=.25,y1=.25,
        fillcolor="#546e7a",line=dict(color="#ffca28",width=2))
    fig.add_annotation(x=0,y=Lz/2+.18,text=f"Lz = {Lz:.2f} m",
        showarrow=False,font=dict(color="#ffca28",size=12))
    fig.add_annotation(x=Lx/2+.18,y=0,text=f"Lx = {Lx:.2f} m",
        showarrow=False,textangle=-90,font=dict(color="#ffca28",size=12))
    fig.update_layout(title="Plan View — Bottom Reinforcement",
        xaxis=dict(showgrid=False,zeroline=False,scaleanchor="y",scaleratio=1,
                   range=[-Lx/2-.4,Lx/2+.4]),
        yaxis=dict(showgrid=False,zeroline=False,range=[-Lz/2-.4,Lz/2+.4]),
        paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"),height=400,margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <h1 style='margin-bottom:0'>🏗️ Foundation Design Tool</h1>
    <p style='color:#78909c;margin-top:2px;font-size:.93rem'>
    Paste STAAD Reactions Directly &nbsp;|&nbsp; LRFD → Strength Design &nbsp;|&nbsp;
    ASD → Bearing Check &nbsp;|&nbsp; ACI 318-19 / ASCE 7-22 &nbsp;|&nbsp;
    Inputs: kip / kip-ft
    </p>""", unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Design Parameters")
        st.markdown("### 🧱 Materials")
        fc  = st.number_input("f'c (MPa)",  20.,80.,28.,1.)
        fy  = st.number_input("fy  (MPa)", 300.,600.,420.,10.)
        gc  = st.number_input("γ concrete (kN/m³)",20.,26.,24.,.5)
        gs  = st.number_input("γ soil     (kN/m³)",14.,22.,18.,.5)
        st.markdown("### 🌍 Soil & Footing")
        qa  = st.number_input("Allowable Bearing (kPa)",50.,1000.,150.,10.)
        Df  = st.number_input("Footing Depth Df (m)",.5,4.,1.5,.1)
        cov = st.number_input("Clear Cover (mm)",50,100,75,5)
        st.markdown("### 🏛️ Column")
        cw  = st.number_input("Column bx (m)",.2,1.5,.5,.05)
        cd  = st.number_input("Column bz (m)",.2,1.5,.5,.05)
        st.divider()
        st.markdown("""
        **LC Classification:**
        - Prefix **`5-`** → LRFD (strength)
        - Prefix **`4-`** → ASD (service)
        - Auto-detected from STAAD format
        """)

    # ── TABS ─────────────────────────────────────────────────────────────────
    t1,t2,t3 = st.tabs(["📋 Load Input + 3D View","📐 Sizing","📊 Results & Drawings"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════════════════════════════════
    with t1:
        sec("① Paste STAAD Reaction Table  (kip / kip-ft)")
        info("""Paste the full STAAD output table here — including the header row.
        Format: <code>Node &nbsp; LC &nbsp; FX &nbsp; FY &nbsp; FZ &nbsp; MX &nbsp; MY &nbsp; MZ</code><br>
        <span class="lrfd-tag">LRFD</span> combos detected by prefix <b>5-</b> &nbsp;|&nbsp;
        <span class="asd-tag">ASD</span> combos detected by prefix <b>4-</b>""")

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

        paste_txt = st.text_area("Paste STAAD output here:", value=SAMPLE, height=250,
                                 placeholder="Paste tab-separated STAAD reactions...")

        col_up, col_btn = st.columns([2,1])
        with col_up:
            uploaded = st.file_uploader("OR upload Excel (.xlsx)", type=["xlsx","xls"])
        with col_btn:
            st.markdown("<br>",unsafe_allow_html=True)
            parse_btn = st.button("🔄 Parse & Preview Loads")

        if parse_btn or "df_si" not in st.session_state:
            df_kip = None
            # Try Excel first
            if uploaded:
                try:
                    raw=pd.read_excel(uploaded,header=None)
                    hrow=0
                    for i,row in raw.iterrows():
                        rs=" ".join([str(v).lower() for v in row.values])
                        if any(k in rs for k in ["lc","fx","fy","node"]): hrow=i; break
                    df_xl=pd.read_excel(uploaded,header=hrow)
                    df_xl.columns=[str(c).strip() for c in df_xl.columns]
                    df_xl=df_xl.dropna(how="all")
                    df_kip=parse_manual_or_excel(df_xl)
                except Exception as e:
                    st.error(f"Excel parse error: {e}")
            # Fallback to paste
            if df_kip is None and paste_txt.strip():
                df_kip = parse_staad_paste(paste_txt)
            if df_kip is not None and len(df_kip)>0:
                df_si = to_kn(df_kip)
                st.session_state["df_kip"]=df_kip
                st.session_state["df_si"]=df_si
            else:
                st.error("Could not parse loads. Check format.")

        if "df_si" in st.session_state:
            df_si = st.session_state["df_si"]
            lrfd_n = (df_si["Type"]=="LRFD").sum()
            asd_n  = (df_si["Type"]=="ASD").sum()

            c1,c2,c3 = st.columns(3)
            mcard("Total Combos",str(len(df_si)),c1)
            mcard("LRFD Combos",f"{lrfd_n}",c2)
            mcard("ASD Combos", f"{asd_n}", c3)

            sec("Load Table (converted to kN / kN·m)")
            disp_cols=["LC","Type","FX (kN)","FY (kN)","FZ (kN)","MX (kN·m)","MY (kN·m)","MZ (kN·m)"]
            def style_type(val):
                return "background:#0d3b6e;color:#90caf9" if val=="LRFD" else "background:#0d3b1e;color:#a5d6a7"
            st.dataframe(
                df_si[disp_cols].style
                    .applymap(style_type, subset=["Type"])
                    .format({c:"{:.2f}" for c in disp_cols[2:]}),
                use_container_width=True, height=300)

            st.divider()
            sec("② 3D Load Visualisation")
            info("🔴 Thick = LRFD critical &nbsp;|&nbsp; 🟡 Thick = ASD critical &nbsp;|&nbsp; "
                 "All arrows shown at column top.  Dotted arcs = governing moments.")
            L_pre = 3.0
            st.plotly_chart(fig_3d(df_si, L_pre, L_pre, cw, cd), use_container_width=True)

            # Envelope chart
            st.divider()
            sec("Load Envelope — All Combinations")
            fig_env = go.Figure()
            for lbl,col_,clr in [("FY (kN)","FY (kN)","#42a5f5"),
                                   ("MZ (kN·m)","MZ (kN·m)","#ef5350"),
                                   ("MX (kN·m)","MX (kN·m)","#ab47bc")]:
                for typ,dash in [("LRFD","solid"),("ASD","dot")]:
                    sub=df_si[df_si["Type"]==typ]
                    fig_env.add_trace(go.Bar(
                        x=sub["LC"], y=sub[col_].abs(),
                        name=f"{typ} |{lbl}|", marker_color=clr,
                        opacity=.8 if typ=="LRFD" else .5,
                        visible=True if lbl=="FY (kN)" else "legendonly"))
            fig_env.update_layout(
                barmode="group",paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"),height=320,
                xaxis=dict(tickangle=-45,tickfont=dict(size=8)),
                legend=dict(bgcolor="#13192b"),
                margin=dict(l=0,r=0,t=20,b=80))
            st.plotly_chart(fig_env, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════════════════════════════════
    with t2:
        sec("③ Critical Combo Selection & Foundation Sizing")
        if "df_si" not in st.session_state:
            st.warning("Parse loads in Tab 1 first.")
        else:
            df_si=st.session_state["df_si"]
            lrfd_r, asd_r = critical_combos(df_si)

            c1,c2=st.columns(2)
            with c1:
                st.markdown('<span class="lrfd-tag">LRFD — Strength Design</span>',
                            unsafe_allow_html=True)
                if lrfd_r is not None:
                    info(f"<b>{lrfd_r['LC']}</b><br>"
                         f"Fy={lrfd_r['FY (kN)']:.2f} kN &nbsp;|&nbsp; "
                         f"Fx={lrfd_r['FX (kN)']:.2f} kN &nbsp;|&nbsp; "
                         f"Fz={lrfd_r['FZ (kN)']:.2f} kN<br>"
                         f"Mx={lrfd_r['MX (kN·m)']:.2f} kN·m &nbsp;|&nbsp; "
                         f"Mz={lrfd_r['MZ (kN·m)']:.2f} kN·m")
                else:
                    st.warning("No LRFD combos found.")
            with c2:
                st.markdown('<span class="asd-tag">ASD — Bearing Pressure Check</span>',
                            unsafe_allow_html=True)
                if asd_r is not None:
                    info(f"<b>{asd_r['LC']}</b><br>"
                         f"Fy={asd_r['FY (kN)']:.2f} kN &nbsp;|&nbsp; "
                         f"Fx={asd_r['FX (kN)']:.2f} kN &nbsp;|&nbsp; "
                         f"Fz={asd_r['FZ (kN)']:.2f} kN<br>"
                         f"Mx={asd_r['MX (kN·m)']:.2f} kN·m &nbsp;|&nbsp; "
                         f"Mz={asd_r['MZ (kN·m)']:.2f} kN·m")
                else:
                    st.warning("No ASD combos found.")

            info(f"Self-weight computed automatically: γ_c={gc} kN/m³ × Lx × Lz × Df={Df} m, "
                 f"added to ASD vertical load for bearing check and subtracted from LRFD for net upward pressure.")

            if st.button("🔩 Run Foundation Sizing — ACI 318-19"):
                if lrfd_r is None or asd_r is None:
                    st.error("Need both LRFD and ASD combos.")
                else:
                    with st.spinner("Computing..."):
                        res=size_footing(
                            P_ult=float(lrfd_r["FY (kN)"]),
                            Mx_ult=float(lrfd_r["MX (kN·m)"]),
                            Mz_ult=float(lrfd_r["MZ (kN·m)"]),
                            P_svc=float(asd_r["FY (kN)"]),
                            Mx_svc=float(asd_r["MX (kN·m)"]),
                            Mz_svc=float(asd_r["MZ (kN·m)"]),
                            qa=qa,fc=fc,fy=fy,cover=cov,
                            col_w=cw,col_d=cd,Df=Df,gc=gc,gs=gs)
                        st.session_state["res"]=res
                        st.success("✓ Sizing complete — see Results tab")

            if "res" in st.session_state:
                res=st.session_state["res"]
                st.divider()
                sec("📏 Footing Dimensions")
                c1,c2,c3,c4=st.columns(4)
                mcard("Length Lx",f"{res['Lx']:.2f} m",c1)
                mcard("Width Lz",f"{res['Lz']:.2f} m",c2)
                mcard("Thickness",f"{res['t_mm']} mm",c3)
                mcard("d effective",f"{res['d_eff']} mm",c4)
                sec("⚖️ Bearing (ASD + Self-Weight)")
                c1,c2,c3,c4=st.columns(4)
                mcard("Wf (self-wt)",f"{res['Wf']:.1f} kN",c1)
                mcard("q max",f"{res['q_max']:.1f} kPa",c2)
                mcard("q allow",f"{qa:.0f} kPa",c3)
                mcard("eX / eZ",f"{res['eX']:.3f} / {res['eZ']:.3f} m",c4)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════════════════════════════════
    with t3:
        sec("④ Detailed Results, Drawings & Export")
        if "res" not in st.session_state:
            st.warning("Run sizing in Tab 2 first.")
        else:
            res=st.session_state["res"]
            # Code checks
            sec("✅ ACI 318-19 Code Checks")
            checks=[
                ("Bearing Pressure (ASD)",      res["q_max"],  qa,           "kPa", res["bearing_ok"]),
                ("Two-Way Punching (LRFD)",      res["Vu2"],    res["Vc2"],   "kN",  res["punch_ok"]),
                ("One-Way Shear X (LRFD)",       res["Vu1x"],   res["Vc1x"],  "kN",  res["shear_x_ok"]),
                ("One-Way Shear Z (LRFD)",       res["Vu1z"],   res["Vc1z"],  "kN",  res["shear_z_ok"]),
            ]
            rows=[]
            for name,D,C,unit,ok in checks:
                rows.append({"Check":name,
                    "Demand":f"{D:.2f} {unit}",
                    "Capacity":f"{C:.2f} {unit}",
                    "DCR":f"{D/(C+1e-9):.3f}",
                    "Status":"PASS ✓" if ok else "FAIL ✗"})
            ck_df=pd.DataFrame(rows)
            def hl(row):
                c="#1b5e20" if "PASS" in row["Status"] else "#b71c1c"
                return [""]*4+[f"background:{c};color:#fff;font-weight:700"]
            st.dataframe(ck_df.style.apply(hl,axis=1),use_container_width=True,hide_index=True)

            sec("🔩 Reinforcement Schedule")
            c1,c2,c3,c4=st.columns(4)
            mcard("Bars — X dir",f"{res['nX']} × ⌀{res['bd']}",c1)
            mcard("Spacing X",f"@ {res['sX']} mm",c2)
            mcard("Bars — Z dir",f"{res['nZ']} × ⌀{res['bd']}",c3)
            mcard("Spacing Z",f"@ {res['sZ']} mm",c4)
            info(f"Bottom mat (LRFD flexure): "
                 f"⌀{res['bd']} @ {res['sX']} mm (X-dir), As={res['As_X']:.0f} mm²  |  "
                 f"⌀{res['bd']} @ {res['sZ']} mm (Z-dir), As={res['As_Z']:.0f} mm²  |  "
                 f"Cover={cov} mm  |  Mu_X={res['Mu_X']:.1f} kN·m  |  Mu_Z={res['Mu_Z']:.1f} kN·m")

            # Drawings
            st.divider()
            cl,cr=st.columns(2)
            with cl: st.plotly_chart(fig_heatmap(res,qa),use_container_width=True)
            with cr: st.plotly_chart(fig_section(res,cw,cov),use_container_width=True)
            st.plotly_chart(fig_plan(res),use_container_width=True)

            # Final 3D
            sec("3D — Final Footing + All Loads")
            if "df_si" in st.session_state:
                st.plotly_chart(
                    fig_3d(st.session_state["df_si"],res["Lx"],res["Lz"],cw,cd),
                    use_container_width=True)

            # Export
            sec("📄 Export")
            export={
                "Parameter":["Lx (m)","Lz (m)","Thickness (mm)","d eff (mm)",
                    "Plan Area (m²)","Self-Weight (kN)",
                    "q_max ASD (kPa)","q_min ASD (kPa)","qa (kPa)",
                    "Bearing (ASD)","Punching (LRFD)","Shear X","Shear Z",
                    "Mu_X (kN·m)","As_X (mm²)","nX","Spacing X (mm)",
                    "Mu_Z (kN·m)","As_Z (mm²)","nZ","Spacing Z (mm)",
                    "Bar dia (mm)","Cover (mm)"],
                "Value":[res["Lx"],res["Lz"],res["t_mm"],res["d_eff"],
                    round(res["A"],2),round(res["Wf"],1),
                    round(res["q_max"],1),round(res["q_min"],1),qa,
                    "PASS" if res["bearing_ok"] else "FAIL",
                    "PASS" if res["punch_ok"] else "FAIL",
                    "PASS" if res["shear_x_ok"] else "FAIL",
                    "PASS" if res["shear_z_ok"] else "FAIL",
                    round(res["Mu_X"],1),round(res["As_X"],0),res["nX"],res["sX"],
                    round(res["Mu_Z"],1),round(res["As_Z"],0),res["nZ"],res["sZ"],
                    res["bd"],cov]}
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as w:
                pd.DataFrame(export).to_excel(w,index=False,sheet_name="Foundation Design")
                if "df_si" in st.session_state:
                    st.session_state["df_si"].to_excel(w,index=False,sheet_name="All Combos (SI)")
                if "df_kip" in st.session_state:
                    st.session_state["df_kip"].to_excel(w,index=False,sheet_name="All Combos (kip)")
            buf.seek(0)
            st.download_button("⬇️ Download Full Report (.xlsx)",data=buf,
                file_name="Foundation_Design_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()
