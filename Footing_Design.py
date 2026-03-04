"""
Foundation Design Tool — STAAD RCDC / MAT3D Style
Isolated Footing Sizing with 3D Load Visualization
ACI 318-19 | ASCE 7-22 Methodology
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io, math

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
.metric-value{font-size:1.8rem;font-weight:700;color:#4fc3f7}
.metric-label{font-size:.8rem;color:#90a4ae;text-transform:uppercase;letter-spacing:1px}
.sec-hdr{background:linear-gradient(90deg,#1565c0,#0d47a1);color:#fff;
  padding:8px 16px;border-radius:6px;font-weight:700;font-size:1.05rem;margin:12px 0 8px}
.info-box{background:#0d2137;border-left:4px solid #1565c0;
  padding:10px 14px;border-radius:4px;margin:6px 0;font-size:.88rem}
div[data-testid="stSidebar"]{background:#13192b}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;
  border:none;border-radius:6px;font-weight:700;padding:10px 28px;width:100%}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-hdr">{t}</div>', unsafe_allow_html=True)
def info(t): st.markdown(f'<div class="info-box">{t}</div>', unsafe_allow_html=True)
def mcard(lbl, val, col):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOAD_COLS = ["Fx (kN)", "Fy (kN)", "Fz (kN)", "Mx (kN·m)", "My (kN·m)", "Mz (kN·m)"]

ACI_COMBOS = {
    "1.4DL":              {"DL":1.4,"LL":0.0,"WL":0.0,"EQ":0.0},
    "1.2DL+1.6LL":        {"DL":1.2,"LL":1.6,"WL":0.0,"EQ":0.0},
    "1.2DL+1.0LL+1.6WL":  {"DL":1.2,"LL":1.0,"WL":1.6,"EQ":0.0},
    "1.2DL+1.0LL+1.0EQ":  {"DL":1.2,"LL":1.0,"WL":0.0,"EQ":1.0},
    "0.9DL+1.6WL":        {"DL":0.9,"LL":0.0,"WL":1.6,"EQ":0.0},
    "0.9DL+1.0EQ":        {"DL":0.9,"LL":0.0,"WL":0.0,"EQ":1.0},
    "SLS (1.0DL+1.0LL)":  {"DL":1.0,"LL":1.0,"WL":0.0,"EQ":0.0},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD COMBINATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def apply_combos(load_df: pd.DataFrame) -> pd.DataFrame:
    case_data = {}
    for _, r in load_df.iterrows():
        case_data[str(r["Case"]).strip()] = {
            "Fx": float(r.get("Fx (kN)", 0) or 0),
            "Fy": float(r.get("Fy (kN)", 0) or 0),
            "Fz": float(r.get("Fz (kN)", 0) or 0),
            "Mx": float(r.get("Mx (kN·m)", 0) or 0),
            "My": float(r.get("My (kN·m)", 0) or 0),
            "Mz": float(r.get("Mz (kN·m)", 0) or 0),
        }
    rows = []
    for combo, factors in ACI_COMBOS.items():
        T = {k: 0.0 for k in ["Fx","Fy","Fz","Mx","My","Mz"]}
        for case_type, f in factors.items():
            for cname, vals in case_data.items():
                if case_type.lower() in cname.lower():
                    for k in T: T[k] += f * vals[k]
        rows.append([combo] + [round(T[k], 2) for k in ["Fx","Fy","Fz","Mx","My","Mz"]])
    return pd.DataFrame(rows, columns=["Combo"] + LOAD_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# 3D VISUALISER
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d(load_df, Lx, Lz, col_w, col_d):
    fig = go.Figure()
    hw, hd, ht = Lx/2, Lz/2, 0.6
    # footing mesh
    vx=[-hw,hw,hw,-hw,-hw,hw,hw,-hw]; vy=[-hd,-hd,hd,hd,-hd,-hd,hd,hd]
    vz=[-ht,-ht,-ht,-ht,0,0,0,0]
    I=[0,0,0,4,4,4,0,2,2,0,6,4]; J=[1,2,3,5,6,7,1,3,6,4,7,5]; K=[2,3,0,6,7,4,5,7,5,6,2,1]
    fig.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=I,j=J,k=K,color="#37474f",opacity=.55,name="Footing"))
    # column
    cw,cd,ch=col_w/2,col_d/2,1.5
    cvx=[-cw,cw,cw,-cw,-cw,cw,cw,-cw]; cvy=[-cd,-cd,cd,cd,-cd,-cd,cd,cd]; cvz=[0]*4+[ch]*4
    fig.add_trace(go.Mesh3d(x=cvx,y=cvy,z=cvz,i=I,j=J,k=K,color="#546e7a",opacity=.8,name="Column"))
    # ground plane
    gx=np.linspace(-hw*2,hw*2,4); gy=np.linspace(-hd*2,hd*2,4)
    GX,GY=np.meshgrid(gx,gy); GZ=np.zeros_like(GX)-ht
    fig.add_trace(go.Surface(x=GX,y=GY,z=GZ,colorscale=[[0,"#1a237e"],[1,"#283593"]],
                             opacity=.18,showscale=False,name="Soil"))
    # Critical load row
    vc = "Fy (kN)"
    idx = load_df[vc].abs().idxmax()
    row = load_df.iloc[idx]
    ref = max(abs(row.get("Fy (kN)",0)),abs(row.get("Fx (kN)",0)),abs(row.get("Fz (kN)",0)),1.0)
    sc  = max(Lx,Lz)*0.5/ref
    cz  = ch  # arrow origin at column top

    def arrow(dx,dy,dz,lbl,clr):
        if max(abs(dx),abs(dy),abs(dz))<1e-3: return
        fig.add_trace(go.Scatter3d(x=[0,dx],y=[0,dy],z=[cz,cz+dz],
            mode="lines",line=dict(color=clr,width=5),name=lbl))
        fig.add_trace(go.Cone(x=[dx],y=[dy],z=[cz+dz],u=[dx*.2],v=[dy*.2],w=[dz*.2],
            colorscale=[[0,clr],[1,clr]],showscale=False,
            sizemode="absolute",sizeref=max(Lx,Lz)*.12,showlegend=False))

    arrow(row.get("Fx (kN)",0)*sc,0,0,"Fx","#ef5350")
    arrow(0,row.get("Fz (kN)",0)*sc,0,"Fz","#42a5f5")
    arrow(0,0,-abs(row.get("Fy (kN)",0))*sc,"Fy (Vertical)","#66bb6a")

    ref_m=max(abs(row.get("Mx (kN·m)",0)),abs(row.get("My (kN·m)",0)),
              abs(row.get("Mz (kN·m)",0)),1.0)
    for moment,lbl,clr,axis in [
        (row.get("Mx (kN·m)",0),"Mx","#ff7043","x"),
        (row.get("My (kN·m)",0),"My","#ab47bc","y"),
        (row.get("Mz (kN·m)",0),"Mz","#26c6da","z")]:
        if abs(moment)<1e-3: continue
        r=max(Lx,Lz)*0.38; t_=np.linspace(0,1.5*math.pi,40)
        if axis=="x":   ax_x=np.zeros_like(t_);      ax_y=r*np.cos(t_); ax_z=cz+.4+r*np.sin(t_)
        elif axis=="y": ax_x=r*np.cos(t_);            ax_y=np.zeros_like(t_); ax_z=cz+.4+r*np.sin(t_)
        else:           ax_x=r*np.cos(t_);            ax_y=r*np.sin(t_); ax_z=np.zeros_like(t_)+cz+.3
        fig.add_trace(go.Scatter3d(x=ax_x,y=ax_y,z=ax_z,mode="lines",
            line=dict(color=clr,width=4,dash="dot"),name=f"{lbl}={moment:.1f} kN·m"))

    # dimension annotations
    dim_z=-ht-.1
    fig.add_trace(go.Scatter3d(x=[-hw,hw],y=[-hd-.35,-hd-.35],z=[dim_z,dim_z],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lx={Lx:.2f}m"],textposition="middle right",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[hw+.35,hw+.35],y=[-hd,hd],z=[dim_z,dim_z],
        mode="lines+text",line=dict(color="#ffca28",width=2),
        text=["",f"Lz={Lz:.2f}m"],textposition="middle right",showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            yaxis=dict(title="Y (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            zaxis=dict(title="Z (m)",backgroundcolor="#0d1117",gridcolor="#1e2a3a",showbackground=True),
            bgcolor="#0d1117",camera=dict(eye=dict(x=1.6,y=1.6,z=1.3))),
        paper_bgcolor="#0f1117",font=dict(color="#cfd8dc"),
        legend=dict(bgcolor="#13192b",bordercolor="#263238",font=dict(size=10)),
        height=580,margin=dict(l=0,r=0,t=36,b=0),
        title=dict(text="3D Applied Loads — Critical Combination",font=dict(color="#90caf9",size=13)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FOUNDATION SIZING (ACI 318-19)
# ─────────────────────────────────────────────────────────────────────────────
def size_footing(P_ult,Mx_ult,Mz_ult,P_svc,Mx_svc,Mz_svc,
                 qa,fc,fy,cover,col_w,col_d,Df,gamma_c,gamma_s):
    # ── Preliminary plan size from SLS bearing ─────────────────────────────
    qa_net = qa - gamma_s*Df
    L = math.sqrt(max(P_svc/max(qa_net,1),0.5))
    L = math.ceil(L*10)/10
    for _ in range(15):
        Wf = L*L*Df*gamma_c
        L_new = math.sqrt((P_svc+Wf)/max(qa,1))
        L_new = math.ceil(L_new*10)/10
        if abs(L_new-L)<.05: break
        L = L_new
    eX = abs(Mz_svc)/(P_svc+Wf+1e-9)
    eZ = abs(Mx_svc)/(P_svc+Wf+1e-9)
    Lx = math.ceil(max(L, 6*eX+col_w+.6)*20)/20
    Lz = math.ceil(max(L, 6*eZ+col_d+.6)*20)/20
    A  = Lx*Lz
    Wf = A*Df*gamma_c
    Ptot = P_svc+Wf
    # ── Bearing ────────────────────────────────────────────────────────────
    Sx=Lx**2*Lz/6; Sz=Lx*Lz**2/6
    q_avg = Ptot/A
    q_max = Ptot/A + abs(Mz_svc)/Sx + abs(Mx_svc)/Sz
    q_min = Ptot/A - abs(Mz_svc)/Sx - abs(Mx_svc)/Sz
    bearing_ok = q_max<=qa
    # ── Net upward pressure (ULS) ──────────────────────────────────────────
    qu = P_ult/A   # uniform net factored (conservative)
    phi=0.75
    # ── Two-way shear ──────────────────────────────────────────────────────
    d=300; Vc2=0; Vu2=P_ult
    for _ in range(40):
        b0=2*(col_w*1000+d)+2*(col_d*1000+d)
        Vc2=phi*0.33*math.sqrt(fc)*b0*d/1e6
        punch_area=((col_w+d/1000)*(col_d+d/1000))
        Vu2=P_ult-qu*punch_area
        if Vc2>=Vu2: break
        d+=10
    d_punch=d
    # ── One-way shear ──────────────────────────────────────────────────────
    d=d_punch; Vc1x=0; Vu1x=0; Vc1z=0; Vu1z=0
    for _ in range(40):
        dx=max(Lx/2-col_w/2-d/1000,0)
        dz=max(Lz/2-col_d/2-d/1000,0)
        Vu1x=qu*Lz*dx; Vc1x=phi*0.17*math.sqrt(fc)*(Lz*1000)*d/1e6
        Vu1z=qu*Lx*dz; Vc1z=phi*0.17*math.sqrt(fc)*(Lx*1000)*d/1e6
        if Vc1x>=Vu1x and Vc1z>=Vu1z: break
        d+=10
    d_req=max(d_punch,d)
    t_mm=math.ceil((d_req+cover+12)/50)*50
    d_eff=t_mm-cover-12
    # ── Flexure ────────────────────────────────────────────────────────────
    lx_c=max(Lx/2-col_w/2,0.05); lz_c=max(Lz/2-col_d/2,0.05)
    Mu_X=qu*Lz*lx_c**2/2*1e6  # N·mm
    Mu_Z=qu*Lx*lz_c**2/2*1e6
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
    )

# ─────────────────────────────────────────────────────────────────────────────
# SOIL PRESSURE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(res, qa):
    Lx,Lz=res["Lx"],res["Lz"]
    qmax,qmin=res["q_max"],res["q_min"]
    x=np.linspace(-Lx/2,Lx/2,40); z=np.linspace(-Lz/2,Lz/2,40)
    X,Z=np.meshgrid(x,z)
    Q=(qmax+qmin)/2+(qmax-qmin)/Lx*X
    fig=go.Figure(go.Heatmap(z=Q,x=x,y=z,colorscale="RdYlGn_r",
        zmin=max(0,qmin-5),zmax=qa*1.05,
        colorbar=dict(title="q (kPa)",tickfont=dict(color="#cfd8dc"),
                      titlefont=dict(color="#cfd8dc"))))
    fig.add_shape(type="rect",x0=-.25,y0=-.25,x1=.25,y1=.25,
        line=dict(color="#ffca28",width=2),fillcolor="rgba(255,202,40,.15)")
    fig.update_layout(
        title="Soil Contact Pressure (kPa)",
        xaxis_title="X (m)",yaxis_title="Z (m)",
        paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"),height=380,margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SECTION SKETCH
# ─────────────────────────────────────────────────────────────────────────────
def fig_section(res, col_w, cover):
    Lx=res["Lx"]; t=res["t_mm"]/1000; d=res["d_eff"]/1000
    cover_m=(res["t_mm"]-res["d_eff"]-16)/1000
    fig=go.Figure()
    fig.add_shape(type="rect",x0=-Lx/2,y0=-t,x1=Lx/2,y1=0,
        fillcolor="#37474f",line=dict(color="#90a4ae",width=1.5))
    fig.add_shape(type="rect",x0=-col_w/2,y0=0,x1=col_w/2,y1=1.2,
        fillcolor="#546e7a",line=dict(color="#90a4ae",width=1.5))
    fig.add_shape(type="line",x0=-Lx/2+.02,y0=-(t-cover_m),
        x1=Lx/2-.02,y1=-(t-cover_m),line=dict(color="#ef5350",width=1,dash="dash"))
    # rebar dots
    n=min(8,res["nX"])
    xs=np.linspace(-Lx/2+.1,Lx/2-.1,n)
    fig.add_trace(go.Scatter(x=xs,y=[-(t-cover_m)]*n,mode="markers",
        marker=dict(symbol="circle",size=9,color="#42a5f5"),name=f"⌀{res['bd']} bars"))
    # dimensions
    for ann in [
        dict(x=0,y=-t-.18,text=f"Lx = {Lx:.2f} m",xanchor="center"),
        dict(x=Lx/2+.1,y=-t/2,text=f"t = {res['t_mm']} mm",textangle=-90),
        dict(x=0,y=-(t-cover_m)+.08,text=f"d = {res['d_eff']} mm",
             font=dict(color="#ef5350",size=10),xanchor="center"),
    ]:
        fig.add_annotation(showarrow=False,font=dict(color="#ffca28",size=11),**ann)
    fig.add_shape(type="line",x0=-Lx/2-.3,y0=0,x1=Lx/2+.3,y1=0,
        line=dict(color="#66bb6a",width=2,dash="dot"))
    fig.add_annotation(x=Lx/2+.25,y=.06,text="GL",showarrow=False,
        font=dict(color="#66bb6a",size=11))
    fig.update_layout(
        title="Section Elevation",
        xaxis=dict(showgrid=False,zeroline=False,range=[-Lx/2-.5,Lx/2+.5]),
        yaxis=dict(showgrid=False,zeroline=False,scaleanchor="x",scaleratio=1,
                   range=[-t-.4,1.6]),
        paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
        font=dict(color="#cfd8dc"),height=380,margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <h1 style='margin-bottom:0'>🏗️ Foundation Design Tool</h1>
    <p style='color:#78909c;margin-top:2px;font-size:.95rem'>
    Isolated Footing &nbsp;|&nbsp; ACI 318-19 / ASCE 7-22 &nbsp;|&nbsp;
    STAAD RCDC + MAT3D Logic &nbsp;|&nbsp; 3D Load Visualisation
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
        qa  = st.number_input("Allowable Bearing (kPa)",50.,1000.,200.,10.)
        Df  = st.number_input("Footing Depth Df (m)",.5,4.,1.5,.1)
        cov = st.number_input("Clear Cover (mm)",50,100,75,5)
        st.markdown("### 🏛️ Column")
        cw  = st.number_input("Column bx (m)",.2,1.5,.5,.05)
        cd  = st.number_input("Column bz (m)",.2,1.5,.5,.05)

    # ── TABS ─────────────────────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs(["📂 Load Input","🔩 Combinations","📐 Sizing","📊 Results"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════════════════════════════════
    with t1:
        sec("① Load Case Entry")
        method = st.radio("Input method",
            ["📋 Manual Entry","📁 Upload Excel (.xlsx)"], horizontal=True)

        if method == "📁 Upload Excel (.xlsx)":
            up = st.file_uploader("Upload your Foundation_Loads.xlsx", type=["xlsx","xls"])
            if up:
                try:
                    xl = pd.ExcelFile(up)
                    sheet = xl.sheet_names[0]
                    raw = pd.read_excel(up, sheet_name=sheet, header=None)
                    hrow=0
                    for i,row in raw.iterrows():
                        rs=" ".join([str(v).lower() for v in row.values])
                        if any(k in rs for k in ["case","load","fx","fy","fz","mx"]): hrow=i; break
                    df=pd.read_excel(up,sheet_name=sheet,header=hrow)
                    df.columns=[str(c).strip() for c in df.columns]
                    df=df.dropna(how="all")
                    st.success(f"✓ Sheet '{sheet}' loaded — {len(df)} rows")
                    st.dataframe(df, use_container_width=True)
                    cols=df.columns.tolist()
                    c1,c2,c3=st.columns(3)
                    def pick(label,hints,c):
                        idx=next((i for i,x in enumerate(cols)
                                  if any(h in x.lower() for h in hints)),0)
                        return c.selectbox(label,cols,index=idx)
                    case_c=pick("Case col",    ["case","name","load"],c1)
                    fy_c  =pick("Fy — Vertical",["fy","vertical","p","fv"],c1)
                    fx_c  =pick("Fx",           ["fx"],c2)
                    fz_c  =pick("Fz",           ["fz"],c2)
                    mx_c  =pick("Mx",           ["mx"],c3)
                    my_c  =pick("My",           ["my"],c3)
                    mz_c  =pick("Mz",           ["mz"],c3)
                    ldf=df[[case_c,fy_c,fx_c,fz_c,mx_c,my_c,mz_c]].copy()
                    ldf.columns=["Case","Fy (kN)","Fx (kN)","Fz (kN)",
                                 "Mx (kN·m)","My (kN·m)","Mz (kN·m)"]
                    ldf=ldf.dropna(subset=["Fy (kN)"])
                    st.session_state["ldf"]=ldf
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            default={"Case":["DL","LL","WLx","WLz","EQx","EQz"],
                     "Fy (kN)":[1200,600,50,50,80,80],
                     "Fx (kN)":[0,0,120,0,150,0],
                     "Fz (kN)":[0,0,0,120,0,150],
                     "Mx (kN·m)":[0,0,0,180,0,220],
                     "My (kN·m)":[0,0,0,0,0,0],
                     "Mz (kN·m)":[0,0,180,0,220,0]}
            if "ldf" not in st.session_state:
                st.session_state["ldf"]=pd.DataFrame(default)
            info("Sign convention: <b>Fy = vertical (gravity) positive downward</b>.  "
                 "Fx, Fz = horizontal.  Moments per right-hand rule.")
            edited=st.data_editor(st.session_state["ldf"],use_container_width=True,
                num_rows="dynamic",
                column_config={"Case":st.column_config.TextColumn(width="medium"),
                    **{c:st.column_config.NumberColumn(format="%.2f") for c in LOAD_COLS}},
                key="le")
            st.session_state["ldf"]=edited

        if "ldf" in st.session_state and len(st.session_state["ldf"])>0:
            st.divider()
            sec("② 3D Load Visualisation (preliminary geometry)")
            ldf=st.session_state["ldf"]
            L_pre=3.0
            res_pre={"Lx":L_pre,"Lz":L_pre}
            st.plotly_chart(fig_3d(ldf,L_pre,L_pre,cw,cd),use_container_width=True)
            info("🔴 Fx (X-horiz) &nbsp;|&nbsp; 🔵 Fz (Z-horiz) &nbsp;|&nbsp; "
                 "🟢 Fy (vertical) &nbsp;|&nbsp; Dotted arcs = Moments &nbsp;|&nbsp; "
                 "Yellow = dimensions")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════════════════════════════════
    with t2:
        sec("③ Load Combinations — ACI 318-19 / ASCE 7-22")
        if "ldf" not in st.session_state:
            st.warning("Enter load cases in Tab 1 first.")
        else:
            cdf=apply_combos(st.session_state["ldf"])
            st.session_state["cdf"]=cdf
            info("<b>ULS combos</b> (factored) used for shear & flexure design. "
                 "<b>SLS combo</b> (1.0DL+1.0LL) used for bearing pressure check.")
            st.dataframe(
                cdf.style.format({c:"{:.2f}" for c in LOAD_COLS})
                         .background_gradient(subset=["Fy (kN)"],cmap="Blues"),
                use_container_width=True,height=290)
            fig_b=px.bar(cdf,x="Combo",y="Fy (kN)",color="Fy (kN)",
                color_continuous_scale="Blues",
                title="Factored Vertical Load per Combination")
            fig_b.update_layout(paper_bgcolor="#0f1117",plot_bgcolor="#0d1117",
                font=dict(color="#cfd8dc"),height=320,xaxis_tickangle=-30,
                showlegend=False,margin=dict(l=0,r=0,t=40,b=60))
            st.plotly_chart(fig_b,use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════════════════════════════════
    with t3:
        sec("④ Foundation Sizing — ACI 318-19")
        if "cdf" not in st.session_state:
            st.warning("Complete Tab 2 first.")
        else:
            cdf=st.session_state["cdf"]
            uls_idx=cdf["Fy (kN)"].idxmax()
            sls_mask=cdf["Combo"].str.contains("SLS|1\\.0DL",case=False)
            sls_idx=cdf[sls_mask].index[0] if sls_mask.any() else cdf["Fy (kN)"].idxmin()
            ur=cdf.iloc[uls_idx]; sr=cdf.iloc[sls_idx]
            c1,c2=st.columns(2)
            c1.info(f"**Critical ULS:** `{ur['Combo']}`  —  Fy = **{ur['Fy (kN)']:.1f} kN**")
            c2.info(f"**Bearing (SLS):** `{sr['Combo']}`  —  Fy = **{sr['Fy (kN)']:.1f} kN**")
            info(f"Footing self-weight will be computed automatically using γ_c = {gc} kN/m³ "
                 f"and Df = {Df} m and added to the vertical load for bearing check.")
            if st.button("🔩 Run Foundation Sizing"):
                with st.spinner("Computing..."):
                    res=size_footing(
                        P_ult=float(ur["Fy (kN)"]),
                        Mx_ult=float(ur["Mx (kN·m)"]),Mz_ult=float(ur["Mz (kN·m)"]),
                        P_svc=float(sr["Fy (kN)"]),
                        Mx_svc=float(sr["Mx (kN·m)"]),Mz_svc=float(sr["Mz (kN·m)"]),
                        qa=qa,fc=fc,fy=fy,cover=cov,
                        col_w=cw,col_d=cd,Df=Df,gamma_c=gc,gamma_s=gs)
                    st.session_state["res"]=res
            if "res" in st.session_state:
                res=st.session_state["res"]
                st.divider()
                sec("📏 Final Footing Dimensions")
                c1,c2,c3,c4=st.columns(4)
                mcard("Length Lx",f"{res['Lx']:.2f} m",c1)
                mcard("Width Lz",f"{res['Lz']:.2f} m",c2)
                mcard("Thickness",f"{res['t_mm']} mm",c3)
                mcard("d effective",f"{res['d_eff']} mm",c4)
                sec("⚖️ Bearing (incl. self-weight)")
                c1,c2,c3,c4=st.columns(4)
                mcard("Self-Weight Wf",f"{res['Wf']:.1f} kN",c1)
                mcard("q max",f"{res['q_max']:.1f} kPa",c2)
                mcard("q allow",f"{qa:.0f} kPa",c3)
                mcard("eX / eZ",f"{res['eX']:.2f} / {res['eZ']:.2f} m",c4)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4
    # ════════════════════════════════════════════════════════════════════════
    with t4:
        sec("⑤ Detailed Results, Drawings & Export")
        if "res" not in st.session_state:
            st.warning("Run sizing in Tab 3 first.")
        else:
            res=st.session_state["res"]
            # ── Code checks table ─────────────────────────────────────────
            sec("✅ Code Checks — ACI 318-19")
            checks=[
                ("Bearing Pressure",        res["q_max"],  qa,          "kPa", res["bearing_ok"]),
                ("Two-Way Punching Shear",   res["Vu2"],    res["Vc2"],  "kN",  res["punch_ok"]),
                ("One-Way Shear (X-dir)",    res["Vu1x"],   res["Vc1x"], "kN",  res["shear_x_ok"]),
                ("One-Way Shear (Z-dir)",    res["Vu1z"],   res["Vc1z"], "kN",  res["shear_z_ok"]),
            ]
            rows=[]
            for name,D,C,unit,ok in checks:
                rows.append({"Check":name,
                    "Demand":f"{D:.1f} {unit}",
                    "Capacity φVc":f"{C:.1f} {unit}",
                    "Utilization":f"{D/(C+1e-9)*100:.1f} %",
                    "Status":"PASS ✓" if ok else "FAIL ✗"})
            ck_df=pd.DataFrame(rows)
            def hl(row):
                c="#1b5e20" if "PASS" in row["Status"] else "#b71c1c"
                return [""]*4+[f"background:{c};color:#fff;font-weight:700"]
            st.dataframe(ck_df.style.apply(hl,axis=1),use_container_width=True,hide_index=True)

            # ── Reinforcement ─────────────────────────────────────────────
            sec("🔩 Reinforcement Schedule")
            c1,c2,c3,c4=st.columns(4)
            mcard("Bars — X dir",f"{res['nX']} × ⌀{res['bd']}",c1)
            mcard("Spacing X",f"@ {res['sX']} mm",c2)
            mcard("Bars — Z dir",f"{res['nZ']} × ⌀{res['bd']}",c3)
            mcard("Spacing Z",f"@ {res['sZ']} mm",c4)
            info(f"Bottom mat: ⌀{res['bd']} @ {res['sX']} mm (X-dir)  +  "
                 f"⌀{res['bd']} @ {res['sZ']} mm (Z-dir)  |  "
                 f"Cover = {cov} mm  |  As_X = {res['As_X']:.0f} mm²  |  As_Z = {res['As_Z']:.0f} mm²")

            # ── Drawings ──────────────────────────────────────────────────
            st.divider()
            cl,cr=st.columns(2)
            with cl: st.plotly_chart(fig_heatmap(res,qa),use_container_width=True)
            with cr: st.plotly_chart(fig_section(res,cw,cov),use_container_width=True)

            # ── Final 3D ──────────────────────────────────────────────────
            sec("3D Loads on Final Footing Geometry")
            if "ldf" in st.session_state:
                st.plotly_chart(fig_3d(st.session_state["ldf"],
                    res["Lx"],res["Lz"],cw,cd),use_container_width=True)

            # ── Export ────────────────────────────────────────────────────
            sec("📄 Export Design Report")
            export={"Parameter":[
                "Footing Lx (m)","Footing Lz (m)","Thickness (mm)","d eff (mm)",
                "Plan Area (m²)","Self-Weight (kN)",
                "q_max (kPa)","q_min (kPa)","Allowable q (kPa)",
                "Bearing","Punching Shear","Shear X","Shear Z",
                "Mu_X (kN·m)","As_X (mm²)","nX bars",
                "Mu_Z (kN·m)","As_Z (mm²)","nZ bars",
                "Bar dia (mm)","Spacing X (mm)","Spacing Z (mm)"],
                "Value":[
                res["Lx"],res["Lz"],res["t_mm"],res["d_eff"],
                round(res["A"],2),round(res["Wf"],1),
                round(res["q_max"],1),round(res["q_min"],1),qa,
                "PASS" if res["bearing_ok"] else "FAIL",
                "PASS" if res["punch_ok"] else "FAIL",
                "PASS" if res["shear_x_ok"] else "FAIL",
                "PASS" if res["shear_z_ok"] else "FAIL",
                round(res["Mu_X"],1),round(res["As_X"],0),res["nX"],
                round(res["Mu_Z"],1),round(res["As_Z"],0),res["nZ"],
                res["bd"],res["sX"],res["sZ"]]}
            ex_df=pd.DataFrame(export)
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as w:
                ex_df.to_excel(w,index=False,sheet_name="Foundation Design")
                if "ldf" in st.session_state:
                    st.session_state["ldf"].to_excel(w,index=False,sheet_name="Load Cases")
                if "cdf" in st.session_state:
                    st.session_state["cdf"].to_excel(w,index=False,sheet_name="Load Combinations")
            buf.seek(0)
            st.download_button("⬇️ Download Full Report (.xlsx)",data=buf,
                file_name="Foundation_Design_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
