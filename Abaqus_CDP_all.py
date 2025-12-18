import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import math

APP_TITLE = "Abaqus CDP Material Generator (Multi-Model)"

# ---------------------------
# Unit systems & Constants
# ---------------------------
G_IN = 386.089            # in/s^2
G_FT = 32.174049          # ft/s^2
G_SI = 9.80665            # m/s^2

UNIT_SYSTEMS = {
    "US (in, lbf, s)": {
        "E_unit": "psi", "density_unit": "lbf·s²/in⁴", "stress_label": "Stress (psi)",
        "len_unit": "in", "to_psi": 1.0, "to_MPa": 0.00689476
    },
    "US (in, kip, s)": { 
        "E_unit": "ksi", "density_unit": "kip·s²/in⁴", "stress_label": "Stress (ksi)",
        "len_unit": "in", "to_psi": 1000.0, "to_MPa": 6.89476
    },
    "SI (m, N, kg, s)": {
        "E_unit": "Pa", "density_unit": "kg/m³", "stress_label": "Stress (Pa)",
        "len_unit": "m", "to_psi": 0.000145038, "to_MPa": 1e-6
    },
    "SI (mm, N, tonne, s)": {
        "E_unit": "MPa", "density_unit": "tonne/mm³", "stress_label": "Stress (MPa)",
        "len_unit": "mm", "to_psi": 145.038, "to_MPa": 1.0
    },
}

# ---------------------------
# Helper Functions
# ---------------------------
def pcf_to_density_US(pcf: float, unit_sys: str):
    rho_kg_m3 = pcf * 16.018463
    
    if unit_sys == "SI (m, N, kg, s)":
        return rho_kg_m3
    elif unit_sys == "SI (mm, N, tonne, s)":
        return rho_kg_m3 * 1.0e-12
    else:
        gamma_in3 = pcf / 1728.0
        if unit_sys == "US (in, lbf, s)": 
            return gamma_in3 / G_IN
        elif unit_sys == "US (in, kip, s)": 
            return (gamma_in3 / 1000.0) / G_IN
            
    return 0.0

def filter_damage_arrays(damage_arr, x_arr):
    pos_indices = np.where(damage_arr > 0)[0]
    if len(pos_indices) == 0:
        return np.array([0.0]), np.array([x_arr[0]])
    first_pos_idx = pos_indices[0]
    start_idx = max(0, first_pos_idx - 1)
    return damage_arr[start_idx:], x_arr[start_idx:]

def format_table_for_copy(data_list):
    return "\n".join([f"{val1:.6g}\t{val2:.6g}" for val1, val2 in data_list])

def snippet_header(name): return f"*MATERIAL, NAME={name}\n"
def snippet_density(val): return f"*DENSITY\n{val:.6g}\n"
def snippet_elastic(E, nu): return f"*ELASTIC\n{E:.6g}, {nu:.6g}\n"

# ---------------------------
# Stress-Strain Models
# ---------------------------
def calc_sargin_1971(eta, fc, k):
    # Sargin (1971) Generalized with D=1 (Asymptotic tail)
    # This avoids negative stresses at high strains
    numerator = k * eta
    denominator = 1 + (k - 2) * eta + (eta**2)
    return fc * (numerator / denominator)

def calc_eurocode_2(eta, fc, k):
    # Eurocode 2 (EN 1992-1-1) / Sargin D=0
    # Parabolic with sharp drop (can go negative)
    numerator = k * eta - (eta**2)
    denominator = 1 + (k - 2) * eta
    # Fix division by zero if any
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = fc * (numerator / denominator)
    sig[sig < 0] = 0.0 # Cutoff negative stresses
    return sig

def calc_carreira_chu(eta, fc, beta):
    # Carreira & Chu (1985) Generalized Power Law
    # sig = fc * [beta*eta] / [beta - 1 + eta^beta]
    numerator = beta * eta
    denominator = beta - 1 + (eta**beta)
    return fc * (numerator / denominator)

def calc_modified_hognestad(eps_arr, fc, eps_0, eps_u):
    # Modified Hognestad
    # Parabola up to eps_0, then linear descent to 0.85fc at eps_u
    sig = np.zeros_like(eps_arr)
    for i, eps in enumerate(eps_arr):
        if eps <= eps_0:
            sig[i] = fc * (2*(eps/eps_0) - (eps/eps_0)**2)
        elif eps <= eps_u:
            # Linear interpolation
            slope = (0.85*fc - fc) / (eps_u - eps_0)
            val = fc + slope * (eps - eps_0)
            sig[i] = max(0.0, val)
        else:
            sig[i] = 0.0 # Failure
    return sig

# ---------------------------
# Main Calculation Logic
# ---------------------------
def calculate_cdp_data(inputs):
    fc = inputs['fc'] 
    wc_pcf = inputs['wc_pcf']
    unit_sys = inputs['unit_sys']
    E_out = inputs['E_modulus'] 
    u_props = UNIT_SYSTEMS[unit_sys]
    
    rho_out = pcf_to_density_US(wc_pcf, unit_sys)

    # --- COMPRESSION SETUP ---
    eps_c1 = inputs['eps_peak'] 
    eps_cu1 = inputs['eps_u']
    
    # 1. Strain Arrays
    eps_c_arr = np.linspace(0, eps_cu1, inputs['n_comp_pts'])
    eta = eps_c_arr / eps_c1
    
    # 2. Model Parameters
    # k for Sargin/EC2
    k_val = 1.05 * E_out * eps_c1 / fc
    
    # Beta for Carreira & Chu
    # Derived definition: beta = 1 / (1 - fc / (E*eps_c1))
    # Note: Beta must be > 1. If E*eps is too close to fc, this spikes.
    # Fallback to empirical if needed, but using constitutive def here:
    beta_denom = 1 - (fc / (E_out * eps_c1))
    if beta_denom > 0.01:
        beta_val = 1.0 / beta_denom
    else:
        beta_val = (fc * u_props['to_ksi'] / 4.7)**3 + 1.55 # Empirical backup (in ksi)

    # 3. Calculate All Curves (for comparison plot)
    sig_sargin = calc_sargin_1971(eta, fc, k_val)
    sig_ec2 = calc_eurocode_2(eta, fc, k_val)
    sig_cc = calc_carreira_chu(eta, fc, beta_val)
    sig_hog = calc_modified_hognestad(eps_c_arr, fc, eps_c1, eps_cu1) # Using eps_c1 as eps_0

    # 4. Select Active Model for Abaqus
    model_choice = inputs['comp_model']
    if model_choice == "Sargin (1971)":
        sig_c_arr = sig_sargin
    elif model_choice == "Eurocode 2":
        sig_c_arr = sig_ec2
    elif model_choice == "Carreira & Chu (1985)":
        sig_c_arr = sig_cc
    else: # Modified Hognestad
        sig_c_arr = sig_hog

    # 5. Inelastic Strain & Hardening
    sig_c_arr = np.maximum(sig_c_arr, 0.0)
    c_inel = eps_c_arr - (sig_c_arr / E_out)
    
    yield_limit = 0.40 * fc
    idx_start = np.argmax(sig_c_arr >= yield_limit)
    sig_c_abaqus = sig_c_arr[idx_start:]
    inel_c_abaqus = c_inel[idx_start:]
    
    inel_c_abaqus = np.maximum(inel_c_abaqus, 0.0)
    if len(inel_c_abaqus) > 0:
        inel_c_abaqus[0] = 0.0
    for i in range(1, len(inel_c_abaqus)):
        if inel_c_abaqus[i] < inel_c_abaqus[i-1]: inel_c_abaqus[i] = inel_c_abaqus[i-1] 
    
    # 6. Compressive Damage
    d_c = np.zeros_like(sig_c_abaqus)
    abq_strains = eps_c_arr[idx_start:]
    for i, s in enumerate(sig_c_abaqus):
        if abq_strains[i] > eps_c1:
            ratio = s / fc
            d_c[i] = min(0.99, max(0.0, 1.0 - ratio))

    # --- TENSION (Belarbi/Hsu) ---
    fc_mpa = fc * u_props['to_MPa']
    ft_mpa = 0.30 * (fc_mpa ** (2.0/3.0))
    ft_out = ft_mpa / u_props['to_MPa']
    
    eps_cr = ft_out / E_out
    max_strain_mult = 150.0 
    eps_t_total = np.linspace(eps_cr, eps_cr * max_strain_mult, inputs['n_tens_pts'])
    
    ratio_eps = eps_cr / eps_t_total
    sig_t_arr = ft_out * (ratio_eps ** 0.4)
    sig_t_arr[0] = ft_out
    
    d_t = 1.0 - (sig_t_arr / ft_out)
    d_t = np.maximum(0.0, np.minimum(0.99, d_t))

    l_char = inputs['l_char'] if inputs['l_char'] > 0 else 1.0
    eps_ck_arr = np.maximum(eps_t_total - (sig_t_arr / E_out), 0.0)
    
    if inputs['tens_type'] == "Strain":
        x_tens_arr = eps_ck_arr
        inp_type = "STRAIN"
        x_label = "Cracking Strain"
    else:
        x_tens_arr = eps_ck_arr * l_char
        inp_type = "DISPLACEMENT"
        x_label = f"Displacement ({u_props['len_unit']})"

    # --- PACKAGING RESULTS ---
    comp_table = list(zip(sig_c_abaqus, inel_c_abaqus))
    tens_table = list(zip(sig_t_arr, x_tens_arr))
    
    d_c_filt, inel_c_filt = filter_damage_arrays(d_c, inel_c_abaqus)
    d_t_filt, x_t_filt = filter_damage_arrays(d_t, x_tens_arr)
    
    if len(inel_c_filt) > 0: inel_c_filt[0] = 0.0

    comp_dmg_table = list(zip(d_c_filt, inel_c_filt))
    tens_dmg_table = list(zip(d_t_filt, x_t_filt))
    
    return {
        'comp_table': comp_table, 'comp_dmg': comp_dmg_table,
        'tens_table': tens_table, 'tens_dmg': tens_dmg_table,
        'rho': rho_out, 'E': E_out,
        'inp_type': inp_type, 'x_label': x_label,
        'comparison_data': {
            'strain': eps_c_arr,
            'Sargin': sig_sargin, 'EC2': sig_ec2,
            'CC': sig_cc, 'Hog': sig_hog
        },
        'plot_data': {
            'c_eps': eps_c_arr, 'c_sig': sig_c_arr,
            'c_inel_filt': inel_c_filt, 'd_c_filt': d_c_filt,
            't_x': x_tens_arr, 't_sig': sig_t_arr,
            't_x_filt': x_t_filt, 'd_t_filt': d_t_filt,
            'eps_c1': eps_c1, 'fc': fc, 'ft': ft_out
        }
    }

def generate_inp_text(res, inputs, name):
    out = io.StringIO()
    out.write(snippet_header(name))
    out.write(snippet_density(res['rho']))
    out.write(snippet_elastic(res['E'], inputs['nu']))
    out.write("*CONCRETE DAMAGED PLASTICITY\n")
    out.write(f"{inputs['dil']:.4f}, {inputs['ecc']:.6f}, {inputs['fb0']:.6f}, {inputs['K']:.6f}, {inputs['visc']:.6f}\n")
    out.write("*CONCRETE COMPRESSION HARDENING\n")
    out.write(format_table_for_copy(res['comp_table']).replace('\t', ', ') + "\n")
    out.write(f"*CONCRETE TENSION STIFFENING, TYPE={res['inp_type']}\n")
    out.write(format_table_for_copy(res['tens_table']).replace('\t', ', ') + "\n")
    out.write("*CONCRETE COMPRESSION DAMAGE\n")
    out.write(format_table_for_copy(res['comp_dmg']).replace('\t', ', ') + "\n")
    out.write(f"*CONCRETE TENSION DAMAGE, TYPE={res['inp_type']}\n")
    out.write(format_table_for_copy(res['tens_dmg']).replace('\t', ', ') + "\n")
    return out.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🧱 " + APP_TITLE)
st.markdown("Generate Abaqus input data comparing **Sargin, EC2, Carreira-Chu, and Modified Hognestad**.")

# Sidebar
with st.sidebar:
    gen_clicked = st.button("🚀 GENERATE DATA", type="primary", use_container_width=True)
    st.divider()

    st.header("1. Settings")
    unit_sys = st.selectbox("Unit System", list(UNIT_SYSTEMS.keys()), index=1)
    u_props = UNIT_SYSTEMS[unit_sys]
    
    st.header("2. Material Inputs")
    def_fc = 5800.0 if "psi" in u_props['E_unit'] else (5.8 if "ksi" in u_props['E_unit'] else 40.0)
    fc = st.number_input(f"f'c (Compressive Strength) [{u_props['E_unit']}]", value=def_fc, format="%.4f")
    wc_pcf = st.number_input("Unit Weight (pcf)", 100.0, 160.0, 145.0, 1.0)
    nu = st.number_input("Poisson's Ratio", 0.15, 0.25, 0.20)
    
    st.subheader("Elastic Modulus (E)")
    e_method = st.selectbox("Calculation Method", ["ACI 318 (Simple)", "ACI 318 (Density)", "Eurocode 2", "Manual Input"])
    
    fc_psi = fc * u_props['to_psi']
    fc_mpa = fc * u_props['to_MPa']
    E_aci_s = (57000.0 * math.sqrt(fc_psi)) / u_props['to_psi']
    E_aci_d = (33.0 * (wc_pcf ** 1.5) * math.sqrt(fc_psi)) / u_props['to_psi']
    E_ec = (22000.0 * ((fc_mpa + 8) / 10.0) ** 0.3) / u_props['to_MPa']
    
    if e_method == "Manual Input":
        E_final = st.number_input(f"E ({u_props['E_unit']})", value=E_aci_s, format="%.2f")
    elif e_method == "ACI 318 (Simple)":
        E_final = E_aci_s; st.info(f"Calc: **{E_final:,.0f}**")
    elif e_method == "ACI 318 (Density)":
        E_final = E_aci_d; st.info(f"Calc: **{E_final:,.0f}**")
    elif e_method == "Eurocode 2":
        E_final = E_ec; st.info(f"Calc: **{E_final:,.0f}**")

    st.header("3. Compressive Model")
    # --- MODEL SELECTION ---
    comp_model = st.radio("Select Model for Abaqus Input", 
                          ["Sargin (1971)", "Eurocode 2", "Carreira & Chu (1985)", "Modified Hognestad"],
                          help="Select which curve to use for the downloadable .inp file.")
    
    with st.expander("Shape Parameters", expanded=True):
        eps_peak = st.number_input("Strain at peak (ε_c1)", 0.0015, 0.0040, 0.0022, 0.0001, format="%.6f")
        eps_u = st.number_input("Ultimate strain (ε_cu1)", 0.003, 0.10, 0.0200, 0.0005, format="%.6f")
        n_comp = st.slider("Num Points (Comp)", 10, 200, 50)
        
    st.header("4. Tension Settings")
    tens_type = st.radio("Abaqus Input Type", ["Strain", "Displacement"], index=0, horizontal=True)
    l_char = 1.0
    if tens_type == "Displacement":
        l_char = st.number_input(f"Char. Length ({u_props['len_unit']})", value=1.0, format="%.4f")
    n_tens = st.slider("Num Points (Tens)", 5, 50, 15)

    st.header("5. Plasticity")
    with st.expander("CDP Parameters"):
        dil = st.number_input("Dilation Angle", value=36.0); ecc = st.number_input("Eccentricity", value=0.1)
        fb0 = st.number_input("fb0/fc0", value=1.16); K_val = st.number_input("K", value=0.667)
        visc = st.number_input("Viscosity", value=0.0)

    input_dict = {'fc': fc, 'wc_pcf': wc_pcf, 'unit_sys': unit_sys, 'E_modulus': E_final, 'nu': nu,
                  'eps_peak': eps_peak, 'eps_u': eps_u, 'n_comp_pts': n_comp, 'comp_model': comp_model,
                  'n_tens_pts': n_tens, 'tens_type': tens_type, 'l_char': l_char,
                  'dil': dil, 'ecc': ecc, 'fb0': fb0, 'K': K_val, 'visc': visc}

if 'cdp_res' not in st.session_state: st.session_state['cdp_res'] = None
if gen_clicked: st.session_state['cdp_res'] = calculate_cdp_data(input_dict)

tab1, tab2 = st.tabs(["📊 Results & Copy Data", "📝 Theory & References"])

with tab1:
    if st.session_state['cdp_res']:
        res = st.session_state['cdp_res']
        pd_data = res['plot_data']
        comp_data = res['comparison_data']
        u_label = UNIT_SYSTEMS[input_dict['unit_sys']]['E_unit']
        
        # --- COMPARISON PLOT ---
        st.subheader("1. Model Comparison")
        st.caption("Comparing four standard constitutive models based on your inputs.")
        
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        ax_comp.plot(comp_data['strain'], comp_data['Sargin'], label='Sargin (1971) [D=1]', lw=2, linestyle='-')
        ax_comp.plot(comp_data['strain'], comp_data['EC2'], label='Eurocode 2 [D=0]', lw=2, linestyle='--')
        ax_comp.plot(comp_data['strain'], comp_data['CC'], label='Carreira & Chu (1985)', lw=2, linestyle='-.')
        ax_comp.plot(comp_data['strain'], comp_data['Hog'], label='Modified Hognestad', lw=2, linestyle=':')
        
        ax_comp.scatter([pd_data['eps_c1']], [pd_data['fc']], c='black', zorder=5, label='Peak Point')
        ax_comp.set_xlabel("Strain"); ax_comp.set_ylabel(f"Stress ({u_label})")
        ax_comp.set_title(f"Compressive Models (f'c = {pd_data['fc']})")
        ax_comp.legend(); ax_comp.grid(True, alpha=0.3)
        st.pyplot(fig_comp)
        
        st.divider()
        st.subheader(f"2. Selected Model: {input_dict['comp_model']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Compressive Hardening** ({u_label} | Inelastic)")
            st.code(format_table_for_copy(res['comp_table']), language="csv")
            st.markdown(f"**Compressive Damage** (Damage | Inelastic)")
            st.code(format_table_for_copy(res['comp_dmg']), language="csv")
        with c2:
            st.markdown(f"**Tensile Stiffening** ({u_label} | {res['x_label']})")
            st.code(format_table_for_copy(res['tens_table']), language="csv")
            st.markdown(f"**Tensile Damage** (Damage | {res['x_label']})")
            st.code(format_table_for_copy(res['tens_dmg']), language="csv")
            
        st.download_button("💾 Download .inp file", generate_inp_text(res, input_dict, "CONCRETE"), "concrete.inp")

with tab2:
    st.header("Constitutive Models & References")
    
    st.subheader("1. Sargin (1971) - Generalized")
    st.markdown("The Generalized Sargin model (often used in Explicit dynamics with $D=1$) provides an asymptotic tail, preventing negative stress at high strains.")
    st.latex(r"\sigma_c = f'_c \frac{k \cdot \eta}{1 + (k-2)\eta + \eta^2}")
    st.markdown("**Reference:** Sargin, M. (1971). *Stress-Strain Relationships for Concrete and the Analysis of Structural Concrete Sections*. Study No. 4, Solid Mechanics Division, University of Waterloo.")

    st.subheader("2. Eurocode 2 (EN 1992-1-1)")
    st.markdown("The standard design code formulation (Sargin with $D=0$). It is parabolic and crosses zero at high strains, which can cause numerical instability in blast simulations.")
    st.latex(r"\sigma_c = f'_c \frac{k \cdot \eta - \eta^2}{1 + (k-2)\eta}")
    st.markdown("**Reference:** *Eurocode 2: Design of concrete structures - Part 1-1: General rules and rules for buildings*. (2004). CEN, Equation 3.14.")

    st.subheader("3. Carreira & Chu (1985)")
    st.markdown("A generalized power-law model that determines the shape factor $\\beta$ based on material properties.")
    st.latex(r"\sigma_c = f'_c \frac{\beta \cdot \eta}{\beta - 1 + \eta^\beta}")
    st.markdown(r"Where $\beta = \frac{1}{1 - f'_c / (\epsilon_{c1} E_c)}$.")
    st.markdown("**Reference:** Carreira, D. J., & Chu, K. H. (1985). *Stress-strain relationship for plain concrete in compression*. ACI Journal, 82(6), 797-804.")

    st.subheader("4. Modified Hognestad")
    st.markdown("The traditional US design model consisting of a parabolic ascent and a linear descent.")
    st.markdown(r"- **Ascending:** $\sigma_c = f'_c [2(\epsilon/\epsilon_0) - (\epsilon/\epsilon_0)^2]$")
    st.markdown(r"- **Descending:** Linear decay to $0.85 f'_c$ at $\epsilon_u$.")
    st.markdown("**Reference:** Hognestad, E. (1951). *A study of combined bending and axial load in reinforced concrete members*. University of Illinois Engineering Experiment Station, Bulletin No. 399.")
