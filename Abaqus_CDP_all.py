import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import math

APP_TITLE = "Abaqus CDP Material Generator (Advanced)"

# ---------------------------
# Unit systems & Constants
# ---------------------------
G_IN = 386.089            # in/s^2
UNIT_SYSTEMS = {
    "US (in, lbf, s)": {
        "E_unit": "psi", "density_unit": "lbf·s²/in⁴", "stress_label": "Stress (psi)",
        "len_unit": "in", "to_psi": 1.0, "to_MPa": 0.00689476, "to_ksi": 0.001
    },
    "US (in, kip, s)": { 
        "E_unit": "ksi", "density_unit": "kip·s²/in⁴", "stress_label": "Stress (ksi)",
        "len_unit": "in", "to_psi": 1000.0, "to_MPa": 6.89476, "to_ksi": 1.0
    },
    "SI (m, N, kg, s)": {
        "E_unit": "Pa", "density_unit": "kg/m³", "stress_label": "Stress (Pa)",
        "len_unit": "m", "to_psi": 0.000145038, "to_MPa": 1e-6, "to_ksi": 1.45038e-7
    },
    "SI (mm, N, tonne, s)": {
        "E_unit": "MPa", "density_unit": "tonne/mm³", "stress_label": "Stress (MPa)",
        "len_unit": "mm", "to_psi": 145.038, "to_MPa": 1.0, "to_ksi": 0.145038
    },
}

# ---------------------------
# Helper Functions
# ---------------------------
def pcf_to_density_US(pcf: float, unit_sys: str):
    rho_kg_m3 = pcf * 16.018463
    if unit_sys == "SI (m, N, kg, s)": return rho_kg_m3
    elif unit_sys == "SI (mm, N, tonne, s)": return rho_kg_m3 * 1.0e-12
    else:
        gamma_in3 = pcf / 1728.0
        if unit_sys == "US (in, lbf, s)": return gamma_in3 / G_IN
        elif unit_sys == "US (in, kip, s)": return (gamma_in3 / 1000.0) / G_IN
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
# Stress-Strain Model Equations
# ---------------------------
def calc_sargin_1971(eta, fc, k):
    # Sargin D=1 (Asymptotic)
    numerator = k * eta
    denominator = 1 + (k - 2) * eta + (eta**2)
    return fc * (numerator / denominator)

def calc_eurocode_2(eta, fc, k):
    # EC2 / Sargin D=0 (Parabolic)
    numerator = k * eta - (eta**2)
    denominator = 1 + (k - 2) * eta
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = fc * (numerator / denominator)
    return sig

def calc_carreira_chu(eta, fc, beta):
    # Carreira & Chu Power Law
    numerator = beta * eta
    denominator = beta - 1 + (eta**beta)
    return fc * (numerator / denominator)

def calc_modified_hognestad(eps_arr, fc, eps_0, eps_u):
    sig = np.zeros_like(eps_arr)
    for i, eps in enumerate(eps_arr):
        if eps <= eps_0:
            sig[i] = fc * (2*(eps/eps_0) - (eps/eps_0)**2)
        elif eps <= eps_u:
            slope = (0.85*fc - fc) / (eps_u - eps_0)
            sig[i] = fc + slope * (eps - eps_0)
        else:
            sig[i] = 0.0
    return sig

def apply_cutoff(strain_arr, stress_arr, fc, threshold_ratio=0.05):
    """
    Cuts off the curve if stress drops below (threshold * fc) AFTER the peak.
    """
    peak_idx = np.argmax(stress_arr)
    threshold_stress = threshold_ratio * fc
    
    # Check points after peak
    cutoff_idx = len(stress_arr) 
    for i in range(peak_idx + 1, len(stress_arr)):
        if stress_arr[i] < threshold_stress:
            cutoff_idx = i
            break
            
    return strain_arr[:cutoff_idx], stress_arr[:cutoff_idx]

# ---------------------------
# Main Logic
# ---------------------------
def calculate_cdp_data(inputs):
    fc = inputs['fc'] 
    wc_pcf = inputs['wc_pcf']
    unit_sys = inputs['unit_sys']
    E_out = inputs['E_modulus'] 
    u_props = UNIT_SYSTEMS[unit_sys]
    
    rho_out = pcf_to_density_US(wc_pcf, unit_sys)

    # --- COMPRESSION ---
    eps_c1 = inputs['eps_peak'] 
    eps_cu1 = inputs['eps_u']
    
    # 1. Strain Arrays
    # Generate plenty of points to catch the drop
    eps_c_arr = np.linspace(0, eps_cu1, int(inputs['n_comp_pts']*1.5))
    eta = eps_c_arr / eps_c1
    
    # 2. Parameters
    k_val = 1.05 * E_out * eps_c1 / fc
    
    # Beta (Carreira)
    beta_denom = 1 - (fc / (E_out * eps_c1))
    beta_val = 1.0 / beta_denom if beta_denom > 0.01 else (fc * u_props['to_ksi'] / 4.7)**3 + 1.55

    # 3. Calculate All Curves (for comparison)
    sig_sargin = calc_sargin_1971(eta, fc, k_val)
    sig_ec2 = calc_eurocode_2(eta, fc, k_val)
    sig_cc = calc_carreira_chu(eta, fc, beta_val)
    sig_hog = calc_modified_hognestad(eps_c_arr, fc, eps_c1, eps_cu1)

    # 4. Select Active Model
    model_choice = inputs['comp_model']
    if model_choice == "Sargin (1971)": sig_raw = sig_sargin
    elif model_choice == "Eurocode 2": sig_raw = sig_ec2
    elif model_choice == "Carreira & Chu (1985)": sig_raw = sig_cc
    else: sig_raw = sig_hog
    
    # 5. APPLY CUTOFF (Force stop at 95% drop)
    eps_final, sig_final = apply_cutoff(eps_c_arr, sig_raw, fc, threshold_ratio=0.05)
    
    # Apply Cutoff to comparison curves too for plotting cleanliness
    # (Just basic masking for the plot)
    def clean_plot_data(sig_arr):
        e, s = apply_cutoff(eps_c_arr, sig_arr, fc, 0.05)
        return e, s
        
    e_sarg, s_sarg = clean_plot_data(sig_sargin)
    e_ec2, s_ec2 = clean_plot_data(sig_ec2)
    e_cc, s_cc = clean_plot_data(sig_cc)
    e_hog, s_hog = clean_plot_data(sig_hog)

    # 6. Inelastic Strain & Hardening (Abaqus Input)
    sig_final = np.maximum(sig_final, 0.0)
    c_inel = eps_final - (sig_final / E_out)
    
    # Find yield point (40% fc)
    yield_limit = 0.40 * fc
    idx_start = np.argmax(sig_final >= yield_limit)
    
    # Filter for Abaqus (must be monotonic inelastic strain, positive stress)
    sig_c_abaqus = sig_final[idx_start:]
    inel_c_abaqus = c_inel[idx_start:]
    inel_c_abaqus = np.maximum(inel_c_abaqus, 0.0)
    
    # Enforce strictly increasing or constant inelastic strain
    if len(inel_c_abaqus) > 0: inel_c_abaqus[0] = 0.0
    for i in range(1, len(inel_c_abaqus)):
        if inel_c_abaqus[i] < inel_c_abaqus[i-1]: 
            inel_c_abaqus[i] = inel_c_abaqus[i-1] 
    
    # 7. Compressive Damage
    d_c = np.zeros_like(sig_c_abaqus)
    abq_strains = eps_final[idx_start:]
    for i, s in enumerate(sig_c_abaqus):
        if abq_strains[i] > eps_c1:
            ratio = s / fc
            d_c[i] = min(0.99, max(0.0, 1.0 - ratio))

    # --- TENSION ---
    fc_mpa = fc * u_props['to_MPa']
    # Eurocode 2 Tensile Strength
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

    # --- PACKAGING ---
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
            'e_sarg': e_sarg, 's_sarg': s_sarg,
            'e_ec2': e_ec2, 's_ec2': s_ec2,
            'e_cc': e_cc, 's_cc': s_cc,
            'e_hog': e_hog, 's_hog': s_hog
        },
        'plot_data': {
            'c_eps': eps_final, 'c_sig': sig_final,
            'c_inel_filt': inel_c_filt, 'd_c_filt': d_c_filt,
            't_x': x_tens_arr, 't_sig': sig_t_arr,
            't_x_filt': x_t_filt, 'd_t_filt': d_t_filt,
            'eps_c1': eps_c1, 'fc': fc, 'ft': ft_out
        },
        'calc_info': {
            'k': k_val, 'beta': beta_val, 'ft': ft_out, 'E': E_out
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
st.markdown("""
Generate Abaqus input data using **Sargin, EC2, Carreira-Chu, or Hognestad** models.
**Note:** All models automatically stop when stress drops by **95%** (5% residual strength) to prevent numerical instability.
""")

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
    
    # E Calculation Equations
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
    comp_model = st.radio("Select Model for Abaqus", 
                          ["Sargin (1971)", "Eurocode 2", "Carreira & Chu (1985)", "Modified Hognestad"])
    
    with st.expander("Shape Parameters", expanded=True):
        eps_peak = st.number_input("Strain at peak (ε_c1)", 0.0015, 0.0040, 0.0022, 0.0001, format="%.6f")
        eps_u = st.number_input("Ultimate strain (ε_cu1)", 0.003, 0.10, 0.0200, 0.0005, format="%.6f", help="Curves stops if this strain is reached, OR if stress drops by 95%.")
        n_comp = st.slider("Num Points (Comp)", 10, 300, 60)
        
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

if 'cdp_res_v3' not in st.session_state: st.session_state['cdp_res_v3'] = None
if gen_clicked: st.session_state['cdp_res_v3'] = calculate_cdp_data(input_dict)

tab1, tab2 = st.tabs(["📊 Results & Copy Data", "📝 Theory & Calculations"])

with tab1:
    if st.session_state['cdp_res_v3']:
        res = st.session_state['cdp_res_v3']
        pd_data = res['plot_data']
        comp_data = res['comparison_data']
        u_label = UNIT_SYSTEMS[input_dict['unit_sys']]['E_unit']
        
        # --- COMPARISON PLOT ---
        st.subheader("1. Model Comparison (with 95% Cutoff)")
        
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        ax_comp.plot(comp_data['e_sarg'], comp_data['s_sarg'], label='Sargin (1971) [D=1]', lw=2)
        ax_comp.plot(comp_data['e_ec2'], comp_data['s_ec2'], label='Eurocode 2 [D=0]', lw=2, linestyle='--')
        ax_comp.plot(comp_data['e_cc'], comp_data['s_cc'], label='Carreira & Chu', lw=2, linestyle='-.')
        ax_comp.plot(comp_data['e_hog'], comp_data['s_hog'], label='Hognestad', lw=2, linestyle=':')
        
        ax_comp.scatter([pd_data['eps_c1']], [pd_data['fc']], c='black', zorder=5, label='Peak')
        ax_comp.set_xlabel("Strain"); ax_comp.set_ylabel(f"Stress ({u_label})")
        ax_comp.set_title(f"Compressive Models (f'c = {pd_data['fc']})")
        ax_comp.legend(); ax_comp.grid(True, alpha=0.3)
        st.pyplot(fig_comp)
        
        st.divider()
        st.subheader(f"2. Selected Data: {input_dict['comp_model']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Compressive Hardening**")
            st.code(format_table_for_copy(res['comp_table']), language="csv")
            st.markdown(f"**Compressive Damage**")
            st.code(format_table_for_copy(res['comp_dmg']), language="csv")
        with c2:
            st.markdown(f"**Tensile Stiffening**")
            st.code(format_table_for_copy(res['tens_table']), language="csv")
            st.markdown(f"**Tensile Damage**")
            st.code(format_table_for_copy(res['tens_dmg']), language="csv")
            
        st.download_button("💾 Download .inp file", generate_inp_text(res, input_dict, "CONCRETE"), "concrete.inp")

with tab2:
    st.header("Theory & Parameter Calculations")
    st.markdown("This section details exactly how every parameter in the code is calculated based on your inputs.")
    
    if st.session_state['cdp_res_v3']:
        calc = st.session_state['cdp_res_v3']['calc_info']
        u_e = UNIT_SYSTEMS[input_dict['unit_sys']]['E_unit']
        
        st.subheader("1. Elastic Modulus ($E_c$)")
        st.markdown(f"Selected Method: **{e_method}**")
        if "ACI 318" in e_method:
            st.latex(r"E_c = 57,000 \sqrt{f'_c} \quad \text{(psi)}")
            st.markdown(f"Value Used: **{calc['E']:,.0f} {u_e}**")
        elif "Eurocode" in e_method:
            st.latex(r"E_{cm} = 22,000 \left(\frac{f_{cm}}{10}\right)^{0.3} \quad \text{(MPa)}")
            st.markdown(f"Value Used: **{calc['E']:,.0f} {u_e}**")
            
        st.subheader("2. Shape Factors ($k$ & $\\beta$)")
        st.markdown("Used to control the steepness of the stress-strain curve.")
        
        st.markdown("**A. Eurocode / Sargin Shape Factor ($k$):**")
        st.latex(r"k = 1.05 E_{cm} \frac{\epsilon_{c1}}{f_{cm}}")
        st.markdown(f"Calculated $k$: **{calc['k']:.4f}**")
        
        st.markdown("**B. Carreira & Chu Shape Factor ($\\beta$):**")
        st.latex(r"\beta = \frac{1}{1 - \frac{f'_c}{\epsilon_{c1} E_c}}")
        st.markdown(f"Calculated $\\beta$: **{calc['beta']:.4f}**")
        
        st.subheader("3. Tensile Strength ($f_t$)")
        st.markdown("Derived from Eurocode 2 (EN 1992-1-1) empirical relationships.")
        st.latex(r"f_{ctm} = 0.30 \cdot f_{ck}^{(2/3)} \quad \text{(MPa)}")
        st.markdown(f"Calculated $f_t$: **{calc['ft']:.4f} {u_e}**")
        
    st.divider()
    
    st.header("Constitutive Model References")
    
    st.subheader("1. Sargin (1971) - Asymptotic")
    st.markdown("Selected for Explicit Dynamics to prevent negative stress (D=1).")
    st.latex(r"\sigma_c = f'_c \frac{k \cdot \eta}{1 + (k-2)\eta + \eta^2}")
    
    st.subheader("2. Eurocode 2 (EN 1992-1-1)")
    st.markdown("Standard Design Code formulation (Sargin D=0). Cutoff applied at 95% drop.")
    st.latex(r"\sigma_c = f'_c \frac{k \cdot \eta - \eta^2}{1 + (k-2)\eta}")

    st.subheader("3. Carreira & Chu (1985)")
    st.markdown("Generalized Power Law model.")
    st.latex(r"\sigma_c = f'_c \frac{\beta \cdot \eta}{\beta - 1 + \eta^\beta}")
