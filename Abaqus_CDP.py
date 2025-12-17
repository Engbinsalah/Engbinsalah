import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import math

APP_TITLE = "Abaqus CDP Material Generator"

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
    gamma_in3 = pcf / 1728.0
    if unit_sys == "US (in, lbf, s)": return gamma_in3 / G_IN
    elif unit_sys == "US (in, kip, s)": return (gamma_in3 / 1000.0) / G_IN
    elif unit_sys == "SI (m, N, kg, s)": return 2400.0
    elif unit_sys == "SI (mm, N, tonne, s)": return 2.4e-9
    return 0.0

def filter_damage_arrays(damage_arr, x_arr):
    # Keep last zero before damage starts
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
# Calculation Logic
# ---------------------------
def calculate_cdp_data(inputs):
    fc = inputs['fc'] 
    wc_pcf = inputs['wc_pcf']
    unit_sys = inputs['unit_sys']
    E_input = inputs['E_modulus'] 
    u_props = UNIT_SYSTEMS[unit_sys]
    
    # Scale Factors
    if u_props['len_unit'] == 'm': scale_disp = 0.001 
    elif u_props['len_unit'] == 'mm': scale_disp = 1.0 
    else: scale_disp = 1/25.4 # inches

    E_out = E_input
    fc_out = fc
    rho_out = pcf_to_density_US(wc_pcf, unit_sys)

    # --- COMPRESSION (Carreira-Chu) ---
    eps_c1 = inputs['eps_peak'] 
    eps_cu1 = inputs['eps_u']
    
    k_val = 1.05 * E_out * eps_c1 / fc_out
    
    eps_c_arr = np.linspace(0, eps_cu1, inputs['n_comp_pts'])
    eta = eps_c_arr / eps_c1
    numerator = k_val * eta
    denominator = 1 + (k_val - 2) * eta + (eta**2)
    sig_c_arr = fc_out * (numerator / denominator)
    sig_c_arr = np.maximum(sig_c_arr, 0.0)

    # Inelastic Strain
    c_inel = eps_c_arr - (sig_c_arr / E_out)
    
    # Abaqus Hardening
    yield_limit = 0.40 * fc_out
    idx_start = np.argmax(sig_c_arr >= yield_limit)
    sig_c_abaqus = sig_c_arr[idx_start:]
    inel_c_abaqus = c_inel[idx_start:]
    
    inel_c_abaqus = np.maximum(inel_c_abaqus, 0.0)
    inel_c_abaqus[0] = 0.0
    for i in range(1, len(inel_c_abaqus)):
        if inel_c_abaqus[i] < inel_c_abaqus[i-1]: inel_c_abaqus[i] = inel_c_abaqus[i-1] 
    
    # Compressive Damage
    d_c = np.zeros_like(sig_c_abaqus)
    abq_strains = eps_c_arr[idx_start:]
    for i, s in enumerate(sig_c_abaqus):
        if abq_strains[i] > eps_c1:
            ratio = s / fc_out
            d_c[i] = min(0.99, max(0.0, 1.0 - ratio))

    # --- TENSION (Hordijk) ---
    fc_psi = fc_out * u_props['to_psi']
    ft_psi = inputs['ft_factor'] * math.sqrt(fc_psi)
    ft_out = ft_psi / u_props['to_psi']
    
    fc_mpa = fc_out * u_props['to_MPa']
    Gf_n_mm = 0.073 * (fc_mpa**0.18)
    
    ft_mpa = ft_out * u_props['to_MPa']
    wc_mm = 5.14 * (Gf_n_mm / ft_mpa)
    wc_out = wc_mm * scale_disp
    
    w_arr = np.linspace(0, wc_out, inputs['n_tens_pts'])
    ratio = w_arr / wc_out
    
    term1 = 1 + (3 * ratio**3)
    term2 = np.exp(-6.93 * ratio)
    sig_t_arr = ft_out * term1 * term2
    sig_t_arr = np.maximum(sig_t_arr, 0.0)
    sig_t_arr[0] = ft_out
    
    d_t = 1.0 - (sig_t_arr / ft_out)
    d_t = np.maximum(0.0, np.minimum(0.99, d_t))

    # --- TENSION TYPE ---
    if inputs['tens_type'] == "Strain":
        l_char = inputs['l_char']
        if l_char <= 0: l_char = 1.0
        # Cracking Strain = w / l_char
        x_tens_arr = w_arr / l_char
        x_label = "Cracking Strain"
        inp_type = "STRAIN"
    else:
        x_tens_arr = w_arr
        x_label = f"Displacement ({u_props['len_unit']})"
        inp_type = "DISPLACEMENT"

    # --- FILTERING ---
    comp_table = list(zip(sig_c_abaqus, inel_c_abaqus))
    tens_table = list(zip(sig_t_arr, x_tens_arr))
    
    d_c_filt, inel_c_filt = filter_damage_arrays(d_c, inel_c_abaqus)
    d_t_filt, x_t_filt = filter_damage_arrays(d_t, x_tens_arr)
    
    comp_dmg_table = list(zip(d_c_filt, inel_c_filt))
    tens_dmg_table = list(zip(d_t_filt, x_t_filt))
    
    return {
        'comp_table': comp_table, 'comp_dmg': comp_dmg_table,
        'tens_table': tens_table, 'tens_dmg': tens_dmg_table,
        'rho': rho_out, 'E': E_out,
        'inp_type': inp_type, 'x_label': x_label,
        'plot_data': {
            'c_eps': eps_c_arr, 'c_sig': sig_c_arr,
            'c_inel_filt': inel_c_filt, 'd_c_filt': d_c_filt,
            't_x': x_tens_arr, 't_sig': sig_t_arr,
            't_x_filt': x_t_filt, 'd_t_filt': d_t_filt,
            'eps_c1': eps_c1, 'fc': fc_out, 'ft': ft_out
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
st.markdown("Generate Abaqus input data using **Carreira-Chu (Compression)** and **Hordijk (Tension)** models.")

# Sidebar
with st.sidebar:
    # 1. GENERATE BUTTON (FIXED TOP)
    gen_clicked = st.button("🚀 GENERATE DATA", type="primary", use_container_width=True)
    st.divider()

    st.header("1. Settings")
    unit_sys = st.selectbox("Unit System", list(UNIT_SYSTEMS.keys()), index=1)
    u_props = UNIT_SYSTEMS[unit_sys]
    
    st.header("2. Material Inputs")
    st.caption(f"Enter values in **{u_props['E_unit']}**")
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
        E_final = E_aci_s; st.info(f"Calculated: **{E_final:,.2f}** {u_props['E_unit']}")
    elif e_method == "ACI 318 (Density)":
        E_final = E_aci_d; st.info(f"Calculated: **{E_final:,.2f}** {u_props['E_unit']}")
    elif e_method == "Eurocode 2":
        E_final = E_ec; st.info(f"Calculated: **{E_final:,.2f}** {u_props['E_unit']}")

    st.header("3. Curve Shape")
    with st.expander("Compression", expanded=True):
        eps_peak = st.number_input("Strain at peak (ε_c1)", 0.0015, 0.0035, 0.0022, 0.0001, format="%.6f")
        eps_u = st.number_input("Ultimate strain (ε_cu1)", 0.003, 0.05, 0.0200, 0.0005, format="%.6f")
        n_comp = st.slider("Num Points (Comp)", 10, 200, 50)
        
    with st.expander("Tension (Hordijk)", expanded=True):
        tens_type = st.radio("Tension Type", ["Strain", "Displacement"], index=0, horizontal=True)
        
        # Smart Default for Characteristic Length
        if u_props['len_unit'] == 'mm': def_l_char = 100.0
        elif u_props['len_unit'] == 'm': def_l_char = 0.1
        else: def_l_char = 4.0 # inches
            
        l_char = 1.0
        if tens_type == "Strain":
            l_char = st.number_input(f"Characteristic Element Length ({u_props['len_unit']})", 
                                     value=def_l_char, format="%.4f",
                                     help=f"Representative finite element size. Used to convert displacement to strain: ε = w / l_char")
        
        ft_fac = st.number_input("ft factor (x sqrt(f'c))", 4.0, 12.0, 6.7)
        n_tens = st.slider("Num Points (Tens)", 5, 50, 15)

    st.header("4. Plasticity")
    with st.expander("CDP Parameters"):
        dil = st.number_input("Dilation Angle", value=36.0, format="%.4f")
        ecc = st.number_input("Eccentricity", value=0.1, format="%.6f")
        fb0 = st.number_input("fb0/fc0", value=1.16, format="%.6f")
        K_val = st.number_input("K", value=0.667, format="%.6f")
        visc = st.number_input("Viscosity", value=0.0, format="%.6f")

    input_dict = {'fc': fc, 'wc_pcf': wc_pcf, 'unit_sys': unit_sys, 'E_modulus': E_final, 'nu': nu,
                  'eps_peak': eps_peak, 'eps_u': eps_u, 'n_comp_pts': n_comp, 
                  'ft_factor': ft_fac, 'n_tens_pts': n_tens, 'tens_type': tens_type, 'l_char': l_char,
                  'dil': dil, 'ecc': ecc, 'fb0': fb0, 'K': K_val, 'visc': visc}

if 'cdp_results_v22' not in st.session_state: st.session_state['cdp_results_v22'] = None
if gen_clicked: st.session_state['cdp_results_v22'] = calculate_cdp_data(input_dict)

tab1, tab2 = st.tabs(["📊 Results & Copy Data", "📝 Theory"])

with tab1:
    if st.session_state['cdp_results_v22']:
        res = st.session_state['cdp_results_v22']
        pd = res['plot_data']
        u_label = UNIT_SYSTEMS[input_dict['unit_sys']]['E_unit']
        
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Compression")
            st.caption(f"Using E = **{res['E']:,.0f} {u_label}**")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
            ax1.plot(pd['c_eps'], pd['c_sig'], label='Total Stress', lw=2)
            ax1.scatter([pd['eps_c1']], [pd['fc']], c='red', zorder=5)
            ax1.text(pd['eps_c1'], pd['fc']*1.02, f"Peak: {pd['fc']:.3f}", color='red', ha='center')
            ax1.set_ylabel(f"Stress ({u_label})"); ax1.set_xlabel("Total Strain"); ax1.grid(True, alpha=0.3)
            ax2.plot(pd['c_inel_filt'], pd['d_c_filt'], c='orange', ls='--', lw=2, marker='.')
            ax2.set_ylabel("Damage (0-1)"); ax2.set_xlabel("Inelastic Strain")
            ax2.set_title("Compression Damage (Filtered)"); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig)
            
        with colB:
            st.subheader("Tension")
            st.caption(f"Type: **{res['inp_type']}**")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
            ax1.plot(pd['t_x'], pd['t_sig'], c='green', label='Hordijk', lw=2)
            ax1.scatter([0], [pd['ft']], c='red', zorder=5)
            ax1.text(0, pd['ft']*1.02, f"Peak: {pd['ft']:.4f}", color='red', ha='left')
            ax1.set_ylabel(f"Stress ({u_label})"); ax1.set_xlabel(res['x_label']); ax1.grid(True, alpha=0.3)
            ax2.plot(pd['t_x_filt'], pd['d_t_filt'], c='purple', ls='--', lw=2, marker='.')
            ax2.set_ylabel("Damage (0-1)"); ax2.set_xlabel(res['x_label'])
            ax2.set_title("Tension Damage (Filtered)"); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig)

        st.divider()
        st.subheader("📋 Tabulated Data")
        st.info("Hover over a block to copy. Tables include the last 0 value plus all positive values.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Comp. Hardening** (Stress | Inelastic)")
            st.code(format_table_for_copy(res['comp_table']), language="csv")
            st.markdown("**Comp. Damage** (Damage | Inelastic)")
            st.code(format_table_for_copy(res['comp_dmg']), language="csv")
        with c2:
            st.markdown(f"**Tens. Stiffening** (Stress | {res['x_label']})")
            st.code(format_table_for_copy(res['tens_table']), language="csv")
            st.markdown(f"**Tens. Damage** (Damage | {res['x_label']})")
            st.code(format_table_for_copy(res['tens_dmg']), language="csv")
        
        st.download_button("💾 Download .inp file", generate_inp_text(res, input_dict, "CONCRETE"), "concrete.inp")

with tab2:
    st.header("Constitutive Theory & References")
    st.markdown("This generator creates input data for the **Concrete Damaged Plasticity (CDP)** model in Abaqus. The stress-strain behavior is derived from established empirical relationships, and damage variables are calculated based on stiffness degradation assumptions.")
    
    st.divider()

    # --- 1. COMPRESSION ---
    st.subheader("1. Compressive Behavior")
    st.markdown("**Stress-Strain Model (Carreira & Chu, 1985):**")
    st.markdown("The compressive stress $\sigma_c$ is calculated as:")
    st.latex(r"\sigma_c = f'_c \frac{k \cdot \eta}{1 + (k-2)\eta + \eta^2}")
    st.markdown("Where:")
    st.latex(r"\eta = \frac{\epsilon_c}{\epsilon_{c1}}, \quad k = 1.05 E_{cm} \frac{\epsilon_{c1}}{f'_c}")
    
    st.markdown("**Inelastic Strain:**")
    st.markdown("Abaqus requires the inelastic strain, not total strain. It is calculated by removing the elastic component:")
    st.latex(r"\tilde{\epsilon}_c^{in} = \epsilon_c - \frac{\sigma_c}{E_0}")

    st.markdown("**Compressive Damage ($d_c$):**")
    st.markdown("In this tool, damage is assumed to evolve linearly with stress loss on the softening branch (post-peak):")
    st.latex(r"d_c = 1 - \frac{\sigma_c}{f'_c} \quad (\text{for } \epsilon_c > \epsilon_{c1})")
    st.caption("*Note: This assumes stiffness degrades proportionally to the loss of load-carrying capacity.*")

    st.divider()

    # --- 2. TENSION ---
    st.subheader("2. Tensile Behavior")
    st.markdown("**Stress-Displacement Model (Hordijk, 1991):**")
    st.markdown("Tensile softening is modeled using a crack opening displacement ($w$) formulation:")
    st.latex(r"\sigma_t = f_{t} \left[ 1 + \left(3 \frac{w}{w_c}\right)^3 \right] \exp\left(-6.93 \frac{w}{w_c}\right)")
    st.markdown("The critical crack opening $w_c$ (where $\sigma_t \to 0$) is derived from fracture energy:")
    st.latex(r"w_c = 5.14 \frac{G_F}{f_{t}}")
    
    st.markdown("**Tensile Damage ($d_t$):**")
    st.markdown("Similar to compression, tensile damage is defined by the ratio of current stress to peak strength:")
    st.latex(r"d_t = 1 - \frac{\sigma_t}{f_t}")

    st.markdown("**Regularization ($l_{char}$):**")
    st.markdown("To minimize mesh dependency, Abaqus allows defining tension by **Displacement** ($w$) or **Cracking Strain** ($\tilde{\epsilon}_t^{ck}$). If 'Strain' is selected, the characteristic length $l_{char}$ (representative element size) is used to convert displacement to strain:")
    st.latex(r"\tilde{\epsilon}_t^{ck} = \frac{w}{l_{char}}")

    st.divider()

    # --- 3. PARAMETERS ---
    st.subheader("3. Material Parameters")
    
    st.markdown("**Fracture Energy ($G_F$):**")
    st.markdown("Estimated using the **CEB-FIP Model Code (1990/2010)** correlation:")
    st.latex(r"G_F = 0.073 \cdot (f'_{c,MPa})^{0.18} \quad \text{(N/mm)}")

    st.markdown("**Elastic Modulus ($E_c$):**")
    st.markdown("Calculated based on the selected standard:")
    st.markdown("- **ACI 318:** $E_c = 57,000 \sqrt{f'_{c, psi}}$ (Simple) or $33 w_c^{1.5} \sqrt{f'_{c, psi}}$ (Density)")
    st.markdown("- **Eurocode 2:** $E_{cm} = 22,000 \cdot [0.1(f'_{c,MPa} + 8)]^{0.3}$")

    st.divider()
    
    # --- REFERENCES ---
    st.caption("**References:**")
    st.caption("1. Carreira, D. J., & Chu, K. H. (1985). *Stress-strain relationship for plain concrete in compression*. ACI Journal.")
    st.caption("2. Hordijk, D. A. (1991). *Local approach to fatigue of concrete*. PhD Thesis, Delft University of Technology.")
    st.caption("3. ACI Committee 318 (2019). *Building Code Requirements for Structural Concrete*.")
    st.caption("4. Dassault Systèmes. *Abaqus Analysis User's Guide*, Section 23.6.3: Concrete Damaged Plasticity.")
