import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import io

# ====================================================================
# 1. CORE MATHCAD/PYTHON CDP GENERATION LOGIC (from COMat_Generator.py)
# ====================================================================

def generate_cdp_data(E_gpa, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Ref_Length):
    """Generates stress-strain and damage data for Abaqus CDP model."""
    
    # Convert E from GPa to MPa
    E = E_gpa * 1000.0

    # --- Compression Part ---
    # Check strain constraints (from COMat_Generator.py)
    e_lin_end = S_cu / E
    e_par_end = 2 * S_cu / E
    if not (e_lin_end <= e_cu <= e_par_end):
        st.error(f"Ultimate crushing strain (e_cu) must be between {e_lin_end:.4e} and {e_par_end:.4e} for parabolic hardening.")
        return None, None, None, None

    # Parabolic Hardening calculation
    S_c0 = 2 * S_cu - E * e_cu
    e_0 = S_c0 / E

    # Stress-Strain Curve Generation (Compression)
    e_c_lin = np.linspace(0.0, e_0, num=10, endpoint=False)
    e_para = np.linspace(e_0, e_cu, num=10, endpoint=False)
    S_c_lin = E * e_c_lin
    S_c_para = -((S_cu - S_c0) / (e_cu - e_0)**2) * (e_para - e_0) * (e_para - e_0 - 2 * (e_cu - e_0)) + S_c0

    # Weibull Softening
    # Calculate end strain where stress is 0.01*S_cu
    e_wb_end = e_cu + e_60 * np.power(-np.log(0.001 / 0.99), 1 / Alpha)
    e_wb = np.linspace(e_cu, e_wb_end, 30, endpoint=True)
    S_c_wb = S_cu * (0.99 * np.exp(-np.power((e_wb - e_cu) / e_60, Alpha)) + 0.01)

    e_c_total = np.concatenate((e_c_lin, e_para, e_wb))
    S_c_total = np.concatenate((S_c_lin, S_c_para, S_c_wb))
    
    # Abaqus Compression Input (*CONCRETE COMPRESSION HARDENING)
    e_c_pla = np.concatenate((e_para, e_wb))
    S_c_pla = np.concatenate((S_c_para, S_c_wb))
    e_c_inelastic = e_c_pla - S_c_pla / E
    comp_ss_data = np.stack((S_c_pla, e_c_inelastic), axis=1)

    # Abaqus Compression Damage (*CONCRETE COMPRESSION DAMAGE)
    dc = 1.0 - S_c_pla / S_cu
    # Damage only starts after peak stress (e_cu)
    e_cu_index = np.where(e_c_pla <= e_cu)[0][-1] if np.any(e_c_pla <= e_cu) else 0
    dc[:e_cu_index] = 0
    comp_damage_data = np.stack((dc, e_c_inelastic), axis=1)

    # --- Tension Part ---
    e_t0 = S_tu / E
    if e_end <= e_t0:
        st.error(f"End strain (e_end) must be larger than cracking strain ({e_t0:.4e}).")
        return None, None, None, None

    # Stress-Strain Curve Generation (Tension)
    e_t_lin = np.linspace(0.0, e_t0, num=10, endpoint=False)
    e_t_power = np.linspace(e_t0, e_end, num=30, endpoint=True)
    S_t_lin = E * e_t_lin
    # Power Law Tension stiffening
    S_t_power = S_tu * (np.power(np.abs((e_end - e_t_power) / (e_end - e_t0)), Beta))

    e_t_total = np.concatenate((e_t_lin, e_t_power))
    S_t_total = np.concatenate((S_t_lin, S_t_power))

    # Abaqus Tension Input (*CONCRETE TENSION STIFFENING)
    e_t_pla = e_t_power
    S_t_pla = S_t_power

    # Filter out very low stress points (as done in COMat_Generator.py)
    mask = S_t_power > 0.01 * S_tu
    e_t_pla = e_t_power[mask]
    S_t_pla = S_t_power[mask]

    # Convert to Cracking Displacement for Abaqus Input
    e_t_cracking = e_t_pla - S_t_pla / E
    u_t_cracking = e_t_cracking * Ref_Length
    tens_ss_data = np.stack((S_t_pla, u_t_cracking), axis=1)

    # Abaqus Tension Damage (*CONCRETE TENSION DAMAGE)
    dt = 1.0 - S_t_pla / S_tu
    tens_damage_data = np.stack((dt, u_t_cracking), axis=1)

    return (e_c_total, S_c_total, e_t_total, S_t_total,
            comp_ss_data, comp_damage_data, tens_ss_data, tens_damage_data)

# ====================================================================
# 2. FILE GENERATOR LOGIC (from COMat_Generator.py)
# ====================================================================

def create_abaqus_input_file(params, data, is_meter):
    """Formats and writes the Abaqus input file content."""
    
    E_gpa, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Ref_Length, Tension_Recovery, Compression_Recovery = params
    comp_ss_data, comp_damage_data, tens_ss_data, tens_damage_data = data

    E = E_gpa * 1000.0
    Density = 2.4E-9
    Poisson = 0.2
    
    # Adjust units for meters (from COMat_Generator.py)
    if is_meter:
        E_out = E * 1E6
        Density_out = Density * 1E12
        # Data must also be converted to SI units (Pa)
        comp_ss_data[:, 0] = comp_ss_data[:, 0] * 1E6
        tens_ss_data[:, 0] = tens_ss_data[:, 0] * 1E6
    else:
        E_out = E
        Density_out = Density

    # Use a StringIO buffer to write the file content
    output = io.StringIO()

    output.write("**************************\n")
    output.write("*** CDP Mat Definition ***\n")
    output.write("**************************\n")
    output.write(f"** E: {E_gpa:.2f} GPa, S_cu: {S_cu:.2f} MPa, e_cu: {e_cu:.4e}, e_63: {e_60:.4e}, Alpha: {Alpha:.3f} **\n")
    output.write(f"** S_tu: {S_tu:.2f} MPa, e_end: {e_end:.4e}, Beta: {Beta:.3f}, Ref_Length: {Ref_Length:.2f} **\n")
    output.write("*Material, name=CDP\n")
    output.write("*Density\n")
    output.write(f"{Density_out:.6e}\n")
    output.write("*Elastic\n")
    output.write(f"{E_out:.6e}, {Poisson}\n")
    output.write(f"*Concrete Damaged Plasticity, REF LENGTH={Ref_Length}\n")
    # Standard CDP parameters (default values from COMat_Generator.py)
    output.write("40., 0.1, 1.16, 0.66667, 0.001\n")
    
    # Abaqus data tables
    output.write("*Concrete Compression Hardening\n")
    np.savetxt(output, comp_ss_data, fmt='%.6e', delimiter=", ")
    
    output.write("*Concrete Tension Stiffening, type=DISPLACEMENT\n")
    np.savetxt(output, tens_ss_data, fmt='%.6e', delimiter=", ")
    
    output.write(f"*Concrete Compression Damage, tension recovery={Tension_Recovery}\n")
    np.savetxt(output, comp_damage_data, fmt='%.6e', delimiter=", ")
    
    output.write(f"*Concrete Tension Damage, type=DISPLACEMENT, compression recovery={Compression_Recovery}\n")
    np.savetxt(output, tens_damage_data, fmt='%.6e', delimiter=", ")
    
    output.write("*************************\n")
    output.write("*** End of Definition ***\n")
    output.write("*************************\n")

    return output.getvalue()


# ====================================================================
# 3. FIT FUNCTIONALITY (from CoMatFIT.py - Adapted for Streamlit)
# ====================================================================

def guess_delimiter_and_load(uploaded_file):
    """Guess delimiter and load data from Streamlit UploadedFile."""
    try:
        # Read the first line as text
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0) # Reset file pointer

        tab_count = first_line.count('\t')
        comma_count = first_line.count(',')
        delimiter = '\t' if tab_count > comma_count else ','

        # Load data using numpy
        return np.genfromtxt(uploaded_file, delimiter=delimiter)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def fit_cdp_parameters(E_gpa, S_cu, e_cu, S_tu, comp_file, tens_file):
    """Performs regression to fit Alpha, e_60, Beta, e_end."""
    
    E = E_gpa * 1000.0 # Convert to MPa
    CompTarget = guess_delimiter_and_load(comp_file)
    TensTarget = guess_delimiter_and_load(tens_file)

    if CompTarget is None or TensTarget is None:
        return None

    st.subheader("Compression Fit Results")
    e_c_Target, S_c_Target = CompTarget[:,0], CompTarget[:,1]
    
    # Parabolic Hardening determined by E, S_cu, e_cu
    S_c0 = 2 * S_cu - E * e_cu
    e_0 = S_c0 / E

    # Weibull Softening Error Function
    def Error_Comp(params):
        Alpha, e_60 = params
        e_wb = e_c_Target[e_c_Target > e_cu]
        S_c_wb = S_cu * (0.99 * np.exp(-np.power((e_wb - e_cu) / e_60, Alpha)) + 0.01)
        # Interpolate target to smooth out target data error
        Comp_Spline = interpolate.PchipInterpolator(e_c_Target, S_c_Target)
        ErrComp = np.sum(np.power((Comp_Spline(e_wb) - S_c_wb), 2))
        return ErrComp

    # Optimization Setup (Initial Guess and Bounds from CoMatFIT.py)
    Alpha_min, Alpha_max = 0.5, 8.0
    e_60_min, e_60_max = 0.0001, 0.01
    
    np.random.seed(42)
    Alpha_samples = Alpha_min + (Alpha_max - Alpha_min) * np.random.rand(100)
    e_60_samples = e_60_min + (e_60_max - e_60_min) * np.random.rand(100)
    errors_comp = [Error_Comp((Alpha, e_60)) for Alpha, e_60 in zip(Alpha_samples, e_60_samples)]
    best_index = np.argmin(errors_comp)
    initial_point_comp = [Alpha_samples[best_index], e_60_samples[best_index]]
    bounds_comp = [(Alpha_min, Alpha_max), (e_60_min, e_60_max)]

    result_comp = minimize(Error_Comp, initial_point_comp, method='Nelder-Mead', bounds=bounds_comp, options={'maxiter': 10000})
    Alpha_fit, e_60_fit = result_comp.x
    st.write(f"Optimized Alpha: **{Alpha_fit:.3f}**")
    st.write(f"Optimized $\epsilon_{{0.63}}$: **{e_60_fit:.4e}**")
    
    
    st.subheader("Tension Fit Results")
    e_t_Target, S_t_Target = TensTarget[:,0], TensTarget[:,1]
    e_tu = S_tu / E

    # Power Softening Error Function
    def Error_Tens(params):
        Beta, e_end = params
        e_t_power = e_t_Target[e_t_Target > e_tu]
        
        # Guard against zero division in tension softening formula
        if np.isclose(e_end, e_tu) or e_end < e_tu: return 1e9 

        S_t_power = S_tu * (np.power(np.abs((e_end - e_t_power) / (e_end - e_tu)), Beta))
        
        Tens_Spline = interpolate.PchipInterpolator(e_t_Target, S_t_Target)
        ErrTens = np.sum(np.power((Tens_Spline(e_t_power) - S_t_power), 2))
        return ErrTens

    # Optimization Setup
    Beta_min, Beta_max = 1.0, 5.0
    e_end_min, e_end_max = 1.0 * max(e_t_Target), 5.0 * max(e_t_Target)
    
    Beta_samples = Beta_min + (Beta_max - Beta_min) * np.random.rand(100)
    e_end_samples = e_end_min + (e_end_max - e_end_min) * np.random.rand(100)
    errors_tens = [Error_Tens((Beta, e_end)) for Beta, e_end in zip(Beta_samples, e_end_samples)]
    best_index = np.argmin(errors_tens)
    initial_point_tens = [Beta_samples[best_index], e_end_samples[best_index]]
    bounds_tens = [(Beta_min, Beta_max), (e_end_min, e_end_max)]

    result_tens = minimize(Error_Tens, initial_point_tens, method='Nelder-Mead', bounds=bounds_tens, options={'maxiter': 10000})
    Beta_fit, e_end_fit = result_tens.x
    st.write(f"Optimized Beta: **{Beta_fit:.3f}**")
    st.write(f"Optimized $\epsilon_{{end}}$: **{e_end_fit:.4e}**")
    
    return Alpha_fit, e_60_fit, Beta_fit, e_end_fit

# ====================================================================
# 4. STREAMLIT APP STRUCTURE
# ====================================================================

# --- Define the new default values ---
DEFAULT_E = 30.0    # GPa (from CoMatFIT.py)
DEFAULT_SCU = 26.5  # MPa (from CoMatFIT.py)
DEFAULT_ECU = 0.0013 # Strain (from CoMatFIT.py)
DEFAULT_STU = 2.63  # MPa (from CoMatFIT.py)


# Set up the Streamlit page configuration
st.set_page_config(
    page_title="CoMat: Abaqus CDP Input Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Header ---
st.title("🧱 CoMat: Abaqus Concrete Damaged Plasticity (CDP) Generator")
st.markdown("---")

# --- Theory Section (Omitted for brevity, assuming it remains the same) ---
with st.expander("📝 CDP Model Theory & Equations (Based on Input Files)"):
    # ... (Theory Markdown content here)
    st.markdown("""
        This application uses Eurocode-style (compression) and Power-Law (tension) functions 
        to generate the stress-strain and damage evolution curves required for the Abaqus CDP model.
        
        ### Compressive Behavior (Hardening/Softening)
        * **Inelastic Strain (Abaqus Input):**
            $$\epsilon_{c}^{in} = \epsilon_c - \\frac{\sigma_c(\epsilon_c)}{E_{cm}}$$
        
        * **Hardening (Parabolic):** Defined by $E$, $\sigma_{cu}$, and $\epsilon_{cu}$.
        * **Softening (Weibull/Exponential):** Governed by the parameters **$\alpha$** and **$\epsilon_{0.63}$** ($\epsilon_{60}$ in the code).
        
        ### Tensile Behavior (Stiffening/Softening)
        * **Cracking Displacement (Abaqus Input):**
            $$u_{cr} = \left( \epsilon_t - \\frac{\sigma_t(\epsilon_t)}{E_{cm}} \\right) \cdot l_{ref}$$
    """)

# --- Main Tabs ---
tab_manual, tab_fit = st.tabs(["⚙️ Manual Input & Generation", "📈 Fit Parameters from Data"])

# --- Tab 1: Manual Generation ---
with tab_manual:
    st.header("1. Material Parameters")
    
    col_comp, col_tens, col_cdp = st.columns(3)
    
    # 1. Compressive Parameters - USING NEW DEFAULTS
    with col_comp:
        st.subheader("Compression")
        E_gpa = st.number_input("Elastic Modulus, E (GPa)", value=DEFAULT_E, min_value=1.0)
        S_cu = st.number_input("Max Compressive Stress, $\sigma_{cu}$ (MPa)", value=DEFAULT_SCU, min_value=0.1)
        e_cu = st.number_input("Strain at $\sigma_{cu}$, $\epsilon_{cu}$", value=DEFAULT_ECU, format="%.5f", min_value=0.0)
        e_60 = st.number_input("Weibull Softening $\epsilon_{0.63}$ (e_60)", value=0.005, format="%.5f", min_value=0.0)
        Alpha = st.number_input("Weibull Softening $\\alpha$", value=2.0, min_value=0.1)

    # 2. Tensile Parameters - USING NEW DEFAULTS
    with col_tens:
        st.subheader("Tension")
        S_tu = st.number_input("Max Tensile Stress, $\sigma_{tu}$ (MPa)", value=DEFAULT_STU, min_value=0.1)
        e_end = st.number_input("End of Cracking Strain, $\epsilon_{end}$", value=0.002, format="%.5f", min_value=0.0)
        Beta = st.number_input("Power Softening $\\beta$", value=2.0, min_value=1.0)
        Ref_Length = st.number_input("Reference Length, $l_{ref}$", value=1.0, min_value=0.01)

    # 3. CDP Damage and Output Options
    with col_cdp:
        st.subheader("CDP / Output")
        Tension_Recovery = st.number_input("Tension Recovery (Wt)", value=1.0, min_value=0.0, max_value=1.0)
        Compression_Recovery = st.number_input("Compression Recovery (Wc)", value=0.0, min_value=0.0, max_value=1.0)
        is_meter = st.checkbox("Output units in SI (N, m, Pa)", value=False, help="Converts stress units from MPa to Pa and length units from mm to m in the Abaqus input file.")
        st.caption("Abaqus requires the remaining 5 CDP parameters (Dilation angle, Viscosity, etc.) to be defined by default values in the generator script.")
    
    st.markdown("---")
    
    # --- Generation Button Logic (Same as before) ---
    if st.button("Generate CDP Data & File", key="manual_gen", type="primary"):
        results = generate_cdp_data(E_gpa, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Ref_Length)

        if results:
            e_c_total, S_c_total, e_t_total, S_t_total, comp_ss_data, comp_damage_data, tens_ss_data, tens_damage_data = results
            
            # --- Visualization (Same as before) ---
            st.header("2. Generated Stress-Strain & Damage Curves")
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
            
            # 1. Compressive Stress-Strain
            axs[0, 0].plot(e_c_total, S_c_total, 'b-o', label='Compression SS', markersize=3)
            axs[0, 0].set_xlabel('Total Strain ($\\epsilon_c$)')
            axs[0, 0].set_ylabel('Stress ($\sigma_c$ / MPa)')
            axs[0, 0].set_title('Compressive Stress-Strain Curve')
            axs[0, 0].grid(True)
            
            # 2. Compressive Hardening (Abaqus Input)
            axs[0, 1].plot(comp_ss_data[:, 1], comp_ss_data[:, 0], 'r-o', label='Comp. Hardening', markersize=3)
            axs[0, 1].set_xlabel('Inelastic Strain ($\\epsilon_{c}^{in}$)')
            axs[0, 1].set_ylabel('Yield Stress ($\sigma_c$ / MPa)')
            axs[0, 1].set_title('Abaqus Compressive Hardening')
            axs[0, 1].grid(True)
            
            # 3. Tensile Stress-Cracking Displacement (Abaqus Input)
            axs[1, 0].plot(tens_ss_data[:, 1], tens_ss_data[:, 0], 'g-o', label='Tension Stiffening', markersize=3)
            axs[1, 0].set_xlabel('Cracking Displacement ($u_{cr}$ / mm)')
            axs[1, 0].set_ylabel('Stress ($\sigma_t$ / MPa)')
            axs[1, 0].set_title('Abaqus Tension Stiffening (Disp.)')
            axs[1, 0].grid(True)
            
            # 4. Damage Evolution (Compression & Tension)
            axs[1, 1].plot(comp_damage_data[:, 1], comp_damage_data[:, 0], 'r--', label='Comp. Damage ($d_c$)')
            axs[1, 1].plot(tens_damage_data[:, 1], tens_damage_data[:, 0], 'g--', label='Tens. Damage ($d_t$)')
            axs[1, 1].set_xlabel('Inelastic/Cracking Input')
            axs[1, 1].set_ylabel('Damage Parameter (d)')
            axs[1, 1].set_title('Compressive and Tensile Damage')
            axs[1, 1].legend()
            axs[1, 1].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- Abaqus File Download (Same as before) ---
            st.header("3. Download Abaqus Input File")
            params = (E_gpa, S_cu, e_cu, e_60, Alpha, S_tu, e_end, Beta, Ref_Length, Tension_Recovery, Compression_Recovery)
            data = (comp_ss_data, comp_damage_data, tens_ss_data, tens_damage_data)
            abaqus_file_content = create_abaqus_input_file(params, data, is_meter)

            st.download_button(
                label="Download CDP_Mat.inp",
                data=abaqus_file_content,
                file_name="CDP_Mat.inp",
                mime="text/plain"
            )

# --- Tab 2: Fit Parameters from Data ---
with tab_fit:
    st.header("Fit $\\alpha, \epsilon_{0.63}, \\beta, \epsilon_{end}$ from Experimental Data")
    
    col_fit_params, col_fit_data = st.columns(2)

    with col_fit_params:
        st.subheader("Fixed Material Parameters")
        # USING NEW DEFAULTS
        E_fit = st.number_input("Elastic Modulus, E (GPa)", value=DEFAULT_E, min_value=1.0, key="E_fit")
        S_cu_fit = st.number_input("Max Compressive Stress, $\sigma_{cu}$ (MPa)", value=DEFAULT_SCU, min_value=0.1, key="S_cu_fit")
        e_cu_fit = st.number_input("Strain at $\sigma_{cu}$, $\epsilon_{cu}$", value=DEFAULT_ECU, format="%.5f", min_value=0.0, key="e_cu_fit")
        S_tu_fit = st.number_input("Max Tensile Stress, $\sigma_{tu}$ (MPa)", value=DEFAULT_STU, min_value=0.1, key="S_tu_fit")

    with col_fit_data:
        st.subheader("Experimental Data Upload")
        comp_file = st.file_uploader("Upload Compression SS Data (.txt or .csv)", type=['txt', 'csv'], key="comp_file")
        tens_file = st.file_uploader("Upload Tension SS Data (.txt or .csv)", type=['txt', 'csv'], key="tens_file")
        st.markdown("**Note:** Data should be in (Strain, Stress) format, separated by comma or tab.")

    if st.button("Run Parameter Fitting", key="run_fit", type="primary"):
        if comp_file and tens_file:
            with st.spinner('Optimizing parameters...'):
                fitted_params = fit_cdp_parameters(E_fit, S_cu_fit, e_cu_fit, S_tu_fit, comp_file, tens_file)

            if fitted_params:
                Alpha_fit, e_60_fit, Beta_fit, e_end_fit = fitted_params
                st.success("Fitting Complete!")
                
                st.subheader("Use Fitted Parameters for Generation")
                st.info(f"Fitted Alpha: **{Alpha_fit:.3f}**, $\epsilon_{{0.63}}$: **{e_60_fit:.4e}**")
                st.info(f"Fitted Beta: **{Beta_fit:.3f}**, $\epsilon_{{end}}$: **{e_end_fit:.4e}**")
                
                # Automatically populate manual input fields with fitted values for easy generation
                st.markdown("You can now switch to the **Manual Input** tab to generate the Abaqus file using these fitted values.")
            else:
                st.error("Fitting failed. Check your input file format and data range.")
        else:
            st.warning("Please upload both compression and tension data files.")
