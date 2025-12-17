import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

APP_TITLE = "Ramberg-Osgood Material Generator"

# ---------------------------
# Unit Systems
# ---------------------------
UNIT_SYSTEMS = {
    "SI (MPa)": {"E_unit": "MPa", "stress_label": "Stress (MPa)"},
    "SI (Pa)":  {"E_unit": "Pa", "stress_label": "Stress (Pa)"},
    "US (psi)": {"E_unit": "psi", "stress_label": "Stress (psi)"},
    "US (ksi)": {"E_unit": "ksi", "stress_label": "Stress (ksi)"},
}

def snippet_header(name): return f"*MATERIAL, NAME={name}\n"
def snippet_elastic(E, nu): return f"*ELASTIC\n{E:.6g}, {nu:.6g}\n"
def format_table_for_copy(data_list):
    # Changed from "{val1:.6f}, {val2:.6f}" to "{val1:.6f} {val2:.6f}"
    return "\n".join([f"{val1:.2f} {val2:.4f}" for val1, val2 in data_list])

# ---------------------------
# Calculation Logic
# ---------------------------
def calculate_ramberg_osgood(inputs):
    E = inputs['E']
    S_y = inputs['S_y']
    n = inputs['n']
    alpha = inputs['alpha']
    
    # 1. Define Stress Range
    # Start exactly at Yield (where plastic strain is small/defined by offset)
    # Go up to a user defined max stress or strain limit estimate
    # We'll calculate until a high strain is reached (e.g. 20%)
    
    # Create stress array: from 0.1*Sy to 1.5*Sy (or higher depending on n)
    # We strictly need pairs starting from first yield for Abaqus *PLASTIC
    
    # Generate points
    # We iterate on Stress to ensure smooth curve
    # Limit max stress to avoid runaway values if n is low
    max_stress_factor = inputs.get('max_stress_factor', 2.0)
    stress_points = np.linspace(S_y, max_stress_factor * S_y, 50)
    
    # Ramberg-Osgood Equation:
    # eps_total = (sigma / E) + alpha * (sigma_y / E) * (sigma / sigma_y)^n
    
    # Plastic Strain component only:
    # eps_plastic = alpha * (sigma_y / E) * (sigma / sigma_y)^n
    
    plastic_strains = []
    stress_output = []
    
    # Add Initial Yield Point (Plastic Strain = 0.0)
    # Abaqus requires the first point of *PLASTIC to be (YieldStress, 0.0)
    # The R-O model implies some plastic strain AT yield (the offset).
    # To satisfy Abaqus, we shift the curve or start at 0.
    # Standard practice: Define first point as (S_y, 0.0)
    plastic_strains.append(0.0)
    stress_output.append(S_y)
    
    # Calculate subsequent points
    for sigma in stress_points[1:]:
        # Formula: eps_p = alpha * (S_y/E) * (sigma/S_y)^n
        # We subtract the offset at yield to make the curve start at 0 plastic strain relative to S_y
        # OR we just output the raw R-O values. 
        # Usually, users want the raw R-O curve starting from the offset point.
        
        term_p = alpha * (S_y / E) * ((sigma / S_y) ** n)
        
        # Calculate offset at yield to shift (optional, but cleaner for FEA initialization)
        offset_at_yield = alpha * (S_y / E) * ((S_y / S_y) ** n) # = alpha * S_y / E
        
        # Shifted Plastic Strain (so at sigma=Sy, eps_p=0)
        # eps_p_shifted = term_p - offset_at_yield
        
        # However, R-O is often used to model the 'knee'. 
        # If we strict enforce (Sy, 0), we might lose the smooth transition if n is low.
        # Let's use the raw value but ensure the first point is (S_y, 0).
        # We will simply discard points where sigma < S_y (elastic).
        
        # To make it compatible with Abaqus *PLASTIC:
        # We will use the calculated plastic strain.
        # But we must insert (S_y, 0) as the very first line.
        
        if term_p > 0:
            plastic_strains.append(term_p)
            stress_output.append(sigma)
            
    total_strains = np.array(stress_output)/E + np.array(plastic_strains)
    
    return {
        'stress': stress_output,
        'plastic_strain': plastic_strains,
        'total_strain': total_strains,
        'E': E, 'S_y': S_y
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("📈 " + APP_TITLE)
st.markdown("""
Generate **True Stress** vs **Plastic Strain** data for Abaqus using the Ramberg-Osgood equation.
Useful when experimental data is unavailable.
""")

st.latex(r"\varepsilon = \frac{\sigma}{E} + \alpha \frac{\sigma_y}{E} \left( \frac{\sigma}{\sigma_y} \right)^n")

with st.sidebar:
    st.header("1. Parameters")
    unit_sys = st.selectbox("Unit System", list(UNIT_SYSTEMS.keys()), index=0)
    props = UNIT_SYSTEMS[unit_sys]
    
    E = st.number_input(f"Modulus (E) [{props['E_unit']}]", value=210000.0 if "MPa" in unit_sys else 29000.0)
    nu = st.number_input("Poisson's Ratio", value=0.3)
    S_y = st.number_input(f"Yield Strength ($\sigma_y$) [{props['E_unit']}]", value=250.0 if "MPa" in unit_sys else 36.0)
    
    st.divider()
    st.subheader("Shape Parameters")
    n = st.number_input("Hardening Exponent ($n$)", value=5.0, min_value=1.0, help="Typical steel: 4-8. Higher values = more 'perfectly plastic'.")
    
    calc_alpha = st.checkbox("Calculate Alpha from Offset?", value=True)
    if calc_alpha:
        offset = st.number_input("Yield Offset (Strain)", value=0.002, format="%.4f")
        # alpha = offset / (Sy/E)
        alpha_val = offset / (S_y / E)
        st.caption(f"Calculated $\\alpha$: **{alpha_val:.4f}**")
    else:
        alpha_val = st.number_input("Alpha ($\alpha$)", value=1.0)

    st.divider()
    max_fac = st.slider("Max Stress Factor", 1.1, 3.0, 1.5, help="Generates curve up to this multiple of Yield Strength.")

if 'ro_results' not in st.session_state: st.session_state['ro_results'] = None

# Calculate automatically
inputs = {'E': E, 'S_y': S_y, 'n': n, 'alpha': alpha_val, 'max_stress_factor': max_fac}
res = calculate_ramberg_osgood(inputs)

# ---------------------------
# Results Display
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Computed Curve")
    fig, ax = plt.subplots()
    ax.plot(res['total_strain'], res['stress'], label="Ramberg-Osgood", color='blue', lw=2)
    ax.scatter([res['total_strain'][0]], [res['stress'][0]], color='red', zorder=5, label="Yield Point")
    
    # Plot elastic line for reference
    max_e_strain = max(res['total_strain'])
    ax.plot([0, max_e_strain], [0, max_e_strain*E], ls='--', color='gray', alpha=0.5, label="Elastic Slope")
    
    ax.set_xlabel("Total Strain")
    ax.set_ylabel(props['stress_label'])
    ax.set_ylim(0, max(res['stress'])*1.1)
    ax.set_xlim(0, max(res['total_strain'])*1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("Abaqus Input Data")
    st.info("Copy this directly into the `*PLASTIC` keyword.")
    
    # Zip data
    data_pairs = list(zip(res['stress'], res['plastic_strain']))
    text_data = format_table_for_copy(data_pairs)
    
    # --- UPDATED SECTION START ---
    # Using st.code instead of st.text_area to get the built-in copy button
    st.text("Yield Stress    Plastic Strain")
    st.code(text_data, language="text")
    # --- UPDATED SECTION END ---
    
    # Download
    inp_content = snippet_header("RO_STEEL") + snippet_elastic(E, nu) + "*PLASTIC\n" + text_data
    st.download_button("Download .inp file", inp_content, "ramberg_osgood.inp")

# Theory Section
with st.expander("📝 Theory & Abaqus Notes"):
    st.markdown("""
    ### The Ramberg-Osgood Model
    The model describes the total strain as the sum of elastic and plastic components:
    
    $$ \\varepsilon_{total} = \\varepsilon_{el} + \\varepsilon_{pl} $$
    $$ \\varepsilon_{total} = \\frac{\sigma}{E} + \\alpha \\frac{\sigma_y}{E} \\left( \\frac{\sigma}{\sigma_y} \\right)^n $$
    
    ### For Abaqus
    Abaqus standard plasticity (`*PLASTIC`) requires **(Yield Stress, Plastic Strain)** pairs.
    This tool extracts the plastic term:
    
    $$ \\varepsilon_{pl} = \\alpha \\frac{\sigma_y}{E} \\left( \\frac{\sigma}{\sigma_y} \\right)^n $$
    
    **Note:** We enforce the first point as $(\sigma_y, 0.0)$ to satisfy Abaqus requirements for the onset of plasticity.
    """)
