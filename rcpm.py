# --- Tab 1: Interaction Diagram ---
with tab1:
    # (Your existing plotting code here...)
    st.plotly_chart(fig_pm, use_container_width=True)

    # --- ADD THIS SECTION FOR CALCULATIONS BELOW THE DIAGRAM ---
    st.divider()
    st.subheader("📑 Detailed Calculation Summary")
    
    with st.expander("View Engineering Formulas & Sample Calculation (Balanced Point)"):
        # 1. Geometry & Materials Summary
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**Material Properties:**")
            st.latex(rf"f'_c = {fc} \text{{ MPa}}, \quad f_y = {fy} \text{{ MPa}}")
            st.latex(rf"E_s = 200,000 \text{{ MPa}}, \quad \epsilon_{{cu}} = 0.003")
        with col_c2:
            st.write("**Section Constants:**")
            st.latex(rf"\beta_1 = {round(beta1, 3)}")
            st.latex(rf"d = {d} \text{{ mm}}, \quad d' = {d_prime} \text{{ mm}}")

        st.divider()

        # 2. Balanced Point Logic
        st.write("**Balanced Point Calculation ($c_{bal}$):**")
        c_bal = (ecu / (ecu + (fy/200000))) * d
        a_bal = beta1 * c_bal
        st.latex(rf"c_{{bal}} = \frac{{\epsilon_{{cu}}}}{{\epsilon_{{cu}} + \epsilon_y}} \cdot d = \frac{{0.003}}{{0.003 + {round(fy/200000, 4)}}} \cdot {d} = {round(c_bal, 2)} \text{{ mm}}")
        st.latex(rf"a_{{bal}} = \beta_1 \cdot c_{{bal}} = {round(a_bal, 2)} \text{{ mm}}")
        
        # 3. Nominal Capacity at Balanced Point
        Cc_bal = 0.85 * fc * a_bal * b / 1000
        st.write(f"Concrete Compression Force: $C_c = 0.85 \cdot f'_c \cdot a \cdot b = {round(Cc_bal, 2)} \text{{ kN}}$")

    # 4. Full Data Table
    st.write("**Interaction Points Data Table:**")
    # Adding a column for Eccentricity (e = M/P) for better engineering insight
    display_df = pm_df.copy()
    display_df['Eccentricity (mm)'] = (display_df['Mn'] / display_df['Pn'] * 1000).fillna(0).apply(lambda x: round(abs(x), 1))
    
    st.dataframe(
        display_df[['Pn', 'Mn', 'Eccentricity (mm)']].rename(
            columns={'Pn': 'Axial Pn (kN)', 'Mn': 'Moment Mn (kNm)'}
        ), 
        use_container_width=True
    )
