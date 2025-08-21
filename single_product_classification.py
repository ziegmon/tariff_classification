import pandas as pd
import streamlit as st
from helper_functions import (
    configure_genai, generate_hs_codes, extract_hs_codes, 
    find_relevant_chapters, load_all_pdf_data,
    header, load_text_files_for_country
)




def show_single_product_classification_page():
    header("Single Footwear HS Code Classification")

    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "pdf_cache" not in st.session_state:
        st.session_state.pdf_cache = {}

    # Configure Gemini API
    try:
        if st.session_state.model is None:
            st.session_state.model = configure_genai(st.secrets["API_KEY"])
            st.success("✅ AI Model configured successfully!")
    except Exception as e:
        st.error(f"❌ Error configuring AI model: {str(e)}")


    # Input form
    st.subheader("👟 Product Information")
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox(
            "Country",
            ["Canada", "USA", "Australia", "Norway", "Switzerland"],
            help="Select the target country for classification"
        )
        
        product_type = st.text_input(
            "Product Type",
            placeholder="e.g., Running Shoe, Boot, Sandal",
            help="Specify the type of footwear"
        )
        
        upper_material = st.text_input(
            "Upper Material",
            placeholder="e.g., Leather, Textile, Synthetic",
            help="Material of the shoe upper"
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Men", "Women", "Kids", "Boys", "Girls", "Unisex"],
            help="Target gender for the footwear"
        )
        
        sole_material = st.text_input(
            "Sole Material", 
            placeholder="e.g., Rubber, Leather, EVA",
            help="Material of the outer sole"
        )
        
        construction = st.text_input(
            "Construction Method",
            placeholder="e.g., Molded, Cemented, Sewn",
            help="How the shoe is constructed"
        )

    # Additional details
    st.subheader("🔧 Additional Details")
    col1, col2 = st.columns(2)

    with col1:
        use_case = st.selectbox(
            "Primary Use",
            ["Athletic/Sports", "Casual", "Dress/Formal", "Work/Safety", "Fashion", "Other"],
            help="Primary intended use of the footwear"
        )

    with col2:
        special_features = st.text_input(
            "Special Features",
            placeholder="e.g., Waterproof, Steel toe, Non-slip",
            help="Any special features or characteristics"
        )

    # Classification button
    if st.button("🔍 Classify Footwear", type="primary"):
        if not all([country, product_type, upper_material, sole_material, gender]):
            st.error("❌ Please fill in all required fields!")
        else:
            # Build product description
            desc_parts = [f"Product for {country.upper()}:"]
            if gender:
                desc_parts.append(f"{gender}'s")
            if product_type:
                desc_parts.append(product_type)
            if upper_material:
                desc_parts.append(f"Upper Material: {upper_material}.")
            if sole_material:
                desc_parts.append(f"Sole Material: {sole_material}.")
            if construction:
                desc_parts.append(f"Construction: {construction}.")
            if use_case and use_case != "Other":
                desc_parts.append(f"Use: {use_case}.")
            if special_features:
                desc_parts.append(f"Features: {special_features}.")
            
            product_description = " ".join(desc_parts).strip()
            
            # Load PDF cache if not already loaded
            if not st.session_state.pdf_cache:
                with st.spinner("📚 Loading tariff documentation..."):
                    st.session_state.pdf_cache = load_all_pdf_data()
                st.success("✅ Documentation loaded!")
            
            country_lower = country.lower()
            
            if country_lower not in st.session_state.pdf_cache:
                st.error(f"❌ No tariff documentation available for {country}")
            else:
                with st.spinner("🤖 Generating HS code classifications..."):
                    # Get country-specific data
                    processed_pdfs = st.session_state.pdf_cache[country_lower]
                    relevant_chapters = find_relevant_chapters(product_description, country_lower, processed_pdfs)
                    legal_notes = processed_pdfs.get("legal_notes", "")
                    classification_guide = processed_pdfs.get("classification_guide", "")
                    gri = processed_pdfs.get("gri", "")
                    
                    # Load country-specific text files
                    country_texts = load_text_files_for_country("chapter_data", country_lower)
                    guidelines = country_texts.get(f"{country_lower}_guidelines", "")
                    
                    try:
                        # Generate classifications
                        generated_response = generate_hs_codes(
                            st.session_state.model,
                            product_description,
                            country_lower,
                            relevant_chapters,
                            legal_notes,
                            classification_guide,
                            gri=gri,
                            guidelines=guidelines
                        )
                        
                        # Extract and display results
                        results_df = extract_hs_codes(generated_response)
                        
                        if not results_df.empty:
                            st.success("✅ Classification complete!")
                            
                            # Display results
                            st.subheader("📊 Classification Results")
                            
                            # Show product description
                            st.write(f"**Product Description:** {results_df['product_description'].iloc[0]}")
                            
                            # Display each option
                            for i in range(1, 4):
                                hs_code = results_df.get(f'hs_code_{i}', [''])[0] if f'hs_code_{i}' in results_df.columns else ''
                                certainty = results_df.get(f'certainty_{i}', [0])[0] if f'certainty_{i}' in results_df.columns else 0
                                reasoning = results_df.get(f'reasoning_{i}', [''])[0] if f'reasoning_{i}' in results_df.columns else ''
                                
                                if hs_code and hs_code.strip():
                                    with st.expander(f"🎯 Option {i}: {hs_code} ({certainty}% certainty)", expanded=(i==1)):
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.write("**HS Code:**")
                                            st.code(hs_code, language=None)
                                            
                                            st.write("**Reasoning:**")
                                            st.write(reasoning)
                                        
                                        with col2:
                                            st.metric("Certainty", f"{certainty}%")
                                            
                                            if st.button(f"❌ Mark Incorrect", key=f"reject_single_{i}"):
                                                from helper_functions import save_rejected_code
                                                save_rejected_code(product_description, country_lower, hs_code)
                                                st.warning(f"Marked Option {i} as incorrect!")
                            
                            # Export option
                            st.subheader("💾 Export Results")
                            results_csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📄 Download Classification Results (CSV)",
                                data=results_csv,
                                file_name=f"single_footwear_classification_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error("❌ Failed to extract classification results!")
                            
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")

    # Example products section
    with st.expander("💡 Example Products"):
        st.markdown("""
        ### Try these example footwear products:
        
        **Athletic Shoe:**
        - Product Type: Running Shoe
        - Upper Material: Textile
        - Sole Material: Rubber
        - Gender: Men
        - Use: Athletic/Sports
        
        **Dress Shoe:**
        - Product Type: Oxford Shoe
        - Upper Material: Leather
        - Sole Material: Leather
        - Gender: Men
        - Use: Dress/Formal
        
        **Work Boot:**
        - Product Type: Safety Boot
        - Upper Material: Leather
        - Sole Material: Rubber
        - Gender: Unisex
        - Use: Work/Safety
        - Special Features: Steel toe, Waterproof
        
        **Children's Shoe:**
        - Product Type: Sneaker
        - Upper Material: Synthetic
        - Sole Material: Rubber
        - Gender: Kids
        - Use: Casual
        """)

    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        ### Classification Tips:
        
        **Upper Material:**
        - **Leather**: Natural animal hide
        - **Textile**: Fabric materials (canvas, mesh, etc.)
        - **Synthetic**: Man-made materials (PU, PVC, etc.)
        
        **Sole Material:**
        - **Rubber**: Natural or synthetic rubber
        - **Leather**: Natural leather sole
        - **EVA**: Ethylene-vinyl acetate foam
        - **PU**: Polyurethane
        
        **Construction Methods:**
        - **Molded**: Sole is molded directly to upper
        - **Cemented**: Sole is glued to upper
        - **Sewn**: Sole is stitched to upper
        - **Vulcanized**: Rubber sole is vulcanized to upper
        
        **Use Cases:**
        - **Athletic/Sports**: Performance footwear for sports activities
        - **Casual**: Everyday wear footwear
        - **Dress/Formal**: Business and formal occasion footwear
        - **Work/Safety**: Protective footwear for work environments
        - **Fashion**: Style-focused footwear
        
        ### Common Chapter 64 Classifications:
        - **6401**: Waterproof footwear with rubber/plastic
        - **6402**: Other footwear with rubber/plastic sole
        - **6403**: Footwear with leather sole
        - **6404**: Footwear with textile upper and rubber/plastic sole
        - **6405**: Other footwear
        """)