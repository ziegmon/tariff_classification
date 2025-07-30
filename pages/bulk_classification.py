import streamlit as st
import pandas as pd
import time
from helper_functions import (
    configure_genai, process_bulk_data, add_bg_from_local, header,
    load_all_pdf_data, regenerate_single_product, check_password,
    save_rejected_code
)

# Page config
st.set_page_config(
    page_title="Footwear Bulk Classification",
    page_icon="👟",
    layout="wide"
)

add_bg_from_local("images/on_box.png")

# Authentication
if not st.session_state.get("authenticated", False):
    if not check_password():
        st.stop()

header("Footwear Bulk HS Code Classification")

# Initialize session state
if "bulk_results_df" not in st.session_state:
    st.session_state.bulk_results_df = None
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}
if "model" not in st.session_state:
    st.session_state.model = None
if "product_selections" not in st.session_state:
    st.session_state.product_selections = {}
if "regenerate_queue" not in st.session_state:
    st.session_state.regenerate_queue = set()
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = {}

# Configure Gemini API
try:
    if st.session_state.model is None:
        st.session_state.model = configure_genai(st.secrets["API_KEY"])
        st.success("✅ AI Model configured successfully!")
except Exception as e:
    st.error(f"❌ Error configuring AI model: {str(e)}")
    st.stop()

# File upload section
st.subheader("📁 Upload Your Footwear Data")
uploaded_file = st.file_uploader(
    "Upload CSV file with footwear products",
    type=['csv'],
    help="CSV should contain columns: country, product_type, material, construction, gender"
)

if uploaded_file is not None:
    try:
        # Read uploaded file
        df_input = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded {len(df_input)} products successfully!")
        
        # Show preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df_input.head())
        
        # Column mapping
        st.subheader("🔗 Column Mapping")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            country_col = st.selectbox("Country Column", df_input.columns, 
                                     index=list(df_input.columns).index("tariff_country_description") 
                                     if "tariff_country_description" in df_input.columns else 0)
            name_col = st.selectbox("Product Name Column", df_input.columns,
                                  index=list(df_input.columns).index("customs_description") 
                                  if "customs_description" in df_input.columns else 0)
        
        with col2:
            material_col = st.selectbox("Material Column", df_input.columns,
                                      index=list(df_input.columns).index("material_composition") 
                                      if "material_composition" in df_input.columns else 0)
            construction_col = st.selectbox("Construction Column", df_input.columns,
                                          index=list(df_input.columns).index("outsole_material") 
                                          if "outsole_material" in df_input.columns else 0)
        
        with col3:
            gender_col = st.selectbox("Gender Column", df_input.columns,
                                    index=list(df_input.columns).index("gender_name") 
                                    if "gender_name" in df_input.columns else 0)
            name_col2 = st.selectbox("Secondary Product Column", ["None"] + list(df_input.columns),
                                   index=list(["None"] + list(df_input.columns)).index("product_type") 
                                   if "product_type" in df_input.columns else 0)
        
        name_col2 = None if name_col2 == "None" else name_col2
        
        # Processing section
        st.subheader("⚡ Process Products")
        
        if st.button("🚀 Start Bulk Classification", type="primary"):
            # Add original index for tracking
            df_input['original_index'] = df_input.index
            
            # Load PDF cache if not already loaded
            if not st.session_state.pdf_cache:
                with st.spinner("📚 Loading tariff documentation..."):
                    st.session_state.pdf_cache = load_all_pdf_data()
                st.success("✅ Documentation loaded!")
            
            # Process data
            with st.spinner("🔄 Processing footwear classifications..."):
                results_df = process_bulk_data(
                    df_input=df_input,
                    model=st.session_state.model,
                    pdf_data_cache=st.session_state.pdf_cache,
                    country_col=country_col,
                    name_col=name_col,
                    name_col2=name_col2,
                    material_col=material_col,
                    construction_col=construction_col,
                    gender_col=gender_col
                )
            
            st.session_state.bulk_results_df = results_df
            st.success("✅ Bulk classification complete!")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Results display section
if st.session_state.bulk_results_df is not None:
    st.subheader("📊 Classification Results")
    
    # Summary statistics
    df = st.session_state.bulk_results_df
    total_products = len(df)
    successful = len(df[df['hs_code_1'] != 'N/A'])
    errors = len(df[df['hs_code_1'] == 'ERROR'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", total_products)
    with col2:
        st.metric("Successfully Processed", successful)
    with col3:
        st.metric("Errors", errors)
    with col4:
        st.metric("Success Rate", f"{(successful/total_products)*100:.1f}%")
    
    # Filter options
    st.subheader("🔍 Filter Results")
    col1, col2 = st.columns(2)
    
    with col1:
        filter_country = st.selectbox("Filter by Country", 
                                    ["All"] + list(df['input_country'].unique()))
    with col2:
        filter_status = st.selectbox("Filter by Status", 
                                   ["All", "Success", "Error", "Skipped"])
    
    # Apply filters
    filtered_df = df.copy()
    if filter_country != "All":
        filtered_df = filtered_df[filtered_df['input_country'] == filter_country]
    
    if filter_status == "Success":
        filtered_df = filtered_df[filtered_df['hs_code_1'].notna() & 
                                (filtered_df['hs_code_1'] != 'N/A') & 
                                (filtered_df['hs_code_1'] != 'ERROR')]
    elif filter_status == "Error":
        filtered_df = filtered_df[filtered_df['hs_code_1'] == 'ERROR']
    elif filter_status == "Skipped":
        filtered_df = filtered_df[filtered_df['hs_code_1'] == 'N/A']
    
    # Display results
    st.write(f"Showing {len(filtered_df)} of {len(df)} products")
    
    # Interactive results table
    for idx, row in filtered_df.iterrows():
        original_index = row['original_index']
        
        with st.expander(f"👟 Product {original_index + 1}: {row['input_gender']}'s {row['input_product']} - {row['input_country']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Product Description:** {row['product_description']}")
                st.write(f"**Material:** {row['input_material']}")
                st.write(f"**Construction:** {row['input_construction']}")
            
            with col2:
                # Regenerate button
                if st.button(f"🔄 Regenerate", key=f"regen_{original_index}"):
                    st.session_state.regenerate_queue.add(original_index)
                    regenerate_single_product(original_index)
            
            # Display classification options
            if row['hs_code_1'] not in ['N/A', 'ERROR']:
                st.write("### Classification Options:")
                
                for i in range(1, 4):
                    hs_code = row.get(f'hs_code_{i}', '')
                    certainty = row.get(f'certainty_{i}', 0)
                    reasoning = row.get(f'reasoning_{i}', '')
                    
                    if hs_code and hs_code.strip():
                        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                        
                        with col_a:
                            st.write(f"**Option {i}:** `{hs_code}` ({certainty}% certainty)")
                        
                        with col_b:
                            # Show reasoning button (FIXED - no nested expanders)
                            reasoning_key = f"reasoning_{original_index}_{i}"
                            if st.button(f"📝 Reasoning", key=f"show_reasoning_{original_index}_{i}"):
                                if reasoning_key not in st.session_state.show_reasoning:
                                    st.session_state.show_reasoning[reasoning_key] = True
                                else:
                                    st.session_state.show_reasoning[reasoning_key] = not st.session_state.show_reasoning[reasoning_key]
                        
                        with col_c:
                            # Selection button
                            if st.button(f"✅ Select", key=f"select_{original_index}_{i}"):
                                st.session_state.product_selections[original_index] = {
                                    'status': 'selected',
                                    'data': {
                                        'hs_code': hs_code,
                                        'certainty': certainty,
                                        'reasoning': reasoning,
                                        'option': i
                                    }
                                }
                                st.success(f"Selected Option {i} for Product {original_index + 1}")
                        
                        with col_d:
                            # Reject button
                            if st.button(f"❌ Reject", key=f"reject_{original_index}_{i}"):
                                save_rejected_code(row['product_description'], 
                                                 row['input_country'].lower(), 
                                                 hs_code)
                                st.warning(f"Rejected Option {i} for Product {original_index + 1}")
                        
                        # Show reasoning below if button was clicked (FIXED - outside expander)
                        reasoning_key = f"reasoning_{original_index}_{i}"
                        if st.session_state.show_reasoning.get(reasoning_key, False):
                            st.markdown(f"**Reasoning for Option {i}:**")
                            st.text_area(
                                "Detailed reasoning:",
                                value=reasoning,
                                height=150,
                                key=f"reasoning_text_{original_index}_{i}",
                                disabled=True
                            )
            else:
                st.error(f"Classification failed: {row['reasoning_1']}")
    
    # Export section
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export all results
        csv_all = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download All Results (CSV)",
            data=csv_all,
            file_name=f"footwear_classification_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export selected only
        if st.session_state.product_selections:
            selected_results = []
            for original_index, selection in st.session_state.product_selections.items():
                if selection['status'] == 'selected':
                    product_row = df[df['original_index'] == original_index].iloc[0]
                    selected_results.append({
                        'Original_Index': original_index + 1,
                        'Country': product_row['input_country'],
                        'Product': product_row['input_product'],
                        'Material': product_row['input_material'],
                        'Construction': product_row['input_construction'],
                        'Gender': product_row['input_gender'],
                        'Selected_HS_Code': selection['data']['hs_code'],
                        'Certainty': selection['data']['certainty'],
                        'Option_Number': selection['data']['option'],
                        'Reasoning': selection['data']['reasoning']
                    })
            
            if selected_results:
                selected_df = pd.DataFrame(selected_results)
                csv_selected = selected_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="✅ Download Selected Only (CSV)",
                    data=csv_selected,
                    file_name=f"footwear_selected_classifications_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No selections made yet")
        else:
            st.info("No selections made yet")
    
    with col3:
        # Export summary
        summary_data = {
            'Metric': ['Total Products', 'Successfully Processed', 'Errors', 'Skipped', 'Success Rate %'],
            'Value': [total_products, successful, errors, total_products - successful - errors, 
                     f"{(successful/total_products)*100:.1f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📈 Download Summary (CSV)",
            data=csv_summary,
            file_name=f"footwear_classification_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Instructions
with st.expander("📖 How to Use Bulk Classification"):
    st.markdown("""
    ### Steps to classify your footwear products:
    
    1. **📁 Upload CSV File** - Your file should contain columns for:
       - Country (e.g., "Australia", "Canada")
       - Product type (e.g., "Running Shoe", "Boot")
       - Material (e.g., "Upper: Leather / Sole: Rubber")
       - Construction (e.g., "Rubber", "Leather")
       - Gender (e.g., "Men", "Women", "Kids")
    
    2. **🔗 Map Columns** - Select which columns correspond to each data type
    
    3. **🚀 Start Processing** - Click the button to begin bulk classification
    
    4. **📊 Review Results** - Examine each product's classification options
    
    5. **✅ Select Classifications** - Choose the best option for each product
    
    6. **💾 Export Results** - Download your classifications as CSV
    
    ### Tips:
    - The system uses AI to analyze footwear characteristics and tariff documents
    - Each product gets 3 classification options with confidence scores
    - You can reject incorrect codes to improve future classifications
    - Use the regenerate button if classifications seem wrong """)