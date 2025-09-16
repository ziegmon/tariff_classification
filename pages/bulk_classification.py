import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import io
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

add_bg_from_local(st.secrets["box"])

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
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []

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
        col1, col2, col3, col4 = st.columns(4)
        
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
            
        with col4:
            size_col = st.selectbox("Size Column", df_input.columns,
                                    index=list(df_input.columns).index("size_code") 
                                    if "size_code" in df_input.columns else 0)
        
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
                    gender_col=gender_col,
                    size_col=size_col
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
                if st.button(f"🔄 Regenerate", key=f"regen_{original_index}"):
                    st.session_state.regenerate_queue.add(original_index)
                    regenerate_single_product(original_index)
            
            # Classification options with radio buttons
            if row['hs_code_1'] not in ['N/A', 'ERROR']:
                st.write("### Select the correct classification:")
                
                # Create a list of options with their reasoning
                classification_options = []
                for i in range(1, 4):
                    hs_code_key = f'hs_code_{i}'
                    certainty_key = f'certainty_{i}'
                    reasoning_key = f'reasoning_{i}'
                    
                    if pd.notna(row[hs_code_key]) and row[hs_code_key] not in ['N/A', 'ERROR']:
                        option_text = f"Option {i}: {row[hs_code_key]} ({row[certainty_key]}% certainty)"
                        reasoning_text = row[reasoning_key]
                        st.markdown(f"**{option_text}**")
                        st.markdown(f"**Reasoning:** {reasoning_text}")
                        classification_options.append(option_text)
                    
                classification_options.append("None of these")
                
                selection = st.radio(
                    "Choose an option:",
                    classification_options,
                    key=f"selection_{original_index}"
                )
                
                if selection == "None of these":
                    manual_code = st.text_input(
                        "Enter the correct HS code (optional):",
                        key=f"manual_{original_index}"
                    )
                else:
                    manual_code = None
                
                if st.button("💾 Save Selection", key=f"save_{original_index}"):
                    if selection == "None of these":
                        st.session_state.product_selections[original_index] = {
                            'status': 'none',
                            'correct_hs_code': manual_code if manual_code else None,
                            'certainty': 0  # Score is 0% for "None of these"
                        }
                    else:
                        # Extract HS code and certainty from selection
                        selected_option = selection.split(": ")[1].split(" (")[0]
                        option_number = int(selection.split(": ")[0].split(" ")[1])
                        certainty = row[f'certainty_{option_number}']
                        st.session_state.product_selections[original_index] = {
                            'status': 'selected',
                            'hs_code': selected_option,
                            'option': option_number,
                            'certainty': certainty  # Store the certainty of the selected option
                        }
                    st.success(f"Saved selection for Product {original_index + 1}")
            else:
                st.error(f"Classification failed: {row['reasoning_1']}")
    
    # Accuracy calculation section
    st.subheader("📈 Calculate and Visualize Accuracy")
    
    # Load existing accuracy history
    json_file = "accuracy_history.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                st.session_state.accuracy_history = json.load(f)
            except json.JSONDecodeError:
                st.session_state.accuracy_history = []

    if st.button("Calculate Accuracy"):
        if not st.session_state.product_selections:
            st.warning("No selections have been saved yet.")
        else:
            total_selected = len(st.session_state.product_selections)
            top1_correct = 0
            top3_correct = 0
            weighted_sum = 0
            
            for original_index, selection in st.session_state.product_selections.items():
                product_row = df[df['original_index'] == original_index].iloc[0]
                
                if selection['status'] == 'selected':
                    correct_hs_code = selection['hs_code']
                    if product_row['hs_code_1'] == correct_hs_code:
                        top1_correct += 1
                    if correct_hs_code in [product_row['hs_code_1'], product_row['hs_code_2'], product_row['hs_code_3']]:
                        top3_correct += 1
                    weighted_sum += selection['certainty']
                else:
                    weighted_sum += 0
            
            top1_accuracy = (top1_correct / total_selected) * 100 if total_selected > 0 else 0
            top3_accuracy = (top3_correct / total_selected) * 100 if total_selected > 0 else 0
            weighted_score = (weighted_sum / total_selected) if total_selected > 0 else 0
            
            new_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_selected": total_selected,
                "top1_accuracy": top1_accuracy,
                "top3_accuracy": top3_accuracy,
                "weighted_score": weighted_score
            }
            st.session_state.accuracy_history.append(new_entry)
            
            with open(json_file, 'w') as f:
                json.dump(st.session_state.accuracy_history, f, indent=4)
            
            st.write(f"**Number of products with selections:** {total_selected}")
            st.write(f"**Top-1 Accuracy:** {top1_accuracy:.2f}%")
            st.write(f"**Top-3 Accuracy:** {top3_accuracy:.2f}%")
            st.write(f"**Weighted Score (Certainty-based):** {weighted_score:.2f}%")
    
    # Diagnostic plots section
    st.subheader("🔬 Diagnostic Plots")
    group_by_column = st.selectbox(
        "Group by column",
        ["input_country", "input_product", "input_material", "input_construction", "input_gender"]
    )

    if st.button("Generate Diagnostic Plot"):
        if not st.session_state.product_selections:
            st.warning("No selections have been saved yet. Please make selections for products before generating the plot.")
        else:
            # Filter to only selected products
            selected_indices = list(st.session_state.product_selections.keys())
            selected_df = st.session_state.bulk_results_df[
                st.session_state.bulk_results_df['original_index'].isin(selected_indices)
            ].copy()

            # Add selection columns
            for idx in selected_indices:
                selection = st.session_state.product_selections[idx]
                row = selected_df[selected_df['original_index'] == idx]
                if selection['status'] == 'selected':
                    selected_df.loc[row.index, 'selected_status'] = 'selected'
                    selected_df.loc[row.index, 'selected_option'] = selection['option']
                    selected_df.loc[row.index, 'is_correct_top1'] = (selection['option'] == 1)
                    selected_df.loc[row.index, 'is_correct_top3'] = True
                    selected_df.loc[row.index, 'selected_certainty'] = selection['certainty']
                else:
                    selected_df.loc[row.index, 'selected_status'] = 'none'
                    selected_df.loc[row.index, 'selected_option'] = None
                    selected_df.loc[row.index, 'is_correct_top1'] = False
                    selected_df.loc[row.index, 'is_correct_top3'] = False
                    selected_df.loc[row.index, 'selected_certainty'] = 0
                    if 'correct_hs_code' in selection and selection['correct_hs_code']:
                        selected_df.loc[row.index, 'manual_hs_code'] = selection['correct_hs_code']

            # Group by the selected column
            group_metrics = selected_df.groupby(group_by_column).agg(
                num_products=('original_index', 'count'),
                top1_accuracy=('is_correct_top1', 'mean'),
                top3_accuracy=('is_correct_top3', 'mean'),
                weighted_score=('selected_certainty', 'mean')
            ).reset_index()

            # Convert accuracies to percentages
            group_metrics['top1_accuracy'] *= 100
            group_metrics['top3_accuracy'] *= 100

            # Sort by number of products descending
            group_metrics = group_metrics.sort_values('num_products', ascending=False)

            # Plot
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(8, 5))
            bar_width = 0.25
            index = range(len(group_metrics))

            ax.bar(index, group_metrics['top1_accuracy'], bar_width, label='Top-1 Accuracy', color='#1f77b4')
            ax.bar([i + bar_width for i in index], group_metrics['top3_accuracy'], bar_width, label='Top-3 Accuracy', color='#2ca02c')
            ax.bar([i + 2 * bar_width for i in index], group_metrics['weighted_score'], bar_width, label='Weighted Score', color='#ff7f0e')

            ax.set_xlabel(group_by_column, fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(f'Accuracy and Weighted Score by {group_by_column}', fontsize=14, pad=10)
            ax.set_xticks([i + bar_width for i in index])
            ax.set_xticklabels(group_metrics[group_by_column], rotation=45, ha='right', fontsize=10)
            ax.set_ylim(0, 110)
            ax.legend(fontsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=10)

            # Add number of products as text
            for i, row in group_metrics.iterrows():
                max_val = max(row['top1_accuracy'], row['top3_accuracy'], row['weighted_score'])
                ax.text(i + bar_width, max_val + 2, f'n={int(row["num_products"])}', ha='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Visualize accuracy history
    if st.session_state.accuracy_history:
        history_df = pd.DataFrame(st.session_state.accuracy_history)
        history_df['calculation_number'] = range(1, len(history_df) + 1)
        
        # First plot: Top-1 and Top-3 Accuracy
        st.subheader("📉 Accuracy Trend Over Time")
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_df['calculation_number'], history_df['top1_accuracy'], label='Top-1 Accuracy', marker='o', color='#1f77b4')
        ax.plot(history_df['calculation_number'], history_df['top3_accuracy'], label='Top-3 Accuracy', marker='o', color='#2ca02c')
        ax.set_xlabel('Calculation Number', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Trend Over Time', fontsize=14, pad=10)
        ax.set_xticks(history_df['calculation_number'])
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        st.image(buf, width=600)
        plt.close(fig)
        
        # Second plot: Weighted Score
        st.subheader("📊 Weighted Score Over Time")
        if 'weighted_score' in history_df.columns:
            history_df_weighted = history_df.dropna(subset=['weighted_score'])
            if not history_df_weighted.empty:
                plt.style.use('seaborn-v0_8')
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(history_df_weighted['calculation_number'], history_df_weighted['weighted_score'], 
                        label='Weighted Score', marker='o', color='#ff7f0e')
                ax.set_xlabel('Calculation Number', fontsize=12)
                ax.set_ylabel('Score (%)', fontsize=12)
                ax.set_title('Weighted Score Over Time', fontsize=14, pad=10)
                ax.set_xticks(history_df_weighted['calculation_number'])
                ax.set_ylim(0, 110)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', labelsize=10)
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                st.image(buf, width=600)
                plt.close(fig)

# Instructions
with st.expander("📖 How to Use Bulk Classification"):
    st.markdown("""
    ### Steps to classify your footwear products and track accuracy:
    
    1. **📁 Upload CSV File** - Your file should contain columns for:
       - Country (e.g., "Australia", "Canada")
       - Product type (e.g., "Running Shoe", "Boot")
       - Material (e.g., "Upper: Leather / Sole: Rubber")
       - Construction (e.g., "Rubber", "Leather")
       - Gender (e.g., "Men", "Women", "Kids")
    
    2. **🔗 Map Columns** - Select which columns correspond to each data type
    
    3. **🚀 Start Processing** - Click the button to begin bulk classification
    
    4. **📊 Review Results** - Examine each product's classification options
    
    5. **✅ Select Correct Classifications** - For each product:
       - Choose one of the three options if correct
       - Select "None of these" and optionally enter the correct HS code
       - Click "Save Selection" to record your choice
    
    6. **📈 Calculate and Visualize Accuracy** - Click "Calculate Accuracy" to:
       - See Top-1 and Top-3 accuracy
       - See the weighted score based on the certainty of the selected option
       - Save results to `accuracy_history.json`
       - Update the accuracy and weighted score trend graphs
    
    7. **🔬 Generate Diagnostic Plots** - Select a column and click "Generate Diagnostic Plot" to:
       - Visualize Top-1 accuracy, Top-3 accuracy, and weighted score by category
       - See the number of products per category as text labels
    
    ### Tips:
    - The system uses AI to analyze footwear characteristics and tariff documents
    - Each product gets 3 classification options with confidence scores
    - Use the regenerate button if classifications seem wrong
    - Selecting "None of these" without a manual code marks all options as incorrect
    - The JSON file (`accuracy_history.json`) is automatically saved and can be pushed to Git
    - Diagnostic plots help identify performance variations across different attributes
    """)