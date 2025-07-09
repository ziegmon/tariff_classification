#__Imports__#
import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
from helper_functions_2 import (
    header,
    st_info,
    configure_genai,
    generate_hs_codes,
    add_bg_from_local,
    load_all_pdf_data,
    extract_hs_codes,
    save_rejected_code,
    load_rejected_codes, 
    regenerate_single_product,
    find_relevant_chapters,
    load_text_files_for_country,
    extract_simplified_hs_codes
)
import time
from streamlit_option_menu import option_menu


#___Page Setupt___#

page_icon = st.secrets["logo"]

st.set_page_config(
    page_title="HS Code Results",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)


add_bg_from_local(st.secrets["box"])

# custom CSS for styling expanders and info boxes
custom_css = """
<style>
.stExpander div[data-testid="stVerticalBlock"] {
    background-color: rgba(233, 220, 193, 0.3);
    border-radius: 5px;
    padding: 10px;
}

.stInfo {
    background-color: rgba(233, 220, 193, 0.3);
    border-radius: 5px;
    padding: 10px;
}
</style>
"""

st.write(custom_css, unsafe_allow_html=True)


max_rows = 14


if "menu_option" not in st.session_state:
    st.session_state["menu_option"] = 0

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

if "navigate" not in st.session_state:
    st.session_state["navigate"] = False

#___Navigation Menu - horizontal layout___#
selected_page = option_menu(
    None,
    ["Home", "Single Product Classification", "Bulk Classification"],
    icons=["key", "pen", "upload"],
    orientation="horizontal",
    key="menu_4",
    default_index=["Home", "Single Product Classification", "Bulk Classification"].index(st.session_state["selected_page"])
)


## logic to handle page navigation based on menu selection
if selected_page != st.session_state["selected_page"]:
    st.session_state["selected_page"] = selected_page
    st.session_state["navigate"] = True
else:
    st.session_state["navigate"] = False


#___Page Logic with Navigation Control__#
if st.session_state["navigate"]:
    if st.session_state["selected_page"] == "Single Product Classification":
        print(st.session_state["selected_page"])
        switch_page("single product classification")
    elif st.session_state["selected_page"] == "Bulk Classification":
        print(st.session_state["selected_page"])
        switch_page("Bulk Classification")
    st.session_state["navigate"] = False # Reset the flag after navigation attempt
else:
    # optionally display content based on the current selected_page without navigation
    if st.session_state["selected_page"] == "Home":
        pass # already handled in the navigate block or initial display
    elif st.session_state["selected_page"] == "Single Product Classification":
        pass # navigation will happen when "navigate" is True
    elif st.session_state["selected_page"] == "Bulk Classification":
        pass # navigation will happen when "navigate" is True


#___Reset Classification - starts a new one___#
col1, col2, col3 = st.columns([6, 3, 5])
with col3:
    if st.button("New Bulk Classification", type="secondary"):
        for key in ['model','product_selections', 'regenerate_queue', 'bulk_results_df',
                    'pdf_cache', 'input_df_for_mapping']:
            if key in st.session_state:
                del st.session_state[key]
        switch_page("bulk classification")


#___Documentation Path___#
PDF_DIRECTORY = "./chapter_data"

#___Session State Initialization___#
if "model" not in st.session_state:
    st.session_state.model = None   # initialized as None first
if "product_selections" not in st.session_state:
    st.session_state.product_selections = {}
if "regenerate_queue" not in st.session_state:
    st.session_state.regenerate_queue = set()
if "bulk_results_df" not in st.session_state:
    st.session_state.bulk_results_df = None
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}   # cache for PDF content
if "uploaded_file_state" not in st.session_state:
    st.session_state.uploaded_file_state = None   # tracks the uploaded file name
if "input_df_for_mapping" not in st.session_state:
    st.session_state.input_df_for_mapping = None   # holds the loaded df


#___Configuration & Setup___#
if st.session_state.model is None:
    try:
        # loading API key from secrets.toml
        api_key = st.secrets["API_KEY"]
        st.session_state.model = configure_genai(api_key) 
    except KeyError:
        st.error("API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to configure Generative AI model: {e}")
        st.stop()

#___Page Content___#
st.title("Bulk HS Code Classification Tool")

if not st.session_state.pdf_cache: # only loads if the cache is empty
    with st.spinner("Loading all country-specific PDF documentation..."):
        st.session_state.pdf_cache = load_all_pdf_data(PDF_DIRECTORY)
        if not st.session_state.pdf_cache:
            st.error("Failed to load any PDF documentation. Please ensure files are in the correct directory and named properly.")
            st.stop() # stops if no documentation is loaded


#___File Upload___#
uploaded_file = st.file_uploader(
    "Upload a CSV file containing product details",
    type=["csv"],
    key="bulk_file_uploader",
    help="CSV should contain columns for Country, Product Name/Type, Material, Construction and Gender."
)

# checking if a new file has been uploaded - resets session states
new_upload = False
if uploaded_file is not None:
    # is the name different from the stored state?
    if st.session_state.uploaded_file_state != uploaded_file.name:
        st.session_state.uploaded_file_state = uploaded_file.name
        # clears previous results when a new file is uploaded
        st.session_state.bulk_results_df = None
        st.session_state.product_selections = {}
        st.session_state.regenerate_queue = set()
        st.session_state.input_df_for_mapping = None   # also clears stored df
        new_upload = True

if uploaded_file is not None:
    try:
        # loads data only if it's a new upload or no df is stored yet
        if new_upload or st.session_state.input_df_for_mapping is None:
            df_input = pd.read_csv(uploaded_file).drop(columns=["level_0", "Unnamed: 0"], errors="ignore")
            df_input = df_input.reset_index(drop=True)  # Ensure clean index

            # limiting the number of rows for testing - *remove or increase limit*
            if len(df_input) > max_rows:
                df_input = df_input[df_input["customs_description"] != "Woven Vest"]
                df_input = df_input.iloc[:20]

            if "idx" not in df_input.columns:
                df_input = df_input.reset_index().rename(columns={"index": "idx"})
            else:
                df_input = df_input.reset_index(drop=True) 

            st.session_state.input_df_for_mapping = df_input    # stored for mapping
            st.session_state.bulk_results_df = None    # ensures results are reset

        # display mapping and generation button only if df exists in state
        if st.session_state.input_df_for_mapping is not None:
            df_input_for_mapping = st.session_state.input_df_for_mapping
            st.write("Uploaded Data Sample:")
            st.dataframe(df_input_for_mapping.head(min(max_rows, len(df_input_for_mapping))))

            country_col = "tariff_country_description"
            name_col = "customs_description"
            name_col2 = "product_type"
            material_col = "composition"
            construction_col = "material_type"
            gender_col = "division"

            disable_generate = (st.session_state.bulk_results_df is not None)

            if st.button("Generate HS Codes for Uploaded File", key="generate_bulk", type="primary"):
            #if st.button("Generate HS Codes for Uploaded File", key="generate_bulk", type="primary", disabled=disable_generate):
                if not country_col or not name_col or not material_col:
                    st.error("Please map at least Country, Product Name/Type, and Material columns.")
                else:
                    st.session_state.product_selections = {}
                    st.session_state.regenerate_queue = set()

                    try:
                        if st.session_state.model is None:
                            st.error("Model not initialized. Cannot generate codes.")
                            st.stop()
                        model = st.session_state.model

                        all_results_list = []
                        total_rows = len(df_input_for_mapping)
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        start_time = time.time()

                        for df_idx, row in df_input_for_mapping.iterrows():
                            idx = row["idx"]

                            current_progress = (df_idx + 1) / total_rows
                            progress_bar.progress(current_progress)
                            elapsed_time = time.time() - start_time
                            est_remaining = (elapsed_time / (df_idx + 1)) * (total_rows - (df_idx + 1)) if df_idx > 0 else 0
                            progress_text.text(f"Processing row {df_idx + 1}/{total_rows}... Est. time remaining: {int(est_remaining)}s")

                            country = str(row[country_col]).strip().lower() if pd.notna(row[country_col]) else "unknown"

                            #processed_pdfs_for_current_country = st.session_state.pdf_cache[country]

                            product_type = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
                            if name_col2 and name_col2 in row and pd.notna(row[name_col2]):
                                product_type += " " + str(row[name_col2]).strip()
                            material = str(row[material_col]).strip() if pd.notna(row[material_col]) else ""
                            construction = str(row[construction_col]).strip() if construction_col and construction_col in row and pd.notna(row[construction_col]) else ""
                            gender = str(row[gender_col]).strip() if gender_col and gender_col in row and pd.notna(row[gender_col]) else ""
                            
                            desc_parts = [f"Product for {country.upper()}:"]
                            if gender: 
                                desc_parts.append(f"{gender}'s")
                            if product_type: 
                                desc_parts.append(product_type)
                            if material: 
                                desc_parts.append(f"Material: {material}.")
                            if construction: 
                                desc_parts.append(f"Construction: {construction}.")
                            product_description = " ".join(desc_parts).strip()
                            print("------____------")
                            print(idx)
                            print("____-----____")

                            base_result_row = {
                                "idx": idx, 
                                "input_country": country.upper(),
                                "input_product": product_type,
                                "input_material": material,
                                "input_construction": construction,
                                "input_gender": gender,
                                "product_description": product_description,
                                "hs_code_1": "N/A", "certainty_1": 0, "reasoning_1": "Skipped",
                                "hs_code_2": "", "certainty_2": 0, "reasoning_2": "",
                                "hs_code_3": "", "certainty_3": 0, "reasoning_3": ""
                            }


                            if not product_type or not material or country == "unknown":
                                st.warning(f"Skipping row {df_idx + 1} (Original Index: {idx}): Insufficient data (missing Product Name/Type, Material, or Country).")
                                base_result_row["reasoning_1"] = "Skipped due to missing essential data"
                                all_results_list.append(base_result_row)
                                continue  

                            if country not in st.session_state.pdf_cache:
                                st.warning(f"Skipping row {df_idx + 1} (Original Index: {idx}): No cached PDF data available for country '{country}'.")
                                base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
                                all_results_list.append(base_result_row)
                                continue

                            processed_pdfs_for_current_country = st.session_state.pdf_cache[country] 
                            relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs_for_current_country) #
                            legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
                            classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
                            gri = processed_pdfs_for_current_country.get("gri", "")

                            if not st.session_state.pdf_cache.get(country):
                                st.warning(f"Skipping row {df_idx + 1} (Original Index: {idx}): No PDF data loaded for country '{country}'.")
                                base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
                                all_results_list.append(base_result_row)
                                continue

                            processed_pdfs = st.session_state.pdf_cache[country]

                            country_texts = load_text_files_for_country(PDF_DIRECTORY, country)
                            guidelines = "" # initializes guidelines as empty string

                            # dynamically checks for guidelines for the current country
                            guidelines_key = f"{country}_guidelines"
                            if guidelines_key in country_texts:
                                guidelines = country_texts[guidelines_key]
                            else:
                                # warning if guidelines are not found for the specific country
                                # but don't stop processing, as the model should still attempt classification
                                st.warning(f"No specific guidelines found for {country.upper()}. Continuing without country-specific guidelines.")
                                guidelines = "" # guidelines is an empty string if not found

                            country_texts = load_text_files_for_country(PDF_DIRECTORY, country)
                            guidelines_key = f"{country}_guidelines"
                            guidelines = country_texts.get(guidelines_key, "") 
                            if not guidelines:
                                st.warning(f"No specific guidelines found for {country.upper()}. Continuing without country-specific guidelines.")

                            if country.lower() != 'canada': # Focus on non-Canada countries
                                st.write(f"--- Debugging for {country.upper()} (Index: {idx}) ---")
                                st.write(f"Product description: {product_description}")
                                st.write(f"PDFs for current country: {processed_pdfs_for_current_country.keys()}")
                                st.write(f"Relevant chapters found: {[c[0] for c in relevant_chapters]}")
                                st.write(f"Legal notes length: {len(legal_notes)}")
                                st.write(f"Classification guide length: {len(classification_guide)}")
                                st.write(f"GRI length: {len(gri)}")
                                st.write(f"Guidelines length: {len(guidelines)}")
                                st.write("------------------------------------")

                            if not relevant_chapters and not legal_notes and not classification_guide and not gri:
                                st.warning(f"Row {df_idx + 1} (Original Index: {idx}): No tariff context (Chapters, Notes, Guide, GRI) found for {country}. Results might be inaccurate.")

                            try:
                                rejected_codes_snapshot = load_rejected_codes(product_description, country)

                                generated_response = generate_hs_codes(
                                    model,
                                    product_description,
                                    country,
                                    relevant_chapters,
                                    legal_notes,
                                    classification_guide,
                                    gri=gri,
                                    guidelines=guidelines,
                                    rejected_codes_snapshot=rejected_codes_snapshot
                                )

                                product_df_row = extract_hs_codes(generated_response)
                                
                                if not product_df_row.empty:
                                    extracted_data = product_df_row.iloc[0].to_dict()
                                    for i in range(1, 4):
                                        hs_col = f"hs_code_{i}"
                                        cert_col = f"certainty_{i}"
                                        reas_col = f"reasoning_{i}"
                                        if hs_col in extracted_data:
                                            base_result_row[hs_col] = extracted_data.get(hs_col, "")
                                            base_result_row[cert_col] = extracted_data.get(cert_col, 0)
                                            base_result_row[reas_col] = extracted_data.get(reas_col, "")
                                        else:  # clears if not present in new result
                                            base_result_row[hs_col] = ""
                                            base_result_row[cert_col] = 0
                                            base_result_row[reas_col] = ""

                                else:
                                    st.error(f"Failed to extract codes for row {df_idx + 1} (Original Index: {idx}). Response format might be unexpected.")
                                    base_result_row["reasoning_1"] = "Error: Failed to parse response"

                                all_results_list.append(base_result_row)

                            except Exception as gen_e:
                                st.error(f"Error generating code for row {df_idx + 1} (Original Index: {idx}): {gen_e}")
                                base_result_row["hs_code_1"] = "ERROR"
                                base_result_row["reasoning_1"] = str(gen_e)
                                all_results_list.append(base_result_row)


                        progress_bar.progress(1.0)
                        progress_text.text("Initial Processing Complete!")
                        progress_text.empty()
                        progress_bar.empty()

                        if all_results_list:
                            final_df = pd.DataFrame(all_results_list)
                            cols_order = [
                                "idx", "product_description",
                                "input_product", "input_material", "input_construction", "input_gender","input_country",
                                "hs_code_1", "certainty_1", "reasoning_1",
                                "hs_code_2", "certainty_2", "reasoning_2",
                                "hs_code_3", "certainty_3", "reasoning_3"
                            ]
                            final_df = final_df.reindex(columns=cols_order)

                            st.session_state.bulk_results_df = final_df
                            for idx in final_df["idx"]:
                                st.session_state.product_selections[idx] = {"status": "pending", "data": {}}

                            st.success("HS Code suggestions generated. Please review below.")
                            st.rerun() 

                        else:
                            st.warning("No HS codes were generated. Check input data and PDF availability.")
                            st.session_state.bulk_results_df = None


                    except Exception as e:
                        st.error(f"An error occurred during bulk processing: {str(e)}")
                        st.session_state.bulk_results_df = None


    except Exception as file_e:
        st.error(f"Error reading or processing CSV file: {file_e}")
        st.session_state.bulk_results_df = None
        st.session_state.product_selections = {}
        st.session_state.regenerate_queue = set()
        st.session_state.uploaded_file_state = None
        st.session_state.input_df_for_mapping = None





#___Regenerating a wrongly classified product___#
if st.session_state.regenerate_queue:
    items_to_process = list(st.session_state.regenerate_queue)
    st.session_state.regenerate_queue.clear()
    st.info(f"Processing regeneration for {len(items_to_process)} product(s)...")
    for index_to_regenerate in items_to_process: 
        regenerate_single_product(index_to_regenerate)
    st.info("Regeneration processing finished.")
    st.rerun()


#___Displaying the results___#
if st.session_state.bulk_results_df is not None:
    st.subheader("Review and Select HS Codes")
    st.markdown("""
    Review suggestions below. Choose one option, mark as incorrect (the code will be regenerated for that product), or review later (product and classification is skipped from final export).
    Click **"Process All Selections"** at the bottom when finished to see the final table with the classified products.
    """)

    results_df = st.session_state.bulk_results_df
    all_indices = results_df["idx"].unique()
    for index in all_indices:
        if index not in st.session_state.product_selections:
            st.session_state.product_selections[index] = {"status": "pending", "data": {}}
    current_selection_keys = list(st.session_state.product_selections.keys())
    for key in current_selection_keys:
        if key not in all_indices: 
            del st.session_state.product_selections[key]

    for idx, row_data in results_df.iterrows():
        idx = row_data["idx"]
        product_desc = row_data["product_description"]
        selection_info = st.session_state.product_selections.get(idx, {"status": "pending", "data": {}})
        current_selection_status = selection_info["status"]
        incorrect_options = selection_info["data"].get("incorrect_options", set()) 
        display_key_base = f"product_{idx}"

        hs_code_from_csv = None
        if st.session_state.input_df_for_mapping is not None:
            original_row = st.session_state.input_df_for_mapping[st.session_state.input_df_for_mapping["idx"] == idx]
            if not original_row.empty:
                hs_code_from_csv = original_row.iloc[0].get("tariff_code", "N/A")

        header(f"{product_desc} (HS Code from file uploaded: {hs_code_from_csv})")

        #___Displaying the Status___#
        if current_selection_status == "selected":
            selected_code = selection_info["data"].get("selected_hs_code", "N/A")
            st.success(f"Status: Selected ({selected_code})")
        elif current_selection_status == "review_later":
            st_info("Status: Marked for Review Later (will be skipped)")
        elif incorrect_options:
            st.warning(f"Status: Option(s) {', '.join(map(str, incorrect_options))} marked Incorrect")
        elif current_selection_status == "regenerate":
            st.session_state.regenerate_queue
        else:
            st_info("Status: Pending Review")

        #___Displaying Options___#
        cols = st.columns(3)

        def display_option(col_idx, opt_num, row, key_base):
            with cols[col_idx]:
                hs, cert, reas = row.get(f"hs_code_{opt_num}"), row.get(f"certainty_{opt_num}", 0), row.get(f"reasoning_{opt_num}", "")
                is_disabled = (current_selection_status == "selected" or current_selection_status == "review_later" or opt_num in incorrect_options)
                if pd.notna(hs) and hs not in ["N/A", "ERROR", "", None]:
                    st.markdown(f"**Option {opt_num}:** `{hs}` ({cert}%)")
                    with st.expander(f"Reasoning {opt_num}"):
                        st.markdown(f"<small>{reas}</small>", unsafe_allow_html=True)
                    if st.button("Select this Code", key=f"{key_base}_select_{opt_num}", disabled=is_disabled):
                        st.session_state.product_selections[idx] = {"status": "selected", "data": {"selected_hs_code": hs, "selected_reasoning": reas, "certainty": cert, "selected_option": opt_num, "incorrect_options": incorrect_options}}
                        st.rerun()
                else:
                    st.markdown(f"_(Option {opt_num} N/A)_")

        display_option(0, 1, row_data, display_key_base)
        display_option(1, 2, row_data, display_key_base)
        display_option(2, 3, row_data, display_key_base)

        #___Action Buttons Row___#
        incorrect_cols = st.columns(3)
        incorrect_count = 0  
        for i in range(3):
            with incorrect_cols[i]:
                opt_num = i + 1
                hs_code = row_data.get(f"hs_code_{opt_num}", None)
                is_disabled_incorrect = (current_selection_status == "selected" or current_selection_status == "review_later" or opt_num in incorrect_options)
                if hs_code and hs_code not in ["N/A", "ERROR", "", None]:
                    if st.button("Mark as Incorrect", key=f"{display_key_base}_incorrect_opt{opt_num}", disabled=is_disabled_incorrect):
                        if idx not in st.session_state.product_selections:
                            st.session_state.product_selections[idx] = {"status": "pending", "data": {}} 

                        if "data" not in st.session_state.product_selections[idx]:
                            st.session_state.product_selections[idx]["data"] = {}

                        current_incorrect = st.session_state.product_selections[idx]["data"].get("incorrect_options", set())
                        current_incorrect.add(opt_num)
                        st.session_state.product_selections[idx]["data"]["incorrect_options"] = current_incorrect

                        try:
                            country = str(row_data.get("input_country", "")).strip().lower()
                            prod_desc_rej = row_data.get("product_description", "")
                            if country and prod_desc_rej:
                                save_rejected_code(prod_desc_rej, country, hs_code)
                                st.toast(f"Option {opt_num} code rejected and saved.")
                            else:
                                st.warning("Cannot save rejected code: Missing country or description.")
                        except Exception as save_e:
                            st.warning(f"Error saving rejected code: {save_e}")
                        st.rerun() 
                    if opt_num in st.session_state.product_selections[idx]["data"].get("incorrect_options", set()):
                        incorrect_count += 1
                elif hs_code:
                    st.write(f"Option {opt_num}: N/A")
                else:
                    st.write(f"Option {opt_num}: N/A")
        if incorrect_count == 3:
            st.session_state.regenerate_queue.add(idx)
            regenerate_single_product(idx)
            st.rerun()

        action_cols = st.columns(4)
        with action_cols[1]:
            all_incorrect_disabled = (
                current_selection_status == "selected"
                or current_selection_status == "review_later"
            )
            if st.button(
                "None are Correct (Regenerate)",
                key=f"{display_key_base}_incorrect_all",
                disabled=all_incorrect_disabled,
                help="Mark all suggestions incorrect and try generating new ones.",
            ):
                print("all incorrect")
                st.session_state.product_selections[idx] = {
                    "status": "incorrect",
                    "data": {"incorrect_options": {1, 2, 3}},
                }
                codes_rejected_count = 0
                try:
                    country = str(row_data.get("input_country", "")).strip().lower()
                    prod_desc_rej = row_data.get("product_description", "")
                    if country and prod_desc_rej:
                        for i in range(1, 4):
                            hs_code_key = f"hs_code_{i}"
                            code_rej = row_data.get(hs_code_key)
                            if (code_rej and isinstance(code_rej, str) and code_rej not in ["N/A", "ERROR", "", None]):
                                save_rejected_code(prod_desc_rej, country, code_rej)
                                codes_rejected_count += 1
                        print(f"Saved {codes_rejected_count} codes as rejected.")
                    else:
                        st.warning("Cannot save rejected: Missing country/desc.")
                except Exception as save_e:
                    st.warning(f"Index {idx}: Err saving rejected - {save_e}")
                st.write(" ")
                regenerate_single_product(idx)
                st.rerun()

        with action_cols[2]:
             review_later_disabled = (current_selection_status == "selected")
             if st.button("Review Later (Skip Export)", key=f"{display_key_base}_review_later", disabled=review_later_disabled, help="Mark this product to be skipped in the final export."):
                  st.session_state.product_selections[idx] = {"status": "review_later", "data": {}}
                  st.rerun()


    #___Process All Selections Button___#
    st.markdown("---")
    st.markdown("### Finalize Selections")
    if st.button("Process All Selections", key="process_all_bulk", type="primary"):
        processed_data = []
        pending_count = 0
        error_count = 0
        review_later_count = 0 

        current_results_df = st.session_state.bulk_results_df
        for current_idx in current_results_df["idx"].unique():
             selection_info = st.session_state.product_selections.get(current_idx, {"status": "pending", "data": {}})
             original_row_series = current_results_df[current_results_df["idx"] == current_idx]
             if original_row_series.empty: 
                 continue
             original_row = original_row_series.iloc[0]

             #___Exclude "review_later" items___#
             if selection_info["status"] == "review_later":
                  review_later_count += 1
                  continue

             output_row = {"idx": current_idx, "product_description": original_row.get("product_description", ""), "input_product": original_row.get("input_product", ""), "input_material": original_row.get("input_material", ""), "input_construction": original_row.get("input_construction", ""), "input_gender": original_row.get("input_gender", ""), "input_country": original_row.get("input_country", ""), "selected_hs_code": "N/A", "selected_reasoning": "N/A", "certainty (%)": None, "status": selection_info["status"]}

             if selection_info["status"] == "selected":
                 output_row["selected_hs_code"] = selection_info["data"].get("selected_hs_code", "Error")
                 output_row["selected_reasoning"] = selection_info["data"].get("selected_reasoning", "Error")
                 output_row["certainty (%)"] = selection_info["data"].get("certainty")
                 output_row["status"] = f"Selected Option {selection_info["data"].get("selected_option", "?")}"
             elif selection_info["status"] == "incorrect":
                 output_row["selected_hs_code"] = "Marked Incorrect"
                 output_row["selected_reasoning"] = "User marked as incorrect."
                 output_row["status"] = "Marked Incorrect"
             elif selection_info["status"] == "pending":
                  if original_row.get("hs_code_1") in ["N/A", "ERROR"]:
                      output_row["selected_hs_code"] = original_row.get("hs_code_1")
                      output_row["selected_reasoning"] = original_row.get("reasoning_1")
                      output_row["status"] = "Skipped/Error"
                      error_count += 1
                  else: 
                    output_row["selected_hs_code"] = "Pending Review"
                    output_row["selected_reasoning"] = "User review needed."
                    output_row["status"] = "Pending Review"
                    pending_count += 1
             else: 
                output_row["status"] = f"Unknown Status: {selection_info["status"]}"
                pending_count += 1

             processed_data.append(output_row)

        #___Validation & Final Output___#
        if pending_count > 0:
            st.warning(f"{pending_count} products are still pending review. Please address them before finalizing.")
            st.session_state.final_selection_df = None
        else:
            final_df = pd.DataFrame(processed_data) if processed_data else pd.DataFrame() # Handle case where all are skipped
            final_cols_order = ["idx", "input_country", "input_product", "input_material", "input_construction", "input_gender", "product_description", "selected_hs_code", "certainty (%)", "status", "selected_reasoning"]
            final_df = final_df.reindex(columns=[col for col in final_cols_order if col in final_df.columns])
            st.session_state.final_selection_df = final_df

            success_msg = "Selections processed successfully!"
            if review_later_count > 0: 
                success_msg += f" ({review_later_count} items marked 'Review Later' were excluded)."
            if error_count > 0:
                success_msg += f" ({error_count} items had original errors/were skipped)."
            st.success(success_msg)
            st.rerun()


#___Display Final Selected Data and Download___#
if "final_selection_df" in st.session_state and st.session_state.final_selection_df is not None:
    st.subheader("Final Selected HS Codes (Excluding 'Review Later')")
    final_df_display = st.session_state.final_selection_df

    if not final_df_display.empty:
        st.dataframe(final_df_display, use_container_width=True, hide_index=True)
        try:
            csv_data = final_df_display.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Final Codes as CSV", data=csv_data, file_name="final_hs_codes.csv", mime="text/csv", key="download_bulk_final")
        except Exception as e:
            st.error(f"Failed to generate CSV: {e}")
    else:
        st.info("No products were finalized for export (they might have been marked 'Review Later' or encountered errors).")


if st.session_state.bulk_results_df is not None:
    st.subheader("Raw Generated HS Code Suggestions (Before Selection)")
    # Select only the relevant columns for a simplified overview
    simplified_bulk_overview_df = st.session_state.bulk_results_df[[
        "idx",
        "input_product",
        "input_country",
        "hs_code_1",
        "certainty_1",
        "hs_code_2",
        "certainty_2",
        "hs_code_3",
        "certainty_3"
    ]].copy() #.copy() to avoid SettingWithCopyWarning

    simplified_bulk_overview_df.to_csv("raw_df.csv")

    st.dataframe(simplified_bulk_overview_df, use_container_width=True, hide_index=True)
    st.markdown("---")


    #___Process All Selections Button___#
    st.markdown("---")
    st.markdown("### Finalize Selections")
    if st.button("Process All Selections", key="process_all_bulk", type="primary"):
        processed_data = []
        pending_count = 0
        error_count = 0
        review_later_count = 0

        current_results_df = st.session_state.bulk_results_df
        for current_idx in current_results_df["idx"].unique():
             selection_info = st.session_state.product_selections.get(current_idx, {"status": "pending", "data": {}})
             original_row_series = current_results_df[current_results_df["idx"] == current_idx]
             if original_row_series.empty:
                 continue
             original_row = original_row_series.iloc[0]

             # retrieves the original input row from the input_df_for_mapping
             # assumes 'idx' column in bulk_results_df maps directly to original_index from input_df_for_mapping
             input_row = st.session_state.input_df_for_mapping[st.session_state.input_df_for_mapping["idx"] == current_idx].iloc[0]
             actual_hs_code = input_row.get("tariff_code", "N/A - Not in original file") # Use 'tariff_code' or your actual HS code column name from the input CSV

             #___Exclude "review_later" items___#
             if selection_info["status"] == "review_later":
                  review_later_count += 1
                  continue

             output_row = {
                 "idx": current_idx,
                 "product_description": original_row.get("product_description", ""),
                 "input_product": original_row.get("input_product", ""),
                 "input_material": original_row.get("input_material", ""),
                 "input_construction": original_row.get("input_construction", ""),
                 "input_gender": original_row.get("input_gender", ""),
                 "input_country": original_row.get("input_country", ""),
                 "suggested_hs_code_1": original_row.get("hs_code_1", "N/A"), # Option 1 HS Code
                 "certainty_1_percent": original_row.get("certainty_1", 0),  # Option 1 Certainty
                 "actual_hs_code_from_dataset": actual_hs_code, # Historical/Actual HS Code
                 "selected_hs_code": "N/A",
                 "selected_reasoning": "N/A",
                 "certainty (%)": None,
                 "status": selection_info["status"]
             }

             if selection_info["status"] == "selected":
                 output_row["selected_hs_code"] = selection_info["data"].get("selected_hs_code", "Error")
                 output_row["selected_reasoning"] = selection_info["data"].get("selected_reasoning", "Error")
                 output_row["certainty (%)"] = selection_info["data"].get("certainty")
                 output_row["status"] = f"Selected Option {selection_info["data"].get("selected_option", "?")}"
             elif selection_info["status"] == "incorrect":
                 output_row["selected_hs_code"] = "Marked Incorrect"
                 output_row["selected_reasoning"] = "User marked as incorrect."
                 output_row["status"] = "Marked Incorrect"
             elif selection_info["status"] == "pending":
                  if original_row.get("hs_code_1") in ["N/A", "ERROR"]:
                      output_row["selected_hs_code"] = original_row.get("hs_code_1")
                      output_row["selected_reasoning"] = original_row.get("reasoning_1")
                      output_row["status"] = "Skipped/Error"
                      error_count += 1
                  else:
                    output_row["selected_hs_code"] = "Pending Review"
                    output_row["selected_reasoning"] = "User review needed."
                    output_row["status"] = "Pending Review"
                    pending_count += 1
             else:
                output_row["status"] = f"Unknown Status: {selection_info["status"]}"
                pending_count += 1

             processed_data.append(output_row)

        #___Validation & Final Output___#
        if pending_count > 0:
            st.warning(f"{pending_count} products are still pending review. Please address them before finalizing.")
            st.session_state.final_selection_df = None
        else:
            final_df = pd.DataFrame(processed_data) if processed_data else pd.DataFrame() # Handle case where all are skipped
            final_cols_order = ["idx", "input_country", "input_product", "input_material", "input_construction", "input_gender", "product_description", "suggested_hs_code_1", "certainty_1_percent", "actual_hs_code_from_dataset", "selected_hs_code", "certainty (%)", "status", "selected_reasoning"] # Updated columns
            final_df = final_df.reindex(columns=[col for col in final_cols_order if col in final_df.columns])
            st.session_state.final_selection_df = final_df

            success_msg = "Selections processed successfully!"
            if review_later_count > 0:
                success_msg += f" ({review_later_count} items marked 'Review Later' were excluded)."
            if error_count > 0:
                success_msg += f" ({error_count} items had original errors/were skipped)."
            st.success(success_msg)
            st.rerun()


#___Display Final Selected Data and Download___#
if "final_selection_df" in st.session_state and st.session_state.final_selection_df is not None:
    st.subheader("Final Selected HS Codes (Excluding 'Review Later')")
    final_df_display = st.session_state.final_selection_df

    if not final_df_display.empty:
        st.dataframe(final_df_display, use_container_width=True, hide_index=True)
        try:
            csv_data = final_df_display.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download User-Selected Codes as CSV", data=csv_data, file_name="selected_hs_codes.csv", mime="text/csv", key="download_bulk_final_user_selection")
        except Exception as e:
            st.error(f"Failed to generate CSV: {e}")

        st.subheader("Download Data for Model Evaluation")
        evaluation_df = final_df_display[[
            "idx",
            "product_description",
            "suggested_hs_code_1",
            "certainty_1_percent",
            "actual_hs_code_from_dataset"
        ]].copy()

        # filter out rows where actual_hs_code_from_dataset is "N/A - Not in original file"
        # as these cannot be used for direct evaluation
        evaluation_df = evaluation_df[evaluation_df["actual_hs_code_from_dataset"] != "N/A - Not in original file"]

        if not evaluation_df.empty:
            eval_csv_data = evaluation_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Evaluation Data (Option 1 vs. Actual) as CSV",
                data=eval_csv_data,
                file_name="model_evaluation_data.csv",
                mime="text/csv",
                key="download_bulk_evaluation_data"
            )
        else:
            st.info("No valid historical data found for evaluation.")


    else:
        st.info("No products were finalized for export (they might have been marked 'Review Later' or encountered errors).")

with st.expander("Instructions (Updated)", expanded=False):
    st.markdown("""
    1.  **Upload:** Upload a CSV file (*).
    2.  **Generate Codes:** Click "Generate HS Codes...". Wait for processing.
    3.  **Review & Select:** For each product:
        * Review the 3 options.
        * Click **"Select this Code"** for the correct one.
        * Click **"Mark as Incorrect"** if the code is not applicable to that product.
        * Click **"None are Correct (Regenerate)"** to get new suggestions for that specific product.
        * Click **"Review Later (Skip Export)"** to exclude this product from the final download without marking it incorrect. 
    4.  **Process Selections:** When *all* products are either "Selected", "Marked Incorrect", "Review Later", or were skipped/had errors (i.e., no "Pending Review"), click **"Process All Selections"**.
    5.  **Download:** A table appears with finalized items (excluding "Review Later"). Click "Download Final Codes as CSV".
    """)

