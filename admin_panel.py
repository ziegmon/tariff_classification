import streamlit as st
import os
import json
import pandas as pd
import PyPDF2
from pathlib import Path
from helper_functions import header, load_certainty_config, save_certainty_config, DEFAULT_CERTAINTY_WEIGHTS #  Import certainty functions

PDF_DIRECTORY = "chapter_data"

def _extract_text_from_pdf_for_edit(pdf_path):
    """Extracts text from a PDF for display/editing in the admin panel."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# MODIFIED: Made list_chapter_files more robust to always return a DataFrame
def list_chapter_files(directory=PDF_DIRECTORY):
    """
    Lists all chapter files in the directory with parsed info.
    Ensures a DataFrame is always returned, even if directory is empty or errors occur.
    """
    # Ensure directory exists, create if not
    if not os.path.exists(directory):
        os.makedirs(directory)
        # Return an empty DataFrame immediately if directory was just created
        return pd.DataFrame(columns=['Filename', 'Country', 'Document Type', 'Extension', 'Full Path'])

    files_data = []
    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                filename_stem = Path(file).stem
                extension = Path(file).suffix.lower()

                parts = filename_stem.split('_')
                country = parts[0] if len(parts) > 0 else "unknown"
                doc_type = "_".join(parts[1:]) if len(parts) > 1 else "misc"

                files_data.append({
                    'Filename': file,
                    'Country': country.capitalize(),
                    'Document Type': doc_type.replace('_', ' ').title(),
                    'Extension': extension,
                    'Full Path': file_path
                })
    except Exception as e:
        # Catch any error during listing or parsing, return an empty DataFrame and show error
        st.error(f"Error listing or parsing chapter files in '{directory}': {e}")
        return pd.DataFrame(columns=['Filename', 'Country', 'Document Type', 'Extension', 'Full Path'])

    return pd.DataFrame(files_data)


def save_uploaded_chapter_file(uploaded_file, country, doc_type, directory=PDF_DIRECTORY):
    """Saves an uploaded file to the chapter_data directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Sanitize inputs for filename
    sanitized_country = country.strip().lower()
    sanitized_doc_type = doc_type.strip().lower().replace(" ", "_")

    # Determine extension from the uploaded file's original name
    file_extension = Path(uploaded_file.name).suffix.lower()

    new_filename = f"{sanitized_country}_{sanitized_doc_type}{file_extension}"
    save_path = os.path.join(directory, new_filename)

    # Check for existing file with same name
    if os.path.exists(save_path):
        st.warning(f"A file named '{new_filename}' already exists. It will be overwritten.")

    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{new_filename}' saved successfully.")
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def delete_chapter_file(filename, directory=PDF_DIRECTORY):
    """Deletes a chapter file from the directory."""
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            st.success(f"File '{filename}' deleted successfully.")
            return True
        except Exception as e:
            st.error(f"Error deleting file '{filename}': {e}")
            return False
    else:
        st.warning(f"File '{filename}' not found.")
        return False

def update_text_file_content(filename, new_content, directory=PDF_DIRECTORY):
    """Updates the content of a text file."""
    file_path = os.path.join(directory, filename)
    if not file_path.lower().endswith(".txt"):
        st.error("Only text (.txt) files can be edited directly via text area.")
        return False
    if os.path.exists(file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            st.success(f"Content for '{filename}' updated successfully.")
            return True
        except Exception as e:
            st.error(f"Error updating file content: {e}")
            return False
    else:
        st.error(f"File '{filename}' not found for update.")
        return False

def show_admin_panel_page():
    header("Admin Panel: Manage Chapter Data and Certainty Calculations")

    st.markdown("""
        This panel allows you to manage the HS chapter documentation used by the classification models
        and configure how certainty scores are calculated.
        
        **Caution:** Modifications here directly affect the AI's knowledge base and classification confidence. Proceed with care.
    """)

    # Tabs for different functionalities
    admin_tab = st.tabs(["View / Edit Chapters", "Add New Chapter", "Delete Chapter", "Certainty Calculation Settings"])

    with admin_tab[0]: # View / Edit Chapters Tab
        st.subheader("Existing Chapter Documents")
        files_df = list_chapter_files() # Ensure files_df is loaded here

        if files_df.empty:
            st.info("No chapter documents found. Please add some using the 'Add New Chapter' tab.")
        else:
            st.dataframe(files_df[['Filename', 'Country', 'Document Type', 'Extension']], use_container_width=True)

            st.subheader("Edit Chapter Content")
            selected_file = st.selectbox(
                "Select a file to view/edit",
                options=[""] + files_df['Filename'].tolist(),
                key="edit_file_select"
            )

            if selected_file:
                file_info = files_df[files_df['Filename'] == selected_file].iloc[0]
                full_path = file_info['Full Path']
                extension = file_info['Extension']

                st.write(f"**Selected File:** `{selected_file}`")
                st.write(f"**Country:** {file_info['Country']}")
                st.write(f"**Document Type:** {file_info['Document Type']}")

                if extension == ".txt":
                    # Display and allow editing of text files
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    edited_content = st.text_area(
                        "Edit content",
                        value=content,
                        height=400,
                        key=f"text_editor_{selected_file}"
                    )

                    if st.button(f"Save Changes to {selected_file}", key=f"save_edit_{selected_file}"):
                        if update_text_file_content(selected_file, edited_content):
                            # Invalidate PDF cache to ensure changes are reloaded
                            if "pdf_cache" in st.session_state:
                                del st.session_state.pdf_cache
                            st.rerun() # Refresh to show updated content if needed
                elif extension == ".pdf":
                    st.info("PDF files cannot be edited directly as text. To update a PDF, please delete the old one and upload a new version using the 'Add New Chapter' tab.")
                    # Optionally, display a download button for the PDF
                    with open(full_path, "rb") as pdf_file:
                        st.download_button(
                            label=f"Download {selected_file}",
                            data=pdf_file,
                            file_name=selected_file,
                            mime="application/pdf",
                            key=f"download_pdf_{selected_file}"
                        )
                else:
                    st.warning(f"Unsupported file type for direct editing: {extension}")


    with admin_tab[1]: # Add New Chapter Tab
        st.subheader("Upload New Chapter Document")
        uploaded_file = st.file_uploader("Upload PDF or Text file", type=["pdf", "txt"], key="upload_chapter_file")

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                country_input = st.text_input(
                    "Country (e.g., 'canada', 'europe')",
                    help="Enter the country this document applies to. This will be used in the filename."
                )
            with col2:
                doc_type_input = st.text_input(
                    "Document Type (e.g., 'chapter_64', 'legal_notes', 'guidelines')",
                    help="Enter the type of document. This will be used in the filename."
                )

            st.info(f"The file will be saved as: `{country_input.strip().lower()}_{doc_type_input.strip().lower().replace(' ', '_')}{Path(uploaded_file.name).suffix.lower()}`")

            if st.button("Save New Chapter", type="primary", key="save_new_chapter_btn"):
                if country_input and doc_type_input:
                    if save_uploaded_chapter_file(uploaded_file, country_input, doc_type_input):
                        # Invalidate PDF cache to ensure new files are loaded
                        if "pdf_cache" in st.session_state:
                            del st.session_state.pdf_cache
                        st.rerun()
                else:
                    st.error("Please provide both country and document type.")

    with admin_tab[2]: # Delete Chapter Tab
        st.subheader("Delete Chapter Document")
        files_df = list_chapter_files() # Ensure files_df is loaded here

        if files_df.empty:
            st.info("No chapter documents to delete.")
        else:
            file_to_delete = st.selectbox(
                "Select a file to delete",
                options=[""] + files_df['Filename'].tolist(),
                key="delete_file_select"
            )

            if file_to_delete:
                st.warning(f"You are about to delete: `{file_to_delete}`. This action cannot be undone.")
                if st.button(f"Confirm Delete {file_to_delete}", key=f"confirm_delete_btn_{file_to_delete}"):
                    if delete_chapter_file(file_to_delete):
                        # Invalidate PDF cache after deletion
                        if "pdf_cache" in st.session_state:
                            del st.session_state.pdf_cache
                        st.rerun()

    with admin_tab[3]: # Certainty Calculation Settings Tab
        st.subheader("Certainty Calculation Settings")
        
        # Load current configuration
        current_config = load_certainty_config()
        current_mode = current_config["calculation_mode"]
        current_weights = current_config["custom_weights"]

        st.markdown("**Current Setting:**")
        if current_mode == "none":
            st.info("Certainty calculation is currently **DISABLED**.")
            if st.button("Enable Standard Certainty Calculation", key="enable_certainty_btn"):
                new_config = {"calculation_mode": "standard", "custom_weights": DEFAULT_CERTAINTY_WEIGHTS.copy()}
                save_certainty_config(new_config)
                st.session_state.certainty_config = new_config # Update session state
                st.rerun()
        else:
            st.info(f"Mode: **{current_mode.replace('_', ' ').title()}**")
            st.write("Current Weights:")
            # Safely display weights, ensuring all keys are present
            weights_to_display = {
                'historical': current_weights.get('historical', DEFAULT_CERTAINTY_WEIGHTS['historical']),
                'reasoning': current_weights.get('reasoning', DEFAULT_CERTAINTY_WEIGHTS['reasoning']),
                'format': current_weights.get('format', DEFAULT_CERTAINTY_WEIGHTS['format']),
                'citation': current_weights.get('citation', DEFAULT_CERTAINTY_WEIGHTS['citation']),
                'agreement': current_weights.get('agreement', DEFAULT_CERTAINTY_WEIGHTS['agreement'])
            }
            weights_display = ", ".join([f"{k.replace('_', ' ').title()}: {v}%" for k,v in weights_to_display.items()])
            st.code(weights_display)

        st.markdown("---")
        st.markdown("### Choose Calculation Mode")
        
        radio_options = ["Standard Option", "Custom Weights", "No Certainty Calculation"]
        
        # Determine the initial index for the radio button based on current_mode
        initial_radio_index = 0 # Default to "Standard Option"
        if current_mode == "custom":
            initial_radio_index = 1
        elif current_mode == "none":
            initial_radio_index = 2

        new_mode_selection = st.radio(
            "Select a mode for certainty calculation:",
            radio_options,
            index=initial_radio_index,
            key="certainty_mode_radio"
        )
        
        # Map radio button text back to internal config keys
        config_mode_map = {
            "Standard Option": "standard",
            "Custom Weights": "custom",
            "No Certainty Calculation": "none"
        }
        selected_config_mode = config_mode_map[new_mode_selection]

        # Display custom weight sliders if "Custom Weights" is selected
        if selected_config_mode == "custom":
            st.markdown("#### Set Custom Weights (Total must sum to 100%)")
            
            # Initialize or update temp_weights in session_state.
            # Only reset if the mode *just* switched to custom or if it was not previously set.
            # This prevents slider values from resetting every rerun if already in custom mode.
            if 'temp_weights' not in st.session_state or st.session_state.last_selected_config_mode != "custom":
                st.session_state.temp_weights = current_weights.copy()
                # Ensure all default weight keys exist in temp_weights (add if missing)
                for key, default_val in DEFAULT_CERTAINTY_WEIGHTS.items():
                    if key not in st.session_state.temp_weights:
                        st.session_state.temp_weights[key] = default_val
            
            # Store the current selection to detect changes in the next run
            st.session_state.last_selected_config_mode = selected_config_mode


            col_hist, col_reason, col_format = st.columns(3)
            with col_hist:
                st.session_state.temp_weights['historical'] = st.slider(
                    "Historical Data Similarity (0-100%)",
                    0, 100, st.session_state.temp_weights.get('historical', DEFAULT_CERTAINTY_WEIGHTS['historical']),
                    key="slider_historical"
                )
            with col_reason:
                st.session_state.temp_weights['reasoning'] = st.slider(
                    "Reasoning Length (0-100%)",
                    0, 100, st.session_state.temp_weights.get('reasoning', DEFAULT_CERTAINTY_WEIGHTS['reasoning']),
                    key="slider_reasoning"
                )
            with col_format:
                st.session_state.temp_weights['format'] = st.slider(
                    "HS Code Format (0-100%)",
                    0, 100, st.session_state.temp_weights.get('format', DEFAULT_CERTAINTY_WEIGHTS['format']),
                    key="slider_format"
                )
            
            col_citation, col_agreement, col_empty = st.columns(3)
            with col_citation:
                st.session_state.temp_weights['citation'] = st.slider(
                    "Legal Basis Citation Quality (0-100%)",
                    0, 100, st.session_state.temp_weights.get('citation', DEFAULT_CERTAINTY_WEIGHTS['citation']),
                    key="slider_citation"
                )
            with col_agreement:
                st.session_state.temp_weights['agreement'] = st.slider(
                    "Inter-Option Agreement (0-100%)",
                    0, 100, st.session_state.temp_weights.get('agreement', DEFAULT_CERTAINTY_WEIGHTS['agreement']),
                    key="slider_agreement"
                )
            
            total_current_weights = sum(st.session_state.temp_weights.values())
            st.info(f"Current total weight: **{total_current_weights}%**")

            if st.button("Normalize and Save Custom Weights", type="primary", key="save_custom_weights_btn"):
                if total_current_weights == 0:
                    st.error("Total custom weights cannot be zero. Please set at least one weight > 0.")
                else:
                    normalized_weights = {k: round((v / total_current_weights) * 100) for k, v in st.session_state.temp_weights.items()}
                    
                    # Distribute rounding errors if any to ensure it sums to exactly 100
                    current_sum = sum(normalized_weights.values())
                    if current_sum != 100:
                        diff = 100 - current_sum
                        # Add/subtract from the largest weight(s) to hit 100
                        sorted_keys = sorted(normalized_weights, key=normalized_weights.get, reverse=True)
                        for k in sorted_keys:
                            if diff == 0:
                                break
                            if diff > 0:
                                normalized_weights[k] += 1
                                diff -= 1
                            elif diff < 0 and normalized_weights[k] > 0: # Only decrease if > 0
                                normalized_weights[k] -= 1
                                diff += 1
                                
                    final_custom_weights = {k: max(0, v) for k, v in normalized_weights.items()} # Ensure no negative weights
                    
                    new_config = {"calculation_mode": "custom", "custom_weights": final_custom_weights}
                    save_certainty_config(new_config)
                    st.session_state.certainty_config = new_config # Update session state immediately
                    # Clear temp weights so they reload from the new saved config on next run if still in custom mode
                    if 'temp_weights' in st.session_state:
                        del st.session_state.temp_weights
                    st.rerun()
        
        # Save button for Standard or No Calculation mode
        if st.button("Apply Mode Change", key="apply_mode_change_btn", type="secondary"):
            if selected_config_mode == "standard":
                new_config = {"calculation_mode": "standard", "custom_weights": DEFAULT_CERTAINTY_WEIGHTS.copy()}
            elif selected_config_mode == "none":
                new_config = {"calculation_mode": "none", "custom_weights": DEFAULT_CERTAINTY_WEIGHTS.copy()} # Keep default weights for potential re-enable
            else: # If 'custom' is selected but not saved via its own button, show warning.
                 st.warning("Please use 'Normalize and Save Custom Weights' button for custom settings if you want to apply them.")
                 new_config = None # Do not save if custom weights not handled
            
            if new_config:
                save_certainty_config(new_config)
                st.session_state.certainty_config = new_config # Update session state
                if 'temp_weights' in st.session_state:
                    del st.session_state.temp_weights # Clear temp weights
                st.rerun()
