
import streamlit as st
import os
import json
import pandas as pd
import PyPDF2
from pathlib import Path
from helper_functions import header 

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

def list_chapter_files(directory=PDF_DIRECTORY):
    """Lists all chapter files in the directory with parsed info."""
    if not os.path.exists(directory):
        os.makedirs(directory) # Create if it doesn't exist
        return pd.DataFrame(columns=['Filename', 'Country', 'Document Type', 'Extension', 'Full Path'])

    files_data = []
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
    header("Admin Panel: Manage Chapter Data")

    st.markdown("""
        This panel allows you to manage the HS chapter documentation used by the classification models.
        You can view existing files, upload new ones, edit text-based files, or delete outdated documents.
    """)

    st.warning("🚨 **Caution:** Modifications here directly affect the AI's knowledge base. Proceed with care.")

    # Tabs for different functionalities
    admin_tab = st.tabs(["View / Edit Chapters", "Add New Chapter", "Delete Chapter"])

    with admin_tab[0]: # View / Edit Chapters Tab
        st.subheader("Existing Chapter Documents")
        files_df = list_chapter_files()

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
                        st.rerun()
                else:
                    st.error("Please provide both country and document type.")

    with admin_tab[2]: # Delete Chapter Tab
        st.subheader("Delete Chapter Document")
        files_df = list_chapter_files()

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
                        st.rerun() 