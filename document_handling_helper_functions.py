import os
import json

import pandas as pd
import pypdf as PyPDF2 
from pathlib import Path


#___Variables___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "composition"
construction_col = "material_type"
gender_col = "division"
hs_code_col = "tariff_code"



def read_file_content(file_path: str) -> str:
    """
    Reads content from a file and returns it as a string.
    Supports PDF, CSV, JSON, and plain text formats.

    Args:
        file_path: The full path to the file.

    Returns:
        The content of the file as a single string.
    """
    try:
        extension = Path(file_path).suffix.lower()

        if extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + " "
            return text.strip()

        elif extension == '.csv':
            df = pd.read_csv(file_path)
            # convert CSV to a Markdown string, which is better for LLMs
            return df.to_markdown(index=False)

        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)

        elif extension in ['.txt', '.html', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        else:
            print(f"Warning: Unsupported file extension '{extension}' for {file_path}. Attempting to read as plain text.")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    except Exception as e:
        print(f"ERROR: Could not read file {file_path}. Reason: {e}")
        return ""

def load_all_documents(directory: str) -> tuple[dict, list]:
    doc_cache = {}
    country_set = set()

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return {}, []

    for filename in os.listdir(directory):
        if filename == '.DS_Store': # <--- ADD THIS LINE
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            base_name = Path(filename).stem
            parts = base_name.lower().split('_')

            if len(parts) > 1:
                country, doc_type_key = parts[0], "_".join(parts[1:])

                if country not in doc_cache:
                    doc_cache[country] = {}

                country_set.add(country)
                content = read_file_content(file_path)
                if content:
                    doc_cache[country][doc_type_key] = content
            else:
                print(f"Warning: Skipping '{filename}'. Naming convention 'country_doctype.ext' not met.")

    return doc_cache, sorted(list(country_set))

def load_documents_for_country(directory: str, country: str) -> dict:
    processed_docs = {}
    country_lower = country.lower()

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return {}

    for filename in os.listdir(directory):
        if filename.lower().startswith(country_lower + '_'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                key = "_".join(Path(filename).stem.lower().split('_')[1:])
                content = read_file_content(file_path)
                if content:
                    processed_docs[key] = content

    return processed_docs


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        # (f"Extracted text from {pdf_path}: {text[:500]}...")  # debug statement if needed
        return text
    except Exception as e:
        # Changed st.error to print for Flask compatibility
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""


def load_all_pdf_data(pdf_directory):
    """
    Loads and caches all PDF data for all countries found in the PDF_DIRECTORY.
    The structure will be {country: {doc_type: text_content, ...}, ...}
    """
    pdf_cache = {}
    country_set = set() # To keep track of all found countries

    if not os.path.exists(pdf_directory):
        # Changed st.error to print for Flask compatibility
        print(f"Directory {pdf_directory} does not exist!")
        return {}

    for file in os.listdir(pdf_directory):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, file)
            filename = Path(file).stem.lower()

            parts = filename.split('_')
            if len(parts) > 1:
                current_country = parts[0]
                doc_type_key = "_".join(parts[1:])

                if current_country not in pdf_cache:
                    pdf_cache[current_country] = {}

                if '_chapter_' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
                elif '_tariff_schedule' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
                elif '_legal_notes' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
                elif '_classification_guide' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
                elif '_gri' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
            else:
                # Changed st.warning to print for Flask compatibility
                print(f"Warning: Skipping PDF: '{file}' due to unconventional naming. Expected 'country_doc_type.pdf'.")

    # st.session_state.country_list = sorted(list(country_set)) # This is Streamlit-specific
    return pdf_cache


#___Load .txt Files___#
# files with classification examples and other country specific guidelines
def load_text_files_for_country(text_directory, country, file_suffix=".txt"):
    processed_texts = {}

    if not os.path.exists(text_directory):
        # Changed st.error to print for Flask compatibility
        print(f"Directory {text_directory} does not exist!")
        return {}

    for file in os.listdir(text_directory):
        # checking if the file starts with the country name (case-insensitive) and has the correct suffix
        if not file.lower().startswith(country.lower()) or not file.lower().endswith(file_suffix):
            continue

        file_path = os.path.join(text_directory, file)
        filename = Path(file).stem

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            processed_texts[filename] = text
        except Exception as e:
            # Changed st.error to print for Flask compatibility
            print(f"Error reading file {file_path}: {str(e)}")

    return processed_texts


def find_relevant_chapters(product_description, country, country_specific_pdf_data):
    keywords = {
        "shirt": ["61", "62"],
        "t-shirt": ["61", "62"],
        "jacket": ["61", "62"],
        "pant": ["61", "62"],
        "shorts": ["61", "62"],
        "vest": ["61", "62"],
        "tank": ["61", "62"],
        "trouser": ["61", "62"],
        "legging": ["61", "62"],
        "tights": ["61", "62"],
        "coat": ["61", "62"],
        "bra": ["61", "62"],
        "shoe": ["64"],
        "boot": ["64"],
        "bag": ["42"],
        "backpack": ["42"],
        "leather": ["42"],
        "headgear": ["65"],
        "hat": ["65"],
        "cap": ["65"],
        "balaclava": ["65"],
        "visor": ["65"],
        "beanie": ["65"],
    }

    product_lower = product_description.lower()
    potential_chapters = set()

    for keyword, chapters in keywords.items():
        if keyword in product_lower:
            for chapter in chapters:
                potential_chapters.add(chapter)

    if any(term in product_lower for term in ["knit", "knitted", "crochet", "crocheted"]):
        if "61" in potential_chapters and "62" in potential_chapters:
            potential_chapters.remove("62")

    if "woven" in product_lower:
        if "61" in potential_chapters and "62" in potential_chapters:
            potential_chapters.remove("61")

    relevant_chapters_content = []
    for chapter_num in potential_chapters:
        chapter_key_in_cache = f"chapter_{chapter_num}"
        if chapter_key_in_cache in country_specific_pdf_data:
            relevant_chapters_content.append((chapter_num, country_specific_pdf_data[chapter_key_in_cache]))

    return relevant_chapters_content

