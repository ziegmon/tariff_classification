import streamlit as st
import os
import json
import google.generativeai as genai
import pandas as pd
import PyPDF2
from pathlib import Path
import base64
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documentation Path
PDF_DIRECTORY = "chapter_data"
CSV_PATH = "train_ftw.csv"
REJECTED_CODES_FILE = "rejected_classifications_footwear.json"
CERTAINTY_CONFIG_FILE = "certainty_config.json" # File for certainty calculation settings

# Variables for footwear data structure
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "material_composition"
construction_col = "outsole_material"
gender_col = "gender_name"
hs_code_col = "tariff_code"

# Define expected HS code lengths per country for format validation
HS_CODE_LENGTHS = {
    "switzerland": 11,
    "new zealand": 11, # 10 digits + 1 letter suffix (we'll treat as 11 for length check simplicity)
    "europe": 10,
    "canada": 10,
    "australia": 10,
    "united states": 10,
    "japan": 9,
    "brazil": 8,
    "norway": 8,
    "south korea": 10,
    # Add other countries if needed with their specific lengths
}

# Default weights for certainty calculation (sum to 100)
DEFAULT_CERTAINTY_WEIGHTS = {
    'historical': 50,
    'reasoning': 20,
    'format': 10,
    'citation': 10,
    'agreement': 10
}

# Function to load certainty configuration
def load_certainty_config():
    if not os.path.exists(CERTAINTY_CONFIG_FILE):
        # Default config if file doesn't exist
        return {
            "calculation_mode": "standard",
            "custom_weights": DEFAULT_CERTAINTY_WEIGHTS.copy()
        }
    try:
        with open(CERTAINTY_CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Ensure all necessary keys are present for robustness
            if "calculation_mode" not in config:
                config["calculation_mode"] = "standard"
            if "custom_weights" not in config or not isinstance(config["custom_weights"], dict):
                config["custom_weights"] = DEFAULT_CERTAINTY_WEIGHTS.copy()
            # Ensure all default weight keys exist in custom_weights (add if missing)
            for key, default_val in DEFAULT_CERTAINTY_WEIGHTS.items():
                if key not in config["custom_weights"]:
                    config["custom_weights"][key] = default_val
            return config
    except (json.JSONDecodeError, FileNotFoundError):
        st.error(f"Error reading {CERTAINTY_CONFIG_FILE}. Reverting to default settings.")
        return {
            "calculation_mode": "standard",
            "custom_weights": DEFAULT_CERTAINTY_WEIGHTS.copy()
        }

#  Function to save certainty configuration
def save_certainty_config(config_data):
    try:
        with open(CERTAINTY_CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=2)
        st.success("Certainty calculation settings saved successfully!")
    except Exception as e:
        st.error(f"Error saving certainty settings: {str(e)}")


# Aesthetics' Functions (No changes needed here)
def add_bg_from_local(image_file, opacity=0.3):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
    <style> .stApp {{ background-image: linear-gradient( rgba(255, 255, 255, {1 - opacity}), rgba(255, 255, 255, {1 - opacity}) ), url(data:image/{"png"};base64,{encoded_string}); background-size: cover; }} .stTextInput > div > div > input {{ background-color: #f0f0f0; }} .stSelectbox [data-baseweb="select"] {{ background-color: #f0f0f0; }} .stSelectbox [data-baseweb="select"] > div {{ color: black; }} </style>
    """,
    unsafe_allow_html=True
    )

def header(url):
    st.markdown(f'<p style="background-color:#898e94;color:#fefefe;font-size:24px;border-radius:35px;text-align:center;padding-left:20px;padding-right:20px;">{url}</p>', unsafe_allow_html=True)

def st_info(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:center;">{url}</p>', unsafe_allow_html=True)

def highlight(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:left;padding-left:20px;">{url}</p>', unsafe_allow_html=True)

# Read PDF (No changes needed here)
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

#Load All PDF Data#
def load_all_pdf_data(pdf_directory=PDF_DIRECTORY):
    """Loads and caches all PDF data for all countries"""
    pdf_cache = {}
    country_set = set()
    os.makedirs(pdf_directory, exist_ok=True)

    if not os.path.exists(pdf_directory):
        st.error(f"Directory {pdf_directory} does not exist!")
        return {}

    for file in os.listdir(pdf_directory):
        if file.lower().endswith(".pdf") or file.lower().endswith(".txt"):
            file_path = os.path.join(pdf_directory, file)
            filename = Path(file).stem.lower()

            parts = filename.split('_')
            if len(parts) > 1:
                current_country = parts[0]
                doc_type_key = "_".join(parts[1:])

                if current_country not in pdf_cache:
                    pdf_cache[current_country] = {}
                
                if file.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif file.lower().endswith(".txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except Exception as e:
                        st.error(f"Error reading text file {file_path}: {str(e)}")
                        text = ""
                else:
                    text = "" # Should not happen with the suffix check

                if text: # Only add if content was successfully extracted
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)

    st.session_state.country_list = sorted(list(country_set))
    return pdf_cache

#Load Text Files#
def load_text_files_for_country(text_directory, country, file_suffix=".txt"):
    processed_texts = {}

    if not os.path.exists(text_directory):
        st.error(f"Directory {text_directory} does not exist!")
        return {}

    for file in os.listdir(text_directory):
        if not file.lower().startswith(country.lower()) or not file.lower().endswith(file_suffix):
            continue

        file_path = os.path.join(text_directory, file)
        filename = Path(file).stem 

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            processed_texts[filename] = text
        except Exception as e:
            st.error(f"Error reading file {file_path}: {str(e)}")

    return processed_texts

#Gemini API Config#
def configure_genai(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
    return model

#Rejection System#
def save_rejected_code(product, country, code):
    entry = {
        "product_description": product,
        "country": country,
        "rejected_code": code
    }
    try:
        if not os.path.exists(REJECTED_CODES_FILE):
            with open(REJECTED_CODES_FILE, "w") as f:
                json.dump([entry], f, indent=2)
        else:
            with open(REJECTED_CODES_FILE, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {REJECTED_CODES_FILE}. Starting with an empty list.")
                    data = []
                except Exception as e:
                    print(f"An unexpected error occurred while reading {REJECTED_CODES_FILE}: {e}")
                    raise
                else:
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
        print("Code saved successfully.")
    except Exception as e:
        print(f"Error saving code: {e}")

def load_rejected_codes(product_description, country):
    if not os.path.exists(REJECTED_CODES_FILE):
        return []

    try:
        with open(REJECTED_CODES_FILE, "r") as f:
            all_rejections = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    product_specific_rejections = [
        entry for entry in all_rejections
        if entry.get("product_description") == product_description and entry.get("country") == country
    ]
    return product_specific_rejections

#Historical Data Integration#
def format_historical_data_from_csv(
    csv_file_path=CSV_PATH,
    target_country=None,
    target_full_product_description=None,
    target_gender=None,
    country_col_hist="tariff_country_description",
    name_col_hist="customs_description",
    name_col2_hist="product_type",
    material_col_hist="material_composition",
    construction_col_hist="outsole_material",
    gender_col_hist="gender_name",
    hs_code_col_hist="tariff_code",
    similarity_threshold=0.85,
    top_n=5
):
    """
    Retrieves and formats historical data for the AI prompt.
    Does NOT return the raw DataFrame for external certainty calculation,
    as that is now handled by _get_historical_similarities_df.
    """
    try:
        historical_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return f"Error: CSV file not found at {csv_file_path}"
    except pd.errors.EmptyDataError:
        return f"Error: CSV file at {csv_file_path} is empty"
    except Exception as e:
        return f"Error reading CSV file: {e}"

    if not target_country or not target_full_product_description:
        return "Error: Target country and target full product description must be specified."

    # Filter by country
    historical_df[country_col_hist] = historical_df[country_col_hist].astype(str)
    historical_df = historical_df[historical_df[country_col_hist].str.lower() == target_country.lower()]

    if historical_df.empty:
        return f"No historical footwear data found for {target_country.upper()}."

    # Filter by gender if provided
    if target_gender:
        historical_df[gender_col_hist] = historical_df[gender_col_hist].astype(str).str.lower()
        target_gender_lower = str(target_gender).lower()

        historical_df = historical_df[
            (historical_df[gender_col_hist] == target_gender_lower) |
            (historical_df[gender_col_hist] == 'unisex') |
            (target_gender_lower == 'unisex')
        ]
        if historical_df.empty:
            return f"No historical footwear data found for {target_country.upper()} with gender '{target_gender}'."

    def build_description(row):
        parts = []
        if gender_col_hist in row and pd.notna(row[gender_col_hist]):
            parts.append(str(row[gender_col_hist]))
        if name_col_hist in row and pd.notna(row[name_col_hist]):
            parts.append(str(row[name_col_hist]))
        if name_col2_hist in row and pd.notna(row[name_col2_hist]):
            parts.append(str(row[name_col2_hist]))
        if material_col_hist in row and pd.notna(row[material_col_hist]):
            parts.append(str(row[material_col_hist]))
        if construction_col_hist in row and pd.notna(row[construction_col_hist]):
            parts.append(str(row[construction_col_hist]))
        return " ".join(parts).strip()

    historical_df['full_description'] = historical_df.apply(build_description, axis=1)

    # TF-IDF similarity matching
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(historical_df['full_description'])
    target_vector = vectorizer.transform([target_full_product_description])

    cosine_similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
    historical_df['similarity'] = cosine_similarities

    relevant_data = historical_df[historical_df['similarity'] >= similarity_threshold]
    top_matches = relevant_data.nlargest(top_n, 'similarity')

    output_header = (
        f"PREVIOUS FOOTWEAR CLASSIFICATIONS (SIMILAR PRODUCTS) IN {target_country.upper()}:\n"
        f"(Target Product: '{target_full_product_description}')\n"
    )
    formatted_data = output_header
    if top_matches.empty:
        return (
            f"No similar historical footwear data found for '{target_full_product_description}' "
            f"in {target_country.upper()} (Similarity Threshold: {similarity_threshold}).\n"
        )

    for index, row in top_matches.iterrows():
        display_parts = []
        if pd.notna(row.get(gender_col_hist)):
            display_parts.append(f"Gender: {row[gender_col_hist]}")
        if pd.notna(row.get(name_col_hist)):
            display_parts.append(f"Desc1: {row[name_col_hist]}")
        if pd.notna(row.get(name_col2_hist)):
            display_parts.append(f"Desc2: {row[name_col2_hist]}")
        if pd.notna(row.get(material_col_hist)):
            display_parts.append(f"Material: {row[material_col_hist]}")
        if pd.notna(row.get(construction_col_hist)):
            display_parts.append(f"Construction: {row[construction_col_hist]}")

        full_product_details_output = ", ".join(display_parts)
        hs_code_val = str(row[hs_code_col_hist]) if hs_code_col_hist in row and pd.notna(row[hs_code_col_hist]) else "N/A"
        similarity = row['similarity']
        formatted_data += (
            f"- Similar Product (Similarity: {similarity:.2f}): "
            f"{full_product_details_output}, HS Code: {hs_code_val}\n"
        )

    return formatted_data

# Internal helper for getting historical similarities as a DataFrame
def _get_historical_similarities_df(
    csv_file_path=CSV_PATH,
    target_country=None,
    target_full_product_description=None,
    target_gender=None,
    country_col_hist="tariff_country_description",
    name_col_hist="customs_description",
    name_col2_hist="product_type",
    material_col_hist="material_composition",
    construction_col_hist="outsole_material",
    gender_col_hist="gender_name",
    hs_code_col_hist="tariff_code"
):
    """
    Internal helper to get similarity scores for historical data, returning a DataFrame.
    """
    try:
        historical_df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame() # Return empty if file not found or empty

    if not target_country or not target_full_product_description:
        return pd.DataFrame()

    historical_df[country_col_hist] = historical_df[country_col_hist].astype(str)
    historical_df = historical_df[historical_df[country_col_hist].str.lower() == target_country.lower()]

    if historical_df.empty:
        return pd.DataFrame()

    if target_gender:
        historical_df[gender_col_hist] = historical_df[gender_col_hist].astype(str).str.lower()
        target_gender_lower = str(target_gender).lower()
        historical_df = historical_df[
            (historical_df[gender_col_hist] == target_gender_lower) |
            (historical_df[gender_col_hist] == 'unisex') |
            (target_gender_lower == 'unisex')
        ]
        if historical_df.empty:
            return pd.DataFrame()

    def build_description(row):
        parts = []
        if gender_col_hist in row and pd.notna(row[gender_col_hist]): parts.append(str(row[gender_col_hist]))
        if name_col_hist in row and pd.notna(row[name_col_hist]): parts.append(str(row[name_col_hist]))
        if name_col2_hist in row and pd.notna(row[name_col2_hist]): parts.append(str(row[name_col2_hist]))
        if material_col_hist in row and pd.notna(row[material_col_hist]): parts.append(str(row[material_col_hist]))
        if construction_col_hist in row and pd.notna(row[construction_col_hist]): parts.append(str(row[construction_col_hist]))
        return " ".join(parts).strip()

    historical_df['full_description'] = historical_df.apply(build_description, axis=1)

    vectorizer = TfidfVectorizer()
    # Handle the case where tfidf_matrix is empty if historical_df is empty or full_description is all NaN/empty
    if historical_df['full_description'].empty or historical_df['full_description'].isnull().all() or not any(historical_df['full_description'].astype(bool)):
        return pd.DataFrame()

    tfidf_matrix = vectorizer.fit_transform(historical_df['full_description'])
    target_vector = vectorizer.transform([target_full_product_description])

    cosine_similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
    historical_df['similarity'] = cosine_similarities

    return historical_df[[hs_code_col_hist, 'similarity']].rename(columns={hs_code_col_hist: 'hs_code'})

# Component 1 - Historical Data Certainty
def _calculate_historical_certainty(product_description, country, proposed_hs_code, product_gender=None):
    """
    Calculates certainty (0-100%) for a proposed HS code based purely on historical data similarity.
    """
    if not proposed_hs_code or not isinstance(proposed_hs_code, str):
        return 0 # Cannot calculate certainty for invalid code

    historical_similarities_df = _get_historical_similarities_df(
        target_country=country,
        target_full_product_description=product_description,
        target_gender=product_gender
    )

    if historical_similarities_df.empty:
        return 0 # No historical data at all, cannot provide historical certainty

    # Normalize HS codes for robust matching
    def normalize_code(code):
        return re.sub(r'[^0-9a-zA-Z]', '', str(code)).strip().lower()

    normalized_proposed_code = normalize_code(proposed_hs_code)
    historical_similarities_df['normalized_hs_code'] = historical_similarities_df['hs_code'].apply(normalize_code)

    matching_historical_entries = historical_similarities_df[
        historical_similarities_df['normalized_hs_code'] == normalized_proposed_code
    ]

    if matching_historical_entries.empty:
        return 0 # Proposed code not found in historical data for similar products

    max_similarity = matching_historical_entries['similarity'].max()

    return int(max_similarity * 100) # Convert to 0-100 scale

# Component 2 - Reasoning Length Score
def _calculate_reasoning_score(reasoning_text):
    """
    Scores reasoning quality based on its length. Max 20 points.
    """
    if not reasoning_text or not isinstance(reasoning_text, str):
        return 0

    text_length = len(reasoning_text.strip())

    # Define thresholds for scoring
    MIN_REASONING_LENGTH = 100 # Below this, score is 0 or very low
    MID_REASONING_LENGTH = 300 # A decent length
    GOOD_REASONING_LENGTH = 800 # Excellent length, full points (20 points)

    if text_length < MIN_REASONING_LENGTH:
        score = 0
    elif text_length < MID_REASONING_LENGTH:
        # Scale linearly from 0 to 10 points for intermediate length
        score = 10 * (text_length - MIN_REASONING_LENGTH) / (MID_REASONING_LENGTH - MIN_REASONING_LENGTH)
    elif text_length < GOOD_REASONING_LENGTH:
        # Scale linearly from 10 to 20 points for longer reasoning
        score = 10 + 10 * (text_length - MID_REASONING_LENGTH) / (GOOD_REASONING_LENGTH - MIN_REASONING_LENGTH) # Corrected denominator for scaling
    else:
        score = 20 # Full points for very long reasoning

    return int(min(20, max(0, score))) # Ensure score is between 0 and 20

# Component 3 - HS Code Format Score
def _calculate_format_score(proposed_hs_code, country):
    """
    Scores HS code format validity. Returns 10 points if perfect, 0 otherwise.
    """
    if not proposed_hs_code or not isinstance(proposed_hs_code, str):
        return 0

    expected_length = HS_CODE_LENGTHS.get(country.lower())

    if expected_length is None:
        return 0 # No length defined for this country, cannot validate

    # Remove non-alphanumeric characters for length check, handle New Zealand's letter suffix
    cleaned_code = re.sub(r'[^0-9a-zA-Z]', '', proposed_hs_code)

    if country.lower() == "new zealand":
        # New Zealand specific: 10 digits followed by 1 letter.
        # Our HS_CODE_LENGTHS is set to 11 for NZ to match total length,
        # but actual validation needs to check numeric + letter parts.
        if len(cleaned_code) == 11 and cleaned_code[:10].isdigit() and cleaned_code[10:].isalpha() and len(cleaned_code[10:]) == 1:
            return 10
        else:
            return 0
    else:
        # For other countries, just check exact digit length and numeric content
        if len(cleaned_code) == expected_length and cleaned_code.isdigit():
            return 10
        else:
            return 0

# Component 4 - Legal Basis Citation Quality
def _calculate_citation_quality(legal_basis_text):
    """
    Scores the quality of citations in the legal basis text. Max 10 points.
    Looks for GRI rules, Chapter notes, and HS heading/subheading patterns.
    """
    if not legal_basis_text or not isinstance(legal_basis_text, str):
        return 0

    citations = set() # Use a set to count unique citations

    # Regex patterns for different types of citations
    # GRI Rules: e.g., "GRI 1", "Rule 3(b)"
    gri_pattern = re.compile(r'(GRI\s+\d(?:$[a-zA-Z0-9]+$)?)|(Rule\s+\d(?:$[a-zA-Z0-9]+$)*)', re.IGNORECASE)
    # Chapter 64 Notes: e.g., "Note 3 to Chapter 64", "Chapter 64, Note 2"
    chapter_note_pattern = re.compile(r'(?:Chapter\s+64,\s*)?Note\s+\d(?:\s*$\w+$)?\s*(?:to\s+Chapter\s+64)?', re.IGNORECASE)
    # HS Headings/Subheadings: e.g., "6403", "6403.12", "Heading 6403", "Subheading 6404.11"
    hs_code_pattern = re.compile(r'(?:(?:Heading|Subheading)\s+)?(64\d{2}(?:\.\d{2})?(?:\.\d{2})?)')

    # Find matches and add to set
    for match in gri_pattern.finditer(legal_basis_text):
        citations.add(match.group(0).strip())
    for match in chapter_note_pattern.finditer(legal_basis_text):
        citations.add(match.group(0).strip())
    for match in hs_code_pattern.finditer(legal_basis_text):
        citations.add(match.group(0).strip())

    # Score based on number of unique, relevant citations
    num_citations = len(citations)

    if num_citations >= 3:
        score = 10 # Excellent citation quality
    elif num_citations == 2:
        score = 7
    elif num_citations == 1:
        score = 3
    else:
        score = 0

    return score

# Component 5 - Inter-Option Agreement
def _calculate_inter_option_agreement(hs_codes_list):
    """
    Calculates certainty based on agreement among the three proposed HS codes. Max 10 points.
    - 10 points if all three share the first 6 digits.
    - 5 points if all three share the first 4 digits (but not 6).
    - 0 points otherwise.
    """
    if not hs_codes_list or len(hs_codes_list) < 3:
        return 0

    # Clean and extract numerical parts of HS codes
    cleaned_codes = []
    for code in hs_codes_list:
        if pd.notna(code) and isinstance(code, str):
            # For "New Zealand" specific formatting (10 digits + 1 letter),
            # extract only the numeric part for comparison of first 4/6 digits
            if re.match(r'^[0-9]{10}[a-zA-Z]{1}$', code, re.IGNORECASE):
                cleaned_codes.append(code[:10]) # Use only the numeric part for prefix comparison
            elif re.match(r'^[0-9.]+$', code): # Standard numeric HS code
                cleaned_codes.append(re.sub(r'\.', '', code).strip())
            else:
                cleaned_codes.append("") # Mark as invalid for agreement calculation
        else:
            cleaned_codes.append("") # Mark as invalid for agreement calculation

    # Filter out empty codes for agreement check
    valid_codes_for_agreement = [c for c in cleaned_codes if c]

    if len(valid_codes_for_agreement) < 3: # Ensure exactly 3 valid codes for agreement
        return 0

    # Check 6-digit agreement
    first_six_digits = [code[:6] for code in valid_codes_for_agreement if len(code) >= 6]
    if len(first_six_digits) == 3 and len(set(first_six_digits)) == 1:
        return 10 # All three share the same first 6 digits

    # Check 4-digit agreement
    first_four_digits = [code[:4] for code in valid_codes_for_agreement if len(code) >= 4]
    if len(first_four_digits) == 3 and len(set(first_four_digits)) == 1:
        return 5 # All three share the same first 4 digits

    return 0 # No significant agreement

#  Main function to calculate combined certainty (MODIFIED RETURN & WEIGHTS)
def calculate_final_certainty(
    product_description,
    country,
    proposed_hs_code,
    product_gender,
    reasoning_text,
    legal_basis_text, 
    all_proposed_hs_codes_for_product, # input: list of [hs1, hs2, hs3] for inter-option
    certainty_config #  Pass certainty configuration
):
    """
    Calculates a combined certainty score (0-100%) for a proposed HS code
    based on historical data, reasoning length, HS code format,
    legal basis citation quality, and inter-option agreement.
    Returns a dictionary with final certainty and breakdown.
    """
    #  Handle 'none' calculation mode
    if certainty_config['calculation_mode'] == "none":
        return {
            'final_certainty': 0,
            'historical_score': 0,
            'reasoning_score': 0,
            'format_score': 0,
            'citation_score': 0,
            'inter_option_agreement_score': 0
        }

    # Determine active weights based on configuration
    if certainty_config['calculation_mode'] == "standard":
        active_weights = DEFAULT_CERTAINTY_WEIGHTS
    elif certainty_config['calculation_mode'] == "custom":
        active_weights = certainty_config['custom_weights']
        # Normalize custom weights if their sum is not 100 for consistency in scoring
        # (Though admin panel will enforce 100 on save, this is a fallback)
        current_total_weight = sum(active_weights.values())
        if current_total_weight == 0: # Avoid division by zero
            st.warning("All custom weights are 0, certainty will be 0.")
            active_weights = {k: 0 for k in active_weights}
        elif current_total_weight != 100:
            scale_factor = 100 / current_total_weight
            active_weights = {k: v * scale_factor for k, v in active_weights.items()}
            # Re-convert to int after scaling if desired, or keep as float for precision
            # For simplicity, we assume admin panel saves as ints summing to 100
    else:
        # Fallback to standard if an unexpected mode is passed
        st.warning(f"Unknown certainty calculation mode '{certainty_config['calculation_mode']}'. Using standard weights.")
        active_weights = DEFAULT_CERTAINTY_WEIGHTS


    # Calculate individual component scores (on their own scales)
    hist_score = _calculate_historical_certainty(product_description, country, proposed_hs_code, product_gender) # 0-100
    reasoning_score = _calculate_reasoning_score(reasoning_text) # 0-20
    format_score = _calculate_format_score(proposed_hs_code, country) # 0-10
    citation_score = _calculate_citation_quality(legal_basis_text) # 0-10
    inter_option_agreement_score = _calculate_inter_option_agreement(all_proposed_hs_codes_for_product) # 0-10

    # Scale individual scores to their weighted contribution (assuming active_weights sum to 100)
    # Each component score is normalized to 0-1 then multiplied by its weight
    weighted_hist = (hist_score / 100) * active_weights.get('historical', 0)
    weighted_reasoning = (reasoning_score / 20) * active_weights.get('reasoning', 0)
    weighted_format = (format_score / 10) * active_weights.get('format', 0)
    weighted_citation = (citation_score / 10) * active_weights.get('citation', 0)
    weighted_agreement = (inter_option_agreement_score / 10) * active_weights.get('agreement', 0)

    final_certainty = weighted_hist + weighted_reasoning + weighted_format + weighted_citation + weighted_agreement

    # Ensure final certainty is within 0-100 range
    final_certainty = max(0, min(100, final_certainty))

    return {
        'final_certainty': int(final_certainty),
        'historical_score': hist_score, # 0-100
        'reasoning_score': reasoning_score, # 0-20
        'format_score': format_score, # 0-10
        'citation_score': citation_score, # 0-10
        'inter_option_agreement_score': inter_option_agreement_score # 0-10
    }

# Enhanced Generate HS Codes (PROMPT MODIFIED to ask for Legal Basis)
def generate_hs_codes(
    model,
    product_description,
    country,
    relevant_chapters,
    legal_notes,
    classification_guide="",
    gri="",
    rejected_codes_snapshot=None,
    guidelines=None,
):
    print(f"[DEBUG] Product Description: {product_description}")

    # Handle rejected codes
    if rejected_codes_snapshot is None:
        rejected_entries = load_rejected_codes(product_description, country)
    else:
        rejected_entries = rejected_codes_snapshot
    temp_rejected_hs_codes = []
    for entry in rejected_entries:
        code = entry.get("rejected_code")
        if code and isinstance(code, str):
            temp_rejected_hs_codes.append(code.strip())

    rejected_hs_codes_for_prompt = list(set(filter(None, temp_rejected_hs_codes)))
    rejected_section_for_prompt = ""
    if rejected_hs_codes_for_prompt:
        rejected_section_for_prompt += "\n\nIMPORTANT: DO NOT SUGGEST ANY OF THE FOLLOWING HS CODES (these were previously rejected for this item by a specialist):\n"
        rejected_section_for_prompt += "\n".join(f"- {code_str}" for code_str in rejected_hs_codes_for_prompt)
        rejected_section_for_prompt += "\n\nEnsure that none of your three new suggested codes match any of the HS codes listed directly above."


    # Get historical data string for prompt 
    historical_data_string = format_historical_data_from_csv(
        target_country=country,
        target_full_product_description=product_description,
        csv_file_path=CSV_PATH
    )

    prompt = f"""
    **CONTEXT & RESOURCES:**
    - **Product Description:** {product_description}
    - **Target Country:** {country.upper()}
    - **Legal Notes:** {legal_notes}
    - **Country Guidelines:** {guidelines}
    - **Classification Guide:** {classification_guide}
    - **General Rules of Interpretation (GRI):** {gri}
    - **OFFICIAL CHAPTER CONTENT:** (Provided below)
    - **Previously Rejected HS Codes (DO NOT USE):** {", ".join(rejected_hs_codes_for_prompt)}
    - **Previously Excluded Chapters/Sections (DO NOT USE CODES FROM HERE):** {rejected_section_for_prompt}

    **HISTORICAL DATA (SIMILAR FOOTWEAR PRODUCTS & CLASSIFICATIONS):**
    {historical_data_string}

    ---

    **ROLE:** You are an expert customs classifier specializing in FOOTWEAR, with expertise in multiple countries, tasked with accurately classifying footwear products using only the provided official Harmonized System (HS) documents. Precision is paramount.

    **ULTRA-CRITICAL FOOTWEAR CLASSIFICATION RULES (STRICT COMPLIANCE REQUIRED):**

    1.  **ABSOLUTE CODE VALIDITY:** ONLY propose HS codes that appear *VERBATIM* in the official documentation.
    2.  **REJECTED CODES ARE FORBIDDEN:** Absolutely DO NOT suggest any code from `Previously Rejected HS Codes`.
    3.  **CHAPTER 64 FOCUS:** Footwear is primarily classified in Chapter 64.
    4.  **SOLE MATERIAL PRIORITY:** Classification depends heavily on outer sole material (rubber, leather, textile, etc.)
    5.  **UPPER MATERIAL CONSIDERATION:** Upper material affects subheading classification
    6.  **SPORTS vs REGULAR FOOTWEAR:** Athletic/sports footwear has specific subheadings (e.g., 6404.11)
    7.  **GENDER CLASSIFICATION:** Many tariff lines distinguish between men's/boys' and women's/girls' footwear
    8.  **STATISTICAL SUFFIX PRIORITY:** Use the most specific, applicable statistical suffixes
    9.  **DIGIT-LENGTH ENFORCEMENT:** The proposed HS code **must** exactly match the digit length required for the target country. If it does not, **reject and do not propose**.

    **COUNTRY-SPECIFIC CODE LENGTHS (STRICT):**
    - **Switzerland:** EXACTLY 11 digits
    - **New Zealand:** EXACTLY 10 digits followed by 1 letter suffix
    - **Europe, Canada, Australia, United States:** EXACTLY 10 digits
    - **Japan:** EXACTLY 9 digits
    - **Brazil, Norway:** EXACTLY 8 digits
    - **South Korea:** EXACTLY 10 digits

    **FOOTWEAR-SPECIFIC CLASSIFICATION LOGIC:**
    - **Athletic Shoes** (tennis, basketball, running, training, gym): Usually 6404.11 with textile uppers
    - **Dress Shoes:** Often 6403.x with leather uppers
    - **Boots:** Consider height and use (fashion vs work vs hiking)
    - **Sandals:** Open footwear, often 6404.x or 6402.x depending on sole
    - **Casual Shoes:** Broad category, classify by construction and materials
    - **Children's Footwear:** Often has separate subheadings

    **TASK:**
    Based *exclusively* on the provided content, determine the *THREE most likely HS codes* for the footwear product.

    **FORMAT (Strictly Adhere):**

    ### OPTION 1: [HS code]

    #### PRODUCT DESCRIPTION:
    [Re-state the footwear product focusing on classification-relevant details: type of shoe, sole material, upper material, construction, intended use, gender]

    #### REASONING STRUCTURE:
    1. *GRI Application*: Apply General Rules of Interpretation systematically
    2. *Historical Data Consideration*: How historical footwear data influenced this option
    3. *Chapter Determination*: Why Chapter 64 (or other) was chosen for this footwear
    4. *Heading Selection*: Justify the 4-digit heading based on sole material and construction
    5. *Subheading Determination*: Explain 6-digit subheading based on upper material and use
    6. *National Tariff Line*: Final digits based on gender, specific shoe type, etc.
    7. *Footwear-Specific Considerations*: Athletic vs dress, protective features, construction method

    #### LEGAL BASIS:
    Cite specific GRI rules, Chapter 64 notes, and heading/subheading texts from provided documents.

    ### OPTION 2: [HS code]
    [Same structure as Option 1]

    ### OPTION 3: [HS code]
    [Same structure as Option 1]

    **FINAL VERIFICATION:** Before outputting, confirm:
    - No proposed code is in Previously Rejected HS Codes
    - All codes exist verbatim in official documentation
    - All codes exactly match the required digit length for {country.upper()}
    - Footwear-specific rules were applied correctly
    - Gender distinctions were considered
    - Sole and upper materials match classification logic
    """


    # Add chapter content to prompt
    for chapter_num, chapter_text in relevant_chapters:
        prompt += f"\n--- CHAPTER {chapter_num} CONTENT ---\n{chapter_text}\n"

    try:
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.5,
            "top_k": 15,
            "max_output_tokens": 15000,
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        token_count = model.count_tokens(prompt).total_tokens
        print(f"The prompt contains {token_count} tokens.")

        return response.text
    except Exception as e:
        return f"Error generating HS codes: {str(e)}"

# Extract HS Codes from Response (MODIFIED: Now extracts Legal Basis)
def extract_hs_codes(text):
    # Extract product description
    product_desc_match = re.search(r'#### PRODUCT DESCRIPTION:\s*(.*?)(?=####|$)', text, re.DOTALL)
    product_description = product_desc_match.group(1).strip() if product_desc_match else "Not found"

    # The pattern now looks for "### OPTION N: [HS code]" and captures everything until the next option or end of string
    options_raw = re.findall(r'### OPTION (\d+): ([0-9]+(?:\.[0-9]+)*(?:\s+[0-9]+)?)\s*(.*?)(?=### OPTION \d+:|$)',
                             text, re.DOTALL)

    row = {'product_description': product_description}
    all_extracted_hs_codes = [] # To collect codes for inter-option agreement calculation

    # First pass: Extract all codes and basic info, and store Legal Basis
    temp_options_data = {}
    for i, (option_num_str, code, details) in enumerate(options_raw):
        option_num = int(option_num_str)

        reasoning_match = re.search(r'#### REASONING(?:\s+STRUCTURE)?:(.*?)(?=#### LEGAL BASIS:|$)', details, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"

        legal_basis_match = re.search(r'#### LEGAL BASIS:\s*(.*?)(?=\s*### OPTION \d+:|$)', details, re.DOTALL)
        legal_basis = legal_basis_match.group(1).strip() if legal_basis_match else "Not found"


        temp_options_data[option_num] = {
            'hs_code': code.strip(),
            'reasoning': reasoning,
            'legal_basis': legal_basis
        }
        all_extracted_hs_codes.append(code.strip()) # Collect valid HS codes for inter-option check

    # Calculate inter-option agreement once for the whole set of codes
    inter_option_agreement_score = _calculate_inter_option_agreement(all_extracted_hs_codes)

    # Second pass: Populate the row with all calculated scores
    for i in range(1, 4):
        option_num = i
        if option_num in temp_options_data:
            data = temp_options_data[option_num]
            row[f'hs_code_{option_num}'] = data['hs_code']
            row[f'reasoning_{option_num}'] = data['reasoning']
            row[f'legal_basis_{option_num}'] = data['legal_basis'] # Store legal basis
            row[f'certainty_{option_num}'] = 0 # Will be calculated externally, placeholder
            row[f'hist_certainty_{option_num}'] = 0 # Placeholder
            row[f'reasoning_score_{option_num}'] = 0 # Placeholder
            row[f'format_score_{option_num}'] = 0 # Placeholder
            row[f'citation_score_{option_num}'] = 0 # placeholder
            row[f'inter_option_agreement_score_{option_num}'] = inter_option_agreement_score #  shared score for this product
        else:
            # Fill any missing options if AI didn't provide 3
            row[f'hs_code_{option_num}'] = ""
            row[f'certainty_{option_num}'] = 0
            row[f'reasoning_{option_num}'] = ""
            row[f'legal_basis_{option_num}'] = "" # Initialize for missing options
            row[f'hist_certainty_{option_num}'] = 0
            row[f'reasoning_score_{option_num}'] = 0
            row[f'format_score_{option_num}'] = 0
            row[f'citation_score_{option_num}'] = 0
            row[f'inter_option_agreement_score_{option_num}'] = 0 # No agreement if options are missing

    return pd.DataFrame([row])

# Bulk Processing Function (MODIFIED: Calls new final certainty calculation and stores breakdown)
def process_bulk_data(
    df_input,
    model,
    pdf_data_cache,
    country_col,
    name_col,
    name_col2,
    material_col,
    construction_col,
    gender_col,
    certainty_config # Pass certainty configuration here
):
    all_results_list = []
    total_rows = len(df_input)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    for df_idx, row in df_input.iterrows():
        original_index = row["original_index"]

        current_progress = (df_idx + 1) / total_rows
        progress_bar.progress(current_progress)
        elapsed_time = time.time() - start_time
        est_remaining = (elapsed_time / (df_idx + 1)) * (total_rows - (df_idx + 1)) if df_idx > 0 else 0
        progress_text.text(f"Processing row {df_idx + 1}/{total_rows}... Est. time remaining: {int(est_remaining)}s")

        # Build Product Description
        country = str(row[country_col]).strip().lower() if pd.notna(row[country_col]) else "unknown"
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
        full_product_description = " ".join(desc_parts).strip() # Renamed to avoid clash with loop variable 'product_description' in row

        base_result_row = {
            "original_index": original_index,
            "input_country": country.upper(),
            "input_product": product_type,
            "input_material": material,
            "input_construction": construction,
            "input_gender": gender,
            "product_description": full_product_description, # Use the combined description
            "hs_code_1": "N/A", "certainty_1": 0, "reasoning_1": "Skipped", "legal_basis_1": "", # Added legal_basis
            "hist_certainty_1": 0, "reasoning_score_1": 0, "format_score_1": 0,
            "citation_score_1": 0, "inter_option_agreement_score_1": 0, # scores
            "hs_code_2": "", "certainty_2": 0, "reasoning_2": "", "legal_basis_2": "",
            "hist_certainty_2": 0, "reasoning_score_2": 0, "format_score_2": 0,
            "citation_score_2": 0, "inter_option_agreement_score_2": 0,
            "hs_code_3": "", "certainty_3": 0, "reasoning_3": "", "legal_basis_3": "",
            "hist_certainty_3": 0, "reasoning_score_3": 0, "format_score_3": 0,
            "citation_score_3": 0, "inter_option_agreement_score_3": 0
        }

        if not product_type or not material or country == "unknown":
            st.warning(f"Skipping row {df_idx + 1} (Original Index: {original_index}): Insufficient data")
            base_result_row["reasoning_1"] = "Skipped due to missing essential data"
            all_results_list.append(base_result_row)
            continue

        if country not in pdf_data_cache:
            st.warning(f"No PDF data available for {country}.")
            base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
            all_results_list.append(base_result_row)
            continue

        processed_pdfs_for_current_country = pdf_data_cache[country]
        relevant_chapters = find_relevant_chapters(full_product_description, country, processed_pdfs_for_current_country)
        legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
        classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
        gri = processed_pdfs_for_current_country.get("gri", "")

        # Load country-specific text files
        country_texts = load_text_files_for_country(PDF_DIRECTORY, country)
        guidelines = country_texts.get(f"{country}_guidelines", "")

        try:
            rejected_codes_snapshot = load_rejected_codes(full_product_description, country)

            generated_response = generate_hs_codes(
                model,
                full_product_description,
                country,
                relevant_chapters,
                legal_notes,
                classification_guide,
                gri=gri,
                guidelines=guidelines,
                rejected_codes_snapshot=rejected_codes_snapshot
            )

            product_df_row_extracted = extract_hs_codes(generated_response)

            if not product_df_row_extracted.empty:
                extracted_data = product_df_row_extracted.iloc[0].to_dict()

                # Get all proposed HS codes from the new response for this product
                all_current_hs_codes = [
                    extracted_data.get('hs_code_1', ''),
                    extracted_data.get('hs_code_2', ''),
                    extracted_data.get('hs_code_3', '')
                ]
                # Calculate inter-option agreement once
                # Note: inter_option_agreement_score is already part of extracted_data from extract_hs_codes
                # We reuse that if available, otherwise recalculate as fallback.
                inter_option_agreement_score = extracted_data.get('inter_option_agreement_score_1',
                                                                    _calculate_inter_option_agreement(all_current_hs_codes))


                for i in range(1, 4):
                    hs_col = f"hs_code_{i}"
                    cert_col = f"certainty_{i}"
                    reas_col = f"reasoning_{i}"
                    legal_basis_col = f"legal_basis_{i}" 
                    hist_cert_col = f"hist_certainty_{i}"
                    reason_score_col = f"reasoning_score_{i}"
                    format_score_col = f"format_score_{i}"
                    citation_score_col = f"citation_score_{i}" 
                    inter_option_agreement_score_col = f"inter_option_agreement_score_{i}" 

                    if hs_col in extracted_data and extracted_data.get(hs_col) != "":
                        base_result_row[hs_col] = extracted_data.get(hs_col, "")
                        base_result_row[reas_col] = extracted_data.get(reas_col, "")
                        base_result_row[legal_basis_col] = extracted_data.get(legal_basis_col, "") # Store legal basis

                        # Calculate and assign external certainty here, and get breakdown
                        certainty_breakdown = calculate_final_certainty(
                            product_description=full_product_description,
                            country=country,
                            proposed_hs_code=base_result_row[hs_col],
                            product_gender=gender,
                            reasoning_text=base_result_row[reas_col],
                            legal_basis_text=base_result_row[legal_basis_col], # Pass legal basis
                            all_proposed_hs_codes_for_product=all_current_hs_codes, # Pass all codes for this product
                            certainty_config=certainty_config #  Pass certainty config
                        )
                        base_result_row[cert_col] = certainty_breakdown['final_certainty']
                        base_result_row[hist_cert_col] = certainty_breakdown['historical_score']
                        base_result_row[reason_score_col] = certainty_breakdown['reasoning_score']
                        base_result_row[format_score_col] = certainty_breakdown['format_score']
                        base_result_row[citation_score_col] = certainty_breakdown['citation_score'] # Store new citation score
                        base_result_row[inter_option_agreement_score_col] = certainty_breakdown['inter_option_agreement_score'] # Store new agreement score
                    else:
                        base_result_row[hs_col] = ""
                        base_result_row[cert_col] = 0
                        base_result_row[reas_col] = ""
                        base_result_row[legal_basis_col] = ""
                        base_result_row[hist_cert_col] = 0
                        base_result_row[reason_score_col] = 0
                        base_result_row[format_score_col] = 0
                        base_result_row[citation_score_col] = 0
                        base_result_row[inter_option_agreement_score_col] = 0
            else:
                st.error(f"Failed to extract codes for row {df_idx + 1} (Original Index: {original_index}). Response format might be unexpected.")
                base_result_row["reasoning_1"] = "Error: Failed to parse response"

            all_results_list.append(base_result_row)

        except Exception as gen_e:
            st.error(f"Error generating code for row {df_idx + 1} (Original Index: {original_index}): {gen_e}")
            base_result_row["hs_code_1"] = "ERROR"
            base_result_row["reasoning_1"] = str(gen_e)
            all_results_list.append(base_result_row)

    return pd.DataFrame(all_results_list)

def check_password():
    # If "password_correct" is already True, user is authenticated for this run.
    # This comes from a previous successful login within the current session's run.
    if st.session_state.get("password_correct", False):
        return True

    # Use st.form to group inputs and the submit button
    with st.form("login_form", clear_on_submit=False): # Set clear_on_submit=False to keep inputs filled
        username = st.text_input(
            "Username",
            key="username_input", # Unique key for session state persistence
            autocomplete="username", # Explicitly request browser autofill
            placeholder="Enter your username"
        )
        password = st.text_input(
            "Password",
            type="password",
            key="password_input", # Unique key for session state persistence
            autocomplete="current-password", # Explicitly request browser autofill
            placeholder="Enter your password"
        )

        # The form's submit button
        submitted = st.form_submit_button("Login")

    # Check credentials only AFTER the form has been submitted
    if submitted:
        # These 'username' and 'password' variables now hold the values that were
        # present in the input fields *at the moment the form was submitted*.
        # This should include autofilled values.
        if (
            username == st.secrets["AUTH"]["USERNAME"]
            and password == st.secrets["AUTH"]["PASSWORD"]
        ):
            st.session_state["password_correct"] = True
            st.success("Login successful!")
            return True
        else:
            st.session_state["password_correct"] = False
            st.error("Incorrect username or password")
            return False # Return False if credentials are incorrect

    # If the form hasn't been submitted yet, or credentials were incorrect, keep showing the form.
    return False

# Regeneration Functions 
def regenerate_single_product(original_index):
    if st.session_state.bulk_results_df is None:
        print("bulk_results_df is None. Returning.")
        return
    try:
        product_row_series = st.session_state.bulk_results_df[st.session_state.bulk_results_df['original_index'] == original_index]
        if product_row_series.empty:
            print("product_row_series is empty. Returning.")
            return

        product_row = product_row_series.iloc[0]
        print(f"product_row: \n{product_row}")

        if st.session_state.model is None:
            print("Model is None. Returning.")
            return

        model = st.session_state.model
        product_description = product_row['product_description'] # This is the already built full description
        country = product_row['input_country'].strip().lower()
        gender = product_row['input_gender'] # Get gender for certainty calculation

        if country not in st.session_state.pdf_cache or not st.session_state.pdf_cache[country]:
            print(f"PDF cache miss for {country}. Loading PDFs.")
            st.session_state.pdf_cache = load_all_pdf_data() # Reload all PDFs if cache is missing for country
            if not st.session_state.pdf_cache.get(country):
                print(f"Could not load PDFs for {country}. Returning.")
                st.error(f"Error: Could not load tariff documentation for {country}.")
                return

        processed_pdfs = st.session_state.pdf_cache[country]
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs)
        legal_notes = processed_pdfs.get("legal_notes", "")
        guide = processed_pdfs.get("classification_guide", "")
        gri = processed_pdfs.get("gri", "")
        rejected_codes_snapshot = load_rejected_codes(product_description, country)

        # Load country-specific text files for guidelines
        country_texts = load_text_files_for_country(PDF_DIRECTORY, country)
        guidelines = country_texts.get(f"{country}_guidelines", "")

        # Load certainty config for this regeneration
        certainty_config = load_certainty_config()

        with st.spinner(f"Regenerating Index: {original_index}..."):
            new_response = generate_hs_codes(
                model,
                product_description,
                country,
                relevant_chapters,
                legal_notes,
                guide,
                gri=gri,
                guidelines=guidelines,
                rejected_codes_snapshot=rejected_codes_snapshot
            )
            new_product_df_row_extracted = extract_hs_codes(new_response)

            if not new_product_df_row_extracted.empty:
                df = st.session_state.bulk_results_df
                target_df_index = df[df['original_index'] == original_index].index
                if not target_df_index.empty:
                    idx_loc = target_df_index[0]
                    new_data = new_product_df_row_extracted.iloc[0]

                    # Get all proposed HS codes from the new response for this product
                    all_current_hs_codes = [
                        new_data.get('hs_code_1', ''),
                        new_data.get('hs_code_2', ''),
                        new_data.get('hs_code_3', '')
                    ]
                    # Calculate inter-option agreement once for all options of this product
                    # Note: inter_option_agreement_score is already part of new_data from extract_hs_codes
                    inter_option_agreement_score = new_data.get('inter_option_agreement_score_1',
                                                                  _calculate_inter_option_agreement(all_current_hs_codes))


                    for i in range(1, 4):
                        hs_col = f'hs_code_{i}'
                        cert_col = f'certainty_{i}'
                        reas_col = f'reasoning_{i}'
                        legal_basis_col = f'legal_basis_{i}' 
                        hist_cert_col = f"hist_certainty_{i}"
                        reason_score_col = f"reasoning_score_{i}"
                        format_score_col = f"format_score_{i}"
                        citation_score_col = f"citation_score_{i}" 
                        inter_option_agreement_score_col = f"inter_option_agreement_score_{i}" 

                        if hs_col in new_data and new_data.get(hs_col) != "":
                            df.loc[idx_loc, hs_col] = new_data.get(hs_col, '')
                            df.loc[idx_loc, reas_col] = new_data.get(reas_col, '')
                            df.loc[idx_loc, legal_basis_col] = new_data.get(legal_basis_col, '') # Store legal basis

                            # Calculate and assign external certainty here, and get breakdown
                            certainty_breakdown = calculate_final_certainty(
                                product_description=product_description,
                                country=country,
                                proposed_hs_code=df.loc[idx_loc, hs_col], # Use the newly set HS code
                                product_gender=gender,
                                reasoning_text=df.loc[idx_loc, reas_col],
                                legal_basis_text=df.loc[idx_loc, legal_basis_col], # Pass legal basis
                                all_proposed_hs_codes_for_product=all_current_hs_codes, # Pass all codes for this product
                                certainty_config=certainty_config # Pass certainty config
                            )
                            df.loc[idx_loc, cert_col] = certainty_breakdown['final_certainty']
                            df.loc[idx_loc, hist_cert_col] = certainty_breakdown['historical_score']
                            df.loc[idx_loc, reason_score_col] = certainty_breakdown['reasoning_score']
                            df.loc[idx_loc, format_score_col] = certainty_breakdown['format_score']
                            df.loc[idx_loc, citation_score_col] = certainty_breakdown['citation_score'] # Store new citation score
                            df.loc[idx_loc, inter_option_agreement_score_col] = certainty_breakdown['inter_option_agreement_score'] # Store new agreement score
                        else:
                            df.loc[idx_loc, hs_col] = ""
                            df.loc[idx_loc, cert_col] = 0
                            df.loc[idx_loc, reas_col] = ""
                            df.loc[idx_loc, legal_basis_col] = ""
                            df.loc[idx_loc, hist_cert_col] = 0
                            df.loc[idx_loc, reason_score_col] = 0
                            df.loc[idx_loc, format_score_col] = 0
                            df.loc[idx_loc, citation_score_col] = 0
                            df.loc[idx_loc, inter_option_agreement_score_col] = 0

                    if 'product_selections' not in st.session_state:
                        st.session_state.product_selections = {}
                    st.session_state.product_selections[original_index] = {'status': 'pending', 'data': {}}
                    st.session_state.bulk_results_df = df
                    print(f"DataFrame after update: \n{df.head()}")
                    st.toast(f"Regenerated Index {original_index}. Review new options.")
                else:
                    st.error(f"Regen Update Error: Index {original_index} not found.")
            else:
                print("Error: Failed to extract codes from Gemini response.")
                st.error(f"Regen Error Index {original_index}: Failed to extract codes.")
    except Exception as regen_e:
        print(f"Exception during regeneration: {regen_e}")
        st.error(f"Regen Error Index {original_index}: {regen_e}")
    finally:
        if 'regenerate_queue' not in st.session_state:
            st.session_state.regenerate_queue = set()
        st.session_state.regenerate_queue.discard(original_index)

# Find Relevant Chapters for Footwear (No changes needed here)
def find_relevant_chapters(product_description, country, country_specific_pdf_data):
    keywords = {
        # Footwear keywords - Chapter 64
        "shoe": ["64"], "boot": ["64"], "sneaker": ["64"], "sandal": ["64"],
        "slipper": ["64"], "pump": ["64"], "loafer": ["64"], "oxford": ["64"],
        "athletic": ["64"], "running": ["64"], "basketball": ["64"], "tennis": ["64"],
        "training": ["64"], "gym": ["64"], "hiking": ["64"], "work": ["64"],
        "safety": ["64"], "dress": ["64"], "casual": ["64"], "formal": ["64"],
        "heel": ["64"], "flat": ["64"], "wedge": ["64"], "platform": ["64"],
        "moccasin": ["64"], "espadrille": ["64"], "clog": ["64"],
        "flip-flop": ["64"], "thong": ["64"], "footwear": ["64"],

        # Other categories
        "bag": ["42"], "backpack": ["42"],
        "hat": ["65"], "cap": ["65"],
    }

    product_lower = product_description.lower()
    potential_chapters = set()

    for keyword, chapters in keywords.items():
        if keyword in product_lower:
            for chapter in chapters:
                potential_chapters.add(chapter)

    # Default to Chapter 64 for footwear if no specific match
    if any(term in product_lower for term in ["sole", "upper", "leather", "rubber", "canvas"]):
        potential_chapters.add("64")

    relevant_chapters_content = []
    for chapter_num in potential_chapters:
        chapter_key_in_cache = f"chapter_{chapter_num}"
        if chapter_key_in_cache in country_specific_pdf_data:
            relevant_chapters_content.append((chapter_num, country_specific_pdf_data[chapter_key_in_cache]))

    return relevant_chapters_content
