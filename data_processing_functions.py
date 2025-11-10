"""
Data Processing Functions Module.

This module contains functions for processing and extracting information
from text data related to HS codes, as well as formatting historical data
from CSV files."""

import re
import pandas as pd
from persistence_helper_functions import log_interaction_event
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#___Variables___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "composition"
construction_col = "material_type"
gender_col = "division"
hs_code_col = "tariff_code"


def parse_reasoning_text(raw_reasoning_text):
    """Parses the raw reasoning text into structured sections.
    Args:
        raw_reasoning_text (str): The raw reasoning text from the model output.
    Returns:
        dict: A dictionary with structured reasoning sections."""
    structured_reasoning = {}

    # defining the sections and their regex patterns
    sections = [
        ("GRI Application", r"1\.?\s*\*GRI Application\*:\s*(.*?)(?=2\.?\s*\*Historical Data Consideration\*|3\.?\s*\*Chapter & Section Fit\*|4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Historical Data Consideration", r"2\.?\s*\*Historical Data Consideration\*:\s*(.*?)(?=3\.?\s*\*Chapter & Section Fit\*|4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Chapter & Section Fit", r"3\.?\s*\*Chapter & Section Fit\*:\s*(.*?)(?=4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Heading & Subheading Determination", r"4\.?\s*\*Heading & Subheading Determination\*:\s*(.*?)(?=5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("National Tariff Line Determination", r"5\.?\s*\*National Tariff Line Determination\*:\s*(.*?)(?=6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Exclusions & Verifications", r"6\.?\s*\*Exclusions & Verifications\*:\s*(.*?)(?=$)")
    ]

    # attempting to extract each section
    for section_name, pattern in sections:
        match = re.search(pattern, raw_reasoning_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # cleaning up markdown asterisks and leading/trailing whitespace
            content = re.sub(r'\*\s*', '', content).strip()
            content = re.sub(r'\s*\*\s*', ' ', content).strip() # replacing single asterisks with space
            structured_reasoning[section_name] = content
        else:
            structured_reasoning[section_name] = "Not explicitly addressed in response."

    # handling the case where there's a simple "NO-CODE-FOUND"
    if "NO-CODE-FOUND" in raw_reasoning_text and not structured_reasoning:
        structured_reasoning["General Reason"] = raw_reasoning_text.replace("#### REASONING:\n\n", "").strip()

    return structured_reasoning


def extract_hs_codes(text):
    """Extracts HS codes, certainty levels, and reasoning from the provided text.
    Args:
        text (str): The text containing HS code options and reasoning.
    Returns:
        pd.DataFrame: A DataFrame with extracted HS codes, certainty levels, and reasoning.
    """
    # extracting the product description from JSON format
    product_desc_match = re.search(r'"Product Description":"(.*?)"', text)

    # if no JSON format product description, extracts from the first option's product description
    if not product_desc_match:
        product_desc_match = re.search(r'#### PRODUCT DESCRIPTION:\s*(.*?)(?=####|$)', text, re.DOTALL)

    product_description = product_desc_match.group(1).strip() if product_desc_match else "Not found"

    # pattern to capture the full HS code format with all digits, periods, and spaces
    options = re.findall(r'### OPTION \d+: ([0-9]+(?:\.[0-9]+)*(?:[ -][0-9]+)?|NO-CODE-FOUND) - (\d+)% certainty(?:\s+\(Validated\))?\s+(.*?)(?=### OPTION \d+:|$)', 
                     text, re.DOTALL)

    row = {'product_description': product_description}

    for i, (code, certainty, details) in enumerate(options):
        # extracting the main reasoning block
        reasoning_match = re.search(r'#### REASONING(?: STRUCTURE)?:(.*?)(?=#### LEGAL BASIS:|$)', details, re.DOTALL)
        raw_reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."

        parsed_reasoning = parse_reasoning_text(raw_reasoning_text)

        option_num = i + 1
        row[f'hs_code_{option_num}'] = code.strip()
        row[f'certainty_{option_num}'] = int(certainty)
        row[f'reasoning_{option_num}'] = parsed_reasoning
        row[f'raw_reasoning_text_{option_num}'] = raw_reasoning_text
        
        if "NO-CODE-FOUND" in code:
            log_interaction_event(
                country="", # country -> not available here, will be added in app.py
                product_type="", # product type -> not available here, will be added in app.py
                event_type="no_code_found",
                hs_code=code
            )

        option_num = i + 1

    df = pd.DataFrame([row])
    return df


def extract_simplified_hs_codes(text):
    """Extracts simplified HS codes and certainty levels from the provided text.
    Args:
        text (str): The text containing HS code options and certainty.
    Returns:
        pd.DataFrame: A DataFrame with extracted simplified HS codes and certainty levels.
        """
    options_data = []
    # pattern to capture HS code and certainty from "### OPTION X: [HS code] - YY% certainty"
    options = re.findall(r'### OPTION \d+: ([0-9]+(?:\.[0-9]+)*(?:\s+[0-9]+)?) - (\d+)% certainty', text)

    for i, (code, certainty) in enumerate(options):
        options_data.append({
            f'Option {i+1} HS Code': code.strip(),
            f'Option {i+1} Certainty': int(certainty)
        })
    return pd.DataFrame(options_data)


def format_historical_data_from_csv(
    csv_file_path,
    target_country,
    target_full_product_description,
    target_gender=None,
    country_col_hist="tariff_country_description",
    name_col_hist="customs_description",
    name_col2_hist="Product Type",
    material_col_hist="Composition",
    construction_col_hist="material_type",
    gender_col_hist="Division",
    hs_code_col_hist="tariff_code",
    similarity_threshold=0.85,
    top_n=5,
    return_df=False
):
    """Formats and retrieves historical data from a CSV file based on similarity to a target product description.
    Args:
        csv_file_path (str): Path to the historical data CSV file.
        target_country (str): The country to filter historical data by.
        target_full_product_description (str): The full product description to compare against.
        target_gender (str, optional): Gender to filter historical data by. Defaults to None.
        country_col_hist (str): Column name for country in historical data.
        name_col_hist (str): Column name for product name in historical data.
        name_col2_hist (str): Column name for secondary product name in historical data.
        material_col_hist (str): Column name for material in historical data.
        construction_col_hist (str): Column name for construction in historical data.
        gender_col_hist (str): Column name for gender in historical data.
        hs_code_col_hist (str): Column name for HS code in historical data.
        similarity_threshold (float): Minimum cosine similarity threshold to consider a match.
        top_n (int): Number of top similar matches to return.
        return_df (bool): If True, returns the filtered DataFrame instead of formatted string.
    Returns:
        str or pd.DataFrame: Formatted string of similar historical data or the DataFrame if return_df is True.
        """

    try:
        print(f"Attempting to read CSV from: {csv_file_path}")
        historical_df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV with {len(historical_df)} rows.")
    except FileNotFoundError:
        return f"Error: CSV file not found at {csv_file_path}"
    except pd.errors.EmptyDataError:
        return f"Error: CSV file at {csv_file_path} is empty"
    except Exception as e:
        return f"Error reading CSV file: {e}"

    if not target_country or not target_full_product_description:
        return "Error: Target country and target full product description must be specified."

    historical_df[country_col_hist] = historical_df[country_col_hist].astype(str)
    historical_df = historical_df[historical_df[country_col_hist].str.lower() == target_country.lower()]

    if historical_df.empty:
        return (
            f"No historical data found for {target_country.upper()}."
        )

    # filtering by target_gender if provided 
    if target_gender:
        # converting historical gender column to string and lowercase for comparison
        historical_df[gender_col_hist] = historical_df[gender_col_hist].astype(str).str.lower()
        target_gender_lower = str(target_gender).lower()

        # Filter: match exactly, or consider "unisex" if target is men/women and vice versa (adjust logic as needed)
        historical_df = historical_df[
            (historical_df[gender_col_hist] == target_gender_lower) |
            (historical_df[gender_col_hist] == 'unisex') | # Example: unisex can match any gender
            (target_gender_lower == 'unisex')             # Example: if target is unisex, match any historical gender
        ]
        if historical_df.empty:
            return (
                f"No historical data found for {target_country.upper()} with gender '{target_gender}'."
            )

    def build_description(row):
        """Builds a full product description from relevant columns in a DataFrame row."""
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

        full_desc = " ".join(parts).strip()
        return full_desc

    historical_df['full_description'] = historical_df.apply(build_description, axis=1)

    if return_df:
        return historical_df

    # Computing cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(historical_df['full_description'])
    target_vector = vectorizer.transform([target_full_product_description])

    cosine_similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
    historical_df['similarity'] = cosine_similarities

    relevant_data = historical_df[historical_df['similarity'] >= similarity_threshold]
    top_matches = relevant_data.nlargest(top_n, 'similarity')

    output_header = (
        f"PREVIOUS CLASSIFICATIONS (SIMILAR PRODUCTS) IN {target_country.upper()}:\n"
        f"(Target Product: '{target_full_product_description}')\n"
    )
    formatted_data = output_header
    if top_matches.empty:
        return (
            f"No similar historical data found for '{target_full_product_description}' "
            f"in {target_country.upper()} (Similarity Threshold: {similarity_threshold}).\n"
        )
    
    # Formatting each top match
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
        
        # Combining all parts into a single string
        full_product_details_output = ", ".join(display_parts)
        hs_code = str(row[hs_code_col_hist]) if hs_code_col_hist in row and pd.notna(row[hs_code_col_hist]) else "N/A" # Corrected typo: hs_code_col instead of hs_code_col_hist
        similarity = row['similarity']
        formatted_data += (
            f"- Similar Product (Similarity: {similarity:.2f}): "
            f"{full_product_details_output}, HS Code: {hs_code}\n"
        )

    return formatted_data

