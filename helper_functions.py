"""
Helper Functions Module.

This module contains helper functions for file reading, document loading,
Gemini API configuration, and HS code generation.
"""


import csv
from datetime import datetime
import os
import json
import google.generativeai as genai

import pandas as pd
import PyPDF2
from pathlib import Path
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio 


#___Documentation Path___#
PDF_DIRECTORY = "chapter_data" 
CSV_PATH = "PLM_D365_merged_datasets_no_questionary.csv"
METRICS_RESULTS_FILE = str(pathlib.Path(__file__).parent / "all_bulk_results.csv")
FINAL_SELECTED_FILE = str(pathlib.Path(__file__).parent / "final_user_selected_hs_codes.csv")
TOKEN_LOG_FILE = str(pathlib.Path(__file__).parent / "token_usage_log.csv")
INTERACTION_LOG_FILE = str(pathlib.Path(__file__).parent / "interaction_log.csv")


# rejected codes
REJECTED_CODES_FILE = "rejected_classifications.json"
# validated codes
VALIDATED_CODES_FILE = "validated_classifications.json"

REQUEST_DELAY_SECONDS = 3

PROCESSING_TIMES_FILE = str(pathlib.Path(__file__).parent / "processing_times.csv") 


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
    """Loads all documents from the specified directory into a nested dictionary.
    Also returns a sorted list of unique countries found in the filenames.
    Args:
        directory: The directory containing the document files.
    Returns:
        A tuple containing:
        - A nested dictionary structured as {country: {doc_type: content, ...}, ...}
        - A sorted list of unique countries.
    """
    doc_cache = {}
    country_set = set()

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return {}, []

    # Iterate over files in the directory
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
    """Loads documents for a specific country from the specified directory.
    Args:
        directory: The directory containing the document files.
        country: The target country to load documents for.
    Returns:
        A dictionary structured as {doc_type: content, ...} for the specified country.
        """
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
    """Extracts text from a PDF file.
    Args:
        pdf_path: The full path to the PDF file.
    Returns:
        The extracted text as a single string.
    """
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
    Args:
        pdf_directory: The directory containing the PDF files.
    Returns:
        A nested dictionary structured as {country: {doc_type: text_content, ...}, ...}
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
    """Loads text files for a specific country from the specified directory.
    Args:
        text_directory: The directory containing the text files.
        country: The target country to load text files for.
        file_suffix: The suffix of the text files to load (default is ".txt").
    Returns:
        A dictionary structured as {filename: content, ...} for the specified country.
    """

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


#___Gemini API Config___# with API key
def configure_genai(api_key):
    """Configures the Google Generative AI (Gemini) API with the provided API key.
    Args:
        api_key: The API key for Google Generative AI.
    Returns:
        A configured GenerativeModel instance."""
    genai.configure(api_key=api_key)
    # using gemini-1.5-flash seems to be enough, does not exceed quota and seems to performa better from 2.0
    # 1.5 can be fine tuned, 2.0 can't
    model = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
    return model

#___HS Code Length Rules___#
HS_CODE_LENGTH_RULES = {
    'canada': 10,
    'usa': 10,
    'norway': 8,
    'switzerland': 11,
    'australia': 10,
    'europe': 8,
}


def log_token_usage(country, product_type, input_tokens, output_tokens):
    """Logs the token usage for a single API call to a CSV file.
    Args:
        country: The target country for the classification.
        product_type: The type of product being classified.
        input_tokens: The number of input tokens used.
        output_tokens: The number of output tokens generated.
    """
    
    file_exists = os.path.exists(TOKEN_LOG_FILE)
    try:
        with open(TOKEN_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'country', 'product_type', 'input_tokens', 'output_tokens', 'total_tokens']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'country': country,
                'product_type': product_type,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            })
    except Exception as e:
        print(f"ERROR: Could not write to token log file. Reason: {e}")



async def generate_hs_codes(
    model,
    product_description,
    country,
    relevant_chapters,
    legal_notes,
    classification_guide,
    gri="",
    rejected_codes_snapshot=None,
    historical_data=None,
    guidelines=None,
    product_type=""
):
    """Generates HS codes using the Gemini API based on the provided product description and country-specific documents.
    Args:
        model: The configured Gemini model instance.
        product_description: The description of the product to classify.
        country: The target country for the classification.
        relevant_chapters: List of tuples containing relevant chapter numbers and their text content.
        legal_notes: The legal notes document content for the country.
        classification_guide: The classification guide document content for the country.
        gri: The General Rules of Interpretation document content for the country (optional).
        rejected_codes_snapshot: A snapshot list of previously rejected codes for the product (optional).
        historical_data: Historical classification data (optional).
        guidelines: Additional country-specific guidelines (optional).
        product_type: The type of product being classified (optional).
    Returns:
        A formatted string containing the generated HS code options and their reasoning.
    """
    print(f"[DEBUG] Product Description: {product_description}")

    # This logic checks for previously validated codes first to avoid unnecessary API calls
    all_validated_entries = load_validated_codes(product_description, country)
    validated_entries_with_code = [e for e in all_validated_entries if e.get("hs_code")]
    if validated_entries_with_code:
        print(f"[DEBUG] Found validated codes for '{product_description}' in {country}. Skipping Gemini call.")
        formatted_response = ""
        for i, entry in enumerate(validated_entries_with_code[:3]):
            option_num = i + 1
            hs_code = entry.get("hs_code", "N/A")
            reasoning = entry.get("reasoning") or "Previously validated by user (code only)."

            # Format reasoning structure
            if isinstance(reasoning, str) and not re.search(r'#### REASONING(?: STRUCTURE)?:', reasoning):
                 reasoning_text_for_option = f"""
                 #### REASONING STRUCTURE:
                 1. *GRI Application*: Not explicitly provided for validated code.
                 2. *Historical Data Consideration*: This code was previously validated.
                 3. *Chapter & Section Fit*: Not explicitly provided for validated code.
                 4. *Heading & Subheading Determination*: Not explicitly provided for validated code.
                 5. *National Tariff Line Determination*: Not explicitly provided for validated code.
                 6. *Exclusions & Verifications*: {reasoning}
                 """
            elif isinstance(reasoning, dict):
                reasoning_text_for_option = "#### REASONING STRUCTURE:\n"
                for sec_name, sec_content in reasoning.items():
                    reasoning_text_for_option += f"{sec_name}: {sec_content}\n"
            else:
                reasoning_text_for_option = f"#### REASONING STRUCTURE:\nGeneral Reason: {reasoning}"


            formatted_response += f"""
                ### OPTION {option_num}: {hs_code} - 100% certainty (Validated)
                #### PRODUCT DESCRIPTION:
                {product_description}
                {reasoning_text_for_option}
                #### LEGAL BASIS:
                Validated by user, specific legal basis not stored.
                """
        return formatted_response.strip()

    # Prepare rejected codes list for the prompt
    if rejected_codes_snapshot is None:
        rejected_entries = load_rejected_codes(product_description, country)
    else:
        rejected_entries = rejected_codes_snapshot

    temp_rejected_hs_codes = []
    for entry in rejected_entries:
        code = entry.get("rejected_code")
        if code and isinstance(code, str):
            temp_rejected_hs_codes.append(code.strip())
        elif code:
            print(f"DEBUG: Found non-string rejected_hs_code in entry: {entry}")

    rejected_hs_codes_for_prompt = list(set(filter(None, temp_rejected_hs_codes)))
    print(f"[DEBUG] Rejected HS Codes for Prompt: {rejected_hs_codes_for_prompt}")

    # Prepare rejected chapters/sections for the prompt
    rejected_section_for_prompt = ""
    if rejected_hs_codes_for_prompt:
        rejected_section_for_prompt += "\n\nIMPORTANT: DO NOT SUGGEST ANY OF THE FOLLOWING HS CODES (these were previously rejected for this item by a specialist):\n"
        rejected_section_for_prompt += "\n".join(f"- {code_str}" for code_str in rejected_hs_codes_for_prompt)
        rejected_section_for_prompt += "\n\nEnsure that none of your three new suggested codes match any of the HS codes listed directly above."
    else:
        print("DEBUG: Condition 'if rejected_hs_codes_for_prompt:' is FALSE. No rejected codes section to add to prompt.")

    # Prepare historical data string for the prompt
    historical_data_string = format_historical_data_from_csv(
        csv_file_path=CSV_PATH,
        target_country=country,
        target_full_product_description=product_description,
    )
    # Prepare chapter content for the prompt
    chapter_content_for_prompt = ""
    if relevant_chapters:
        for chapter_num, chapter_text in relevant_chapters:
            chapter_content_for_prompt += f"\n--- OFFICIAL CHAPTER CONTENT: CHAPTER {chapter_num} ---\n{chapter_text}\n"
            #print(chapter_content_for_prompt)
    else:
        chapter_content_for_prompt = "\n--- OFFICIAL CHAPTER CONTENT: NOT PROVIDED ---\n"

    length_rule_text = "Ensure the HS Code is the correct length as specified in the official documents."
    country_lower = country.lower()

    if country_lower == 'switzerland':
        length_rule_text = (
            "- **HS CODE STRUCTURE (SWITZERLAND):** All suggested HS codes for Switzerland MUST follow a strict format: "
            "an 8-digit base code, followed by the mandatory, unchanging statistical suffix of `'-000'`. "
            "The final output must ALWAYS look like this: `xxxx.xx.xx-000`. NO EXCEPTIONS."
        )
    elif country_lower in HS_CODE_LENGTH_RULES:
        required_length = HS_CODE_LENGTH_RULES[country_lower]
        length_rule_text = (
            f"- **HS CODE LENGTH:** All suggested HS codes for {country.upper()} MUST be "
            f"exactly {required_length} digits long (not counting the periods). "
            "This is a non-negotiable rule. Verify the final digit count. NO EXCEPTIONS."
        )

    length_rule_prompt_block = f"""
    **CRITICAL RULE - HS CODE STRUCTURE:**
    {length_rule_text}
    """

    # Construct the full prompt
    prompt = f"""
        **CONTEXT & RESOURCES:**
        - **Product Description:** {product_description}
        - **Target Country:** {country.upper()}
        - **Legal Notes:** {legal_notes}
        - **Country Guidelines:** {guidelines}
        - **Classification Guide:** {classification_guide}
        - **General Rules of Interpretation (GRI):** {gri}
        - **OFFICIAL CHAPTER CONTENT:** {chapter_content_for_prompt} 
        - **Previously Rejected HS Codes (DO NOT USE):** {", ".join(rejected_hs_codes_for_prompt)}
        - **Previously Excluded Chapters/Sections (DO NOT USE CODES FROM HERE):** {rejected_section_for_prompt}

        {length_rule_prompt_block}

        **HISTORICAL DATA (SIMILAR PRODUCTS & CLASSIFICATIONS):**
        {historical_data_string}

        ---

        **ROLE:** You are an expert customs classifier, with expertise in a wide number of countries, tasked with accurately classifying products using only the provided official Harmonized System (HS) documents. Precision is paramount.
        Every product has a single respective code. You are tasked with finding the most appropriate one.

        **ULTRA-CRITICAL CLASSIFICATION RULES (MANDATORY COMPLIANCE):**

        1.  **ABSOLUTE CODE VALIDITY:** ONLY propose HS codes (including all digits and statistical suffixes) that appear *VERBATIM* in the `OFFICIAL CHAPTER CONTENT`. Do not invent, assume, or truncate. If a code/suffix is not explicitly listed for a relevant heading/subheading, it does not exist.
        2.  **REJECTED CODES ARE FORBIDDEN:** !!important rule!! Before providing your final answer, you MUST perform a final verification. Compare every potential code against the `Previously Rejected HS Codes for this Item` list. If a code is on this list, it is **FORBIDDEN**. You must discard it and find an alternative, even if you believe the rejected code is a good fit. If all potential codes are on the rejected list, you must use the 'NO-CODE-FOUND' format. There are no exceptions to this rule.
        3.  **EXCLUDED CHAPTERS ARE FORBIDDEN:** DO NOT propose codes from chapters/sections listed in `Previously Excluded Chapters/Sections`. Verify the chapter of any potential code against this list.
        4.  **STATISTICAL SUFFIX PRIORITY:** Use the most specific, applicable statistical suffixes (e.g., '.10', '.25', '.99') provided in `OFFICIAL CHAPTER CONTENT`. Only use '.00' if no other specific suffix is listed or appropriate.
        5.  **KNIT/WOVEN INTEGRITY:**
                * **Knitted/Crocheted:** EXCLUSIVELY consider Chapter 61. NEVER suggest codes from Chapter 62 or other non-knitted chapters.
                * **Woven (Not Knitted/Crocheted):** EXCLUSIVELY consider Chapter 62. NEVER suggest codes from Chapter 61 or other knitted chapters.
        6.  **SINGLE CORRECT CODE PRINCIPLE:** Every product has one single, most correct HS code. OPTION 1 MUST be your highest confidence classification. Options 2 and 3 are for genuine ambiguity after rigorous application of all rules and data.
        7.  ** *TIGHTS* = in this context, tights ARE ALWAYS tight-fitting stretchy trousers, similar to leggings, regardless of being KNITTED or WOVEN. Regardless of country.
            * *TIGHTS SHORT* = tight-fitting stretch shorts.
        8.  **Pima Cotton Classification:** *Pima Cotton* is to be treated as pure *cotton*. If Pima Cotton (or any cotton) is the **predominant material** (i.e., it has the HIGHEST PERCENTAGE in a blend), the product's primary classification **MUST** be under the corresponding "Of cotton" heading. NO EXCEPTIONS.

        * **NO DEFAULT CODES:** Do not assume a general code like `.00` or `.0000` exists if it is not explicitly written in the `OFFICIAL CHAPTER CONTENT`. The absence of a specific code means it is an invalid option.
        * **MANDATORY FALLBACK:** If you follow a logical classification path (e.g., Chapter > Heading > Subheading) but cannot find a complete, verbatim national tariff code in the provided documents, you MUST output `NO-CODE-FOUND` for that option. Do not invent, complete, or assume a code exists.

        **HISTORICAL DATA USAGE PROTOCOL:**
        * **For Same Country & Product:** Treat the *full* historical code (from `HISTORICAL DATA`) as a strong candidate, but not the ground thruth, givng special weight to the Gender of the product. Verify its exact existence and applicability in `OFFICIAL CHAPTER CONTENT`, `LEGAL NOTES`, and other documents. If valid, this significantly increases confidence for OPTION 1.
        * **REJECTION OVERRIDES HISTORY:** If a code suggested by `HISTORICAL DATA` is also present in the `Previously Rejected HS Codes for this Item` list, the rejection list ALWAYS wins. The code is forbidden.
        * **For Different Country:** Focus ONLY on the first six digits (international portion) of the historical code as a guide for Chapter/Heading/Subheading. Then, determine national digits/suffixes *solely* from `OFFICIAL CHAPTER CONTENT` for the target country.
        * **Invalid Historical Data:** If historical data points to a code in `Previously Rejected HS Codes` or an `Excluded Chapter`, it is INVALID, regardless of past use.

        **TASK:**
        Based *exclusively and meticulously* on the provided content and adhering to ALL critical rules, determine the *THREE most likely HS codes + statistical suffixes* for the product.

        1.  **Prioritize Historical Data:**
            * If a highly similar and valid historical code exists (as per `HISTORICAL DATA USAGE PROTOCOL`), propose it as **OPTION 1** with high certainty. Explain its origin and validation.
            * If historical data is partially relevant (e.g., different country) or invalid, explain why, then classify based solely on current documents, using historical data only as a guide where appropriate.
        2.  **Determine Classification Segments:**
            * **Chapter (2 digits):** Choose the most appropriate chapter.
            * **Heading (next 2 digits):** Justify based on text and legal notes.
            * **Subheading (next 2 digits):** Justify based on text and legal notes (forms 6-digit international code).
            * **National Tariff Line/Statistical Suffixes:** Determine final applicable digits *VERBATIM* from `OFFICIAL CHAPTER CONTENT`.
        3.  **MANDATORY FINAL VERIFICATION:** Before outputting, re-confirm that:
            * No proposed code is in `Previously Rejected HS Codes`.
            * No proposed code belongs to an `Excluded Chapter`.
            * All proposed codes (including suffixes) exist *VERBATIM* in `OFFICIAL CHAPTER CONTENT`.
            * Specific statistical suffixes were prioritized over '.00' where applicable.
            * Knit/woven rules were strictly followed.
            * The sum of likelihood for the 3 proposed codes should be 100%.

        **MANDATORY FINAL VERIFICATION (DOUBLE-CHECK BEFORE OUTPUT):**
        Before generating your final output, perform these absolute checks on EACH proposed OPTION:
        1.  **HS Code Length:** Confirm the HS code precisely matches the required length for {country.upper()} (e.g., 10 digits for Canada/USA, 8 digits + "-000" for Switzerland).
        2.  **Knit/Woven Integrity:** Re-verify that knitted/crocheted products are EXCLUSIVELY Chapter 61 and woven products are EXCLUSIVELY Chapter 62.
        3.  **Pima Cotton/Predominance:** If cotton (especially Pima Cotton) is the highest percentage material by weight, the code MUST be from the 'Of cotton' subheading.
        4.  **No Rejected Codes:** ABSOLUTELY ENSURE that none of the proposed codes are present in `Previously Rejected HS Codes for this Item`. If a proposed code is on that list, you MUST find a valid alternative.
        5.  **Final Reasoning Integrity Check:** Before finalizing your response for any option, you MUST re-read your own generated 'REASONING STRUCTURE'. If any part of your reasoning (especially the material classification) contradicts one of the `HARDCODED CLASSIFICATION RULES` (e.g., claiming polyester is not a man-made fiber), that entire option is **INVALID**. You must discard it and find a new one. Do not output options that contain logical errors in their justification.
        
        ---

        **FORMAT (Strictly Adhere):**

        ### OPTION 1: [HS code + suffix] - XX% certainty

        #### PRODUCT DESCRIPTION:
        [Re-state the product description based *solely* on the information received, focusing on classification-relevant details.]

        #### REASONING STRUCTURE:
        Use this detailed structure for justification:

        1.  *GRI Application*: Identify and apply ALL relevant General Rules of Interpretation (GRI 1-6, in order). Explain how each applied GRI leads to the decision.
        2.  *Historical Data Consideration*: Explicitly state how historical data influenced (or didn't influence) this option. If used, explain relevance and verification. If not used, explain why.
        3.  *Chapter & Section Fit*: State chosen Chapter/Section (e.g., Chapter 61: Knitted/crocheted apparel, Section XI). Confirm inclusion based on material, construction, use, and notes. Confirm chapter is NOT excluded. *(IMPORTANT: Only consider chapters provided above.)*
        4.  *Heading & Subheading Determination*: Justify 4-digit heading and 6-digit subheading using texts and legal notes.
        5.  *National Tariff Line Determination*: Explain how specific suffixes/national digits were determined *VERBATIM* from `OFFICIAL CHAPTER CONTENT`. 
        6.  *Exclusions & Verifications*: Explicitly rule out other plausible but incorrect classifications.
        7.  ***Internal Verification Check***: **Before finalizing this option, state the outcome of the following checks:**

            * **Check 1 (Knit/Woven):** Is the product knitted/woven, and does the code chapter (61/62) match? State Yes/No and confirm.
            * **Check 2 (Code in Docs):** Does this exact HS code `[HS code + suffix]` exist verbatim in the provided `OFFICIAL CHAPTER CONTENT`? **This is a non-negotiable, mandatory check. If the exact code is NOT found, this entire option is INVALID and you MUST discard it. If it IS found, you MUST quote the exact line item from the document that contains the code.**
            * **Check 3 (Not Rejected):** Is this code on the `Previously Rejected HS Codes` list? State Yes/No and confirm it is not.
            * **Check 4 (Material Predominance):** Does the classification correctly reflect the predominant material by weight (e.g., classifying as 'Of cotton' if cotton > 50%)? State Yes/No and confirm.

        #### LEGAL BASIS:
        Cite specific GRI rules, Section/Chapter notes, and heading/subheading texts from provided documents. Quote directly or paraphrase precisely. Include `Classification Guide` cross-references if applicable.

        ---

        **HARDCODED CLASSIFICATION RULES (Apply Rigorously - These override general interpretations if applicable):**

        * **No "Same as OPTION X":** Each option requires full, independent reasoning.
        * If the relevant documentation shows hierarchy of codes to be used with hifens *-* but there is no value for that level, look for the following levels of hierarchy.
        * **Knitted vs. Woven Exclusions:** If knitted/crocheted, ALWAYS exclude non-knitted headings (e.g., Chapter 62). If woven, ALWAYS exclude knitted headings (e.g., Chapter 61).
        * **Material:** *Polyester* is a *man-made fiber*. So, a product whose composition is mostly Polyester will always fall within *man-made fiber*.
        * **Material:** *Pima Cotton* is a type of *cotton* and should ALWAYS be classified as such.
        * The DEFINING material for classification is the one with the **HIGHEST PERCENTAGE** when a product is composed of multiple materials.
        * **Chapter Definitions:**
            * Chapter 61: *Articles of Apparel and Clothing Accessories, Knitted or Crocheted*.
            * Chapter 62: *Articles of Apparel and Clothing Accessories, Not Knitted or Crocheted*.
        * **HS Code Structure:**
            * Apply `GRI` from `{gri}`.
            * Use chapter-specific `LEGAL NOTES` from `{legal_notes}`.
            * DO NOT add suffixes/sub-codes unless *VERBATIM* present in `OFFICIAL CHAPTER CONTENT`.
            * Use the *longest HS Code version explicitly shown* in `OFFICIAL CHAPTER CONTENT`.
            * Full code for `{country.upper()}` must include *all visible digits* (e.g., 6103.43.00.15 includes .00; 6109.9000 includes 000). Do not skip intermediate levels.
            * After 4 digits, every pair of digits is separated by `.` (e.g., 6109.10.20, NOT 61091020).
            * NEVER use spaces within HS Code digits.
            * **Switzerland Specific:** For Switzerland, provide an 8-digit HS Code. Add `*-000*` as the statistical suffix ONLY IF the 8-digit code exists in `OFFICIAL CHAPTER CONTENT` and no more specific 11-digit Swiss suffix is listed/applicable.
        * **Product-Specific Overrides:**
            * `6204.43.00.90` does NOT exist on Canada's Tariff Schedule. DO NOT use for Canada.
            * *UNISEX* products should be classified as *WOMEN*.
            * Knitted *Tank Top* = T-shirt.
            * Unpadded *VEST* = `62.11.33`.
            * *Crop Top* is NEVER a t-shirt (except China).
            * *Crop Top* is ALWAYS *Other Garments* unless specific text dictates otherwise.
            * *Crop Top* is NEVER `61.04` (unless overwhelming country-specific legal notes).
            * *Short Tights* = *shorts*.
            * *Pullover* = *Other garments*, not a shirt.
            * For WOMEN, knitted t-shirt (incl. "Tank Top") = `61.09`. Generally NOT `6106` (shirt/blouse).
            * *Race Singlet* is NEVER `62.11`. If woven, typically `62.07`.
            * *TIGHTS* = in this context, tights ARE ALWAYS tight-fitting stretchy trousers, similar to leggings, regardless of being KNITTED or WOVEN.
            * *TIGHTS SHORT* = tight-fitting stretch shorts.
            * Ignore *color* or *identifier terms* (SKU, style names) unless directly affecting material, primary use, or construction relevant to tariff distinctions.
            * For man-made materials (e.g., polyester, polyamide), it is NOT "other textile materials" unless a blend where man-made fiber isn't predominant or specific blend notes apply.
            * Pay VERY SPECIAL ATTENTION to product type (men/women/infant garment, garment type) to avoid suggesting completely incorrect codes (e.g., sock as trouser).

        ---

        In the country specific guidelines (`{guidelines}`), you are provided an example classification that follows an ideal structure. Model your response structure after this example.

        **FINAL REMINDERS:**
        - Do not propose any code segment that does not exist *verbatim* in the official documentation. If a classification path dead-ends, indicate that no valid code can be found.
        - Your primary directive is accuracy based *only* on the provided documents and rules.
        - Double-check all suggestions: 
             -- confirm verbatim existence in Tariff Schedule 
             -- verify that the code is NOT present in the *Previously Rejected HS Codes*.
        """

    try:
        # Define safety settings to be less restrictive for this low-risk task
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        # Define generation configuration
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.5,
            "top_k": 15,
            "max_output_tokens": 8192,
        }
        
        # --- MODIFICATION: Add timeout to the API call ---
        response = await asyncio.wait_for(
            model.generate_content_async(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings  # Apply the safety settings
            ),
            timeout=120.0 # Timeout after 2 minutes
        )
        print(response)
        
        input_tokens = 0
        output_tokens = 0
        # gets input tokens
        try:
            input_tokens = model.count_tokens(prompt).total_tokens
        except Exception as e:
            print(f"Warning: Could not count input tokens. {e}")

        # gets output tokens from response metadata
        if response and hasattr(response, 'usage_metadata'):
            output_tokens = response.usage_metadata.candidates_token_count
        
        # logs the usage
        log_token_usage(country, product_type, input_tokens, output_tokens)

        # checks if the response was blocked
        if not response.parts:
            print(f"--- RESPONSE BLOCKED FOR PRODUCT: {product_description} ---")
            if response.prompt_feedback:
                 print(f"    Finish Reason: {response.prompt_feedback.block_reason}")
                 print(f"    Safety Ratings: {response.prompt_feedback.safety_ratings}")
            print("----------------------------------------------------")
            return "" # returns an empty string if blocked, preventing errors downstream

        token_count = model.count_tokens(prompt).total_tokens
        print(f"The prompt contains {token_count} tokens.")

        # Ensure we always return a string
        validated_response = response.text or ""
        return validated_response
    
    # --- MODIFICATION: Add specific handling for TimeoutError ---
    except asyncio.TimeoutError:
        print(f"An API call timed out for product: {product_description}")
        # Return a structured error message that can be parsed without error
        return """
        ### OPTION 1: TIMEOUT - 0% certainty
        #### REASONING STRUCTURE:
        The request to the generative AI service timed out after 120 seconds. This might be due to high server load.
        """
    except Exception as e:
        import traceback
        print(f"An exception occurred in generate_hs_codes for product '{product_description}': {str(e)}")
        traceback.print_exc()
        # Return a structured error message that can be parsed without error
        return """
        ### OPTION 1: EXCEPTION - 0% certainty
        #### REASONING STRUCTURE:
        An unexpected error occurred during API call. Please check the console logs.
        """


# finding relevant chapters based on product description
def find_relevant_chapters(product_description, country, country_specific_pdf_data):
    """Finds relevant chapters based on keywords in the product description.
    Args:
        product_description: The description of the product to classify.
        country: The target country for the classification.
        country_specific_pdf_data: The country-specific PDF data containing chapter texts.
    Returns:
        A list of tuples containing relevant chapter numbers and their text content.
    """
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


def save_rejected_code(product, country, code):
    """Saves a rejected HS code to a JSON file.
    Args:
        product: The product description.
        country: The target country for the classification.
        code: The rejected HS code.
    """
    
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
                    print(
                        f"Warning: Could not decode JSON from {REJECTED_CODES_FILE}."
                        " Starting with an empty list."
                    )
                    data = []
                except Exception as e:
                    print(
                        f"An unexpected error occurred while reading"
                        f" {REJECTED_CODES_FILE}: {e}"
                    )
                    raise
                else:
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
        #print("Code saved successfully.")
    except Exception as e:
        print(f"Error saving code: {e}")


def load_rejected_codes(product_description, country):
    """Loads rejected HS codes for a given product description and country.
    Returns a list of matching entries.
    Args:
        product_description: The product description.
        country: The target country for the classification.
    Returns:
        A list of rejected code entries matching the product description and country.
    """
    # print(f"Loading REJECTED_CODES_FILE: {REJECTED_CODES_FILE}")
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


def save_validated_code(product_description, country, hs_code, reasoning=None):
    """
    Saves a validated HS code and its reasoning (optional) to a JSON file.
    Args:
        product_description: The product description.
        country: The target country for the classification.
        hs_code: The validated HS code.
        reasoning: The reasoning behind the validation (optional).
    """
    entry = {
        "product_description": product_description,
        "country": country,
        "hs_code": hs_code,
        "reasoning": reasoning, # Can be None if only code is validated
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        if not os.path.exists(VALIDATED_CODES_FILE):
            with open(VALIDATED_CODES_FILE, "w") as f:
                json.dump([entry], f, indent=2)
        else:
            with open(VALIDATED_CODES_FILE, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {VALIDATED_CODES_FILE}."
                        " Starting with an empty list."
                    )
                    data = []
                except Exception as e:
                    print(
                        f"An unexpected error occurred while reading"
                        f" {VALIDATED_CODES_FILE}: {e}"
                    )
                    raise
                else:
                    # Check if this exact product-country-hs_code combination already exists
                    # This prevents duplicate entries for the same validated code
                    existing_entry = next((item for item in data if
                                           item.get("product_description") == product_description and
                                           item.get("country") == country and
                                           item.get("hs_code") == hs_code), None)
                    if existing_entry:
                        print(f"DEBUG: Code '{hs_code}' for '{product_description}' in {country} already exists in validated codes. Updating timestamp.")
                        existing_entry['timestamp'] = entry['timestamp']
                        if reasoning is not None: # Only update reasoning if provided
                            existing_entry['reasoning'] = reasoning
                    else:
                        data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
        print(f"DEBUG: Validated code '{hs_code}' for '{product_description}' in {country} saved successfully.")
    except Exception as e:
        print(f"ERROR: Error saving validated code: {e}")


def load_validated_codes(product_description, country):
    """
    Loads validated HS codes for a given product description and country.
    Returns a list of matching entries.
    Args:
        product_description: The product description.
        country: The target country for the classification.
    """
    if not os.path.exists(VALIDATED_CODES_FILE):
        return []

    try:
        with open(VALIDATED_CODES_FILE, "r") as f:
            all_validated = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    product_specific_validated = [
        entry for entry in all_validated
        if entry.get("product_description") == product_description and entry.get("country") == country
    ]
    return product_specific_validated


def parse_reasoning_text(raw_reasoning_text):
    """Parses the raw reasoning text into a structured dictionary format.
    Args:
        raw_reasoning_text: The raw reasoning text to parse.
    Returns:
        A dictionary with structured reasoning sections.
    """
    structured_reasoning = {}

    # Define the sections and their regex patterns
    sections = [
        ("GRI Application", r"1\.?\s*\*GRI Application\*:\s*(.*?)(?=2\.?\s*\*Historical Data Consideration\*|3\.?\s*\*Chapter & Section Fit\*|4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Historical Data Consideration", r"2\.?\s*\*Historical Data Consideration\*:\s*(.*?)(?=3\.?\s*\*Chapter & Section Fit\*|4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Chapter & Section Fit", r"3\.?\s*\*Chapter & Section Fit\*:\s*(.*?)(?=4\.?\s*\*Heading & Subheading Determination\*|5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Heading & Subheading Determination", r"4\.?\s*\*Heading & Subheading Determination\*:\s*(.*?)(?=5\.?\s*\*National Tariff Line Determination\*|6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("National Tariff Line Determination", r"5\.?\s*\*National Tariff Line Determination\*:\s*(.*?)(?=6\.?\s*\*Exclusions & Verifications\*|$)"),
        ("Exclusions & Verifications", r"6\.?\s*\*Exclusions & Verifications\*:\s*(.*?)(?=$)")
    ]

    # Attempt to extract each section
    for section_name, pattern in sections:
        match = re.search(pattern, raw_reasoning_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Clean up markdown asterisks and leading/trailing whitespace
            content = re.sub(r'\*\s*', '', content).strip()
            content = re.sub(r'\s*\*\s*', ' ', content).strip() # Replace single asterisks with space
            structured_reasoning[section_name] = content
        else:
            structured_reasoning[section_name] = "Not explicitly addressed in response."

    # Handle the case where there's a simple "NO-CODE-FOUND"
    if "NO-CODE-FOUND" in raw_reasoning_text and not structured_reasoning:
        structured_reasoning["General Reason"] = raw_reasoning_text.replace("#### REASONING:\n\n", "").strip()

    return structured_reasoning



def extract_hs_codes(text):
    """Extracts HS codes, certainty levels, and reasoning from the model's response text.
    1. Extracts the product description.
    2. Extracts each option's HS code, certainty, and reasoning.
    Args:
        text: The model's response text.
    Returns:
        A pandas DataFrame with the extracted information."""
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
        # Extract the main reasoning block
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
                country="", # Country is not available here, will be added in app.py
                product_type="", # Product type also not available here
                event_type="no_code_found",
                hs_code=code
            )

        option_num = i + 1

    df = pd.DataFrame([row])
    return df


def process_bulk_data(
    df_input,
    model,
    pdf_data_cache,
    country_col,
    name_col,
    name_col2,
    material_col,
    construction_col,
    gender_col
):
    """Processes bulk data for HS code classification.
    Args:
        df_input: DataFrame containing input data.
        model: The language model to use for classification.
        pdf_data_cache: Cached PDF data for countries.
        country_col: Column name for country.
        name_col: Column name for product name.
        name_col2: Optional second column name for product name.
        material_col: Column name for material.
        construction_col: Column name for construction.
        gender_col: Column name for gender.
    Returns:
        DataFrame with classification results."""
    # This function seems to be for a Streamlit app and is not used in the Flask app.
    # I will keep it as is, assuming it's for a separate Streamlit component or future use.
    all_results_list = []
    total_rows = len(df_input)
    start_time = time.time()

    # Define the delay here, or use a global one if you uncommented it above

    for df_idx, row in df_input.iterrows():
        original_index = row["original_index"]

        #___Build Product Description____#
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
        product_description = " ".join(desc_parts).strip()

        base_result_row = {
            "original_index": original_index,
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

        processed_pdfs_for_current_country = pdf_data_cache[country]

        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs_for_current_country)

        legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
        classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
        gri = processed_pdfs_for_current_country.get("gri", "")


        country_texts = load_text_files_for_country("/Users/joana.duarte/Desktop/bydo/flask_app/chapter_data", country)
        guidelines = country_texts.get(f"{country}_guidelines", "")

        try:
            rejected_codes_snapshot = load_rejected_codes(product_description, country)

            generated_response = generate_hs_codes(
                model,
                product_description,
                country,
                relevant_chapters, # Pass the list of (chapter_num, chapter_text) tuples
                legal_notes,
                classification_guide,
                gri=gri,
                guidelines=guidelines,
                rejected_codes_snapshot = rejected_codes_snapshot
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
                    else:     # clears if not present in new result
                        base_result_row[hs_col] = ""
                        base_result_row[cert_col] = 0
                        base_result_row[reas_col] = ""

            else:
                # Use print for Flask as st.error is Streamlit-specific
                print(f"ERROR: Failed to extract codes for row {df_idx + 1} (Original Index: {original_index}). Response format might be unexpected.")
                base_result_row["reasoning_1"] = "Error: Failed to parse response"

            all_results_list.append(base_result_row)

        except Exception as gen_e:
            print(f"ERROR: Error generating code for row {df_idx + 1} (Original Index: {original_index}): {gen_e}")
            base_result_row["hs_code_1"] = "ERROR"
            base_result_row["reasoning_1"] = str(gen_e)
            all_results_list.append(base_result_row)

        print(f"Waiting for {REQUEST_DELAY_SECONDS} seconds before the next request...")
        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(all_results_list)



def save_processing_time(processing_time, num_rows, filename="processing_times.csv"):
    """Saves processing time and number of rows processed to a CSV file.
    Args:
        processing_time: Total processing time in seconds.
        num_rows: Number of rows processed.
        filename: Name of the CSV file to save the data.
    """
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'rows_processed', 'total_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'rows_processed': num_rows,
            'total_time_seconds': round(processing_time, 2)
        })


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
    """Formats historical data from a CSV file and finds similar products.
    Args:
        csv_file_path: Path to the CSV file containing historical data.
        target_country: The target country to filter historical data.
        target_full_product_description: The full product description to compare against.
        target_gender: The target gender to filter historical data (optional).
        country_col_hist: Column name for country in historical data.
        name_col_hist: Column name for product name in historical data.
        name_col2_hist: Optional second column name for product name in historical data.
        material_col_hist: Column name for material in historical data.
        construction_col_hist: Column name for construction in historical data.
        gender_col_hist: Column name for gender in historical data.
        hs_code_col_hist: Column name for HS code in historical data.
        similarity_threshold: Cosine similarity threshold to consider a product as similar.
        top_n: Number of top similar products to return.
        return_df: If True, returns the filtered DataFrame instead of formatted string.
    Returns:
        A formatted string of similar historical products or a DataFrame if return_df is True.
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

    # Ensure country column is string and filter
    historical_df[country_col_hist] = historical_df[country_col_hist].astype(str)
    historical_df = historical_df[historical_df[country_col_hist].str.lower() == target_country.lower()]

    if historical_df.empty:
        return (
            f"No historical data found for {target_country.upper()}."
        )

    # --- NEW: Filter by target_gender if provided ---
    if target_gender:
        # Convert historical gender column to string and lowercase for robust comparison
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
        """Builds a full product description from relevant columns.
        """
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
        hs_code = str(row[hs_code_col_hist]) if hs_code_col_hist in row and pd.notna(row[hs_code_col_hist]) else "N/A" # Corrected typo: hs_code_col instead of hs_code_col_hist
        similarity = row['similarity']
        formatted_data += (
            f"- Similar Product (Similarity: {similarity:.2f}): "
            f"{full_product_details_output}, HS Code: {hs_code}\n"
        )

    return formatted_data


def extract_simplified_hs_codes(text):
    """Extracts simplified HS codes and certainty levels from the model's response text.
    Args:
        text: The model's response text.
    Returns:
        A pandas DataFrame with the extracted HS codes and certainty levels."""
    options_data = []
    # pattern to capture HS code and certainty from "### OPTION X: [HS code] - YY% certainty"
    options = re.findall(r'### OPTION \d+: ([0-9]+(?:\.[0-9]+)*(?:\s+[0-9]+)?) - (\d+)% certainty', text)

    for i, (code, certainty) in enumerate(options):
        options_data.append({
            f'Option {i+1} HS Code': code.strip(),
            f'Option {i+1} Certainty': int(certainty)
        })
    return pd.DataFrame(options_data)


def save_bulk_classification_results(df_to_save: pd.DataFrame, filename=METRICS_RESULTS_FILE):
    """
    Saves or appends the raw bulk classification results (model's suggestions)
    DataFrame to a CSV file.
    Args:
        df_to_save: DataFrame containing the classification results to save.
        filename: Name of the CSV file to save the data.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_with_time = df_to_save.copy()

    # Ensure all columns from the Streamlit raw_df.csv are present and ordered
    # And add timestamp
    raw_output_cols = [
        "idx", "product_description", "input_product", "input_material",
        "input_construction", "input_gender", "input_country",
        "hs_code_1", "certainty_1", "reasoning_1",
        "hs_code_2", "certainty_2", "reasoning_2",
        "hs_code_3", "certainty_3", "reasoning_3",
        "hs_code_from_csv" # This comes from the original input_df_for_mapping in Streamlit
    ]

    # Add any missing columns with NaN if they weren't generated (e.g., if you only output hs_code_1)
    for col in raw_output_cols:
        if col not in df_with_time.columns:
            df_with_time[col] = pd.NA # Or None, depends on desired type for empty values

    df_with_time['timestamp'] = timestamp # Add timestamp

    # Reorder columns to match Streamlit's raw_df if desired for strict format
    # Note: hs_code_from_csv would ideally be merged into the results_df in app.py
    # before saving, to accurately reflect the Streamlit process where it's added.
    # For now, we'll assume it exists if it's coming from input.
    final_cols_for_raw_save = raw_output_cols + ['timestamp']
    df_to_write = df_with_time.reindex(columns=final_cols_for_raw_save)


    file_path = pathlib.Path(filename)

    print(f"DEBUG: save_bulk_classification_results called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{file_path.resolve()}'")
    print(f"DEBUG: Does file exist before writing? {file_path.exists()}")
    print(f"DEBUG: DataFrame to save shape: {df_to_write.shape}")
    print(f"DEBUG: DataFrame columns: {df_to_write.columns.tolist()}") # Check actual columns

    try:
        if file_path.exists():
            df_to_write.to_csv(file_path, mode='a', header=False, index=False)
            print(f"DEBUG: Successfully appended results to {filename}.")
        else:
            df_to_write.to_csv(file_path, mode='w', header=True, index=False)
            print(f"DEBUG: Successfully created and saved results to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save/append results to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()


def save_final_selected_results(df_to_save: pd.DataFrame, filename=FINAL_SELECTED_FILE):
    """
    Saves or appends the user's final selected classification results DataFrame to a CSV file.
    Args:
        df_to_save: DataFrame containing the final selected classification results to save.
        filename: Name of the CSV file to save the data.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_with_time = df_to_save.copy()
    df_with_time["timestamp"] = timestamp # Add the timestamp to the DataFrame

    file_path = pathlib.Path(filename)

    print(f"DEBUG: save_final_selected_results called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{file_path.resolve()}'")
    print(f"DEBUG: Does file exist before writing? {file_path.exists()}")
    print(f"DEBUG: DataFrame to save shape: {df_to_save.shape}")
    print(f"DEBUG: DataFrame columns: {df_to_save.columns.tolist()}")

    try:
        if file_path.exists():
            df_with_time.to_csv(file_path, mode='a', header=False, index=False)
            print(f"DEBUG: Successfully appended final selected results to {filename}.")
        else:
            df_with_time.to_csv(file_path, mode='w', header=True, index=False)
            print(f"DEBUG: Successfully created and saved final selected results to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save/append final selected results to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()
        
        
def log_interaction_event(country, product_type, event_type, hs_code=None, details=""):
    """Logs user interactions, regenerations, and other key events to a CSV file.
    Args:
        country: The target country for the classification.
        product_type: The product type or description.
        event_type: The type of event (e.g., 'selection', 'rejection', 'regeneration', 'no_code_found').
        hs_code: The HS code involved in the event (if applicable).
        details: Additional details about the event (e.g., which option was selected).
    """
    file_exists = os.path.exists(INTERACTION_LOG_FILE)
    try:
        with open(INTERACTION_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'country', 'product_type', 'event_type', 'hs_code', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'country': country,
                'product_type': product_type,
                'event_type': event_type, # e.g., 'selection', 'rejection', 'regeneration', 'no_code_found'
                'hs_code': hs_code,
                'details': details # e.g., 'option_1', 'option_2'
            })
    except Exception as e:
        print(f"ERROR: Could not write to interaction log file. Reason: {e}")
        

def save_processing_time(processing_time, num_rows, filename=PROCESSING_TIMES_FILE): #
    """
    Saves the processing time for a bulk classification run to a CSV file.
    Args:
        processing_time: Total processing time in seconds.
        num_rows: Number of rows processed.
        filename: Name of the CSV file to save the data.
    """
    file_exists = os.path.exists(filename) #
    
    print(f"DEBUG: save_processing_time called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{pathlib.Path(filename).resolve()}'")
    print(f"DEBUG: Does processing_times file exist before writing? {file_exists}")

    try:
        with open(filename, 'a', newline='') as csvfile: #
            fieldnames = ['timestamp', 'rows_processed', 'total_time_seconds'] #
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) #
            
            if not file_exists: #
                writer.writeheader() #
                
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), #
                'rows_processed': num_rows, #
                'total_time_seconds': round(processing_time, 2) #
            })
        print(f"DEBUG: Successfully saved processing time to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save processing time to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()