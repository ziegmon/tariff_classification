"""
Helper Functions Module (Footwear).
This module contains helper functions for file reading, document loading,
Gemini API configuration, and HS code generation specifically for footwear products.
"""

import os
import json
import google.generativeai as genai
import pandas as pd
import PyPDF2
from pathlib import Path
import base64
import re
import time
from google.api_core import exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import threading

#___Documentation Path___#
PDF_DIRECTORY = "chapter_data"
CSV_PATH = "train_fold_0.csv"

#___Variables for your footwear data structure___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "material_composition"
construction_col = "outsole_material"
gender_col = "gender_name"
hs_code_col = "tariff_code"

#___Read PDF___#    
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file located at pdf_path.
    Returns the extracted text as a single string.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file, strict=False)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

#___Load All PDF Data___#
def load_all_pdf_data(pdf_directory=PDF_DIRECTORY):
    """Loads and caches all PDF data for all countries from the specified directory.
    The PDFs are expected to follow the naming convention 'country_doctype.pdf'.
    Args:
        pdf_directory (str): The path to the directory containing the PDF files.
    Returns:
        dict: A nested dictionary with structure {country: {doc_type: content, ...}, ...}"""
    pdf_cache = {}
    country_set = set()

    if not os.path.exists(pdf_directory):
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
                elif '_gri' in filename:
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
 
    return pdf_cache

#___Load Text Files___#
def load_text_files_for_country(text_directory, country, file_suffix=".txt"):
    """Loads and caches text files for a specific country from the specified directory.
    The text files are expected to start with the country name and end with the specified suffix.
    Args:
        text_directory (str): The path to the directory containing the text files.
        country (str): The country name to filter text files.
        file_suffix (str): The suffix that text files should end with (default is ".txt").
    Returns:
        dict: A dictionary with structure {filename: content, ...} for the specified country
    """
    processed_texts = {}

    if not os.path.exists(text_directory):
        print(f"Directory {text_directory} does not exist!")
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
            print(f"Error reading file {file_path}: {str(e)}")

    return processed_texts

#___Gemini API Config___#
def configure_genai(api_key):
    """Configures the Google Generative AI (Gemini) API with the provided API key.
    Args:
        api_key (str): The API key for authenticating with the Gemini API.
    Returns:
        generative_model: An instance of the configured GenerativeModel."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
    return model

#___Historical Data Integration___#
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
    size_col_hist="size_code",
    hs_code_col_hist="tariff_code",
    similarity_threshold=0.85,
    top_n=5,
    return_df=False
):
    """Formats historical footwear data from a CSV file for similarity matching.
    Args:
        csv_file_path (str): Path to the CSV file containing historical data.
        target_country (str): The target country for filtering historical data.
        target_full_product_description (str): The full product description to match against.
        target_gender (str, optional): The target gender for filtering historical data.
        country_col_hist (str): Column name for country in historical data.
        name_col_hist (str): Column name for product name in historical data.
        name_col2_hist (str): Column name for secondary product name in historical data.
        material_col_hist (str): Column name for material in historical data.
        construction_col_hist (str): Column name for construction in historical data.
        gender_col_hist (str): Column name for gender in historical data.
        size_col_hist (str): Column name for size in historical data.
        hs_code_col_hist (str): Column name for HS code in historical data.
        similarity_threshold (float): Cosine similarity threshold for matching.
        top_n (int): Number of top similar matches to return.
        return_df (bool): If True, returns the filtered DataFrame instead of formatted string.
    Returns:
        str or pd.DataFrame: Formatted historical data string or DataFrame if return_df is True.
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
        """Builds a full product description from relevant columns."""
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
        if size_col_hist in row and pd.notna(row[size_col_hist]):
            parts.append(str(row[size_col_hist]))
        return " ".join(parts).strip()

    historical_df['full_description'] = historical_df.apply(build_description, axis=1)

    if return_df:
        return historical_df

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
        if pd.notna(row.get(size_col_hist)):
            display_parts.append(f"Size: {row[size_col_hist]}")

        full_product_details_output = ", ".join(display_parts)
        hs_code = str(row[hs_code_col_hist]) if hs_code_col_hist in row and pd.notna(row[hs_code_col_hist]) else "N/A"
        similarity = row['similarity']
        formatted_data += (
            f"- Similar Product (Similarity: {similarity:.2f}): "
            f"{full_product_details_output}, HS Code: {hs_code}\n"
        )

    return formatted_data

#___Enhanced Generate HS Codes___#
def generate_hs_codes_ftw(
    model,
    product_description,
    country,
    relevant_chapters,
    gri=""
):
    """Generates HS code suggestions for footwear products using the Gemini API.
    Args:
        model: The configured Gemini generative model.
        product_description (str): The full product description for classification.
        country (str): The target country for classification.
        relevant_chapters (list): List of tuples containing relevant chapter numbers and their content.
        gri (str, optional): Country-specific General Rules of Interpretation. Defaults to "".
    Returns:
        str: The generated HS code suggestions and reasoning.
    """
    max_retries = 1
    retry_delay = 5  # seconds
    print(f"[DEBUG] Product Description: {product_description}")

    # Get historical data
    historical_data_string = format_historical_data_from_csv(
        target_country=country,
        target_full_product_description=product_description,
        csv_file_path=CSV_PATH
    )

    for attempt in range(max_retries):
        prompt = f"""
            **CONTEXT & RESOURCES:**
            - **Product Description:** {product_description}
            - **Target Country:** {country.upper()}
            - **General Rules of Interpretation (GRI):** {gri}
            - **OFFICIAL CHAPTER CONTENT:** (Provided below)

            **HISTORICAL DATA (SIMILAR FOOTWEAR PRODUCTS & CLASSIFICATIONS):**
            {historical_data_string}

            ---

            **ROLE:** You are an expert customs classifier specializing in FOOTWEAR, with expertise in multiple countries, tasked with accurately classifying footwear products using only the provided official Harmonized System (HS) documents. Precision is paramount.

            For the most part the PDF contains a table with two columns: 'Tariff Item' and 'Description of Goods'. The Tariff Item is a numerical code, and the Description of Goods is the corresponding product description.
            
            HS Codes are hierarchical and the provided PDF includes different hierarchical levels. Consider always the most granular/longest code of the hierarchy.

            **ULTRA-CRITICAL FOOTWEAR CLASSIFICATION RULES (STRICT COMPLIANCE REQUIRED):**

            1. **ABSOLUTE CODE VALIDITY:** ONLY propose HS codes that appear *VERBATIM* in the official documentation.
            2. **SOLE MATERIAL PRIORITY:** Classification depends heavily on outer sole material (rubber, leather, textile, etc.)
            3. **UPPER MATERIAL CONSIDERATION:** Upper material affects subheading classification
            4. **SPORTS vs REGULAR FOOTWEAR:** Athletic/sports footwear has specific subheadings (e.g., 6404.11)
            5. **GENDER CLASSIFICATION:** Many tariff lines distinguish between men's/boys' and women's/girls' footwear
            6. **STATISTICAL SUFFIX PRIORITY:** Use the most specific, applicable statistical suffixes
            7. **DIGIT-LENGTH ENFORCEMENT:** The proposed HS code **must** exactly match the digit length required for the target country. If it does not, **reject and do not propose**.
            8. **SIZE CODE:** A size code of -1 means that size is not relevant for this country and thus can be excluded from reasoning.

            **COUNTRY-SPECIFIC CODE LENGTHS (STRICT):**
            - **Switzerland:** all HS codes **must** be 11 digits and end with "-000" (e.g., 64041100-000) even if documentation is only 8 digits
            - **Europe, Canada, Australia, United States:** EXACTLY 10 digits
            - **Japan:** EXACTLY 9 digits
            - **Brazil, Norway:** EXACTLY 8 digits
            - **South Korea:** EXACTLY 10 digits

            **TASK:**
            Based *exclusively* on the provided content, determine the *THREE most likely HS codes* for the footwear product. Always propose exactly three distinct HS codes that best match, even if no perfect fit—assign low certainty if needed and explain limitations.

            **FORMAT (Strictly Adhere):**

            ### OPTION 1: [HS code] - XX% certainty

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

            ### OPTION 2: [HS code] - XX% certainty
            [Same structure as Option 1]

            ### OPTION 3: [HS code] - XX% certainty
            [Same structure as Option 1]

            **FINAL VERIFICATION:** Before outputting, confirm:
            - Ensure exactly three different codes are outputted, no exceptions.
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
                "top_p": 0.3,
                "top_k": 5,
                "max_output_tokens": 10000,
            }

            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            token_count = model.count_tokens(prompt).total_tokens
            print(f"The prompt contains {token_count} tokens.")

            return response.text
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                print(f"Rate limit hit, retrying in {retry_delay} seconds... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"Error: API rate limit exceeded after {max_retries} retries: {str(e)}"
        except Exception as e:
            return f"Error generating HS codes: {str(e)}"

#___Find Relevant Chapters for Footwear___#
def find_relevant_chapters(product_description, country, country_specific_pdf_data):
    """Finds relevant chapters for footwear products based on keywords in the product description.
    Args:
        product_description (str): The full product description for classification.
        country (str): The target country for classification.
        country_specific_pdf_data (dict): Cached PDF data for the specific country.
    Returns:
        list: List of tuples containing relevant chapter numbers and their content."""
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

#___Extract HS Codes from Response___#
def extract_hs_codes(text):
    """Extracts HS codes, certainty percentages, product description, and reasoning from the model's response text.
    Args:
        text (str): The model's response text containing HS code suggestions.
    Returns:
        pd.DataFrame: A DataFrame with columns for product description, HS codes, certainty, and reasoning."""
    # Extract product description
    product_desc_match = re.search(r'#### PRODUCT DESCRIPTION:\s*(.*?)(?=####|$)', text, re.DOTALL)
    product_description = product_desc_match.group(1).strip() if product_desc_match else "Not found"
    
    # Extract options
    options = re.findall(r'### OPTION \d+:\s*([\d][\d.\s-]{3,15})\s*[-–]\s*(\d+)% certainty\s*(.*?)(?=### OPTION \d+:|$)', 
                     text, re.DOTALL)
    print("DEBUG: Regex matches ->", options)
    
    row = {'product_description': product_description}
    
    for i, (code, certainty, details) in enumerate(options):
        reasoning_match = re.search(r'#### REASONING(?:\s+STRUCTURE)?:(.*?)(?=#### LEGAL BASIS:|$)', details, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"
        
        option_num = i + 1
        row[f'hs_code_{option_num}'] = code.strip()
        row[f'certainty_{option_num}'] = int(certainty)
        row[f'reasoning_{option_num}'] = reasoning
    
    return pd.DataFrame([row])
#___Bulk Processing Function___#
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
    size_col
):
    """Processes bulk footwear data to generate HS code suggestions using the Gemini API.
    Args:
        df_input (pd.DataFrame): DataFrame containing footwear data to process.
        model: The configured Gemini generative model.
        pdf_data_cache (dict): Cached PDF data for countries.
        country_col (str): Column name for country in input data.
        name_col (str): Column name for product name in input data.
        name_col2 (str): Column name for secondary product name in input data.
        material_col (str): Column name for material in input data.
        construction_col (str): Column name for construction in input data.
        gender_col (str): Column name for gender in input data.
        size_col (str): Column name for size in input data.
    Returns:
        pd.DataFrame: DataFrame containing original data with added HS code suggestions and reasoning.
    """
    all_results_list = []
    total_rows = len(df_input)
    class DummyProgress:
        """A dummy progress bar class for demonstration purposes."""
        def progress(self, value):
            """Updates the progress bar to the specified value.
            Args:
                value (float): A float between 0 and 1 indicating progress.
            """
            print(f"Progress: {value*100:.2f}%")

    class DummyText:
        """A dummy text display class for demonstration purposes."""
        def text(self, msg):
            """Displays a text message.
            Args:
                msg (str): The message to display.
            """
            print(msg)

    progress_bar = DummyProgress()
    progress_text = DummyText()
    start_time = time.time()
    lock = threading.Lock()

    def process_row(row):
        """Processes a single row of footwear data to generate HS code suggestions.
        Args:
            row (pd.Series): A row from the input DataFrame.
        Returns:
            dict: A dictionary containing the original data with added HS code suggestions and reasoning.""" 
        
        original_index = row["original_index"]
        country = str(row[country_col]).strip().lower() if pd.notna(row[country_col]) else "unknown"
        product_type = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        
        # Append secondary name if available
        if name_col2 and name_col2 in row and pd.notna(row[name_col2]):
            product_type += " " + str(row[name_col2]).strip()
        material = str(row[material_col]).strip() if pd.notna(row[material_col]) else ""
        construction = str(row[construction_col]).strip() if construction_col and construction_col in row and pd.notna(row[construction_col]) else ""
        gender = str(row[gender_col]).strip() if gender_col and gender_col in row and pd.notna(row[gender_col]) else ""
        size = str(row[size_col]).strip() if size_col and size_col in row and pd.notna(row[size_col]) else ""

        # Build product description
        desc_parts = [f"Product for {country.upper()}:"]
        if gender:
            desc_parts.append(f"{gender}'s")
        if product_type:
            desc_parts.append(product_type)
        if material:
            desc_parts.append(f"Material: {material}.")
        if construction:
            desc_parts.append(f"Construction: {construction}.")
        if size and country.upper() in ["EUROPE", "UNITED KINGDOM"]:
            desc_parts.append(f"Size: {size}.")
        product_description = " ".join(desc_parts).strip()

        # Prepare base result row
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

        # Validate essential data
        if not product_type or not material or country == "unknown":
            base_result_row["reasoning_1"] = "Skipped due to missing essential data"
            return base_result_row

        if country not in pdf_data_cache or not pdf_data_cache[country]:
            with lock:
                print(f"Reloading PDF data for {country}...")
            pdf_data_cache[country] = load_all_pdf_data()
            if country not in pdf_data_cache or not pdf_data_cache[country]:
                base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
                return base_result_row

        # Get processed PDF data for the country
        processed_pdfs_for_current_country = pdf_data_cache[country]
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs_for_current_country)
        gri = processed_pdfs_for_current_country.get("gri", "")

        # Generate HS codes using the Gemini API
        try:
            generated_response = generate_hs_codes_ftw(
                model,
                product_description,
                country,
                relevant_chapters,
                gri=gri
            )

            product_df_row = extract_hs_codes(generated_response)
            if not product_df_row.empty:
                extracted_data = product_df_row.iloc[0].to_dict()
            else:
                extracted_data = {}

            print("==== RAW MODEL OUTPUT ====")
            print(generated_response)
            print("==== EXTRACTED DATA ====")
            print(extracted_data)
            print("==========================")

            # Warning for missing HS codes
            if all(not extracted_data.get(f"hs_code_{i}", "").strip() for i in range(1, 4)):
                print(f"⚠️ All HS codes missing for row (Original Index: {original_index})")
            # Populate result row with extracted data
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
                    else:
                        base_result_row[hs_col] = ""
                        base_result_row[cert_col] = 0
                        base_result_row[reas_col] = ""
            else:
                base_result_row["reasoning_1"] = "Error: Failed to parse response"

            return base_result_row

        except Exception as gen_e:
            base_result_row["hs_code_1"] = "ERROR"
            base_result_row["reasoning_1"] = str(gen_e)
            return base_result_row
    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_input.iterrows()]
        for df_idx, future in enumerate(futures):
            with lock:
                all_results_list.append(future.result())
                current_progress = (df_idx + 1) / total_rows
                progress_bar.progress(current_progress)
                elapsed_time = time.time() - start_time
                est_remaining = (elapsed_time / (df_idx + 1)) * (total_rows - (df_idx + 1)) if df_idx > 0 else 0
                progress_text.text(f"Processing row {df_idx + 1}/{total_rows}... Est. time remaining: {int(est_remaining)}s")

    return pd.DataFrame(all_results_list)