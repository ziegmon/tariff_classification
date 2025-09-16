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
from google.api_core import exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#___Documentation Path___#
PDF_DIRECTORY = "chapter_data"
CSV_PATH = "train_fold_EU_0.csv"
REJECTED_CODES_FILE = "rejected_classifications_footwear.json"

#___Variables for your footwear data structure___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "material_composition"
construction_col = "outsole_material"
gender_col = "gender_name"
hs_code_col = "tariff_code"

#___Aesthetics' Functions___#
def add_bg_from_local(image_file, opacity=0.3):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(
            rgba(255, 255, 255, {1 - opacity}),
            rgba(255, 255, 255, {1 - opacity})
        ), url(data:image/{"png"};base64,{encoded_string});
        background-size: cover;
    }}
    .stTextInput > div > div > input {{
        background-color: #f0f0f0;
    }}
    .stSelectbox [data-baseweb="select"] {{
        background-color: #f0f0f0;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        color: black;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def header(url):
    st.markdown(f'<p style="background-color:#898e94;color:#fefefe;font-size:24px;border-radius:35px;text-align:center;padding-left:20px;padding-right:20px;">{url}</p>', unsafe_allow_html=True)

def st_info(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:center;">{url}</p>', unsafe_allow_html=True)

def highlight(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:left;padding-left:20px;">{url}</p>', unsafe_allow_html=True)

#___Read PDF___#    
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file, strict=False)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

#___Load All PDF Data___#
def load_all_pdf_data(pdf_directory=PDF_DIRECTORY):
    """Loads and caches all PDF data for all countries"""
    pdf_cache = {}
    country_set = set()

    if not os.path.exists(pdf_directory):
        st.error(f"Directory {pdf_directory} does not exist!")
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

    st.session_state.country_list = sorted(list(country_set)) 
    return pdf_cache

#___Load Text Files___#
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

#___Gemini API Config___#
def configure_genai(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
    return model

#___Rejection System___#
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
def generate_hs_codes(
    model,
    product_description,
    country,
    relevant_chapters,
    legal_notes,
    classification_guide="",
    gri="",
    rejected_codes_snapshot=None,
    historical_data=None,
    guidelines=None,
):
    max_retries = 3
    retry_delay = 5  # seconds
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
            - **Switzerland:** EXACTLY 11 digits
            - **New Zealand:** EXACTLY 10 digits followed by 1 letter suffix
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
                "top_p": 0.5,
                "top_k": 15,
                "max_output_tokens": 20000,
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
        size = str(row[size_col]).strip() if size_col and size_col in row and pd.notna(row[size_col]) else ""

        desc_parts = [f"Product for {country.upper()}:"]
        if gender:
            desc_parts.append(f"{gender}'s")
        if product_type:
            desc_parts.append(product_type)
        if material:
            desc_parts.append(f"Material: {material}.")
        if construction:
            desc_parts.append(f"Construction: {construction}.")
        if size:
            desc_parts.append(f"Size: {size}.")
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

        if not product_type or not material or country == "unknown":
            st.warning(f"Skipping row {df_idx + 1} (Original Index: {original_index}): Insufficient data")
            base_result_row["reasoning_1"] = "Skipped due to missing essential data"
            all_results_list.append(base_result_row)
            continue

        # Reload PDF cache if missing or empty
        if country not in pdf_data_cache or not pdf_data_cache[country]:
            st.warning(f"Reloading PDF data for {country}...")
            pdf_data_cache[country] = load_all_pdf_data()
            if country not in pdf_data_cache or not pdf_data_cache[country]:
                st.error(f"No PDF data available for {country}.")
                base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
                all_results_list.append(base_result_row)
                continue

        processed_pdfs_for_current_country = pdf_data_cache[country]
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs_for_current_country)
        legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
        classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
        gri = processed_pdfs_for_current_country.get("gri", "")

        # Load country-specific text files
        country_texts = load_text_files_for_country(PDF_DIRECTORY, country)
        guidelines = country_texts.get(f"{country}_guidelines", "")

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
            else:
                extracted_data = {}

            # Debug: Always print
            print("==== RAW MODEL OUTPUT ====")
            print(generated_response)
            print("==== EXTRACTED DATA ====")
            print(extracted_data)
            print("==========================")

            # Extra check: Are all HS codes blank or missing?
            if all(not extracted_data.get(f"hs_code_{i}", "").strip() for i in range(1, 4)):
                print(f"⚠️ All HS codes missing for row {df_idx + 1} (Original Index: {original_index})")

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
                st.error(f"Failed to extract codes for row {df_idx + 1} (Original Index: {original_index}). Response format might be unexpected.")
                base_result_row["reasoning_1"] = "Error: Failed to parse response"

            all_results_list.append(base_result_row)

        except Exception as gen_e:
            st.error(f"Error generating code for row {df_idx + 1} (Original Index: {original_index}): {gen_e}")
            base_result_row["hs_code_1"] = "ERROR"
            base_result_row["reasoning_1"] = str(gen_e)
            all_results_list.append(base_result_row)

    return pd.DataFrame(all_results_list)

#___Authentication___#
def check_password():
    def password_entered():
        if (
            st.session_state["username"] == st.secrets["AUTH"]["USERNAME"]
            and st.session_state["password"] == st.secrets["AUTH"]["PASSWORD"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input(
            "Password", type="password", key="password", on_change=password_entered
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input(
            "Password", type="password", key="password", on_change=password_entered
        )
        st.error("Incorrect combination")
        return False
    else:
        return True

#___Regeneration Functions___#
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
        product_description = product_row['product_description']
        country = product_row['input_country'].strip().lower()

        if country not in st.session_state.pdf_cache or not st.session_state.pdf_cache[country]:
            print(f"PDF cache miss for {country}. Loading PDFs.")
            st.session_state.pdf_cache[country] = load_all_pdf_data()
            if not st.session_state.pdf_cache.get(country):
                print(f"Could not load PDFs for {country}. Returning.")
                return

        processed_pdfs = st.session_state.pdf_cache[country]
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs)
        legal_notes = processed_pdfs.get("legal_notes", "")
        guide = processed_pdfs.get("classification_guide", "")
        gri = processed_pdfs.get("gri", "")
        rejected_codes_snapshot = load_rejected_codes(product_description, country)

        with st.spinner(f"Regenerating Index: {original_index}..."):
            new_response = generate_hs_codes(model, product_description, country, relevant_chapters, legal_notes, guide, gri=gri, rejected_codes_snapshot=rejected_codes_snapshot)
            new_product_df_row = extract_hs_codes(new_response)
            
            if not new_product_df_row.empty:
                df = st.session_state.bulk_results_df
                target_df_index = df[df['original_index'] == original_index].index
                if not target_df_index.empty:
                    idx_loc = target_df_index[0]
                    new_data = new_product_df_row.iloc[0]
                    for i in range(1, 4):
                        hs, cert, reas = f'hs_code_{i}', f'certainty_{i}', f'reasoning_{i}'
                        df.loc[idx_loc, hs] = new_data.get(hs, '')
                        df.loc[idx_loc, cert] = new_data.get(cert, 0)
                        df.loc[idx_loc, reas] = new_data.get(reas, '')
                    
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