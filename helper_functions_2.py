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


#___Documentation Path___#
PDF_DIRECTORY = "chapter_data"


#___Variables___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "composition"
construction_col = "material_type"
gender_col = "division"
hs_code_col = "tariff_code"


#___Aesthetics' Functions___#
# These functions should handle the page background and formatting
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

    """,
    unsafe_allow_html=True
    )

def header(url):
    st.markdown(f'<p style="background-color:#898e94;color:#fefefe;font-size:24px;border-radius:35px;text-align:center;;padding-left:20px;;padding-right:20px;">{url}</p>', unsafe_allow_html=True)


def st_info(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:center;">{url}</p>', unsafe_allow_html=True)


def highlight(url):
    st.markdown(f'<p style="background-color:rgba(137, 142, 148, 0.5);color:#fefefe;font-size:24px;border-radius:30px;text-align:left;padding-left:20px;">{url}</p>', unsafe_allow_html=True)


#___Read PDF___#
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
        st.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""


#___Country Specific Documentation___#    
# loading one country at the time to reduce the token bloat, by reducing the prompt length
def load_pdfs_for_country(pdf_directory, country):
    processed_pdfs = {}
    if not os.path.exists(pdf_directory):
        st.error(f"Directory {pdf_directory} does not exist!")
        return {}

    country_set = set(["canada"]) # More countries to be added or set to be replaced

    for file in os.listdir(pdf_directory):
        if not file.lower().startswith(country.lower()):
            continue

        file_path = os.path.join(pdf_directory, file)
        filename = Path(file).stem  # filename without extension

        # extracting the country from filename for the relevant documents
        if '_chapter_' in filename:
            country = filename.split('_chapter_')[0].lower()
            chapter = filename.split('_chapter_')[1]
            key = f"{country}_chapter_{chapter}"
            text = extract_text_from_pdf(file_path)
            processed_pdfs[key] = text
            country_set.add(country)
        elif '_tariff_schedule' in filename:
            country = filename.split('_tariff_schedule')[0].lower()
            key = f"{country}_tariff_schedule"
            text = extract_text_from_pdf(file_path)
            processed_pdfs[key] = text
            country_set.add(country)
        elif '_legal_notes' in filename:
            country = filename.split('_legal_notes')[0].lower()
            key = f"{country}_legal_notes"
            text = extract_text_from_pdf(file_path)
            processed_pdfs[key] = text
            country_set.add(country)
        elif '_classification_guide' in filename:
            country = filename.split('_classification_guide')[0].lower()
            key = f"{country}_classification_guide"
            text = extract_text_from_pdf(file_path)
            processed_pdfs[key] = text
            country_set.add(country)
        elif '_gri' in filename:
            country = filename.split('_gri')[0].lower()
            key = f"{country}_gri"
            text = extract_text_from_pdf(file_path)
            processed_pdfs[key] = text
            country_set.add(country)

    st.session_state.country_list = sorted(list(country_set))
    return processed_pdfs




# remove the previous function if this works
def load_all_pdf_data(pdf_directory):
    """
    Loads and caches all PDF data for all countries found in the PDF_DIRECTORY.
    The structure will be {country: {doc_type: text_content, ...}, ...}
    """
    pdf_cache = {}
    country_set = set() # To keep track of all found countries

    if not os.path.exists(pdf_directory):
        st.error(f"Directory {pdf_directory} does not exist!")
        return {}

    for file in os.listdir(pdf_directory):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, file)
            filename = Path(file).stem.lower() # Get filename without extension, lowercase

            # Extract country and document type from filename
            parts = filename.split('_')
            if len(parts) > 1:
                current_country = parts[0] # Assuming country name is the first part
                doc_type_key = "_".join(parts[1:]) # Remaining parts form the doc type key

                if current_country not in pdf_cache:
                    pdf_cache[current_country] = {}
                
                # Check for specific document types and store them
                if '_chapter_' in filename:
                    # Example: "canada_chapter_61" -> doc_type_key "chapter_61"
                    text = extract_text_from_pdf(file_path)
                    pdf_cache[current_country][doc_type_key] = text
                    country_set.add(current_country)
                elif '_tariff_schedule' in filename:
                    # Example: "canada_tariff_schedule" -> doc_type_key "tariff_schedule"
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
                st.warning(f"Skipping PDF: '{file}' due to unconventional naming. Expected 'country_doc_type.pdf'.")

    st.session_state.country_list = sorted(list(country_set)) 
    return pdf_cache


#___Gemini API Config___#
def configure_genai(api_key):
    genai.configure(api_key=st.secrets["API_KEY"])
    # using gemini-1.5-flash seems to be enough, does not exceed quota and seems to performa better from 2.0
    # 1.5 can be fine tuned, 2.0 can't
    model = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
    return model


def generate_hs_codes(
    model,
    product_description,
    country,
    relevant_chapters,  # Changed from chapter_content
    legal_notes,
    gri="",
    guidelines=None,
):
    print("_______---_______")

    prompt = f"""
    **CONTEXT & RESOURCES:**
    - **Product Description:** {product_description}
    - **Target Country:** {country.upper()}
    - **General Rules of Interpretation (GRI):** {gri}
    - **Official Chapter Content Table:**  
        • **Tariff Item (6‑digit):** top‑level headings  
        • **Tariff Item (8‑digit):** more specific sub‑headings under each 6‑digit  
        • **SS (10‑digit):** statistical suffixes extending the 8‑digit item  
        • **Description of Goods:** use to choose the correct heading, sub‑heading AND gender

    ---

    **ROLE:** You are an expert customs classifier, with expertise in a wide number of countries, tasked with accurately classifying products using only the provided official Harmonized System (HS) documents. Precision is paramount.

    **ULTRA-CRITICAL CLASSIFICATION RULES (STRICT COMPLIANCE REQUIRED):**

    1. **ABSOLUTE CODE VALIDITY:** ONLY propose HS codes (including all digits and statistical suffixes) that appear *VERBATIM* in the `OFFICIAL CHAPTER CONTENT`. Do not invent, assume, or truncate. If a code/suffix is not explicitly listed for a relevant heading/subheading, it does not exist.
    2. **STATISTICAL SUFFIX PRIORITY:** Use the most specific, applicable statistical suffixes (e.g., '.10', '.25', '.99') provided in `OFFICIAL CHAPTER CONTENT`. Only use '.00' if no other specific suffix is listed or appropriate.
    3. **SINGLE CORRECT CODE PRINCIPLE:** Every product has one single, most correct HS code. OPTION 1 MUST be your highest confidence classification. Options 2 and 3 are for genuine ambiguity after rigorous application of all rules and data.
    4. **FOOTWEAR CLASSIFICATION:**
    - For products identified as footwear (e.g., 'shoe', 'boot', 'sandal'), exclusively consider Chapter 64.
    - Classification must be based on both the material of the upper and the material of the outer sole, as specified in the heading descriptions of Chapter 64.
    - Additionally, consider the type of footwear (e.g., sports, casual, protective) and any special features (e.g., waterproof, orthopedic) as specified in the subheadings of Chapter 64.
    - Example: A shoe with a rubber sole and textile upper should be classified under heading 6404 (uppers of textile materials), not 6403 (uppers of leather).

    **TASK:**
    Based *exclusively and meticulously* on the provided content and adhering to ALL critical rules, determine the *THREE most likely HS codes + statistical suffixes* for the product.

    Parse the table in exactly three steps:

    1. **Drill down to the 8‑digit sub‑heading.**  
    - Under the Tariff Item column, find the 8‑digit Tariff Item row that refines by construction or material.
    - Quote the row’s DESCRIPTION to show your match.
    - **Do not** look at or list any SS codes in this step—stop at 8 digits.
    - Only after identifying a 8-digit code move to the column on the right for each 8 digit code you identified. There are almost always multiple options.

    2. **Select the 10‑digit SS suffix.**  
    - a. **List all numeric SS codes** that appear *verbatim* under your selected 8‑digit (e.g. 00, 10, 91, 92, 93). At this step you have 8 digits already and the SS are the ninth and tenth digit.  
    - b. **Reject any SS** not in that list—if it’s not in your table, it doesn’t exist.  
    - c. **Ignore “nan”** rows, they’re just the parent heading.  
    - d. Choose the SS whose DESCRIPTION explicitly states the correct gender/age (“Men’s or boys’” → 91, “Women’s or girls’” → 92, “Children’s” → 93).

    **FORMAT (Strictly Adhere):**

    ### OPTION 1: [HS code + suffix] - XX% certainty

    #### PRODUCT DESCRIPTION:
    [Re-state the product description based *solely* on the information received, focusing on classification-relevant details such as materials (e.g., upper and sole for footwear), construction methods, and intended use.]

    #### REASONING STRUCTURE:
    Use this detailed structure for justification:

    1. *GRI Application*: Identify and apply ALL relevant General Rules of Interpretation (GRI 1-6, in order). Explain how each applied GRI leads to the decision.
    2. *Chapter & Section Fit*: State chosen Chapter/Section (e.g., Chapter 64: Footwear). Confirm inclusion based on material, construction, use, and notes. Confirm chapter is NOT excluded.
    3. *Heading & Subheading Determination*: Justify the 4-digit heading and 6-digit subheading using the texts and legal notes. For footwear, specify how the materials of the upper and outer sole, as well as the type of footwear, align with the heading and subheading descriptions.
    4. *National Tariff Line / Statistical-Suffix Determination*  
        a. Examine the **SS (Statistical Suffix) column** that appears directly after the 8-digit sub-heading in the `OFFICIAL CHAPTER CONTENT`, you can only look at the SS when you have identified a 8-digit code.   
        b. **Output the full code including the verbatim suffix**, e.g. 6403.99.11, 6203.42.12, etc. Never truncate or invent digits.
        c. **Cross-check the Description of goods column** that appears **beside each SS suffix**.  
            The correct suffix is the one whose description contains the gender/age term that matches the product (e.g., “Men’s footwear”, “Women’s footwear”, “Infants’ footwear”).  
    5. *Exclusions & Verifications*: Explicitly rule out other plausible but incorrect classifications (e.g., Chapter 62 for footwear). Confirm code is NOT rejected and adheres to all critical rules.

    #### LEGAL BASIS:
    Cite specific GRI rules, Section/Chapter notes, and heading/subheading texts from provided documents. Quote directly or paraphrase precisely. Include `Classification Guide` cross-references if applicable.

    ---

    **HARDCODED CLASSIFICATION RULES (Apply Rigorously - These override general interpretations if applicable):**

    - **Mandatory gender check:** always read the “Description of goods” column that accompanies each SS suffix and pick the suffix whose description explicitly states the correct gender/age group.
    - **Gender/Age Suffix Rule:** Always check the SS column for an explicit “men / women / boys / girls / infants” suffix before accepting .00 or .90.
    - **No "Same as OPTION X":** Each option requires full, independent reasoning.
    - **Knitted vs. Woven Exclusions:** If knitted/crocheted, ALWAYS exclude non-knitted headings (e.g., Chapter 62). If woven, ALWAYS exclude knitted headings (e.g., Chapter 61). **Note:** These rules apply only to apparel products classified under Chapters 61 and 62.
    - **Material:** *Polyester* is a *man-made fiber*.
    - **Footwear Material Consideration:** Always consider both the upper and outer sole materials for footwear classification. Do not base the classification solely on one material.
    - **Mixed Materials in Footwear:** If footwear has uppers or soles made of multiple materials, refer to the specific subheadings or notes in Chapter 64 that address such cases, such as subheadings for 'other' materials or specific combinations.
    - **Chapter Definitions:**
    - Chapter 61: *Articles of Apparel and Clothing Accessories, Knitted or Crocheted*.
    - Chapter 62: *Articles of Apparel and Clothing Accessories, Not Knitted or Crocheted*.
    - Chapter 64: *Footwear, gaiters and the like; parts of such articles*.
    - **HS Code Structure:**
    - Apply `GRI` from `{gri}`.
    - Use chapter-specific `LEGAL NOTES` from `{legal_notes}`.
    - DO NOT add suffixes/sub-codes unless *VERBATIM* present in `OFFICIAL CHAPTER CONTENT`.
    - Use the *longest HS Code version explicitly shown* in `OFFICIAL CHAPTER CONTENT`.
    - Full code for `{country.upper()}` must include *all visible digits* (e.g., 6103.43.00.15 includes .00; 6109.9000 includes 000). Do not skip intermediate levels.
    - After 4 digits, every pair of digits is separated by `.` (e.g., 6109.10.20, NOT 61091020).
    - NEVER use spaces within HS Code digits.
    - **Switzerland Specific:** For Switzerland, provide an 8-digit HS Code. Add `*-000*` as the statistical suffix ONLY IF the 8-digit code exists in `OFFICIAL CHAPTER CONTENT` and no more specific 11-digit Swiss suffix is listed/applicable.
    - **Product-Specific Overrides:**
    - `6204.43.00.90` does NOT exist on Canada's Tariff Schedule. DO NOT use for Canada.
    - Knitted *Tank Top* = T-shirt.
    - Unpadded *VEST* = `62.11.33`.
    - *Crop Top* is NEVER a t-shirt (except China).
    - *Crop Top* is ALWAYS *Other Garments* unless specific text dictates otherwise.
    - *Crop Top* is NEVER `61.04` (unless overwhelming country-specific legal notes).
    - *Short Tights* = *shorts*.
    - *Pullover* = *Other garments*, not a shirt.
    - For WOMEN, knitted t-shirt (incl. "Tank Top") = `61.09`. Generally NOT `6106` (shirt/blouse).
    - *Race Singlet* is NEVER `62.11`. If woven, typically `62.07`.
    - *TIGHTS* = tight-fitting stretch trousers.
    - *TIGHTS SHORT* = tight-fitting stretch shorts.
    - Ignore *color* or *identifier terms* (SKU, style names) unless directly affecting material, primary use, or construction relevant to tariff distinctions.
    - For man-made materials (e.g., polyester, polyamide), it is NOT "other textile materials" unless a blend where man-made fiber isn't predominant or specific blend notes apply.
    - Pay VERY SPECIAL ATTENTION to product type (men/women/infant garment, garment type) to avoid suggesting completely incorrect codes (e.g., sock as trouser).

    ---

    In the country-specific guidelines (`{guidelines}`), you are provided an example classification that follows an ideal structure. Model your response structure after this example.

    **FINAL REMINDERS:**
    - Do not propose any code segment that  that does not exist *verbatim* in the official documentation. If a classification path dead-ends, indicate that no valid code can be found.
    - Your primary directive is accuracy based *only* on the provided documents and rules.
    - Double-check all suggestions: confirm verbatim existence in Tariff Schedule and verify against `Previously Rejected HS Codes`.
    """

    # for chapter_num, chapter_text in relevant_chapters:
    #     prompt += f"\n--- CHAPTER {chapter_num} ---\n{chapter_text}\n"

    # print("\n--- FULL PROMPT SENT TO GEMINI (for current product and country) ---")
    # print(prompt)
    # print("--- END OF PROMPT ---")

    try:
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.5,
            "top_k": 15,
            "max_output_tokens": 8192,
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        token_count = model.count_tokens(prompt).total_tokens

        print(f"The prompt contains {token_count} tokens.")

        validated_response = response.text
        return validated_response
    except Exception as e:
        return f"Error generating HS codes: {str(e)}"
    

# finding relevant chapters based on product description
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


def save_rejected_code(product, country, code): # Accepts 3 arguments
    entry = {
        "product_description": product,
        "country": country,
        "rejected_code": code 
    }



def extract_hs_codes(text):
    # extracting the product description from JSON format
    product_desc_match = re.search(r'"Product Description":"(.*?)"', text)
    
    # if no JSON format product description, extracts from the first option's product description
    if not product_desc_match:
        product_desc_match = re.search(r'#### PRODUCT DESCRIPTION:\s*(.*?)(?=####|$)', text, re.DOTALL)
    
    product_description = product_desc_match.group(1).strip() if product_desc_match else "Not found"
    
    # pattern to capture the full HS code format with all digits, periods, and spaces
    options = re.findall(r'### OPTION \d+: ([0-9]+(?:\.[0-9]+)*(?:\s+[0-9]+)?) - (\d+)% certainty\s+(.*?)(?=### OPTION \d+:|$)', 
                         text, re.DOTALL)
    
    # Create a row dict starting with product description
    row = {'product_description': product_description}
    
    # adds each code, likelihood, and reasoning as separate columns
    for i, (code, certainty, details) in enumerate(options):
        # reasoning section - looking for both REASONING and REASONING STRUCTURE
        reasoning_match = re.search(r'#### REASONING(?:\s+STRUCTURE)?:(.*?)(?=#### LEGAL BASIS:|$)', details, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"
        
        option_num = i + 1
        row[f'hs_code_{option_num}'] = code.strip()
        row[f'certainty_{option_num}'] = int(certainty)
        row[f'reasoning_{option_num}'] = reasoning
    
    df = pd.DataFrame([row])
    
    return df



def mark_as_rejected(product_id, option_num, product_description, country):
    hs_code = None
    for _, row in st.session_state.final_df.iterrows():
        if row['product_id'] == product_id and row['option_num'] == option_num:
            hs_code = row['hs_code']
            break
    
    if hs_code:
        save_rejected_code(product_description, country, hs_code)
        
        if product_id not in st.session_state.rejected_codes_count:
            st.session_state.rejected_codes_count[product_id] = []
        
        if hs_code not in st.session_state.rejected_codes_count[product_id]:
            st.session_state.rejected_codes_count[product_id].append(hs_code)
        
        if len(st.session_state.rejected_codes_count[product_id]) >= 2:
            st.session_state.regenerate_products.add(product_id)
            st.success(f"Product {product_id + 1} is marked for regeneration after 2+ rejections.")
        else:
            st.info(f"Code marked as incorrect. {2 - len(st.session_state.rejected_codes_count[product_id])} more rejection(s) needed before regeneration.")


# display products with selection options
def display_products_with_selection(final_df):
    st.write("### Select the correct HS code for each product:")
    
    product_ids = final_df['product_id'].unique()
    
    for product_id in product_ids:
        group = final_df[final_df['product_id'] == product_id]
        product_info = st.session_state[f'product_info_{product_id}']
        
        with st.expander(f"Product {product_id + 1}: {product_info['gender']}'s {product_info['product_type']}"):
            st.write(f"**Full Description**: {group['product_description'].iloc[0]}")
            
            cols = st.columns(3)
            
            for i, (_, row) in enumerate(group.iterrows()):
                with cols[i % 3]:
                    option_num = row['option']
                    st.write(f"**Option {option_num}**")
                    st.write(f"HS Code: {row['hs_code']}")
                    st.write(f"Certainty: {row['certainty']}")
                    
                    with st.expander("Show reasoning"):
                        st.write(row['reasoning'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.checkbox("Select", key=f"select_{product_id}_{option_num}"):
                            st.session_state.selected_codes[product_id] = {
                                'hs_code': row['hs_code'],
                                'certainty': row['certainty'],
                                'reasoning': row['reasoning'],
                                'option': option_num
                            }
                    
                    with col2:
                        if st.button("Mark Incorrect", key=f"reject_{product_id}_{option_num}"):
                            mark_as_rejected(
                                product_id, 
                                option_num, 
                                product_info['product_description'], 
                                product_info['country']
                            )
    
    if st.button("Generate Final Table with Selections"):
        if st.session_state.selected_codes:
            final_selections = []
            
            for product_id in product_ids:
                if product_id in st.session_state.selected_codes:
                    product_info = st.session_state[f'product_info_{product_id}']
                    selected = st.session_state.selected_codes[product_id]
                    
                    row_data = {
                        'Product ID': product_id + 1,
                        'Country': product_info['country'].upper(),
                        'Product Type': product_info['product_type'],
                        'Material': product_info['material'],
                        'Construction': product_info['construction'],
                        'Gender': product_info['gender'],
                        'Selected HS Code': selected['hs_code'],
                        'Certainty': selected['certainty'],
                        'Reasoning': selected['reasoning']
                    }
                    
                    final_selections.append(row_data)
            
            selections_df = pd.DataFrame(final_selections)
            
            st.write("### Final Selected HS Codes:")
            st.dataframe(selections_df)
            
            csv = selections_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Selected HS Codes as CSV",
                data=csv,
                file_name="selected_hs_codes.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one HS code before generating the final table.")


# regenerate products with 2+ rejections
def regenerate_products():
    if not st.session_state.regenerate_products:
        st.info("No products need regeneration.")
        return
    
    st.write("### Regenerating HS codes for products with 2+ rejections:")
    
    model = st.session_state.model
    
    for product_id in st.session_state.regenerate_products:
        product_info = st.session_state[f'product_info_{product_id}']
        st.write(f"Regenerating for Product {product_id + 1}: {product_info['product_description']}")
        
        country = product_info['country']
        product_description = product_info['product_description']
    
        
        chapter_content = st.session_state.chapter_content
        legal_notes = st.session_state.legal_notes
        classification_guide = st.session_state.classification_guide
        
        try:
            generated_response = generate_hs_codes(
                model,
                product_description,
                country,
                chapter_content,
                legal_notes,
                classification_guide
            )
            
            product_df = extract_hs_codes(generated_response)
            
            product_df['product_id'] = product_id
            product_df['product_description'] = product_description
            
            st.session_state.final_df = st.session_state.final_df[st.session_state.final_df['product_id'] != product_id]
            
            st.session_state.final_df = pd.concat([st.session_state.final_df, product_df], ignore_index=True)
            
            st.success(f"Successfully regenerated options for Product {product_id + 1}")
        except Exception as e:
            st.error(f"Error regenerating codes for product {product_id + 1}: {str(e)}")
        
        st.session_state.rejected_codes_count[product_id] = []
    
    st.session_state.regenerate_products = set()
    
    st.success("Regeneration complete! New HS code options have been generated.")
    display_products_with_selection(st.session_state.final_df)


def regenerate_single_product(original_index):
    if st.session_state.bulk_results_df is None:
        print("bulk_results_df is None. Returning.") 
        return
    try:
        product_row_series = st.session_state.bulk_results_df[st.session_state.bulk_results_df['idx'] == original_index]
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
            st.session_state.pdf_cache[country] = load_all_pdf_data(PDF_DIRECTORY, country)
            if not st.session_state.pdf_cache.get(country):
                print(f"Could not load PDFs for {country}. Returning.") 
                return

        processed_pdfs = st.session_state.pdf_cache[country]
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs)
        legal_notes = processed_pdfs.get(f"{country}_legal_notes", "")
        guide = processed_pdfs.get(f"{country}_classification_guide", "")
        gri = processed_pdfs.get(f"{country}_gri", "")

        with st.spinner(f"Regenerating Index: {original_index}..."):
            new_response = generate_hs_codes(model, product_description, country, relevant_chapters, legal_notes, guide, gri=gri)
            new_product_df_row = extract_hs_codes(new_response)
            if not new_product_df_row.empty:
                df = st.session_state.bulk_results_df
                target_df_index = df[df['idx'] == original_index].index
                if not target_df_index.empty:
                    idx_loc = target_df_index[0]
                    new_data = new_product_df_row.iloc[0]
                    for i in range(1, 4):
                        hs, cert, reas = f'hs_code_{i}', f'certainty_{i}', f'reasoning_{i}'
                        df.loc[idx_loc, hs] = new_data.get(hs, '')
                        df.loc[idx_loc, cert] = new_data.get(cert, 0)
                        df.loc[idx_loc, reas] = new_data.get(reas, '')
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
        st.session_state.regenerate_queue.discard(original_index)


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


def load_all_pdf_data(pdf_directory):
    """
    Loads and caches all PDF data for all countries.
    This should be called *once* before processing any data.
    """
    pdf_cache = {}
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            country = filename.split('_')[0].lower() 
            if country not in pdf_cache:
                pdf_cache[country] = {}
            key = "_".join(filename.split('_')[1:]).replace(".pdf", "")
            pdf_path = os.path.join(pdf_directory, filename)
            pdf_cache[country][key] = extract_text_from_pdf(pdf_path)  
    return pdf_cache


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

        if not product_type or not material or country == "unknown":
            st.warning(f"Skipping row {df_idx + 1} (Original Index: {original_index}): Insufficient data (missing Product Name/Type, Material, or Country).")
            base_result_row["reasoning_1"] = "Skipped due to missing essential data"
            all_results_list.append(base_result_row)
            continue

        if country not in pdf_data_cache:
            st.warning(f"No PDF data available for {country}.")
            base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
            all_results_list.append(base_result_row)
            continue

        processed_pdfs_for_current_country = pdf_data_cache[country] # This is correct

        # Correctly get relevant chapters
        relevant_chapters = find_relevant_chapters(product_description, country, processed_pdfs_for_current_country)

        # Correctly get other documents by their keys (which are derived from filename without country prefix)
        legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
        classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
        gri = processed_pdfs_for_current_country.get("gri", "")

        # Load country-specific text files (e.g., guidelines)
    



        if not relevant_chapters and not legal_notes and not classification_guide and not gri:
            st.warning(f"Row {df_idx + 1} (Original Index: {original_index}): No tariff context (Chapters, Notes, Guide, GRI, Guidelines) found for {country}. Results might be inaccurate.")


        try:
            generated_response = generate_hs_codes(
                model,
                product_description,
                country,
                relevant_chapters, # Pass the list of (chapter_num, chapter_text) tuples
                legal_notes,
                classification_guide,
                gri=gri
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
                st.error(f"Failed to extract codes for row {df_idx + 1} (Original Index: {original_index}). Response format might be unexpected.")
                base_result_row["reasoning_1"] = "Error: Failed to parse response"

            all_results_list.append(base_result_row)

        except Exception as gen_e:
            st.error(f"Error generating code for row {df_idx + 1} (Original Index: {original_index}): {gen_e}")
            base_result_row["hs_code_1"] = "ERROR"
            base_result_row["reasoning_1"] = str(gen_e)
            all_results_list.append(base_result_row)

    return pd.DataFrame(all_results_list)


def extract_simplified_hs_codes(text):
    options_data = []
    # pattern to capture HS code and certainty from "### OPTION X: [HS code] - YY% certainty"
    options = re.findall(r'### OPTION \d+: ([0-9]+(?:\.[0-9]+)*(?:\s+[0-9]+)?) - (\d+)% certainty', text)

    for i, (code, certainty) in enumerate(options):
        options_data.append({
            f'Option {i+1} HS Code': code.strip(),
            f'Option {i+1} Certainty': int(certainty)
        })
    return pd.DataFrame(options_data)
