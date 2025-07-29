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

"""     **Input:**
        - Country: {country}
        - Product Description: {product_description}
        - Chapter Data: {relevant_chapters}
        - GRI: {gri} 

        """
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
        You are an expert in customs tariff classification with access to two PDFs: the General Rules of Interpretation (GRI) and the Harmonized Tariff Schedule (HTS) chapter pages for footwear (e.g., Chapter 64). Given user inputs via a Streamlit app—**country** (used for context only, not hardcoded logic), **product description** (including material, construction, gender, and use, with a focus on sports footwear like tennis shoes, basketball shoes, gym shoes, training shoes, or hiking footwear), and the provided PDFs—output three distinct, maximally granular HTS codes (e.g., 8-digit or 10-digit like 6404.11.99.21 or 6404.11.90) with their descriptions, ensuring accurate handling of hierarchical codes and gender specificity. Follow these steps:

        1. Parse the product description to extract key details (e.g., outer sole material, upper material, construction, protective features, gender, and use). Prioritize sports footwear characteristics (e.g., tennis, basketball, gym, training, or hiking).
        2. Analyze the HTS PDF to determine its hierarchical structure (e.g., detailed subheadings with multiple levels or row-based indentation) and whether it includes gender-specific subheadings.
        3. Apply the GRI sequentially to navigate the HTS codes in the PDF, starting with sports footwear subheadings (e.g., 6404.11 or 6402.12 for textile uppers with rubber/plastic soles) when the description suggests sports use.
        4. Identify three distinct HTS codes at the deepest subheading level available in the PDF (e.g., 6404.11.99.21 or 6404.11.90, not 6-digit like 6404.11 unless no deeper codes exist), prioritizing gender-specific subheadings when available (e.g., 6404.11.99.21 for men's/boys') and ensuring codes exist in the PDF to avoid inventing codes. Each code must be unique and not a parent of another.
        5. For each HTS code, provide a brief explanation as bullet points, detailing why each hierarchical level (e.g., heading, subheading) was chosen, referencing the GRI and tariff item description, and noting if gender is not distinguished in the PDF.
        6. Account for country-specific tariff rules from the PDF, if present, using the country input for context.
        7. Format the output with the HTS code in bold and larger font, followed by bullet points for the hierarchical explanations, addressing sports use and gender if specified.

        **Input:**
        - Country: {country}
        - Product Description: {product_description}
        - Chapter Data: {relevant_chapters}
        - GRI: {gri}

        **Output Format:**
        - **HTS Code: [Code, e.g., 6404.11.99.21 or 6404.11.90]** (bold, larger font)
        - 64.04: [Explanation, e.g., "Footwear with rubber soles and textile uppers per GRI 1."]
        - 6404.11: [Explanation, e.g., "Sports footwear (training shoes) identified in description."]
        - 6404.11.99: [Explanation, e.g., "Other sports footwear, not specific to canvas or hiking."]
        - 6404.11.99.21: [Explanation, e.g., "Men's/boys' matches gender, if in PDF; otherwise, no gender-specific subheading."]
        - **HTS Code: [Code, e.g., 6404.11.99.90 or 6402.19.90]** (bold, larger font)
        - 64.04: [Explanation, e.g., "Footwear with rubber soles and textile uppers per GRI 1."]
        - 6404.11: [Explanation, e.g., "Sports footwear (training shoes) identified."]
        - 6404.11.99: [Explanation, e.g., "Other sports footwear, not specific to canvas or hiking."]
        - 6404.11.99.90: [Explanation, e.g., "Other, no gender-specific codes in PDF."]
        - **HTS Code: [Code, e.g., 6404.19.90.91 or 6404.19.90]** (bold, larger font)
        - 64.04: [Explanation, e.g., "Footwear with rubber soles and textile uppers per GRI 1."]
        - 6404.19: [Explanation, e.g., "Other non-sports footwear, less likely given training use."]
        - 6404.19.90: [Explanation, e.g., "Other, not specific to canvas."]
        - 6404.19.90.91: [Explanation, e.g., "Men's/boys' matches gender, if in PDF; otherwise, no gender-specific subheading."]

        Ensure the response is concise, avoids code output, prioritizes sports footwear subheadings (e.g., 6404.11 or 6402.12), selects three distinct HTS codes at the deepest granularity available in the PDF (e.g., 8-digit or 10-digit, not 6-digit unless no deeper codes exist), formats HTS codes in bold and larger font with bullet-point explanations for each hierarchical level, uses the provided PDFs as the primary reference to select only valid HTS codes, dynamically adapts to the PDF’s hierarchical structure (detailed subheadings or row-based), and notes when gender distinctions are absent in the PDF.
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
