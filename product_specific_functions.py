"""
Product-specific classification functions for Apparel and Footwear
"""
from data_processing_functions import format_historical_data_from_csv
from flask import current_app

def find_relevant_chapters_apparel(product_description, country, country_specific_pdf_data):
    """Find relevant chapters for apparel products"""
    keywords = {
        "shirt": ["61", "62"], "t-shirt": ["61", "62"], "jacket": ["61", "62"],
        "pant": ["61", "62"], "shorts": ["61", "62"], "vest": ["61", "62"],
        "tank": ["61", "62"], "trouser": ["61", "62"], "legging": ["61", "62"],
        "tights": ["61", "62"], "coat": ["61", "62"], "bra": ["61", "62"],
        "bag": ["42"], "backpack": ["42"], "leather": ["42"],
        "headgear": ["65"], "hat": ["65"], "cap": ["65"],
        "balaclava": ["65"], "visor": ["65"], "beanie": ["65"],
    }
    
    product_lower = product_description.lower()
    potential_chapters = set()
    
    for keyword, chapters in keywords.items():
        if keyword in product_lower:
            for chapter in chapters:
                potential_chapters.add(chapter)
    
    # Knit vs Woven logic
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

def find_relevant_chapters_footwear(product_description, country, country_specific_pdf_data):
    """Find relevant chapters for footwear products"""
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

def get_apparel_prompt_additions(country, product_description, historical_data_string, rejected_hs_codes_for_prompt):
    """Generate apparel-specific prompt sections"""
    return f"""
    **ROLE:** You are an expert customs classifier, with expertise in a wide number of countries, tasked with accurately classifying APPAREL products using only the provided official Harmonized System (HS) documents. Precision is paramount.

    **ULTRA-CRITICAL CLASSIFICATION RULES (MANDATORY COMPLIANCE):**
    
    5.  **KNIT/WOVEN INTEGRITY:**
        * **Knitted/Crocheted:** EXCLUSIVELY consider Chapter 61. NEVER suggest codes from Chapter 62.
        * **Woven (Not Knitted/Crocheted):** EXCLUSIVELY consider Chapter 62. NEVER suggest codes from Chapter 61.
    
    7.  **TIGHTS** = tight-fitting stretchy trousers, similar to leggings, regardless of being KNITTED or WOVEN.
    8.  **Pima Cotton Classification:** Treat as pure cotton. If predominant material, classify under "Of cotton" heading.
    
    **HARDCODED CLASSIFICATION RULES (Apply Rigorously):**
    * **Knitted vs. Woven Exclusions:** If knitted/crocheted, ALWAYS exclude Chapter 62. If woven, ALWAYS exclude Chapter 61.
    * **Material:** Polyester is a man-made fiber. Pima Cotton is cotton.
    * **Chapter Definitions:**
        * Chapter 61: Articles of Apparel and Clothing Accessories, Knitted or Crocheted
        * Chapter 62: Articles of Apparel and Clothing Accessories, Not Knitted or Crocheted
    * **Product-Specific Overrides:**
        * UNISEX products → classify as WOMEN
        * Knitted Tank Top = T-shirt
        * Crop Top = Other Garments (not t-shirt except China)
        * Short Tights = shorts
        * TIGHTS = tight-fitting trousers
    """

def get_footwear_prompt_additions(country, product_description, historical_data_string, rejected_hs_codes_for_prompt):
    """Generate footwear-specific prompt sections"""
    return f"""
    **ROLE:** You are an expert customs classifier specializing in FOOTWEAR, with expertise in multiple countries, tasked with accurately classifying footwear products using only the provided official Harmonized System (HS) documents. Precision is paramount.

    **ULTRA-CRITICAL FOOTWEAR CLASSIFICATION RULES (STRICT COMPLIANCE REQUIRED):**

    1. **SOLE MATERIAL PRIORITY:** Classification depends heavily on outer sole material (rubber, leather, textile, etc.)
    2. **UPPER MATERIAL CONSIDERATION:** Upper material affects subheading classification
    3. **SPORTS vs REGULAR FOOTWEAR:** Athletic/sports footwear has specific subheadings (e.g., 6404.11)
    4. **GENDER CLASSIFICATION:** Many tariff lines distinguish between men's/boys' and women's/girls' footwear
    5. **SIZE CODE:** A size code of -1 means that size is not relevant for this country
    
    **FOOTWEAR-SPECIFIC REASONING:**
    1. *Footwear Type*: Athletic vs dress, protective features, construction method
    2. *Sole Material Analysis*: Primary determinant for heading selection
    3. *Upper Material Analysis*: Secondary determinant for subheading
    4. *Gender & Size Considerations*: Final classification refinement
    
    **Common Chapter 64 Classifications:**
    - 6401: Waterproof footwear with rubber/plastic
    - 6402: Other footwear with rubber/plastic sole
    - 6403: Footwear with leather sole
    - 6404: Footwear with textile upper and rubber/plastic sole
    - 6405: Other footwear
    """

def get_historical_data(product_category, country, product_description, target_gender=None, app_config=None):
    """Get historical data based on product category"""
    if app_config is None:
        from flask import current_app
        app_config = current_app.config
    
    if product_category == 'apparel':
        csv_path = app_config['APPAREL_CSV_PATH']
        col_mapping = app_config['APPAREL_COLUMNS']
    else:  # footwear
        csv_path = app_config['FOOTWEAR_CSV_PATH']
        col_mapping = app_config['FOOTWEAR_COLUMNS']
    
    return format_historical_data_from_csv(
        csv_file_path=csv_path,
        target_country=country,
        target_full_product_description=product_description,
        target_gender=target_gender,
        country_col_hist=col_mapping['country_col'],
        name_col_hist=col_mapping['name_col'],
        name_col2_hist=col_mapping['name_col2'],
        material_col_hist=col_mapping['material_col'],
        construction_col_hist=col_mapping['construction_col'],
        gender_col_hist=col_mapping['gender_col'],
        hs_code_col_hist=col_mapping['hs_code_col'],
        size_col_hist=col_mapping.get('size_col') if product_category == 'footwear' else None
    )

def build_product_description(row, product_category, col_mapping):
    """Build product description based on category"""
    country = str(row.get(col_mapping['country_col'], '')).strip().lower() if col_mapping['country_col'] in row else "unknown"
    product_type = str(row.get(col_mapping['name_col'], '')).strip() if col_mapping['name_col'] in row else ""
    
    if col_mapping['name_col2'] in row and row[col_mapping['name_col2']]:
        product_type += " " + str(row[col_mapping['name_col2']]).strip()
    
    material = str(row.get(col_mapping['material_col'], '')).strip() if col_mapping['material_col'] in row else ""
    construction = str(row.get(col_mapping['construction_col'], '')).strip() if col_mapping['construction_col'] in row else ""
    gender = str(row.get(col_mapping['gender_col'], '')).strip() if col_mapping['gender_col'] in row else ""
    
    desc_parts = [f"Product for {country.upper()}:"]
    if gender:
        desc_parts.append(f"{gender}'s")
    if product_type:
        desc_parts.append(product_type)
    if material:
        if product_category == 'apparel':
            desc_parts.append(f"Material: {material}.")
        else:  # footwear
            desc_parts.append(f"Upper Material: {material}.")
    if construction:
        if product_category == 'apparel':
            desc_parts.append(f"Construction: {construction}.")
        else:  # footwear
            desc_parts.append(f"Sole Material: {construction}.")
    
    # Footwear-specific: add size if relevant
    if product_category == 'footwear' and 'size_col' in col_mapping:
        size = str(row.get(col_mapping['size_col'], '')).strip() if col_mapping['size_col'] in row else ""
        if size and country.upper() in ["EUROPE", "UNITED KINGDOM"]:
            desc_parts.append(f"Size: {size}.")
    
    return " ".join(desc_parts).strip(), country, product_type, material, construction, gender