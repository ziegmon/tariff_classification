from persistence_helper_functions import load_rejected_codes, load_validated_codes, log_token_usage
from product_specific_functions import (
    get_apparel_prompt_additions, 
    get_footwear_prompt_additions,
    get_historical_data
)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from flask import current_app
import re
import asyncio

def configure_genai(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
    return model

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
    product_type="",
    product_category="apparel",
    app_config=None  # Add this parameter
):
    print(f"[DEBUG] Product Description: {product_description}")
    print(f"[DEBUG] Product Category: {product_category}")

    # If app_config not provided, try to get it from current_app
    if app_config is None:
        try:
            from flask import current_app
            app_config = current_app.config
        except RuntimeError:
            # If we're outside application context, we need app_config passed in
            raise RuntimeError("app_config must be provided when running outside Flask application context")

    # Check for previously validated codes first
    all_validated_entries = load_validated_codes(product_description, country)
    validated_entries_with_code = [e for e in all_validated_entries if e.get("hs_code")]

    if validated_entries_with_code:
        print(f"[DEBUG] Found validated codes for '{product_description}' in {country}. Skipping Gemini call.")
        formatted_response = ""
        for i, entry in enumerate(validated_entries_with_code[:3]):
            option_num = i + 1
            hs_code = entry.get("hs_code", "N/A")
            reasoning = entry.get("reasoning") or "Previously validated by user (code only)."

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

    # Prepare rejected codes list
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
    print(f"[DEBUG] Rejected HS Codes for Prompt: {rejected_hs_codes_for_prompt}")

    rejected_section_for_prompt = ""
    if rejected_hs_codes_for_prompt:
        rejected_section_for_prompt += "\n\nIMPORTANT: DO NOT SUGGEST ANY OF THE FOLLOWING HS CODES (these were previously rejected for this item by a specialist):\n"
        rejected_section_for_prompt += "\n".join(f"- {code_str}" for code_str in rejected_hs_codes_for_prompt)
        rejected_section_for_prompt += "\n\nEnsure that none of your three new suggested codes match any of the HS codes listed directly above."

    # Get historical data using product-specific function
    historical_data_string = get_historical_data(
        product_category=product_category,
        country=country,
        product_description=product_description,
        app_config=app_config  # Pass the config
    )
    # Prepare chapter content
    chapter_content_for_prompt = ""
    if relevant_chapters:
        for chapter_num, chapter_text in relevant_chapters:
            chapter_content_for_prompt += f"\n--- OFFICIAL CHAPTER CONTENT: CHAPTER {chapter_num} ---\n{chapter_text}\n"
    else:
        chapter_content_for_prompt = "\n--- OFFICIAL CHAPTER CONTENT: NOT PROVIDED ---\n"

    # Length rule text
    length_rule_text = "Ensure the HS Code is the correct length as specified in the official documents."
    country_lower = country.lower()

    if country_lower == 'switzerland':
        length_rule_text = (
            "- HS CODE STRUCTURE (SWITZERLAND): All suggested HS codes for Switzerland MUST follow a strict format: "
            "an 8-digit base code, followed by the mandatory, unchanging statistical suffix of '-000'. "
            "The final output must ALWAYS look like this: xxxx.xx.xx-000. NO EXCEPTIONS."
        )
    elif country_lower in app_config['HS_CODE_LENGTH_RULES']:  # Changed from current_app.config
        required_length = app_config['HS_CODE_LENGTH_RULES'][country_lower]
        length_rule_text = (
            f"- HS CODE LENGTH: All suggested HS codes for {country.upper()} MUST be "
            f"exactly {required_length} digits long (not counting the periods). "
            "This is a non-negotiable rule. Verify the final digit count. NO EXCEPTIONS."
        )

    length_rule_prompt_block = f"""
CRITICAL RULE - HS CODE STRUCTURE:
{length_rule_text}
"""

    # Get product-specific prompt additions
    if product_category == 'apparel':
        product_specific_rules = get_apparel_prompt_additions(
            country, product_description, historical_data_string, rejected_hs_codes_for_prompt
        )
    else:  # footwear
        product_specific_rules = get_footwear_prompt_additions(
            country, product_description, historical_data_string, rejected_hs_codes_for_prompt
        )

    # Build the main prompt
    prompt = f"""
CONTEXT & RESOURCES:
- Product Description: {product_description}
- Product Category: {product_category.upper()}
- Target Country: {country.upper()}
- Legal Notes: {legal_notes}
- Country Guidelines: {guidelines}
- Classification Guide: {classification_guide}
- General Rules of Interpretation (GRI): {gri}
- OFFICIAL CHAPTER CONTENT: {chapter_content_for_prompt}
- Previously Rejected HS Codes (DO NOT USE): {", ".join(rejected_hs_codes_for_prompt)}
- Previously Excluded Chapters/Sections (DO NOT USE CODES FROM HERE): {rejected_section_for_prompt}

{length_rule_prompt_block}

**HISTORICAL DATA (SIMILAR PRODUCTS & CLASSIFICATIONS):**
{historical_data_string}

---

{product_specific_rules}

**ULTRA-CRITICAL CLASSIFICATION RULES (MANDATORY COMPLIANCE):**

1.  **ABSOLUTE CODE VALIDITY:** ONLY propose HS codes (including all digits and statistical suffixes) that appear *VERBATIM* in the `OFFICIAL CHAPTER CONTENT`. Do not invent, assume, or truncate.
2.  **REJECTED CODES ARE FORBIDDEN:** Before providing your final answer, you MUST perform a final verification. Compare every potential code against the `Previously Rejected HS Codes for this Item` list. If a code is on this list, it is **FORBIDDEN**.
3.  **EXCLUDED CHAPTERS ARE FORBIDDEN:** DO NOT propose codes from chapters/sections listed in `Previously Excluded Chapters/Sections`.
4.  **STATISTICAL SUFFIX PRIORITY:** Use the most specific, applicable statistical suffixes provided in `OFFICIAL CHAPTER CONTENT`.
6.  **SINGLE CORRECT CODE PRINCIPLE:** Every product has one single, most correct HS code. OPTION 1 MUST be your highest confidence classification.

* **NO DEFAULT CODES:** Do not assume a general code like `.00` or `.0000` exists if it is not explicitly written in the `OFFICIAL CHAPTER CONTENT`.
* **MANDATORY FALLBACK:** If you cannot find a complete, verbatim national tariff code in the provided documents, you MUST output `NO-CODE-FOUND` for that option.

**HISTORICAL DATA USAGE PROTOCOL:**
* **For Same Country & Product:** Treat the *full* historical code as a strong candidate, but not ground truth, giving special weight to the Gender of the product.
* **REJECTION OVERRIDES HISTORY:** If a code suggested by `HISTORICAL DATA` is also present in the `Previously Rejected HS Codes for this Item` list, the rejection list ALWAYS wins.
* **For Different Country:** Focus ONLY on the first six digits (international portion) of the historical code as a guide for Chapter/Heading/Subheading.

**TASK:**
Based *exclusively and meticulously* on the provided content and adhering to ALL critical rules, determine the *THREE most likely HS codes + statistical suffixes* for the product.

1.  **Prioritize Historical Data:** If a highly similar and valid historical code exists, propose it as **OPTION 1** with high certainty.
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
    * The sum of likelihood for the 3 proposed codes should be 100%.

**MANDATORY FINAL VERIFICATION (DOUBLE-CHECK BEFORE OUTPUT):**
Before generating your final output, perform these absolute checks on EACH proposed OPTION:
1.  **HS Code Length:** Confirm the HS code precisely matches the required length for {country.upper()}.
2.  **No Rejected Codes:** ABSOLUTELY ENSURE that none of the proposed codes are present in `Previously Rejected HS Codes for this Item`.
3.  **Material Predominance:** Does the classification correctly reflect the predominant material by weight?

---

**FORMAT (Strictly Adhere):**

### OPTION 1: [HS code + suffix] - XX% certainty

#### PRODUCT DESCRIPTION:
[Re-state the product description based *solely* on the information received, focusing on classification-relevant details.]

#### REASONING STRUCTURE:
Use this detailed structure for justification:

1.  *GRI Application*: Identify and apply ALL relevant General Rules of Interpretation (GRI 1-6, in order). Explain how each applied GRI leads to the decision.
2.  *Historical Data Consideration*: Explicitly state how historical data influenced (or didn't influence) this option.
3.  *Chapter & Section Fit*: State chosen Chapter/Section. Confirm inclusion based on material, construction, use, and notes.
4.  *Heading & Subheading Determination*: Justify 4-digit heading and 6-digit subheading using texts and legal notes.
5.  *National Tariff Line Determination*: Explain how specific suffixes/national digits were determined *VERBATIM* from `OFFICIAL CHAPTER CONTENT`. 
6.  *Exclusions & Verifications*: Explicitly rule out other plausible but incorrect classifications.
7.  ***Internal Verification Check***: **Before finalizing this option, state the outcome of the following checks:**
    * **Check 1:** Does the product type match the classification chapter?
    * **Check 2 (Code in Docs):** Does this exact HS code `[HS code + suffix]` exist verbatim in the provided `OFFICIAL CHAPTER CONTENT`? If it IS found, quote the exact line item.
    * **Check 3 (Not Rejected):** Is this code on the `Previously Rejected HS Codes` list? State Yes/No and confirm it is not.
    * **Check 4 (Material Predominance):** Does the classification correctly reflect the predominant material by weight?

#### LEGAL BASIS:
Cite specific GRI rules, Section/Chapter notes, and heading/subheading texts from provided documents. Quote directly or paraphrase precisely.

---

**HARDCODED CLASSIFICATION RULES (Apply Rigorously):**

* **No "Same as OPTION X":** Each option requires full, independent reasoning.
* If the relevant documentation shows hierarchy of codes with hyphens but there is no value for that level, look for the following levels of hierarchy.
* The DEFINING material for classification is the one with the **HIGHEST PERCENTAGE** when a product is composed of multiple materials.
* **HS Code Structure:**
    * Apply `GRI` from `{gri}`.
    * Use chapter-specific `LEGAL NOTES` from `{legal_notes}`.
    * DO NOT add suffixes/sub-codes unless *VERBATIM* present in `OFFICIAL CHAPTER CONTENT`.
    * Use the *longest HS Code version explicitly shown  in `OFFICIAL CHAPTER CONTENT`.
    * Full code for `{country.upper()}` must include *all visible digits* (e.g., 6103.43.00.15 includes .00; 6109.9000 includes 000). Do not skip intermediate levels.
    * After 4 digits, every pair of digits is separated by `.` (e.g., 6109.10.20, NOT 61091020).
    * NEVER use spaces within HS Code digits.
    * **Switzerland Specific:** For Switzerland, provide an 8-digit HS Code. Add `*-000*` as the statistical suffix ONLY IF the 8-digit code exists in `OFFICIAL CHAPTER CONTENT` and no more specific 11-digit Swiss suffix is listed/applicable.

---

In the country specific guidelines (`{guidelines}`), you are provided an example classification that follows an ideal structure. Model your response structure after this example.

**FINAL REMINDERS:**
- Do not propose any code segment that does not exist *verbatim* in the official documentation.
- Your primary directive is accuracy based *only* on the provided documents and rules.
- Double-check all suggestions: 
     -- confirm verbatim existence in Tariff Schedule 
     -- verify that the code is NOT present in the *Previously Rejected HS Codes*.
"""

    try:
        # Define safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        generation_config = {
            "temperature": 0.0,
            "top_p": 0.5,
            "top_k": 15,
            "max_output_tokens": 8192,
        }

        # Add timeout to the API call
        response = await asyncio.wait_for(
            model.generate_content_async(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            ),
            timeout=120.0
        )
        print(response)

        input_tokens = 0
        output_tokens = 0
        
        # Get input tokens
        try:
            input_tokens = model.count_tokens(prompt).total_tokens
        except Exception as e:
            print(f"Warning: Could not count input tokens. {e}")

        # Get output tokens from response metadata
        if response and hasattr(response, 'usage_metadata'):
            output_tokens = response.usage_metadata.candidates_token_count

        # Log the usage
        log_token_usage(country, product_type, input_tokens, output_tokens)

        # Check if the response was blocked
        if not response.parts:
            print(f"--- RESPONSE BLOCKED FOR PRODUCT: {product_description} ---")
            if response.prompt_feedback:
                print(f"    Finish Reason: {response.prompt_feedback.block_reason}")
                print(f"    Safety Ratings: {response.prompt_feedback.safety_ratings}")
            print("----------------------------------------------------")
            return ""

        token_count = model.count_tokens(prompt).total_tokens
        print(f"The prompt contains {token_count} tokens.")

        validated_response = response.text or ""
        return validated_response
        
    except asyncio.TimeoutError:
        print(f"An API call timed out for product: {product_description}")
        return """
### OPTION 1: TIMEOUT - 0% certainty
#### REASONING STRUCTURE:
The request to the generative AI service timed out after 120 seconds. This might be due to high server load.
"""
    except Exception as e:
        import traceback
        print(f"An exception occurred in generate_hs_codes for product '{product_description}': {str(e)}")
        traceback.print_exc()
        return """
### OPTION 1: EXCEPTION - 0% certainty
#### REASONING STRUCTURE:
An unexpected error occurred during API call. Please check the console logs.
"""