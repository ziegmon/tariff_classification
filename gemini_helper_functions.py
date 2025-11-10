"""
Gemini Helper Functions Module.
This module contains helper functions for configuring the Gemini API
and generating HS code suggestions using the Gemini model.
"""

from persistence_helper_functions import load_rejected_codes, load_validated_codes, log_token_usage
from data_processing_functions import format_historical_data_from_csv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from flask import Flask
from config import Config
import re
import asyncio 

#__Flask App Configuration___#
app = Flask(__name__)
app.config.from_object(Config)




#___Variables___#
country_col = "tariff_country_description"
name_col = "customs_description"
name_col2 = "product_type"
material_col = "composition"
construction_col = "material_type"
gender_col = "division"
hs_code_col = "tariff_code"

HS_CODE_LENGTH_RULES = app.config["HS_CODE_LENGTH_RULES"]

#___Gemini API Config___# with API key
def configure_genai(api_key):
    """Configures the Google Generative AI (Gemini) API with the provided API key.
     Args:
        api_key (str): The API key for authenticating with the Gemini API.
    Returns:
        genai.GenerativeModel: An instance of the configured GenerativeModel.
    """
    genai.configure(api_key=api_key)
    # using gemini-1.5-flash seems to be enough, does not exceed quota and seems to performa better from 2.0
    # 1.5 can be fine tuned, 2.0 can't
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
    product_type=""
):
    """Generates HS code suggestions using the Gemini model based on the provided product description and context.
    Args:
        model (genai.GenerativeModel): The configured Gemini generative model.
        product_description (str): The description of the product to classify.
        country (str): The target country for classification.
        relevant_chapters (list): List of tuples containing relevant chapter numbers and their content.
        legal_notes (str): Legal notes relevant to the classification.
        classification_guide (str): Classification guide content.
        gri (str, optional): General Rules of Interpretation content. Defaults to.
        rejected_codes_snapshot (list, optional): Snapshot of previously rejected codes. Defaults to None.
        historical_data (str, optional): Historical data content. Defaults to None.
        guidelines (str, optional): Country-specific guidelines. Defaults to None.
        product_type (str, optional): The type of product. Defaults to "".
    Returns:
        str: Formatted response with HS code suggestions and reasoning.
        """
    print(f"[DEBUG] Product Description: {product_description}")

    # this logic checks for previously validated codes first to avoid unnecessary API calls
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

    # prepare rejected codes list for the prompt
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

    rejected_section_for_prompt = ""
    if rejected_hs_codes_for_prompt:
        rejected_section_for_prompt += "\n\nIMPORTANT: DO NOT SUGGEST ANY OF THE FOLLOWING HS CODES (these were previously rejected for this item by a specialist):\n"
        rejected_section_for_prompt += "\n".join(f"- {code_str}" for code_str in rejected_hs_codes_for_prompt)
        rejected_section_for_prompt += "\n\nEnsure that none of your three new suggested codes match any of the HS codes listed directly above."
    else:
        print("DEBUG: Condition 'if rejected_hs_codes_for_prompt:' is FALSE. No rejected codes section to add to prompt.")

    historical_data_string = format_historical_data_from_csv(
        csv_file_path = app.config["CSV_PATH"],
        target_country = country,
        target_full_product_description=product_description,
    )

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

    # Construct the detailed prompt
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