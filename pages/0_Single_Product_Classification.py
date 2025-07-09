import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from helper_functions_2 import (add_bg_from_local, 
                              highlight,
                              load_pdfs_for_country, 
                              configure_genai, 
                              generate_hs_codes, 
                              find_relevant_chapters)

# PDF directory path
PDF_DIRECTORY = "chapter_data"

page_icon = st.secrets["logo"]

if "generated_response" not in st.session_state:
    st.session_state.generated_response = ""


st.set_page_config(
    page_title="HS Code Generator",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local(st.secrets["box"])

st.markdown("""
<style>
    .stSelectbox [data-baseweb="select"] {
        background-color: #918c6c;
    }
    .stSelectbox [data-baseweb="select"] > div {
        color: black;
    }
</style>
""", unsafe_allow_html=True)

if "menu_option" not in st.session_state:
    st.session_state["menu_option"] = 0

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

if "navigate" not in st.session_state:
    st.session_state["navigate"] = False

# Navigation menu
selected_page = option_menu(
    None,
    ["Home", "Single Product Classification", "Bulk Classification"],
    icons=["house", "pen", "upload"],
    orientation="horizontal",
    key="menu_4",
    default_index=["Home", "Single Product Classification", "Bulk Classification"].index(st.session_state["selected_page"])
)

# Update selected_page in session state only if the user makes a new selection
if selected_page != st.session_state["selected_page"]:
    st.session_state["selected_page"] = selected_page
    st.session_state["navigate"] = True
else:
    st.session_state["navigate"] = False

# --- Page Logic with Navigation Control ---
if st.session_state["navigate"]:
    if st.session_state["selected_page"] == "Single Product Classification":
        print(st.session_state["selected_page"])
        switch_page("single product classification")
    elif st.session_state["selected_page"] == "Bulk Classification":
        print(st.session_state["selected_page"])
        switch_page("Bulk Classification")
    st.session_state["navigate"] = False # Reset the flag after navigation attempt
else:
    if st.session_state["selected_page"] == "Single Product Classification":
        pass # Navigation will happen when "navigate" is True
    elif st.session_state["selected_page"] == "Bulk Classification":
        pass # Navigation will happen when "navigate" is True


# initializing session state variables if they don't exist
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = {}
if "country_list" not in st.session_state:
    # using USA as default option
    st.session_state.country_list = ["canada", "norway", "usa", "switzerland"]


st.title("HS Code Generator")


country_display_labels = {
    "usa": "🇺🇸 USA",
    "canada": "🇨🇦 Canada",
    "norway": "🇳🇴 Norway",
    "switzerland": "🇨🇭 Switzerland"
}

country_options = st.session_state.country_list if st.session_state.country_list else ["usa"]

country_display = [country_display_labels[c] for c in country_options]
country_mapping = {country_display_labels[c]: c for c in country_options}


if not st.session_state.get("country"):
    # clearing all classification-related session state
    for key in [
        "product_description",
        "generated_response",
        "response_generated",
        "chapter_content",
        "legal_notes",
        "classification_guide",
        "chapter_info",
        "legal_notes_info",
        "classification_guide_info",
        "model",
    ]:
        if key in st.session_state:
            del st.session_state[key]

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2, 0.5, 2, 0.5, 3, 0.5, 2, 0.5, 1.5])
with col1:
    country_display_selected = st.selectbox(
        "Country",
        options=country_display,
        index=0 if "USA" in country_display else 0
    )
    country = country_mapping[country_display_selected]

with col3:
    gender = st.selectbox("Gender", options=["Men", "Women", "Kids", "Unisex", "N/A"])

with col5:
    product_type = st.text_input(
        "Product Type (e.g., T-shirt, Jacket, Backpack)",
        key="product_type_input"
    )

with col7:
    construction = st.selectbox("Construction", options=["Knitted", "Woven"])

with col9:
    available_chapters = [ch.split("_chapter_")[1] for ch in st.session_state.processed_pdfs.keys()
        if f"{country}_chapter_" in ch] if st.session_state.processed_pdfs else []
    chapter = st.selectbox(
        "Chapter ",
        options=["auto"] + available_chapters,
        help="If known, otherwise leave as AUTO"
    )

# material input outside form to prevent Enter key submission
st.write("")
material = st.text_input(
    "Material Composition (e.g., 100% Polyester (Mechanical Recycled) - rPES, 80% Polyamide (Nylon) - PA, 20% Elastane - EL, etc.)",
    key="material_input"
)

st.write("")
product_description = f"Product for {country.upper()}: {gender}'s {product_type}. Material: {material}. {construction}"

st.write("Product Description Preview:")
highlight(product_description)

use_enhanced_search = st.checkbox(
    "Use enhanced chapter search",
    value=True,
    help="Automatically identifies the most relevant chapters based on product description"
)

# only handles submission when the button is clicked
with st.form("hs_code_form"):
    submit = st.form_submit_button("Generate HS Codes")

if submit:
    if not country:
        st.error("Please select a country")
    elif not product_type or not material:
        st.error("Please provide the product type and material composition")
    else:
        try:
            # loading PDFs only after form submission
            if "country" in st.session_state and st.session_state.country != country:
                st.session_state.processed_pdfs = {}  # Clear old country PDFs

            st.session_state.country = country
            st.session_state.product_description = product_description
            st.session_state.chapter = chapter
            st.session_state.use_enhanced_search = use_enhanced_search

            if not st.session_state.get("processed_pdfs"):
                with st.spinner(f"Loading HS data for {country.upper()}..."):
                    st.session_state.processed_pdfs = load_pdfs_for_country(PDF_DIRECTORY, country)

            model = configure_genai(st.secrets["API_KEY"])
            st.session_state.model = model

            chapter_content = ""
            if chapter != "auto":
                chapter_key = f"{country}chapter{chapter}"
                if chapter_key in st.session_state.processed_pdfs:
                    chapter_content = st.session_state.processed_pdfs[chapter_key]
                    st.session_state.chapter_info = f"Using chapter {chapter} for {country.upper()} with {len(chapter_content)} characters of content."
                else:
                    st.session_state.chapter_info = f"Chapter {chapter} not found for {country}. Attempting to find the most relevant chapter."

            if not chapter_content and use_enhanced_search:
                relevant_chapter_data = find_relevant_chapters(
                    product_description,
                    country,
                    st.session_state.processed_pdfs
                )
                if relevant_chapter_data:
                    relevant_chapters = [content for _, content in relevant_chapter_data]
                    chapter_keys = [key for key, _ in relevant_chapter_data]
                    chapter_content = "\n\n".join(relevant_chapters[:3])
                    st.session_state.chapter_info = f"Using chapter selection: {', '.join([k.split('_')[-1] for k in chapter_keys[:3]])} for {country.upper()}"
                else:
                    st.session_state.chapter_info = "No relevant chapters found based on product description."

            if not chapter_content:
                relevant_chapters = [content for key, content in st.session_state.processed_pdfs.items()
                                     if f"{country}chapter" in key]
                if relevant_chapters:
                    chapter_content = "\n\n".join(relevant_chapters[:3])
                    st.session_state.chapter_info = f"Using {len(relevant_chapters[:3])} chapters for {country.upper()} with a total of {len(chapter_content)} characters."
                else:
                    st.session_state.chapter_info = f"No chapter information found for {country}. Results may be less accurate."

            legal_notes = st.session_state.processed_pdfs.get(f"{country}_legal_notes", "")
            st.session_state.legal_notes_info = f"Using legal notes for {country.upper()} with {len(legal_notes)} characters of content." if legal_notes else f"No legal notes found for {country.title()}. Results may be less accurate."

            guidelines = st.session_state.processed_pdfs.get(f"{country}_guidelines", "")
            st.session_state.guidelines_info = f"Using guidelines for {country.upper()} with {len(guidelines)} characters of content." if guidelines else f"No guidelines found for {country.title()}. Results may be less accurate."


            classification_guide = st.session_state.processed_pdfs.get(f"{country}_classification_guide", "")
            st.session_state.classification_guide_info = f"Using classification guide for {country.upper()} with {len(classification_guide)} characters of content." if classification_guide else f"No classification guide found for {country.title()}. Results may be less accurate."

            st.session_state.chapter_content = chapter_content
            st.session_state.legal_notes = legal_notes
            st.session_state.classification_guide = classification_guide

            if not chapter_content and not legal_notes and not classification_guide:
                st.error("No tariff data available. Please ensure PDF files are in the correct directory.")
            else:
                with st.spinner("Generating HS codes..."):
                    st.session_state.generated_response = generate_hs_codes(
                        model,
                        product_description,
                        country,
                        chapter_content,
                        legal_notes,
                        classification_guide
                    )
                    st.session_state.response_generated = True

        except Exception as e:
            st.error(f"Error: {str(e)}")


# displaying the HS Codes
if "response_generated" in st.session_state and st.session_state.response_generated:
    st.spinner("In Progress")
    switch_page("suggested options")

# instructions section
with st.expander("How to Use This App"):
    st.markdown("""
    ## Instructions

    1. **Generate HS Codes**:
       - Select the destination country
       - Choose the gender category
       - Enter the product type (e.g., t-shirt, jacket)
       - Provide the material composition
       - Provide the Construction Method
       - Optionally select a specific chapter if you know it
       - Toggle enhanced chapter search if needed
       - Click "Generate HS Codes"

    2. **Results**:
       - The app will display 3 possible HS codes, ordered from most to least likely
       - Each code will be exactly 10 digits long
       - Detailed reasoning will be provided for each option
       - The certainty percentage indicates the model's confidence in each classification

    ## PDF File Naming Convention

    PDF files in the directory should follow this naming pattern:
    - Chapter files: `{country}_chapter_{number}.pdf` (e.g., `usa_chapter_62.pdf`)
    - Legal notes files: `{country}_legal_notes.pdf` (e.g., `usa_legal_notes.pdf`)
    - Classification guide files: `{country}_classification_guide.pdf` (e.g., `canada_classification_guide.pdf`)
    """)
