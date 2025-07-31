import streamlit as st
from helper_functions_2 import (
    add_bg_from_local, highlight, load_pdfs_for_country,
    configure_genai, generate_hs_codes
)

PDF_DIRECTORY = "chapter_data"
page_icon = st.secrets["logo"]

st.set_page_config(
    page_title="HS Code Generator",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local(st.secrets["box"])
st.title("HS Code Generator - FTW")

# Basic Input Form
col1, _, col2, _, col3, _, col4 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
with col1:
    gender = st.selectbox("Gender", ["Men", "Women", "Kids", "Youth", "N/A"])

with col2:
    upper = st.text_input("Upper Composition", key="upper_input")

with col3:
    sole = st.selectbox("Outer Sole Composition", ["Rubber", "Leather", "Plastic", "Other"])

with col4:
    country = st.selectbox("Destination Country", sorted(["Canada", "Switzerland", "Norway", "Australia", "USA", "Brazil", "Hong Kong", "Japan", "New Zealand", "South Korea", "UK"]))

product_description = f"Product: {gender}'s footwear. Upper made of {upper}, outer sole of {sole}. Destination Country: {country}"
highlight(product_description)

# Submit
with st.form("hs_code_form"):
    submit = st.form_submit_button("Generate HS Code")

if submit:
    try:
        with st.spinner("Loading data..."):
            # Only load Chapter 64 and GRI PDFs
            country_key = country.lower()
            processed_pdfs = load_pdfs_for_country(PDF_DIRECTORY, country_key)
            chapter_64 = processed_pdfs.get(f"{country_key}_chapter_64", "")
            gri = processed_pdfs.get(f"{country_key}_gri", "")

        if not chapter_64 and not gri:
            st.error("Chapter 64 and GRI not found. Ensure correct PDFs are present.")
        else:
            with st.spinner("Generating HS Code..."):
                model = configure_genai(st.secrets["API_KEY"])
                response = generate_hs_codes(
                    model, product_description, country_key,
                    chapter_64, gri, ""
                )
                st.success("HS Code generated:")
                st.markdown(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
