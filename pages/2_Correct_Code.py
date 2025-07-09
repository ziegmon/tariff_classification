import streamlit as st
import re
import base64
from streamlit_extras.switch_page_button import switch_page
from helper_functions_2 import add_bg_from_local, generate_hs_codes, validate_hs_codes, configure_genai, generate_hs_codes_description

# page configuration
page_icon = st.secrets["logo"]

st.set_page_config(
    page_title="HS Code Results",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local(st.secrets["box"])

col1, col2, col3 = st.columns([10, 2, 2])
with col2:
    if st.button("← Back to Start", type = "primary"):
        st.switch_page("app.py")
with col3:
    if st.button("← Back to Codes", type = "primary"):
        st.switch_page("./streamlit_app/pages/1_Suggested_Options.py")



if 'generated_response_description' not in st.session_state or not st.session_state.generated_response_description:
    with st.spinner("Generating HS Codes... (this may take a minute)"):
        result = generate_hs_codes_description(
            st.session_state.model,
            st.session_state.product_description,
            st.session_state.country,
            st.session_state.chapter_content,
            st.session_state.legal_notes,
            st.session_state.correct_code
        )

        st.session_state.generated_response_description = result


response = st.session_state.generated_response_description
st.write(response)