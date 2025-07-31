import streamlit as st
import pandas as pd
from helper_functions_2 import (
    add_bg_from_local, configure_genai
)

page_icon = st.secrets["logo"]

st.set_page_config(
    page_title="Leaderboard",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local(st.secrets["box"])
st.title("FTW Leaderboard")

# File uploader
csv_file = st.file_uploader("Upload labeled CSV", type=["csv"])

if st.button("Submit", type="primary") and csv_file:
    df = pd.read_csv(csv_file)

    # Validate necessary columns
    required_columns = {"division", "material_composition", "tariff_country_description", "tariff_code"}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")

    else:
        try:
            with st.spinner("Setting up model..."):
                model = configure_genai(st.secrets["API_KEY"])

            # Store per-country PDFs so we don't reload them multiple times
            country_pdfs = {}
            results = []
            