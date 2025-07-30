import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from helper_functions import add_bg_from_local, check_password

# PDF directory path
PDF_DIRECTORY = "chapter_data"
page_icon = "👟"

if "generated_response" not in st.session_state:
    st.session_state.generated_response = ""

st.set_page_config(
    page_title="Footwear HS Code Generator",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local("images/on_box.png")

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

# Initialize session state
if "menu_option" not in st.session_state:
    st.session_state["menu_option"] = 0

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

if "navigate" not in st.session_state:
    st.session_state["navigate"] = False

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- Authentication Check ---
if not st.session_state["authenticated"]:
    if not check_password():
        st.stop()
    else:
        st.session_state['authenticated'] = True
        st.success("🔐 Authentication successful!")
        st.info("👟 Welcome to the Footwear HS Code Classification System!")

# Navigation menu
selected_page = option_menu(
    None,
    ["Home", "Single Product Classification", "Bulk Classification"],
    icons=["house", "shoe", "upload"],
    orientation="horizontal",
    key="menu_4",
    default_index=["Home", "Single Product Classification", "Bulk Classification"].index(st.session_state["selected_page"])
)

# --- Logout Logic ---
if st.session_state["authenticated"] and selected_page == "Home" and st.session_state["selected_page"] != "Home":
    if st.session_state.get("logout_confirmation"):
        st.session_state["authenticated"] = False
        st.session_state["selected_page"] = "Home"
        del st.session_state["logout_confirmation"]
        st.rerun()
    else:
        st.warning("Are you sure you want to logout?")
        logout_col, cancel_col = st.columns(2)
        with logout_col:
            if st.button("Logout"):
                st.session_state["logout_confirmation"] = True
                st.rerun()
        with cancel_col:
            if st.button("Cancel"):
                st.session_state["selected_page"] = st.session_state.get("last_non_home_page", "Single Product Classification")
                st.rerun()
else:
    if selected_page != "Home":
        st.session_state["last_non_home_page"] = selected_page

# --- Page Navigation ---
if st.session_state["authenticated"]:
    if selected_page == "Single Product Classification":
        switch_page("single product classification")
    elif selected_page == "Bulk Classification":
        switch_page("bulk classification")
    else:
        # Home page content
        st.title("👟 Footwear HS Code Classification System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🦶 Single Product Classification
            Classify individual footwear products with detailed analysis:
            - Interactive product input
            - 3 classification options with reasoning
            - Historical data matching
            - Expert validation tools
            """)
            
        with col2:
            st.markdown("""
            ### 📊 Bulk Classification  
            Process multiple footwear products efficiently:
            - CSV file upload
            - Batch processing
            - Export results
            - Quality control tools
            """)
        
        st.markdown("""
        ---
        ### 🎯 Features
        - **AI-Powered Classification**: Advanced machine learning for accurate HS code determination
        - **Multi-Country Support**: Handles different tariff schedules and regulations
        - **Historical Data Integration**: Learns from previous classifications
        - **Rejection Tracking**: Improves accuracy by avoiding previously rejected codes
        - **Footwear Expertise**: Specialized in Chapter 64 classifications with material and construction logic
        
        ### 👟 Supported Footwear Types
        - Athletic shoes (running, basketball, tennis, training)
        - Dress shoes and formal footwear
        - Boots (hiking, work, fashion)
        - Sandals and casual footwear
        - Safety and protective footwear
        - Children's and specialty footwear
        """)