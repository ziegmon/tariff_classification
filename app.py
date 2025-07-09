import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu

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

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


# Navigation menu
selected_page = option_menu(
    None,
    ["Home", "Single Product Classification", "Bulk Classification"],
    icons=["house", "pen", "upload"],
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
        print(selected_page)
        switch_page("single product classification")
    elif selected_page == "Bulk Classification":
        print(selected_page)
        switch_page("bulk classification")