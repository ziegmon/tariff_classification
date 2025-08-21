import streamlit as st
from streamlit_option_menu import option_menu
from helper_functions import check_password, add_bg_from_local, header
# Import the functions from your other files
from bulk_classification import show_bulk_classification_page
from single_product_classification import show_single_product_classification_page


# --- GLOBAL PAGE CONFIGURATION (MUST BE THE VERY FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Footwear HS Code Generator",
    page_icon=st.secrets["logo"],
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE INITIALIZATION ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"



# --- AUTHENTICATION LOGIC ---
# The check_password() function now implicitly handles its own rendering and state update

if not st.session_state.get("authenticated", False):
    st.subheader("Login to Access the Footwear HS Code Classification Playground")
    # check_password() will return True ONLY when the button is pressed AND credentials are correct.
    if check_password(): # This function now directly handles displaying inputs and the button
        st.session_state["authenticated"] = True
        st.session_state["selected_page"] = "Home" # Ensure user starts on Home after login
        st.rerun() # Rerun the script to now display the main app content
    else:
        st.stop() # Stop rendering if not authenticated

else:


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

    # --- Navigation Menu ---
    selected_page_from_menu = option_menu(
        None,
        ["Home", "Single Product Classification", "Bulk Classification"],
        icons=["house", "pen", "upload"],
        orientation="horizontal",
        key="main_navigation_menu",
        default_index=["Home", "Single Product Classification", "Bulk Classification"].index(st.session_state["selected_page"])
    )

    # Update session state with the current page selected from the menu
    st.session_state["selected_page"] = selected_page_from_menu

    # --- Page Content & Navigation Handling ---
    if st.session_state["selected_page"] == "Home":
        header("Welcome to the Footwear HS Code Classification Playground!")
        st.markdown("""
        This application helps you classify footwear products using an AI model.
        Choose an option from the navigation menu above to get started:

        -   **Single Product Classification:** Classify one product at a time.
        -   **Bulk Classification:** Upload a CSV file to classify multiple products and track accuracy.
        """)
        st.info("Your AI model is ready. Select a classification task from the menu above.")

        # --- Logout Button on Home Page ---
        if st.button("Logout", key="home_logout_btn"):

            # Clear all relevant session state keys to force re-authentication
            if "authenticated" in st.session_state:
                del st.session_state["authenticated"]
            # *** CRITICAL FIX: Delete the password_correct flag! ***
            # This is still important as check_password() uses it.
            if "password_correct" in st.session_state:
                del st.session_state["password_correct"]
            # To re-enable browser autofill, we should NOT delete the actual input values (username_input, password_input)
            # here. The browser typically links autofill to the input field's 'name' or 'autocomplete' attribute.
            # Deleting the session state keys for the input values might actually hurt autofill.

            st.session_state["selected_page"] = "Home" # Reset page selection for next login
            st.toast("Logged out successfully!")

            st.rerun()

    elif st.session_state["selected_page"] == "Single Product Classification":
        show_single_product_classification_page()

    elif st.session_state["selected_page"] == "Bulk Classification":
        show_bulk_classification_page()