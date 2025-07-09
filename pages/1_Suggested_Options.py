import streamlit as st
import re
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from helper_functions_2 import (add_bg_from_local, 
                              generate_hs_codes, 
                              save_rejected_code)

# page configuration
page_icon = st.secrets["logo"]

st.set_page_config(
    page_title="HS Code Results",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

add_bg_from_local(st.secrets["box"])

# custom CSS
custom_css = """
<style>
.stExpander div[data-testid="stVerticalBlock"] {
    background-color: rgba(233, 220, 193, 0.3); 
    border-radius: 5px;
    padding: 10px;
}

.stInfo {
    background-color: rgba(233, 220, 193, 0.3); 
    border-radius: 5px;
    padding: 10px;
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)


# if "menu_option" not in st.session_state:
#     st.session_state["menu_option"] = 0

# if "selected_page" not in st.session_state:
#     st.session_state["selected_page"] = "Home"

# if "navigate" not in st.session_state:
#     st.session_state["navigate"] = False

# # Navigation menu
# selected_page = option_menu(
#     None,
#     ["Home", "Single Product Classification", "Bulk Classification"],
#     icons=["key", "pen", "upload"],
#     orientation="horizontal",
#     key="menu_4",
#     default_index=["Home", "Single Product Classification", "Bulk Classification"].index(st.session_state["selected_page"])
# )

# # Update selected_page in session state only if the user makes a new selection
# if selected_page != st.session_state["selected_page"]:
#     st.session_state["selected_page"] = selected_page
#     st.session_state["navigate"] = True
# else:
#     st.session_state["navigate"] = False

# # --- Page Logic with Navigation Control ---
# if st.session_state["navigate"]:
#     if st.session_state["selected_page"] == "Single Product Classification":
#         print(st.session_state["selected_page"])
#         switch_page("single product classification")
#     elif st.session_state["selected_page"] == "Bulk Classification":
#         print(st.session_state["selected_page"])
#         switch_page("Bulk Classification")
#     st.session_state["navigate"] = False # Reset the flag after navigation attempt
# else:
#     # Optionally display content based on the current selected_page without navigation
#     if st.session_state["selected_page"] == "Home":
#         pass # Already handled in the navigate block or initial display
#     elif st.session_state["selected_page"] == "Single Product Classification":
#         pass # Navigation will happen when "navigate" is True
#     elif st.session_state["selected_page"] == "Bulk Classification":
#         pass # Navigation will happen when "navigate" is True


if "menu_option" not in st.session_state:
    st.session_state["menu_option"] = 0 # This seems unused in the provided logic, but kept for consistency

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home" # Default starting page

if "navigate" not in st.session_state:
    st.session_state["navigate"] = False

# --- Determine the highlight for the option_menu ---
# Define the menu items
menu_items = ["Home", "Single Product Classification", "Bulk Classification"]
current_page_from_session = st.session_state.get("selected_page", "Home") # Get current page, default to Home

# If the actual current page is "Suggested Options", we want to highlight "Single Product Classification"
if current_page_from_session == "Suggested Options":
    default_menu_selection = "Single Product Classification"
else:
    # Ensure the current page (if in menu_items) is highlighted, otherwise default to "Home"
    if current_page_from_session in menu_items:
        default_menu_selection = current_page_from_session
    else:
        default_menu_selection = "Home"


try:
    default_idx = menu_items.index(default_menu_selection)
except ValueError:
    # Fallback if default_menu_selection is somehow not in menu_items (should not happen with above logic)
    default_idx = 0
    st.warning(f"Warning: Default menu selection '{default_menu_selection}' not found in menu items. Defaulting to 'Home'.")


# Navigation menu
selected_by_option_menu = option_menu(
    None,
    options=menu_items,
    icons=["house", "search", "upload"], # Changed icons for variety, adjust as needed
    orientation="horizontal",
    key="menu_4", # Using a unique key is good practice
    default_index=default_idx
)

# --- Logic to handle page switching and state updates ---

# If the user clicked a different option in the menu
if selected_by_option_menu != default_menu_selection: 
    st.session_state["selected_page"] = selected_by_option_menu
    st.session_state["navigate"] = True # Trigger navigation
else:
    if not st.session_state.get("navigate", False): 
        st.session_state["navigate"] = False


# --- Page Logic with Navigation Control ---
# This part now primarily reacts to st.session_state["navigate"] being True
if st.session_state.get("navigate", False): # Use .get for safety
    current_target_page = st.session_state["selected_page"]
    st.session_state["navigate"] = False  # Reset the flag immediately after deciding to navigate

    if current_target_page == "Single Product Classification":
        switch_page("single product classification")
    elif current_target_page == "Bulk Classification":
        switch_page("Bulk Classification")
    elif current_target_page == "Home":
        switch_page("app") 


# back to start button
col1, col2, col3 = st.columns([6, 5, 3])
with col2:
    if st.button("New Classification", type="secondary"):
        for key in ['product_description', 'country', 'generated_response',
                    'correct_code', 'chapter_info', 'legal_notes_info']:
            if key in st.session_state:
                del st.session_state[key]
        switch_page("single product classification")

# File path to store rejected codes
REJECTED_CODES_FILE = "rejected_classifications.json"

if 'rejected_codes' not in st.session_state:
    st.session_state.rejected_codes = []

if 'product_description' not in st.session_state or 'country' not in st.session_state:
    st.error("Please start from the main page to input product details.")
    if st.button("Go to Main Page"):
        switch_page("app")
else:
    title = re.sub(r'^.*?(?=:)', '', st.session_state.product_description, 1)

    st.header(f"HS Code Results for {title}. {st.session_state.country.title()}", anchor=False, divider="blue")

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    .bold-font {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    product = st.session_state.product_description
    st.markdown(f'<p class="big-font">{product}</p>', unsafe_allow_html=True)

    if 'generated_response' not in st.session_state or not st.session_state.generated_response:
        with st.spinner("Generating HS Codes... (this may take a minute)"):
            result = generate_hs_codes(
                st.session_state.model,
                st.session_state.product_description,
                st.session_state.country,
                st.session_state.chapter_content,
                st.session_state.legal_notes,
                st.session_state.classification_guide,
                st.session_state.gri
            )
            st.session_state.generated_response = result

    st.divider()

    options = st.session_state.generated_response.split('OPTION ')[1:]
    cols = st.columns(len(options))

    for i, option in enumerate(options):
        lines = option.strip().split('\n', 1)
        title_line = lines[0].strip()
        start_index = title_line.index(": ") + 2
        end_index = title_line.index(" -", start_index)
        extracted_value = title_line[start_index:end_index]
      
        details = lines[1].strip() if len(lines) > 1 else "No explanation provided."

        with cols[i]:
            st.markdown(f"### OPTION {title_line}")

            with st.expander("Click to see explanation and legal reasoning"):
                st.write(details)

            if st.button(f"❌ Mark OPTION {i+1} as incorrect", key=f"incorrect_{i+1}"):
                if title_line not in st.session_state.rejected_codes:
                    st.session_state.rejected_codes.append(title_line)
                    save_rejected_code(
                        st.session_state.product_description,
                        st.session_state.country,
                        extracted_value
                    )

    # Check if all options are rejected and regenerate
    if len(st.session_state.rejected_codes) >= 3:
        st.warning("All 3 options marked as incorrect. Regenerating suggestions...")
        rejected_snapshot = st.session_state.rejected_codes.copy()
        st.session_state.rejected_codes = []

        with st.spinner("Re-generating HS Codes with feedback..."):
            st.session_state.generated_response = generate_hs_codes(
                st.session_state.model,
                st.session_state.product_description,
                st.session_state.country,
                st.session_state.chapter_content,
                st.session_state.legal_notes,
                st.session_state.classification_guide,
                rejected_snapshot
            )
        switch_page("suggested options")
