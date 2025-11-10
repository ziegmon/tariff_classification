"""
Main Flask Application for HS Code Classification.

This application provides endpoints for single and bulk product classification
using HS codes. It integrates with a GenAI model for generating HS code suggestions
based on product descriptions and relevant tariff documentation.
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import pandas as pd
from datetime import datetime
import time
import re
import pathlib 
import asyncio
import collections 
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from config import Config
from concurrent.futures import ThreadPoolExecutor


from persistence_helper_functions import (load_validated_codes, 
                                          load_rejected_codes, 
                                          save_rejected_code, 
                                          save_validated_code, 
                                          save_bulk_classification_results, 
                                          save_processing_time, 
                                          save_final_selected_results)
from document_handling_helper_functions import (load_all_documents, 
                                                find_relevant_chapters, 
                                                load_text_files_for_country)
from gemini_helper_functions import configure_genai, generate_hs_codes
from data_processing_functions import log_interaction_event, extract_hs_codes
from metrics_handlings_functions import log_detailed_accuracy
from metrics_service import get_dashboard_data
from helper_functions_ftw import load_all_pdf_data, configure_genai as configure_genai_footwear, generate_hs_codes_ftw as generate_hs_codes_footwear, find_relevant_chapters as find_relevant_chapters_footwear, extract_hs_codes as extract_hs_codes_footwear, format_historical_data_from_csv


def run_async_task(task, *args):
    """
    Runs an async function in a new event loop on a separate thread.
    This is used to bridge the sync Flask world with our async processing.
    """
    asyncio.run(task(*args))

# File paths for logs
ALL_BULK_RESULTS_FILE = str(pathlib.Path(__file__).parent / "data/logs/all_bulk_results.csv")
PROCESSING_TIMES_FILE = str(pathlib.Path(__file__).parent / "data/logs/processing_times.csv")
TOKEN_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/token_usage_log.csv")
INTERACTION_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/interaction_log.csv")
ACCURACY_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/accuracy_log_combined.csv")

# --- Flask App Setup ---
app = Flask(__name__)
app.config.from_object(Config)


# Initializes model and doc_cache once
# API Key Setup
with app.app_context():
    app.doc_cache, _ = load_all_documents(app.config['PDF_DIRECTORY'])
    if not app.doc_cache:
        print("ERROR: Failed to load any PDF documentation!")



@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple login page to restrict access.
     In a production app, use secure authentication methods.
     """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == app.config['AUTH_USERNAME'] and password == app.config['AUTH_PASSWORD']:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Incorrect username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs out the user by clearing the session.
     In a production app, ensure proper session management.
     """
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# --- Home Page ---
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')


class AsyncRateLimiter:
    """
    An asynchronous rate limiter that enforces a rate limit using a token bucket algorithm.
    Attributes:
        rate_limit (int): Maximum number of requests allowed in the specified period.
        period_seconds (int): Time period in seconds for the rate limit.
    Methods:
        acquire(): Waits if necessary to ensure the rate limit is not exceeded.
    """
    def __init__(self, rate_limit: int, period_seconds: int = 60):
        self.rate_limit = rate_limit
        self.period_seconds = period_seconds
        self._timestamps = collections.deque()

    async def acquire(self):
        """Waits if necessary to ensure the rate limit is not exceeded."""
        while True:
            now = time.monotonic()
            
            # Removing timestamps older than the current period
            while self._timestamps and self._timestamps[0] <= now - self.period_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) < self.rate_limit:
                self._timestamps.append(now)
                break
            
            # Calculating wait time until the oldest request expires
            wait_time = self._timestamps[0] - (now - self.period_seconds)
            await asyncio.sleep(wait_time)


# --- Single Product Classification ---
@app.route('/single_classification', methods=['GET', 'POST'])
async def single_classification():
    """Handles single product classification requests.
     On GET: renders the input form.
     On POST: processes the form data, generates HS code suggestions, and redirects to results."""
    if request.method == 'POST':
        country = request.form['country'].lower()
        gender = request.form['gender']
        product_type = request.form['product_type']
        construction = request.form['construction']
        material = request.form['material']
        use_enhanced_search = 'use_enhanced_search' in request.form
        chapter_input = request.form.get('chapter', 'auto')

        product_description = f"Product for {country.upper()}: {gender}'s {product_type}. Material: {material}. {construction}"

        # Checking for validated codes first (no API call needed)
        validated_codes = load_validated_codes(product_description, country)
        if validated_codes:
            # If validated codes exist, uses the first one as the primary suggestion
            # and formats it to match the expected structure of generate_hs_codes output
            first_validated_code = validated_codes[0].get('hs_code')
            first_validated_reasoning = validated_codes[0].get('reasoning', 'Previously validated by a specialist.')

            # Construct a mock response that extract_hs_codes can parse
            mock_generated_response = f"""
                ### OPTION 1: {first_validated_code} - 100% certainty (Validated)

                #### PRODUCT DESCRIPTION:
                {product_description}

                #### REASONING STRUCTURE:
                {first_validated_reasoning}

                #### LEGAL BASIS:
                Previously validated.
            """
            # Adds placeholders for Option 2 and 3 if needed by extract_hs_codes
            if len(validated_codes) > 1:
                mock_generated_response += f"\n### OPTION 2: {validated_codes[1].get('hs_code', 'N/A')} - 100% certainty (Validated)\n#### PRODUCT DESCRIPTION:\n{product_description}\n#### REASONING STRUCTURE:\n{validated_codes[1].get('reasoning', 'Previously validated.')}\n#### LEGAL BASIS:\nPreviously validated."
            if len(validated_codes) > 2:
                mock_generated_response += f"\n### OPTION 3: {validated_codes[2].get('hs_code', 'N/A')} - 100% certainty (Validated)\n#### PRODUCT DESCRIPTION:\n{product_description}\n#### REASONING STRUCTURE:\n{validated_codes[2].get('reasoning', 'Previously validated.')}\n#### LEGAL BASIS:\nPreviously validated."

            session['last_generated_response'] = mock_generated_response
            session['last_product_description'] = product_description
            session['last_country'] = country
            flash("Found a previously validated HS code for this product!", 'success')
            return redirect(url_for('suggested_options'))


        # If no validated codes are found, proceeds with generation
        # Loads country-specific docs for the current request
        processed_docs = app.doc_cache.get(country)
        if not processed_docs:
            flash(f"No tariff data available for {country.title()}. Please ensure PDF files are in the correct directory.", 'danger')
            return redirect(url_for('single_classification'))

        chapter_content = ""
        if chapter_input != "auto":
            chapter_key = f"chapter_{chapter_input}"
            chapter_content = processed_docs.get(chapter_key, "")

        if not chapter_content and use_enhanced_search:
            relevant_chapter_data = find_relevant_chapters(
                product_description, country, processed_docs
            )
            if relevant_chapter_data:
                chapter_content = "\n\n".join([content for _, content in relevant_chapter_data][:3])

        legal_notes = processed_docs.get("legal_notes", "")
        classification_guide = processed_docs.get("classification_guide", "")
        gri = processed_docs.get("gri", "")
        guidelines = load_text_files_for_country(app.config['PDF_DIRECTORY'], country).get(f"{country}_guidelines", "")

        rejected_codes_snapshot = load_rejected_codes(product_description, country) # To inform generation of previously rejected codes

        model = configure_genai(app.config['API_KEY']) # Configure model for single classification
        generated_response = await generate_hs_codes(
            model,
            product_description,
            country,
            relevant_chapter_data if use_enhanced_search else chapter_content, 
            legal_notes,
            classification_guide,
            gri=gri,
            guidelines=guidelines,
            rejected_codes_snapshot=rejected_codes_snapshot
        )

        session['last_generated_response'] = generated_response # Store for retrieval in results page
        session['last_product_description'] = product_description # Store for context
        session['last_country'] = country # Store for context

        return redirect(url_for('suggested_options')) # Redirect to results page

    # GET request: render the input form
    country_options = sorted(list(app.doc_cache.keys()))
    gender_options = ["Men", "Women", "Kids", "Unisex", "N/A"]
    construction_options = ["Knitted", "Woven"]
    available_chapters = []
    # If a country is selected, populate chapters (for GET request)
    if country_options:
        default_country_docs = app.doc_cache.get(country_options[0])
        if default_country_docs:
             available_chapters = sorted([ch.split("_chapter_")[1] for ch in default_country_docs.keys() if "_chapter_" in ch])

    return render_template('single_product.html',
                           country_options=country_options,
                           gender_options=gender_options,
                           construction_options=construction_options,
                           available_chapters=available_chapters)

@app.route('/footwear_bulk_classification', methods=['GET', 'POST'])
async def footwear_bulk_classification():
    """Handles bulk footwear classification requests.
     On GET: renders the upload form or active batch status.
     On POST: processes the uploaded CSV file and starts background processing.
     """
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                df_input = pd.read_csv(file)
                df_input = df_input.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})

                batch_id = f"footwear_bulk_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
                app.bulk_processing_state[batch_id] = {
                    'input_df': df_input,
                    'processed_results': [],
                    'selection_status': {row['idx']: {"status": "pending", "data": {}} for _, row in df_input.iterrows()},
                    'regenerate_queue': set(),
                    'total_rows': len(df_input),
                    'processed_count': 0,
                    'status': 'processing',
                    'error_message': None
                }

                session['current_footwear_batch_id'] = batch_id  # Keep footwear batches separate

                # --- Run async footwear processing in background thread ---
                thread = threading.Thread(
                    target=run_async_task,
                    args=(process_bulk_footwear_flask, batch_id, app.config['API_KEY'])
                )
                thread.start()

                # ✅ redirect while batch_id exists
                return redirect(url_for('review_bulk_results', batch_id=batch_id))

            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
                print(f"ERROR: Footwear bulk classification failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV.', 'danger')
            return redirect(request.url)

    # ✅ Handle normal GET request (no file uploaded yet)
    active_batch_id = session.get('current_footwear_batch_id')
    if active_batch_id and active_batch_id in app.bulk_processing_state:
        return render_template('footwear_bulk.html', active_batch_id=active_batch_id)

    # Render the upload form for new session
    return render_template('footwear_bulk.html', active_batch_id=None)

@app.route('/suggested_options', methods=['GET', 'POST'])
async def suggested_options():
    """Displays suggested HS code options and handles user feedback.
     On GET: renders the suggested options.
     On POST: processes user feedback (marking options as incorrect) and regenerates suggestions if needed.
     """
    generated_response = session.get('last_generated_response')
    product_description = session.get('last_product_description')
    country = session.get('last_country')

    if not generated_response:
        flash("No HS codes generated. Please perform a classification first.", 'warning')
        return redirect(url_for('single_classification'))

    options_data = []
    # Updated regex to handle "(Validated)" tag in the option title line
    raw_options = re.split(r'### OPTION \d+:', generated_response)[1:] # Split by the option header
    for i, option_text in enumerate(raw_options):
        lines = option_text.strip().split('\n', 1)
        title_line = lines[0].strip()
        hs_code_match = re.search(r'([0-9\.\s]+(?:-[0-9]+)?) - (\d+)% certainty', title_line) # Adjusted regex for HS code and certainty
        
        hs_code = hs_code_match.group(1).strip() if hs_code_match else "N/A"
        certainty = int(hs_code_match.group(2)) if hs_code_match else 0
        
        # Check for (Validated) tag
        is_validated = "(Validated)" in title_line

        details = lines[1].strip() if len(lines) > 1 else "No explanation provided."
        
        options_data.append({
            'option_num': i + 1,
            'title_line': title_line,
            'hs_code': hs_code,
            'certainty': certainty,
            'details': details,
            'is_validated': is_validated
        })

    # Handle 'Mark as Incorrect' submission
    if request.method == 'POST' and request.form.get('action') == 'mark_incorrect':
        action = request.form.get('action')
        option_num_str = request.form.get('option_num')

        if action == 'mark_incorrect' and option_num_str:
            option_num = int(option_num_str)
            
            # Find the HS code for the rejected option
            rejected_hs_code = next((opt['hs_code'] for opt in options_data if opt['option_num'] == option_num), None)
            
            if rejected_hs_code:
                save_rejected_code(product_description, country, rejected_hs_code)
                # Keep track of rejected options in session to disable buttons in UI
                if 'rejected_options_for_current_product' not in session:
                    session['rejected_options_for_current_product'] = []
                session['rejected_options_for_current_product'].append(option_num)
                
                # Check if all options are rejected
                if len(session['rejected_options_for_current_product']) >= len(options_data):
                    flash("All options marked as incorrect. Regenerating suggestions...", 'info')
                    rejected_snapshot = [opt['hs_code'] for opt in options_data if opt['option_num'] in session['rejected_options_for_current_product']]
                    
                    # Clear session rejected options for new generation
                    session.pop('rejected_options_for_current_product', None)

                    # Re-generate HS codes
                    # Need to retrieve chapter_content, legal_notes, classification_guide, gri, guidelines again
                    processed_docs = app.doc_cache.get(country, {})
                    relevant_chapter_data = find_relevant_chapters(product_description, country, processed_docs)
                    legal_notes = processed_docs.get("legal_notes", "")
                    classification_guide = processed_docs.get("classification_guide", "")
                    gri = processed_docs.get("gri", "")
                    guidelines = load_text_files_for_country(app.config['PDF_DIRECTORY'], country).get(f"{country}_guidelines", "")

                    model = configure_genai(app.config['API_KEY'])
                    regenerated_response = await generate_hs_codes(
                        model,
                        product_description,
                        country,
                        relevant_chapter_data,
                        legal_notes,
                        classification_guide,
                        gri=gri,
                        guidelines=guidelines,
                        rejected_codes_snapshot=rejected_snapshot
                    )
                    session['last_generated_response'] = regenerated_response
                    return redirect(url_for('suggested_options'))
                
                flash(f'Option {option_num} marked as incorrect.', 'success')
                # If not regenerating, just re-render the page to show the update
                return redirect(url_for('suggested_options')) # Rerun is like redirecting to self in Flask

    rejected_options_for_current_product = session.get('rejected_options_for_current_product', [])
    return render_template('suggested_options.html',
                           product_description=product_description,
                           country=country,
                           options_data=options_data,
                           rejected_options_for_current_product=rejected_options_for_current_product)

# Example of an AJAX endpoint for regeneration in bulk (more complex)
@app.route('/regenerate_bulk_item/<int:idx>', methods=['POST'])
def regenerate_bulk_item(idx):
    """AJAX endpoint to regenerate HS code suggestions for a specific bulk item.
    This would involve loading product info, rejected codes, calling generate_hs_codes
    and then returning JSON with new options for the specific item to update the frontend via JS
    """
    return jsonify({'status': 'success', 'new_options': []}) # Placeholder


app.bulk_processing_state = {} # Stores results keyed by a unique batch ID

@app.route('/bulk_classification', methods=['GET', 'POST'])
async def bulk_classification():
    """Handles bulk classification requests.
     On GET: renders the upload form or active batch status.
     On POST: processes the uploaded CSV file and starts background processing."""
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                df_input = pd.read_csv(file)
                df_input = df_input.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})

                batch_id = f"bulk_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
                app.bulk_processing_state[batch_id] = {
                    'input_df': df_input,
                    'processed_results': [],
                    'selection_status': {row['idx']: {"status": "pending", "data": {}} for _, row in df_input.iterrows()},
                    'regenerate_queue': set(),
                    'total_rows': len(df_input),
                    'processed_count': 0,
                    'status': 'processing',
                    'error_message': None
                }
                
                session['current_batch_id'] = batch_id

                # --- Use threading to run the async task ---
                thread = threading.Thread(
                    target=run_async_task,
                    args=(process_bulk_data_flask, batch_id, app.config['API_KEY'])
                )
                thread.start()

                return redirect(url_for('processing_status_page', batch_id=batch_id))

            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
                print(f"ERROR: Bulk classification failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV.', 'danger')
            return redirect(request.url)

    active_batch_id = session.get('current_batch_id')
    if active_batch_id and active_batch_id in app.bulk_processing_state:
        return render_template('bulk_classification.html', active_batch_id=active_batch_id)

    return render_template('bulk_classification.html', active_batch_id=None)


@app.route('/start_new_bulk')
def start_new_bulk():
    """Clears the current batch ID from the session to allow a new upload."""
    if 'current_batch_id' in session:
        session.pop('current_batch_id', None)
        flash('Previous review session cleared. You can now start a new bulk classification.', 'info')
    return redirect(url_for('bulk_classification'))


rate_limiter = AsyncRateLimiter(10, 60)

async def process_bulk_footwear_flask(batch_id, api_key, pdf_directory='data/chapter_data'):
    """Processes bulk footwear classification in the background.
     Updates the app.bulk_processing_state with results.
     Args:
        batch_id (str): Unique identifier for the bulk processing batch.
        api_key (str): API key for configuring the GenAI model.
        pdf_directory (str): Directory containing PDF chapter data.
     """
    print(f"!!!!!!!!!!!!!! BACKGROUND TASK STARTED FOR FOOTWEAR BATCH {batch_id} !!!!!!!!!!!!!!")
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        print(f"ERROR: process_bulk_footwear_flask called for a non-existent batch_id: {batch_id}")
        return

    try:
        df_input = state['input_df']
        model = configure_genai_footwear(api_key)
        all_results_list = []
        total_rows = len(df_input)
        start_time = time.time()
        lock = threading.Lock()

        pdf_data_cache = load_all_pdf_data(pdf_directory)

        def process_row(row):
            """Processes a single row for footwear classification."""
            original_index = row["idx"]
            country_col = "tariff_country_description"
            name_col = "customs_description"
            name_col2 = "product_type"
            material_col = "material_composition"
            construction_col = "outsole_material"
            gender_col = "gender_name"
            size_col = "size_code"

            country = str(row[country_col]).strip().lower() if pd.notna(row[country_col]) else "unknown"
            product_type = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
            if name_col2 in row and pd.notna(row[name_col2]):
                product_type += " " + str(row[name_col2]).strip()
            material = str(row[material_col]).strip() if pd.notna(row[material_col]) else ""
            construction = str(row[construction_col]).strip() if construction_col in row and pd.notna(row[construction_col]) else ""
            gender = str(row[gender_col]).strip() if gender_col in row and pd.notna(row[gender_col]) else ""
            size = str(row[size_col]).strip() if size_col in row and pd.notna(row[size_col]) else ""

            desc_parts = [f"Product for {country.upper()}:"]
            if gender:
                desc_parts.append(f"{gender}'s")
            if product_type:
                desc_parts.append(product_type)
            if material:
                desc_parts.append(f"Material: {material}.")
            if construction:
                desc_parts.append(f"Construction: {construction}.")
            if size and country.upper() in ["EUROPE", "UNITED KINGDOM"]:
                desc_parts.append(f"Size: {size}.")
            product_description = " ".join(desc_parts).strip()

            # ✅ Use 'idx' key instead of 'original_index' to align with general bulk schema
            base_result_row = {
                "idx": original_index,
                "input_country": country.upper(),
                "input_product": product_type,
                "input_material": material,
                "input_construction": construction,
                "input_gender": gender,
                "product_description": product_description,
                "hs_code_1": "N/A", "certainty_1": 0, "reasoning_1": "Skipped",
                "hs_code_2": "", "certainty_2": 0, "reasoning_2": "",
                "hs_code_3": "", "certainty_3": 0, "reasoning_3": ""
            }

            if not product_type or not material or country == "unknown":
                base_result_row["reasoning_1"] = "Skipped due to missing essential data"
                return base_result_row

            if country not in pdf_data_cache or not pdf_data_cache[country]:
                with lock:
                    print(f"Reloading PDF data for {country}...")
                pdf_data_cache.update(load_all_pdf_data(pdf_directory))
                if country not in pdf_data_cache or not pdf_data_cache[country]:
                    base_result_row["reasoning_1"] = f"Skipped: No PDF data for {country}"
                    return base_result_row

            processed_pdfs_for_current_country = pdf_data_cache[country]
            relevant_chapters = find_relevant_chapters_footwear(product_description, country, processed_pdfs_for_current_country)
            gri = processed_pdfs_for_current_country.get("gri", "")

            try:
                generated_response = generate_hs_codes_footwear(
                    model,
                    product_description,
                    country,
                    relevant_chapters,
                    gri=""
                )

                product_df_row = extract_hs_codes_footwear(generated_response)
                if not product_df_row.empty:
                    extracted_data = product_df_row.iloc[0].to_dict()
                else:
                    extracted_data = {}

                print("==== RAW MODEL OUTPUT ====")
                print(generated_response)
                print("==== EXTRACTED DATA ====")
                print(extracted_data)
                print("==========================")

                if all(not extracted_data.get(f"hs_code_{i}", "").strip() for i in range(1, 4)):
                    print(f"⚠️ All HS codes missing for row (Idx: {original_index})")

                if not product_df_row.empty:
                    extracted_data = product_df_row.iloc[0].to_dict()
                    for i in range(1, 4):
                        hs_col = f"hs_code_{i}"
                        cert_col = f"certainty_{i}"
                        reas_col = f"reasoning_{i}"
                        base_result_row[hs_col] = extracted_data.get(hs_col, "")
                        base_result_row[cert_col] = extracted_data.get(cert_col, 0)
                        base_result_row[reas_col] = extracted_data.get(reas_col, "")
                else:
                    base_result_row["reasoning_1"] = "Error: Failed to parse response"

                return base_result_row

            except Exception as gen_e: # Catch generation/extraction errors
                base_result_row["hs_code_1"] = "ERROR"
                base_result_row["reasoning_1"] = str(gen_e)
                return base_result_row

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_row, row) for _, row in df_input.iterrows()]
            for df_idx, future in enumerate(futures):
                with lock:
                    all_results_list.append(future.result())
                    state['processed_count'] = df_idx + 1
                    elapsed_time = time.time() - start_time
                    est_remaining = (elapsed_time / (df_idx + 1)) * (total_rows - (df_idx + 1)) if df_idx > 0 else 0
                    print(f"Processing row {df_idx + 1}/{total_rows}... Est. time remaining: {int(est_remaining)}s")

        # ✅ Add the missing state setup to match the general bulk process
        state['processed_results'] = all_results_list
        state['selection_status'] = {row['idx']: {"status": "pending", "data": {}} for row in all_results_list}
        state['total_rows'] = len(all_results_list)
        state['status'] = 'completed'

        # ✅ Save results in the same structure as general bulk flow
        save_bulk_classification_results(pd.DataFrame(all_results_list))

    except Exception as e:
        state['status'] = 'error'
        state['error_message'] = str(e)
        print(f"ERROR in process_bulk_footwear_flask: {str(e)}")
        import traceback
        traceback.print_exc()

async def process_bulk_data_flask(batch_id, api_key):
    """Processes bulk classification in the background.
     Updates the app.bulk_processing_state with results.
     Args:
        batch_id (str): Unique identifier for the bulk processing batch.
        api_key (str): API key for configuring the GenAI model.
     """
    print(f"!!!!!!!!!!!!!! BACKGROUND TASK STARTED FOR BATCH {batch_id} !!!!!!!!!!!!!!")
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        print(f"ERROR: process_bulk_data_flask called for a non-existent batch_id: {batch_id}")
        return

    try: # Main processing logic with error handling
        df_input = state['input_df']
        model = configure_genai(api_key)
        all_results_list = []
        items_for_api_call = []
        classification_cache_for_batch = {}
        start_time = time.time()
        
        for df_idx, row in df_input.iterrows():
            idx = row["idx"]
            country = str(row["tariff_country_description"]).strip().lower() if pd.notna(row["tariff_country_description"]) else "unknown"
            product_type = str(row["customs_description"]).strip() if pd.notna(row["customs_description"]) else ""
            if "product_type" in row and pd.notna(row["product_type"]):
                product_type += " " + str(row["product_type"]).strip()
            material = str(row["material_composition"]).strip() if pd.notna(row["material_composition"]) else ""
            construction = str(row["material_type"]).strip() if "material_type" in row and pd.notna(row["material_type"]) else ""
            gender = str(row["division"]).strip() if "division" in row and pd.notna(row["division"]) else ""
            hs_code_from_csv = str(row.get('tariff_code', 'N/A')) if pd.notna(row.get('tariff_code')) else 'N/A'

            desc_parts = [f"Product for {country.upper()}:"]
            if gender: 
                desc_parts.append(f"{gender}'s")
            if product_type: 
                desc_parts.append(product_type)
            if material: 
                desc_parts.append(f"Material: {material}.")
            if construction: 
                desc_parts.append(f"Construction: {construction}.")
            product_description = " ".join(desc_parts).strip()
            
            product_key = (country, gender.lower(), product_type.lower(), material.lower(), construction.lower())

            base_result_row = {
                "idx": idx, "input_country": country.upper(), "input_product": product_type,
                "input_material": material, "input_construction": construction, "input_gender": gender,
                "product_description": product_description,
                "hs_code_1": "N/A", "certainty_1": 0, "reasoning_1": "Skipped", "raw_reasoning_text_1": "",
                "hs_code_2": "", "certainty_2": 0, "reasoning_2": "", "raw_reasoning_text_2": "",
                "hs_code_3": "", "certainty_3": 0, "reasoning_3": "", "raw_reasoning_text_3": "",
                "hs_code_from_csv": hs_code_from_csv
            }

            if not product_type or not material or country == "unknown" or country not in app.doc_cache:
                reason = "Skipped due to missing essential data" if (not product_type or not material or country == "unknown") else f"Skipped: No PDF data for {country}"
                base_result_row["reasoning_1"] = reason
                all_results_list.append(base_result_row)
                state['processed_count'] += 1
                continue

            all_validated_entries = load_validated_codes(product_description, country) # Check for validated codes first
            validated_entries_with_code = [e for e in all_validated_entries if e.get("hs_code")] # Filter for entries with HS code
            if validated_entries_with_code:
                for i, entry in enumerate(validated_entries_with_code[:3]):
                    reasoning = entry.get("reasoning") or "Previously validated by user (code only)."
                    base_result_row[f"hs_code_{i+1}"] = entry.get("hs_code", "N/A")
                    base_result_row[f"certainty_{i+1}"] = 100
                    base_result_row[f"reasoning_{i+1}"] = reasoning
                state['selection_status'][idx] = {"status": "validated_with_reasoning", "data": {"selected_hs_code": base_result_row["hs_code_1"]}}
                all_results_list.append(base_result_row)
                state['processed_count'] += 1
                continue
                
            if product_key in classification_cache_for_batch:
                cached_result = classification_cache_for_batch[product_key]
                for i in range(1, 4):
                    base_result_row[f"hs_code_{i}"] = cached_result.get(f"hs_code_{i}", "")
                    base_result_row[f"certainty_{i}"] = cached_result.get(f"certainty_{i}", 0)
                    base_result_row[f"reasoning_{i}"] = cached_result.get(f"reasoning_{i}", "")
                    base_result_row[f"raw_reasoning_text_{i}"] = cached_result.get(f"raw_reasoning_text_{i}", "")
                all_results_list.append(base_result_row)
                state['processed_count'] += 1
                continue
                
            items_for_api_call.append({
                "base_row": base_result_row, "product_key": product_key,
                "args": {
                    "product_description": product_description, "country": country,
                    "relevant_chapters": find_relevant_chapters(product_description, country, app.doc_cache[country]),
                    "legal_notes": app.doc_cache[country].get("legal_notes", ""),
                    "classification_guide": app.doc_cache[country].get("classification_guide", ""),
                    "gri": app.doc_cache[country].get("gri", ""),
                    "guidelines": load_text_files_for_country(app.config['PDF_DIRECTORY'], country).get(f"{country}_guidelines", ""),
                    "rejected_codes_snapshot": load_rejected_codes(product_description, country),
                    "product_type": product_type
                }
            })
        
        if items_for_api_call:
            semaphore = asyncio.Semaphore(10)
            async def run_classification_with_semaphore(item):
                """Runs classification with semaphore to limit concurrency.
                    Args:
                        item (dict): Dictionary containing base_row, product_key, and args for classification.
                    Returns:
                        dict: Result row with classification results.
                     """
                async with semaphore:
                    await rate_limiter.acquire()
                    response = await generate_hs_codes(model, **item['args'])
                    result_row = item['base_row'].copy()
                    product_key = item['product_key']
                    if response and isinstance(response, str):
                        product_df_row = extract_hs_codes(response)
                        if not product_df_row.empty:
                            extracted_data = product_df_row.iloc[0].to_dict()
                            for i in range(1, 4):
                                result_row[f"hs_code_{i}"] = extracted_data.get(f"hs_code_{i}", "")
                                result_row[f"certainty_{i}"] = extracted_data.get(f"certainty_{i}", 0)
                                result_row[f"reasoning_{i}"] = extracted_data.get(f"reasoning_{i}", "")
                                result_row[f"raw_reasoning_text_{i}"] = extracted_data.get(f"raw_reasoning_text_{i}", "")
                            classification_cache_for_batch[product_key] = extracted_data
                        else: 
                            result_row["reasoning_1"] = "Error: Failed to parse API response"
                    else: 
                        result_row["reasoning_1"] = "Error: Model returned invalid or empty response"
                    return result_row

            tasks = [run_classification_with_semaphore(item) for item in items_for_api_call]
            api_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in api_results:
                if isinstance(result, Exception):
                    raise result
                else:
                    all_results_list.append(result)
                state['processed_count'] += 1
                print(f"DEBUG: Processed {state['processed_count']}/{state['total_rows']}.")

        all_results_list.sort(key=lambda x: x['idx']) # Ensure original order is maintained
        processed_df = pd.DataFrame(all_results_list) if all_results_list else pd.DataFrame()
        
        state['processed_results'] = processed_df.to_dict('records')
        save_bulk_classification_results(processed_df, filename=ALL_BULK_RESULTS_FILE)
        log_detailed_accuracy(processed_df)
        
        total_processing_time = time.time() - start_time
        save_processing_time(total_processing_time, len(df_input))

        state['status'] = 'completed'
        print(f"DEBUG: Background processing for batch {batch_id} completed in {total_processing_time:.2f}s.")

    except Exception as e:
        import traceback
        print(f"!!!!!!!!!!!!!! ERROR IN BACKGROUND TASK FOR BATCH {batch_id} !!!!!!!!!!!!!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        state['status'] = 'error'
        state['error_message'] = str(e)


@app.route('/processing_status/<batch_id>', methods=['GET'])
def processing_status_page(batch_id):
    """Renders the processing status page for a given bulk classification batch."""
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        flash('Processing batch not found or already completed. Please start a new bulk classification.', 'danger')
        return redirect(url_for('bulk_classification'))

    return render_template('processing_status.html', batch_id=batch_id, state=state)


@app.route('/get_bulk_progress/<batch_id>', methods=['GET'])
def get_bulk_progress(batch_id):
    """AJAX endpoint to get the current progress of bulk classification."""
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        return jsonify({'status': 'error', 'message': 'Batch not found or completed'}), 404

    return jsonify({
        'status': state.get('status', 'processing'),
        'processed_count': state.get('processed_count', 0),
        'total_rows': state.get('total_rows', 0),
        'error_message': state.get('error_message', None)
    })

# ... (the rest of your app.py file from review_bulk_table onwards remains unchanged) ...

@app.route('/review_bulk_table/<batch_id>', methods=['GET'])
def review_bulk_table(batch_id):
    """Renders the review table for bulk classification results."""
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        flash('Bulk processing batch not found or completed. Please start a new one.', 'danger')
        return redirect(url_for('bulk_classification'))

    # If processing is somehow still underway, keep them on the status page
    if state.get('status', 'processing') == 'processing':
        return redirect(url_for('processing_status_page', batch_id=batch_id))

    processed_results_df = pd.DataFrame(state['processed_results'])
    selection_status = state['selection_status']

    return render_template('review_bulk_table.html',
                           batch_id=batch_id,
                           results=processed_results_df.to_dict('records'),
                           selection_status=selection_status,
                           total_rows=state['total_rows'])


@app.route('/review_bulk_results/<batch_id>', methods=['GET', 'POST'])
async def review_bulk_results(batch_id):
    """Displays the review page for bulk classification results and handles user actions."""
    state = app.bulk_processing_state.get(batch_id)
    if not state:
        flash('Bulk processing batch not found or completed. Please start a new one.', 'danger')
        return redirect(url_for('bulk_classification'))

    if state['status'] == 'processing':
        flash("Processing is still underway. Please wait...", 'info')
        return redirect(url_for('processing_status_page', batch_id=batch_id))

    processed_results_df = pd.DataFrame(state['processed_results'])
    selection_status = state['selection_status']

    if request.method == 'POST':
        action_type = request.form.get('action_type')
        item_idx = int(request.form.get('item_idx'))
        option_num = request.form.get('option_num')

        current_item_state = selection_status.get(item_idx, {"status": "pending", "data": {}})

        current_processed_df_for_lookup = pd.DataFrame(app.bulk_processing_state[batch_id]['processed_results'])
        original_row_for_product_info = current_processed_df_for_lookup[current_processed_df_for_lookup['idx'] == item_idx].iloc[0]
        product_description_for_action = original_row_for_product_info['product_description']
        country_for_action = original_row_for_product_info['input_country'].lower()

        response_message = ""
        success_status = True
        
        product_type_for_action = original_row_for_product_info['input_product']

        if action_type == 'regenerate_all':
            for i in range(1, 4):
                hs_code_to_reject = original_row_for_product_info.get(f'hs_code_{i}')
                if hs_code_to_reject and hs_code_to_reject not in ["N/A", "ERROR", "", None]:
                    save_rejected_code(product_description_for_action, country_for_action, hs_code_to_reject)

            current_item_state['status'] = 'regenerate_pending'
            selection_status[item_idx] = current_item_state
            log_interaction_event(country_for_action, product_type_for_action, 'regeneration')
            
            await process_single_regeneration_in_background(batch_id, item_idx)
            
            return jsonify({
                'status': 'success',
                'message': f"Product {item_idx} regeneration complete.",
                'item_idx': item_idx,
                'reload_page': True
            })

        elif action_type == 'select_code':
            selected_hs_code = request.form.get(f'hs_code_{option_num}')
            selected_certainty = request.form.get(f'certainty_{option_num}', type=int)
            selected_reasoning = request.form.get(f'reasoning_{option_num}')
            current_item_state['status'] = 'selected'
            current_item_state['data'] = {
                'selected_hs_code': selected_hs_code,
                'selected_certainty': selected_certainty,
                'selected_reasoning': selected_reasoning,
                'selected_option': int(option_num)
            }
            log_interaction_event(country_for_action, product_type_for_action, 'selection', selected_hs_code, details=f'option_{option_num}')
            response_message = f"Product {item_idx} selected successfully!"

        elif action_type == 'mark_incorrect':
            rejected_hs_code = request.form.get('rejected_hs_code')
            if rejected_hs_code:
                save_rejected_code(product_description_for_action, country_for_action, rejected_hs_code)
                if 'incorrect_options' not in current_item_state['data']:
                    current_item_state['data']['incorrect_options'] = set()
                current_item_state['data']['incorrect_options'].add(int(option_num))
                log_interaction_event(country_for_action, product_type_for_action, 'rejection', rejected_hs_code, details=f'option_{option_num}')
                response_message = f"Option {option_num} for Product {item_idx} marked incorrect."

        elif action_type == 'review_later':
            current_item_state['status'] = 'review_later'
            current_item_state['data'] = {}
            response_message = f"Product {item_idx} marked for review later."

        elif action_type == 'save_code_only':
            hs_code_to_save = request.form.get(f'hs_code_{option_num}')
            if hs_code_to_save:
                save_validated_code(product_description_for_action, country_for_action, hs_code_to_save, reasoning=None)
                current_item_state['status'] = 'validated_code_only'
                current_item_state['data'] = {'selected_hs_code': hs_code_to_save, 'selected_option': int(option_num)}
                response_message = f"Product {item_idx} HS Code {hs_code_to_save} saved successfully (code only)!"
            else:
                success_status = False
                response_message = "Error: HS Code to save not provided."

        elif action_type == 'save_code_with_reasoning':
            hs_code_to_save = request.form.get(f'hs_code_{option_num}')
            reasoning_to_save = request.form.get(f'reasoning_{option_num}')
            if hs_code_to_save and reasoning_to_save:
                save_validated_code(product_description_for_action, country_for_action, hs_code_to_save, reasoning=reasoning_to_save)
                current_item_state['status'] = 'validated_with_reasoning'
                current_item_state['data'] = {
                    'selected_hs_code': hs_code_to_save,
                    'selected_reasoning': reasoning_to_save,
                    'selected_option': int(option_num)
                }
                response_message = f"Product {item_idx} HS Code and reasoning saved successfully!"
            else:
                success_status = False
                response_message = "Error: HS Code or reasoning to save not provided."
        
        else:
            success_status = False
            response_message = "Unknown action type."

        selection_status[item_idx] = current_item_state
        app.bulk_processing_state[batch_id]['selection_status'] = selection_status

        return jsonify({
            'status': 'success' if success_status else 'error',
            'message': response_message,
            'item_idx': item_idx,
            'new_status': current_item_state['status'],
            'reload_page': True # Instruct JS to reload
        })

    return render_template('review_bulk_results.html',
                           batch_id=batch_id,
                           results=processed_results_df.to_dict('records'),
                           selection_status=selection_status,
                           total_rows=state['total_rows'])


async def process_single_regeneration_in_background(batch_id, item_idx):
    """Processes regeneration for a single item in the bulk classification batch.
     Updates the app.bulk_processing_state with the new results.
     Args:
        batch_id (str): Unique identifier for the bulk processing batch.
        item_idx (int): Index of the item to regenerate.
     """
    state = app.bulk_processing_state[batch_id]
    selection_status = state['selection_status']
    results_list = state['processed_results']
    item_to_update = next((item for item in results_list if item['idx'] == item_idx), None)

    if item_to_update is None:
        print(f"ERROR: Could not find item with idx {item_idx} in state for regeneration.")
        selection_status[item_idx] = {"status": "error", "data": {"reason": "Internal error: Item not found."}}
        return

    selection_status[item_idx]['status'] = 'regenerating'

    product_description_regen = item_to_update['product_description']
    country_lower = item_to_update['input_country'].lower()
    product_type_regen = item_to_update['input_product']

    rejected_codes_snapshot = load_rejected_codes(product_description_regen, country_lower)
    print(f"DEBUG: Found {len(rejected_codes_snapshot)} rejected codes for '{product_description_regen}' in {country_lower}.")

    processed_pdfs_for_current_country = app.doc_cache.get(country_lower, {})
    relevant_chapters = find_relevant_chapters(product_description_regen, country_lower, processed_pdfs_for_current_country)
    legal_notes = processed_pdfs_for_current_country.get("legal_notes", "")
    classification_guide = processed_pdfs_for_current_country.get("classification_guide", "")
    gri = processed_pdfs_for_current_country.get("gri", "")
    guidelines = load_text_files_for_country(app.config['PDF_DIRECTORY'], country_lower).get(f"{country_lower}_guidelines", "")

    try:
        model = configure_genai(app.config['API_KEY'])
        new_response = await generate_hs_codes(
            model, product_description_regen, country_lower, relevant_chapters,
            legal_notes, classification_guide, gri=gri, guidelines=guidelines,
            rejected_codes_snapshot=rejected_codes_snapshot,
            product_type=product_type_regen
        )
        new_options_df = extract_hs_codes(new_response)

        if not new_options_df.empty:
            new_data_dict = new_options_df.iloc[0].to_dict()
            for i in range(1, 4):
                item_to_update[f'hs_code_{i}'] = new_data_dict.get(f'hs_code_{i}', '')
                item_to_update[f'certainty_{i}'] = new_data_dict.get(f'certainty_{i}', 0)
                item_to_update[f'reasoning_{i}'] = new_data_dict.get(f'reasoning_{i}', {})
                item_to_update[f'raw_reasoning_text_{i}'] = new_data_dict.get(f'raw_reasoning_text_{i}', '')
            selection_status[item_idx] = {"status": "pending", "data": {}}
            state['regenerate_queue'].discard(item_idx)
            print(f"DEBUG: Product {item_idx} regenerated successfully.")
        else:
            selection_status[item_idx] = {"status": "error", "data": {"reason": "Regen failed: No codes extracted from response."}}
            print(f"DEBUG: Failed to extract codes for Product {item_idx} during regeneration.")
            
    except Exception as regen_e:
        import traceback
        print(f"ERROR: Exception during regeneration for Product {item_idx}: {regen_e}")
        traceback.print_exc()
        selection_status[item_idx] = {"status": "error", "data": {"reason": str(regen_e)}}

    state['selection_status'] = selection_status


@app.route('/finalize_bulk_results/<batch_id>', methods=['POST'])
def finalize_bulk_results(batch_id):
    """Finalizes the bulk classification results and prepares the final output.
     Args:
        batch_id (str): Unique identifier for the bulk processing batch."""
    if batch_id not in app.bulk_processing_state:
        flash('Bulk processing batch not found.', 'danger')
        return redirect(url_for('bulk_classification'))

    state = app.bulk_processing_state[batch_id]
    processed_results_df = pd.DataFrame(state['processed_results'])
    selection_status = state['selection_status']
    original_input_df = state['input_df']

    processed_data = []
    pending_count = 0
    error_count = 0
    review_later_count = 0
    validated_count = 0 

    for original_idx_in_batch, current_row_processed in processed_results_df.iterrows():
        item_specific_idx = current_row_processed['idx']
        item_status = selection_status.get(item_specific_idx, {"status": "pending", "data": {}})

        if item_status["status"] == "review_later":
            review_later_count += 1
            continue

        original_source_row_filtered = original_input_df[original_input_df['idx'] == item_specific_idx]
        actual_hs_code = "N/A - Not in original file"
        if not original_source_row_filtered.empty:
            if 'tariff_code' in original_source_row_filtered.columns:
                actual_hs_code = str(original_source_row_filtered.iloc[0]['tariff_code'])
        
        output_row = {
            "idx": item_specific_idx, "product_description": current_row_processed.get("product_description", ""),
            "input_product": current_row_processed.get("input_product", ""), "input_material": current_row_processed.get("input_material", ""),
            "input_construction": current_row_processed.get("input_construction", ""), "input_gender": current_row_processed.get("input_gender", ""),
            "input_country": current_row_processed.get("input_country", ""), "suggested_hs_code_1": current_row_processed.get("hs_code_1", "N/A"),
            "certainty_1_percent": current_row_processed.get("certainty_1", 0), "actual_hs_code_from_dataset": actual_hs_code,
            "selected_hs_code": "N/A", "selected_reasoning": "N/A", "certainty (%)": None, "status": item_status["status"]
        }

        if item_status["status"] == "selected":
            output_row["selected_hs_code"] = item_status["data"].get("selected_hs_code", "Error")
            output_row["selected_reasoning"] = item_status["data"].get("selected_reasoning", "Error")
            output_row["certainty (%)"] = item_status["data"].get("selected_certainty")
            output_row["status"] = f"Selected Option {item_status['data'].get('selected_option', '?')}"
        elif item_status["status"] == "incorrect":
            output_row["selected_hs_code"] = "Marked Incorrect"
            output_row["selected_reasoning"] = "User marked as incorrect."
            output_row["status"] = "Marked Incorrect"
        elif item_status["status"] == "validated_code_only":
            output_row["selected_hs_code"] = item_status["data"].get("selected_hs_code", "Error")
            output_row["selected_reasoning"] = "Validated (Code Only)"
            output_row["certainty (%)"] = 100
            output_row["status"] = "Validated (Code Only)"
            validated_count += 1
        elif item_status["status"] == "validated_with_reasoning":
            output_row["selected_hs_code"] = item_status["data"].get("selected_hs_code", "Error")
            output_row["selected_reasoning"] = item_status["data"].get("selected_reasoning", "Validated (with reasoning)")
            output_row["certainty (%)"] = 100
            output_row["status"] = "Validated (with Reasoning)"
            validated_count += 1
        elif item_status["status"] == "pending" or item_status["status"] == "regenerate_pending":
             if current_row_processed.get("hs_code_1") in ["N/A", "ERROR"]:
                 output_row["selected_hs_code"] = current_row_processed.get("hs_code_1")
                 output_row["selected_reasoning"] = current_row_processed.get("reasoning_1")
                 output_row["status"] = "Skipped/Error"
                 error_count += 1
             else:
               output_row["selected_hs_code"] = "Pending Review"
               output_row["selected_reasoning"] = "User review needed."
               output_row["status"] = "Pending Review"
               pending_count += 1
        elif item_status["status"] == "error":
            output_row["selected_hs_code"] = "Error During Regeneration"
            output_row["selected_reasoning"] = item_status["data"].get("reason", "Unknown error")
            output_row["status"] = "Error"
            error_count += 1

        processed_data.append(output_row)

    if pending_count > 0:
        flash(f"{pending_count} products are still pending review. Please address them before finalizing.", 'warning')
        return redirect(url_for('review_bulk_table', batch_id=batch_id))

    final_df = pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
    final_cols_order = [
        "idx", "input_country", "input_product", "input_material", "input_construction", "input_gender", "product_description",
        "suggested_hs_code_1", "certainty_1_percent", "actual_hs_code_from_dataset", "selected_hs_code", "certainty (%)", "status", "selected_reasoning"
    ]
    final_df = final_df.reindex(columns=[col for col in final_cols_order if col in final_df.columns])
    save_final_selected_results(final_df, filename='final_hs_codes.csv')
    
    flash_message = "Selections processed successfully!"
    if review_later_count > 0: 
        flash_message += f" ({review_later_count} items marked 'Review Later' were excluded)."
    if error_count > 0: 
        flash_message += f" ({error_count} items had original errors/were skipped)."
    if validated_count > 0: 
        flash_message += f" ({validated_count} items were previously validated)."
    flash(flash_message, 'success')

    if batch_id in app.bulk_processing_state: 
        del app.bulk_processing_state[batch_id]
    if 'current_batch_id' in session: 
        session.pop('current_batch_id', None)

    return render_template('final_bulk_report.html', final_results=final_df.to_dict('records'))


def simplify_product_type(description):
    """Simplifies a detailed product description to a general category.
    Args:
        description (str): Detailed product description."""
    if not isinstance(description, str): 
        return "Unknown"
    desc_lower = description.lower()
    if "t-shirt" in desc_lower or "t shirt" in desc_lower: 
        return "T-Shirt"
    if "tank" in desc_lower: 
        return "Tank Top"
    if "short sleeves" in desc_lower: 
        return "T-Shirt"
    if "short" in desc_lower: 
        return "Shorts"
    if "pant" in desc_lower or "trouser" in desc_lower: 
        return "Pants"
    if "tight" in desc_lower: 
        return "Tights"
    if "jacket" in desc_lower: 
        return "Jacket"
    if "hoodie" in desc_lower or "sweatshirt" in desc_lower: 
        return "Hoodie/Sweatshirt"
    if "bra" in desc_lower: 
        return "Bra"
    if "cap" in desc_lower or "beanie" in desc_lower: 
        return "Headwear"
    if "singlet" in desc_lower: 
        return "Singlet"
    return "Other"


@app.route("/metrics")
def display_metrics():
    """
    Renders the metrics dashboard page.
    All data processing is delegated to the metrics_service module.
    """
    dashboard_data = get_dashboard_data()

    # Check if the service returned an error to display
    if "error" in dashboard_data:
        flash(dashboard_data["error"], "danger")
        # Render the template with empty data to avoid errors
        return render_template("metrics.html", 
                               metrics={}, processing_times=[], overall_avg_time_per_row=0,
                               token_data={}, cost_data={}, quality_data={}, 
                               usage_data={}, interaction_data={}, 
                               completeness_data={}, evolution_data={})
        
    return render_template("metrics.html", **dashboard_data)    # the ** operator unpacks the dictionary, passing its keys and values as keyword arguments to the render_template function.


from flask import send_file

@app.route("/download_final_report")
def download_final_report():
    """
    Endpoint to download the final HS code report as a CSV file.
    Returns:
        Response: Flask response to send the file for download.
    """
    file_path = os.path.join(os.path.dirname(__file__), "final_hs_codes.csv")
    if not os.path.exists(file_path):
        flash("Please finalize the bulk results to generate and download the final report.", "warning")
        return redirect(url_for('bulk_classification'))
    return send_file(file_path, as_attachment=True, download_name="final_hs_codes.csv")

# --- Run the Flask App --- 
if __name__ == '__main__':
    app.run(debug=True)