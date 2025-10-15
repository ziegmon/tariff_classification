import csv
from datetime import datetime
import os
import json
from flask import Flask
from config import Config
import pathlib
import pandas as pd

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


TOKEN_LOG_FILE = app.config["TOKEN_LOG_FILE"]
REJECTED_CODES_FILE = app.config["REJECTED_CODES_FILE"]
VALIDATED_CODES_FILE = app.config["VALIDATED_CODES_FILE"]
METRICS_RESULTS_FILE = app.config["METRICS_RESULTS_FILE"]
FINAL_SELECTED_FILE = app.config["FINAL_SELECTED_FILE"]
INTERACTION_LOG_FILE = app.config["INTERACTION_LOG_FILE"]
PROCESSING_TIMES_FILE = app.config["PROCESSING_TIMES_FILE"]


def log_token_usage(country, product_type, input_tokens, output_tokens):
    """Logs the token usage for a single API call to a CSV file."""
    file_exists = os.path.exists(TOKEN_LOG_FILE)
    try:
        with open(TOKEN_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'country', 'product_type', 'input_tokens', 'output_tokens', 'total_tokens']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'country': country,
                'product_type': product_type,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            })
    except Exception as e:
        print(f"ERROR: Could not write to token log file. Reason: {e}")
        


def save_rejected_code(product, country, code): # Accepts 3 arguments
    entry = {
        "product_description": product,
        "country": country,
        "rejected_code": code
    }
    try:
        if not os.path.exists(REJECTED_CODES_FILE):
            with open(REJECTED_CODES_FILE, "w") as f:
                json.dump([entry], f, indent=2)
        else:
            with open(REJECTED_CODES_FILE, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {REJECTED_CODES_FILE}."
                        " Starting with an empty list."
                    )
                    data = []
                except Exception as e:
                    print(
                        f"An unexpected error occurred while reading"
                        f" {REJECTED_CODES_FILE}: {e}"
                    )
                    raise
                else:
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
        #print("Code saved successfully.")
    except Exception as e:
        print(f"Error saving code: {e}")
        

def load_rejected_codes(product_description, country):
    # print(f"Loading REJECTED_CODES_FILE: {REJECTED_CODES_FILE}")
    if not os.path.exists(REJECTED_CODES_FILE):
        return []

    try:
        with open(REJECTED_CODES_FILE, "r") as f:
            all_rejections = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    product_specific_rejections = [
        entry for entry in all_rejections
        if entry.get("product_description") == product_description and entry.get("country") == country
    ]
    return product_specific_rejections


def save_validated_code(product_description, country, hs_code, reasoning=None):
    """
    Saves a validated HS code and its reasoning (optional) to a JSON file.
    """
    entry = {
        "product_description": product_description,
        "country": country,
        "hs_code": hs_code,
        "reasoning": reasoning, # Can be None if only code is validated
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        if not os.path.exists(VALIDATED_CODES_FILE):
            with open(VALIDATED_CODES_FILE, "w") as f:
                json.dump([entry], f, indent=2)
        else:
            with open(VALIDATED_CODES_FILE, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {VALIDATED_CODES_FILE}."
                        " Starting with an empty list."
                    )
                    data = []
                except Exception as e:
                    print(
                        f"An unexpected error occurred while reading"
                        f" {VALIDATED_CODES_FILE}: {e}"
                    )
                    raise
                else:
                    # Check if this exact product-country-hs_code combination already exists
                    # This prevents duplicate entries for the same validated code
                    existing_entry = next((item for item in data if
                                           item.get("product_description") == product_description and
                                           item.get("country") == country and
                                           item.get("hs_code") == hs_code), None)
                    if existing_entry:
                        print(f"DEBUG: Code '{hs_code}' for '{product_description}' in {country} already exists in validated codes. Updating timestamp.")
                        existing_entry['timestamp'] = entry['timestamp']
                        if reasoning is not None: # Only update reasoning if provided
                            existing_entry['reasoning'] = reasoning
                    else:
                        data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
        print(f"DEBUG: Validated code '{hs_code}' for '{product_description}' in {country} saved successfully.")
    except Exception as e:
        print(f"ERROR: Error saving validated code: {e}")


def load_validated_codes(product_description, country):
    """
    Loads validated HS codes for a given product description and country.
    Returns a list of matching entries.
    """
    if not os.path.exists(VALIDATED_CODES_FILE):
        return []

    try:
        with open(VALIDATED_CODES_FILE, "r") as f:
            all_validated = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    product_specific_validated = [
        entry for entry in all_validated
        if entry.get("product_description") == product_description and entry.get("country") == country
    ]
    return product_specific_validated



# def save_processing_time(processing_time, num_rows, filename="processing_times.csv"):
#     file_exists = os.path.exists(filename)

#     with open(filename, 'a', newline='') as csvfile:
#         fieldnames = ['timestamp', 'rows_processed', 'total_time_seconds']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         if not file_exists:
#             writer.writeheader()

#         writer.writerow({
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'rows_processed': num_rows,
#             'total_time_seconds': round(processing_time, 2)
#         })


def save_bulk_classification_results(df_to_save: pd.DataFrame, filename=METRICS_RESULTS_FILE):
    """
    Saves or appends the raw bulk classification results (model's suggestions)
    DataFrame to a CSV file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_with_time = df_to_save.copy()

    # Ensure all columns from the Streamlit raw_df.csv are present and ordered
    # And add timestamp
    raw_output_cols = [
        "idx", "product_description", "input_product", "input_material",
        "input_construction", "input_gender", "input_country",
        "hs_code_1", "certainty_1", "reasoning_1",
        "hs_code_2", "certainty_2", "reasoning_2",
        "hs_code_3", "certainty_3", "reasoning_3",
        "hs_code_from_csv" # This comes from the original input_df_for_mapping in Streamlit
    ]

    # Add any missing columns with NaN if they weren't generated (e.g., if you only output hs_code_1)
    for col in raw_output_cols:
        if col not in df_with_time.columns:
            df_with_time[col] = pd.NA # Or None, depends on desired type for empty values

    df_with_time['timestamp'] = timestamp # Add timestamp

    # Reorder columns to match Streamlit's raw_df if desired for strict format
    # Note: hs_code_from_csv would ideally be merged into the results_df in app.py
    # before saving, to accurately reflect the Streamlit process where it's added.
    # For now, we'll assume it exists if it's coming from input.
    final_cols_for_raw_save = raw_output_cols + ['timestamp']
    df_to_write = df_with_time.reindex(columns=final_cols_for_raw_save)


    file_path = pathlib.Path(filename)

    print(f"DEBUG: save_bulk_classification_results called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{file_path.resolve()}'")
    print(f"DEBUG: Does file exist before writing? {file_path.exists()}")
    print(f"DEBUG: DataFrame to save shape: {df_to_write.shape}")
    print(f"DEBUG: DataFrame columns: {df_to_write.columns.tolist()}") # Check actual columns

    try:
        if file_path.exists():
            df_to_write.to_csv(file_path, mode='a', header=False, index=False)
            print(f"DEBUG: Successfully appended results to {filename}.")
        else:
            df_to_write.to_csv(file_path, mode='w', header=True, index=False)
            print(f"DEBUG: Successfully created and saved results to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save/append results to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()
        

def save_final_selected_results(df_to_save: pd.DataFrame, filename=FINAL_SELECTED_FILE):
    """
    Saves or appends the user's final selected classification results DataFrame to a CSV file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_with_time = df_to_save.copy()
    df_with_time["timestamp"] = timestamp # Add the timestamp to the DataFrame

    file_path = pathlib.Path(filename)

    print(f"DEBUG: save_final_selected_results called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{file_path.resolve()}'")
    print(f"DEBUG: Does file exist before writing? {file_path.exists()}")
    print(f"DEBUG: DataFrame to save shape: {df_to_save.shape}")
    print(f"DEBUG: DataFrame columns: {df_to_save.columns.tolist()}")

    try:
        if file_path.exists():
            df_with_time.to_csv(file_path, mode='a', header=False, index=False)
            print(f"DEBUG: Successfully appended final selected results to {filename}.")
        else:
            df_with_time.to_csv(file_path, mode='w', header=True, index=False)
            print(f"DEBUG: Successfully created and saved final selected results to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save/append final selected results to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()
        
        
def log_interaction_event(country, product_type, event_type, hs_code=None, details=""):
    """Logs user interactions, regenerations, and other key events to a CSV file."""
    file_exists = os.path.exists(INTERACTION_LOG_FILE)
    try:
        with open(INTERACTION_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'country', 'product_type', 'event_type', 'hs_code', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'country': country,
                'product_type': product_type,
                'event_type': event_type, # e.g., 'selection', 'rejection', 'regeneration', 'no_code_found'
                'hs_code': hs_code,
                'details': details # e.g., 'option_1', 'option_2'
            })
    except Exception as e:
        print(f"ERROR: Could not write to interaction log file. Reason: {e}")
        


def save_processing_time(processing_time, num_rows, filename=PROCESSING_TIMES_FILE): #
    """
    Saves the processing time for a bulk classification run to a CSV file.
    """
    file_exists = os.path.exists(filename) #
    
    print(f"DEBUG: save_processing_time called for file: '{filename}'")
    print(f"DEBUG: Absolute path attempting to write to: '{pathlib.Path(filename).resolve()}'")
    print(f"DEBUG: Does processing_times file exist before writing? {file_exists}")

    try:
        with open(filename, 'a', newline='') as csvfile: #
            fieldnames = ['timestamp', 'rows_processed', 'total_time_seconds'] #
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) #
            
            if not file_exists: #
                writer.writeheader() #
                
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), #
                'rows_processed': num_rows, #
                'total_time_seconds': round(processing_time, 2) #
            })
        print(f"DEBUG: Successfully saved processing time to {filename}.")
    except Exception as e:
        print(f"ERROR: Could not save processing time to {filename}. Reason: {e}")
        import traceback
        traceback.print_exc()