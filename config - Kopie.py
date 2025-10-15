
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or ""
    API_KEY = ''
    AUTH_USERNAME = os.environ.get('AUTH_USERNAME') or 'your_username'
    AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD') or 'your_password'
    
    # CSV Paths for historical data
    APPAREL_CSV_PATH = "data/raw/PLM_D365_merged_datasets_no_questionary.csv"
    FOOTWEAR_CSV_PATH = "data/raw/train_fold_0.csv"
    
    # Common paths
    REJECTED_CODES_FILE = "data/processed/rejected_classifications.json"
    PDF_DIRECTORY = os.path.join(BASE_DIR, "data/chapter_data")
    
    # Output/Log File Paths
    METRICS_RESULTS_FILE = os.path.join(BASE_DIR, "data/logs/all_bulk_results.csv")
    FINAL_SELECTED_FILE = os.path.join(BASE_DIR, "data/raw/final_user_selected_hs_codes.csv")
    TOKEN_LOG_FILE = os.path.join(BASE_DIR, "data/logs/token_usage_log.csv")
    INTERACTION_LOG_FILE = os.path.join(BASE_DIR, "data/logs/interaction_log.csv")
    PROCESSING_TIMES_FILE = os.path.join(BASE_DIR, "data/logs/processing_times.csv")
    ACCURACY_LOG_FILE = os.path.join(BASE_DIR, "data/logs/accuracy_log_combined.csv")
    
    # JSON "Database" Files
    VALIDATED_CODES_FILE = os.path.join(BASE_DIR, "data/processed/validated_classifications.json")
    
    # HS Code Length Rules
    HS_CODE_LENGTH_RULES = {
        'canada': 10,
        'usa': 10,
        'norway': 8,
        'switzerland': 11,
        'australia': 10,
        'europe': 8,
        'japan': 9,
        'brazil': 8,
        'south korea': 10,
    }
    
    # Product Category Column Mappings
    APPAREL_COLUMNS = {
        'country_col': 'tariff_country_description',
        'name_col': 'customs_description',
        'name_col2': 'product_type',
        'material_col': 'composition',
        'construction_col': 'material_type',
        'gender_col': 'division',
        'hs_code_col': 'tariff_code'
    }
    
    FOOTWEAR_COLUMNS = {
        'country_col': 'tariff_country_description',
        'name_col': 'customs_description',
        'name_col2': 'product_type',
        'material_col': 'material_composition',
        'construction_col': 'outsole_material',
        'gender_col': 'gender_name',
        'size_col': 'size_code',
        'hs_code_col': 'tariff_code'
    }