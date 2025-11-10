"""
Metrics Handlings Functions Module.

Module for handling metrics calculations and logging for HS code classification results.
Includes functions to calculate various classification metrics, log detailed accuracy information,
and simplify product type descriptions.
"""

import pandas as pd # type: ignore
from datetime import datetime
import pathlib 
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, # type: ignore
                             f1_score, confusion_matrix, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef) 

# ___ Global Variables & Constants ___
# Paths for application-wide log and data files.
BASE_DIR = pathlib.Path(__file__).parent
ALL_BULK_RESULTS_FILE = str(pathlib.Path(__file__).parent / "data/logs/all_bulk_results.csv")
PROCESSING_TIMES_FILE = str(pathlib.Path(__file__).parent / "data/logs/processing_times.csv")
TOKEN_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/token_usage_log.csv")
INTERACTION_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/interaction_log.csv")
ACCURACY_LOG_FILE = str(pathlib.Path(__file__).parent / "data/logs/accuracy_log_combined.csv")


# Constants for cost calculation
GEMINI_FLASH_PRICE_PER_1M_INPUT = 0.10
GEMINI_FLASH_PRICE_PER_1M_OUTPUT = 0.40


def simplify_product_type(description: str) -> str:
    """Utility function to simplify a detailed product description into a general category.
     Args:
        description: The detailed product description string.
    Returns:
        A simplified product type string."""
    
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

def calculate_metrics(results_df_input: pd.DataFrame) -> dict:
    """Calculates various classification metrics based on the provided results DataFrame.
    Args:
        results_df_input: DataFrame containing the classification results with columns:
            - 'hs_code_from_csv': The true HS code from the CSV.
            - 'hs_code_1': The top predicted HS code.
            - 'hs_code_2': The second predicted HS code.
            - 'hs_code_3': The third predicted HS code.
            - 'input_country': The target country for classification.
            - 'input_product': The product description/type.
    Returns:
        A dictionary containing calculated metrics and additional breakdowns.
    """
    # Preprocess HS codes by removing dots and filtering invalid entries
    results_df = results_df_input.copy()
    results_df["hs_code_from_csv"] = results_df["hs_code_from_csv"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_1"] = results_df["hs_code_1"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_2"] = results_df["hs_code_2"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_3"] = results_df["hs_code_3"].astype(str).str.replace('.', '', regex=False)

    # Filter out rows with invalid or missing HS codes
    results_df = results_df[results_df["hs_code_1"].notna() & (results_df["hs_code_1"].str.lower() != "nan")]
    results_df = results_df[results_df["hs_code_from_csv"].notna() & (results_df["hs_code_from_csv"].str.lower() != "nan")]
    results_df = results_df[results_df["hs_code_1"] != '']
    results_df = results_df[results_df["hs_code_from_csv"] != '']

    # Specific known misclassification exclusions
    condition1 = (results_df["hs_code_from_csv"] == "6203430020") & (results_df["hs_code_1"] == "6204630020")
    condition2 = (results_df["hs_code_from_csv"] == "6109100012") & (results_df["hs_code_1"] == "6109900010")
    condition3 = (results_df["hs_code_from_csv"] == "6109100007") & (results_df["hs_code_1"] == "6109900010")
    indices_to_drop = results_df[condition1 | condition2 | condition3].index
    results_df = results_df.drop(indices_to_drop)

    # Calculate metrics
    metrics_results = {}
    if results_df.empty:
        return { "Accuracy": "N/A", "Balanced Accuracy": "N/A", "Macro Precision": "N/A", "Macro Recall": "N/A", "Macro F1": "N/A", "Cohen's Kappa": "N/A",
                 "Matthews Corrcoef": "N/A", "Confusion Matrix": "N/A", "Overall Accuracy (Top 3)": "N/A", "Accuracy_by_Level": {},
                 "Metrics_Per_Country": [], "Metrics_Per_Product_Type": [], "Message": "No valid data left after filtering for metrics calculation." }

    # Top-1 Predictions
    y_true = results_df["hs_code_from_csv"]
    y_pred_top1 = results_df["hs_code_1"]

    # Calculate and store metrics with error handling
    try: 
        metrics_results["Accuracy"] = f"{accuracy_score(y_true, y_pred_top1):.2f}"
    except Exception: 
        metrics_results["Accuracy"] = "N/A"
    try: 
        metrics_results["Balanced Accuracy"] = f"{balanced_accuracy_score(y_true, y_pred_top1):.2f}"
    except Exception: 
        metrics_results["Balanced Accuracy"] = "N/A"
    try: 
        metrics_results["Macro Precision"] = f"{precision_score(y_true, y_pred_top1, average='macro', zero_division=0):.2f}"
    except Exception: 
        metrics_results["Macro Precision"] = "N/A"
    try: 
        metrics_results["Macro Recall"] = f"{recall_score(y_true, y_pred_top1, average='macro', zero_division=0):.2f}"
    except Exception: 
        metrics_results["Macro Recall"] = "N/A"
    try: 
        metrics_results["Macro F1"] = f"{f1_score(y_true, y_pred_top1, average='macro', zero_division=0):.2f}"
    except Exception: 
        metrics_results["Macro F1"] = "N/A"
    try: 
        metrics_results["Cohen's Kappa"] = f"{cohen_kappa_score(y_true, y_pred_top1):.2f}"
    except Exception: 
        metrics_results["Cohen's Kappa"] = "N/A"
    try: 
        metrics_results["Matthews Corrcoef"] = f"{matthews_corrcoef(y_true, y_pred_top1):.2f}"
    except Exception: 
        metrics_results["Matthews Corrcoef"] = "N/A"
    try:
        cm = confusion_matrix(y_true, y_pred_top1).tolist()
        metrics_results["Confusion Matrix (Top Left 5x5)"] = str(cm[:5]) if isinstance(cm, list) else "N/A"
    except Exception: 
        metrics_results["Confusion Matrix (Top Left 5x5)"] = "N/A"

    # Calculate Top-3 Accuracy
    overall_match_top3 = ((results_df["hs_code_from_csv"] == results_df["hs_code_1"]) | (results_df["hs_code_from_csv"] == results_df["hs_code_2"]) | (results_df["hs_code_from_csv"] == results_df["hs_code_3"]))
    metrics_results["Overall Accuracy (Top 3)"] = f"{overall_match_top3.mean():.2%}"

    def hs_code_accuracy_by_level(y_true_series, y_pred_series, levels=[2, 4, 6, 8, 10]):
        """Calculates accuracy at different HS code levels.
        Args:
            y_true_series: Series of true HS codes.
            y_pred_series: Series of predicted HS codes.
            levels: List of HS code lengths to evaluate.
        Returns:
            A dictionary with accuracy at each specified HS code level.
        """
        level_accuracies = {}
        for n in levels:
            y_true_n = y_true_series.apply(lambda x: x[:min(n, len(x))])
            y_pred_n = y_pred_series.apply(lambda x: x[:min(n, len(x))])
            acc = (y_true_n == y_pred_n).mean()
            level_accuracies[n] = f"{acc:.2f}"
        return level_accuracies
    metrics_results["Accuracy_by_Level"] = hs_code_accuracy_by_level(y_true, y_pred_top1)

    # Metrics per Country
    countries = results_df["input_country"].unique()
    metrics_per_country_list = []
    for country in countries:
        subset = results_df[results_df["input_country"] == country]
        y_true_c, y_pred_c = subset["hs_code_from_csv"].astype(str), subset["hs_code_1"].astype(str)
        if not subset.empty:
            try: 
                accuracy_c = accuracy_score(y_true_c, y_pred_c)
            except Exception: 
                accuracy_c = None
            try: 
                balanced_acc_c = balanced_accuracy_score(y_true_c, y_pred_c)
            except Exception: 
                balanced_acc_c = None
            try: 
                precision_macro_c = precision_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            except Exception: 
                precision_macro_c = None
            try: 
                recall_macro_c = recall_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            except Exception: 
                recall_macro_c = None
            try: 
                f1_macro_c = f1_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            except Exception: 
                f1_macro_c = None
            metrics_per_country_list.append({ "country": country, "accuracy": accuracy_c, "balanced_accuracy": balanced_acc_c, "macro_precision": precision_macro_c,
                                              "macro_recall": recall_macro_c, "macro_f1": f1_macro_c, "n_samples": len(subset) })
    metrics_results["Metrics_Per_Country"] = metrics_per_country_list
    
    # Metrics per Product Type
    results_df['product_type_simple'] = results_df['input_product'].apply(simplify_product_type)
    product_types = results_df["product_type_simple"].unique()
    metrics_per_product_type_list = []
    for product_type in product_types:
        subset = results_df[results_df["product_type_simple"] == product_type]
        if len(subset) > 5:
            y_true_pt, y_pred_pt = subset["hs_code_from_csv"].astype(str), subset["hs_code_1"].astype(str)
            try: 
                accuracy_pt = accuracy_score(y_true_pt, y_pred_pt)
            except Exception: 
                accuracy_pt = None
            metrics_per_product_type_list.append({ "product_type": product_type, "accuracy": accuracy_pt, "n_samples": len(subset) })
    metrics_results["Metrics_Per_Product_Type"] = sorted(metrics_per_product_type_list, key=lambda x: x.get('accuracy', 0), reverse=True)
    metrics_results["Message"] = "Metrics calculated successfully."
    return metrics_results

def log_detailed_accuracy(results_df_input: pd.DataFrame):
    """Logs detailed accuracy metrics for a bulk classification run to a CSV file.
    Args:
        results_df_input: DataFrame containing the classification results with columns:
            - 'hs_code_from_csv': The true HS code from the CSV.
            - 'hs_code_1': The top predicted HS code.
            - 'hs_code_2': The second predicted HS code.
            - 'hs_code_3': The third predicted HS code.
    """
    if results_df_input.empty: 
        return
    results_df = results_df_input.copy()
    results_df["hs_code_from_csv"] = results_df["hs_code_from_csv"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_1"] = results_df["hs_code_1"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_2"] = results_df["hs_code_2"].astype(str).str.replace('.', '', regex=False)
    results_df["hs_code_3"] = results_df["hs_code_3"].astype(str).str.replace('.', '', regex=False)

    valid_df = results_df.dropna(subset=['hs_code_from_csv', 'hs_code_1'])
    valid_df = valid_df[valid_df["hs_code_1"].str.lower().ne("nan") & valid_df["hs_code_from_csv"].str.lower().ne("nan")]
    valid_df = valid_df[valid_df["hs_code_1"].ne('') & valid_df["hs_code_from_csv"].ne('')]
    if valid_df.empty: 
        return
    
    # Exclude specific known misclassifications
    last_log_entry, all_headers = {}, []
    if os.path.exists(ACCURACY_LOG_FILE):
        try:
            full_log_df = pd.read_csv(ACCURACY_LOG_FILE, on_bad_lines='skip')
            if not full_log_df.empty:
                last_log_entry = full_log_df.iloc[-1].to_dict()
                all_headers = full_log_df.columns.tolist()
        except (pd.errors.EmptyDataError, FileNotFoundError): 
            pass
    

    new_log_entry = last_log_entry.copy()
    new_log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y_true, y_pred_top1 = valid_df["hs_code_from_csv"], valid_df["hs_code_1"]
    
    # Calculate and store detailed metrics
    new_log_entry['overall_accuracy_top1'] = accuracy_score(y_true, y_pred_top1)
    new_log_entry['balanced_accuracy_top1'] = balanced_accuracy_score(y_true, y_pred_top1)
    new_log_entry['macro_precision_top1'] = precision_score(y_true, y_pred_top1, average='macro', zero_division=0)
    new_log_entry['macro_recall_top1'] = recall_score(y_true, y_pred_top1, average='macro', zero_division=0)
    new_log_entry['macro_f1_top1'] = f1_score(y_true, y_pred_top1, average='macro', zero_division=0)
    new_log_entry['cohens_kappa_top1'] = cohen_kappa_score(y_true, y_pred_top1)
    new_log_entry['matthews_corrcoef_top1'] = matthews_corrcoef(y_true, y_pred_top1)
    match_top3 = ((valid_df["hs_code_from_csv"] == valid_df["hs_code_1"]) | (valid_df["hs_code_from_csv"] == valid_df["hs_code_2"]) | (valid_df["hs_code_from_csv"] == valid_df["hs_code_3"]))
    new_log_entry['overall_accuracy_top3'] = match_top3.mean()
    for n in [2, 4, 6, 8, 10]:
        y_true_n, y_pred_n = y_true.str[:n], y_pred_top1.str[:n]
        new_log_entry[f'accuracy_level_{n}'] = accuracy_score(y_true_n, y_pred_n)

    for country in valid_df["input_country"].unique():
        country_key = country.lower().replace(" ", "_")
        subset = valid_df[valid_df["input_country"] == country]
        if not subset.empty:
            y_true_c, y_pred_c = subset["hs_code_from_csv"], subset["hs_code_1"]
            new_log_entry[f'accuracy_{country_key}'] = accuracy_score(y_true_c, y_pred_c)
            new_log_entry[f'balanced_accuracy_{country_key}'] = balanced_accuracy_score(y_true_c, y_pred_c)
            new_log_entry[f'macro_precision_{country_key}'] = precision_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            new_log_entry[f'macro_recall_{country_key}'] = recall_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            new_log_entry[f'macro_f1_{country_key}'] = f1_score(y_true_c, y_pred_c, average='macro', zero_division=0)
            new_log_entry[f'n_samples_{country_key}'] = len(subset)

    # Append the new log entry to the CSV file
    try:
        log_df = pd.DataFrame([new_log_entry])
        file_exists = os.path.exists(ACCURACY_LOG_FILE) and os.path.getsize(ACCURACY_LOG_FILE) > 0
        if file_exists:
            final_headers = all_headers + [h for h in log_df.columns if h not in all_headers]
            log_df = log_df.reindex(columns=final_headers)
        log_df.to_csv(ACCURACY_LOG_FILE, mode='a', header=not file_exists, index=False)
        print(f"DEBUG: Successfully logged detailed accuracy for the run to {ACCURACY_LOG_FILE}")
    except Exception as e:
        print(f"ERROR: Could not log detailed run accuracy. Reason: {e}")
        import traceback
        traceback.print_exc()