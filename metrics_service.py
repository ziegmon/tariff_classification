"""Metrics Service Module.

This module contains all the business logic for calculating and aggregating
metrics for the dashboard. It is responsible for reading data from various
log files (bulk results, tokens, interactions, etc.), performing the necessary
calculations using pandas and scikit-learn, and returning a structured
dictionary of data ready to be rendered by the Flask template.
"""

import os
import pandas as pd
from flask import Flask
from config import Config
from datetime import datetime
import pathlib
from typing import Dict, Any
from metrics_handlings_functions import calculate_metrics

# ___ Constants & File Paths ___
# Centralize file paths here or import from a config object.
BASE_DIR = pathlib.Path(__file__).parent
ALL_BULK_RESULTS_FILE = str(BASE_DIR / "data/logs/all_bulk_results.csv")
PROCESSING_TIMES_FILE = str(BASE_DIR / "data/logs/processing_times.csv")
TOKEN_LOG_FILE = str(BASE_DIR / "data/logs/token_usage_log.csv")
INTERACTION_LOG_FILE = str(BASE_DIR / "data/logs/interaction_log.csv")
ACCURACY_LOG_FILE = str(BASE_DIR / "data/logs/accuracy_log_combined.csv")


app = Flask(__name__)
app.config.from_object(Config)

TOKEN_LOG_FILE = app.config["TOKEN_LOG_FILE"]
REJECTED_CODES_FILE = app.config["REJECTED_CODES_FILE"]
VALIDATED_CODES_FILE = app.config["VALIDATED_CODES_FILE"]
ALL_BULK_RESULTS_FILE = app.config["METRICS_RESULTS_FILE"]
FINAL_SELECTED_FILE = app.config["FINAL_SELECTED_FILE"]
INTERACTION_LOG_FILE = app.config["INTERACTION_LOG_FILE"]
PROCESSING_TIMES_FILE = app.config["PROCESSING_TIMES_FILE"]
ACCURACY_LOG_FILE = app.config["ACCURACY_LOG_FILE"]


# Constants for cost calculation
GEMINI_FLASH_PRICE_PER_1M_INPUT = 0.10
GEMINI_FLASH_PRICE_PER_1M_OUTPUT = 0.40

def get_dashboard_data() -> Dict[str, Any]:
    """
    The main orchestrator function that gathers all data for the metrics dashboard.

    Returns:
        A dictionary containing all aggregated data needed by the metrics template.
    """
    try:
        if not os.path.exists(ALL_BULK_RESULTS_FILE):
            return {"error": "No bulk classification results found. Please run a bulk classification first."}

        all_results_df = pd.read_csv(ALL_BULK_RESULTS_FILE)

        # Each function call gathers a specific part of the dashboard data
        metrics_data = _get_accuracy_metrics(all_results_df)
        evolution_data = _get_evolution_data()
        processing_times_data, overall_avg_time_per_row = _get_processing_time_data()
        token_data, cost_data, usage_data = _get_token_and_cost_data()
        quality_data, interaction_data = _get_interaction_and_quality_data(token_data.get('total_requests', 0))
        completeness_data = _get_completeness_data(all_results_df)

        # Combine all data into a single dictionary
        return {
            "metrics": metrics_data,
            "processing_times": processing_times_data,
            "overall_avg_time_per_row": overall_avg_time_per_row,
            "token_data": token_data,
            "cost_data": cost_data,
            "quality_data": quality_data,
            "usage_data": usage_data,
            "interaction_data": interaction_data,
            "completeness_data": completeness_data,
            "evolution_data": evolution_data,
        }

    except Exception as e:
        print(f"ERROR: Failed to generate dashboard data. Reason: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"An unexpected error occurred while generating metrics: {e}"}


def _get_accuracy_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates the core classification accuracy metrics."""
    return calculate_metrics(df)


def _get_evolution_data() -> Dict[str, Any]:
    """Processes the accuracy log to show performance evolution over time."""
    evolution_data = {}
    if not os.path.exists(ACCURACY_LOG_FILE):
        return evolution_data

    try:
        accuracy_all_df = pd.read_csv(ACCURACY_LOG_FILE)
        if accuracy_all_df.empty:
            return evolution_data
            
        accuracy_all_df['timestamp'] = pd.to_datetime(accuracy_all_df['timestamp'])
        accuracy_all_df.sort_values('timestamp', inplace=True)

        # --- Overall Performance Evolution ---
        overall_cols = ['timestamp', 'overall_accuracy_top1', 'balanced_accuracy_top1', 'macro_f1_top1']
        existing_overall_cols = [col for col in overall_cols if col in accuracy_all_df.columns]
        if len(existing_overall_cols) > 1:
            overall_df = accuracy_all_df[existing_overall_cols].set_index('timestamp')
            # Resample to daily, forward-fill missing days
            resampled_df = overall_df.resample('D').last()
            full_range = pd.date_range(start=resampled_df.index.min(), end=datetime.now(), freq='D')
            daily_df = resampled_df.reindex(full_range).ffill()
            
            evo_dict = daily_df.reset_index().to_dict('list')
            evo_dict['timestamp'] = [ts.strftime('%Y-%m-%d') for ts in evo_dict['index']]
            del evo_dict['index']
            evolution_data['overall'] = evo_dict

        # --- Country Accuracy Evolution ---
        country_cols = [col for col in accuracy_all_df.columns if col.startswith('accuracy_') and '_level_' not in col]
        if country_cols:
            country_df = accuracy_all_df[['timestamp'] + country_cols].set_index('timestamp')
            country_df.rename(columns=lambda c: c.replace('accuracy_', '').replace('_', ' ').title(), inplace=True)
            
            resampled_df = country_df.resample('D').last()
            full_range = pd.date_range(start=resampled_df.index.min(), end=datetime.now(), freq='D')
            daily_df = resampled_df.reindex(full_range).ffill()
            
            evo_dict = daily_df.reset_index().to_dict('list')
            evo_dict['timestamp'] = [ts.strftime('%Y-%m-%d') for ts in evo_dict['index']]
            del evo_dict['index']
            evolution_data['by_country'] = evo_dict

    except Exception as e:
        print(f"Warning: Could not process accuracy evolution logs: {e}")
        
    return evolution_data


def _get_processing_time_data() -> (list, float):
    """Reads and calculates processing time statistics."""
    if not os.path.exists(PROCESSING_TIMES_FILE):
        return [], 0
        
    proc_times_df = pd.read_csv(PROCESSING_TIMES_FILE)
    if proc_times_df.empty:
        return [], 0
        
    proc_times_df['timestamp'] = pd.to_datetime(proc_times_df['timestamp'])
    proc_times_df['average'] = (proc_times_df['total_time_seconds'] / proc_times_df['rows_processed'])
    overall_avg = proc_times_df['average'].mean()
    
    return proc_times_df.to_dict('records'), overall_avg


def _get_token_and_cost_data() -> (Dict[str, Any], Dict[str, Any], Dict[str, Any]):
    """Reads the token log and calculates usage and estimated cost metrics."""
    token_data, cost_data, usage_data = {}, {}, {}
    if not os.path.exists(TOKEN_LOG_FILE):
        return token_data, cost_data, usage_data
        
    token_df = pd.read_csv(TOKEN_LOG_FILE)
    if token_df.empty:
        return token_data, cost_data, usage_data

    token_df['timestamp'] = pd.to_datetime(token_df['timestamp'])
    token_data['total_requests'] = len(token_df)

    # Token Usage
    daily_summary = token_df.resample('D', on='timestamp').agg(total_tokens=('total_tokens', 'sum'), request_count=('total_tokens', 'size')).reset_index()
    token_data['daily'] = daily_summary.to_dict('records')
    token_data['by_country'] = token_df.groupby('country')['total_tokens'].sum().sort_values(ascending=False).reset_index().to_dict('records')
    token_data['by_product_type'] = token_df.groupby('product_type')['total_tokens'].sum().sort_values(ascending=False).head(15).reset_index().to_dict('records')
    
    # Cost Estimation
    token_df['estimated_cost'] = ((token_df['input_tokens'] / 1_000_000) * GEMINI_FLASH_PRICE_PER_1M_INPUT) + \
                               ((token_df['output_tokens'] / 1_000_000) * GEMINI_FLASH_PRICE_PER_1M_OUTPUT)
    
    cost_data['daily'] = token_df.resample('D', on='timestamp').agg(estimated_cost=('estimated_cost', 'sum')).reset_index().to_dict('records')
    cost_data['by_country'] = token_df.groupby('country')['estimated_cost'].sum().sort_values(ascending=False).reset_index().to_dict('records')
    cost_data['by_product_type'] = token_df.groupby('product_type')['estimated_cost'].sum().sort_values(ascending=False).head(15).reset_index().to_dict('records')
    
    total_cost = token_df['estimated_cost'].sum()
    cost_data['cost_per_1000'] = (total_cost / len(token_df)) * 1000 if len(token_df) > 0 else 0

    # Usage Patterns
    usage_data['by_hour'] = token_df.groupby(token_df['timestamp'].dt.hour).size().reset_index(name='request_count').to_dict('records')

    return token_data, cost_data, usage_data


def _get_interaction_and_quality_data(total_requests: int) -> (Dict[str, Any], Dict[str, Any]):
    """Analyzes user interactions to derive quality metrics."""
    quality_data, interaction_data = {}, {}
    if not os.path.exists(INTERACTION_LOG_FILE):
        return quality_data, interaction_data

    interaction_df = pd.read_csv(INTERACTION_LOG_FILE)
    if interaction_df.empty:
        return quality_data, interaction_data

    # Quality Metrics
    if total_requests > 0:
        regenerations = (interaction_df['event_type'] == 'regeneration').sum()
        quality_data['regeneration_rate'] = (regenerations / total_requests) * 100
        no_code_found = (interaction_df['event_type'] == 'no_code_found').sum()
        quality_data['no_code_found_rate'] = (no_code_found / total_requests) * 100
        
    # Interaction Analysis
    selections_df = interaction_df[interaction_df['event_type'] == 'selection']
    if not selections_df.empty:
        interaction_data['selection_distribution'] = selections_df['details'].value_counts().reset_index().rename(columns={'index': 'option', 'details': 'count'}).to_dict('records')
    
    rejections_df = interaction_df[interaction_df['event_type'] == 'rejection']
    if not rejections_df.empty:
        interaction_data['top_rejected_codes'] = rejections_df['hs_code'].value_counts().head(10).reset_index().rename(columns={'index': 'hs_code', 'hs_code': 'count'}).to_dict('records')
        
    return quality_data, interaction_data


def _get_completeness_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes the results data for content and certainty patterns."""
    completeness_data = {}
    
    df["hs_code_from_csv"] = df["hs_code_from_csv"].astype(str).str.replace('.', '', regex=False)
    df["hs_code_1"] = df["hs_code_1"].astype(str).str.replace('.', '', regex=False)
    correct_mask = df["hs_code_from_csv"] == df["hs_code_1"]

    # Certainty analysis
    completeness_data['avg_certainty'] = {
        'correct': df[correct_mask]['certainty_1'].mean(),
        'incorrect': df[~correct_mask]['certainty_1'].mean()
    }
    bins, labels = [0, 80, 90, 95, 101], ['<80%', '80-89%', '90-94%', '95-100%']
    df['certainty_bucket'] = pd.cut(df['certainty_1'], bins=bins, labels=labels, right=False)
    completeness_data['certainty_distribution'] = df['certainty_bucket'].value_counts().reset_index().rename(columns={'index': 'bucket', 'certainty_bucket': 'count'}).to_dict('records')

    # Content analysis
    completeness_data['top_materials'] = df['input_material'].value_counts().head(10).reset_index().rename(columns={'index': 'material', 'input_material': 'count'}).to_dict('records')
    completeness_data['top_constructions'] = df['input_construction'].value_counts().head(10).reset_index().rename(columns={'index': 'construction', 'input_construction': 'count'}).to_dict('records')

    # Misclassification analysis
    misclassified_df = df[~correct_mask & df['hs_code_from_csv'].notna() & df['hs_code_1'].notna()]
    if not misclassified_df.empty:
        top_pairs = misclassified_df.groupby(['hs_code_from_csv', 'hs_code_1']).size().reset_index(name='count').nlargest(10, 'count')
        completeness_data['top_misclassified_pairs'] = top_pairs.to_dict('records')
        
    return completeness_data