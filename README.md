# AI-Powered HS Code Classification Tool
This project is a Flask-based web application that uses the Google Gemini AI model to assist customs and trade specialists in accurately classifying products with their Harmonized System (HS) codes for international trade.

The application has evolved from a Streamlit prototype into a more robust and scalable solution, featuring a custom user interface, background processing for bulk tasks, and a detailed performance dashboard.


## Features
- Single Product Classification: A user-friendly form to classify a single product by providing its details (country, material, construction, etc.). The tool returns up to three HS code suggestions with detailed, structured reasoning.
- Bulk Classification: Allows users to upload a CSV file containing products. The classification is performed as a background task.
- Review Interface: After bulk processing, results are displayed in an interactive table. Users can sort data, review individual suggestions, accept a code, mark incorrect suggestions, or trigger a regeneration for a specific item on the fly.
- Human-in-the-Loop Learning: The tool learns from expert feedback. When a user marks a code as "incorrect" or validates a correct one, this information is saved and used to improve future suggestions for the same product.
- Performance Dashboard: A comprehensive metrics page visualizes the tool's performance, tracking overall accuracy, accuracy per country, model confidence, user interaction patterns, and operational costs over time.
- Asynchronous Processing: The application leverages asyncio to make concurrent API calls to the Gemini model during bulk processing, reducing the total time required.


## Tech Stack
Backend: Flask

AI Model: Google Gemini 2.0 Flash (via google-generativeai)

Data Processing: Pandas, Scikit-learn

Frontend: Jinja2 Templating, HTML5, CSS3, Vanilla JavaScript, Chart.js

Document Parsing: PyPDF2

Asynchronous Operations: asyncio, threading


## Project Structure
The project has been refactored from a single-script prototype into a modular, service-oriented architecture to improve maintainability and scalability.

bydo_flask_app/

├── data/
│   ├── chapter_data/             # Contains all source PDF/TXT tariff documents.
│   ├── datasets/                 # For test datasets.
│   └── logs/                     # Output location for all generated CSV logs.
├── notebooks/                    # Jupyter notebooks for experimentation and analysis.
├── static/
│   ├── css/style.css
│   └── img/
├── templates/
│   └── all  HTML templates
├── app.py                        # Main Flask application: routes and controller logic.
├── config.py                     # Central configuration for paths and secrets.
├── data_processing_functions.py  # Logic for parsing AI responses and historical data.
├── document_handling_helper_functions.py # Logic for reading and parsing source documents.
├── gemini_helper_functions.py    # Manages all interaction with the Gemini API.
├── persistence_helper_functions.py # Handles all file I/O for logs and user feedback.
├── requirements.txt              # Project dependencies.
└── README.md  


## Setup and Installation

### 1. Prerequisites
Python 3.12
A Google Cloud Project with the Generative AI API enabled.
An API Key for the Google Generative AI API.

### 2. Create and Activate a Virtual Environment
#### Create the environment
    -m venv venv

#### Activate on macOS/Linux
    source venv/bin/activate

#### Activate on Windows
    .\venv\Scripts\activate


### 3. Install Dependencies
Install all required Python packages from the requirements.txt file.
    pip install -r requirements.txt


### 4. Configure the Application
All configuration is managed in the config.py file. 


### 5. Prepare Data Directories

>> At the moment, the files are stored locally. Moving forward, files generated and used for generation purposes should be migrated to a proper database.   

**chapter_data/**: The user should make sure this directory exists and is populated with the official tariff PDF documents and any text-based guidelines. The files must follow the naming convention: {country}_chapter_{number}.pdf (e.g., usa_chapter_62.pdf).

**Historical Data CSV**: Ensure that the historical data file (e.g., PLM_D365_merged_datasets_no_questionary.csv) is present in the root directory and its name matches the one in config.py.


## Running the Application
Once the setup is complete, the user can run the Flask development server:
    python app.py

The application will be available at http://127.0.0.1:5000 (web browser).


## Usage Guide
- Login: The user should access the application and log in with the credentials set in configuration (hardcoded for development purposes, but **WIP**).
- Single Classification: Navigate to the "Single Product" page, fill in the product details, and generate suggestions. Review the reasoning and provide feedback by marking incorrect options.
- Bulk Classification:
    - Navigate to the "Bulk Classification" page.
    - Upload a CSV file with your product data. The user will be redirected to a progress page and should wait for the processing to complete.
    - On the review table, the user should inspect the proposed codes. Use the icons to accept (✓), reject (✗), or regenerate (↻) suggestions for each item.
    - Once you are done reviewing, click "Process All Selections" to finalize the batch. (**WIP**)
Download the final report as a CSV file.
