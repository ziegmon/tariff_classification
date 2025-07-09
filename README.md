## Tariff Classification

### Install dependencies
```
pip install -r requirements.txt
```

### Download essential files
1. Go to the company's Google Drive and download:
  - the images folder
  - the sepcific csv CSV and place it in this project folder

### Setup secrets.toml file
1. Create a secrets.toml file inside the .streamlit folder

```
API_KEY = ""

[AUTH]
USERNAME = ""
PASSWORD = ""

logo = 
box = 

CSV_PATH = 
```

2. Get a Gemini API Key if you don't have one already and copy it in the secrets.toml file
3. Username and Password can be found in the Google Drive's secrets.toml file
4. For logo, box and CSV_PATH, copy the respective paths to the files in the secrets.toml file

### Run Streamlit App
```
streamlit run app.py
```
