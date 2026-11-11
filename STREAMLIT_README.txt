SMART REAL ESTATE AI — COMPLETE STREAMLIT DASHBOARD
===================================================

FILES INCLUDED
app.py                  Interactive Streamlit dashboard
train_models.py         Reproducible training code
requirements.txt        Required Python packages
RUN_DASHBOARD_WINDOWS.bat  Double-click launcher (if 'python' is on PATH)
dataset/                Original user-supplied California Housing CSV
models/                 Already trained Decision Tree and Random Forest models
results/                Real model metrics and test predictions
plots/                  Four generated static figures

HOW TO RUN ON WINDOWS WITH PYTHON IDLE
1. Extract the ZIP file completely to a folder (do not run inside the ZIP).
2. Open Windows Command Prompt in that extracted folder.
3. Install packages:
   python -m pip install -r requirements.txt
4. Start the dashboard:
   python -m streamlit run app.py
5. Open the URL shown by Streamlit, usually http://localhost:8501
6. Stop the dashboard by pressing Ctrl+C in Command Prompt.

IMPORTANT: IDLE is suitable for editing app.py, but Streamlit must be launched
from Command Prompt or a terminal, not by pressing F5 in IDLE.

IF 'NO MODULE NAMED JOBLIB' OR ANOTHER MODULE APPEARS
In IDLE, execute:
    import sys
    print(sys.executable)
Copy the exact path it prints and install dependencies using that interpreter:
    "C:\full\path\to\python.exe" -m pip install -r requirements.txt
Then launch Streamlit with the same interpreter:
    "C:\full\path\to\python.exe" -m streamlit run app.py

DASHBOARD TABS
- Predict: enter 8 block-group characteristics, choose model, predict
- Model comparison: actual MAE, RMSE, R² metrics
- Prediction diagnostics: scatter, residuals, Random Forest feature importance
- Geographic analysis: interactive California map
- Dataset: records, preview, summary statistics

NOTES
- Data: California Housing, 20,640 census block groups.
- Target: MedHouseVal in units of $100,000 (not an individual home price).
- Input MedInc in units of $10,000.
- Model training used an 80/20 split and random_state=42.
- The dataset's target is capped at approximately $500,000.
- This is a project demonstration, not an appraisal or financial decision tool.
- Trained joblib files should only be loaded from trusted sources.
- Training time and software versions may affect repeatability of exact output.
