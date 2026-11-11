SMART REAL ESTATE AI - CALIFORNIA HOUSING

CONTENTS
- dataset/california_housing.csv: your supplied dataset
- train_models.py: complete reproducible training program
- models/: trained Decision Tree and Random Forest models
- results/: metrics, test predictions, feature importance, summary
- plots/: four generated charts

RUN IN PYTHON IDLE
1. Extract this ZIP to a normal folder.
2. Open train_models.py in IDLE.
3. Press F5 (Run > Run Module).
4. Results are written into models, results, and plots.

INSTALL PACKAGES
Use Command Prompt (not the IDLE Shell):
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn joblib

If IDLE uses a different Python installation, run the install command with
the exact interpreter path printed by this in IDLE:
import sys
print(sys.executable)

METHODOLOGY
- Target: MedHouseVal
- Features: all other eight columns
- Train/test split: 80% / 20%
- random_state: 42
- Decision Tree: min_samples_leaf=2
- Random Forest: 200 trees, min_samples_leaf=1
- Metrics: MAE, RMSE, R-squared
- Model selection: lowest held-out test RMSE

IMPORTANT DATASET INTERPRETATION
California Housing rows represent census block groups. MedHouseVal is the
median house value for a block group, expressed in units of $100,000.
Predictions are regional estimates and must not be presented as precise
valuations of individual properties.

The included model files and results were generated from the uploaded CSV.
Training/prediction times vary by computer.
