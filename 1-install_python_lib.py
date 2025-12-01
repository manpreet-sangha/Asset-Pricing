# Import required libraries for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
import sklearn

# Statistical and econometric libraries
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import reset_ramsey

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure plotting parameters
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Set random seed for reproducibility
np.random.seed(42)

# Define global constants
BASE_PATH = Path(r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1")
RAW_DATA_PATH = BASE_PATH / "Data" / "Raw Data"
CLEANED_DATA_PATH = BASE_PATH / "Data" / "Cleaned Data"
OUTPUT_PATH = BASE_PATH / "Python Scripts" / "Asset-Pricing" / "outputs"

# Create output directories if they don't exist
for path in [OUTPUT_PATH, OUTPUT_PATH / "tables", OUTPUT_PATH / "figures"]:
    path.mkdir(parents=True, exist_ok=True)

# Key dates for analysis
TRAINING_START = '1990-01-31'  # Start of training period
FORECAST_START = '2015-10-31'  # Start of out-of-sample period
FORECAST_END = '2025-10-31'    # End of evaluation period
FORECAST_HORIZON = 6           # 6-month forecast horizon

print("Libraries imported successfully!")
print(f"Analysis period: {TRAINING_START} to {FORECAST_END}")
print(f"Out-of-sample evaluation: {FORECAST_START} to {FORECAST_END}")
print(f"Forecast horizon: {FORECAST_HORIZON} months")
print(f"Output directory: {OUTPUT_PATH}")

# Display key thresholds from the paper
print("\n" + "="*60)
print("EXPECTED RESULTS BASED ON PESARAN & TIMMERMANN (1994)")
print("="*60)
print("Expected R²: 0.15 - 0.30 (between quarterly and annual)")
print("Expected Sign Accuracy: 55% - 70%")
print("Expected PT Statistic: 1.5 - 2.5")
print("Critical Values - PT Test: 1.645 (5%), 2.326 (1%)")
print("="*60)