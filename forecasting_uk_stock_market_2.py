# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
from datetime import datetime, timedelta

# Import specific statsmodels components instead of the full api
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import jarque_bera, durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some statsmodels features unavailable: {e}")
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define paths
BASE_PATH = Path(r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1")
OUTPUT_PATH = BASE_PATH / "Python Scripts" / "Asset-Pricing" / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

class PesaranTimmermannMultiPredictor:
    """
    Pesaran & Timmermann (1995) replication using UK data with multiple predictors
    Predictability model using Dividend Yield AND UK CPI to predict stock market excess returns
    
    METHODOLOGY (Rolling Window Approach with Multiple Predictors):
    Model: Excess_Return = α + β₁ × Dividend_Yield + β₂ × CPI_Growth + ε
    
    1. Initial Training: Apr 1997 - Sep 2015 (estimate α̂, β̂₁, β̂₂)
    2. Rolling Predictions: Re-estimate coefficients every 6 months
       - Oct 2015: Train on Apr 1997→Sep 2015 → Get α̂₁, β̂₁₁, β̂₁₂ → Predict Oct 2015-Mar 2016
       - Apr 2016: Train on Apr 1997→Mar 2016 → Get α̂₂, β̂₂₁, β̂₂₂ → Predict Apr 2016-Sep 2016
       - Oct 2016: Train on Apr 1997→Sep 2016 → Get α̂₃, β̂₃₁, β̂₃₂ → Predict Oct 2016-Mar 2017
       - ... and so on
    3. Investment Strategy: Switch between stocks and InterBank based on predicted excess returns
    """
    
    def __init__(self):
        # TRAINING PHASE: Apr 1997 → Sep 2015 (to estimate α̂ and β̂)
        self.training_start = '1997-04-30'
        self.training_end = '2015-09-30'  # Train until September 2015
        
        # PREDICTION PHASE: Oct 2015 → Oct 2025 (forward-looking prediction)
        self.prediction_start = '2015-10-31'  # Start predicting from October 2015
        self.prediction_end = '2025-04-30'  # Predict until October 2025
        
        self.data = None
        self.training_data = None
        self.prediction_data = None
        self.model = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load merged data file with InterBank rate and UK CPI"""
        print("PESARAN & TIMMERMANN MULTI-PREDICTOR PREDICTABILITY MODEL")
        print("="*75)
        print("📊 Loading merged data with multiple predictors...")
        
        # Load the FTSE Index and Dividend Yield data
        index_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265_Index_DivYield_Monthly.xlsx"
        
        # Load the InterBank rate data
        interbank_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265_UK_InterBank_Rate_Monthly.xlsx"
        
        # Load UK CPI data
        cpi_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265 - UKRPCJYR Index - UK CPI YoY - Monthly.xlsx"
        
        try:
            # Load the index and dividend yield data
            index_data = pd.read_excel(index_file)
            
            # Load the InterBank rate data
            interbank_data = pd.read_excel(interbank_file)
            
            # Load UK CPI data
            try:
                cpi_data = pd.read_excel(cpi_file)
                print(f"   ✅ UK CPI data loaded: {cpi_data.shape}")
            except FileNotFoundError:
                print(f"   ⚠️  UK CPI file not found at {cpi_file}")
                print(f"   📋 Creating synthetic CPI data for demonstration purposes...")
                # Create synthetic CPI data if file doesn't exist
                cpi_data = self.create_synthetic_cpi_data(index_data)
            
            # Debug: Show what we loaded
            print(f"   🔍 Index data shape: {index_data.shape}")
            print(f"   🔍 Index columns: {list(index_data.columns)}")
            print(f"   🔍 InterBank data shape: {interbank_data.shape}")
            print(f"   🔍 InterBank columns: {list(interbank_data.columns)}")
            print(f"   🔍 CPI data shape: {cpi_data.shape}")
            print(f"   🔍 CPI columns: {list(cpi_data.columns)}")
            
            # Show first few rows of all datasets
            print(f"\n📋 INDEX DATA SAMPLE:")
            print(index_data.head(3))
            print(f"\n📋 INTERBANK DATA SAMPLE:")
            print(interbank_data.head(3))
            print(f"\n📋 CPI DATA SAMPLE:")
            print(cpi_data.head(3))
            
            # Rename columns in index data
            index_data = index_data.rename(columns={
                'astrx_index': 'ftse_price',
                'astrx_div_yield': 'dividend_yield'
            })
            
            # Ensure InterBank data has correct column names
            if 'interbank_rate' not in interbank_data.columns:
                # Try common column names for the rate
                rate_columns = [col for col in interbank_data.columns if 'rate' in col.lower() or 'yield' in col.lower()]
                if rate_columns:
                    interbank_data = interbank_data.rename(columns={rate_columns[0]: 'interbank_rate'})
                elif len(interbank_data.columns) >= 2:
                    # Assume second column is the rate if first is date
                    interbank_data = interbank_data.rename(columns={interbank_data.columns[1]: 'interbank_rate'})
            
            # Ensure CPI data has correct column names
            if 'cpi_yoy' not in cpi_data.columns:
                # Try common column names for CPI
                cpi_columns = [col for col in cpi_data.columns if 'cpi' in col.lower() or 'inflation' in col.lower()]
                if cpi_columns:
                    cpi_data = cpi_data.rename(columns={cpi_columns[0]: 'cpi_yoy'})
                elif len(cpi_data.columns) >= 2:
                    # Assume second column is the CPI if first is date
                    cpi_data = cpi_data.rename(columns={cpi_data.columns[1]: 'cpi_yoy'})
            
            # Convert dates to datetime - using lowercase 'date' for all
            index_data['date'] = pd.to_datetime(index_data['date'])
            interbank_data['date'] = pd.to_datetime(interbank_data['date'])
            cpi_data['date'] = pd.to_datetime(cpi_data['date'])
            
            # Create year-month columns for matching (CCYY-MM format)
            index_data['year_month'] = index_data['date'].dt.to_period('M')
            interbank_data['year_month'] = interbank_data['date'].dt.to_period('M')
            cpi_data['year_month'] = cpi_data['date'].dt.to_period('M')
            
            print(f"   🔍 Index date range: {index_data['year_month'].min()} to {index_data['year_month'].max()}")
            print(f"   🔍 InterBank date range: {interbank_data['year_month'].min()} to {interbank_data['year_month'].max()}")
            print(f"   🔍 CPI date range: {cpi_data['year_month'].min()} to {cpi_data['year_month'].max()}")
            
            # Find common year-months across all three datasets
            common_months_step1 = set(index_data['year_month']) & set(interbank_data['year_month'])
            common_months_all = common_months_step1 & set(cpi_data['year_month'])
            print(f"   🔍 Common year-months (Index + InterBank): {len(common_months_step1)}")
            print(f"   🔍 Common year-months (All three datasets): {len(common_months_all)}")
            
            # First merge: Index + InterBank
            temp_merged = pd.merge(
                index_data, 
                interbank_data[['year_month', 'interbank_rate']],
                left_on='year_month', 
                right_on='year_month',
                how='inner'
            )
            
            # Second merge: Add CPI data to get the complete multi-predictor dataset
            self.data = pd.merge(
                temp_merged,
                cpi_data[['year_month', 'cpi_yoy']],
                left_on='year_month',
                right_on='year_month', 
                how='inner'
            )
            
            # Remove the temporary year_month column from merged data
            self.data = self.data.drop('year_month', axis=1)
            
            # Rename for consistency with existing code
            self.data = self.data.rename(columns={
                'interbank_rate': 'm_interbank_rate',  # Use descriptive variable name
                'date': 'Date'  # Standardize to capital D for consistency with rest of code
            })
            
            # Sort by date
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            # Save merged data for verification
            output_dir = BASE_PATH / "Data" / "Output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "SMM265_PT_Model_DivYield_CPI_Merged_Data.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.data.to_excel(writer, sheet_name='Data', index=False)
            
            print(f"   💾 Merged data saved to: {output_file.name}")
            print(f"   ✅ Multi-predictor dataset: {len(self.data)} observations")
            print(f"   📅 Period: {self.data['Date'].min().strftime('%Y-%m')} to {self.data['Date'].max().strftime('%Y-%m')}")
            print(f"   📊 Columns: {list(self.data.columns)}")
            print(f"   📈 Predictors: Dividend Yield + UK CPI YoY + UK InterBank Rate")
            
            # Debug: Show first and last few dates
            print(f"   🔍 First 5 dates: {self.data['Date'].head().dt.strftime('%Y-%m-%d').tolist()}")
            print(f"   🔍 Last 5 dates: {self.data['Date'].tail().dt.strftime('%Y-%m-%d').tolist()}")
            
            # Show sample of merged data for verification
            print(f"\n📋 MERGED DATA SAMPLE:")
            print(self.data.head(3).to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            print(f"   📁 Expected index file: {index_file}")
            print(f"   📁 Expected InterBank file: {interbank_file}")
            return False
    
    def compute_returns(self):
        """Prepare data structure for rolling window - returns computed dynamically for each window"""
        print("📈 Preparing data for rolling window predictions...")
        print("   🎯 Returns will be computed dynamically for each expanding window:")
        print("      • 1st prediction (Oct 2015): Training Apr 1997 - Sep 2015 (~222 obs)")
        print("      • 2nd prediction (Apr 2016): Training Apr 1997 - Mar 2016 (~228 obs)")
        print("      • 3rd prediction (Oct 2016): Training Apr 1997 - Sep 2016 (~234 obs)")
        print("      • Each window expands by 6 months of new data")
        
        df = self.data.copy()
        
        # Sort by date to ensure proper chronological order
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Data prepared for rolling window analysis")
        print(f"   📊 Total available observations: {len(df)}")
        print(f"   � Data range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   � Predictors available: Dividend Yield + CPI Growth + InterBank Rate")
        print(f"   � Returns and excess returns will be computed for each training window")
        
        # Store the full dataset for rolling window use
        self.full_data = df  # Keep full data for rolling window
        self.data = df

    def prepare_initial_training_data(self):
        """Prepare initial training data for reference (ROLLING WINDOW USES EXPANDING WINDOWS)"""
        print("📚 Preparing initial training data for reference...")
        print("⚠️  NOTE: This is NOT used in rolling window - each iteration uses expanding training windows")
        
        # First, let's debug what data we have (use full_data for rolling window)
        if hasattr(self, 'full_data'):
            print(f"   📊 Full data range: {self.full_data['Date'].min().strftime('%Y-%m-%d')} to {self.full_data['Date'].max().strftime('%Y-%m-%d')}")
            print(f"   📊 Total observations: {len(self.full_data)}")
            data_source = self.full_data
        else:
            print(f"   📊 Available data range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
            print(f"   📊 Total observations: {len(self.data)}")
            data_source = self.data
        
        # Convert date strings to datetime
        training_start = pd.to_datetime(self.training_start)
        training_end = pd.to_datetime(self.training_end)
        prediction_start = pd.to_datetime(self.prediction_start)
        prediction_end = pd.to_datetime(self.prediction_end)
        
        print(f"   🎯 Initial training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
        print(f"   🎯 Rolling prediction starts: {prediction_start.strftime('%Y-%m-%d')}")
        
        # Training period: Apr 1997 → Sep 2015 - contains PREDICTORS ONLY (no excess returns)
        # Excess returns are computed dynamically in rolling window for each training period
        training_mask = (self.data['Date'] >= training_start) & (self.data['Date'] <= training_end)
        self.training_data = self.data[training_mask].copy().reset_index(drop=True)
        
        print(f"   📊 Initial training data range: {self.training_data['Date'].min().strftime('%Y-%m-%d')} to {self.training_data['Date'].max().strftime('%Y-%m-%d')} ({len(self.training_data)} obs)")
        
        # Verify that training data has the required PREDICTOR columns 
        required_cols = ['dividend_yield', 'cpi_yoy', 'ftse_price', 'm_interbank_rate']
        missing_cols = [col for col in required_cols if col not in self.training_data.columns]
        if missing_cols:
            print(f"   ⚠️  Missing columns in training data: {missing_cols}")
        else:
            print(f"   ✅ Training data has all required PREDICTOR columns: {required_cols}")
            # Show data quality for predictors
            valid_dividend = self.training_data['dividend_yield'].notna().sum()
            valid_cpi = self.training_data['cpi_yoy'].notna().sum()
            print(f"   📊 Valid dividend yield: {valid_dividend}/{len(self.training_data)} observations")
            print(f"   📊 Valid CPI data: {valid_cpi}/{len(self.training_data)} observations")
            print("   📝 NOTE: Excess returns computed dynamically in rolling window!")
        
        # Prediction period: Oct 2015 → Oct 2025 (use full_data for rolling window)
        max_available_date = data_source['Date'].max()
        if prediction_end > max_available_date:
            print(f"   ⚠️  Warning: Prediction end ({prediction_end.strftime('%Y-%m-%d')}) beyond available data ({max_available_date.strftime('%Y-%m-%d')})")
            print(f"   📅 Using available data until: {max_available_date.strftime('%Y-%m-%d')}")
            prediction_end = max_available_date
        
        prediction_mask = (data_source['Date'] >= prediction_start) & (data_source['Date'] <= prediction_end)
        self.prediction_data = data_source[prediction_mask].copy().reset_index(drop=True)
        
        print(f"   📚 Initial training period: {len(self.training_data)} observations")
        if len(self.training_data) > 0:
            print(f"      {self.training_data['Date'].min().strftime('%Y-%m-%d')} to {self.training_data['Date'].max().strftime('%Y-%m-%d')}")
        else:
            print("      ❌ No training data found!")
        
        print(f"   🔮 Rolling prediction data: {len(self.prediction_data)} observations available")
        if len(self.prediction_data) > 0:
            print(f"      {self.prediction_data['Date'].min().strftime('%Y-%m-%d')} to {self.prediction_data['Date'].max().strftime('%Y-%m-%d')}")
        else:
            print("      ❌ No prediction data found!")
            
        print(f"\n💡 ACTUAL ROLLING WINDOW METHODOLOGY:")
        print(f"   1. EXPANDING WINDOWS: Each prediction uses progressively more training data")
        print(f"   2. Oct 2015 prediction: Train on Apr 1997 → Mar 2015 (get α̂₁, β̂₁)")
        print(f"   3. Apr 2016 prediction: Train on Apr 1997 → Sep 2015 (get α̂₂, β̂₂)")
        print(f"   4. Oct 2016 prediction: Train on Apr 1997 → Mar 2016 (get α̂₃, β̂₃)")
        print(f"   5. Each iteration ADDS 6 months of new training data")

    def train_model(self):
        """REFERENCE METHOD - Shows multi-predictor model structure
        
        NOTE: This method is for reference only. In the actual rolling window methodology,
        excess returns are computed dynamically for each expanding training window.
        The real training happens in train_rolling_model() during rolling_window_predictions().
        """
        print("\n🎓 TRAINING PHASE - MULTI-PREDICTOR MODEL (REFERENCE)")
        print("="*60)
        print("OBJECTIVE: Estimate α̂, β̂₁ and β̂₂ coefficients")
        print("Model: Excess_Return = α + β₁ × Dividend_Yield + β₂ × CPI_Growth + ε")
        print(f"Training period: {self.training_start} to {self.training_end}")
        print(f"For predicting: {self.prediction_start} onwards")
        print("\n📝 METHODOLOGY OVERVIEW:")
        print("1. 🔄 Each rolling window computes excess returns dynamically for that training period")
        print("2. 🔄 Uses EXPANDING windows: Apr 1997 → Sep 2015, then Apr 1997 → Mar 2016, etc.")
        print("3. ✅ Uses TWO predictors: X₁ = Dividend_Yield_t, X₂ = CPI_Growth_t")
        print("4. 🔄 Runs OLS regression: Y = α + β₁ × X₁ + β₂ × X₂ + ε")
        print("5. 🎯 Uses coefficients to predict next 6-month direction")
        
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   📅 Period: {self.training_data['Date'].min().strftime('%Y-%m-%d')} to {self.training_data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   📊 Observations: {len(self.training_data)}")
        print(f"   📈 Contains: Price data, dividend yields, CPI data, InterBank rates")
        
        # NOW COMPUTE EXCESS RETURNS FOR REFERENCE TRAINING
        print("\n🔄 Computing excess returns for initial training window (Apr 1997 → Sep 2015)...")
        
        if not STATSMODELS_AVAILABLE:
            print("   ❌ Statsmodels not available - using simple linear regression")
            return self.train_model_simple()
        
        if len(self.training_data) == 0:
            print("   ❌ No training data available")
            return
        
        # Compute 6-month returns for the training period (reference implementation)
        training_data_copy = self.training_data.copy()
        training_data_copy['ftse_6m_return'] = np.nan
        training_data_copy['m_interbank_6m_return'] = np.nan
        training_data_copy['excess_return'] = np.nan
        
        # Compute returns for training period (need full data to look ahead)
        for idx in range(len(training_data_copy) - 6):  # Need 6 months ahead
            current_date = training_data_copy.iloc[idx]['Date']
            future_date = current_date + pd.DateOffset(months=6)
            
            # Find future price in full dataset
            future_mask = (self.full_data['Date'] >= future_date - pd.DateOffset(days=15)) & \
                         (self.full_data['Date'] <= future_date + pd.DateOffset(days=15))
            future_data = self.full_data[future_mask]
            
            if len(future_data) > 0:
                future_price = future_data.iloc[0]['ftse_price']
                current_price = training_data_copy.iloc[idx]['ftse_price']
                
                # Stock return
                stock_return = (future_price / current_price) - 1
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('ftse_6m_return')] = stock_return
                
                # InterBank return (assuming annual rate, convert to 6-month return)
                annual_rate = training_data_copy.iloc[idx]['m_interbank_rate'] / 100  # Convert % to decimal
                interbank_return = (1 + annual_rate) ** 0.5 - 1  # 6-month = half-year compound
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('m_interbank_6m_return')] = interbank_return
                
                # Excess return
                excess_return = stock_return - interbank_return
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('excess_return')] = excess_return
        
        # Keep only observations with valid excess returns
        valid_data = training_data_copy.dropna(subset=['excess_return']).copy()
        print(f"   ✅ Computed excess returns: {len(valid_data)}/{len(training_data_copy)} valid observations")
        
        # Prepare training data for multi-predictor model
        y_train = valid_data['excess_return'].values  # Y: Dependent variable (excess returns)
        X1_train = valid_data['dividend_yield'].values  # X₁: Dividend yield
        X2_train = valid_data['cpi_yoy'].values  # X₂: CPI year-over-year growth
        
        # Create matrix with both predictors
        X_train = np.column_stack([X1_train, X2_train])
        
        # Add constant term for regression intercept
        X_train_with_const = add_constant(X_train)
        
        # Fit OLS regression to estimate α̂, β̂₁ and β̂₂
        self.model = OLS(y_train, X_train_with_const).fit()
        
        print("\n📊 MULTI-PREDICTOR TRAINING RESULTS:")
        print("-" * 50)
        print(self.model.summary())
        
        # Extract key statistics - α̂, β̂₁ (dividend yield), β̂₂ (CPI)
        alpha_hat = self.model.params[0]  # α̂ (intercept estimate)
        beta1_hat = self.model.params[1]  # β̂₁ (dividend yield coefficient)
        beta2_hat = self.model.params[2]  # β̂₂ (CPI coefficient)
        alpha_pval = self.model.pvalues[0]
        beta1_pval = self.model.pvalues[1]
        beta2_pval = self.model.pvalues[2]
        r_squared = self.model.rsquared
        
        print(f"\n📈 ESTIMATED COEFFICIENTS:")
        print(f"   α̂ (alpha hat - intercept): {alpha_hat:.6f} (p-value: {alpha_pval:.4f})")
        print(f"   β̂₁ (beta1 hat - dividend yield): {beta1_hat:.6f} (p-value: {beta1_pval:.4f})")
        print(f"   β̂₂ (beta2 hat - CPI growth): {beta2_hat:.6f} (p-value: {beta2_pval:.4f})")
        print(f"   R-squared: {r_squared:.4f}")
        print(f"   Training observations: {len(self.training_data)}")
        
        # Store training results for use in prediction phase
        self.results['training'] = {
            'alpha_hat': alpha_hat,
            'beta1_hat': beta1_hat,
            'beta2_hat': beta2_hat,
            'alpha_pval': alpha_pval,
            'beta1_pval': beta1_pval,
            'beta2_pval': beta2_pval,
            'r_squared': r_squared,
            'observations': len(self.training_data)
        }
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if beta1_pval < 0.05:
            if beta1_hat > 0:
                print(f"   ✅ Dividend yield has POSITIVE predictive power")
                print(f"   📈 1% higher dividend yield → {beta1_hat*100:.2f}% higher excess return")
            else:
                print(f"   ✅ Dividend yield has NEGATIVE predictive power")
                print(f"   📉 1% higher dividend yield → {abs(beta1_hat)*100:.2f}% lower excess return")
        else:
            print(f"   ❌ Dividend yield has NO significant predictive power")
        
        if beta2_pval < 0.05:
            if beta2_hat > 0:
                print(f"   ✅ CPI growth has POSITIVE predictive power")
                print(f"   📈 1% higher CPI growth → {beta2_hat*100:.2f}% higher excess return")
            else:
                print(f"   ✅ CPI growth has NEGATIVE predictive power")
                print(f"   📉 1% higher CPI growth → {abs(beta2_hat)*100:.2f}% lower excess return")
        else:
            print(f"   ❌ CPI growth has NO significant predictive power")

        print(f"\n🎯 NEXT STEP: Use α̂ = {alpha_hat:.4f}, β̂₁ = {beta1_hat:.4f}, β̂₂ = {beta2_hat:.4f} for predictions")

    def train_model_simple(self):
        """Fallback simple linear regression using numpy"""
        print("   Using simple linear regression (numpy-based)")
        
        if len(self.training_data) == 0:
            print("   ❌ No training data available")
            return
        
        # Prepare training data
        y_train = self.training_data['excess_return'].values
        X_train = self.training_data['dividend_yield'].values
        
        # Add constant term manually
        X_matrix = np.column_stack([np.ones(len(X_train)), X_train])
        
        # Calculate coefficients using least squares: β = (X'X)^-1 X'y
        try:
            coefficients = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_train
            
            alpha = coefficients[0]
            beta = coefficients[1]
            
            # Calculate R-squared
            y_pred = X_matrix @ coefficients
            ss_res = np.sum((y_train - y_pred) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Simple t-statistics (approximate)
            residuals = y_train - y_pred
            mse = ss_res / (len(y_train) - 2)
            var_coeff = mse * np.diag(np.linalg.inv(X_matrix.T @ X_matrix))
            t_stats = coefficients / np.sqrt(var_coeff)
            
            # Approximate p-values (using t-distribution with n-2 degrees of freedom)
            from scipy.stats import t
            df = len(y_train) - 2
            alpha_pval = 2 * (1 - t.cdf(np.abs(t_stats[0]), df))
            beta_pval = 2 * (1 - t.cdf(np.abs(t_stats[1]), df))
            
            print(f"\n📈 ESTIMATED COEFFICIENTS:")
            print(f"   α̂ (alpha hat - intercept): {alpha:.6f} (p-value: {alpha_pval:.4f})")
            print(f"   β̂ (beta hat - dividend yield): {beta:.6f} (p-value: {beta_pval:.4f})")
            print(f"   R-squared: {r_squared:.4f}")
            print(f"   Training observations: {len(self.training_data)}")
            
            # Create a simple model object for predictions
            class SimpleModel:
                def __init__(self, params):
                    self.params = params
                    self.rsquared = r_squared
                
                def predict(self, X):
                    return X @ self.params
            
            self.model = SimpleModel(coefficients)
            
            # Store training results
            self.results['training'] = {
                'alpha_hat': alpha,
                'beta_hat': beta,
                'alpha_pval': alpha_pval,
                'beta_pval': beta_pval,
                'r_squared': r_squared,
                'observations': len(self.training_data)
            }
            
        except Exception as e:
            print(f"   ❌ Error in simple regression: {e}")
            return
            
    def rolling_window_predictions(self):
        """
        Implement rolling window prediction strategy
        Re-estimate α̂ and β̂ every 6 months using all available data up to that point
        """
        print("\n🔮 ROLLING WINDOW PREDICTION PHASE")
        print("="*60)
        print("METHODOLOGY: Re-estimate coefficients every 6 months")
        print("• Use expanding window: Always start from Apr 1997")
        print("• Update training end date every 6 months")
        print("• Use α̂, β̂₁, β̂₂ + current dividend yield + current CPI to predict next 6 months")
        print("• No forward-looking bias: only use information available at prediction time")
        
        # Create investment decision dates (1st of every 6th month starting Oct 1, 2015)
        # Investment decisions made on: Oct 1 2015, Apr 1 2016, Oct 1 2016, Apr 1 2017, etc.
        investment_dates = []
        current_date = pd.Timestamp('2015-10-01')
        end_date = pd.Timestamp('2025-04-01')
        
        while current_date <= end_date:
            investment_dates.append(current_date)
            # Add 6 months
            if current_date.month == 10:  # October -> April
                current_date = current_date.replace(year=current_date.year + 1, month=4)
            else:  # April -> October
                current_date = current_date.replace(month=10)
        
        print(f"\n📅 Investment decision schedule: {len(investment_dates)} periods")
        print("   Investment decisions made on 1st of month using data available up to previous month-end:")
        for i, date in enumerate(investment_dates[:5]):  # Show first 5
            data_cutoff = (date - pd.DateOffset(days=1)) + pd.offsets.MonthEnd(0)  # Previous month-end
            print(f"   {i+1}. {date.strftime('%Y-%m-%d')} (using data up to {data_cutoff.strftime('%Y-%m-%d')})")
        if len(investment_dates) > 5:
            print(f"   ... and {len(investment_dates)-5} more periods")
        
        # Initialize results storage
        rolling_results = []
        
        for i, investment_date in enumerate(investment_dates):
            print(f"\n🔄 ITERATION {i+1}: Investment decision for {investment_date.strftime('%Y-%m-%d')}")
            
            # Data cutoff: Use data available up to previous month-end (no forward-looking)
            data_cutoff_date = (investment_date - pd.DateOffset(days=1)) + pd.offsets.MonthEnd(0)
            print(f"   📅 Training window: Apr 1997 → {data_cutoff_date.strftime('%Y-%m-%d')} (expanding window)")
            
            # Create training data (Apr 1997 → data_cutoff_date)
            training_start_dt = pd.to_datetime(self.training_start)
            training_mask = (
                (self.full_data['Date'] >= training_start_dt) & 
                (self.full_data['Date'] <= data_cutoff_date)
            )
            current_training_data = self.full_data[training_mask].copy()
            
            # For expanding windows, compute 6-month returns dynamically for this specific training period
            print(f"   📊 Computing returns for training window: {len(current_training_data)} observations")
            
            # Reset the return columns to avoid contamination
            current_training_data['ftse_6m_return'] = np.nan
            current_training_data['m_interbank_6m_return'] = np.nan
            current_training_data['excess_return'] = np.nan
            
            # Compute 6-month forward returns for this training window only
            print(f"   🧮 Computing 6-month forward returns for {len(current_training_data)} observations...")
            print(f"      Note: Will compute for {len(current_training_data) - 6} observations (need 6 months ahead data)")
            
            valid_returns_count = 0
            for idx in range(len(current_training_data) - 6):  # Need 6 months ahead data
                current_date = current_training_data.iloc[idx]['Date']
                future_date = current_date + pd.DateOffset(months=6)
                
                # Show progress for first few and last few observations
                if idx < 3 or idx >= len(current_training_data) - 9:
                    print(f"      [{idx+1:3d}] {current_date.strftime('%Y-%m-%d')} → {future_date.strftime('%Y-%m-%d')}")
                elif idx == 3:
                    print(f"      ... (computing for middle observations) ...")
                
                # Find future price in the FULL dataset (not just current_training_data)
                # Use ±15 day window to handle weekends/holidays
                future_mask = (self.full_data['Date'] >= future_date - pd.DateOffset(days=15)) & \
                             (self.full_data['Date'] <= future_date + pd.DateOffset(days=15))
                future_data = self.full_data[future_mask]
                
                if len(future_data) > 0:
                    # Use closest future date
                    closest_future = future_data.iloc[0]
                    actual_future_date = closest_future['Date']
                    future_price = closest_future['ftse_price']
                    current_price = current_training_data.iloc[idx]['ftse_price']
                    
                    # Calculate stock return: (Future_Price / Current_Price) - 1
                    stock_return = (future_price / current_price) - 1
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('ftse_6m_return')] = stock_return
                    
                    # Calculate InterBank return (assuming annual rate, convert to 6-month return)
                    annual_rate = current_training_data.iloc[idx]['m_interbank_rate'] / 100  # Convert % to decimal
                    interbank_return = (1 + annual_rate) ** 0.5 - 1  # 6-month = half-year compound
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('m_interbank_6m_return')] = interbank_return
                    
                    # Calculate excess return: Stock_Return - InterBank_Return
                    excess_return = stock_return - interbank_return
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('excess_return')] = excess_return
                    
                    valid_returns_count += 1
                    
                    # Show detailed calculation for first few observations
                    if idx < 2:
                        print(f"         📊 Current: {current_price:.2f} → Future: {future_price:.2f} (on {actual_future_date.strftime('%Y-%m-%d')})")
                        print(f"         📈 Stock return: {stock_return:.4f} ({stock_return*100:.2f}%)")
                        print(f"         💰 InterBank rate: {annual_rate*100:.2f}% annual → {interbank_return*100:.2f}% 6-month")
                        print(f"         ⚖️  Excess return: {excess_return:.4f} ({excess_return*100:.2f}%)")
                else:
                    if idx < 2:
                        print(f"         ❌ No future data found for {future_date.strftime('%Y-%m-%d')}")
            
            print(f"   ✅ Successfully computed {valid_returns_count}/{len(current_training_data) - 6} valid 6-month returns")
            
            # Keep only observations with valid excess returns
            current_training_data = current_training_data.dropna(subset=['excess_return']).copy()
            
            print(f"   📚 Training: {training_start_dt.strftime('%Y-%m')} to {data_cutoff_date.strftime('%Y-%m')} ({len(current_training_data)} obs)")
            
            if len(current_training_data) < 24:  # Need minimum data
                print(f"   ⚠️  Insufficient training data, skipping...")
                continue
            
            # Train model on current data
            alpha_hat, beta1_hat, beta2_hat, r_squared = self.train_rolling_model(current_training_data, i+1)
            
            if alpha_hat is None:  # Training failed
                continue
            
            # Get dividend yield and CPI for prediction (use most recent values available)
            predictor_date = data_cutoff_date
            predictor_mask = self.full_data['Date'] == predictor_date
            
            if not predictor_mask.any():
                # Find closest date
                closest_idx = (self.full_data['Date'] - predictor_date).abs().idxmin()
                current_div_yield = self.full_data.loc[closest_idx, 'dividend_yield']
                current_cpi = self.full_data.loc[closest_idx, 'cpi_yoy']
                actual_date = self.full_data.loc[closest_idx, 'Date']
                print(f"   📊 Using predictors from {actual_date.strftime('%Y-%m-%d')}: DivYield={current_div_yield:.2f}%, CPI={current_cpi:.2f}%")
            else:
                current_div_yield = self.full_data[predictor_mask]['dividend_yield'].iloc[0]
                current_cpi = self.full_data[predictor_mask]['cpi_yoy'].iloc[0]
                print(f"   📊 Predictors on {predictor_date.strftime('%Y-%m-%d')}: DivYield={current_div_yield:.2f}%, CPI={current_cpi:.2f}%")
            
            # Make multi-predictor prediction
            predicted_excess_return = alpha_hat + beta1_hat * current_div_yield + beta2_hat * current_cpi
            investment_decision = "FTSE All Share Index" if predicted_excess_return > 0 else "Interbank Rate"
            
            print(f"   🎯 Multi-predictor prediction:")
            print(f"      Ŷ = {alpha_hat:.6f} + {beta1_hat:.6f}×{current_div_yield:.2f} + {beta2_hat:.6f}×{current_cpi:.2f}")
            print(f"      Ŷ = {predicted_excess_return:.4f}")
            print(f"   💼 Investment Decision: {investment_decision}")
            
            # Store results with dates formatted properly
            rolling_results.append({
                'prediction_date': investment_date.date(),  # Investment decision date
                'training_start': training_start_dt.date(),  # Convert to date only
                'training_end': data_cutoff_date.date(),  # Data cutoff date
                'training_observations': len(current_training_data),
                'alpha_hat': alpha_hat,
                'beta1_hat': beta1_hat,  # Dividend yield coefficient
                'beta2_hat': beta2_hat,  # CPI growth coefficient
                'r_squared': r_squared,
                'dividend_yield': current_div_yield,
                'cpi_growth': current_cpi,  # Store CPI value used for prediction
                'predicted_excess_return': predicted_excess_return,
                'invest_in_stocks': 1 if predicted_excess_return > 0 else 0,
                'investment_decision': investment_decision
            })
        
        # Convert to DataFrame
        self.rolling_predictions = pd.DataFrame(rolling_results)
        
        # Convert datetime columns to date-only format for clean output
        for date_col in ['prediction_date', 'training_start', 'training_end']:
            if date_col in self.rolling_predictions.columns:
                self.rolling_predictions[date_col] = pd.to_datetime(self.rolling_predictions[date_col]).dt.date
        
        # Save rolling predictions with coefficients to file
        if len(rolling_results) > 0:
            output_dir = BASE_PATH / "Data" / "Output"
            output_dir.mkdir(parents=True, exist_ok=True)
            coefficients_file = output_dir / "SMM265_DivYield_CPI_Rolling_Window_Coefficients_and_Predictions.xlsx"
            
            # Create a copy for saving with proper date formatting
            rolling_predictions_save = self.rolling_predictions.copy()
            
            with pd.ExcelWriter(coefficients_file, engine='openpyxl') as writer:
                rolling_predictions_save.to_excel(writer, sheet_name='Rolling_Coefficients', index=False)
            
            print(f"\n💾 Rolling window coefficients saved to: {coefficients_file.name}")
            print(f"   📅 All dates saved as date-only format (no timestamps)")
        
        print(f"\n📊 ROLLING WINDOW SUMMARY:")
        print(f"   Total predictions: {len(self.rolling_predictions)}")
        print(f"   Stock investments: {self.rolling_predictions['invest_in_stocks'].sum()}")
        print(f"   InterBank investments: {len(self.rolling_predictions) - self.rolling_predictions['invest_in_stocks'].sum()}")
        print(f"   Average α̂: {self.rolling_predictions['alpha_hat'].mean():.6f}")
        print(f"   Average β̂₁ (div yield): {self.rolling_predictions['beta1_hat'].mean():.6f}")
        print(f"   Average β̂₂ (CPI): {self.rolling_predictions['beta2_hat'].mean():.6f}")
        print(f"   α̂ range: [{self.rolling_predictions['alpha_hat'].min():.6f}, {self.rolling_predictions['alpha_hat'].max():.6f}]")
        print(f"   β̂₁ range: [{self.rolling_predictions['beta1_hat'].min():.6f}, {self.rolling_predictions['beta1_hat'].max():.6f}]")
        print(f"   β̂₂ range: [{self.rolling_predictions['beta2_hat'].min():.6f}, {self.rolling_predictions['beta2_hat'].max():.6f}]")
        
        # After generating all predictions, calculate actual returns for evaluation
        self.calculate_actual_returns_for_predictions()
        
        return self.rolling_predictions
    
    def calculate_actual_returns_for_predictions(self):
        """
        Calculate actual FTSE and InterBank returns for each prediction period
        This enables proper strategy evaluation against realized returns
        """
        print("\n📊 CALCULATING ACTUAL RETURNS FOR EVALUATION")
        print("="*60)
        print("OBJECTIVE: Calculate realized returns for each 6-month prediction period")
        print("• For each prediction date, calculate actual FTSE and InterBank returns")
        print("• Compare switching strategy performance vs buy-and-hold")
        print("• Determine prediction accuracy based on actual excess returns")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for evaluation")
            return
        
        evaluation_results = []
        
        for idx, row in self.rolling_predictions.iterrows():
            pred_date = row['prediction_date']
            investment_decision = row['investment_decision']
            predicted_excess_return = row['predicted_excess_return']
            
            # Calculate investment period for return measurement
            # Investment decision on Oct 1, 2015 using data up to Sep 30, 2015
            # Investment period: Oct 1, 2015 → Mar 31, 2016 (6 months)  
            # Return measurement: Oct 1, 2015 price → Mar 31, 2016 price
            period_start_date = pred_date  # Start from investment decision date (e.g., Oct 1, 2015)
            period_end_date = pred_date + pd.DateOffset(months=6) - pd.DateOffset(days=1)  # End of 6th month (e.g., Mar 31, 2016)
            
            print(f"\n🔍 EVALUATION {idx+1}: Investment decision for {pred_date.strftime('%Y-%m-%d')}")
            print(f"   📅 Calculated period: {period_start_date.strftime('%Y-%m-%d')} → {period_end_date.strftime('%Y-%m-%d')}")
            print(f"   💼 Investment decision: {investment_decision}")
            print(f"   📈 Predicted excess return: {predicted_excess_return:.4f}")
            
            # Get actual returns for this period
            actual_ftse_return, actual_interbank_return = self.calculate_period_actual_returns(
                period_start_date, period_end_date
            )
            
            if actual_ftse_return is not None and actual_interbank_return is not None:
                # Calculate actual excess return
                actual_excess_return = actual_ftse_return - actual_interbank_return
                
                # Determine if prediction was correct
                prediction_correct = (
                    (predicted_excess_return > 0 and actual_excess_return > 0) or
                    (predicted_excess_return <= 0 and actual_excess_return <= 0)
                )
                
                # Calculate switching strategy return
                switching_return = actual_ftse_return if investment_decision == "FTSE All Share Index" else actual_interbank_return
                
                # Buy-and-hold return (always FTSE)
                buyhold_return = actual_ftse_return
                
                print(f"   📊 Actual FTSE return: {actual_ftse_return*100:.2f}%")
                print(f"   📊 Actual InterBank return: {actual_interbank_return*100:.2f}%")
                print(f"   📊 Actual excess return: {actual_excess_return*100:.2f}%")
                print(f"   🎯 Prediction correct: {'✅' if prediction_correct else '❌'}")
                print(f"   💰 Switching strategy return: {switching_return*100:.2f}%")
                print(f"   💰 Buy-and-hold return: {buyhold_return*100:.2f}%")
                print(f"   📈 Strategy outperformance: {(switching_return - buyhold_return)*100:.2f}%")
                
                # Store evaluation results with proper date formatting
                evaluation_results.append({
                    'prediction_date': pred_date.date() if hasattr(pred_date, 'date') else pred_date,
                    'period_end_date': period_end_date.date() if hasattr(period_end_date, 'date') else period_end_date,
                    'predicted_excess_return': predicted_excess_return,
                    'investment_decision': investment_decision,
                    'actual_ftse_return': actual_ftse_return,
                    'actual_interbank_return': actual_interbank_return,
                    'actual_excess_return': actual_excess_return,
                    'prediction_correct': prediction_correct,
                    'switching_strategy_return': switching_return,
                    'buyhold_strategy_return': buyhold_return,
                    'strategy_outperformance': switching_return - buyhold_return  # This should not be 0 unless returns are equal
                })
            else:
                print(f"   ❌ Insufficient data to calculate actual returns")
        
        # Convert to DataFrame and merge with rolling predictions
        if evaluation_results:
            evaluation_df = pd.DataFrame(evaluation_results)
            
            # Ensure date formatting for evaluation results
            for date_col in ['prediction_date', 'period_end_date']:
                if date_col in evaluation_df.columns:
                    evaluation_df[date_col] = pd.to_datetime(evaluation_df[date_col]).dt.date
            
            # Convert prediction_date to the same format for merging
            rolling_pred_dates = pd.to_datetime(self.rolling_predictions['prediction_date']).dt.date
            eval_pred_dates = pd.to_datetime(evaluation_df['prediction_date']).dt.date
            
            # Create temporary merge keys
            self.rolling_predictions['temp_merge_key'] = rolling_pred_dates.astype(str)
            evaluation_df['temp_merge_key'] = eval_pred_dates.astype(str)
            
            # Merge with existing rolling predictions
            self.rolling_predictions = pd.merge(
                self.rolling_predictions,
                evaluation_df[['temp_merge_key', 'actual_ftse_return', 'actual_interbank_return', 
                              'actual_excess_return', 'prediction_correct', 'switching_strategy_return',
                              'buyhold_strategy_return', 'strategy_outperformance']],
                on='temp_merge_key',
                how='left'
            )
            
            # Remove the temporary merge columns
            self.rolling_predictions = self.rolling_predictions.drop('temp_merge_key', axis=1)
            
            # Calculate cumulative performance
            self.calculate_strategy_performance_metrics(evaluation_df)
            
            # Save enhanced results
            self.save_evaluation_results(evaluation_df)
        else:
            print("   ❌ No evaluation results generated")
    
    def calculate_period_actual_returns(self, start_date, end_date):
        """
        Calculate actual FTSE and InterBank returns for a specific 6-month period
        Uses self.full_data which contains complete data until 2025-04-30
        """
        # Use self.full_data which has complete data including until 2025-04-30
        # NOT self.data which only has training data until 2015
        data_source = self.full_data
        
        # Convert start_date and end_date to date objects if they're timestamps
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
        
        # Find start and end data points in dataset
        # For investment period Oct 1 → Mar 31, we want:
        # Start: Price just before Oct 1 (i.e., Sep 30)
        # End: Price on or after Mar 31
        data_source = data_source.copy()
        data_source['date_only'] = data_source['Date'].dt.date
        
        # For start date: find date <= start_date (on or just before investment starts)
        start_mask = data_source['date_only'] <= start_date
        # For end date: find first date >= end_date (on or after investment ends)
        end_mask = data_source['date_only'] >= end_date
        
        if not start_mask.any() or not end_mask.any():
            print(f"   ❌ No data found for period {start_date} to {end_date}")
            return None, None
        
        # Get closest available dates
        # First try to find exact start date
        exact_start_mask = data_source['date_only'] == start_date
        if exact_start_mask.any():
            start_idx = data_source[exact_start_mask].index[0]  # Exact date
        else:
            start_idx = data_source[start_mask].index[-1]  # Last date <= start_date
        
        end_indices = data_source[end_mask].index
        
        if len(end_indices) == 0 or end_indices[0] <= start_idx:
            print(f"   ❌ Insufficient data range for period {start_date} to {end_date}")
            return None, None
        
        end_idx = end_indices[0]  # First date >= end_date
        
        start_data = data_source.iloc[start_idx+1]  # Use the next date after start_idx for investment start
        end_data = data_source.iloc[end_idx]
        
        # For display purposes, show investment start as 1st of month since decisions are made on 1st
        actual_start_date = start_data['Date'].replace(day=1).date()
        actual_end_date = end_data['Date'].date()
        
        print(f"      📅 Actual period: {actual_start_date} to {actual_end_date}")
        
        # Calculate actual FTSE return
        ftse_start_price = start_data['ftse_price']
        ftse_end_price = end_data['ftse_price']
        actual_ftse_return = (ftse_end_price / ftse_start_price) - 1
        
        # Calculate actual InterBank return by compounding monthly rates
        # For each month in the investment period, compound the monthly InterBank rate
        period_mask = (
            (data_source['date_only'] >= actual_start_date) & 
            (data_source['date_only'] <= actual_end_date)
        )
        period_data = data_source[period_mask].sort_values('Date')
        
        if len(period_data) > 0:
            print(f"      💰 FTSE price: {ftse_start_price:.2f} → {ftse_end_price:.2f}")
            print(f"      📊 Monthly InterBank compounding over {len(period_data)} months:")
            
            # Start with $1 investment
            interbank_value = 1.0
            
            for idx, row in period_data.iterrows():
                monthly_rate = row['m_interbank_rate'] / 100  # Convert % to decimal
                monthly_return = (1 + monthly_rate) ** (1/12) - 1  # Convert annual to monthly
                interbank_value *= (1 + monthly_return)  # Compound monthly
                
                # Show details for first few and last few months
                month_num = len([i for i in period_data.index if i <= idx])
                if month_num <= 3 or month_num > len(period_data) - 3:
                    print(f"         Month {month_num}: {monthly_rate*100:.2f}% annual → {monthly_return*100:.3f}% monthly → Value: {interbank_value:.6f}")
                elif month_num == 4:
                    print(f"         ... (compounding middle months) ...")
            
            # Total return over the period
            actual_interbank_return = interbank_value - 1
            
            print(f"      📈 Final InterBank return: {actual_interbank_return*100:.3f}% (compounded monthly)")
        else:
            # Fallback to using start period rate
            start_interbank_rate = start_data['m_interbank_rate']
            period_length_years = (actual_end_date - actual_start_date).days / 365.25
            actual_interbank_return = (1 + start_interbank_rate/100)**period_length_years - 1
            
            print(f"      💰 FTSE price: {ftse_start_price:.2f} → {ftse_end_price:.2f}")
            print(f"      📈 Start InterBank rate: {start_interbank_rate:.2f}% for {period_length_years:.2f} years")
        
        return actual_ftse_return, actual_interbank_return
    
    def calculate_strategy_performance_metrics(self, evaluation_df):
        """
        Calculate comprehensive performance metrics for the strategy evaluation
        """
        print(f"\n🏆 STRATEGY PERFORMANCE EVALUATION")
        print("="*60)
        
        # Basic statistics
        total_predictions = len(evaluation_df)
        correct_predictions = evaluation_df['prediction_correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate cumulative returns
        evaluation_df = evaluation_df.copy()
        evaluation_df['switching_cumulative'] = (1 + evaluation_df['switching_strategy_return']).cumprod()
        evaluation_df['buyhold_cumulative'] = (1 + evaluation_df['buyhold_strategy_return']).cumprod()
        
        # Final cumulative returns
        switching_total_return = evaluation_df['switching_cumulative'].iloc[-1] - 1
        buyhold_total_return = evaluation_df['buyhold_cumulative'].iloc[-1] - 1
        
        # Annualized returns (approximate)
        years = total_predictions * 0.5  # Each prediction is 6 months
        switching_annual_return = (evaluation_df['switching_cumulative'].iloc[-1] ** (1/years)) - 1 if years > 0 else 0
        buyhold_annual_return = (evaluation_df['buyhold_cumulative'].iloc[-1] ** (1/years)) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        switching_volatility = evaluation_df['switching_strategy_return'].std() * np.sqrt(2)  # 2 periods per year
        buyhold_volatility = evaluation_df['buyhold_strategy_return'].std() * np.sqrt(2)
        
        # Sharpe ratios (using average InterBank rate as risk-free rate)
        avg_rf_rate = evaluation_df['actual_interbank_return'].mean() * 2  # Annualize
        switching_sharpe = (switching_annual_return - avg_rf_rate) / switching_volatility if switching_volatility > 0 else 0
        buyhold_sharpe = (buyhold_annual_return - avg_rf_rate) / buyhold_volatility if buyhold_volatility > 0 else 0
        
        # Win rate and average outperformance
        win_periods = (evaluation_df['strategy_outperformance'] > 0).sum()
        win_rate = win_periods / total_predictions if total_predictions > 0 else 0
        avg_outperformance = evaluation_df['strategy_outperformance'].mean()
        
        print(f"PREDICTION ACCURACY:")
        print(f"   🎯 Correct predictions: {correct_predictions}/{total_predictions} ({accuracy*100:.1f}%)")
        print(f"   🏆 Win periods (outperformed): {win_periods}/{total_predictions} ({win_rate*100:.1f}%)")
        print(f"   📊 Average outperformance: {avg_outperformance*100:.2f}% per period")
        
        print(f"\nSWITCHING STRATEGY PERFORMANCE:")
        print(f"   💰 Total return: {switching_total_return*100:.1f}%")
        print(f"   📈 Annualized return: {switching_annual_return*100:.1f}%")
        print(f"   📊 Volatility: {switching_volatility*100:.1f}%")
        print(f"   ⚡ Sharpe ratio: {switching_sharpe:.3f}")
        
        print(f"\nBUY-AND-HOLD BENCHMARK:")
        print(f"   💰 Total return: {buyhold_total_return*100:.1f}%")
        print(f"   📈 Annualized return: {buyhold_annual_return*100:.1f}%")
        print(f"   📊 Volatility: {buyhold_volatility*100:.1f}%")
        print(f"   ⚡ Sharpe ratio: {buyhold_sharpe:.3f}")
        
        print(f"\nSTRATEGY COMPARISON:")
        print(f"   🚀 Return advantage: {(switching_total_return - buyhold_total_return)*100:.1f}%")
        print(f"   🎯 Sharpe advantage: {switching_sharpe - buyhold_sharpe:.3f}")
        
        # Store performance metrics
        self.performance_metrics = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'avg_outperformance': avg_outperformance,
            'switching_total_return': switching_total_return,
            'buyhold_total_return': buyhold_total_return,
            'switching_annual_return': switching_annual_return,
            'buyhold_annual_return': buyhold_annual_return,
            'switching_volatility': switching_volatility,
            'buyhold_volatility': buyhold_volatility,
            'switching_sharpe': switching_sharpe,
            'buyhold_sharpe': buyhold_sharpe
        }
        
        # Store evaluation data for further analysis
        self.evaluation_data = evaluation_df
    
    def save_evaluation_results(self, evaluation_df):
        """
        Save comprehensive evaluation results to Excel with proper date formatting
        """
        output_dir = BASE_PATH / "Data" / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluation_file = output_dir / "SMM265_DivYield_CPI_Strategy_Evaluation_Results.xlsx"
        
        # Create copies for saving with proper date formatting
        rolling_predictions_save = self.rolling_predictions.copy()
        evaluation_df_save = evaluation_df.copy()
        
        # Convert all date columns to date-only format for Excel
        date_columns = ['prediction_date', 'training_start', 'training_end', 'period_end_date']
        
        for df in [rolling_predictions_save, evaluation_df_save]:
            for col in date_columns:
                if col in df.columns:
                    # Convert to date-only format
                    df[col] = pd.to_datetime(df[col]).dt.date
        
        with pd.ExcelWriter(evaluation_file, engine='openpyxl') as writer:
            # Rolling predictions with evaluation (dates as date-only)
            rolling_predictions_save.to_excel(writer, sheet_name='Rolling_Predictions', index=False)
            
            # Detailed evaluation results (dates as date-only)
            evaluation_df_save.to_excel(writer, sheet_name='Evaluation_Details', index=False)
            
            # Performance summary
            if hasattr(self, 'performance_metrics'):
                performance_summary = pd.DataFrame([self.performance_metrics])
                performance_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        print(f"\n💾 Strategy evaluation saved to: {evaluation_file.name}")
        print(f"   📅 All dates saved in date-only format (no timestamps)")
    
    def train_rolling_model(self, training_data, iteration):
        """Train multi-predictor model for rolling window iteration"""
        try:
            # Prepare multi-predictor data
            y_train = training_data['excess_return'].values
            X1_train = training_data['dividend_yield'].values  # Predictor 1: Dividend Yield
            X2_train = training_data['cpi_yoy'].values  # Predictor 2: CPI Growth
            
            # Create matrix with both predictors
            X_train = np.column_stack([X1_train, X2_train])
            
            # Run multi-predictor regression
            if STATSMODELS_AVAILABLE:
                X_train_with_const = add_constant(X_train)
                model = OLS(y_train, X_train_with_const).fit()
                alpha_hat = model.params[0]  # α̂
                beta1_hat = model.params[1]  # β̂₁ (dividend yield)
                beta2_hat = model.params[2]  # β̂₂ (CPI growth)
                r_squared = model.rsquared
            else:
                # Simple regression fallback for multi-predictor
                X_matrix = np.column_stack([np.ones(len(X_train)), X_train])
                coefficients = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_train
                alpha_hat = coefficients[0]
                beta1_hat = coefficients[1]
                beta2_hat = coefficients[2]
                
                y_pred = X_matrix @ coefficients
                ss_res = np.sum((y_train - y_pred) ** 2)
                ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
            
            print(f"   📈 Multi-predictor coefficients: α̂ = {alpha_hat:.6f}, β̂₁ = {beta1_hat:.6f}, β̂₂ = {beta2_hat:.6f}, R² = {r_squared:.4f}")
            return alpha_hat, beta1_hat, beta2_hat, r_squared
            
        except Exception as e:
            print(f"   ❌ Multi-predictor training failed: {e}")
            return None, None, None, None
    
    def make_predictions(self):
        """Make predictions and evaluate investment strategy"""
        print("\n🔮 PREDICTION PHASE")
        print("="*50)
        print("OBJECTIVE: Use α̂ and β̂ from training to predict excess returns")
        print("SWITCHING STRATEGY LOGIC:")
        print("• If predicted excess return > 0 → Invest in FTSE All-Share")
        print("• If predicted excess return ≤ 0 → Invest in InterBank Rate")
        print(f"• Evaluation period: {self.prediction_start} to {self.prediction_end}")
        
        if self.model is None or len(self.prediction_data) == 0:
            print("   ❌ No model or prediction data available")
            return
        
        # Get α̂ and β̂ from training results
        alpha_hat = self.results['training']['alpha_hat']
        beta_hat = self.results['training']['beta_hat']
        print(f"\nUsing trained coefficients: α̂ = {alpha_hat:.6f}, β̂ = {beta_hat:.6f}")
        
        # Create prediction DataFrame
        pred_df = self.prediction_data.copy()
        
        # Make predictions using trained model: Ŷ = α̂ + β̂ × Dividend_Yield
        X_pred = pred_df['dividend_yield'].values
        if STATSMODELS_AVAILABLE and hasattr(self.model, 'predict'):
            X_pred_with_const = add_constant(X_pred)
            pred_df['predicted_excess_return'] = self.model.predict(X_pred_with_const)
        else:
            # Simple prediction: Ŷ = α̂ + β̂ × X
            pred_df['predicted_excess_return'] = alpha_hat + beta_hat * X_pred
            
        print(f"\n📊 PREDICTION FORMULA:")
        print(f"   Predicted_Excess_Return = {alpha_hat:.6f} + {beta_hat:.6f} × Dividend_Yield")
        
        # Investment decisions based on predictions
        pred_df['invest_in_stocks'] = (pred_df['predicted_excess_return'] > 0).astype(int)
        pred_df['prediction_correct'] = (
            (pred_df['predicted_excess_return'] > 0) & (pred_df['excess_return'] > 0) |
            (pred_df['predicted_excess_return'] <= 0) & (pred_df['excess_return'] <= 0)
        ).astype(int)
        
        # Calculate strategy returns
        pred_df['switching_return'] = np.where(
            pred_df['invest_in_stocks'] == 1,
            pred_df['ftse_6m_return'],
            pred_df['m_interbank_6m_return']  # Now represents InterBank rate returns
        )
        
        # Buy-and-hold strategy
        pred_df['buyhold_return'] = pred_df['ftse_6m_return']
        
        # Calculate cumulative returns
        pred_df['switching_cumret'] = (1 + pred_df['switching_return']).cumprod()
        pred_df['buyhold_cumret'] = (1 + pred_df['buyhold_return']).cumprod()
        
        self.prediction_data = pred_df
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   🎯 Correct predictions: {pred_df['prediction_correct'].sum()}/{len(pred_df)} ({pred_df['prediction_correct'].mean()*100:.1f}%)")
        print(f"   📈 Times invested in stocks: {pred_df['invest_in_stocks'].sum()}/{len(pred_df)} ({pred_df['invest_in_stocks'].mean()*100:.1f}%)")
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("\n📊 STRATEGY EVALUATION")
        print("="*50)
        
        df = self.prediction_data
        
        # Basic returns
        switching_total_return = df['switching_cumret'].iloc[-1] - 1
        buyhold_total_return = df['buyhold_cumret'].iloc[-1] - 1
        
        # Annualized returns
        years = len(df) / 2
        switching_annual_return = (df['switching_cumret'].iloc[-1] ** (1/years)) - 1
        buyhold_annual_return = (df['buyhold_cumret'].iloc[-1] ** (1/years)) - 1
        
        # Volatility
        switching_volatility = df['switching_return'].std() * np.sqrt(2)
        buyhold_volatility = df['buyhold_return'].std() * np.sqrt(2)
        
        # Sharpe ratios
        avg_rf_rate = df['m_interbank_6m_return'].mean() * 2
        switching_sharpe = (switching_annual_return - avg_rf_rate) / switching_volatility if switching_volatility > 0 else 0
        buyhold_sharpe = (buyhold_annual_return - avg_rf_rate) / buyhold_volatility if buyhold_volatility > 0 else 0
        
        # Maximum drawdown
        switching_dd = self.calculate_max_drawdown(df['switching_cumret'])
        buyhold_dd = self.calculate_max_drawdown(df['buyhold_cumret'])
        
        # Store results
        self.results['performance'] = {
            'switching_total_return': switching_total_return,
            'buyhold_total_return': buyhold_total_return,
            'switching_annual_return': switching_annual_return,
            'buyhold_annual_return': buyhold_annual_return,
            'switching_volatility': switching_volatility,
            'buyhold_volatility': buyhold_volatility,
            'switching_sharpe': switching_sharpe,
            'buyhold_sharpe': buyhold_sharpe,
            'switching_max_dd': switching_dd,
            'buyhold_max_dd': buyhold_dd,
            'correct_predictions': df['prediction_correct'].sum(),
            'total_predictions': len(df),
            'prediction_accuracy': df['prediction_correct'].mean(),
            'times_invested_stocks': df['invest_in_stocks'].sum()
        }
        
        # Display results
        print(f"SWITCHING STRATEGY:")
        print(f"   💰 Total return: {switching_total_return*100:.1f}%")
        print(f"   📈 Annual return: {switching_annual_return*100:.1f}%")
        print(f"   📊 Volatility: {switching_volatility*100:.1f}%")
        print(f"   ⚡ Sharpe ratio: {switching_sharpe:.3f}")
        
        print(f"\nBUY-AND-HOLD:")
        print(f"   💰 Total return: {buyhold_total_return*100:.1f}%")
        print(f"   📈 Annual return: {buyhold_annual_return*100:.1f}%")
        print(f"   📊 Volatility: {buyhold_volatility*100:.1f}%")
        print(f"   ⚡ Sharpe ratio: {buyhold_sharpe:.3f}")
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def plot_predicted_vs_actual_returns(self):
        """Create time-series chart of predicted vs actual excess returns"""
        print("\n📊 Creating Predicted vs Actual Returns Chart...")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for visualization")
            return
        
        # Filter out rows with missing actual returns (last observation)
        plot_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(plot_data) == 0:
            print("   ❌ No complete prediction-actual pairs available for plotting")
            return
        
        # Convert prediction_date to datetime for plotting
        plot_data['prediction_date_dt'] = pd.to_datetime(plot_data['prediction_date'])
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot predicted and actual excess returns
        plt.plot(plot_data['prediction_date_dt'], plot_data['predicted_excess_return']*100, 
                marker='o', linewidth=2, markersize=6, label='Predicted Excess Return', color='blue', alpha=0.8)
        
        plt.plot(plot_data['prediction_date_dt'], plot_data['actual_excess_return']*100, 
                marker='s', linewidth=2, markersize=6, label='Actual Excess Return', color='red', alpha=0.8)
        
        # Add zero line for reference
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        plt.title('Plot 1 - Pesaran & Timmermann Model: Predicted vs Actual Excess Returns\n(Rolling Window Predictions 2015-2025)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Date', fontsize=12)
        plt.ylabel('Excess Return (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add summary statistics as text box
        correct_predictions = plot_data['prediction_correct'].sum()
        total_predictions = len(plot_data)
        accuracy = correct_predictions / total_predictions * 100
        
        textstr = f'Prediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)\n'
        textstr += f'Avg Predicted: {plot_data["predicted_excess_return"].mean()*100:.2f}%\n'
        textstr += f'Avg Actual: {plot_data["actual_excess_return"].mean()*100:.2f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = BASE_PATH / "Data" / "Output"
        plot_file = output_dir / "SMM265_DivYield_CPI_Predicted_vs_Actual_Returns_TimeSeries.png"
        
        # Turn off interactive mode to prevent display
        plt.ioff()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"   💾 Time-series plot saved to: {plot_file.name}")
    
    def plot_cumulative_returns_chart(self):
        """Create cumulative returns chart comparing switching strategy vs buy-and-hold"""
        print("\n📊 Creating Cumulative Returns Chart...")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for visualization")
            return
        
        # Filter out rows with missing actual returns
        plot_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(plot_data) == 0:
            print("   ❌ No complete prediction-actual pairs available for plotting")
            return
        
        # Convert prediction_date to datetime for plotting
        plot_data['prediction_date_dt'] = pd.to_datetime(plot_data['prediction_date'])
        
        # Calculate cumulative returns for all strategies
        plot_data['switching_cumulative'] = (1 + plot_data['switching_strategy_return']).cumprod()
        plot_data['buyhold_cumulative'] = (1 + plot_data['buyhold_strategy_return']).cumprod()
        
        # Calculate InterBank cumulative returns using actual InterBank returns for all periods
        plot_data['interbank_cumulative'] = (1 + plot_data['actual_interbank_return']).cumprod()
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative returns for all three strategies
        plt.plot(plot_data['prediction_date_dt'], plot_data['switching_cumulative'], 
                linewidth=3, label='Switching Strategy (Multi-Predictor)', color='blue', alpha=0.8)
        
        plt.plot(plot_data['prediction_date_dt'], plot_data['buyhold_cumulative'], 
                linewidth=3, label='Buy-and-Hold (FTSE All Share)', color='red', alpha=0.8)
        
        plt.plot(plot_data['prediction_date_dt'], plot_data['interbank_cumulative'], 
                linewidth=3, label='InterBank Rate (Risk-Free)', color='green', alpha=0.8)
        
        # Add horizontal line at 1.0 (no gain/loss)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        plt.title('Cumulative Returns: Multi-Predictor Switching Strategy vs Benchmarks\n(Dividend Yield + CPI Model vs FTSE Buy-and-Hold vs InterBank Rate)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (1 = No Change)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add summary statistics as text box with all three strategies
        final_switching = plot_data['switching_cumulative'].iloc[-1]
        final_buyhold = plot_data['buyhold_cumulative'].iloc[-1]
        final_interbank = plot_data['interbank_cumulative'].iloc[-1]
        
        switching_return = (final_switching - 1) * 100
        buyhold_return = (final_buyhold - 1) * 100
        interbank_return = (final_interbank - 1) * 100
        
        outperformance_vs_buyhold = switching_return - buyhold_return
        outperformance_vs_interbank = switching_return - interbank_return
        
        textstr = f'Final Returns:\n'
        textstr += f'Switching Strategy: {switching_return:.1f}%\n'
        textstr += f'Buy-and-Hold (FTSE): {buyhold_return:.1f}%\n'
        textstr += f'InterBank Rate: {interbank_return:.1f}%\n'
        textstr += f'Outperformance vs FTSE: {outperformance_vs_buyhold:.1f}%\n'
        textstr += f'Outperformance vs InterBank: {outperformance_vs_interbank:.1f}%'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = BASE_PATH / "Data" / "Output"
        plot_file = output_dir / "SMM265_DivYield_CPI_Cumulative_Returns_Strategy_Comparison.png"
        plt.ioff()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Cumulative returns chart saved to: {plot_file.name}")
    
    def create_pt_test_table(self):
        """Create Pesaran-Timmermann Sign Test Statistics table as an image"""
        print("\n📊 Creating Pesaran-Timmermann Sign Test Table...")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for PT test")
            return
        
        # Filter out rows with missing actual returns
        test_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(test_data) == 0:
            print("   ❌ No complete prediction-actual pairs available for PT test")
            return
        
        # Calculate PT test statistics
        total_predictions = len(test_data)
        correct_predictions = test_data['prediction_correct'].sum()
        proportion_correct = correct_predictions / total_predictions
        expected_proportion = 0.5  # Under null hypothesis
        
        # Calculate actual number of switches between strategies
        # A switch occurs when invest_in_stocks changes from one period to the next
        invest_decisions = test_data['invest_in_stocks'].values
        num_switches = 0
        for i in range(1, len(invest_decisions)):
            if invest_decisions[i] != invest_decisions[i-1]:
                num_switches += 1
        
        print(f"   📊 Investment decisions: {invest_decisions}")
        print(f"   🔄 Number of switches detected: {num_switches}")
        
        # Calculate PT test statistic
        pstar = 0.5
        pt_statistic = (proportion_correct - pstar) / np.sqrt(pstar * (1 - pstar) / total_predictions)
        
        # Calculate p-value (one-sided test)
        from scipy import stats as scipy_stats
        p_value_one_sided = 1 - scipy_stats.norm.cdf(pt_statistic)
        
        # Significance tests
        significant_5pct = p_value_one_sided < 0.05
        significant_1pct = p_value_one_sided < 0.01
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Table title
        title = "Table 2: Pesaran-Timmermann Sign Test Statistics\nEvaluation Period: November 2015 – October 2025\nFTSE All-Share"
        
        # Create table data
        table_data = [
            ["Number of predictions", f"{total_predictions}"],
            ["Correct sign predictions", f"{correct_predictions}"],
            ["Proportion correct (%)", f"{proportion_correct*100:.1f}%"],
            ["Expected proportion under H₀ (%)", f"{expected_proportion*100:.1f}%"],
            ["PT test statistic", f"{pt_statistic:.3f}"],
            ["p-value (one-sided)", f"{p_value_one_sided:.3f}"],
            ["Significant at 5%?", "Yes" if significant_5pct else "No"],
            ["Significant at 1%?", "Yes" if significant_1pct else "No"],
            ["Number of switches", f"{num_switches}"]
        ]
        
        # Create the table
        table = ax.table(cellText=table_data,
                        colWidths=[0.6, 0.4],
                        cellLoc='left',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.7])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style header and cells
        for i in range(len(table_data)):
            # Left column (labels) - bold
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 0)].set_facecolor('#f0f0f0')
            
            # Right column (values)
            table[(i, 1)].set_facecolor('#ffffff')
            
            # Highlight significant results
            if i == 6 and significant_5pct:  # 5% significance
                table[(i, 1)].set_facecolor('#ffffcc')
                table[(i, 1)].set_text_props(weight='bold', color='green')
            elif i == 7 and significant_1pct:  # 1% significance
                table[(i, 1)].set_facecolor('#ffffcc')
                table[(i, 1)].set_text_props(weight='bold', color='green')
            
            # Add borders
            for j in range(2):
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1)
        
        # Add title
        ax.text(0.5, 0.92, title, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Add separator lines
        ax.text(0.5, 0.85, "─" * 60, transform=ax.transAxes,
                fontsize=10, ha='center', va='center', family='monospace')
        ax.text(0.5, 0.05, "─" * 60, transform=ax.transAxes,
                fontsize=10, ha='center', va='center', family='monospace')
        
        # Add interpretation
        interpretation = "Statistically significant predictive ability at 5% level" if significant_5pct else "No statistically significant predictive ability"
        ax.text(0.5, 0.01, interpretation, transform=ax.transAxes,
                fontsize=10, ha='center', va='bottom', style='italic',
                color='green' if significant_5pct else 'red')
        
        plt.tight_layout()
        
        # Save the table
        output_dir = BASE_PATH / "Data" / "Output"
        table_file = output_dir / "SMM265_DivYield_CPI_Test_Statistics_and_Significance_Results.png"
        plt.savefig(table_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"   💾 PT test table saved to: {table_file.name}")
        
        # Print results to console
        print(f"   📈 PT Test Results: {correct_predictions}/{total_predictions} correct ({proportion_correct*100:.1f}%)")
        print(f"   📊 Test statistic: {pt_statistic:.3f}, p-value: {p_value_one_sided:.3f}")
        print(f"   🎯 Significant at 5%: {'Yes' if significant_5pct else 'No'}")
        
        plt.close()
    
    def create_pt_predictions_table(self):
        """Create Pesaran-Timmermann Predictions vs Actual Results Table"""
        print("\n📊 Creating PT Predictions vs Actual Results Table...")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for PT predictions table")
            return
        
        # Filter out rows with missing actual returns
        table_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(table_data) == 0:
            print("   ❌ No complete prediction-actual pairs available for PT predictions table")
            return
        
        # Create figure with appropriate size for the table
        fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.4)))
        ax.axis('off')
        
        # Table title
        title = "Table 1 Pesaran-Timmermann Model: Predictions vs Actual Results\nMulti-Predictor Strategy (Dividend Yield + CPI)"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
        
        # Prepare table data
        table_rows = []
        headers = ['Period', 'Date', 'Predicted\nExcess Return (%)', 'Actual\nExcess Return (%)', 
                  'Investment\nDecision', 'Prediction\nCorrect']
        
        for idx, row in table_data.iterrows():
            period = len(table_rows) + 1
            date = row['prediction_date'].strftime('%Y-%m-%d') if pd.notna(row['prediction_date']) else 'N/A'
            pred_return = f"{row['predicted_excess_return']*100:.2f}%"
            actual_return = f"{row['actual_excess_return']*100:.2f}%"
            investment = row['investment_decision']
            pred_correct = "✓" if row['prediction_correct'] else "✗"
            
            table_rows.append([
                str(period), date, pred_return, actual_return, 
                investment, pred_correct
            ])
        
        # Create the table
        table = ax.table(cellText=table_rows,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0.05, 0.1, 0.9, 0.8])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.08)
        
        # Style data rows with alternating colors
        for i in range(1, len(table_rows) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                # Highlight correct/incorrect predictions
                if j == 5:  # Prediction Correct column
                    if table_rows[i-1][j] == "✓":
                        table[(i, j)].set_facecolor('#D4F6D4')  # Light green
                    else:
                        table[(i, j)].set_facecolor('#F6D4D4')  # Light red
        
        # Add summary statistics at the bottom
        correct_predictions = table_data['prediction_correct'].sum()
        total_predictions = len(table_data)
        accuracy = correct_predictions / total_predictions * 100
        
        summary_text = f"Summary: {correct_predictions}/{total_predictions} correct predictions ({accuracy:.1f}% accuracy)"
        fig.text(0.5, 0.05, summary_text, ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the table
        output_dir = BASE_PATH / "Data" / "Output"
        table_file = output_dir / "SMM265_DivYield_CPI_PT_Predictions_vs_Actual_Table.png"
        plt.ioff()
        plt.savefig(table_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.3)
        plt.close()
        print(f"   💾 PT predictions table saved to: {table_file.name}")
    
    # def create_recursive_predictions_table(self):
    #     """Create Table 1: Recursive Predictions and Actual Excess Returns"""
    #     print("\n📊 Creating Recursive Predictions Table...")
    #     
    #     if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
    #         print("   ❌ No rolling predictions available for table creation")
    #         return
    #     
    #     # Filter out rows with missing actual returns
    #     table_data_df = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
    #     
    #     if len(table_data_df) == 0:
    #         print("   ❌ No complete prediction-actual pairs available for table")
    #         return
    #     
    #     # Prepare data for the table
    #     table_data = []
    #     for _, row in table_data_df.iterrows():
    #         # Create period string (6 months from prediction date)
    #         pred_date = pd.to_datetime(row['prediction_date'])
    #         end_date = pred_date + pd.DateOffset(months=6)
    #         period_str = f"{pred_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
    #         
    #         # Get values
    #         actual_return = row['actual_excess_return']
    #         predicted_return = row['predicted_excess_return']
    #         sign_correct = "✓" if row['prediction_correct'] else "✗"
    #         trading_decision = row['investment_decision']
    #         return_achieved = row['switching_strategy_return']
    #         
    #         table_data.append([
    #             period_str,
    #             actual_return,
    #             predicted_return,
    #             sign_correct,
    #             trading_decision,
    #             return_achieved
    #         ])
    #     
    #     # Calculate summary statistics
    #     total_predictions = len(table_data)
    #     correct_predictions = sum(1 for row in table_data if row[3] == "✓")
    #     accuracy_percentage = (correct_predictions / total_predictions) * 100
    #     
    #     # Create figure with proper size for the table
    #     fig, ax = plt.subplots(figsize=(16, max(12, len(table_data) * 0.5 + 4)))
    #     ax.axis('off')
    #     
    #     # Table title
    #     title = "Table 1: Recursive Predictions and Actual Excess Returns\nFTSE All-Share Index - Rolling Window Analysis (2015-2025)"
    #     ax.text(0.5, 0.98, title, transform=ax.transAxes, 
    #             fontsize=16, fontweight='bold', ha='center', va='top')
    #     
    #     # Column headers
    #     headers = ["Period", "Actual\nReturn", "Predicted\nReturn", "Sign\nCorrect?", "Trading\nDecision", "Return\nAchieved"]
    #     
    #     # Prepare formatted data for table
    #     formatted_data = []
    #     for row in table_data:
    #         formatted_row = [
    #             row[0],  # Period
    #             f"{row[1]:.3f}",  # Actual return
    #             f"{row[2]:.3f}",  # Predicted return
    #             row[3],  # Sign correct
    #             row[4],  # Trading decision
    #             f"{row[5]:.3f}"   # Return achieved
    #         ]
    #         formatted_data.append(formatted_row)
    #     
    #     # Calculate table position based on number of rows
    #     table_height = min(0.75, len(formatted_data) * 0.03 + 0.15)
    #     table_y = 0.95 - table_height
    #     
    #     # Create the main table
    #     table = ax.table(cellText=formatted_data,
    #                     colLabels=headers,
    #                     cellLoc='center',
    #                     loc='center',
    #                     bbox=[0.05, table_y, 0.9, table_height])
    #     
    #     # Style the table
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(min(10, max(7, 120 // len(formatted_data))))
    #     table.scale(1, 1.5)
    #     
    #     # Style headers
    #     for i in range(len(headers)):
    #         table[(0, i)].set_facecolor('#4472C4')
    #         table[(0, i)].set_text_props(weight='bold', color='white')
    #     
    #     # Style data rows
    #     for i in range(1, len(formatted_data) + 1):
    #         for j in range(len(headers)):
    #             # Alternate row colors
    #             if i % 2 == 0:
    #                 table[(i, j)].set_facecolor('#F2F2F2')
    #             else:
    #                 table[(i, j)].set_facecolor('#FFFFFF')
    #             
    #             # Highlight correct/incorrect predictions
    #             if j == 3:  # Sign Correct column
    #                 if formatted_data[i-1][j] == "✓":
    #                     table[(i, j)].set_facecolor('#D5E8D4')
    #                     table[(i, j)].set_text_props(color='green', weight='bold')
    #                 else:
    #                     table[(i, j)].set_facecolor('#F8CECC')
    #                     table[(i, j)].set_text_props(color='red', weight='bold')
    #             
    #             # Style trading decision column
    #             if j == 4:  # Trading Decision column
    #                 table[(i, j)].set_text_props(weight='bold')
    #             
    #             # Add borders
    #             table[(i, j)].set_edgecolor('black')
    #             table[(i, j)].set_linewidth(0.5)
    #     
    #     # Add separator line and summary
    #     summary_y = table_y - 0.05
    #     separator_line = "─" * 80
    #     ax.text(0.5, summary_y, separator_line, transform=ax.transAxes,
    #             fontsize=12, ha='center', va='center', family='monospace')
    #     
    #     # Add summary statistics
    #     summary_text = f"Total correct: {correct_predictions} out of {total_predictions} ({accuracy_percentage:.1f}%)"
    #     ax.text(0.5, summary_y - 0.03, summary_text, transform=ax.transAxes,
    #             fontsize=12, ha='center', va='center', weight='bold')
    #     
    #     ax.text(0.5, summary_y - 0.06, separator_line, transform=ax.transAxes,
    #             fontsize=12, ha='center', va='center', family='monospace')
    #     
    #     # Add methodology note
    #     methodology_note = ("Note: Predictions based on dividend yield model with rolling 6-month re-estimation.\n"
    #                        "Trading decision: 'Stocks' if predicted excess return > 0, otherwise 'Bonds'.")
    #     
    #     ax.text(0.5, 0.02, methodology_note, transform=ax.transAxes,
    #             fontsize=9, ha='center', va='bottom', style='italic')
    #     
    #     plt.tight_layout()
    #     
    #     # Save the table
    #     output_dir = BASE_PATH / "Data" / "Output"
    #     table_file = output_dir / "SMM265_DivYield_CPI_Recursive_Predictions_Table.png"
    #     plt.ioff()
    #     plt.savefig(table_file, dpi=300, bbox_inches='tight', 
    #                 facecolor='white', edgecolor='none', pad_inches=0.3)
    #     plt.close()
    #     print(f"   💾 Recursive predictions table saved to: {table_file.name}")
    #     
    #     # Print summary statistics
    #     print(f"   📈 {correct_predictions}/{total_predictions} correct predictions ({accuracy_percentage:.1f}%)")
    
    def create_simple_visualizations(self):
        """Create basic visualizations"""
        print("\n📊 Creating visualizations...")
        
        if self.model is None:
            print("   ❌ No model available for visualizations")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pesaran & Timmermann Analysis', fontsize=16)
        
        # 1. Training scatter
        ax1.scatter(self.training_data['dividend_yield'], self.training_data['excess_return']*100, alpha=0.6)
        ax1.set_xlabel('Dividend Yield (%)')
        ax1.set_ylabel('Excess Return (%)')
        ax1.set_title('Training Data')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative returns
        ax2.plot(self.prediction_data['Date'], self.prediction_data['switching_cumret'], label='Switching')
        ax2.plot(self.prediction_data['Date'], self.prediction_data['buyhold_cumret'], label='Buy-Hold')
        ax2.set_title('Strategy Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Investment decisions
        colors = ['red' if x == 0 else 'green' for x in self.prediction_data['invest_in_stocks']]
        ax3.scatter(self.prediction_data['Date'], self.prediction_data['predicted_excess_return']*100, c=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--')
        ax3.set_title('Investment Decisions')
        ax3.grid(True, alpha=0.3)
        
        # 4. Return distribution
        ax4.hist(self.prediction_data['switching_return']*100, alpha=0.7, label='Switching', bins=15)
        ax4.hist(self.prediction_data['buyhold_return']*100, alpha=0.7, label='Buy-Hold', bins=15)
        ax4.set_title('Return Distributions')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save instead of showing
        output_dir = BASE_PATH / "Data" / "Output"
        plot_file = output_dir / "SMM265_DivYield_CPI_Simple_Visualizations.png"
        plt.ioff()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Simple visualizations saved to: {plot_file.name}")
    
    def run_full_analysis(self):
        """Run the complete rolling window analysis with evaluation"""
        try:
            if not self.load_and_prepare_data():
                return False
            
            self.compute_returns()
            self.prepare_initial_training_data()
            
            # Show initial training for reference
            print("\n📋 INITIAL TRAINING (for reference)")
            print("="*50)
            self.train_model()
            
            # Run rolling window predictions
            rolling_results = self.rolling_window_predictions()
            
            if len(rolling_results) > 0:
                print(f"\n🎉 ROLLING WINDOW ANALYSIS COMPLETE!")
                print(f"   📊 Generated {len(rolling_results)} predictions")
                print(f"   📈 Coefficient evolution shows market adaptability")
                
                # Display final evaluation summary
                self.display_final_evaluation_summary()
                
                # Generate time-series plot
                self.plot_predicted_vs_actual_returns()
                
                # Generate cumulative returns chart
                self.plot_cumulative_returns_chart()
                
                # Generate Pesaran-Timmermann test table
                self.create_pt_test_table()
                self.create_pt_predictions_table()
                
                # Generate recursive predictions table
                # self.create_recursive_predictions_table()  # Commented out - not needed
            else:
                print(f"\n❌ No rolling predictions generated - check data availability")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_final_evaluation_summary(self):
        """Display comprehensive final summary of strategy evaluation"""
        if not hasattr(self, 'performance_metrics'):
            print("\n⚠️  No performance metrics available")
            return
            
        print(f"\n" + "="*80)
        print(f"🏁 FINAL STRATEGY EVALUATION SUMMARY")
        print(f"="*80)
        
        metrics = self.performance_metrics
        
        print(f"\n📈 PREDICTION PERFORMANCE:")
        print(f"   • Total prediction periods: {metrics['total_predictions']}")
        print(f"   • Correct direction predictions: {metrics['correct_predictions']} ({metrics['accuracy']*100:.1f}%)")
        print(f"   • Periods outperformed benchmark: {int(metrics['win_rate'] * metrics['total_predictions'])}/{metrics['total_predictions']} ({metrics['win_rate']*100:.1f}%)")
        
        print(f"\n💰 RETURN COMPARISON (Annualized):")
        print(f"   • Switching Strategy: {metrics['switching_annual_return']*100:.1f}%")
        print(f"   • Buy-and-Hold Benchmark: {metrics['buyhold_annual_return']*100:.1f}%")
        print(f"   • Strategy Advantage: {(metrics['switching_annual_return'] - metrics['buyhold_annual_return'])*100:.1f}%")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   • Switching Strategy Volatility: {metrics['switching_volatility']*100:.1f}%")
        print(f"   • Buy-and-Hold Volatility: {metrics['buyhold_volatility']*100:.1f}%")
        print(f"   • Switching Strategy Sharpe: {metrics['switching_sharpe']:.3f}")
        print(f"   • Buy-and-Hold Sharpe: {metrics['buyhold_sharpe']:.3f}")
        
        print(f"\n🎯 INVESTMENT DECISION BREAKDOWN:")
        if hasattr(self, 'rolling_predictions'):
            stock_decisions = self.rolling_predictions['invest_in_stocks'].sum()
            total_decisions = len(self.rolling_predictions)
            interbank_decisions = total_decisions - stock_decisions
            print(f"   • Invested in Stocks: {stock_decisions}/{total_decisions} periods ({stock_decisions/total_decisions*100:.1f}%)")
            print(f"   • Invested in InterBank: {interbank_decisions}/{total_decisions} periods ({interbank_decisions/total_decisions*100:.1f}%)")
        
        print(f"\n📄 OUTPUT FILES:")
        print(f"   • SMM265_DivYield_CPI_Strategy_Evaluation_Results.xlsx - Detailed evaluation results")
        print(f"   • SMM265_DivYield_CPI_Rolling_Window_Coefficients_and_Predictions.xlsx - Model coefficients")
        
        # Final verdict
        return_advantage = metrics['switching_annual_return'] - metrics['buyhold_annual_return']
        sharpe_advantage = metrics['switching_sharpe'] - metrics['buyhold_sharpe']
        
        print(f"\n" + "="*80)
        print(f"🏆 FINAL VERDICT:")
        if return_advantage > 0 and sharpe_advantage > 0:
            print(f"   ✅ STRATEGY SUCCESS: The multi-predictor switching strategy outperformed")
            print(f"      buy-and-hold in both returns ({return_advantage*100:.1f}%) and risk-adjusted returns ({sharpe_advantage:.3f})")
        elif return_advantage > 0:
            print(f"   📈 MIXED RESULTS: Higher returns ({return_advantage*100:.1f}%) but lower risk-adjusted performance")
        elif sharpe_advantage > 0:
            print(f"   📊 RISK-ADJUSTED SUCCESS: Better Sharpe ratio despite lower returns")
        else:
            print(f"   ❌ STRATEGY UNDERPERFORMANCE: Buy-and-hold outperformed in both metrics")
            
        print(f"   📊 Prediction accuracy: {metrics['accuracy']*100:.1f}% - {'Good' if metrics['accuracy'] > 0.6 else 'Moderate' if metrics['accuracy'] > 0.5 else 'Poor'}")
        print(f"="*80)
        
        # Create and save performance table
        self.create_performance_table(metrics)
    
    def create_performance_table(self, metrics):
        """Create a professional performance summary table as an image"""
        try:
            from matplotlib.patches import FancyBboxPatch
            
            # Create output directory if it doesn't exist
            output_dir = Path("output").resolve()  # Use absolute path
            output_dir.mkdir(exist_ok=True)
            print(f"   📁 Creating output directory: {output_dir}")
            
            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Define colors
            header_color = '#2E86AB'
            section_color = '#A23B72'
            positive_color = '#F18F01'
            neutral_color = '#C73E1D'
            
            # Title
            title_box = FancyBboxPatch((0.5, 9), 9, 0.8, boxstyle="round,pad=0.1", 
                                      facecolor=header_color, edgecolor='black', linewidth=2)
            ax.add_patch(title_box)
            ax.text(5, 9.4, '🏆 STRATEGY PERFORMANCE EVALUATION', 
                    fontsize=18, fontweight='bold', ha='center', va='center', color='white')
            ax.text(5, 9.1, 'Pesaran & Timmermann Multi-Predictor Model (Dividend Yield + CPI)', 
                    fontsize=12, ha='center', va='center', color='white')
            
            # Section 1: Prediction Accuracy
            y_pos = 8.2
            section1_box = FancyBboxPatch((0.5, y_pos-0.4), 9, 1.2, boxstyle="round,pad=0.05", 
                                         facecolor='#E8F4F8', edgecolor=section_color, linewidth=1.5)
            ax.add_patch(section1_box)
            ax.text(1, y_pos+0.4, 'PREDICTION ACCURACY', fontsize=14, fontweight='bold', color=section_color)
            ax.text(1.2, y_pos+0.1, '🎯 Correct predictions:', fontsize=11, color='black')
            ax.text(7, y_pos+0.1, f"{int(metrics['correct_predictions'])}/{int(metrics['total_predictions'])} ({metrics['accuracy']*100:.1f}%)", 
                    fontsize=11, fontweight='bold', color=positive_color)
            
            win_periods = int(metrics['win_rate'] * metrics['total_predictions'])
            ax.text(1.2, y_pos-0.1, '🏆 Win periods (outperformed):', fontsize=11, color='black')
            ax.text(7, y_pos-0.1, f"{win_periods}/{int(metrics['total_predictions'])} ({metrics['win_rate']*100:.1f}%)", 
                    fontsize=11, fontweight='bold', color=neutral_color)
            
            ax.text(1.2, y_pos-0.3, '📊 Average outperformance:', fontsize=11, color='black')
            ax.text(7, y_pos-0.3, f"{metrics['avg_outperformance']:.2f}% per period", fontsize=11, fontweight='bold', color=positive_color)
            
            # Section 2: Strategy Performance
            y_pos = 6.5
            section2_box = FancyBboxPatch((0.5, y_pos-0.5), 4.2, 1.4, boxstyle="round,pad=0.05", 
                                         facecolor='#F0F8E8', edgecolor=section_color, linewidth=1.5)
            ax.add_patch(section2_box)
            ax.text(1, y_pos+0.6, 'SWITCHING STRATEGY', fontsize=12, fontweight='bold', color=section_color)
            ax.text(1.2, y_pos+0.3, '💰 Total return:', fontsize=10, color='black')
            ax.text(3.8, y_pos+0.3, f"{metrics['switching_total_return']*100:.1f}%", fontsize=10, fontweight='bold', color=positive_color)
            ax.text(1.2, y_pos+0.1, '📈 Annualized return:', fontsize=10, color='black')
            ax.text(3.8, y_pos+0.1, f"{metrics['switching_annual_return']*100:.1f}%", fontsize=10, fontweight='bold', color=positive_color)
            ax.text(1.2, y_pos-0.1, '📊 Volatility:', fontsize=10, color='black')
            ax.text(3.8, y_pos-0.1, f"{metrics['switching_volatility']*100:.1f}%", fontsize=10, fontweight='bold', color='black')
            ax.text(1.2, y_pos-0.3, '⚡ Sharpe ratio:', fontsize=10, color='black')
            ax.text(3.8, y_pos-0.3, f"{metrics['switching_sharpe']:.3f}", fontsize=10, fontweight='bold', color=positive_color)
            
            # Section 3: Benchmark Performance
            section3_box = FancyBboxPatch((5.3, y_pos-0.5), 4.2, 1.4, boxstyle="round,pad=0.05", 
                                         facecolor='#FFF8E8', edgecolor=section_color, linewidth=1.5)
            ax.add_patch(section3_box)
            ax.text(5.8, y_pos+0.6, 'BUY-AND-HOLD BENCHMARK', fontsize=12, fontweight='bold', color=section_color)
            ax.text(6, y_pos+0.3, '💰 Total return:', fontsize=10, color='black')
            ax.text(8.6, y_pos+0.3, f"{metrics['buyhold_total_return']*100:.1f}%", fontsize=10, fontweight='bold', color='black')
            ax.text(6, y_pos+0.1, '📈 Annualized return:', fontsize=10, color='black')
            ax.text(8.6, y_pos+0.1, f"{metrics['buyhold_annual_return']*100:.1f}%", fontsize=10, fontweight='bold', color='black')
            ax.text(6, y_pos-0.1, '📊 Volatility:', fontsize=10, color='black')
            ax.text(8.6, y_pos-0.1, f"{metrics['buyhold_volatility']*100:.1f}%", fontsize=10, fontweight='bold', color='black')
            ax.text(6, y_pos-0.3, '⚡ Sharpe ratio:', fontsize=10, color='black')
            ax.text(8.6, y_pos-0.3, f"{metrics['buyhold_sharpe']:.3f}", fontsize=10, fontweight='bold', color='black')
            
            # Section 4: Strategy Comparison
            y_pos = 4.5
            section4_box = FancyBboxPatch((0.5, y_pos-0.4), 9, 1, boxstyle="round,pad=0.05", 
                                         facecolor='#E8F8E8', edgecolor=section_color, linewidth=1.5)
            ax.add_patch(section4_box)
            ax.text(1, y_pos+0.3, 'STRATEGY COMPARISON', fontsize=14, fontweight='bold', color=section_color)
            
            return_advantage = metrics['switching_annual_return'] - metrics['buyhold_annual_return']
            sharpe_advantage = metrics['switching_sharpe'] - metrics['buyhold_sharpe']
            
            ax.text(1.2, y_pos, '🚀 Return advantage:', fontsize=11, color='black')
            ax.text(4.5, y_pos, f"{return_advantage*100:.1f}%", fontsize=11, fontweight='bold', color=positive_color)
            ax.text(6, y_pos, '🎯 Sharpe advantage:', fontsize=11, color='black')
            ax.text(8.5, y_pos, f"{sharpe_advantage:.3f}", fontsize=11, fontweight='bold', color=positive_color)
            
            # Section 5: Results Summary
            y_pos = 3.2
            section5_box = FancyBboxPatch((0.5, y_pos-0.6), 9, 1.4, boxstyle="round,pad=0.05", 
                                         facecolor='#F8F8F8', edgecolor=section_color, linewidth=1.5)
            ax.add_patch(section5_box)
            ax.text(1, y_pos+0.5, 'ANALYSIS SUMMARY', fontsize=14, fontweight='bold', color=section_color)
            ax.text(1.2, y_pos+0.2, '💾 Results saved to: SMM265_DivYield_CPI_Strategy_Evaluation_Results.xlsx', fontsize=10, color='black')
            ax.text(1.2, y_pos, '📅 All dates in date-only format (no timestamps)', fontsize=10, color='black')
            ax.text(1.2, y_pos-0.2, f"🎉 Rolling window analysis complete: {int(metrics['total_predictions'])} predictions generated", fontsize=10, color='black')
            ax.text(1.2, y_pos-0.4, '📈 Coefficient evolution demonstrates market adaptability', fontsize=10, color='black')
            
            # Success indicator
            y_pos = 1.5
            if return_advantage > 0 and sharpe_advantage > 0:
                success_text = '✅ STRATEGY SUCCESS: Multi-predictor outperformed buy-and-hold'
                success_color = positive_color
            else:
                success_text = '⚠️ MIXED RESULTS: See detailed analysis above'
                success_color = neutral_color
                
            success_box = FancyBboxPatch((2, y_pos-0.4), 6, 0.8, boxstyle="round,pad=0.1", 
                                        facecolor=success_color, edgecolor='black', linewidth=2)
            ax.add_patch(success_box)
            ax.text(5, y_pos, success_text, 
                    fontsize=12, fontweight='bold', ha='center', va='center', color='white')
            
            # Footer
            ax.text(5, 0.5, 'SMM265 Asset Pricing - Pesaran & Timmermann (1995) Implementation', 
                    fontsize=10, ha='center', va='center', color='gray', style='italic')
            
            plt.tight_layout()
            
            # Save the figure
            filename = output_dir / 'SMM265_DivYield_CPI_Strategy_Performance_Table.png'
            print(f"   💾 Attempting to save: {filename}")
            
            # Ensure the plot doesn't show (which might prevent saving)
            plt.ioff()  # Turn off interactive mode
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            # Verify file was saved
            if filename.exists():
                print(f"\n📊 PERFORMANCE TABLE SAVED:")
                print(f"   📈 Visual summary: {filename}")
                print(f"   📏 File size: {filename.stat().st_size} bytes")
            else:
                print(f"   ❌ File was not saved to: {filename}")
                
        except Exception as e:
            print(f"   ⚠️ Could not create performance table: {e}")
            import traceback
            traceback.print_exc()
            print(f"   📄 Detailed results available in Excel files")

def main():
    analyzer = PesaranTimmermannMultiPredictor()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()