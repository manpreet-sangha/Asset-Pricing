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
        cpi_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265_UK_CPI_Monthly.xlsx"
        
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
            
            # Convert dates to datetime - using lowercase 'date' for both
            index_data['date'] = pd.to_datetime(index_data['date'])
            interbank_data['date'] = pd.to_datetime(interbank_data['date'])
            
            # Create year-month columns for matching (CCYY-MM format)
            index_data['year_month'] = index_data['date'].dt.to_period('M')
            interbank_data['year_month'] = interbank_data['date'].dt.to_period('M')
            
            print(f"   🔍 Index date range: {index_data['year_month'].min()} to {index_data['year_month'].max()}")
            print(f"   🔍 InterBank date range: {interbank_data['year_month'].min()} to {interbank_data['year_month'].max()}")
            
            # Find common year-months
            common_months = set(index_data['year_month']) & set(interbank_data['year_month'])
            print(f"   🔍 Common year-months: {len(common_months)}")
            
            # Merge on year-month, keeping the original date from index file
            self.data = pd.merge(
                index_data, 
                interbank_data[['year_month', 'interbank_rate']],
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
            output_file = output_dir / "SMM265_Index_DivYield_InterBank_Monthly_Merged.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.data.to_excel(writer, sheet_name='Data', index=False)
            
            print(f"   💾 Merged data saved to: {output_file.name}")
            print(f"   ✅ Loaded dataset: {len(self.data)} observations")
            print(f"   📅 Period: {self.data['Date'].min().strftime('%Y-%m')} to {self.data['Date'].max().strftime('%Y-%m')}")
            print(f"   📊 Columns: {list(self.data.columns)}")
            print(f"   📈 Using UK InterBank Rate")
            
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
        """Compute 6-month returns and excess returns for training data only"""
        print("📈 Computing 6-month returns and excess returns...")
        print("   🎯 Computing returns ONLY for initial training period")
        print("   📅 Training period: Apr 1997 - Sep 2015")
        print("   📅 Need 6m ahead data until Mar 2016 to compute training returns")
        
        df = self.data.copy()
        
        # Define the cutoff for training data - we need historical data until Mar 2016 to compute
        # realized 6-month returns for observations up to Sep 2015
        training_end = pd.to_datetime('2015-09-30')
        data_needed_until = pd.to_datetime('2016-03-31')  # Need historical data 6 months past training end
        
        # Find observations that can be used for training
        # (observations from Apr 1997 to Sep 2015 where we can calculate realized returns)
        training_start = pd.to_datetime('1997-04-30')
        
        # Filter data to only include training period where we can compute 6m returns
        training_mask = (df['Date'] >= training_start) & (df['Date'] <= training_end)
        training_indices = df[training_mask].index.tolist()
        
        print(f"   📊 Training observations (Apr 1997 - Sep 2015): {len(training_indices)}")
        
        # Only compute returns for training observations where we have historical data
        # to calculate their realized 6-month returns (need stock prices 6 months later)
        valid_training_indices = []
        for idx in training_indices:
            # Check if we have historical stock price data 6 months later to compute realized returns
            if idx + 6 < len(df):
                future_date = df.iloc[idx + 6]['Date']
                if future_date <= data_needed_until:
                    valid_training_indices.append(idx)
        
        print(f"   📊 Valid training observations with historical data for return calculation: {len(valid_training_indices)}")
        
        # Initialize return columns
        df['ftse_6m_return'] = np.nan
        df['m_interbank_6m_return'] = np.nan
        df['excess_return'] = np.nan
        
        # Compute 6-month stock returns only for valid training indices
        for idx in valid_training_indices:
            if idx + 6 < len(df):
                df.iloc[idx, df.columns.get_loc('ftse_6m_return')] = (
                    df.iloc[idx + 6]['ftse_price'] / df.iloc[idx]['ftse_price'] - 1
                )
        
        # Compute 6-month InterBank returns using current period rates
        # This calculates the expected 6m return if investing at current InterBank rate
        df['m_interbank_6m_return'] = (1 + df['m_interbank_rate']/100)**(0.5) - 1
        
        print(df['m_interbank_6m_return'])

        # Compute excess returns only where we have stock returns
        mask = ~df['ftse_6m_return'].isna()
        df.loc[mask, 'excess_return'] = (
            df.loc[mask, 'ftse_6m_return'] - df.loc[mask, 'm_interbank_6m_return']
        )
        
        # Keep only observations with valid excess returns for training
        df_training = df[~df['excess_return'].isna()].copy().reset_index(drop=True)
        
        print(f"   ✅ Computed returns for {len(df_training)} training observations")
        
        if len(df_training) > 0:
            print(f"   📊 Average 6m stock return: {df_training['ftse_6m_return'].mean()*100:.2f}%")
            print(f"   📊 Average 6m InterBank return: {df_training['m_interbank_6m_return'].mean()*100:.2f}%")
            print(f"   📊 Average excess return: {df_training['excess_return'].mean()*100:.2f}%")
            print(f"   📅 Training period: {df_training['Date'].min().strftime('%Y-%m-%d')} to {df_training['Date'].max().strftime('%Y-%m-%d')}")
        
        # Store both full data (for rolling window) and training data
        self.full_data = df  # Keep full data for rolling window
        self.data = df_training  # Use training data for initial model

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
        
        print(f"   📊 Training data with returns: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')} ({len(self.data)} obs)")
        
        # Convert date strings to datetime
        training_start = pd.to_datetime(self.training_start)
        training_end = pd.to_datetime(self.training_end)
        prediction_start = pd.to_datetime(self.prediction_start)
        prediction_end = pd.to_datetime(self.prediction_end)
        
        print(f"   🎯 Initial training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
        print(f"   🎯 Rolling prediction starts: {prediction_start.strftime('%Y-%m-%d')}")
        
        # Training period: Apr 1997 → Sep 2015 (ONLY FOR REFERENCE)
        training_mask = (self.data['Date'] >= training_start) & (self.data['Date'] <= training_end)
        self.training_data = self.data[training_mask].copy().reset_index(drop=True)
        
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
        """Train the predictive regression model on training data"""
        print("\n🎓 TRAINING PHASE")
        print("="*50)
        print("OBJECTIVE: Estimate α̂ and β̂ coefficients")
        print("Model: Excess_Return = α + β × Dividend_Yield + ε")
        print(f"Training period: {self.training_start} to {self.training_end}")
        print(f"For predicting: {self.prediction_start} onwards")
        print("\nSTEPS:")
        print("1. ✅ Computed 6-month stock returns: R_stock = (FTSE_{t+6} / FTSE_t) - 1")
        print("2. ✅ Computed 6-month InterBank returns: R_interbank = (1 + annual_yield/100)^0.5 - 1") 
        print("3. ✅ Computed excess returns: Y = R_stock - R_interbank")
        print("4. ✅ Used dividend yield as predictor: X = Dividend_Yield_t")
        print("5. 🔄 Running OLS regression: Y = α + β × X + ε")
        
        if not STATSMODELS_AVAILABLE:
            print("   ❌ Statsmodels not available - using simple linear regression")
            return self.train_model_simple()
        
        if len(self.training_data) == 0:
            print("   ❌ No training data available")
            return
        
        # Prepare training data
        y_train = self.training_data['excess_return'].values  # Y: Dependent variable (excess returns)
        X_train = self.training_data['dividend_yield'].values  # X: Independent variable (dividend yield)
        
        # Add constant term for regression intercept
        X_train_with_const = add_constant(X_train)
        
        # Fit OLS regression to estimate α̂ and β̂
        self.model = OLS(y_train, X_train_with_const).fit()
        
        print("\n📊 TRAINING RESULTS:")
        print("-" * 40)
        print(self.model.summary())
        
        # Extract key statistics - these are α̂ and β̂
        alpha_hat = self.model.params[0]  # α̂ (intercept estimate)
        beta_hat = self.model.params[1]   # β̂ (dividend yield coefficient estimate)
        alpha_pval = self.model.pvalues[0]
        beta_pval = self.model.pvalues[1]
        r_squared = self.model.rsquared
        
        print(f"\n📈 ESTIMATED COEFFICIENTS:")
        print(f"   α̂ (alpha hat - intercept): {alpha_hat:.6f} (p-value: {alpha_pval:.4f})")
        print(f"   β̂ (beta hat - dividend yield): {beta_hat:.6f} (p-value: {beta_pval:.4f})")
        print(f"   R-squared: {r_squared:.4f}")
        print(f"   Training observations: {len(self.training_data)}")
        
        # Store training results for use in prediction phase
        self.results['training'] = {
            'alpha_hat': alpha_hat,
            'beta_hat': beta_hat,
            'alpha_pval': alpha_pval,
            'beta_pval': beta_pval,
            'r_squared': r_squared,
            'observations': len(self.training_data)
        }
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if beta_pval < 0.05:
            if beta_hat > 0:
                print(f"   ✅ Dividend yield has POSITIVE predictive power")
                print(f"   📈 1% higher dividend yield → {beta_hat*100:.2f}% higher excess return")
            else:
                print(f"   ✅ Dividend yield has NEGATIVE predictive power")
                print(f"   📉 1% higher dividend yield → {abs(beta_hat)*100:.2f}% lower excess return")
        else:
            print(f"   ❌ Dividend yield has NO significant predictive power")
            
        print(f"\n🎯 NEXT STEP: Use α̂ = {alpha_hat:.4f} and β̂ = {beta_hat:.4f} for prediction phase")
    
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
        print("• Use α̂, β̂ + current dividend yield to predict next 6 months")
        print("• No forward-looking bias: only use information available at prediction time")
        
        # Create prediction periods (every 6 months starting Oct 2015)
        prediction_dates = pd.date_range(
            start=pd.to_datetime(self.prediction_start),
            end=pd.to_datetime(self.prediction_end),
            freq='6M'
        )
        
        print(f"\n📅 Prediction schedule: {len(prediction_dates)} periods")
        for i, date in enumerate(prediction_dates[:5]):  # Show first 5
            print(f"   {i+1}. {date.strftime('%Y-%m-%d')}")
        if len(prediction_dates) > 5:
            print(f"   ... and {len(prediction_dates)-5} more periods")
        
        # Initialize results storage
        rolling_results = []
        
        for i, pred_date in enumerate(prediction_dates):
            print(f"\n🔄 ITERATION {i+1}: Predicting for {pred_date.strftime('%Y-%m-%d')}")
            
            # Define training end as 1 month before prediction date (use all data up to previous month)
            training_end_date = pred_date - pd.DateOffset(months=1)
            print(f"   📅 Training window: Apr 1997 → {training_end_date.strftime('%Y-%m-%d')}")
            
            # Create training data (Apr 1997 → training_end_date)
            training_start_dt = pd.to_datetime(self.training_start)
            training_mask = (
                (self.full_data['Date'] >= training_start_dt) & 
                (self.full_data['Date'] <= training_end_date)
            )
            current_training_data = self.full_data[training_mask].copy()
            
            # For rolling window, we need to compute returns dynamically for this training period
            # Compute returns for current training period
            for idx in range(len(current_training_data)):
                if idx + 6 < len(self.full_data):
                    current_date = current_training_data.iloc[idx]['Date']
                    future_idx = current_training_data.index[idx] + 6
                    if future_idx < len(self.full_data):
                        future_price = self.full_data.iloc[future_idx]['ftse_price']
                        current_price = current_training_data.iloc[idx]['ftse_price']
                        current_training_data.iloc[idx, current_training_data.columns.get_loc('ftse_6m_return')] = (
                            future_price / current_price - 1
                        )
            
            # Compute InterBank returns and excess returns
            current_training_data['m_interbank_6m_return'] = (1 + current_training_data['m_interbank_rate']/100)**(0.5) - 1
            current_training_data['excess_return'] = (
                current_training_data['ftse_6m_return'] - current_training_data['m_interbank_6m_return']
            )
            
            # Keep only observations with valid excess returns
            current_training_data = current_training_data.dropna(subset=['excess_return']).copy()
            
            print(f"   📚 Training: {training_start_dt.strftime('%Y-%m')} to {training_end_date.strftime('%Y-%m')} ({len(current_training_data)} obs)")
            
            if len(current_training_data) < 24:  # Need minimum data
                print(f"   ⚠️  Insufficient training data, skipping...")
                continue
            
            # Train model on current data
            alpha_hat, beta_hat, r_squared = self.train_rolling_model(current_training_data, i+1)
            
            if alpha_hat is None:  # Training failed
                continue
            
            # Get dividend yield for prediction
            div_yield_date = training_end_date  # Use most recent dividend yield
            div_yield_mask = self.full_data['Date'] == div_yield_date
            
            if not div_yield_mask.any():
                # Find closest date
                closest_idx = (self.full_data['Date'] - div_yield_date).abs().idxmin()
                current_div_yield = self.full_data.loc[closest_idx, 'dividend_yield']
                actual_date = self.full_data.loc[closest_idx, 'Date']
                print(f"   📊 Using dividend yield from {actual_date.strftime('%Y-%m-%d')}: {current_div_yield:.2f}%")
            else:
                current_div_yield = self.full_data[div_yield_mask]['dividend_yield'].iloc[0]
                print(f"   📊 Dividend yield on {div_yield_date.strftime('%Y-%m-%d')}: {current_div_yield:.2f}%")
            
            # Make prediction
            predicted_excess_return = alpha_hat + beta_hat * current_div_yield
            investment_decision = "STOCKS" if predicted_excess_return > 0 else "INTERBANK"
            
            print(f"   🎯 Prediction: Ŷ = {alpha_hat:.6f} + {beta_hat:.6f} × {current_div_yield:.2f} = {predicted_excess_return:.4f}")
            print(f"   💼 Investment Decision: {investment_decision}")
            
            # Store results with dates formatted properly
            rolling_results.append({
                'prediction_date': pred_date.date(),  # Convert to date only
                'training_start': training_start_dt.date(),  # Convert to date only
                'training_end': training_end_date.date(),  # Convert to date only
                'training_observations': len(current_training_data),
                'alpha_hat': alpha_hat,
                'beta_hat': beta_hat,
                'r_squared': r_squared,
                'dividend_yield': current_div_yield,
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
            coefficients_file = output_dir / "Rolling_Window_Coefficients_and_Predictions.xlsx"
            
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
        print(f"   Average β̂: {self.rolling_predictions['beta_hat'].mean():.6f}")
        print(f"   α̂ range: [{self.rolling_predictions['alpha_hat'].min():.6f}, {self.rolling_predictions['alpha_hat'].max():.6f}]")
        print(f"   β̂ range: [{self.rolling_predictions['beta_hat'].min():.6f}, {self.rolling_predictions['beta_hat'].max():.6f}]")
        
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
            
            # Calculate 6-month period end date
            period_end_date = pred_date + pd.DateOffset(months=6)
            
            print(f"\n🔍 EVALUATION {idx+1}: Prediction for {pred_date.strftime('%Y-%m-%d')}")
            print(f"   📅 Investment period: {pred_date.strftime('%Y-%m-%d')} → {period_end_date.strftime('%Y-%m-%d')}")
            print(f"   💼 Predicted decision: {investment_decision}")
            print(f"   📈 Predicted excess return: {predicted_excess_return:.4f}")
            
            # Get actual returns for this period
            actual_ftse_return, actual_interbank_return = self.calculate_period_actual_returns(
                pred_date, period_end_date
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
                switching_return = actual_ftse_return if investment_decision == "STOCKS" else actual_interbank_return
                
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
        data_source = data_source.copy()
        data_source['date_only'] = data_source['Date'].dt.date
        
        start_mask = data_source['date_only'] <= start_date
        end_mask = data_source['date_only'] <= end_date
        
        if not start_mask.any() or not end_mask.any():
            print(f"   ❌ No data found for period {start_date} to {end_date}")
            return None, None
        
        # Get closest available dates
        start_idx = data_source[start_mask].index[-1]  # Latest date <= start_date
        end_indices = data_source[end_mask].index
        
        if len(end_indices) == 0 or end_indices[-1] <= start_idx:
            print(f"   ❌ Insufficient data range for period {start_date} to {end_date}")
            return None, None
        
        end_idx = end_indices[-1]  # Latest date <= end_date
        
        start_data = data_source.iloc[start_idx]
        end_data = data_source.iloc[end_idx]
        
        actual_start_date = start_data['Date'].date()
        actual_end_date = end_data['Date'].date()
        
        print(f"      📅 Actual period: {actual_start_date} to {actual_end_date}")
        
        # Calculate actual FTSE return
        ftse_start_price = start_data['ftse_price']
        ftse_end_price = end_data['ftse_price']
        actual_ftse_return = (ftse_end_price / ftse_start_price) - 1
        
        # Calculate actual InterBank return using the rates during the period
        # We'll use the average InterBank rate during the investment period
        period_mask = (
            (data_source['date_only'] >= actual_start_date) & 
            (data_source['date_only'] <= actual_end_date)
        )
        period_data = data_source[period_mask]
        
        if len(period_data) > 0:
            avg_interbank_rate = period_data['m_interbank_rate'].mean()
            # Calculate return for the actual period length
            period_length_years = (actual_end_date - actual_start_date).days / 365.25
            actual_interbank_return = (1 + avg_interbank_rate/100)**period_length_years - 1
            
            print(f"      💰 FTSE price: {ftse_start_price:.2f} → {ftse_end_price:.2f}")
            print(f"      📈 Avg InterBank rate: {avg_interbank_rate:.2f}% for {period_length_years:.2f} years")
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
        
        evaluation_file = output_dir / "Strategy_Evaluation_Results.xlsx"
        
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
        """Train model for rolling window iteration"""
        try:
            # Prepare data
            y_train = training_data['excess_return'].values
            X_train = training_data['dividend_yield'].values
            
            # Run regression
            if STATSMODELS_AVAILABLE:
                X_train_with_const = add_constant(X_train)
                model = OLS(y_train, X_train_with_const).fit()
                alpha_hat = model.params[0]
                beta_hat = model.params[1]
                r_squared = model.rsquared
            else:
                # Simple regression fallback
                X_matrix = np.column_stack([np.ones(len(X_train)), X_train])
                coefficients = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_train
                alpha_hat = coefficients[0]
                beta_hat = coefficients[1]
                
                y_pred = X_matrix @ coefficients
                ss_res = np.sum((y_train - y_pred) ** 2)
                ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
            
            print(f"   📈 Coefficients: α̂ = {alpha_hat:.6f}, β̂ = {beta_hat:.6f}, R² = {r_squared:.4f}")
            return alpha_hat, beta_hat, r_squared
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return None, None, None
    
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
        plt.title('Pesaran & Timmermann Model: Predicted vs Actual Excess Returns\n(Rolling Window Predictions 2015-2025)', 
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
        plot_file = output_dir / "Predicted_vs_Actual_Returns_TimeSeries.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   💾 Time-series plot saved to: {plot_file.name}")
        
        plt.show()
    
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
        
        # Calculate cumulative returns for both strategies
        plot_data['switching_cumulative'] = (1 + plot_data['switching_strategy_return']).cumprod()
        plot_data['buyhold_cumulative'] = (1 + plot_data['buyhold_strategy_return']).cumprod()
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative returns
        plt.plot(plot_data['prediction_date_dt'], plot_data['switching_cumulative'], 
                linewidth=3, label='Switching Strategy', color='blue', alpha=0.8)
        
        plt.plot(plot_data['prediction_date_dt'], plot_data['buyhold_cumulative'], 
                linewidth=3, label='Buy-and-Hold (FTSE)', color='red', alpha=0.8)
        
        # Add horizontal line at 1.0 (no gain/loss)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        plt.title('Cumulative Returns: Switching Strategy vs Buy-and-Hold\n(Pesaran & Timmermann Rolling Window 2015-2025)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (1 = No Change)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add summary statistics as text box
        final_switching = plot_data['switching_cumulative'].iloc[-1]
        final_buyhold = plot_data['buyhold_cumulative'].iloc[-1]
        switching_return = (final_switching - 1) * 100
        buyhold_return = (final_buyhold - 1) * 100
        outperformance = switching_return - buyhold_return
        
        textstr = f'Final Returns:\n'
        textstr += f'Switching Strategy: {switching_return:.1f}%\n'
        textstr += f'Buy-and-Hold: {buyhold_return:.1f}%\n'
        textstr += f'Outperformance: {outperformance:.1f}%'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = BASE_PATH / "Data" / "Output"
        plot_file = output_dir / "Cumulative_Returns_Strategy_Comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   💾 Cumulative returns chart saved to: {plot_file.name}")
        
        plt.show()
    
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
        
        # Number of switches (our model always chose STOCKS)
        num_switches = 0  # All predictions were STOCKS
        
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
        table_file = output_dir / "Pesaran_Timmermann_Sign_Test_Table.png"
        plt.savefig(table_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"   💾 PT test table saved to: {table_file.name}")
        
        # Print results to console
        print(f"   📈 PT Test Results: {correct_predictions}/{total_predictions} correct ({proportion_correct*100:.1f}%)")
        print(f"   📊 Test statistic: {pt_statistic:.3f}, p-value: {p_value_one_sided:.3f}")
        print(f"   🎯 Significant at 5%: {'Yes' if significant_5pct else 'No'}")
        
        plt.show()
    
    def create_recursive_predictions_table(self):
        """Create Table 1: Recursive Predictions and Actual Excess Returns"""
        print("\n📊 Creating Recursive Predictions Table...")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            print("   ❌ No rolling predictions available for table creation")
            return
        
        # Filter out rows with missing actual returns
        table_data_df = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(table_data_df) == 0:
            print("   ❌ No complete prediction-actual pairs available for table")
            return
        
        # Prepare data for the table
        table_data = []
        for _, row in table_data_df.iterrows():
            # Create period string (6 months from prediction date)
            pred_date = pd.to_datetime(row['prediction_date'])
            end_date = pred_date + pd.DateOffset(months=6)
            period_str = f"{pred_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
            
            # Get values
            actual_return = row['actual_excess_return']
            predicted_return = row['predicted_excess_return']
            sign_correct = "✓" if row['prediction_correct'] else "✗"
            trading_decision = row['investment_decision']
            return_achieved = row['switching_strategy_return']
            
            table_data.append([
                period_str,
                actual_return,
                predicted_return,
                sign_correct,
                trading_decision,
                return_achieved
            ])
        
        # Calculate summary statistics
        total_predictions = len(table_data)
        correct_predictions = sum(1 for row in table_data if row[3] == "✓")
        accuracy_percentage = (correct_predictions / total_predictions) * 100
        
        # Create figure with proper size for the table
        fig, ax = plt.subplots(figsize=(16, max(12, len(table_data) * 0.5 + 4)))
        ax.axis('off')
        
        # Table title
        title = "Table 1: Recursive Predictions and Actual Excess Returns\nFTSE All-Share Index - Rolling Window Analysis (2015-2025)"
        ax.text(0.5, 0.98, title, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', ha='center', va='top')
        
        # Column headers
        headers = ["Period", "Actual\nReturn", "Predicted\nReturn", "Sign\nCorrect?", "Trading\nDecision", "Return\nAchieved"]
        
        # Prepare formatted data for table
        formatted_data = []
        for row in table_data:
            formatted_row = [
                row[0],  # Period
                f"{row[1]:.3f}",  # Actual return
                f"{row[2]:.3f}",  # Predicted return
                row[3],  # Sign correct
                row[4],  # Trading decision
                f"{row[5]:.3f}"   # Return achieved
            ]
            formatted_data.append(formatted_row)
        
        # Calculate table position based on number of rows
        table_height = min(0.75, len(formatted_data) * 0.03 + 0.15)
        table_y = 0.95 - table_height
        
        # Create the main table
        table = ax.table(cellText=formatted_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0.05, table_y, 0.9, table_height])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(min(10, max(7, 120 // len(formatted_data))))
        table.scale(1, 1.5)
        
        # Style headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(formatted_data) + 1):
            for j in range(len(headers)):
                # Alternate row colors
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                # Highlight correct/incorrect predictions
                if j == 3:  # Sign Correct column
                    if formatted_data[i-1][j] == "✓":
                        table[(i, j)].set_facecolor('#D5E8D4')
                        table[(i, j)].set_text_props(color='green', weight='bold')
                    else:
                        table[(i, j)].set_facecolor('#F8CECC')
                        table[(i, j)].set_text_props(color='red', weight='bold')
                
                # Style trading decision column
                if j == 4:  # Trading Decision column
                    table[(i, j)].set_text_props(weight='bold')
                
                # Add borders
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(0.5)
        
        # Add separator line and summary
        summary_y = table_y - 0.05
        separator_line = "─" * 80
        ax.text(0.5, summary_y, separator_line, transform=ax.transAxes,
                fontsize=12, ha='center', va='center', family='monospace')
        
        # Add summary statistics
        summary_text = f"Total correct: {correct_predictions} out of {total_predictions} ({accuracy_percentage:.1f}%)"
        ax.text(0.5, summary_y - 0.03, summary_text, transform=ax.transAxes,
                fontsize=12, ha='center', va='center', weight='bold')
        
        ax.text(0.5, summary_y - 0.06, separator_line, transform=ax.transAxes,
                fontsize=12, ha='center', va='center', family='monospace')
        
        # Add methodology note
        methodology_note = ("Note: Predictions based on dividend yield model with rolling 6-month re-estimation.\n"
                           "Trading decision: 'Stocks' if predicted excess return > 0, otherwise 'Bonds'.")
        
        ax.text(0.5, 0.02, methodology_note, transform=ax.transAxes,
                fontsize=9, ha='center', va='bottom', style='italic')
        
        plt.tight_layout()
        
        # Save the table
        output_dir = BASE_PATH / "Data" / "Output"
        table_file = output_dir / "Recursive_Predictions_Table.png"
        plt.savefig(table_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.3)
        print(f"   💾 Recursive predictions table saved to: {table_file.name}")
        
        # Print summary statistics
        print(f"   📈 {correct_predictions}/{total_predictions} correct predictions ({accuracy_percentage:.1f}%)")
        
        plt.show()
    
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
        plt.show()
    
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
                
                # Generate recursive predictions table
                self.create_recursive_predictions_table()
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
        print(f"   • Strategy_Evaluation_Results.xlsx - Detailed evaluation results")
        print(f"   • Rolling_Window_Coefficients_and_Predictions.xlsx - Model coefficients")
        
        # Final verdict
        return_advantage = metrics['switching_annual_return'] - metrics['buyhold_annual_return']
        sharpe_advantage = metrics['switching_sharpe'] - metrics['buyhold_sharpe']
        
        print(f"\n" + "="*80)
        print(f"🏆 FINAL VERDICT:")
        if return_advantage > 0 and sharpe_advantage > 0:
            print(f"   ✅ STRATEGY SUCCESS: The dividend yield switching strategy outperformed")
            print(f"      buy-and-hold in both returns ({return_advantage*100:.1f}%) and risk-adjusted returns ({sharpe_advantage:.3f})")
        elif return_advantage > 0:
            print(f"   📈 MIXED RESULTS: Higher returns ({return_advantage*100:.1f}%) but lower risk-adjusted performance")
        elif sharpe_advantage > 0:
            print(f"   📊 RISK-ADJUSTED SUCCESS: Better Sharpe ratio despite lower returns")
        else:
            print(f"   ❌ STRATEGY UNDERPERFORMANCE: Buy-and-hold outperformed in both metrics")
            
        print(f"   📊 Prediction accuracy: {metrics['accuracy']*100:.1f}% - {'Good' if metrics['accuracy'] > 0.6 else 'Moderate' if metrics['accuracy'] > 0.5 else 'Poor'}")
        print(f"="*80)

def main():
    analyzer = PesaranTimmermannDividendYield()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()