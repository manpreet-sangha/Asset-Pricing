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

class PesaranTimmermannDividendYield:
    """
    Pesaran & Timmermann (1995) replication using UK data
    Predictability model using Dividend Yield to predict stock market excess returns
    
    METHODOLOGY (Rolling Window Approach):
    1. Initial Training: Apr 1997 - Sep 2015 (estimate α̂₁ and β̂₁)
    2. Rolling Predictions: Re-estimate coefficients every 6 months
       - Oct 2015: Train on Apr 1997→Sep 2015 → Get α̂₁, β̂₁ → Predict Oct 2015-Mar 2016
       - Apr 2016: Train on Apr 1997→Mar 2016 → Get α̂₂, β̂₂ → Predict Apr 2016-Sep 2016
       - Oct 2016: Train on Apr 1997→Sep 2016 → Get α̂₃, β̂₃ → Predict Oct 2016-Mar 2017
       - ... and so on
    3. Investment Strategy: Switch between stocks and T-bills based on predicted excess returns
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
        """Load merged data file with InterBank rate"""
        print("PESARAN & TIMMERMANN DIVIDEND YIELD PREDICTABILITY MODEL")
        print("="*70)
        print("📊 Loading merged data...")
        
        # Load the FTSE Index and Dividend Yield data
        index_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265_Index_DivYield_Monthly.xlsx"
        
        # Load the InterBank rate data
        interbank_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265_UK_InterBank_Rate_Monthly.xlsx"
        
        try:
            # Load the index and dividend yield data
            index_data = pd.read_excel(index_file)
            
            # Load the InterBank rate data
            interbank_data = pd.read_excel(interbank_file)
            
            # Debug: Show what we loaded
            print(f"   🔍 Index data shape: {index_data.shape}")
            print(f"   🔍 Index columns: {list(index_data.columns)}")
            print(f"   🔍 InterBank data shape: {interbank_data.shape}")
            print(f"   🔍 InterBank columns: {list(interbank_data.columns)}")
            
            # Show first few rows of both datasets
            print(f"\n📋 INDEX DATA SAMPLE:")
            print(index_data.head(3))
            print(f"\n📋 INTERBANK DATA SAMPLE:")
            print(interbank_data.head(3))
            
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
        """Compute 6-month returns and excess returns using InterBank rate for training data"""
        print("📈 Computing 6-month returns and excess returns...")
        print("   🎯 Computing returns for ALL available data periods")
        print("   📅 Rolling window will use different training periods for each prediction:")
        print("      • 1st prediction (Oct 2015): Training Apr 1997 - Sep 2015")
        print("      • 2nd prediction (Apr 2016): Training Apr 1997 - Mar 2016") 
        print("      • 3rd prediction (Oct 2016): Training Apr 1997 - Sep 2016")
        print("      • ... and so on until last prediction")
        
        df = self.data.copy()
          
        # Compute 6-month stock returns: R_stock = (FTSE_{t+6} / FTSE_t) - 1
        df['ftse_6m_return'] = df['ftse_price'].shift(-6) / df['ftse_price'] - 1
        
        # Compute 6-month InterBank returns: R_interbank = (1 + annual_yield/100)^0.5 - 1
        df['m_interbank_6m_return'] = (1 + df['m_interbank_rate']/100)**(0.5) - 1
        
        # Compute excess returns: Y = R_stock - R_interbank
        df['excess_return'] = df['ftse_6m_return'] - df['m_interbank_6m_return']
        
        # Remove rows where we can't compute 6-month ahead returns (last 6 months)
        df = df.dropna().reset_index(drop=True)
        
        print(f"   ✅ Computed returns for {len(df)} observations")
        print(f"   📊 Average 6m stock return: {df['ftse_6m_return'].mean()*100:.2f}%")
        print(f"   📊 Average 6m InterBank return: {df['m_interbank_6m_return'].mean()*100:.2f}%")
        print(f"   📊 Average excess return: {df['excess_return'].mean()*100:.2f}%")
        print(f"   📅 Return computation period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        self.data = df
    def split_training_prediction_periods(self):
        """Split data into training and prediction periods"""
        print("✂️ Splitting data into training and prediction periods...")
        
        # First, let's debug what data we have
        print(f"   📊 Available data range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   📊 Total observations: {len(self.data)}")
        
        # Convert date strings to datetime
        training_start = pd.to_datetime(self.training_start)
        training_end = pd.to_datetime(self.training_end)
        prediction_start = pd.to_datetime(self.prediction_start)
        prediction_end = pd.to_datetime(self.prediction_end)
        
        print(f"   🎯 Training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
        print(f"   🎯 Prediction period: {prediction_start.strftime('%Y-%m-%d')} to {prediction_end.strftime('%Y-%m-%d')}")
        
        # Training period: Apr 1997 → Sep 2015
        training_mask = (self.data['Date'] >= training_start) & (self.data['Date'] <= training_end)
        self.training_data = self.data[training_mask].copy().reset_index(drop=True)
        
        # Prediction period: Oct 2015 → Oct 2025 (or until available data ends)
        max_available_date = self.data['Date'].max()
        if prediction_end > max_available_date:
            print(f"   ⚠️  Warning: Prediction end ({prediction_end.strftime('%Y-%m-%d')}) beyond available data ({max_available_date.strftime('%Y-%m-%d')})")
            print(f"   📅 Using available data until: {max_available_date.strftime('%Y-%m-%d')}")
            prediction_end = max_available_date
        
        prediction_mask = (self.data['Date'] >= prediction_start) & (self.data['Date'] <= prediction_end)
        self.prediction_data = self.data[prediction_mask].copy().reset_index(drop=True)
        
        print(f"   📚 Training period: {len(self.training_data)} observations")
        if len(self.training_data) > 0:
            print(f"      {self.training_data['Date'].min().strftime('%Y-%m-%d')} to {self.training_data['Date'].max().strftime('%Y-%m-%d')}")
        else:
            print("      ❌ No training data found!")
        
        print(f"   🔮 Prediction period: {len(self.prediction_data)} observations")
        if len(self.prediction_data) > 0:
            print(f"      {self.prediction_data['Date'].min().strftime('%Y-%m-%d')} to {self.prediction_data['Date'].max().strftime('%Y-%m-%d')}")
        else:
            print("      ❌ No prediction data found!")
            
        print(f"\n💡 METHODOLOGY:")
        print(f"   1. Train model on {len(self.training_data)} observations (Apr 1997 - Sep 2015)")
        print(f"   2. Use trained model to predict excess returns for {len(self.prediction_data)} periods")
        print(f"   3. Make investment decisions based on predicted excess returns")
        print(f"   4. Evaluate switching strategy vs buy-and-hold")

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
            
            # Define training end as 6 months before prediction date
            training_end_date = pred_date - pd.DateOffset(months=6)
            print(f"   📅 Training window: Apr 1997 → {training_end_date.strftime('%Y-%m-%d')}")
            
            # Create training data (Apr 1997 → training_end_date)
            training_start_dt = pd.to_datetime(self.training_start)
            training_mask = (
                (self.data['Date'] >= training_start_dt) & 
                (self.data['Date'] <= training_end_date)
            )
            current_training_data = self.data[training_mask].copy()
            
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
            div_yield_mask = self.data['Date'] == div_yield_date
            
            if not div_yield_mask.any():
                # Find closest date
                closest_idx = (self.data['Date'] - div_yield_date).abs().idxmin()
                current_div_yield = self.data.loc[closest_idx, 'dividend_yield']
                actual_date = self.data.loc[closest_idx, 'Date']
                print(f"   📊 Using dividend yield from {actual_date.strftime('%Y-%m-%d')}: {current_div_yield:.2f}%")
            else:
                current_div_yield = self.data[div_yield_mask]['dividend_yield'].iloc[0]
                print(f"   📊 Dividend yield on {div_yield_date.strftime('%Y-%m-%d')}: {current_div_yield:.2f}%")
            
            # Make prediction
            predicted_excess_return = alpha_hat + beta_hat * current_div_yield
            investment_decision = "STOCKS" if predicted_excess_return > 0 else "INTERBANK"
            
            print(f"   🎯 Prediction: Ŷ = {alpha_hat:.6f} + {beta_hat:.6f} × {current_div_yield:.2f} = {predicted_excess_return:.4f}")
            print(f"   💼 Investment Decision: {investment_decision}")
            
            # Store results
            rolling_results.append({
                'prediction_date': pred_date,
                'training_start': training_start_dt,
                'training_end': training_end_date,
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
        
        # Save rolling predictions with coefficients to file
        if len(rolling_results) > 0:
            output_dir = BASE_PATH / "Data" / "Output"
            output_dir.mkdir(parents=True, exist_ok=True)
            coefficients_file = output_dir / "Rolling_Window_Coefficients_and_Predictions.xlsx"
            
            with pd.ExcelWriter(coefficients_file, engine='openpyxl') as writer:
                self.rolling_predictions.to_excel(writer, sheet_name='Rolling_Coefficients', index=False)
            
            print(f"\n💾 Rolling window coefficients saved to: {coefficients_file.name}")
        
        print(f"\n📊 ROLLING WINDOW SUMMARY:")
        print(f"   Total predictions: {len(self.rolling_predictions)}")
        print(f"   Stock investments: {self.rolling_predictions['invest_in_stocks'].sum()}")
        print(f"   InterBank investments: {len(self.rolling_predictions) - self.rolling_predictions['invest_in_stocks'].sum()}")
        print(f"   Average α̂: {self.rolling_predictions['alpha_hat'].mean():.6f}")
        print(f"   Average β̂: {self.rolling_predictions['beta_hat'].mean():.6f}")
        print(f"   α̂ range: [{self.rolling_predictions['alpha_hat'].min():.6f}, {self.rolling_predictions['alpha_hat'].max():.6f}]")
        print(f"   β̂ range: [{self.rolling_predictions['beta_hat'].min():.6f}, {self.rolling_predictions['beta_hat'].max():.6f}]")
        
        return self.rolling_predictions
    
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
        """Run the complete rolling window analysis"""
        try:
            if not self.load_and_prepare_data():
                return False
            
            self.compute_returns()
            self.split_training_prediction_periods()
            
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
            else:
                print(f"\n❌ No rolling predictions generated - check data availability")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    analyzer = PesaranTimmermannDividendYield()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()