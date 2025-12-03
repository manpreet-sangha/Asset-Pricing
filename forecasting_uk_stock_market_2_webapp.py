"""
Pesaran & Timmermann Multi-Predictor Model - Webapp Version
============================================================
This is a modified version of forecasting_uk_stock_market_2.py designed to work
with Streamlit webapp. It accepts data directly instead of reading from files.

Original file: forecasting_uk_stock_market_2.py (unchanged)
"""

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for webapp
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
from datetime import datetime, timedelta
import io

# Import specific statsmodels components
try:
    import statsmodels.api as sm
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
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# PNG optimization settings
SAVE_KWARGS = {
    'dpi': 200,
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'edgecolor': 'none',
    'pad_inches': 0.05,
}


class PesaranTimmermannMultiPredictor:
    """
    Pesaran & Timmermann (1995) replication using UK data with multiple predictors
    Webapp version - accepts data directly instead of reading from files
    """
    
    def __init__(self):
        # TRAINING PHASE: Apr 1997 → Sep 2015
        self.training_start = '1997-04-30'
        self.training_end = '2015-09-30'
        
        # PREDICTION PHASE: Oct 2015 → Oct 2025
        self.prediction_start = '2015-10-31'
        self.prediction_end = '2025-04-30'
        
        self.data = None
        self.training_data = None
        self.prediction_data = None
        self.model = None
        self.results = {}
        self.rolling_predictions = None
        self.performance_metrics = None
        
    def set_input_data(self, index_data, interbank_data, cpi_data):
        """Set input data directly from DataFrames (for webapp use)"""
        print("📊 Setting input data from DataFrames...")
        
        # Make copies to avoid modifying originals
        index_df = index_data.copy()
        interbank_df = interbank_data.copy()
        cpi_df = cpi_data.copy()
        
        # Rename columns in index data
        if 'astrx_index' in index_df.columns:
            index_df = index_df.rename(columns={'astrx_index': 'ftse_price'})
        if 'astrx_div_yield' in index_df.columns:
            index_df = index_df.rename(columns={'astrx_div_yield': 'dividend_yield'})
        
        # Find the date column
        date_cols = [col for col in index_df.columns if 'date' in col.lower()]
        if date_cols:
            index_df = index_df.rename(columns={date_cols[0]: 'date'})
        
        date_cols = [col for col in interbank_df.columns if 'date' in col.lower()]
        if date_cols:
            interbank_df = interbank_df.rename(columns={date_cols[0]: 'date'})
            
        date_cols = [col for col in cpi_df.columns if 'date' in col.lower()]
        if date_cols:
            cpi_df = cpi_df.rename(columns={date_cols[0]: 'date'})
        
        # Ensure InterBank data has correct column names
        if 'interbank_rate' not in interbank_df.columns:
            rate_columns = [col for col in interbank_df.columns if 'rate' in col.lower() or 'yield' in col.lower()]
            if rate_columns:
                interbank_df = interbank_df.rename(columns={rate_columns[0]: 'interbank_rate'})
            elif len(interbank_df.columns) >= 2:
                non_date_cols = [c for c in interbank_df.columns if c != 'date']
                if non_date_cols:
                    interbank_df = interbank_df.rename(columns={non_date_cols[0]: 'interbank_rate'})
        
        # Ensure CPI data has correct column names
        if 'cpi_yoy' not in cpi_df.columns:
            cpi_columns = [col for col in cpi_df.columns if 'cpi' in col.lower() or 'inflation' in col.lower() or 'ukrpcjyr' in col.lower()]
            if cpi_columns:
                cpi_df = cpi_df.rename(columns={cpi_columns[0]: 'cpi_yoy'})
            elif len(cpi_df.columns) >= 2:
                non_date_cols = [c for c in cpi_df.columns if c != 'date']
                if non_date_cols:
                    cpi_df = cpi_df.rename(columns={non_date_cols[0]: 'cpi_yoy'})
        
        # Convert dates to datetime
        index_df['date'] = pd.to_datetime(index_df['date'])
        interbank_df['date'] = pd.to_datetime(interbank_df['date'])
        cpi_df['date'] = pd.to_datetime(cpi_df['date'])
        
        # Create year-month columns for matching
        index_df['year_month'] = index_df['date'].dt.to_period('M')
        interbank_df['year_month'] = interbank_df['date'].dt.to_period('M')
        cpi_df['year_month'] = cpi_df['date'].dt.to_period('M')
        
        print(f"   Index date range: {index_df['year_month'].min()} to {index_df['year_month'].max()}")
        print(f"   InterBank date range: {interbank_df['year_month'].min()} to {interbank_df['year_month'].max()}")
        print(f"   CPI date range: {cpi_df['year_month'].min()} to {cpi_df['year_month'].max()}")
        
        # First merge: Index + InterBank
        temp_merged = pd.merge(
            index_df, 
            interbank_df[['year_month', 'interbank_rate']],
            on='year_month',
            how='inner'
        )
        
        # Second merge: Add CPI data
        self.data = pd.merge(
            temp_merged,
            cpi_df[['year_month', 'cpi_yoy']],
            on='year_month',
            how='inner'
        )
        
        # Remove the temporary year_month column
        self.data = self.data.drop('year_month', axis=1)
        
        # Rename for consistency
        self.data = self.data.rename(columns={
            'interbank_rate': 'm_interbank_rate',
            'date': 'Date'
        })
        
        # Sort by date
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Multi-predictor dataset: {len(self.data)} observations")
        print(f"   📅 Period: {self.data['Date'].min().strftime('%Y-%m')} to {self.data['Date'].max().strftime('%Y-%m')}")
        
        return True

    def compute_returns(self):
        """Prepare data structure for rolling window"""
        print("📈 Preparing data for rolling window predictions...")
        
        df = self.data.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Data prepared: {len(df)} observations")
        print(f"   📅 Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        self.full_data = df
        self.data = df

    def prepare_initial_training_data(self):
        """Prepare initial training data"""
        print("📚 Preparing initial training data...")
        
        data_source = self.full_data if hasattr(self, 'full_data') else self.data
        
        training_start = pd.to_datetime(self.training_start)
        training_end = pd.to_datetime(self.training_end)
        prediction_start = pd.to_datetime(self.prediction_start)
        prediction_end = pd.to_datetime(self.prediction_end)
        
        # Training period
        training_mask = (self.data['Date'] >= training_start) & (self.data['Date'] <= training_end)
        self.training_data = self.data[training_mask].copy().reset_index(drop=True)
        
        print(f"   Training data: {len(self.training_data)} observations")
        
        # Prediction period
        max_available_date = data_source['Date'].max()
        if prediction_end > max_available_date:
            prediction_end = max_available_date
        
        prediction_mask = (data_source['Date'] >= prediction_start) & (data_source['Date'] <= prediction_end)
        self.prediction_data = data_source[prediction_mask].copy().reset_index(drop=True)
        
        print(f"   Prediction data: {len(self.prediction_data)} observations")

    def train_model(self):
        """Train the initial model"""
        print("\n🎓 TRAINING PHASE - MULTI-PREDICTOR MODEL")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("   ❌ Statsmodels not available")
            return
        
        if len(self.training_data) == 0:
            print("   ❌ No training data available")
            return
        
        # Compute 6-month returns for the training period
        training_data_copy = self.training_data.copy()
        training_data_copy['ftse_6m_return'] = np.nan
        training_data_copy['m_interbank_6m_return'] = np.nan
        training_data_copy['excess_return'] = np.nan
        
        for idx in range(len(training_data_copy) - 6):
            current_date = training_data_copy.iloc[idx]['Date']
            future_date = current_date + pd.DateOffset(months=6)
            
            future_mask = (self.full_data['Date'] >= future_date - pd.DateOffset(days=15)) & \
                         (self.full_data['Date'] <= future_date + pd.DateOffset(days=15))
            future_data = self.full_data[future_mask]
            
            if len(future_data) > 0:
                future_price = future_data.iloc[0]['ftse_price']
                current_price = training_data_copy.iloc[idx]['ftse_price']
                
                stock_return = (future_price / current_price) - 1
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('ftse_6m_return')] = stock_return
                
                annual_rate = training_data_copy.iloc[idx]['m_interbank_rate'] / 100
                interbank_return = (1 + annual_rate) ** 0.5 - 1
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('m_interbank_6m_return')] = interbank_return
                
                excess_return = stock_return - interbank_return
                training_data_copy.iloc[idx, training_data_copy.columns.get_loc('excess_return')] = excess_return
        
        valid_data = training_data_copy.dropna(subset=['excess_return']).copy()
        print(f"   ✅ Computed excess returns: {len(valid_data)} valid observations")
        
        # Prepare training data for multi-predictor model
        y_train = valid_data['excess_return'].values
        X1_train = valid_data['dividend_yield'].values
        X2_train = valid_data['cpi_yoy'].values
        
        X_train = np.column_stack([X1_train, X2_train])
        X_train_with_const = add_constant(X_train)
        
        self.model = OLS(y_train, X_train_with_const).fit()
        
        alpha_hat = self.model.params[0]
        beta1_hat = self.model.params[1]
        beta2_hat = self.model.params[2]
        r_squared = self.model.rsquared
        
        print(f"\n📈 ESTIMATED COEFFICIENTS:")
        print(f"   α̂ = {alpha_hat:.6f}")
        print(f"   β̂₁ (dividend yield) = {beta1_hat:.6f}")
        print(f"   β̂₂ (CPI growth) = {beta2_hat:.6f}")
        print(f"   R² = {r_squared:.4f}")
        
        self.results['training'] = {
            'alpha_hat': alpha_hat,
            'beta1_hat': beta1_hat,
            'beta2_hat': beta2_hat,
            'r_squared': r_squared,
            'observations': len(self.training_data)
        }

    def train_rolling_model(self, training_data, iteration):
        """Train multi-predictor model for rolling window iteration"""
        try:
            y_train = training_data['excess_return'].values
            X1_train = training_data['dividend_yield'].values
            X2_train = training_data['cpi_yoy'].values
            
            X_train = np.column_stack([X1_train, X2_train])
            
            if STATSMODELS_AVAILABLE:
                X_train_with_const = add_constant(X_train)
                model = OLS(y_train, X_train_with_const).fit()
                alpha_hat = model.params[0]
                beta1_hat = model.params[1]
                beta2_hat = model.params[2]
                r_squared = model.rsquared
            else:
                X_matrix = np.column_stack([np.ones(len(X_train)), X_train])
                coefficients = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_train
                alpha_hat = coefficients[0]
                beta1_hat = coefficients[1]
                beta2_hat = coefficients[2]
                
                y_pred = X_matrix @ coefficients
                ss_res = np.sum((y_train - y_pred) ** 2)
                ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
            
            return alpha_hat, beta1_hat, beta2_hat, r_squared
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return None, None, None, None

    def rolling_window_predictions(self):
        """Implement rolling window prediction strategy"""
        print("\n🔮 ROLLING WINDOW PREDICTION PHASE")
        print("="*60)
        
        # Create investment decision dates
        investment_dates = []
        current_date = pd.Timestamp('2015-10-01')
        end_date = pd.Timestamp('2025-04-01')
        
        while current_date <= end_date:
            investment_dates.append(current_date)
            if current_date.month == 10:
                current_date = current_date.replace(year=current_date.year + 1, month=4)
            else:
                current_date = current_date.replace(month=10)
        
        print(f"📅 Investment decision schedule: {len(investment_dates)} periods")
        
        rolling_results = []
        
        for i, investment_date in enumerate(investment_dates):
            data_cutoff_date = (investment_date - pd.DateOffset(days=1)) + pd.offsets.MonthEnd(0)
            
            training_start_dt = pd.to_datetime(self.training_start)
            training_mask = (
                (self.full_data['Date'] >= training_start_dt) & 
                (self.full_data['Date'] <= data_cutoff_date)
            )
            current_training_data = self.full_data[training_mask].copy()
            
            # Compute returns
            current_training_data['ftse_6m_return'] = np.nan
            current_training_data['m_interbank_6m_return'] = np.nan
            current_training_data['excess_return'] = np.nan
            
            for idx in range(len(current_training_data) - 6):
                current_date_loop = current_training_data.iloc[idx]['Date']
                future_date = current_date_loop + pd.DateOffset(months=6)
                
                future_mask = (self.full_data['Date'] >= future_date - pd.DateOffset(days=15)) & \
                             (self.full_data['Date'] <= future_date + pd.DateOffset(days=15))
                future_data = self.full_data[future_mask]
                
                if len(future_data) > 0:
                    future_price = future_data.iloc[0]['ftse_price']
                    current_price = current_training_data.iloc[idx]['ftse_price']
                    
                    stock_return = (future_price / current_price) - 1
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('ftse_6m_return')] = stock_return
                    
                    annual_rate = current_training_data.iloc[idx]['m_interbank_rate'] / 100
                    interbank_return = (1 + annual_rate) ** 0.5 - 1
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('m_interbank_6m_return')] = interbank_return
                    
                    excess_return = stock_return - interbank_return
                    current_training_data.iloc[idx, current_training_data.columns.get_loc('excess_return')] = excess_return
            
            current_training_data = current_training_data.dropna(subset=['excess_return']).copy()
            
            if len(current_training_data) < 24:
                continue
            
            alpha_hat, beta1_hat, beta2_hat, r_squared = self.train_rolling_model(current_training_data, i+1)
            
            if alpha_hat is None:
                continue
            
            # Get predictors
            predictor_mask = self.full_data['Date'] == data_cutoff_date
            
            if not predictor_mask.any():
                closest_idx = (self.full_data['Date'] - data_cutoff_date).abs().idxmin()
                current_div_yield = self.full_data.loc[closest_idx, 'dividend_yield']
                current_cpi = self.full_data.loc[closest_idx, 'cpi_yoy']
            else:
                current_div_yield = self.full_data[predictor_mask]['dividend_yield'].iloc[0]
                current_cpi = self.full_data[predictor_mask]['cpi_yoy'].iloc[0]
            
            predicted_excess_return = alpha_hat + beta1_hat * current_div_yield + beta2_hat * current_cpi
            investment_decision = "FTSE All Share Index" if predicted_excess_return > 0 else "Interbank Rate"
            
            rolling_results.append({
                'prediction_date': investment_date.date(),
                'training_start': training_start_dt.date(),
                'training_end': data_cutoff_date.date(),
                'training_observations': len(current_training_data),
                'alpha_hat': alpha_hat,
                'beta1_hat': beta1_hat,
                'beta2_hat': beta2_hat,
                'r_squared': r_squared,
                'dividend_yield': current_div_yield,
                'cpi_growth': current_cpi,
                'predicted_excess_return': predicted_excess_return,
                'invest_in_stocks': 1 if predicted_excess_return > 0 else 0,
                'investment_decision': investment_decision
            })
        
        self.rolling_predictions = pd.DataFrame(rolling_results)
        
        for date_col in ['prediction_date', 'training_start', 'training_end']:
            if date_col in self.rolling_predictions.columns:
                self.rolling_predictions[date_col] = pd.to_datetime(self.rolling_predictions[date_col]).dt.date
        
        print(f"\n📊 ROLLING WINDOW SUMMARY:")
        print(f"   Total predictions: {len(self.rolling_predictions)}")
        print(f"   Stock investments: {self.rolling_predictions['invest_in_stocks'].sum()}")
        
        # Calculate actual returns
        self.calculate_actual_returns_for_predictions()
        
        return self.rolling_predictions

    def calculate_actual_returns_for_predictions(self):
        """Calculate actual returns for evaluation"""
        print("\n📊 CALCULATING ACTUAL RETURNS FOR EVALUATION")
        
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            return
        
        evaluation_results = []
        
        for idx, row in self.rolling_predictions.iterrows():
            pred_date = row['prediction_date']
            investment_decision = row['investment_decision']
            predicted_excess_return = row['predicted_excess_return']
            
            period_start_date = pred_date
            period_end_date = pd.to_datetime(pred_date) + pd.DateOffset(months=6) - pd.DateOffset(days=1)
            
            actual_ftse_return, actual_interbank_return = self.calculate_period_actual_returns(
                period_start_date, period_end_date
            )
            
            if actual_ftse_return is not None and actual_interbank_return is not None:
                actual_excess_return = actual_ftse_return - actual_interbank_return
                
                prediction_correct = (
                    (predicted_excess_return > 0 and actual_excess_return > 0) or
                    (predicted_excess_return <= 0 and actual_excess_return <= 0)
                )
                
                switching_return = actual_ftse_return if investment_decision == "FTSE All Share Index" else actual_interbank_return
                buyhold_return = actual_ftse_return
                
                evaluation_results.append({
                    'prediction_date': pred_date,
                    'period_end_date': period_end_date.date(),
                    'predicted_excess_return': predicted_excess_return,
                    'investment_decision': investment_decision,
                    'actual_ftse_return': actual_ftse_return,
                    'actual_interbank_return': actual_interbank_return,
                    'actual_excess_return': actual_excess_return,
                    'prediction_correct': prediction_correct,
                    'switching_strategy_return': switching_return,
                    'buyhold_strategy_return': buyhold_return,
                    'strategy_outperformance': switching_return - buyhold_return
                })
        
        if evaluation_results:
            evaluation_df = pd.DataFrame(evaluation_results)
            
            # Merge with rolling predictions
            self.rolling_predictions['temp_merge_key'] = pd.to_datetime(self.rolling_predictions['prediction_date']).dt.date.astype(str)
            evaluation_df['temp_merge_key'] = pd.to_datetime(evaluation_df['prediction_date']).dt.date.astype(str)
            
            self.rolling_predictions = pd.merge(
                self.rolling_predictions,
                evaluation_df[['temp_merge_key', 'actual_ftse_return', 'actual_interbank_return', 
                              'actual_excess_return', 'prediction_correct', 'switching_strategy_return',
                              'buyhold_strategy_return', 'strategy_outperformance']],
                on='temp_merge_key',
                how='left'
            )
            
            self.rolling_predictions = self.rolling_predictions.drop('temp_merge_key', axis=1)
            
            self.calculate_strategy_performance_metrics(evaluation_df)

    def calculate_period_actual_returns(self, start_date, end_date):
        """Calculate actual returns for a specific period"""
        data_source = self.full_data.copy()
        
        if hasattr(start_date, 'date'):
            start_date = start_date
        if hasattr(end_date, 'date'):
            end_date = end_date.date() if hasattr(end_date, 'date') else end_date
        
        data_source['date_only'] = data_source['Date'].dt.date
        
        start_mask = data_source['date_only'] <= start_date
        end_mask = data_source['date_only'] >= end_date
        
        if not start_mask.any() or not end_mask.any():
            return None, None
        
        exact_start_mask = data_source['date_only'] == start_date
        if exact_start_mask.any():
            start_idx = data_source[exact_start_mask].index[0]
        else:
            start_idx = data_source[start_mask].index[-1]
        
        end_indices = data_source[end_mask].index
        
        if len(end_indices) == 0 or end_indices[0] <= start_idx:
            return None, None
        
        end_idx = end_indices[0]
        
        if start_idx + 1 >= len(data_source):
            return None, None
            
        start_data = data_source.iloc[start_idx + 1]
        end_data = data_source.iloc[end_idx]
        
        ftse_start_price = start_data['ftse_price']
        ftse_end_price = end_data['ftse_price']
        actual_ftse_return = (ftse_end_price / ftse_start_price) - 1
        
        actual_start_date = start_data['Date'].replace(day=1).date()
        actual_end_date = end_data['Date'].date()
        
        period_mask = (
            (data_source['date_only'] >= actual_start_date) & 
            (data_source['date_only'] <= actual_end_date)
        )
        period_data = data_source[period_mask].sort_values('Date')
        
        if len(period_data) > 0:
            interbank_value = 1.0
            for _, row in period_data.iterrows():
                monthly_rate = row['m_interbank_rate'] / 100
                monthly_return = (1 + monthly_rate) ** (1/12) - 1
                interbank_value *= (1 + monthly_return)
            actual_interbank_return = interbank_value - 1
        else:
            start_interbank_rate = start_data['m_interbank_rate']
            period_length_years = (actual_end_date - actual_start_date).days / 365.25
            actual_interbank_return = (1 + start_interbank_rate/100)**period_length_years - 1
        
        return actual_ftse_return, actual_interbank_return

    def calculate_strategy_performance_metrics(self, evaluation_df):
        """Calculate performance metrics"""
        print(f"\n🏆 STRATEGY PERFORMANCE EVALUATION")
        
        total_predictions = len(evaluation_df)
        correct_predictions = evaluation_df['prediction_correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        evaluation_df = evaluation_df.copy()
        evaluation_df['switching_cumulative'] = (1 + evaluation_df['switching_strategy_return']).cumprod()
        evaluation_df['buyhold_cumulative'] = (1 + evaluation_df['buyhold_strategy_return']).cumprod()
        
        switching_total_return = evaluation_df['switching_cumulative'].iloc[-1] - 1
        buyhold_total_return = evaluation_df['buyhold_cumulative'].iloc[-1] - 1
        
        years = total_predictions * 0.5
        switching_annual_return = (evaluation_df['switching_cumulative'].iloc[-1] ** (1/years)) - 1 if years > 0 else 0
        buyhold_annual_return = (evaluation_df['buyhold_cumulative'].iloc[-1] ** (1/years)) - 1 if years > 0 else 0
        
        switching_volatility = evaluation_df['switching_strategy_return'].std() * np.sqrt(2)
        buyhold_volatility = evaluation_df['buyhold_strategy_return'].std() * np.sqrt(2)
        
        avg_rf_rate = evaluation_df['actual_interbank_return'].mean() * 2
        switching_sharpe = (switching_annual_return - avg_rf_rate) / switching_volatility if switching_volatility > 0 else 0
        buyhold_sharpe = (buyhold_annual_return - avg_rf_rate) / buyhold_volatility if buyhold_volatility > 0 else 0
        
        if 'strategy_outperformance' not in evaluation_df.columns:
            evaluation_df['strategy_outperformance'] = evaluation_df['switching_strategy_return'] - evaluation_df['buyhold_strategy_return']
        
        win_periods = (evaluation_df['strategy_outperformance'] > 0).sum()
        win_rate = win_periods / total_predictions if total_predictions > 0 else 0
        avg_outperformance = evaluation_df['strategy_outperformance'].mean()
        
        print(f"   Accuracy: {accuracy*100:.1f}%")
        print(f"   Switching Total Return: {switching_total_return*100:.1f}%")
        print(f"   Buy-Hold Total Return: {buyhold_total_return*100:.1f}%")
        
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
        
        self.evaluation_data = evaluation_df

    # ==================== WEBAPP PLOTTING METHODS ====================
    # These methods return matplotlib figures instead of saving to files
    
    def plot_predicted_vs_actual_returns_webapp(self):
        """Create time-series chart - returns figure for webapp display"""
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            return None
        
        plot_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(plot_data) == 0:
            return None
        
        plot_data['prediction_date_dt'] = pd.to_datetime(plot_data['prediction_date'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(plot_data['prediction_date_dt'], plot_data['predicted_excess_return']*100, 
                marker='o', linewidth=2, markersize=6, label='Predicted Excess Return', color='blue', alpha=0.8)
        
        ax.plot(plot_data['prediction_date_dt'], plot_data['actual_excess_return']*100, 
                marker='s', linewidth=2, markersize=6, label='Actual Excess Return', color='red', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title('Pesaran & Timmermann Model: Predicted vs Actual Excess Returns\n(Rolling Window Predictions 2015-2025)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Date', fontsize=12)
        ax.set_ylabel('Excess Return (%)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        
        correct_predictions = plot_data['prediction_correct'].sum()
        total_predictions = len(plot_data)
        accuracy = correct_predictions / total_predictions * 100
        
        textstr = f'Prediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)\n'
        textstr += f'Avg Predicted: {plot_data["predicted_excess_return"].mean()*100:.2f}%\n'
        textstr += f'Avg Actual: {plot_data["actual_excess_return"].mean()*100:.2f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        return fig

    def plot_cumulative_returns_chart_webapp(self):
        """Create cumulative returns chart - returns figure for webapp display"""
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            return None
        
        plot_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(plot_data) == 0:
            return None
        
        plot_data['prediction_date_dt'] = pd.to_datetime(plot_data['prediction_date'])
        
        plot_data['switching_cumulative'] = (1 + plot_data['switching_strategy_return']).cumprod()
        plot_data['buyhold_cumulative'] = (1 + plot_data['buyhold_strategy_return']).cumprod()
        plot_data['interbank_cumulative'] = (1 + plot_data['actual_interbank_return']).cumprod()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(plot_data['prediction_date_dt'], plot_data['switching_cumulative'], 
                linewidth=3, label='Switching Strategy (Multi-Predictor)', color='blue', alpha=0.8)
        
        ax.plot(plot_data['prediction_date_dt'], plot_data['buyhold_cumulative'], 
                linewidth=3, label='Buy-and-Hold (FTSE All Share)', color='red', alpha=0.8)
        
        ax.plot(plot_data['prediction_date_dt'], plot_data['interbank_cumulative'], 
                linewidth=3, label='InterBank Rate (Risk-Free)', color='green', alpha=0.8)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title('Cumulative Returns: Multi-Predictor Switching Strategy vs Benchmarks\n(Dividend Yield + CPI Model)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (1 = No Change)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        
        final_switching = plot_data['switching_cumulative'].iloc[-1]
        final_buyhold = plot_data['buyhold_cumulative'].iloc[-1]
        final_interbank = plot_data['interbank_cumulative'].iloc[-1]
        
        switching_return = (final_switching - 1) * 100
        buyhold_return = (final_buyhold - 1) * 100
        interbank_return = (final_interbank - 1) * 100
        
        textstr = f'Final Returns:\n'
        textstr += f'Switching: {switching_return:.1f}%\n'
        textstr += f'Buy-Hold: {buyhold_return:.1f}%\n'
        textstr += f'InterBank: {interbank_return:.1f}%'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        return fig

    def create_pt_test_table_webapp(self):
        """Create PT Sign Test table - returns figure for webapp display"""
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            return None
        
        test_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(test_data) == 0:
            return None
        
        total_predictions = len(test_data)
        correct_predictions = test_data['prediction_correct'].sum()
        proportion_correct = correct_predictions / total_predictions
        expected_proportion = 0.5
        
        invest_decisions = test_data['invest_in_stocks'].values
        num_switches = sum(1 for i in range(1, len(invest_decisions)) if invest_decisions[i] != invest_decisions[i-1])
        
        pstar = 0.5
        pt_statistic = (proportion_correct - pstar) / np.sqrt(pstar * (1 - pstar) / total_predictions)
        
        from scipy import stats as scipy_stats
        p_value_one_sided = 1 - scipy_stats.norm.cdf(pt_statistic)
        
        significant_5pct = p_value_one_sided < 0.05
        significant_1pct = p_value_one_sided < 0.01
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        
        title = "Pesaran-Timmermann Sign Test Statistics\nEvaluation Period: 2015-2025"
        
        table_data = [
            ["Number of predictions", f"{total_predictions}"],
            ["Correct sign predictions", f"{correct_predictions}"],
            ["Proportion correct (%)", f"{proportion_correct*100:.1f}%"],
            ["Expected proportion H₀", f"{expected_proportion*100:.1f}%"],
            ["PT test statistic", f"{pt_statistic:.3f}"],
            ["p-value (one-sided)", f"{p_value_one_sided:.3f}"],
            ["Significant at 5%?", "Yes" if significant_5pct else "No"],
            ["Number of switches", f"{num_switches}"]
        ]
        
        table = ax.table(cellText=table_data,
                        colWidths=[0.65, 0.20],
                        cellLoc='left',
                        loc='center',
                        bbox=[0.08, 0.08, 0.84, 0.72])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        
        for i in range(len(table_data)):
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
            
            if i == 6 and significant_5pct:
                table[(i, 1)].set_facecolor('#ffffcc')
                table[(i, 1)].set_text_props(weight='bold', color='green')
        
        ax.text(0.5, 0.92, title, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='center', va='top')
        
        plt.tight_layout()
        
        return fig

    def create_pt_predictions_table_webapp(self):
        """Create predictions vs actual table - returns figure for webapp display"""
        if not hasattr(self, 'rolling_predictions') or len(self.rolling_predictions) == 0:
            return None
        
        table_data = self.rolling_predictions.dropna(subset=['actual_excess_return']).copy()
        
        if len(table_data) == 0:
            return None
        
        num_rows = len(table_data)
        fig, ax = plt.subplots(figsize=(8, max(4, num_rows * 0.22 + 1.5)))
        ax.axis('off')
        
        title = "Predictions vs Actual Results\nMulti-Predictor Strategy"
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)
        
        table_rows = []
        headers = ['#', 'Date', 'Pred (%)', 'Actual (%)', 'Decision', '✓/✗']
        
        for idx, row in table_data.iterrows():
            period = len(table_rows) + 1
            date = pd.to_datetime(row['prediction_date']).strftime('%Y-%m')
            pred_return = f"{row['predicted_excess_return']*100:.1f}"
            actual_return = f"{row['actual_excess_return']*100:.1f}"
            investment = "FTSE" if "FTSE" in row['investment_decision'] else "IB"
            pred_correct = "✓" if row['prediction_correct'] else "✗"
            
            table_rows.append([str(period), date, pred_return, actual_return, investment, pred_correct])
        
        table = ax.table(cellText=table_rows,
                        colLabels=headers,
                        colWidths=[0.06, 0.14, 0.12, 0.12, 0.10, 0.06],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.15, 0.08, 0.70, 0.82])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_rows) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                if j == 5:
                    if table_rows[i-1][j] == "✓":
                        table[(i, j)].set_facecolor('#D4F6D4')
                    else:
                        table[(i, j)].set_facecolor('#F6D4D4')
        
        correct_predictions = table_data['prediction_correct'].sum()
        total_predictions = len(table_data)
        accuracy = correct_predictions / total_predictions * 100
        
        summary_text = f"{correct_predictions}/{total_predictions} correct ({accuracy:.1f}%)"
        fig.text(0.5, 0.03, summary_text, ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        return fig
