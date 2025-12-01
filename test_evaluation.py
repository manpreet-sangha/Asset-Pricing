from forecasting_uk_stock_market_1 import PesaranTimmermannDividendYield
import pandas as pd

# Quick test of the evaluation function
analyzer = PesaranTimmermannDividendYield()
analyzer.load_and_prepare_data()
analyzer.compute_returns()

# Test calculate_period_actual_returns function
print('Testing actual returns calculation...')
start_date = pd.to_datetime('2015-10-31').date()
end_date = pd.to_datetime('2016-04-30').date()
print(f'Testing period: {start_date} to {end_date}')
print(f'Full data range: {analyzer.full_data["Date"].min().date()} to {analyzer.full_data["Date"].max().date()}')

# Check if the period is within data range
actual_ftse_return, actual_interbank_return = analyzer.calculate_period_actual_returns(start_date, end_date)
if actual_ftse_return is not None:
    print(f'SUCCESS: FTSE return = {actual_ftse_return*100:.2f}%, InterBank return = {actual_interbank_return*100:.2f}%')
else:
    print('FAILED: No actual returns calculated')
