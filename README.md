# Pesaran & Timmermann (1995) UK Stock Market Forecasting Model

A comprehensive replication and extension of the Pesaran & Timmermann (1995) methodology for predicting UK stock market excess returns using economic indicators.

## 📊 Project Overview

This project implements a rolling window prediction strategy to forecast UK stock market performance using dividend yield and other economic indicators. The model evaluates the predictive ability of these variables for stock market excess returns over 6-month periods from 2015-2025.

### Key Features

- **Rolling Window Methodology**: Re-estimates model coefficients every 6 months using expanding training windows
- **Statistical Validation**: Implements Pesaran-Timmermann sign test for statistical significance
- **Multiple Predictors**: Single-predictor (dividend yield) and multi-predictor (dividend yield + UK CPI) versions
- **Comprehensive Evaluation**: Complete strategy performance analysis with visualizations
- **Investment Strategy**: Switching strategy between FTSE All-Share and UK InterBank Rate

## 🎯 Results Summary

### Model Performance (2015-2025)
- **Prediction Accuracy**: 75% directional accuracy (15/20 correct predictions)
- **Statistical Significance**: Significant at 5% level (p-value: 0.013)
- **Strategy Performance**: Outperformed buy-and-hold benchmark
- **Total Predictions**: 20 rolling window predictions over 10-year period

### Key Findings
- Dividend yield demonstrates significant predictive power for UK stock market returns
- Rolling window approach successfully captures time-varying relationships
- Switching strategy provides superior risk-adjusted returns compared to buy-and-hold

## 🗂️ Project Structure

```
Asset-Pricing/
├── forecasting_uk_stock_market_1.py    # Complete single-predictor model
├── forecasting_uk_stock_market_2.py    # Multi-predictor model (with UK CPI)
├── Data/
│   ├── Cleaned Data/Monthly/            # Processed monthly data files
│   └── Output/                          # Generated results and visualizations
├── Visualization Scripts/
│   ├── create_pt_table.py              # Pesaran-Timmermann test table
│   ├── create_recursive_table.py       # Recursive predictions table
│   ├── cumulative_returns_chart.py     # Strategy performance chart
│   └── plot_chart.py                   # General plotting utilities
├── Data Processing Scripts/
│   ├── m_*.py                          # Monthly data processing scripts
│   ├── q_*.py                          # Quarterly data processing scripts
│   └── load_data_1.py                  # Data loading utilities
├── Analysis Notebooks/
│   ├── Pesaran_Timmermann_UK_Replication.ipynb
│   └── Steps.ipynb
└── requirements.txt                     # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- openpyxl

### Running the Analysis

1. **Single-Predictor Model (Dividend Yield)**:
```python
from forecasting_uk_stock_market_1 import PesaranTimmermannDividendYield

# Initialize and run complete analysis
model = PesaranTimmermannDividendYield()
model.load_and_prepare_data()
model.compute_returns()
model.prepare_initial_training_data()
model.train_model()
predictions = model.rolling_window_predictions()

# Generate visualizations
model.plot_predicted_vs_actual_returns()
model.plot_cumulative_returns_chart()
model.create_pt_test_table()
model.create_recursive_predictions_table()
```

2. **Multi-Predictor Model (Dividend Yield + UK CPI)**:
```python
from forecasting_uk_stock_market_2 import PesaranTimmermannMultiPredictor

# Initialize and run multi-predictor analysis
model = PesaranTimmermannMultiPredictor()
model.load_and_prepare_data()
# ... (similar workflow with multiple predictors)
```

## 📈 Methodology

### Rolling Window Approach

The model uses an expanding window methodology:

1. **Initial Training**: April 1997 - September 2015 (estimate α̂₁ and β̂₁)
2. **Rolling Predictions**: Re-estimate coefficients every 6 months
   - October 2015: Train on Apr 1997→Sep 2015 → Predict Oct 2015-Mar 2016
   - April 2016: Train on Apr 1997→Mar 2016 → Predict Apr 2016-Sep 2016
   - October 2016: Train on Apr 1997→Sep 2016 → Predict Oct 2016-Mar 2017
   - ... continuing for 20 prediction periods

### Model Specification

**Single-Predictor Model:**
```
Excess_Return = α + β × Dividend_Yield + ε
```

**Multi-Predictor Model:**
```
Excess_Return = α + β₁ × Dividend_Yield + β₂ × CPI_Growth + ε
```

### Investment Strategy

- **If predicted excess return > 0**: Invest in FTSE All-Share
- **If predicted excess return ≤ 0**: Invest in UK InterBank Rate

## 📊 Generated Outputs

### Data Files
- `Rolling_Window_Coefficients_and_Predictions.xlsx` - Complete coefficient estimates and predictions
- `Strategy_Evaluation_Results.xlsx` - Comprehensive performance analysis
- `SMM265_Index_DivYield_InterBank_Monthly_Merged.xlsx` - Merged dataset

### Visualizations
- `Predicted_vs_Actual_Returns_TimeSeries.png` - Time-series comparison
- `Cumulative_Returns_Strategy_Comparison.png` - Strategy performance chart
- `Pesaran_Timmermann_Sign_Test_Table.png` - Statistical test results
- `Recursive_Predictions_Table.png` - Complete predictions summary

## 🔬 Statistical Testing

### Pesaran-Timmermann Sign Test
- **Null Hypothesis**: No predictive ability (50% accuracy expected)
- **Test Statistic**: 2.486
- **P-value**: 0.013 (one-sided)
- **Result**: Significant at 5% level, confirming predictive ability

## 📋 Data Sources

- **FTSE All-Share Index**: UK stock market performance
- **Dividend Yield**: FTSE All-Share dividend yield
- **UK InterBank Rate**: Risk-free rate proxy
- **UK CPI**: Consumer Price Index (multi-predictor model)
- **Data Period**: April 1997 - April 2025

## ⚙️ Technical Implementation

### Key Classes

1. **PesaranTimmermannDividendYield**: Single-predictor implementation
2. **PesaranTimmermannMultiPredictor**: Multi-predictor extension

### Core Methods

- `load_and_prepare_data()`: Data loading and preprocessing
- `compute_returns()`: Calculate 6-month returns and excess returns
- `rolling_window_predictions()`: Implement rolling prediction strategy
- `calculate_actual_returns_for_predictions()`: Evaluation framework
- `plot_*()` methods: Comprehensive visualization suite

## 📈 Performance Metrics

### Strategy Evaluation
- **Prediction Accuracy**: Directional accuracy of excess return predictions
- **Sharpe Ratio**: Risk-adjusted return comparison
- **Maximum Drawdown**: Downside risk analysis
- **Win Rate**: Percentage of periods outperforming benchmark
- **Volatility**: Strategy return variability

## 🔄 Future Extensions

- Additional economic predictors (unemployment, GDP growth, etc.)
- Machine learning approaches (Random Forest, Neural Networks)
- Different rebalancing frequencies (quarterly, annually)
- International market applications
- Sector-specific predictions

## 📚 References

Pesaran, M. H., & Timmermann, A. (1995). Predictability of stock returns: Robustness and economic significance. *Journal of Finance*, 50(4), 1201-1228.

## 👨‍💻 Author

**Manpreet Sangha**  
SMM265 Asset Pricing Coursework  
City St George's, University of London

## 📄 License

This project is for academic purposes as part of the SMM265 Asset Pricing course.

---

## 🚨 Important Notes

- Ensure all data files are properly formatted and located in the `Data/Cleaned Data/Monthly/` directory
- The model requires continuous monthly data from April 1997 to April 2025
- All results are for academic research purposes only and should not be used for actual investment decisions
- Performance results are based on historical data and do not guarantee future performance

## 🛠️ Troubleshooting

### Common Issues

1. **Data Loading Errors**: Verify file paths and Excel file formats
2. **Missing Dependencies**: Install all required packages from requirements.txt
3. **Date Format Issues**: Ensure consistent date formatting across all data files
4. **Memory Issues**: Large datasets may require increased memory allocation

### Support

For technical issues or questions about the implementation, please refer to the detailed documentation within each Python script or the Jupyter notebooks provided.
