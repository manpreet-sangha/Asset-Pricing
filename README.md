# SMM265 Asset Pricing - Group 7

Regression modelling to predict UK stock index movements and portfolio optimization analysis.

##  Live Demo

**[Launch Web Application](https://smm265-assetpricing.streamlit.app/)**

---

##  Project Overview

This project implements two key analyses from the SMM265 Asset Pricing module:

### Question 1: UK Stock Market Forecasting

A forecasting model adapted from **Pesaran, M.H. and Timmermann, A. (1994)** framework to predict 6-month forward excess returns on the FTSE All-Share Index.

#### Model Specification

`
ExcessReturn(t  t+6) = α + β DY(t-k) + β CPI(t-j) + ε(t+6)
`

Where:
- **DY**: Dividend Yield (12-month trailing) with lag k  {0,1,2,3}
- **CPI**: UK CPI YoY (UKRPCJYR) with lag j  {0,1,2}

#### Data Sources
| Variable | Description |
|----------|-------------|
| FTSE All-Share Index | Total Return Index (ASXTR) |
| Dividend Yield | 12-month trailing |
| UK CPI YoY | UKRPCJYR |
| Risk-Free Rate | UK 1-Month Interbank deposit |

#### Methodology
1. Expanding training window at each 6-month decision point
2. Sequential lag selection approach:
   - Step 1: Calculate optimal DY lag (0-3) based on |t-stat|
   - Step 2: Calculate optimal CPI lag (0-2) based on |t-stat|
   - Step 3: Use optimal lags for prediction

#### Key Results

| Metric | Model Switching | Passive Buy & Hold | Difference |
|--------|-----------------|-------------------|------------|
| Total Return (10yr) | 163.9% | 163.8% | +0.1% |
| Annualised Return | 10.2% | 10.2% | 0.0% |
| Volatility | 11.1% | 12.3% | **-1.2%** |
| Sharpe Ratio | 0.745 | 0.672 | **+0.073** |

| Coefficient | Mean | Range | Interpretation |
|-------------|------|-------|----------------|
| α (Intercept) | -0.056 | -0.059 to -0.051 | Baseline negative excess returns |
| β (Dividend Yield) | +0.029 | +0.024 to +0.034 | Higher DY  Higher returns |
| β (CPI) | -0.008 | -0.013 to -0.005 | Higher CPI  Lower returns |

#### Model Performance
- **90% accuracy** (18/20) for predicting direction of excess returns
- Correctly switched to interbank deposits during Apr-2022 to Oct-2023 (CPI reached 10.4%)
- 3-month lagged DY and current CPI remained consistent throughout

---

### Question 2: Mean-Variance Portfolio Optimization

Analysis of UK stocks using Modern Portfolio Theory with two sample periods:
- **SMPL1**: Up to February 2020 (pre-pandemic)
- **SMPL2**: October 2021 to September 2025 (post-pandemic)

#### Asset Universe
| Company | Sector | Rationale |
|---------|--------|-----------|
| AstraZeneca (1) | Healthcare | Diversification |
| Rio Tinto (2) | Mining | Commodity exposure |
| BP (3) | Energy | Traditional energy |
| Tesco (7) | Consumer | Defensive |
| BAE Systems (8) | Defence | Government contracts |

#### Portfolio Weights Comparison

| Company | SMPL1 Optimum | SMPL1 Min-Var | SMPL2 Optimum | SMPL2 Min-Var |
|---------|---------------|---------------|---------------|---------------|
| AstraZeneca | 48.9% | 30.1% | 0% | 17.0% |
| Rio Tinto | 30.0% | 7.0% | 0% | 17.6% |
| BP | 0% | 19.2% | 13.4% | 14.2% |
| Tesco | 5.5% | 25.3% | 31.2% | 26.4% |
| BAE Systems | 15.6% | 18.5% | 55.4% | 24.8% |

#### Terminal Wealth Results (SMPL2)
| Strategy | Terminal Wealth |
|----------|-----------------|
| Min-Variance Portfolio | £1,742.89 |
| Equal-Weight Portfolio | £1,701.17 |
| SMPL1 Optimum Portfolio | Lower |
| FTSE All-Share | Lower |

#### Efficient Frontier Analysis
- SMPL1 frontier: 0.8-1.1% monthly return at 7-10% volatility
- SMPL2 frontier: 2.5-3% monthly return at 4-6% volatility
- SMPL1 Sharpe ratio: ~0.10-0.12
- SMPL2 Sharpe ratio: ~0.40-0.45

---

##  Getting Started

### Prerequisites
`ash
pip install -r requirements.txt
`

### Running the Analysis
`ash
# Install dependencies
python 1-install_python_lib.py

# Run forecasting model
python forecasting_uk_stock_market_2.py

# Launch web application locally
streamlit run streamlit_app.py
`

### Web Application Usage
1. Navigate to the [Streamlit App](https://smm265-assetpricing.streamlit.app/)
2. Select 'Run Analysis' tab
3. Choose analysis option:
   - **Option 1**: No lags (80% accuracy)
   - **Option 2**: Optimal lags (90% accuracy)
   - **Option 3**: Additional regressors (89.5% accuracy)

---

##  Project Structure

`
 Data/                          # Input datasets
 output/                        # Generated outputs and charts
 1_ASTRX_price_div.py          # FTSE price and dividend processing
 1_uk_treasury_bill.py         # Treasury bill data processing
 2_astrx_idx.py                # Index calculations
 forecasting_uk_stock_market_*.py  # Main forecasting models
 streamlit_app.py              # Web application
 Pesaran_Timmermann_UK_Replication.ipynb  # Jupyter notebook analysis
 requirements.txt              # Python dependencies
 SMM265 Group 7.pdf           # Full project report
`

---

##  Limitations

1. **Parameter Instability**: Coefficient estimates varied while optimal lags remained stable
2. **Non-linearity**: Model may not capture non-linear relationships or structural breaks
3. **Transaction Costs**: Analysis does not account for switching costs between assets
4. **Sample Size**: Limited training sample (Apr-1997 to Oct-2015) with only 20 decision points
5. **Estimation Error**: Mean-variance optimization is highly susceptible to estimation error (Jorion, 1992)

---

##  References

- Pesaran, M.H. and Timmermann, A. (1994) - Forecasting Stock Returns
- Jorion, P. (1992) - Portfolio Optimization in Practice
- Britton-Jones, M. (1999) - The Sampling Error in Estimates of Mean-Variance Efficient Portfolio Weights

---

##  Authors

**SMM265 Group 7** - City St George's, University of London

---

##  License

This project is for educational purposes as part of the MSc Finance program.
