"""
Streamlit Web Application for Pesaran & Timmermann Multi-Predictor Model
=========================================================================
This webapp allows users to:
1. View preloaded input data files
2. Execute the forecasting analysis
3. View output charts and tables in the browser
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
import sys
from contextlib import redirect_stdout
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pesaran & Timmermann Multi-Predictor Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Define paths - using relative paths for webapp
WEBAPP_PATH = Path(__file__).parent
DATA_PATH = WEBAPP_PATH / "Data"

def load_input_data():
    """Load the input data files from the data folder"""
    data_files = {}
    
    file_mappings = {
        'index': ('SMM265_Index_DivYield_Monthly.xlsx', 'FTSE Index & Dividend Yield'),
        'interbank': ('SMM265_UK_InterBank_Rate_Monthly.xlsx', 'UK InterBank Rate'),
        'cpi': ('SMM265 - UKRPCJYR Index - UK CPI YoY - Monthly.xlsx', 'UK CPI Year-over-Year'),
        'ip': ('SMM265 - UKIPIYOY Index - UK IP YoY - Monthly.xlsx', 'UK Industrial Production YoY')
    }
    
    for key, (filename, display_name) in file_mappings.items():
        data_file = DATA_PATH / filename
        
        if data_file.exists():
            try:
                data_files[key] = {
                    'data': pd.read_excel(data_file),
                    'name': display_name,
                    'path': str(data_file),
                    'source': 'repo'
                }
            except Exception as e:
                st.error(f"Error loading {display_name}: {e}")
                data_files[key] = None
        else:
            st.error(f"File not found: {filename}")
            data_files[key] = None
    
    return data_files


def run_analysis(index_data, interbank_data, cpi_data, analysis_type="option1", ip_data=None, progress_callback=None):
    """Run the Pesaran & Timmermann analysis with the provided data
    
    Args:
        index_data: FTSE Index data
        interbank_data: InterBank rate data
        cpi_data: CPI data
        analysis_type: "option1" for standard, "option2" for 2-predictor lag selection, "option3" for 3-predictor
        ip_data: Industrial Production data (required for option3)
        progress_callback: Progress update callback function
    """
    from forecasting_uk_stock_market_2_webapp import PesaranTimmermannMultiPredictor, DividendYieldLagSelectionAnalysis, ThreePredictorLagSelectionAnalysis
    
    # Create predictor instance based on analysis type
    if analysis_type == "option3":
        predictor = ThreePredictorLagSelectionAnalysis()
    elif analysis_type == "option2":
        predictor = DividendYieldLagSelectionAnalysis()
    else:
        predictor = PesaranTimmermannMultiPredictor()
    
    # Set data directly instead of loading from files
    if analysis_type == "option3" and ip_data is not None:
        predictor.set_input_data_with_ip(index_data, interbank_data, cpi_data, ip_data)
    else:
        predictor.set_input_data(index_data, interbank_data, cpi_data)
    
    # Run the analysis steps
    if progress_callback:
        progress_callback(0.1, "Computing returns...")
    predictor.compute_returns()
    
    if progress_callback:
        progress_callback(0.2, "Preparing training data...")
    predictor.prepare_initial_training_data()
    
    if progress_callback:
        progress_callback(0.3, "Training initial model...")
    predictor.train_model()
    
    if progress_callback:
        progress_callback(0.5, "Running rolling window predictions...")
    
    # Run appropriate prediction method based on analysis type
    if analysis_type == "option3" or analysis_type == "option2":
        predictor.rolling_window_predictions_with_lags()
    else:
        predictor.rolling_window_predictions()
    
    if progress_callback:
        progress_callback(0.8, "Generating visualizations...")
    
    # Generate plots and return them
    plots = {}
    
    # Plot 1: Predicted vs Actual Returns
    fig1 = predictor.plot_predicted_vs_actual_returns_webapp()
    if fig1:
        plots['predicted_vs_actual'] = fig1
    
    # Plot 2: Cumulative Returns
    fig2 = predictor.plot_cumulative_returns_chart_webapp()
    if fig2:
        plots['cumulative_returns'] = fig2
    
    # Plot 3: PT Test Table (only for Option 1)
    if analysis_type == "option1":
        fig3 = predictor.create_pt_test_table_webapp()
        if fig3:
            plots['pt_test_table'] = fig3
    
    # Plot 4: Predictions Table
    fig4 = predictor.create_pt_predictions_table_webapp()
    if fig4:
        plots['predictions_table'] = fig4
    
    if progress_callback:
        progress_callback(1.0, "Analysis complete!")
    
    return {
        'predictor': predictor,
        'plots': plots,
        'rolling_predictions': predictor.rolling_predictions if hasattr(predictor, 'rolling_predictions') else None,
        'performance_metrics': predictor.performance_metrics if hasattr(predictor, 'performance_metrics') else None,
        'analysis_type': analysis_type
    }


def main():
    # Header
    st.markdown('<p class="main-header">SMM265 Asset Pricing Coursework</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Group 7 - Question 1</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">📈 Pesaran & Timmermann Multi-Predictor Model<br>UK Stock Market Forecasting using Dividend Yield, CPI, and Industrial Production</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Load input data
    st.sidebar.subheader("📁 Input Data")
    data_files = load_input_data()
    
    # Check if all data is available (base data for options 1 & 2)
    base_data_available = all(data_files.get(key) is not None for key in ['index', 'interbank', 'cpi'])
    # Check if IP data is also available (for option 3)
    ip_data_available = data_files.get('ip') is not None
    all_data_available = base_data_available and ip_data_available
    
    if all_data_available:
        st.sidebar.success("✅ All input files loaded (including IP)!")
    elif base_data_available:
        st.sidebar.success("✅ Base input files loaded!")
        if not ip_data_available:
            st.sidebar.warning("⚠️ IP data not available - Option 3 disabled")
    else:
        st.sidebar.warning("⚠️ Some input files are missing")
        missing = [key for key in ['index', 'interbank', 'cpi', 'ip'] if data_files.get(key) is None]
        for key in missing:
            st.sidebar.error(f"❌ Missing: {key}")
    
    # Show data sources
    for key, info in data_files.items():
        if info:
            st.sidebar.text(f"• {info['name']}: ✓")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Input Data", "▶️ Run Analysis", "📈 Results", "📋 Methodology"])
    
    # Tab 1: Input Data
    with tab1:
        st.header("Input Data Preview")
        
        if base_data_available:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("FTSE Index & Dividend Yield")
                df_index = data_files['index']['data']
                st.dataframe(df_index.head(10), width='stretch')
                st.caption(f"Total rows: {len(df_index)}")
            
            with col2:
                st.subheader("UK InterBank Rate")
                df_interbank = data_files['interbank']['data']
                st.dataframe(df_interbank.head(10), width='stretch')
                st.caption(f"Total rows: {len(df_interbank)}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("UK CPI YoY")
                df_cpi = data_files['cpi']['data']
                st.dataframe(df_cpi.head(10), width='stretch')
                st.caption(f"Total rows: {len(df_cpi)}")
            
            with col4:
                st.subheader("UK Industrial Production YoY")
                if ip_data_available:
                    df_ip = data_files['ip']['data']
                    st.dataframe(df_ip.head(10), width='stretch')
                    st.caption(f"Total rows: {len(df_ip)}")
                else:
                    st.warning("IP data not available")
        else:
            st.warning("Please ensure all input data files are available.")
            st.info("""
            **Required files:**
            1. `SMM265_Index_DivYield_Monthly.xlsx` - FTSE Index and Dividend Yield data
            2. `SMM265_UK_InterBank_Rate_Monthly.xlsx` - UK InterBank Rate data
            3. `SMM265 - UKRPCJYR Index - UK CPI YoY - Monthly.xlsx` - UK CPI data
            4. `SMM265 - UKIPIYOY Index - UK IP YoY - Monthly.xlsx` - UK Industrial Production data (for Option 3)
            
            Place these files in the `Data/` folder.
            """)
    
    # Tab 2: Run Analysis
    with tab2:
        st.header("Run Analysis")
        
        if base_data_available:
            # Analysis Type Selection
            st.subheader("📋 Select Analysis Type")
            
            # Build options list based on available data
            options_list = ["option1", "option2"]
            if ip_data_available:
                options_list.append("option3")
            
            def format_option(x):
                if x == "option1":
                    return "Option 1: Standard Analysis (Current DY + CPI)"
                elif x == "option2":
                    return "Option 2: 2-Predictor Lag Selection (DY + CPI with Lags)"
                else:
                    return "Option 3: 3-Predictor Lag Selection (DY + CPI + IP with Lags)"
            
            analysis_option = st.radio(
                "Choose the analysis method:",
                options=options_list,
                format_func=format_option,
                horizontal=False,
                help="Option 1 uses current values. Option 2 tests DY and CPI lags. Option 3 adds Industrial Production."
            )
            
            if analysis_option == "option1":
                st.info("""
                **Option 1 - Standard Analysis (2 Predictors):**
                - Model: Excess_Return = α + β₁×DY + β₂×CPI + ε
                - Uses **current** dividend yield and CPI values
                - 20 rolling window predictions (Oct 2015 - Apr 2025)
                """)
            elif analysis_option == "option2":
                st.info("""
                **Option 2 - Sequential Lag Selection (2 Predictors):**
                - Model: Excess_Return = α + β₁×DY(lag) + β₂×CPI(lag) + ε
                - **Step 1:** Test DY lags (0, 1, 2, 3) → Select best by |t-stat|
                - **Step 2:** Given optimal DY lag, test CPI lags (0, 1, 2) → Select best by |t-stat|
                - 20 rolling window predictions with optimal lags per iteration
                """)
            else:
                st.info("""
                **Option 3 - Sequential Lag Selection (3 Predictors with IP):**
                - Model: Excess_Return = α + β₁×DY(lag) + β₂×CPI(lag) + β₃×IP(lag) + ε
                - **Step 1:** Test DY lags (0, 1, 2, 3) → Select best by |t-stat|
                - **Step 2:** Given optimal DY lag, test CPI lags (0, 1, 2) → Select best by |t-stat|
                - **Step 3:** Given optimal DY & CPI lags, test IP lags (0, 1, 2) → Select best by |t-stat|
                - 20 rolling window predictions with optimal lags per iteration
                """)
            
            st.divider()
            
            # Initialize session state for results
            if 'analysis_results' not in st.session_state:
                st.session_state.analysis_results = None
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                run_button = st.button("▶️ Run Analysis", type="primary", width='stretch')
            
            if run_button:
                with st.spinner("Running analysis..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(value, text):
                        progress_bar.progress(value)
                        status_text.text(text)
                    
                    # Capture stdout
                    output_capture = io.StringIO()
                    
                    try:
                        # Get IP data if available and option3 selected
                        ip_data = data_files['ip']['data'] if (ip_data_available and analysis_option == "option3") else None
                        
                        # Run the analysis with selected option
                        results = run_analysis(
                            data_files['index']['data'],
                            data_files['interbank']['data'],
                            data_files['cpi']['data'],
                            analysis_type=analysis_option,
                            ip_data=ip_data,
                            progress_callback=update_progress
                        )
                        
                        st.session_state.analysis_results = results
                        
                        option_name = "Standard Analysis" if analysis_option == "option1" else "Lag Selection Analysis"
                        st.success(f"✅ {option_name} completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error running analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Show console output expander
            if st.session_state.analysis_results:
                st.success("Results are ready! Go to the **Results** tab to view plots and metrics.")
        else:
            st.warning("Cannot run analysis - input data files are missing.")
    
    # Tab 3: Results
    with tab3:
        st.header("Analysis Results")
        
        if st.session_state.get('analysis_results'):
            results = st.session_state.analysis_results
            
            # Show which analysis type was run
            analysis_type = results.get('analysis_type', 'option1')
            if analysis_type == "option3":
                st.info("📊 **3-Predictor Lag Selection Results** - Using optimal DY, CPI, and IP lags per iteration")
            elif analysis_type == "option2":
                st.info("📊 **2-Predictor Lag Selection Results** - Using optimal DY and CPI lags per iteration")
            else:
                st.info("📊 **Standard Analysis Results** - Using current dividend yield and CPI")
            
            # Performance Metrics Summary
            if results.get('performance_metrics'):
                st.subheader("📊 Performance Summary")
                metrics = results['performance_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Prediction Accuracy",
                        f"{metrics['accuracy']*100:.1f}%",
                        f"{metrics['correct_predictions']}/{metrics['total_predictions']}"
                    )
                
                with col2:
                    st.metric(
                        "Switching Strategy Return",
                        f"{metrics['switching_total_return']*100:.1f}%",
                        f"{metrics['switching_annual_return']*100:.1f}% annual"
                    )
                
                with col3:
                    st.metric(
                        "Buy-Hold Return",
                        f"{metrics['buyhold_total_return']*100:.1f}%",
                        f"{metrics['buyhold_annual_return']*100:.1f}% annual"
                    )
                
                with col4:
                    advantage = (metrics['switching_total_return'] - metrics['buyhold_total_return']) * 100
                    st.metric(
                        "Strategy Advantage",
                        f"{advantage:.1f}%",
                        "Outperformance" if advantage > 0 else "Underperformance"
                    )
            
            # Plots
            st.subheader("📈 Visualizations")
            
            plots = results.get('plots', {})
            
            if plots.get('predicted_vs_actual'):
                st.pyplot(plots['predicted_vs_actual'])
                st.caption("Plot 1: Predicted vs Actual Excess Returns (Time Series)")
            
            if plots.get('cumulative_returns'):
                st.pyplot(plots['cumulative_returns'])
                st.caption("Plot 2: Cumulative Returns - Strategy Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if plots.get('pt_test_table'):
                    st.pyplot(plots['pt_test_table'])
                    st.caption("Table 1: PT Sign Test Statistics")
            
            with col2:
                if plots.get('predictions_table'):
                    st.pyplot(plots['predictions_table'])
                    st.caption("Table 2: Predictions vs Actual Results")
            
            # Rolling Predictions Data
            if results.get('rolling_predictions') is not None:
                st.subheader("📋 Rolling Predictions Data")
                st.dataframe(results['rolling_predictions'], width='stretch')
                
                # Download button
                csv = results['rolling_predictions'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="rolling_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.info("No results available. Please run the analysis first in the **Run Analysis** tab.")
    
    # Tab 4: Methodology
    with tab4:
        st.header("Methodology")
        
        st.markdown("""
        ## Pesaran & Timmermann (1994) Multi-Predictor Model
        
        ### Model Specification
        
        **2-Predictor Model (Options 1 & 2):**
        ```
        Excess_Return = α + β₁ × Dividend_Yield + β₂ × CPI_Growth + ε
        ```
        
        **3-Predictor Model (Option 3):**
        ```
        Excess_Return = α + β₁ × Dividend_Yield + β₂ × CPI_Growth + β₃ × IP_Growth + ε
        ```
        
        Where:
        - **Excess_Return** = FTSE All-Share 6-month return - InterBank 6-month return
        - **Dividend_Yield** = Dividend yield of FTSE All-Share (lags: 0, 1, 2, 3)
        - **CPI_Growth** = UK Consumer Price Index YoY growth rate (lags: 0, 1, 2)
        - **IP_Growth** = UK Industrial Production YoY growth rate (lags: 0, 1, 2)
        
        ---
        
        ### Option 1: Standard Analysis (2 Predictors)
        
        Uses **current** values of both predictors:
        - Dividend Yield: Current month
        - CPI Growth: Current month
        
        ---
        
        ### Option 2: Lag Selection Analysis (2 Predictors)
        
        Implements **sequential lag selection**:
        
        **Step 1 - Select Optimal DY Lag:**
        - Test lags: 0, 1, 2, 3 months
        - Selection criterion: Highest |t-statistic| for β₁
        
        **Step 2 - Select Optimal CPI Lag:**
        - Given optimal DY lag from Step 1
        - Test lags: 0, 1, 2 months
        - Selection criterion: Highest |t-statistic| for β₂
        
        ---
        
        ### Option 3: Lag Selection Analysis (3 Predictors with IP)
        
        Extends Option 2 with Industrial Production as a third predictor:
        
        **Step 1 - Select Optimal DY Lag:**
        - Test lags: 0, 1, 2, 3 months → Select best by |t-stat| for β₁
        
        **Step 2 - Select Optimal CPI Lag:**
        - Given optimal DY lag, test CPI lags: 0, 1, 2 months
        - Selection criterion: Highest |t-statistic| for β₂
        
        **Step 3 - Select Optimal IP Lag:**
        - Given optimal DY and CPI lags, test IP lags: 0, 1, 2 months
        - Selection criterion: Highest |t-statistic| for β₃
        
        **Step 4 - Make Prediction:**
        - Use final 3-predictor model with all optimal lags
        - Generate 6-month ahead forecast
        
        ---
        
        ### Rolling Window Approach
        
        1. **Initial Training Period**: April 1997 → September 2015
        2. **Expanding Windows**: Each prediction uses all data from April 1997 to the current date
        3. **Re-estimation Frequency**: Every 6 months (April and October)
        4. **Prediction Horizon**: 6 months ahead
        5. **Total Predictions**: 20 (Oct 2015 → Apr 2025)
        
        ### Investment Strategy
        
        The switching strategy makes investment decisions based on predicted excess returns:
        
        - **If predicted excess return > 0** → Invest in FTSE All-Share Index
        - **If predicted excess return ≤ 0** → Invest in InterBank Rate (risk-free)
        
        ### Evaluation Metrics
        
        - **Prediction Accuracy**: Percentage of correct sign predictions
        - **Pesaran-Timmermann Test**: Statistical test for market timing ability
        - **Total Return**: Cumulative return over the evaluation period
        - **Sharpe Ratio**: Risk-adjusted return measure
        """)


if __name__ == "__main__":
    main()
