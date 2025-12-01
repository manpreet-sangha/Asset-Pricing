# Import only the essential libraries needed for this file
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
BASE_PATH = Path(r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1")
CLEANED_DATA_PATH = BASE_PATH / "Data" / "Cleaned Data" / "Monthly"
OUTPUT_PATH = BASE_PATH / "Data" / "Output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def load_ftse_data():
    """Load FTSE All Share Index with Dividend Yield data"""
    file_path = CLEANED_DATA_PATH / "SMM265 - ASTRX - FTSE All Share Index - Monthly with Dividend Yield.xlsx"
    
    print(f"📁 Loading FTSE data: {file_path.name}")
    
    try:
        # Load Excel file - Date is always first column
        df = pd.read_excel(file_path)
        
        # Select columns: date, astrx_index, astrx_div_yield
        df_clean = df.iloc[:, [0, 1, 2]].copy()
        df_clean.columns = ['Date', 'FTSE_Index', 'Dividend_Yield']
        
        # Convert data types
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean['FTSE_Index'] = pd.to_numeric(df_clean['FTSE_Index'], errors='coerce')
        df_clean['Dividend_Yield'] = pd.to_numeric(df_clean['Dividend_Yield'], errors='coerce')
        
        # Remove invalid data
        df_clean = df_clean.dropna().reset_index(drop=True)
        
        # Create YYYY-MM for matching
        df_clean['Year_Month'] = df_clean['Date'].dt.to_period('M')
        
        print(f"   ✅ Loaded {len(df_clean)} observations from {df_clean['Date'].min():%Y-%m} to {df_clean['Date'].max():%Y-%m}")
        
        return df_clean
        
    except Exception as e:
        print(f"   ❌ Error loading FTSE data: {e}")
        return None

def load_tbill_data():
    """Load UK Treasury Bill 3 Month Rate data"""
    file_path = CLEANED_DATA_PATH / "SMM265 - 1126593 Index - IMF UK Treasury Bill 3 Month Rate- Monthly.xlsx"
    
    print(f"📁 Loading T-Bill data: {file_path.name}")
    
    try:
        # Load Excel file - Date is always first column
        df = pd.read_excel(file_path)
        
        # Select columns: Date, uktb_yield
        df_clean = df.iloc[:, [0, 1]].copy()
        df_clean.columns = ['Date', 'TBill_Rate']
        
        # Convert data types
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean['TBill_Rate'] = pd.to_numeric(df_clean['TBill_Rate'], errors='coerce')
        
        # Remove invalid data
        df_clean = df_clean.dropna().reset_index(drop=True)
        
        # Create YYYY-MM for matching
        df_clean['Year_Month'] = df_clean['Date'].dt.to_period('M')
        
        print(f"   ✅ Loaded {len(df_clean)} observations from {df_clean['Date'].min():%Y-%m} to {df_clean['Date'].max():%Y-%m}")
        
        return df_clean
        
    except Exception as e:
        print(f"   ❌ Error loading T-Bill data: {e}")
        return None

def merge_and_save_data(ftse_data, tbill_data):
    """Merge datasets on Year-Month and save to Excel"""
    print(f"🔄 Merging datasets...")
    
    if ftse_data is None or tbill_data is None:
        print("   ❌ Missing data - cannot merge")
        return False
    
    try:
        # Group by Year_Month and take last observation per month
        ftse_monthly = ftse_data.groupby('Year_Month').last().reset_index()
        tbill_monthly = tbill_data.groupby('Year_Month').last().reset_index()
        
        # Merge on Year_Month (inner join - only overlapping months)
        merged = pd.merge(
            ftse_monthly[['Year_Month', 'Date', 'FTSE_Index', 'Dividend_Yield']], 
            tbill_monthly[['Year_Month', 'TBill_Rate']], 
            on='Year_Month', 
            how='inner'
        )
        
        # Sort by date and reorder columns
        merged = merged.sort_values('Year_Month').reset_index(drop=True)
        
        # Convert Date column from timestamp to date format (CCYY-MM-DD)
        merged['Date'] = merged['Date'].dt.date
        
        # Reorder columns
        merged = merged[['Date', 'Year_Month', 'FTSE_Index', 'Dividend_Yield', 'TBill_Rate']]
        
        print(f"   ✅ Merged dataset: {len(merged)} observations ({merged['Year_Month'].min()} to {merged['Year_Month'].max()})")
        print(f"   📅 Date format: {type(merged['Date'].iloc[0])} - {merged['Date'].iloc[0]}")
        
        # Save to Excel
        output_file = OUTPUT_PATH / "SMM265_Index_DivYield_TBill_Monthly.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main data
            merged.to_excel(writer, sheet_name='Data', index=False)
            
            # Summary statistics
            summary = pd.DataFrame({
                'Metric': ['Observations', 'Start Date', 'End Date', 'Avg FTSE Index', 'Avg Dividend Yield (%)', 'Avg T-Bill Rate (%)'],
                'Value': [
                    len(merged),
                    str(merged['Date'].min()),  # Convert date to string for Excel
                    str(merged['Date'].max()),  # Convert date to string for Excel
                    f"{merged['FTSE_Index'].mean():.2f}",
                    f"{merged['Dividend_Yield'].mean():.2f}",
                    f"{merged['TBill_Rate'].mean():.2f}"
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"   💾 Saved to: {output_file}")
        print(f"\n📊 Sample data:")
        print(merged.head())
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error merging/saving: {e}")
        return False

def main():
    """Load, merge and save UK financial data"""
    print("🚀 LOADING UK FINANCIAL DATA")
    print("="*40)
    
    # Load both datasets
    ftse_data = load_ftse_data()
    tbill_data = load_tbill_data()
    
    # Merge and save
    success = merge_and_save_data(ftse_data, tbill_data)
    
    if success:
        print(f"\n🎉 SUCCESS! Data saved to {OUTPUT_PATH}")
    else:
        print(f"\n❌ FAILED to process data")

if __name__ == "__main__":
    main()