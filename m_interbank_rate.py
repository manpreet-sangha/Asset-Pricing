import pandas as pd
from pathlib import Path

def process_interbank_rate_data():
    """
    Simple processing of UK 3 Month InterBank rate data:
    1. Load the Excel file
    2. Sort by 'date' column in ascending order
    3. Remove any NA or blank rows
    4. Save the cleaned data
    """
    
    # Define file paths
    BASE_PATH = Path(r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1")
    input_file = BASE_PATH / "Data" / "Cleaned Data" / "Monthly" / "SMM265 - UK Monthly InterBank rate.xlsx"
    output_dir = BASE_PATH / "Data" / "Cleaned Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "SMM265_UK_InterBank_Rate_Monthly.xlsx"
    
    try:
        # Load the Excel file
        print(f"Loading data from: {input_file.name}")
        df = pd.read_excel(input_file)
        
        print(f"Original data: {len(df)} rows, {len(df.columns)} columns")
        
        # Remove any completely blank rows
        df = df.dropna(how='all')
        
        # Convert date column to date type (not datetime)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            print("Date column converted to date type")
        
        # Sort by 'date' column in ascending order
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        
        print(f"Cleaned data: {len(df)} rows")
        
        # Save the cleaned data
        df.to_excel(output_file, index=False)
        print(f"Data saved to: {output_file.name}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return None
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    process_interbank_rate_data()
