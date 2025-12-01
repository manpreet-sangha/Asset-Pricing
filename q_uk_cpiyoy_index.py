# UK CPI Year-over-Year Index Analysis - Quarterly Data
# SMM265 Asset Pricing - Quarterly Data
# Author: [201148348 - Manpreet Sangha]
# Date: November 30, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_and_clean_data():
    """
    Load and clean the UK CPI Year-over-Year data from Excel file
    """
    # File path
    file_path = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1\Data\Raw Data\Quarterly\SMM265 - UKRPCJYR Index - UK CPI YoY - Quarterly.xlsx"
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the Excel file
        print("Loading UK CPI YoY Excel file...")
        df = pd.read_excel(file_path)
        
        # Display basic info about the dataset
        print(f"Original dataset shape: {df.shape}")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Basic data cleaning steps
        print("\n" + "="*50)
        print("CLEANING UK CPI YOY DATA")
        print("="*50)
        
        # Remove first five rows
        df_cleaned = df.iloc[5:].copy()
        print(f"After removing first 5 rows: {df_cleaned.shape}")
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all').dropna(axis=1, how='all')
        print(f"After removing empty rows/columns: {df_cleaned.shape}")
        
        # Reset index after removing rows
        df_cleaned = df_cleaned.reset_index(drop=True)
        print(f"Index reset. Final shape: {df_cleaned.shape}")
        
        # Rename columns to 'date' and 'ukrpcjyr_index'
        if len(df_cleaned.columns) >= 2:
            df_cleaned.columns = ['date', 'ukrpcjyr_index']
            print(f"Columns renamed to: {list(df_cleaned.columns)}")
        else:
            print(f"Warning: Expected at least 2 columns, found {len(df_cleaned.columns)}")
        
        # Convert date column to datetime and remove timestamp
        if 'date' in df_cleaned.columns:
            try:
                df_cleaned['date'] = pd.to_datetime(df_cleaned['date']).dt.date
                print("Date column converted to date format (no timestamp)")
            except Exception as e:
                print(f"Warning: Could not convert date column: {e}")
        
        # Sort by date in ascending order
        if 'date' in df_cleaned.columns:
            try:
                df_cleaned = df_cleaned.sort_values('date', ascending=True).reset_index(drop=True)
                print("Data sorted by date in ascending order")
            except Exception as e:
                print(f"Warning: Could not sort by date: {e}")
        
        # Display cleaned data info
        print("\nCleaned dataset info:")
        print(df_cleaned.info())
        print("\nData types:")
        print(df_cleaned.dtypes)
        
        # Show first few rows of cleaned data
        print("\nFirst few rows of cleaned data:")
        print(df_cleaned.head())
        
        # Check for missing values
        missing_values = df_cleaned.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found.")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error loading/cleaning UK CPI YoY data: {e}")
        return None

def save_cleaned_data(df, output_path=None):
    """
    Save cleaned UK CPI YoY data to Excel file
    """
    if df is None:
        print("No data to save.")
        return
    
    if output_path is None:
        # Create output directory if it doesn't exist
        output_dir = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1\Data\Cleaned Data\Quarterly"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "SMM265 - UKRPCJYR Index - UK CPI YoY - Quarterly.xlsx")
    
    try:
        df.to_excel(output_path, index=False)
        print(f"\nCleaned UK CPI YoY data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving UK CPI YoY data: {e}")

def main():
    """
    Main function for UK CPI YoY analysis
    """
    print("UK CPI Year-over-Year Analysis - Quarterly")
    print("=" * 40)
    
    # Load and clean the data
    cleaned_data = load_and_clean_data()
    
    if cleaned_data is not None:
        # Save cleaned data
        save_cleaned_data(cleaned_data)
        
        # Display summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(cleaned_data.describe())
    else:
        print("Data cleaning failed. Please check the file path and format.")

if __name__ == "__main__":
    main()
