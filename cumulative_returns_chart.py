"""
Standalone script to create cumulative returns chart for Pesaran & Timmermann analysis
Shows switching strategy vs buy-and-hold performance over time
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, date

# Sample data from the analysis (replace with actual data)
# This represents the 19 prediction periods with complete evaluation data
data = {
    'prediction_date': [
        date(2015, 10, 31), date(2016, 4, 30), date(2016, 10, 31), date(2017, 4, 30),
        date(2017, 10, 31), date(2018, 4, 30), date(2018, 10, 31), date(2019, 4, 30),
        date(2019, 10, 31), date(2020, 4, 30), date(2020, 10, 31), date(2021, 4, 30),
        date(2021, 10, 31), date(2022, 4, 30), date(2022, 10, 31), date(2023, 4, 30),
        date(2023, 10, 31), date(2024, 4, 30), date(2024, 10, 31)
    ],
    'switching_strategy_return': [
        0.005, -0.055, 0.028, 0.174, -0.041, 0.099, -0.136, 0.083,
        -0.027, -0.174, 0.234, 0.086, 0.066, -0.103, 0.096, 0.006,
        -0.015, 0.067, 0.077
    ],
    'buyhold_strategy_return': [
        0.005, -0.055, 0.028, 0.174, -0.041, 0.099, -0.136, 0.083,
        -0.027, -0.174, 0.234, 0.086, 0.066, -0.103, 0.096, 0.006,
        -0.015, 0.067, 0.077
    ]
}

def create_cumulative_returns_chart():
    """Create and save cumulative returns comparison chart"""
    print("📊 Creating Cumulative Returns Chart...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    
    # Calculate cumulative returns
    df['switching_cumulative'] = (1 + df['switching_strategy_return']).cumprod()
    df['buyhold_cumulative'] = (1 + df['buyhold_strategy_return']).cumprod()
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot cumulative returns
    plt.plot(df['prediction_date'], df['switching_cumulative'], 
            linewidth=3, label='Switching Strategy', color='blue', alpha=0.8, marker='o', markersize=5)
    
    plt.plot(df['prediction_date'], df['buyhold_cumulative'], 
            linewidth=3, label='Buy-and-Hold (FTSE)', color='red', alpha=0.8, marker='s', markersize=5)
    
    # Add horizontal line at 1.0 (no gain/loss)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    plt.title('Cumulative Returns: Switching Strategy vs Buy-and-Hold\n(Pesaran & Timmermann Rolling Window 2015-2025)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (1 = No Change)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add summary statistics as text box
    final_switching = df['switching_cumulative'].iloc[-1]
    final_buyhold = df['buyhold_cumulative'].iloc[-1]
    switching_return = (final_switching - 1) * 100
    buyhold_return = (final_buyhold - 1) * 100
    outperformance = switching_return - buyhold_return
    
    textstr = f'Final Returns:\n'
    textstr += f'Switching Strategy: {switching_return:.1f}%\n'
    textstr += f'Buy-and-Hold: {buyhold_return:.1f}%\n'
    textstr += f'Outperformance: {outperformance:.1f}%\n\n'
    textstr += f'Total Periods: {len(df)}'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("Cumulative_Returns_Strategy_Comparison.png", dpi=300, bbox_inches='tight')
    print("   💾 Cumulative returns chart saved to: Cumulative_Returns_Strategy_Comparison.png")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 CUMULATIVE RETURNS SUMMARY:")
    print(f"   • Switching Strategy Final Value: ${final_switching:.3f} (per $1 invested)")
    print(f"   • Buy-and-Hold Final Value: ${final_buyhold:.3f} (per $1 invested)")
    print(f"   • Switching Strategy Total Return: {switching_return:.1f}%")
    print(f"   • Buy-and-Hold Total Return: {buyhold_return:.1f}%")
    print(f"   • Strategy Outperformance: {outperformance:.1f}%")
    
    # Calculate some additional metrics
    switching_volatility = df['switching_strategy_return'].std()
    buyhold_volatility = df['buyhold_strategy_return'].std()
    
    print(f"\n📊 RISK METRICS (6-month periods):")
    print(f"   • Switching Strategy Volatility: {switching_volatility*100:.1f}%")
    print(f"   • Buy-and-Hold Volatility: {buyhold_volatility*100:.1f}%")

if __name__ == "__main__":
    create_cumulative_returns_chart()
