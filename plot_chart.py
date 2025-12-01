import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths
BASE_PATH = Path(r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term1\Coursework\SMM265 Asset Pricing\Q1")

# Create sample data from the analysis (the actual rolling predictions)
data = {
    'prediction_date': ['2015-10-31', '2016-04-30', '2016-10-31', '2017-04-30', '2017-10-31', 
                       '2018-04-30', '2018-10-31', '2019-04-30', '2019-10-31', '2020-04-30',
                       '2020-10-31', '2021-04-30', '2021-10-31', '2022-04-30', '2022-10-31',
                       '2023-04-30', '2023-10-31', '2024-04-30', '2024-10-31', '2025-04-30'],
    'predicted_excess_return': [0.0383, 0.0418, 0.0359, 0.0340, 0.0422, 0.0495, 0.0388, 0.0522, 
                               0.0499, 0.0562, 0.0264, 0.0170, 0.0287, 0.0245, 0.0333, 0.0357, 
                               0.0307, 0.0300, 0.0286, 0.0272],
    'actual_excess_return': [-0.0028, 0.1136, 0.0689, 0.0384, 0.0185, 0.0147, 0.0593, 0.0151,
                            -0.1738, -0.0206, 0.2843, 0.0536, 0.0287, -0.0936, 0.1059, -0.0401,
                            0.1165, 0.0142, 0.0339, 0.1013],
    'prediction_correct': [False, True, True, True, True, True, True, True, False, False, 
                          True, True, True, False, True, False, True, True, True, True]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['prediction_date'] = pd.to_datetime(df['prediction_date'])

# Create the plot
plt.figure(figsize=(14, 8))

# Plot predicted and actual excess returns
plt.plot(df['prediction_date'], df['predicted_excess_return']*100, 
         marker='o', linewidth=2, markersize=6, label='Predicted Excess Return', color='blue', alpha=0.8)

plt.plot(df['prediction_date'], df['actual_excess_return']*100, 
         marker='s', linewidth=2, markersize=6, label='Actual Excess Return', color='red', alpha=0.8)

# Add zero line for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Formatting
plt.title('Forecasting based on Pesaran & Timmermann Model: Predicted vs Actual Excess Returns\n(Rolling Window Predictions 2015-2025)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Prediction Date', fontsize=12)
plt.ylabel('Excess Return (%)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add summary statistics as text box
correct_predictions = df['prediction_correct'].sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions * 100

textstr = f'Prediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)\n'
textstr += f'Avg Predicted: {df["predicted_excess_return"].mean()*100:.2f}%\n'
textstr += f'Avg Actual: {df["actual_excess_return"].mean()*100:.2f}%'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()

# Save the plot
output_dir = BASE_PATH / "Data" / "Output"
output_dir.mkdir(parents=True, exist_ok=True)
plot_file = output_dir / "Predicted_vs_Actual_Returns_TimeSeries.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"📊 Time-series plot saved to: {plot_file}")

plt.show()
