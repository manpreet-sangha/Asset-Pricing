"""
Create Table 1: Recursive Predictions and Actual Excess Returns
Shows all 20 prediction periods with detailed results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def create_recursive_predictions_table():
    """Create Table 1 showing all recursive predictions and actual returns"""
    
    # Data from our analysis (20 prediction periods)
    # Based on the rolling window analysis results
    table_data = [
        # Period, Actual Return, Predicted Return, Sign Correct, Trading Decision, Return Achieved
        ["Nov 2015 - Apr 2016", -0.0028, 0.0383, "✗", "Stocks", 0.0001],
        ["May 2016 - Oct 2016", 0.1136, 0.0418, "✓", "Stocks", 0.1158],
        ["Nov 2016 - Apr 2017", 0.0689, 0.0359, "✓", "Stocks", 0.0706],
        ["May 2017 - Oct 2017", 0.0384, 0.0340, "✓", "Stocks", 0.0397],
        ["Nov 2017 - Apr 2018", 0.0185, 0.0422, "✓", "Stocks", 0.0212],
        ["May 2018 - Oct 2018", 0.0147, 0.0495, "✓", "Stocks", 0.0177],
        ["Nov 2018 - Apr 2019", 0.0593, 0.0388, "✓", "Stocks", 0.0636],
        ["May 2019 - Oct 2019", 0.0151, 0.0522, "✓", "Stocks", 0.0184],
        ["Nov 2019 - Apr 2020", -0.1738, 0.0499, "✗", "Stocks", -0.1702],
        ["May 2020 - Oct 2020", -0.0206, 0.0562, "✗", "Stocks", -0.0195],
        ["Nov 2020 - Apr 2021", 0.2843, 0.0264, "✓", "Stocks", 0.2846],
        ["May 2021 - Oct 2021", 0.0536, 0.0170, "✓", "Stocks", 0.0540],
        ["Nov 2021 - Apr 2022", 0.0287, 0.0287, "✓", "Stocks", 0.0315],
        ["May 2022 - Oct 2022", -0.0936, 0.0245, "✗", "Stocks", -0.0859],
        ["Nov 2022 - Apr 2023", 0.1059, 0.0333, "✓", "Stocks", 0.1251],
        ["May 2023 - Oct 2023", -0.0401, 0.0357, "✗", "Stocks", -0.0187],
        ["Nov 2023 - Apr 2024", 0.1165, 0.0307, "✓", "Stocks", 0.1423],
        ["May 2024 - Oct 2024", 0.0142, 0.0300, "✓", "Stocks", 0.0351],
        ["Nov 2024 - Apr 2025", 0.0339, 0.0286, "✓", "Stocks", 0.0561],
        ["May 2025 - Oct 2025", 0.1013, 0.0272, "✓", "Stocks", 0.1183]
    ]
    
    # Calculate summary statistics
    total_predictions = len(table_data)
    correct_predictions = sum(1 for row in table_data if row[3] == "✓")
    accuracy_percentage = (correct_predictions / total_predictions) * 100
    
    # Create figure with proper size for the table
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.axis('off')
    
    # Table title
    title = "Table 1: Recursive Predictions and Actual Excess Returns\nFTSE All-Share Index - Rolling Window Analysis (2015-2025)"
    ax.text(0.5, 0.98, title, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', ha='center', va='top')
    
    # Column headers
    headers = ["Period", "Actual\nReturn", "Predicted\nReturn", "Sign\nCorrect?", "Trading\nDecision", "Return\nAchieved"]
    
    # Prepare data for table (format numbers)
    formatted_data = []
    for row in table_data:
        formatted_row = [
            row[0],  # Period
            f"{row[1]:.3f}",  # Actual return (formatted)
            f"{row[2]:.3f}",  # Predicted return (formatted)
            row[3],  # Sign correct
            row[4],  # Trading decision
            f"{row[5]:.3f}"   # Return achieved (formatted)
        ]
        formatted_data.append(formatted_row)
    
    # Create the main table
    table = ax.table(cellText=formatted_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0.05, 0.15, 0.9, 0.75])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style headers
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data rows
    for i in range(1, len(formatted_data) + 1):
        for j in range(len(headers)):
            # Alternate row colors
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            
            # Highlight correct/incorrect predictions
            if j == 3:  # Sign Correct column
                if formatted_data[i-1][j] == "✓":
                    table[(i, j)].set_facecolor('#D5E8D4')
                    table[(i, j)].set_text_props(color='green', weight='bold')
                else:
                    table[(i, j)].set_facecolor('#F8CECC')
                    table[(i, j)].set_text_props(color='red', weight='bold')
            
            # Style trading decision column
            if j == 4:  # Trading Decision column
                table[(i, j)].set_text_props(weight='bold')
            
            # Add borders
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(0.5)
    
    # Add separator lines
    separator_line = "─" * 80
    ax.text(0.5, 0.12, separator_line, transform=ax.transAxes,
            fontsize=12, ha='center', va='center', family='monospace')
    
    # Add summary statistics
    summary_text = f"Total correct: {correct_predictions} out of {total_predictions} ({accuracy_percentage:.1f}%)"
    ax.text(0.5, 0.09, summary_text, transform=ax.transAxes,
            fontsize=12, ha='center', va='center', weight='bold')
    
    ax.text(0.5, 0.06, separator_line, transform=ax.transAxes,
            fontsize=12, ha='center', va='center', family='monospace')
    
    # Add methodology note
    methodology_note = ("Note: Predictions based on dividend yield model with rolling 6-month re-estimation.\n"
                       "Trading decision: 'Stocks' if predicted excess return > 0, otherwise 'Bonds'.\n"
                       "All trading decisions resulted in 'Stocks' due to consistently positive predictions.")
    
    ax.text(0.5, 0.02, methodology_note, transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom', style='italic')
    
    plt.tight_layout()
    
    # Save the table
    output_dir = Path("Data/Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Recursive_Predictions_Table.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    
    print(f"📊 Recursive Predictions Table created and saved as: {output_file.name}")
    print(f"   💾 Location: {output_file.absolute()}")
    
    # Print summary to console
    print(f"\n📈 RECURSIVE PREDICTIONS SUMMARY:")
    print(f"   • Total prediction periods: {total_predictions}")
    print(f"   • Correct sign predictions: {correct_predictions}")
    print(f"   • Prediction accuracy: {accuracy_percentage:.1f}%")
    print(f"   • Trading decisions: All {total_predictions} periods invested in Stocks")
    print(f"   • Model consistency: Always predicted positive excess returns")
    
    # Print some key statistics
    actual_returns = [row[1] for row in table_data]
    predicted_returns = [row[2] for row in table_data]
    achieved_returns = [row[5] for row in table_data]
    
    print(f"\n📊 RETURN STATISTICS:")
    print(f"   • Average actual excess return: {np.mean(actual_returns)*100:.2f}%")
    print(f"   • Average predicted excess return: {np.mean(predicted_returns)*100:.2f}%")
    print(f"   • Average achieved return: {np.mean(achieved_returns)*100:.2f}%")
    print(f"   • Volatility of actual returns: {np.std(actual_returns)*100:.2f}%")
    
    plt.show()
    
    return {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy_percentage,
        'table_data': table_data
    }

if __name__ == "__main__":
    create_recursive_predictions_table()
