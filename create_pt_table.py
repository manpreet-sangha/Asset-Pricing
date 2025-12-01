"""
Create Pesaran-Timmermann Sign Test Statistics table as an image
Based on the rolling window analysis results from 2015-2025
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path

def calculate_pt_test_statistics():
    """
    Calculate Pesaran-Timmermann sign test statistics based on our analysis
    From the previous run, we know:
    - Total predictions: 20
    - Correct predictions: 15 (75% accuracy)
    - All predictions were for STOCKS (20/20)
    """
    
    # Data from our analysis
    total_predictions = 20
    correct_predictions = 15
    proportion_correct = correct_predictions / total_predictions
    expected_proportion = 0.5  # Under null hypothesis of no predictability
    
    # Number of switches (model always chose STOCKS, so 0 switches)
    num_switches = 0  # All 20 predictions were STOCKS
    
    # Calculate PT test statistic
    # PT = (P - 0.5) / sqrt(Pstar * (1 - Pstar) / N)
    # where Pstar is the expected success probability under independence
    
    # Under independence, Pstar = 0.5
    pstar = 0.5
    n = total_predictions
    
    # PT test statistic
    pt_statistic = (proportion_correct - pstar) / np.sqrt(pstar * (1 - pstar) / n)
    
    # Calculate p-value (one-sided test)
    # We test H0: no predictability vs H1: predictability exists
    p_value_one_sided = 1 - stats.norm.cdf(pt_statistic)
    
    # Significance tests
    significant_5pct = p_value_one_sided < 0.05
    significant_1pct = p_value_one_sided < 0.01
    
    return {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'proportion_correct': proportion_correct * 100,
        'expected_proportion': expected_proportion * 100,
        'pt_statistic': pt_statistic,
        'p_value': p_value_one_sided,
        'significant_5pct': significant_5pct,
        'significant_1pct': significant_1pct,
        'num_switches': num_switches
    }

def create_pt_table_image():
    """Create the Pesaran-Timmermann table as a high-quality image"""
    
    # Calculate statistics
    stats_data = calculate_pt_test_statistics()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Hide axes
    
    # Table title
    title = "Table 2: Pesaran-Timmermann Sign Test Statistics\nEvaluation Period: November 2015 – October 2025\nFTSE All-Share"
    
    # Create table data
    table_data = [
        ["Number of predictions", f"{stats_data['total_predictions']}"],
        ["Correct sign predictions", f"{stats_data['correct_predictions']}"],
        ["Proportion correct (%)", f"{stats_data['proportion_correct']:.1f}%"],
        ["Expected proportion under H₀ (%)", f"{stats_data['expected_proportion']:.1f}%"],
        ["PT test statistic", f"{stats_data['pt_statistic']:.3f}"],
        ["p-value (one-sided)", f"{stats_data['p_value']:.3f}"],
        ["Significant at 5%?", "Yes" if stats_data['significant_5pct'] else "No"],
        ["Significant at 1%?", "Yes" if stats_data['significant_1pct'] else "No"],
        ["Number of switches", f"{stats_data['num_switches']}"]
    ]
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colWidths=[0.6, 0.4],
                    cellLoc='left',
                    loc='center',
                    bbox=[0.1, 0.1, 0.8, 0.7])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style header and cells
    for i in range(len(table_data)):
        # Left column (labels) - bold
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 0)].set_facecolor('#f0f0f0')
        
        # Right column (values)
        table[(i, 1)].set_facecolor('#ffffff')
        
        # Highlight significant results
        if i == 6 and stats_data['significant_5pct']:  # 5% significance
            table[(i, 1)].set_facecolor('#ffffcc')
            table[(i, 1)].set_text_props(weight='bold', color='green')
        elif i == 7 and stats_data['significant_1pct']:  # 1% significance
            table[(i, 1)].set_facecolor('#ffffcc')
            table[(i, 1)].set_text_props(weight='bold', color='green')
        
        # Add borders
        for j in range(2):
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)
    
    # Add title
    ax.text(0.5, 0.92, title, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='center', va='top')
    
    # Add separator line
    ax.text(0.5, 0.85, "─" * 60, transform=ax.transAxes,
            fontsize=10, ha='center', va='center', family='monospace')
    
    # Add bottom separator line
    ax.text(0.5, 0.05, "─" * 60, transform=ax.transAxes,
            fontsize=10, ha='center', va='center', family='monospace')
    
    # Add interpretation note
    interpretation = ""
    if stats_data['significant_5pct']:
        interpretation = f"✅ The model shows statistically significant predictive ability at 5% level"
    else:
        interpretation = f"❌ The model does not show statistically significant predictive ability"
    
    ax.text(0.5, 0.01, interpretation, transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom', style='italic',
            color='green' if stats_data['significant_5pct'] else 'red')
    
    # Save the table
    output_dir = Path("Data/Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Pesaran_Timmermann_Sign_Test_Table.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print(f"📊 Pesaran-Timmermann Sign Test Table created and saved as: {output_file.name}")
    print(f"   💾 Location: {output_file.absolute()}")
    
    # Print summary to console
    print(f"\n📈 PESARAN-TIMMERMANN SIGN TEST RESULTS:")
    print(f"   • Total predictions: {stats_data['total_predictions']}")
    print(f"   • Correct predictions: {stats_data['correct_predictions']} ({stats_data['proportion_correct']:.1f}%)")
    print(f"   • Expected under H₀: {stats_data['expected_proportion']:.1f}%")
    print(f"   • PT test statistic: {stats_data['pt_statistic']:.3f}")
    print(f"   • p-value (one-sided): {stats_data['p_value']:.3f}")
    print(f"   • Significant at 5%? {'✅ Yes' if stats_data['significant_5pct'] else '❌ No'}")
    print(f"   • Significant at 1%? {'✅ Yes' if stats_data['significant_1pct'] else '❌ No'}")
    print(f"   • Model switches: {stats_data['num_switches']} (always chose stocks)")
    
    plt.show()
    
    return stats_data

if __name__ == "__main__":
    create_pt_table_image()
