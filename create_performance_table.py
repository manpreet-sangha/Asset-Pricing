import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
header_color = '#2E86AB'
section_color = '#A23B72'
positive_color = '#F18F01'
neutral_color = '#C73E1D'

# Title
title_box = FancyBboxPatch((0.5, 9), 9, 0.8, boxstyle="round,pad=0.1", 
                          facecolor=header_color, edgecolor='black', linewidth=2)
ax.add_patch(title_box)
ax.text(5, 9.4, '🏆 STRATEGY PERFORMANCE EVALUATION', 
        fontsize=18, fontweight='bold', ha='center', va='center', color='white')
ax.text(5, 9.1, 'Pesaran & Timmermann Multi-Predictor Model (Dividend Yield + CPI)', 
        fontsize=12, ha='center', va='center', color='white')

# Section 1: Prediction Accuracy
y_pos = 8.2
section1_box = FancyBboxPatch((0.5, y_pos-0.4), 9, 1.2, boxstyle="round,pad=0.05", 
                             facecolor='#E8F4F8', edgecolor=section_color, linewidth=1.5)
ax.add_patch(section1_box)
ax.text(1, y_pos+0.4, 'PREDICTION ACCURACY', fontsize=14, fontweight='bold', color=section_color)
ax.text(1.2, y_pos+0.1, '🎯 Correct predictions:', fontsize=11, color='black')
ax.text(7, y_pos+0.1, '16/20 (80.0%)', fontsize=11, fontweight='bold', color=positive_color)
ax.text(1.2, y_pos-0.1, '🏆 Win periods (outperformed):', fontsize=11, color='black')
ax.text(7, y_pos-0.1, '2/20 (10.0%)', fontsize=11, fontweight='bold', color=neutral_color)
ax.text(1.2, y_pos-0.3, '📊 Average outperformance:', fontsize=11, color='black')
ax.text(7, y_pos-0.3, '0.57% per period', fontsize=11, fontweight='bold', color=positive_color)

# Section 2: Strategy Performance
y_pos = 6.5
section2_box = FancyBboxPatch((0.5, y_pos-0.5), 4.2, 1.4, boxstyle="round,pad=0.05", 
                             facecolor='#F0F8E8', edgecolor=section_color, linewidth=1.5)
ax.add_patch(section2_box)
ax.text(1, y_pos+0.6, 'SWITCHING STRATEGY', fontsize=12, fontweight='bold', color=section_color)
ax.text(1.2, y_pos+0.3, '💰 Total return:', fontsize=10, color='black')
ax.text(3.8, y_pos+0.3, '110.1%', fontsize=10, fontweight='bold', color=positive_color)
ax.text(1.2, y_pos+0.1, '📈 Annualized return:', fontsize=10, color='black')
ax.text(3.8, y_pos+0.1, '7.7%', fontsize=10, fontweight='bold', color=positive_color)
ax.text(1.2, y_pos-0.1, '📊 Volatility:', fontsize=10, color='black')
ax.text(3.8, y_pos-0.1, '11.9%', fontsize=10, fontweight='bold', color='black')
ax.text(1.2, y_pos-0.3, '⚡ Sharpe ratio:', fontsize=10, color='black')
ax.text(3.8, y_pos-0.3, '0.488', fontsize=10, fontweight='bold', color=positive_color)

# Section 3: Benchmark Performance
section3_box = FancyBboxPatch((5.3, y_pos-0.5), 4.2, 1.4, boxstyle="round,pad=0.05", 
                             facecolor='#FFF8E8', edgecolor=section_color, linewidth=1.5)
ax.add_patch(section3_box)
ax.text(5.8, y_pos+0.6, 'BUY-AND-HOLD BENCHMARK', fontsize=12, fontweight='bold', color=section_color)
ax.text(6, y_pos+0.3, '💰 Total return:', fontsize=10, color='black')
ax.text(8.6, y_pos+0.3, '85.7%', fontsize=10, fontweight='bold', color='black')
ax.text(6, y_pos+0.1, '📈 Annualized return:', fontsize=10, color='black')
ax.text(8.6, y_pos+0.1, '6.4%', fontsize=10, fontweight='bold', color='black')
ax.text(6, y_pos-0.1, '📊 Volatility:', fontsize=10, color='black')
ax.text(8.6, y_pos-0.1, '13.0%', fontsize=10, fontweight='bold', color='black')
ax.text(6, y_pos-0.3, '⚡ Sharpe ratio:', fontsize=10, color='black')
ax.text(8.6, y_pos-0.3, '0.345', fontsize=10, fontweight='bold', color='black')

# Section 4: Strategy Comparison
y_pos = 4.5
section4_box = FancyBboxPatch((0.5, y_pos-0.4), 9, 1, boxstyle="round,pad=0.05", 
                             facecolor='#E8F8E8', edgecolor=section_color, linewidth=1.5)
ax.add_patch(section4_box)
ax.text(1, y_pos+0.3, 'STRATEGY COMPARISON', fontsize=14, fontweight='bold', color=section_color)
ax.text(1.2, y_pos, '🚀 Return advantage:', fontsize=11, color='black')
ax.text(4.5, y_pos, '24.3%', fontsize=11, fontweight='bold', color=positive_color)
ax.text(6, y_pos, '🎯 Sharpe advantage:', fontsize=11, color='black')
ax.text(8.5, y_pos, '0.143', fontsize=11, fontweight='bold', color=positive_color)

# Section 5: Results Summary
y_pos = 3.2
section5_box = FancyBboxPatch((0.5, y_pos-0.6), 9, 1.4, boxstyle="round,pad=0.05", 
                             facecolor='#F8F8F8', edgecolor=section_color, linewidth=1.5)
ax.add_patch(section5_box)
ax.text(1, y_pos+0.5, 'ANALYSIS SUMMARY', fontsize=14, fontweight='bold', color=section_color)
ax.text(1.2, y_pos+0.2, '💾 Results saved to: SMM265_DivYield_CPI_Strategy_Evaluation_Results.xlsx', fontsize=10, color='black')
ax.text(1.2, y_pos, '📅 All dates in date-only format (no timestamps)', fontsize=10, color='black')
ax.text(1.2, y_pos-0.2, '🎉 Rolling window analysis complete: 20 predictions generated', fontsize=10, color='black')
ax.text(1.2, y_pos-0.4, '📈 Coefficient evolution demonstrates market adaptability', fontsize=10, color='black')

# Success indicator
y_pos = 1.5
success_box = FancyBboxPatch((2, y_pos-0.4), 6, 0.8, boxstyle="round,pad=0.1", 
                            facecolor=positive_color, edgecolor='black', linewidth=2)
ax.add_patch(success_box)
ax.text(5, y_pos, '✅ STRATEGY SUCCESS: Multi-predictor outperformed buy-and-hold', 
        fontsize=12, fontweight='bold', ha='center', va='center', color='white')

# Footer
ax.text(5, 0.5, 'SMM265 Asset Pricing - Pesaran & Timmermann (1995) Implementation', 
        fontsize=10, ha='center', va='center', color='gray', style='italic')

plt.tight_layout()

# Save the figure
filename = os.path.join(output_dir, 'SMM265_DivYield_CPI_Strategy_Performance_Table.png')
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"✅ Performance table saved to: {filename}")
