"""
Predict CE50 for AUC/Dose Dataset and Analyze Correlation
CORRECTED VERSION: Focus on CE50 (μM) vs AUC/Dose correlation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import joblib
import json
from pathlib import Path
from ce50_ensemble_predictor import DualFingerprintGenerator

print("\n" + "="*80)
print("CE50 PREDICTION FOR AUC/DOSE MOLECULES - CORRELATION ANALYSIS")
print("="*80)

# Load trained CE50 models
print("\nLoading trained CE50 ensemble models from models_300compounds/...")
model_dir = Path('models_300compounds')

# Load all 4 models
models = {}
for model_type in ['rf_binary', 'rf_count', 'xgb_binary', 'xgb_count']:
    model_file = list(model_dir.glob(f'{model_type}_*.pkl'))[0]
    models[model_type] = joblib.load(model_file)
    print(f"  Loaded {model_type}")

# Load applicability domain model
ad_file = list(model_dir.glob('applicability_domain_*.pkl'))[0]
applicability_domain = joblib.load(ad_file)
print(f"  Loaded applicability domain")

# Load AUC/Dose data
print("\nLoading AUC/Dose data from Excel file...")
df = pd.read_excel('CDD Excel Export -AUC-dose.xlsx')
print(f"Loaded {len(df)} compounds")

# Rename columns
df = df.rename(columns={
    'SMILES': 'smiles',
    'AUC/Dose': 'auc_dose'
})

# Remove missing values
df_valid = df.dropna(subset=['smiles', 'auc_dose']).copy()
print(f"Valid compounds (with both SMILES and AUC/Dose): {len(df_valid)}")

print(f"\nAUC/Dose Statistics:")
print(f"  Range: {df_valid['auc_dose'].min():.2f} - {df_valid['auc_dose'].max():.2f}")
print(f"  Mean: {df_valid['auc_dose'].mean():.2f} (±{df_valid['auc_dose'].std():.2f})")
print(f"  Median: {df_valid['auc_dose'].median():.2f}")

# Generate fingerprints for AUC/Dose molecules
print("\nGenerating dual fingerprints for AUC/Dose molecules...")
fp_generator = DualFingerprintGenerator()
fps = fp_generator.generate_both(df_valid['smiles'].values)

# Filter valid molecules
df_final = df_valid[fps['valid_mask']].copy()
X_binary = fps['binary']
X_count = fps['count']

print(f"Valid molecules after fingerprint generation: {len(df_final)}")

# Predict CE50 using all 4 models
print("\n" + "="*80)
print("PREDICTING CE50 VALUES")
print("="*80)

predictions_by_model = {}
for model_name, model in models.items():
    if 'binary' in model_name:
        preds = model.predict(X_binary)
    else:
        preds = model.predict(X_count)
    predictions_by_model[model_name] = preds

# Calculate ensemble prediction (mean of all models)
all_preds = np.array([predictions_by_model[m] for m in models.keys()])
ensemble_mean = np.mean(all_preds, axis=0)
ensemble_std = np.std(all_preds, axis=0)

# Convert model output to CE50 (μM)
# Model outputs -log10(CE50_μM), so CE50_μM = 10^(-model_output)
predicted_ce50 = 10 ** (-ensemble_mean)

# Assess applicability domain
print("\nAssessing applicability domain...")
confidence_scores = []
for i in range(len(X_binary)):
    ad_scores = applicability_domain.assess_all(X_binary[i], X_count[i])
    confidence_scores.append(ad_scores['overall_confidence'])

df_final['predicted_ce50'] = predicted_ce50
df_final['ensemble_std'] = ensemble_std
df_final['confidence'] = confidence_scores

# Log transform both for additional analysis
df_final['log_ce50'] = np.log10(df_final['predicted_ce50'])
df_final['log_auc_dose'] = np.log10(df_final['auc_dose'] + 0.001)  # +0.001 to handle very small values

# Correlation Analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Predicted CE50 (μM) vs Actual AUC/Dose")
print("="*80)

# Primary analysis: CE50 vs AUC/Dose on original scales
pearson_r_orig, pearson_p_orig = pearsonr(df_final['predicted_ce50'], df_final['auc_dose'])
spearman_r_orig, spearman_p_orig = spearmanr(df_final['predicted_ce50'], df_final['auc_dose'])

print(f"\nOriginal Scale (CE50 μM vs AUC/Dose):")
print(f"  Pearson r = {pearson_r_orig:.4f} (p = {pearson_p_orig:.4e})")
print(f"  Spearman r = {spearman_r_orig:.4f} (p = {spearman_p_orig:.4e})")

# Log-log scale analysis
pearson_r_log, pearson_p_log = pearsonr(df_final['log_ce50'], df_final['log_auc_dose'])
spearman_r_log, spearman_p_log = spearmanr(df_final['log_ce50'], df_final['log_auc_dose'])

print(f"\nLog-Log Scale (log(CE50) vs log(AUC/Dose)):")
print(f"  Pearson r = {pearson_r_log:.4f} (p = {pearson_p_log:.4e})")
print(f"  Spearman r = {spearman_r_log:.4f} (p = {spearman_p_log:.4e})")

# Inverse relationship (lower CE50 = more potent, might have higher AUC/Dose?)
pearson_r_inv, pearson_p_inv = pearsonr(-df_final['predicted_ce50'], df_final['auc_dose'])
spearman_r_inv, spearman_p_inv = spearmanr(-df_final['predicted_ce50'], df_final['auc_dose'])

print(f"\nInverse Relationship (Higher Potency vs AUC/Dose):")
print(f"  Pearson r (1/CE50 vs AUC/Dose) = {pearson_r_inv:.4f} (p = {pearson_p_inv:.4e})")
print(f"  Spearman r = {spearman_r_inv:.4f} (p = {spearman_p_inv:.4e})")

# Statistical significance interpretation
def interpret_pvalue(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: CE50 vs AUC/Dose (both on log scale for visibility)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df_final['predicted_ce50'], df_final['auc_dose'],
                       c=df_final['ensemble_std'], cmap='coolwarm',
                       s=100, alpha=0.7, edgecolors='k', linewidths=1.5)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Predicted CE50 (μM) - Lower = More Potent', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUC/Dose', fontsize=12, fontweight='bold')
ax1.set_title('Mass Spec Fragmentation vs Pharmacokinetic Exposure (Log-Log Scale)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Ensemble Std Dev')

textstr = f'Pearson r = {pearson_r_orig:.4f} {interpret_pvalue(pearson_p_orig)}\n'
textstr += f'Spearman ρ = {spearman_r_orig:.4f} {interpret_pvalue(spearman_p_orig)}\n'
textstr += f'p = {pearson_p_orig:.4e}'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Linear scale CE50 vs AUC/Dose
ax2 = axes[0, 1]
ax2.scatter(df_final['predicted_ce50'], df_final['auc_dose'],
           s=100, alpha=0.7, edgecolors='k', linewidths=1.5, color='steelblue')
ax2.set_xlabel('Predicted CE50 (μM) - Lower = More Potent', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUC/Dose', fontsize=12, fontweight='bold')
ax2.set_title('CE50 vs AUC/Dose (Linear Scale)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_final['predicted_ce50'], df_final['auc_dose'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_final['predicted_ce50'].min(), df_final['predicted_ce50'].max(), 100)
ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8,
         label=f'Linear fit: y={z[0]:.0f}x+{z[1]:.0f}')
ax2.legend()

textstr = f'Pearson r = {pearson_r_orig:.4f} {interpret_pvalue(pearson_p_orig)}\n'
textstr += f'Slope = {z[0]:.0f}\n'
textstr += f'p = {pearson_p_orig:.4e}'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 3: Distribution of predicted CE50 colored by AUC/Dose
ax3 = axes[1, 0]
scatter3 = ax3.scatter(range(len(df_final)), df_final['predicted_ce50'],
                       c=np.log10(df_final['auc_dose'] + 0.001), cmap='viridis',
                       s=100, alpha=0.7, edgecolors='k', linewidths=1.5)
ax3.axhline(df_final['predicted_ce50'].median(), color='red', linestyle='--',
           linewidth=2, label=f'Median CE50 = {df_final["predicted_ce50"].median():.2f} μM')
ax3.set_xlabel('Compound Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted CE50 (μM)', fontsize=12, fontweight='bold')
ax3.set_title('Predicted CE50 Distribution (Colored by AUC/Dose)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
cbar = plt.colorbar(scatter3, ax=ax3)
cbar.set_label('log(AUC/Dose)', fontsize=10)

# Plot 4: Confidence levels
ax4 = axes[1, 1]
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
confidence_numeric = [confidence_map[c] for c in df_final['confidence']]
colors = ['red' if c == 1 else 'orange' if c == 2 else 'green' for c in confidence_numeric]
ax4.scatter(df_final['predicted_ce50'], df_final['auc_dose'],
           c=colors, s=100, alpha=0.7, edgecolors='k', linewidths=1.5)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Predicted CE50 (μM)', fontsize=12, fontweight='bold')
ax4.set_ylabel('AUC/Dose', fontsize=12, fontweight='bold')
ax4.set_title('Predictions Colored by Confidence Level', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', edgecolor='k', label='High Confidence'),
                   Patch(facecolor='orange', edgecolor='k', label='Medium Confidence'),
                   Patch(facecolor='red', edgecolor='k', label='Low Confidence')]
ax4.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('ce50_vs_auc_dose_correlation_corrected.png', dpi=300, bbox_inches='tight')
print("Correlation plots saved to ce50_vs_auc_dose_correlation_corrected.png")
plt.close()

# Distribution comparison with better visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# CE50 distribution
ax1 = axes[0, 0]
ax1.hist(df_final['predicted_ce50'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Predicted CE50 (μM)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Predicted CE50 Values', fontsize=12, fontweight='bold')
ax1.axvline(df_final['predicted_ce50'].median(), color='red', linestyle='--',
           linewidth=2, label=f'Median = {df_final["predicted_ce50"].median():.2f} μM')
ax1.axvline(df_final['predicted_ce50'].mean(), color='green', linestyle='--',
           linewidth=2, label=f'Mean = {df_final["predicted_ce50"].mean():.2f} μM')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# AUC/Dose distribution (log scale)
ax2 = axes[0, 1]
ax2.hist(np.log10(df_final['auc_dose'] + 0.001), bins=20, edgecolor='black', alpha=0.7, color='coral')
ax2.set_xlabel('log(AUC/Dose)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of AUC/Dose Values (Log Scale)', fontsize=12, fontweight='bold')
ax2.axvline(np.log10(df_final['auc_dose'].median()), color='red', linestyle='--',
           linewidth=2, label=f'Median = {df_final["auc_dose"].median():.1f}')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Scatter with regression line
ax3 = axes[1, 0]
ax3.scatter(df_final['predicted_ce50'], df_final['auc_dose'],
           s=100, alpha=0.6, edgecolors='k', linewidths=1.5, color='purple')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Predicted CE50 (μM)', fontsize=11, fontweight='bold')
ax3.set_ylabel('AUC/Dose', fontsize=11, fontweight='bold')
ax3.set_title(f'Correlation: r = {pearson_r_orig:.3f}, p = {pearson_p_orig:.3f}',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Box plot by confidence level
ax4 = axes[1, 1]
conf_data = []
conf_labels = []
for conf in ['High', 'Medium', 'Low']:
    mask = df_final['confidence'] == conf
    if mask.any():
        conf_data.append(df_final[mask]['auc_dose'].values)
        conf_labels.append(f'{conf}\n(n={mask.sum()})')

bp = ax4.boxplot(conf_data, labels=conf_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'orange', 'red']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_ylabel('AUC/Dose', fontsize=11, fontweight='bold')
ax4.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
ax4.set_title('AUC/Dose Distribution by Prediction Confidence', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ce50_auc_dose_analysis_corrected.png', dpi=300, bbox_inches='tight')
print("Distribution and analysis plots saved to ce50_auc_dose_analysis_corrected.png")
plt.close()

# Save results to CSV
output_df = df_final[['smiles', 'auc_dose', 'predicted_ce50',
                      'ensemble_std', 'confidence']].copy()

# Add molecule names if available
if 'Molecule Name' in df.columns:
    molecule_names = []
    for idx in df_final.index:
        molecule_names.append(df.loc[idx, 'Molecule Name'])
    output_df.insert(0, 'molecule_name', molecule_names)

output_df.to_csv('ce50_predictions_for_auc_dose_corrected.csv', index=False)
print("\nPredictions saved to ce50_predictions_for_auc_dose_corrected.csv")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY - CE50 vs AUC/DOSE CORRELATION")
print("="*80)

print(f"\nDataset:")
print(f"  Total molecules analyzed: {len(df_final)}")
print(f"  Source: CDD Excel Export -AUC-dose.xlsx")

print(f"\nPredicted CE50 Statistics:")
print(f"  Range: {df_final['predicted_ce50'].min():.2f} - {df_final['predicted_ce50'].max():.2f} μM")
print(f"  Mean: {df_final['predicted_ce50'].mean():.2f} μM (±{df_final['predicted_ce50'].std():.2f})")
print(f"  Median: {df_final['predicted_ce50'].median():.2f} μM")
print(f"  Coefficient of Variation: {100*df_final['predicted_ce50'].std()/df_final['predicted_ce50'].mean():.1f}%")

print(f"\nAUC/Dose Statistics:")
print(f"  Range: {df_final['auc_dose'].min():.2f} - {df_final['auc_dose'].max():.2f}")
print(f"  Mean: {df_final['auc_dose'].mean():.2f} (±{df_final['auc_dose'].std():.2f})")
print(f"  Median: {df_final['auc_dose'].median():.2f}")
print(f"  Fold Range: {df_final['auc_dose'].max()/df_final['auc_dose'].min():.0f}-fold")

print(f"\nConfidence Distribution:")
for conf_level in ['High', 'Medium', 'Low']:
    count = sum(1 for c in df_final['confidence'] if c == conf_level)
    pct = 100 * count / len(df_final)
    print(f"  {conf_level:6s}: {count:3d} ({pct:.1f}%)")

print(f"\n" + "="*60)
print("CORRELATION RESULTS")
print("="*60)

print(f"\n{'Analysis Type':<40s} {'Pearson r':>12s} {'p-value':>12s} {'Sig':>5s}")
print("-" * 72)
print(f"{'CE50 vs AUC/Dose (original scale)':<40s} {pearson_r_orig:12.4f} {pearson_p_orig:12.4e} {interpret_pvalue(pearson_p_orig):>5s}")
print(f"{'log(CE50) vs log(AUC/Dose)':<40s} {pearson_r_log:12.4f} {pearson_p_log:12.4e} {interpret_pvalue(pearson_p_log):>5s}")
print(f"{'Inverse: -CE50 vs AUC/Dose':<40s} {pearson_r_inv:12.4f} {pearson_p_inv:12.4e} {interpret_pvalue(pearson_p_inv):>5s}")

print(f"\n{'Spearman (Rank-based) Correlations':<40s} {'Spearman ρ':>12s} {'p-value':>12s} {'Sig':>5s}")
print("-" * 72)
print(f"{'CE50 vs AUC/Dose (original scale)':<40s} {spearman_r_orig:12.4f} {spearman_p_orig:12.4e} {interpret_pvalue(spearman_p_orig):>5s}")
print(f"{'log(CE50) vs log(AUC/Dose)':<40s} {spearman_r_log:12.4f} {spearman_p_log:12.4e} {interpret_pvalue(spearman_p_log):>5s}")

print(f"\n{'='*72}")
if abs(pearson_r_orig) < 0.3 and pearson_p_orig > 0.05:
    print("CONCLUSION: No significant correlation between CE50 and AUC/Dose")
    print("Mass spectrometry fragmentation energy and pharmacokinetic exposure are independent properties.")
elif pearson_p_orig < 0.05:
    print(f"CONCLUSION: Significant correlation found (r={pearson_r_orig:.3f}, p={pearson_p_orig:.4f})")
    if pearson_r_orig > 0:
        print("Higher CE50 (lower potency) associated with higher AUC/Dose")
    else:
        print("Lower CE50 (higher potency) associated with higher AUC/Dose")
print(f"{'='*72}")

# Top compounds by CE50
print(f"\n{'Top 10 Lowest CE50 Compounds (Easiest Fragmentation) (Lowest CE50):'}")
print(f"{'Rank':<6s} {'Molecule':<20s} {'CE50 (μM)':>12s} {'AUC/Dose':>12s} {'Confidence':>10s}")
print("-" * 65)
top10 = df_final.nsmallest(10, 'predicted_ce50')
for rank, (idx, row) in enumerate(top10.iterrows(), 1):
    mol_name = df.loc[idx, 'Molecule Name'] if 'Molecule Name' in df.columns else f"Mol_{idx}"
    print(f"{rank:<6d} {mol_name:<20s} {row['predicted_ce50']:12.2f} {row['auc_dose']:12.2f} {row['confidence']:>10s}")

print(f"\n{'Top 10 Highest CE50 Compounds (Hardest Fragmentation) (Highest CE50):'}")
print(f"{'Rank':<6s} {'Molecule':<20s} {'CE50 (μM)':>12s} {'AUC/Dose':>12s} {'Confidence':>10s}")
print("-" * 65)
bottom10 = df_final.nlargest(10, 'predicted_ce50')
for rank, (idx, row) in enumerate(bottom10.iterrows(), 1):
    mol_name = df.loc[idx, 'Molecule Name'] if 'Molecule Name' in df.columns else f"Mol_{idx}"
    print(f"{rank:<6d} {mol_name:<20s} {row['predicted_ce50']:12.2f} {row['auc_dose']:12.2f} {row['confidence']:>10s}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")

print("Output files generated:")
print("  - ce50_vs_auc_dose_correlation_corrected.png")
print("  - ce50_auc_dose_analysis_corrected.png")
print("  - ce50_predictions_for_auc_dose_corrected.csv")
