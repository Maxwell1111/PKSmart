"""
Predict CE50 for AUC/Dose Dataset and Analyze Correlation
Uses trained CE50 ensemble to predict mass spectrometry fragmentation energy, then correlates with AUC/Dose
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

# Find most recent model files
model_files = list(model_dir.glob('*.pkl'))
if not model_files:
    raise FileNotFoundError("No model files found in models_300compounds/")

# Load all 4 models
models = {}
for model_type in ['rf_binary', 'rf_count', 'xgb_binary', 'xgb_count']:
    model_file = list(model_dir.glob(f'{model_type}_*.pkl'))[0]
    models[model_type] = joblib.load(model_file)
    print(f"  Loaded {model_type} from {model_file.name}")

# Load applicability domain model
ad_file = list(model_dir.glob('applicability_domain_*.pkl'))[0]
applicability_domain = joblib.load(ad_file)
print(f"  Loaded applicability domain from {ad_file.name}")

# Load metadata
metadata_file = list(model_dir.glob('metadata_*.json'))[0]
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
print(f"  Model timestamp: {metadata['timestamp']}")
print(f"  Best CV score: {max(metadata['model_scores'].values()):.4f}")

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
    print(f"  {model_name}: Generated {len(preds)} predictions")

# Calculate ensemble prediction (mean of all models)
all_preds = np.array([predictions_by_model[m] for m in models.keys()])
ensemble_mean = np.mean(all_preds, axis=0)
ensemble_std = np.std(all_preds, axis=0)

# Convert pCE50 to CE50 (μM)
predicted_ce50 = 10 ** (-ensemble_mean)

# Assess applicability domain
print("\nAssessing applicability domain...")
confidence_scores = []
for i in range(len(X_binary)):
    ad_scores = applicability_domain.assess_all(X_binary[i], X_count[i])
    confidence_scores.append(ad_scores['overall_confidence'])

df_final['predicted_pce50'] = ensemble_mean
df_final['predicted_ce50'] = predicted_ce50
df_final['ensemble_std'] = ensemble_std
df_final['confidence'] = confidence_scores

# Log transform AUC/Dose for better correlation analysis
df_final['log_auc_dose'] = np.log10(df_final['auc_dose'] + 1)

# Correlation Analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Predicted CE50 vs Actual AUC/Dose")
print("="*80)

# Correlation on original scales
pearson_r_orig, pearson_p_orig = pearsonr(df_final['predicted_ce50'], df_final['auc_dose'])
spearman_r_orig, spearman_p_orig = spearmanr(df_final['predicted_ce50'], df_final['auc_dose'])

print(f"\nOriginal Scale (CE50 vs AUC/Dose):")
print(f"  Pearson r = {pearson_r_orig:.4f} (p = {pearson_p_orig:.4e})")
print(f"  Spearman r = {spearman_r_orig:.4f} (p = {spearman_p_orig:.4e})")

# Correlation on log scales
pearson_r_log, pearson_p_log = pearsonr(df_final['predicted_pce50'], df_final['log_auc_dose'])
spearman_r_log, spearman_p_log = spearmanr(df_final['predicted_pce50'], df_final['log_auc_dose'])

print(f"\nLog Scale (pCE50 vs log(AUC/Dose)):")
print(f"  Pearson r = {pearson_r_log:.4f} (p = {pearson_p_log:.4e})")
print(f"  Spearman r = {spearman_r_log:.4f} (p = {spearman_p_log:.4e})")

# Inverse relationship (lower CE50 = higher potency might mean higher AUC/Dose)
# Try negative correlation
pearson_r_inv, pearson_p_inv = pearsonr(-df_final['predicted_pce50'], df_final['log_auc_dose'])
spearman_r_inv, spearman_p_inv = spearmanr(-df_final['predicted_pce50'], df_final['log_auc_dose'])

print(f"\nInverse Relationship (-pCE50 vs log(AUC/Dose)):")
print(f"  Pearson r = {pearson_r_inv:.4f} (p = {pearson_p_inv:.4e})")
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

# Plot 1: pCE50 vs log(AUC/Dose)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df_final['predicted_pce50'], df_final['log_auc_dose'],
                       c=df_final['ensemble_std'], cmap='coolwarm',
                       s=80, alpha=0.7, edgecolors='k', linewidths=1)
ax1.set_xlabel('Predicted pCE50 (Higher = More Potent)', fontsize=12, fontweight='bold')
ax1.set_ylabel('log(AUC/Dose)', fontsize=12, fontweight='bold')
ax1.set_title('Predicted Mass Spec Fragmentation vs Pharmacokinetics (Log Scale)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Ensemble Std Dev')

# Add correlation info
textstr = f'Pearson r = {pearson_r_log:.4f} {interpret_pvalue(pearson_p_log)}\n'
textstr += f'Spearman ρ = {spearman_r_log:.4f} {interpret_pvalue(spearman_p_log)}\n'
textstr += f'p = {pearson_p_log:.4e}'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: CE50 vs AUC/Dose (log-log scale)
ax2 = axes[0, 1]
ax2.scatter(df_final['predicted_ce50'], df_final['auc_dose'],
           s=80, alpha=0.7, edgecolors='k', linewidths=1, color='steelblue')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Predicted CE50 (μM) [Lower = More Potent]', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUC/Dose', fontsize=12, fontweight='bold')
ax2.set_title('Mass Spec Fragmentation vs Pharmacokinetics (Log-Log Scale)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

textstr = f'Pearson r = {pearson_r_orig:.4f} {interpret_pvalue(pearson_p_orig)}\n'
textstr += f'Spearman ρ = {spearman_r_orig:.4f} {interpret_pvalue(spearman_p_orig)}\n'
textstr += f'p = {pearson_p_orig:.4e}'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 3: Inverse relationship (-pCE50 vs log AUC/Dose)
ax3 = axes[1, 0]
ax3.scatter(-df_final['predicted_pce50'], df_final['log_auc_dose'],
           s=80, alpha=0.7, edgecolors='k', linewidths=1, color='crimson')
ax3.set_xlabel('-pCE50 (Higher = Less Potent)', fontsize=12, fontweight='bold')
ax3.set_ylabel('log(AUC/Dose)', fontsize=12, fontweight='bold')
ax3.set_title('Inverse Potency vs Pharmacokinetics', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(-df_final['predicted_pce50'], df_final['log_auc_dose'], 1)
p = np.poly1d(z)
x_trend = np.linspace(-df_final['predicted_pce50'].min(), -df_final['predicted_pce50'].max(), 100)
ax3.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
ax3.legend()

textstr = f'Pearson r = {pearson_r_inv:.4f} {interpret_pvalue(pearson_p_inv)}\n'
textstr += f'Spearman ρ = {spearman_r_inv:.4f} {interpret_pvalue(spearman_p_inv)}\n'
textstr += f'p = {pearson_p_inv:.4e}'
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 4: Model agreement (ensemble std vs confidence)
ax4 = axes[1, 1]
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
confidence_numeric = [confidence_map[c] for c in df_final['confidence']]
scatter4 = ax4.scatter(confidence_numeric, df_final['ensemble_std'],
                      c=df_final['log_auc_dose'], cmap='viridis',
                      s=100, alpha=0.7, edgecolors='k', linewidths=1)
ax4.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
ax4.set_ylabel('Ensemble Standard Deviation (pCE50)', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Confidence vs Model Agreement', fontsize=13, fontweight='bold')
ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(['Low', 'Medium', 'High'])
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter4, ax=ax4, label='log(AUC/Dose)')

plt.tight_layout()
plt.savefig('ce50_vs_auc_dose_correlation.png', dpi=300, bbox_inches='tight')
print("Correlation plots saved to ce50_vs_auc_dose_correlation.png")
plt.close()

# Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of predicted CE50
ax1 = axes[0]
ax1.hist(df_final['predicted_ce50'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Predicted CE50 (μM)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Predicted CE50 Values', fontsize=12, fontweight='bold')
ax1.axvline(df_final['predicted_ce50'].median(), color='red', linestyle='--',
           linewidth=2, label=f'Median = {df_final["predicted_ce50"].median():.2f} μM')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Distribution of AUC/Dose
ax2 = axes[1]
ax2.hist(df_final['auc_dose'], bins=20, edgecolor='black', alpha=0.7, color='coral')
ax2.set_xlabel('AUC/Dose', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Actual AUC/Dose Values', fontsize=12, fontweight='bold')
ax2.axvline(df_final['auc_dose'].median(), color='blue', linestyle='--',
           linewidth=2, label=f'Median = {df_final["auc_dose"].median():.2f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ce50_auc_dose_distributions.png', dpi=300, bbox_inches='tight')
print("Distribution plots saved to ce50_auc_dose_distributions.png")
plt.close()

# Save results to CSV
output_df = df_final[['smiles', 'auc_dose', 'log_auc_dose',
                      'predicted_ce50', 'predicted_pce50',
                      'ensemble_std', 'confidence']].copy()

# Add molecule names if available
if 'Molecule Name' in df.columns:
    molecule_names = []
    for idx in df_final.index:
        molecule_names.append(df.loc[idx, 'Molecule Name'])
    output_df.insert(0, 'molecule_name', molecule_names)

output_df.to_csv('ce50_predictions_for_auc_dose.csv', index=False)
print("\nPredictions saved to ce50_predictions_for_auc_dose.csv")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY - CE50 vs AUC/DOSE CORRELATION")
print("="*80)

print(f"\nDataset:")
print(f"  Total molecules analyzed: {len(df_final)}")
print(f"  Molecules from: CDD Excel Export -AUC-dose.xlsx")

print(f"\nPredicted CE50 Statistics:")
print(f"  Range: {df_final['predicted_ce50'].min():.2f} - {df_final['predicted_ce50'].max():.2f} μM")
print(f"  Mean: {df_final['predicted_ce50'].mean():.2f} μM (±{df_final['predicted_ce50'].std():.2f})")
print(f"  Median: {df_final['predicted_ce50'].median():.2f} μM")

print(f"\nConfidence Distribution:")
for conf_level in ['High', 'Medium', 'Low']:
    count = sum(1 for c in df_final['confidence'] if c == conf_level)
    pct = 100 * count / len(df_final)
    print(f"  {conf_level:6s}: {count:3d} ({pct:.1f}%)")

print(f"\nCorrelation Results:")
print(f"  {'Comparison':<35s} {'Pearson r':>12s} {'p-value':>12s} {'Sig':>5s}")
print("-" * 70)
print(f"  {'pCE50 vs log(AUC/Dose)':<35s} {pearson_r_log:12.4f} {pearson_p_log:12.4e} {interpret_pvalue(pearson_p_log):>5s}")
print(f"  {'CE50 vs AUC/Dose (original)':<35s} {pearson_r_orig:12.4f} {pearson_p_orig:12.4e} {interpret_pvalue(pearson_p_orig):>5s}")
print(f"  {'-pCE50 vs log(AUC/Dose) [inverse]':<35s} {pearson_r_inv:12.4f} {pearson_p_inv:12.4e} {interpret_pvalue(pearson_p_inv):>5s}")

print(f"\nSpearman Correlations (rank-based):")
print(f"  {'Comparison':<35s} {'Spearman ρ':>12s} {'p-value':>12s} {'Sig':>5s}")
print("-" * 70)
print(f"  {'pCE50 vs log(AUC/Dose)':<35s} {spearman_r_log:12.4f} {spearman_p_log:12.4e} {interpret_pvalue(spearman_p_log):>5s}")
print(f"  {'CE50 vs AUC/Dose (original)':<35s} {spearman_r_orig:12.4f} {spearman_p_orig:12.4e} {interpret_pvalue(spearman_p_orig):>5s}")
print(f"  {'-pCE50 vs log(AUC/Dose) [inverse]':<35s} {spearman_r_inv:12.4f} {spearman_p_inv:12.4e} {interpret_pvalue(spearman_p_inv):>5s}")

print(f"\n{'Top 10 Lowest CE50 Compounds (Easiest Fragmentation) (Lowest Predicted CE50):'}")
print(f"{'Rank':<6s} {'Molecule':<20s} {'Pred CE50':>12s} {'AUC/Dose':>12s} {'Confidence':>10s}")
print("-" * 65)
top10 = df_final.nsmallest(10, 'predicted_ce50')
for rank, (idx, row) in enumerate(top10.iterrows(), 1):
    mol_name = df.loc[idx, 'Molecule Name'] if 'Molecule Name' in df.columns else f"Mol_{idx}"
    print(f"{rank:<6d} {mol_name:<20s} {row['predicted_ce50']:12.2f} {row['auc_dose']:12.2f} {row['confidence']:>10s}")

print(f"\n{'Top 10 Highest CE50 Compounds (Hardest Fragmentation) (Highest Predicted CE50):'}")
print(f"{'Rank':<6s} {'Molecule':<20s} {'Pred CE50':>12s} {'AUC/Dose':>12s} {'Confidence':>10s}")
print("-" * 65)
bottom10 = df_final.nlargest(10, 'predicted_ce50')
for rank, (idx, row) in enumerate(bottom10.iterrows(), 1):
    mol_name = df.loc[idx, 'Molecule Name'] if 'Molecule Name' in df.columns else f"Mol_{idx}"
    print(f"{rank:<6d} {mol_name:<20s} {row['predicted_ce50']:12.2f} {row['auc_dose']:12.2f} {row['confidence']:>10s}")

print("\n" + "="*80)
print("PIPELINE COMPLETED")
print("="*80 + "\n")

print("Output files generated:")
print("  - ce50_vs_auc_dose_correlation.png (4-panel correlation analysis)")
print("  - ce50_auc_dose_distributions.png (distribution comparison)")
print("  - ce50_predictions_for_auc_dose.csv (full predictions)")
