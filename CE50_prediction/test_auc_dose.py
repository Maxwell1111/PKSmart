"""
Test CE50 Ensemble Model on AUC/Dose Dataset
Predict AUC/Dose from SMILES and analyze correlation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from ce50_ensemble_predictor import (
    DualFingerprintGenerator,
    EnsembleModel,
    Visualizer
)

print("\n" + "="*80)
print("AUC/DOSE PREDICTION FROM SMILES - ENSEMBLE TEST")
print("="*80)

# Load data
print("\nLoading data from Excel file...")
df = pd.read_excel('CDD Excel Export -AUC-dose.xlsx')

print(f"Loaded {len(df)} compounds")
print(f"Columns: {list(df.columns)}")

# Rename columns for consistency
df = df.rename(columns={
    'SMILES': 'smiles',
    'AUC/Dose': 'auc_dose'
})

# Remove missing AUC/Dose values
df_valid = df.dropna(subset=['smiles', 'auc_dose']).copy()
print(f"\nValid compounds (with AUC/Dose): {len(df_valid)} / {len(df)}")

# Convert AUC/Dose to log scale for better distribution
print("\nConverting AUC/Dose to log scale...")
df_valid['log_auc_dose'] = np.log10(df_valid['auc_dose'] + 1)  # +1 to handle zeros

print(f"\nAUC/Dose Statistics:")
print(f"  Range: {df_valid['auc_dose'].min():.2f} - {df_valid['auc_dose'].max():.2f}")
print(f"  Mean: {df_valid['auc_dose'].mean():.2f} (±{df_valid['auc_dose'].std():.2f})")
print(f"  Median: {df_valid['auc_dose'].median():.2f}")
print(f"\nLog(AUC/Dose) Statistics:")
print(f"  Range: {df_valid['log_auc_dose'].min():.4f} - {df_valid['log_auc_dose'].max():.4f}")
print(f"  Mean: {df_valid['log_auc_dose'].mean():.4f} (±{df_valid['log_auc_dose'].std():.4f})")

# Generate dual fingerprints
print("\nGenerating dual fingerprints...")
fp_generator = DualFingerprintGenerator()
fps = fp_generator.generate_both(df_valid['smiles'].values)

# Filter valid molecules
df_final = df_valid[fps['valid_mask']].copy()
X_binary = fps['binary']
X_count = fps['count']
y = df_final['log_auc_dose'].values

print(f"Valid molecules after fingerprint generation: {len(df_final)}")
print(f"Binary fingerprints shape: {X_binary.shape}")
print(f"Count fingerprints shape: {X_count.shape}")

if len(df_final) < 20:
    print("\n" + "⚠"*40)
    print("WARNING: Dataset too small (n<20) for reliable ML training!")
    print("Results will be exploratory only.")
    print("⚠"*40 + "\n")

# Create and train ensemble
print("\n" + "="*80)
print("TRAINING ENSEMBLE ON AUC/DOSE DATA")
print("="*80)
ensemble = EnsembleModel()
models, scores = ensemble.train_ensemble(X_binary, X_count, y)

# Evaluate ensemble
results = ensemble.evaluate_ensemble()

# Generate predictions for test set
test_smiles = df_final.iloc[len(y)-len(ensemble.y_test):]['smiles'].values
predictions = ensemble.predict_with_confidence(test_smiles)

# Convert predictions back to original scale
y_test_original = 10**ensemble.y_test - 1
y_pred_dict = {}
for model_name, res in results.items():
    y_pred_dict[model_name] = 10**res['predictions'] - 1

# Calculate correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Predicted vs Actual AUC/Dose")
print("="*80)

best_model = max(results.items(), key=lambda x: x[1]['r2'])
y_pred_best = y_pred_dict[best_model[0]]

pearson_r, pearson_p = pearsonr(y_test_original, y_pred_best)
spearman_r, spearman_p = spearmanr(y_test_original, y_pred_best)

print(f"\nBest Model: {best_model[0]}")
print(f"  R² (log scale): {best_model[1]['r2']:.4f}")
print(f"  Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4e})")
print(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Ensemble comparison (log scale)
viz = Visualizer()
viz.plot_ensemble_comparison(ensemble.y_test, results,
                             output_file='ensemble_auc_dose_comparison.png')

# Plot 2: Original scale predictions
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

for idx, (model_name, res) in enumerate(results.items()):
    ax = axes[idx]
    y_pred_orig = y_pred_dict[model_name]

    ax.scatter(y_test_original, y_pred_orig, alpha=0.6, s=80,
              edgecolors='k', linewidths=1)
    ax.plot([y_test_original.min(), y_test_original.max()],
           [y_test_original.min(), y_test_original.max()],
           'r--', lw=2, label='Perfect Prediction')

    # Log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Actual AUC/Dose', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted AUC/Dose', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name.upper()}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate correlation on original scale
    pr, _ = pearsonr(y_test_original, y_pred_orig)
    sr, _ = spearmanr(y_test_original, y_pred_orig)

    textstr = f'R² (log) = {res["r2"]:.4f}\nPearson r = {pr:.4f}\nSpearman r = {sr:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('auc_dose_predictions_original_scale.png', dpi=300, bbox_inches='tight')
print("Original scale predictions saved to auc_dose_predictions_original_scale.png")
plt.close()

# Plot 3: Confidence distribution
viz.plot_confidence_distribution(predictions,
                                 output_file='confidence_auc_dose_distribution.png')

# Save predictions to CSV
predictions_df = pd.DataFrame([{
    'molecule_name': df_final.iloc[len(y)-len(ensemble.y_test)+i].get('Molecule Name', f'Molecule_{i}'),
    'smiles': p['smiles'],
    'actual_auc_dose': df_final.iloc[len(y)-len(ensemble.y_test)+i]['auc_dose'],
    'actual_log_auc_dose': df_final.iloc[len(y)-len(ensemble.y_test)+i]['log_auc_dose'],
    'predicted_log_auc_dose': p['predicted_pce50'],  # Using pce50 field for log value
    'predicted_auc_dose': 10**p['predicted_pce50'] - 1,
    'confidence': p['confidence'],
    'selected_model': p['selected_model'],
    'ensemble_std': p['ensemble_std'],
    'tanimoto_binary': p['applicability_scores']['tanimoto_binary'],
    'tanimoto_count': p['applicability_scores']['tanimoto_count']
} for i, p in enumerate(predictions)])

predictions_df.to_csv('auc_dose_predictions.csv', index=False)
print("\nPredictions saved to auc_dose_predictions.csv")

# Print detailed summary
print("\n" + "="*80)
print("SUMMARY - AUC/DOSE PREDICTION")
print("="*80)

print(f"\nDataset:")
print(f"  Total compounds: {len(df)}")
print(f"  Valid for training: {len(df_final)}")
print(f"  Training set: {len(y) - len(ensemble.y_test)} compounds")
print(f"  Test set: {len(ensemble.y_test)} compounds")

print(f"\nBest performing model (by R² on log scale):")
print(f"  Model: {best_model[0]}")
print(f"  R² (log scale): {best_model[1]['r2']:.4f}")
print(f"  MAE (log scale): {best_model[1]['mae']:.4f}")
print(f"  RMSE (log scale): {best_model[1]['rmse']:.4f}")
print(f"  Pearson r (original): {pearson_r:.4f}")
print(f"  Spearman r (original): {spearman_r:.4f}")

print(f"\nAll model performance:")
for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    y_pred = y_pred_dict[model_name]
    pr, _ = pearsonr(y_test_original, y_pred)
    print(f"  {model_name:15s}: R²={metrics['r2']:.4f}, Pearson r={pr:.4f}")

print(f"\nConfidence distribution ({len(predictions)} test predictions):")
for conf_level in ['High', 'Medium', 'Low']:
    count = sum(1 for p in predictions if p['confidence'] == conf_level)
    pct = 100 * count / len(predictions) if len(predictions) > 0 else 0
    print(f"  {conf_level:6s}: {count:3d} ({pct:.1f}%)")

print(f"\nSample predictions (first 10):")
print(f"{'Molecule':<15s} {'Actual':>10s} {'Predicted':>12s} {'Error %':>10s} {'Conf':>6s}")
print("-" * 60)
for i in range(min(10, len(predictions_df))):
    row = predictions_df.iloc[i]
    error_pct = abs(row['predicted_auc_dose'] - row['actual_auc_dose']) / row['actual_auc_dose'] * 100
    print(f"{row['molecule_name']:<15s} {row['actual_auc_dose']:10.1f} {row['predicted_auc_dose']:12.1f} {error_pct:9.1f}% {row['confidence']:>6s}")

print("\n" + "="*80)
print("PIPELINE COMPLETED")
print("="*80 + "\n")

print("Output files generated:")
print("  - ensemble_auc_dose_comparison.png (log scale)")
print("  - auc_dose_predictions_original_scale.png")
print("  - confidence_auc_dose_distribution.png")
print("  - auc_dose_predictions.csv")
