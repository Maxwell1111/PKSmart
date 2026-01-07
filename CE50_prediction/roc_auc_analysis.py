"""
ROC-AUC Analysis: CE50 as Predictor of Pharmacokinetic Outcomes

This analysis evaluates the predictive power of CE50 (mass spec fragmentation)
for classifying compounds as having favorable vs poor pharmacokinetic profiles.

Based on the original 2016 discovery:
- Lower CE50 → Better PK (higher AUC/Dose)
- Higher CE50 → Poor PK (lower AUC/Dose)

Thresholds:
- CE50: 21.1 eV (Good/Stable if >= 21.1, Poor/Unstable if < 21.1)
- AUC/Dose: 300 (Success if >= 300, Failure if < 300)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n" + "="*80)
print("ROC-AUC ANALYSIS: CE50 AS PREDICTOR OF PHARMACOKINETIC OUTCOMES")
print("="*80)

# Load data
print("\nLoading ML-predicted CE50 data...")
df = pd.read_csv('ce50_predictions_for_auc_dose_corrected.csv')
print(f"Loaded {len(df)} compounds")

# Extract relevant columns
ce50 = df['predicted_ce50'].values
auc_dose = df['auc_dose'].values

print(f"\nData Summary:")
print(f"  CE50 range: {ce50.min():.2f} - {ce50.max():.2f} eV")
print(f"  AUC/Dose range: {auc_dose.min():.2f} - {auc_dose.max():.2f}")

# Define thresholds
CE50_THRESHOLD = 21.1  # eV
AUC_DOSE_THRESHOLD = 300  # Success/Failure cutoff

print(f"\n" + "="*80)
print("THRESHOLDS DEFINED")
print("="*80)
print(f"  CE50 Threshold: {CE50_THRESHOLD} eV")
print(f"    - 'Good/Stable' if CE50 >= {CE50_THRESHOLD} eV")
print(f"    - 'Poor/Unstable' if CE50 < {CE50_THRESHOLD} eV")
print(f"\n  AUC/Dose Threshold: {AUC_DOSE_THRESHOLD}")
print(f"    - 'Success' if AUC/Dose >= {AUC_DOSE_THRESHOLD}")
print(f"    - 'Failure' if AUC/Dose < {AUC_DOSE_THRESHOLD}")

# Create binary classifications
# For CE50: According to original discovery, LOWER CE50 = BETTER PK
# So we need to INVERT the CE50 classification for ROC
# Predictor: 1 if CE50 < 21.1 (predicting good PK), 0 if CE50 >= 21.1 (predicting poor PK)
y_pred_binary = (ce50 < CE50_THRESHOLD).astype(int)

# Ground truth: 1 if AUC/Dose >= 300 (actual good PK), 0 otherwise
y_true = (auc_dose >= AUC_DOSE_THRESHOLD).astype(int)

# For ROC curve, we use the continuous CE50 values
# Since lower CE50 = better PK, we use negative CE50 as the score
# (higher score = better predicted outcome)
y_score = -ce50

print(f"\n" + "="*80)
print("CLASSIFICATION DISTRIBUTION")
print("="*80)

print(f"\nPredicted (based on CE50 < {CE50_THRESHOLD}):")
print(f"  Predicted Good PK: {y_pred_binary.sum()} compounds ({100*y_pred_binary.mean():.1f}%)")
print(f"  Predicted Poor PK: {(1-y_pred_binary).sum()} compounds ({100*(1-y_pred_binary.mean()):.1f}%)")

print(f"\nActual (based on AUC/Dose >= {AUC_DOSE_THRESHOLD}):")
print(f"  Actual Good PK: {y_true.sum()} compounds ({100*y_true.mean():.1f}%)")
print(f"  Actual Poor PK: {(1-y_true).sum()} compounds ({100*(1-y_true.mean()):.1f}%)")

# Confusion Matrix
print(f"\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_true, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

print(f"\n                    Predicted Poor PK  |  Predicted Good PK")
print(f"                    (CE50 >= {CE50_THRESHOLD})     |  (CE50 < {CE50_THRESHOLD})")
print(f"  --------------------------------------------------")
print(f"  Actual Poor PK    |        {tn:3d}        |       {fp:3d}")
print(f"  (AUC/Dose < {AUC_DOSE_THRESHOLD}) |   True Negative   |  False Positive")
print(f"  --------------------------------------------------")
print(f"  Actual Good PK    |        {fn:3d}        |       {tp:3d}")
print(f"  (AUC/Dose >= {AUC_DOSE_THRESHOLD})|  False Negative   |  True Positive")

# Calculate metrics at fixed threshold
accuracy = accuracy_score(y_true, y_pred_binary)
sensitivity = recall_score(y_true, y_pred_binary)  # True Positive Rate
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
precision = precision_score(y_true, y_pred_binary, zero_division=0)
f1 = f1_score(y_true, y_pred_binary, zero_division=0)

print(f"\n" + "="*80)
print(f"PERFORMANCE AT FIXED THRESHOLD (CE50 = {CE50_THRESHOLD} eV)")
print("="*80)
print(f"\n  Accuracy:     {accuracy:.4f} ({100*accuracy:.1f}%)")
print(f"  Sensitivity:  {sensitivity:.4f} (True Positive Rate)")
print(f"  Specificity:  {specificity:.4f} (True Negative Rate)")
print(f"  Precision:    {precision:.4f} (Positive Predictive Value)")
print(f"  F1 Score:     {f1:.4f}")

# ROC Curve Analysis
print(f"\n" + "="*80)
print("ROC CURVE ANALYSIS")
print("="*80)

fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print(f"\n  ROC-AUC Score: {roc_auc:.4f}")

# Find optimal threshold using Youden's J statistic
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold_score = thresholds_roc[optimal_idx]
optimal_ce50 = -optimal_threshold_score  # Convert back to CE50

optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]
optimal_youden_j = youden_j[optimal_idx]

print(f"\n  Optimal Cutoff (Youden's J):")
print(f"    CE50 Threshold: {optimal_ce50:.2f} eV")
print(f"    Youden's J:     {optimal_youden_j:.4f}")
print(f"    Sensitivity:    {optimal_sensitivity:.4f} ({100*optimal_sensitivity:.1f}%)")
print(f"    Specificity:    {optimal_specificity:.4f} ({100*optimal_specificity:.1f}%)")

# Recalculate accuracy at optimal threshold
y_pred_optimal = (ce50 < optimal_ce50).astype(int)
optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
print(f"    Accuracy:       {optimal_accuracy:.4f} ({100*optimal_accuracy:.1f}%)")

# Classification report
print(f"\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(f"\nAt Fixed Threshold (CE50 = {CE50_THRESHOLD} eV):")
print(classification_report(y_true, y_pred_binary,
                          target_names=['Poor PK', 'Good PK'],
                          digits=4))

# Generate visualizations
print(f"\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: ROC Curve
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr, tpr, color='darkblue', lw=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')

# Mark optimal point
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
        label=f'Optimal Cutoff (CE50 < {optimal_ce50:.2f} eV)')

# Mark fixed threshold point
fixed_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fixed_tpr = sensitivity
ax.plot(fixed_fpr, fixed_tpr, 'gs', markersize=12,
        label=f'Fixed Threshold (CE50 < {CE50_THRESHOLD} eV)')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve: CE50 as Predictor of Pharmacokinetic Outcomes',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig('roc_curve_ce50_pk.png', dpi=300, bbox_inches='tight')
print("ROC curve saved to roc_curve_ce50_pk.png")
plt.close()

# Figure 2: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Predicted Poor PK\n(CE50 >= {CE50_THRESHOLD})',
                         f'Predicted Good PK\n(CE50 < {CE50_THRESHOLD})'],
            yticklabels=[f'Actual Poor PK\n(AUC/Dose < {AUC_DOSE_THRESHOLD})',
                         f'Actual Good PK\n(AUC/Dose >= {AUC_DOSE_THRESHOLD})'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_title(f'Confusion Matrix: CE50 Classification at {CE50_THRESHOLD} eV Threshold',
             fontsize=13, fontweight='bold', pad=20)

# Add performance metrics as text
textstr = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPrecision: {precision:.3f}'
ax.text(1.5, -0.15, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('confusion_matrix_ce50_pk.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to confusion_matrix_ce50_pk.png")
plt.close()

# Figure 3: Scatter plot with threshold lines
fig, ax = plt.subplots(figsize=(12, 9))

# Define quadrants
q1 = (ce50 >= CE50_THRESHOLD) & (auc_dose >= AUC_DOSE_THRESHOLD)  # High CE50, High AUC - Incorrect
q2 = (ce50 < CE50_THRESHOLD) & (auc_dose >= AUC_DOSE_THRESHOLD)   # Low CE50, High AUC - Correct
q3 = (ce50 < CE50_THRESHOLD) & (auc_dose < AUC_DOSE_THRESHOLD)    # Low CE50, Low AUC - Incorrect
q4 = (ce50 >= CE50_THRESHOLD) & (auc_dose < AUC_DOSE_THRESHOLD)   # High CE50, Low AUC - Correct

# Plot quadrants with different colors
ax.scatter(ce50[q1], auc_dose[q1], c='orange', s=100, alpha=0.7,
          edgecolors='k', linewidths=1.5, label=f'Q1: High CE50, Good PK (n={q1.sum()}) - FN')
ax.scatter(ce50[q2], auc_dose[q2], c='green', s=100, alpha=0.7,
          edgecolors='k', linewidths=1.5, label=f'Q2: Low CE50, Good PK (n={q2.sum()}) - TP')
ax.scatter(ce50[q3], auc_dose[q3], c='red', s=100, alpha=0.7,
          edgecolors='k', linewidths=1.5, label=f'Q3: Low CE50, Poor PK (n={q3.sum()}) - FP')
ax.scatter(ce50[q4], auc_dose[q4], c='blue', s=100, alpha=0.7,
          edgecolors='k', linewidths=1.5, label=f'Q4: High CE50, Poor PK (n={q4.sum()}) - TN')

# Add threshold lines
ax.axvline(CE50_THRESHOLD, color='red', linestyle='--', linewidth=2.5,
          label=f'CE50 Threshold = {CE50_THRESHOLD} eV', alpha=0.8)
ax.axhline(AUC_DOSE_THRESHOLD, color='blue', linestyle='--', linewidth=2.5,
          label=f'AUC/Dose Threshold = {AUC_DOSE_THRESHOLD}', alpha=0.8)

# Add optimal threshold line
ax.axvline(optimal_ce50, color='purple', linestyle=':', linewidth=2,
          label=f'Optimal CE50 = {optimal_ce50:.2f} eV', alpha=0.8)

ax.set_xlabel('Predicted CE50 (eV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual AUC/Dose', fontsize=12, fontweight='bold')
ax.set_title('CE50 vs AUC/Dose: Classification Quadrants', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale for better visualization

# Add quadrant labels
mid_ce50 = (ce50.min() + CE50_THRESHOLD) / 2
high_ce50 = (CE50_THRESHOLD + ce50.max()) / 2
mid_auc = np.exp((np.log(auc_dose.min() + 0.1) + np.log(AUC_DOSE_THRESHOLD)) / 2)
high_auc = np.exp((np.log(AUC_DOSE_THRESHOLD) + np.log(auc_dose.max())) / 2)

ax.text(high_ce50, high_auc, 'Q1\nFalse Neg', fontsize=10, ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(mid_ce50, high_auc, 'Q2\nTrue Pos', fontsize=10, ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
ax.text(mid_ce50, mid_auc, 'Q3\nFalse Pos', fontsize=10, ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(high_ce50, mid_auc, 'Q4\nTrue Neg', fontsize=10, ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

plt.tight_layout()
plt.savefig('ce50_auc_dose_quadrants.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved to ce50_auc_dose_quadrants.png")
plt.close()

# Save results to CSV
results_df = pd.DataFrame({
    'molecule_name': df['molecule_name'],
    'ce50': ce50,
    'auc_dose': auc_dose,
    'predicted_good_pk': y_pred_binary,
    'actual_good_pk': y_true,
    'correct_classification': (y_pred_binary == y_true),
    'quadrant': np.where(q1, 'Q1_FN', np.where(q2, 'Q2_TP', np.where(q3, 'Q3_FP', 'Q4_TN')))
})

results_df.to_csv('roc_classification_results.csv', index=False)
print("Classification results saved to roc_classification_results.csv")

# Summary Statistics
print(f"\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nQuadrant Distribution:")
print(f"  Q1 (High CE50, Good PK - False Negative): {q1.sum():3d} ({100*q1.sum()/len(df):5.1f}%)")
print(f"  Q2 (Low CE50, Good PK - True Positive):   {q2.sum():3d} ({100*q2.sum()/len(df):5.1f}%)")
print(f"  Q3 (Low CE50, Poor PK - False Positive):  {q3.sum():3d} ({100*q3.sum()/len(df):5.1f}%)")
print(f"  Q4 (High CE50, Poor PK - True Negative):  {q4.sum():3d} ({100*q4.sum()/len(df):5.1f}%)")

correct = (q2.sum() + q4.sum())
total = len(df)
print(f"\n  Correct Classifications (Q2 + Q4): {correct}/{total} ({100*correct/total:.1f}%)")
print(f"  Incorrect Classifications (Q1 + Q3): {total-correct}/{total} ({100*(total-correct)/total:.1f}%)")

print(f"\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if roc_auc > 0.7:
    interpretation = "Good discriminative ability"
elif roc_auc > 0.6:
    interpretation = "Moderate discriminative ability"
elif roc_auc > 0.5:
    interpretation = "Poor discriminative ability"
else:
    interpretation = "No discriminative ability (worse than random)"

print(f"\n  ROC-AUC = {roc_auc:.4f}: {interpretation}")

if roc_auc > 0.5:
    print(f"\n  ✓ CE50 shows predictive power for PK outcomes")
    print(f"  ✓ ML-predicted CE50 can classify compounds better than random")
else:
    print(f"\n  ✗ CE50 does not show predictive power for PK outcomes")
    print(f"  ✗ ML-predicted CE50 cannot classify compounds better than random")

print(f"\n  Note: This analysis uses ML-PREDICTED CE50 values (R² = 0.57)")
print(f"        The original 2016 discovery used EXPERIMENTAL CE50 measurements")
print(f"        which showed significant correlation with rat PK outcomes.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\nOutput files generated:")
print("  1. roc_curve_ce50_pk.png - ROC curve with AUC score")
print("  2. confusion_matrix_ce50_pk.png - Confusion matrix heatmap")
print("  3. ce50_auc_dose_quadrants.png - Scatter plot with threshold lines")
print("  4. roc_classification_results.csv - Per-compound classification results")
print()
