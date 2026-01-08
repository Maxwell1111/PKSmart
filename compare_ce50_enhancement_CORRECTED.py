#!/usr/bin/env python3
"""
CORRECTED CE50 Enhancement Comparison Script

This script performs proper comparison between baseline and CE50-enhanced
pharmacokinetic prediction models using ACTUAL cross-validation results.

BUG FIX: Previous script was comparing:
- Fake baseline (actual values + noise)
- CE50 predictions (not PK predictions!)

This script uses the actual CV results files which contain proper
held-out test predictions from nested cross-validation.

Author: Generated with Claude Code
Date: 2026-01-07 (Corrected)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CE50 ENHANCEMENT COMPARISON ANALYSIS (CORRECTED)")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# ============================================================================
# Step 1: Load Cross-Validation Results
# ============================================================================

print("Step 1: Loading Cross-Validation Results")
print("-" * 80)

# Load rat CV results
rat_baseline_cv = pd.read_csv('Prediction_rat_from_mordred_morgan_fs_baseline.csv')
rat_ce50_cv = pd.read_csv('Prediction_rat_from_mordred_morgan_fs_ce50.csv')
print(f"✓ Loaded rat baseline CV results: {rat_baseline_cv.shape[0]} folds")
print(f"✓ Loaded rat CE50 CV results: {rat_ce50_cv.shape[0]} folds")

# Load human CV results
human_baseline_cv = pd.read_csv('Prediction_human_from_mordred_morgan_baseline.csv')
human_ce50_cv = pd.read_csv('Prediction_human_from_mordred_morgan_ce50.csv')
print(f"✓ Loaded human baseline CV results: {human_baseline_cv.shape[0]} folds")
print(f"✓ Loaded human CE50 CV results: {human_ce50_cv.shape[0]} folds")
print()

# ============================================================================
# Step 2: Aggregate Metrics by Endpoint
# ============================================================================

print("Step 2: Aggregating Metrics by Endpoint")
print("-" * 80)

metrics = ['r2', 'rmse', 'gmfe', 'fold_2', 'fold_3', 'fold_5', 'bias']
comparison_results = []

def aggregate_cv_metrics(df, species, model_type):
    """Aggregate CV metrics by endpoint."""
    results = []

    for endpoint in df['endpoint'].unique():
        subset = df[df['endpoint'] == endpoint]

        for metric in metrics:
            if metric in subset.columns:
                values = subset[metric].values
                results.append({
                    'species': species,
                    'endpoint': endpoint,
                    'metric': metric,
                    'model_type': model_type,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_folds': len(values)
                })

    return pd.DataFrame(results)

# Aggregate all results
print("Aggregating rat baseline...")
rat_base_agg = aggregate_cv_metrics(rat_baseline_cv, 'Rat', 'baseline')
print("Aggregating rat CE50...")
rat_ce50_agg = aggregate_cv_metrics(rat_ce50_cv, 'Rat', 'ce50')
print("Aggregating human baseline...")
human_base_agg = aggregate_cv_metrics(human_baseline_cv, 'Human', 'baseline')
print("Aggregating human CE50...")
human_ce50_agg = aggregate_cv_metrics(human_ce50_cv, 'Human', 'ce50')

# Combine all aggregated results
all_agg = pd.concat([rat_base_agg, rat_ce50_agg, human_base_agg, human_ce50_agg], ignore_index=True)
print(f"✓ Aggregated {len(all_agg)} metric-endpoint combinations")
print()

# ============================================================================
# Step 3: Calculate Improvements and Statistical Tests
# ============================================================================

print("Step 3: Calculating Improvements and Statistical Tests")
print("-" * 80)

comparison_report = []

for species in all_agg['species'].unique():
    for endpoint in all_agg[all_agg['species'] == species]['endpoint'].unique():
        print(f"  {species} - {endpoint}")

        for metric in metrics:
            # Get baseline and CE50 values
            baseline_row = all_agg[
                (all_agg['species'] == species) &
                (all_agg['endpoint'] == endpoint) &
                (all_agg['metric'] == metric) &
                (all_agg['model_type'] == 'baseline')
            ]

            ce50_row = all_agg[
                (all_agg['species'] == species) &
                (all_agg['endpoint'] == endpoint) &
                (all_agg['metric'] == metric) &
                (all_agg['model_type'] == 'ce50')
            ]

            if len(baseline_row) > 0 and len(ce50_row) > 0:
                baseline_mean = baseline_row['mean'].values[0]
                ce50_mean = ce50_row['mean'].values[0]

                # Calculate improvement
                improvement = ce50_mean - baseline_mean

                # Calculate improvement percentage
                if baseline_mean != 0:
                    improvement_pct = (improvement / abs(baseline_mean)) * 100
                else:
                    improvement_pct = np.nan

                # Statistical test using raw fold values
                if species == 'Rat':
                    baseline_cv_data = rat_baseline_cv
                    ce50_cv_data = rat_ce50_cv
                else:
                    baseline_cv_data = human_baseline_cv
                    ce50_cv_data = human_ce50_cv

                baseline_values = baseline_cv_data[
                    baseline_cv_data['endpoint'] == endpoint
                ][metric].values

                ce50_values = ce50_cv_data[
                    ce50_cv_data['endpoint'] == endpoint
                ][metric].values

                # Wilcoxon signed-rank test
                try:
                    if len(baseline_values) > 4 and len(ce50_values) > 4:
                        statistic, p_value = stats.wilcoxon(baseline_values, ce50_values)
                        significant = p_value < 0.05
                    else:
                        p_value = np.nan
                        significant = False
                except:
                    p_value = np.nan
                    significant = False

                comparison_report.append({
                    'species': species,
                    'endpoint': endpoint,
                    'metric': metric,
                    'baseline': baseline_mean,
                    'ce50': ce50_mean,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'p_value': p_value,
                    'significant': significant
                })

comparison_df = pd.DataFrame(comparison_report)
print(f"✓ Calculated {len(comparison_df)} comparisons")
print()

# ============================================================================
# Step 4: Generate Summary Report
# ============================================================================

print("="*80)
print("SUMMARY REPORT")
print("="*80)
print()

print("Overall Performance (Mean across all CV folds):")
print("-" * 80)

for species in ['Rat', 'Human']:
    print(f"\n{species} Endpoints:")
    species_data = comparison_df[comparison_df['species'] == species]

    for endpoint in species_data['endpoint'].unique():
        endpoint_data = species_data[species_data['endpoint'] == endpoint]
        r2_row = endpoint_data[endpoint_data['metric'] == 'r2'].iloc[0]

        print(f"  {endpoint}:")
        print(f"    Baseline R²: {r2_row['baseline']:.3f}")
        print(f"    CE50 R²: {r2_row['ce50']:.3f}")
        print(f"    Improvement: ΔR² = {r2_row['improvement']:+.3f} ({r2_row['improvement_pct']:+.1f}%)")
        print(f"    p-value: {r2_row['p_value']:.4f} {'✓ Significant' if r2_row['significant'] else '✗ Not significant'}")

print()
print("="*80)
print("KEY METRICS COMPARISON")
print("="*80)
print()

# Create detailed comparison table
for metric in ['r2', 'gmfe', 'fold_2']:
    print(f"\n{metric.upper()}:")
    print("-" * 80)

    metric_data = comparison_df[comparison_df['metric'] == metric]

    for species in ['Rat', 'Human']:
        species_metric = metric_data[metric_data['species'] == species]

        if len(species_metric) > 0:
            print(f"\n{species}:")
            for _, row in species_metric.iterrows():
                direction = "↑" if row['improvement'] > 0 else "↓"
                print(f"  {row['endpoint']:30s} | Baseline: {row['baseline']:6.3f} | CE50: {row['ce50']:6.3f} | {direction} {row['improvement']:+.3f}")

print()

# ============================================================================
# Step 5: Save Results
# ============================================================================

print("="*80)
print("SAVING RESULTS")
print("="*80)

# Save comparison report
comparison_df.to_csv('ce50_comparison_report_CORRECTED.csv', index=False)
print("✓ Saved: ce50_comparison_report_CORRECTED.csv")

# Save aggregated metrics
all_agg.to_csv('ce50_aggregated_metrics.csv', index=False)
print("✓ Saved: ce50_aggregated_metrics.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
print("Generated Files:")
print("  • ce50_comparison_report_CORRECTED.csv - Main comparison results")
print("  • ce50_aggregated_metrics.csv - Detailed CV metrics")
print()
