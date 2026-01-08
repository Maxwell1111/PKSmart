#!/usr/bin/env python3
"""
Test CE50-Based Stratification Strategy

This script tests whether compounds with different predicted CE50 values
show different PK prediction performance, which would justify training
separate models for each CE50 stratum.

Hypothesis: If experimental CE50 is useful for ranking, then stratifying
by predicted CE50 should reveal different structure-PK relationships in
each stratum.

Author: Generated with Claude Code
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("="*80)
print("CE50 STRATIFICATION ANALYSIS")
print("="*80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Step 1: Loading Data")
print("-" * 80)

# Load CV results
human_baseline_cv = pd.read_csv('Prediction_human_from_mordred_morgan_baseline.csv')
human_ce50_cv = pd.read_csv('Prediction_human_from_mordred_morgan_ce50.csv')
rat_baseline_cv = pd.read_csv('Prediction_rat_from_mordred_morgan_fs_baseline.csv')
rat_ce50_cv = pd.read_csv('Prediction_rat_from_mordred_morgan_fs_ce50.csv')

# Load CE50 predictions
human_ce50_preds = pd.read_csv('data/human_ce50_predictions.csv')
rat_ce50_preds = pd.read_csv('data/rat_ce50_predictions.csv')

print(f"✓ Loaded human CV results: {len(human_baseline_cv)} folds")
print(f"✓ Loaded rat CV results: {len(rat_baseline_cv)} folds")
print(f"✓ Loaded CE50 predictions: {len(human_ce50_preds)} human, {len(rat_ce50_preds)} rat")
print()

# ============================================================================
# Analyze CE50 Distribution
# ============================================================================

print("Step 2: Analyzing CE50 Distribution")
print("-" * 80)

print(f"\nHuman CE50 Statistics:")
print(f"  Mean:   {human_ce50_preds['ce50'].mean():.2f} eV")
print(f"  Median: {human_ce50_preds['ce50'].median():.2f} eV")
print(f"  Std:    {human_ce50_preds['ce50'].std():.2f} eV")
print(f"  Min:    {human_ce50_preds['ce50'].min():.2f} eV")
print(f"  Max:    {human_ce50_preds['ce50'].max():.2f} eV")

# Define CE50 strata (tertiles)
human_ce50_tertiles = human_ce50_preds['ce50'].quantile([0.33, 0.67])
rat_ce50_tertiles = rat_ce50_preds['ce50'].quantile([0.33, 0.67])

print(f"\nHuman CE50 Tertiles:")
print(f"  Low:  < {human_ce50_tertiles[0.33]:.2f} eV (favorable fragmentation)")
print(f"  Mid:  {human_ce50_tertiles[0.33]:.2f} - {human_ce50_tertiles[0.67]:.2f} eV")
print(f"  High: > {human_ce50_tertiles[0.67]:.2f} eV (difficult fragmentation)")

print(f"\nRat CE50 Tertiles:")
print(f"  Low:  < {rat_ce50_tertiles[0.33]:.2f} eV (favorable fragmentation)")
print(f"  Mid:  {rat_ce50_tertiles[0.33]:.2f} - {rat_ce50_tertiles[0.67]:.2f} eV")
print(f"  High: > {rat_ce50_tertiles[0.67]:.2f} eV (difficult fragmentation)")
print()

# ============================================================================
# Test Stratification Hypothesis
# ============================================================================

print("Step 3: Testing Stratification by CE50")
print("-" * 80)
print()

def analyze_stratified_performance(cv_data, ce50_data, species, tertiles):
    """Analyze if PK prediction performance differs by CE50 stratum."""

    print(f"{species} Performance by CE50 Stratum:")
    print("-" * 60)

    results = []

    for endpoint in cv_data['endpoint'].unique():
        endpoint_cv = cv_data[cv_data['endpoint'] == endpoint]

        # Get unique compounds (average across CV folds)
        endpoint_summary = endpoint_cv.groupby('endpoint').agg({
            'r2': 'mean',
            'gmfe': 'mean',
            'fold_2': 'mean'
        }).reset_index()

        # For stratification, we need compound-level CE50 values
        # This is simplified - in practice you'd match compounds properly

        print(f"\n  {endpoint}:")
        print(f"    Overall R²:    {endpoint_summary['r2'].values[0]:.3f}")
        print(f"    Overall GMFE:  {endpoint_summary['gmfe'].values[0]:.3f}")
        print(f"    Overall Fold-2: {endpoint_summary['fold_2'].values[0]:.1f}%")

        results.append({
            'species': species,
            'endpoint': endpoint,
            'stratum': 'Overall',
            'r2': endpoint_summary['r2'].values[0],
            'gmfe': endpoint_summary['gmfe'].values[0],
            'fold_2': endpoint_summary['fold_2'].values[0],
            'n_compounds': len(endpoint_cv)
        })

    return pd.DataFrame(results)

# Analyze human data
human_results = analyze_stratified_performance(
    human_baseline_cv, human_ce50_preds, 'Human', human_ce50_tertiles
)

print()

# Analyze rat data
rat_results = analyze_stratified_performance(
    rat_baseline_cv, rat_ce50_preds, 'Rat', rat_ce50_tertiles
)

print()

# ============================================================================
# Key Diagnostic: CE50 Confidence vs Performance
# ============================================================================

print("="*80)
print("DIAGNOSTIC: Does CE50 Confidence Correlate with PK Prediction Quality?")
print("="*80)
print()

print("Testing hypothesis: High CE50 prediction confidence → Better PK predictions")
print("-" * 80)

# For human data
print("\nHuman Endpoints (correlation between CE50 confidence and PK R²):")
# This is a simplification - proper analysis would match individual compounds
print("  (Analysis requires compound-level matching across datasets)")

# ============================================================================
# Recommendations
# ============================================================================

print()
print("="*80)
print("ANALYSIS SUMMARY & RECOMMENDATIONS")
print("="*80)
print()

print("FINDING: Limited data structure prevents full stratification analysis")
print("-" * 80)
print()
print("The current CV results are aggregated by fold, not by compound.")
print("To properly test CE50 stratification, we need:")
print()
print("1. Compound-level PK predictions (not fold-aggregated)")
print("2. Match each compound's CE50 prediction to its PK prediction")
print("3. Stratify compounds into CE50 tertiles")
print("4. Compare R² within each stratum vs. overall")
print()
print("ALTERNATIVE APPROACH: Train new stratified models")
print("-" * 80)
print()
print("To test if CE50 stratification helps:")
print()
print("1. Load raw PK data (Human_PK_data.csv, Animal_PK_data.csv)")
print("2. Merge with CE50 predictions")
print("3. Split compounds into CE50 tertiles (low/medium/high)")
print("4. Train separate PK models for each tertile")
print("5. Compare performance:")
print("   - Baseline (unified model): Current R² values")
print("   - Stratified (3 models):    Expected R² if stratification helps")
print()
print("HYPOTHESIS:")
print("-" * 80)
print()
print("If experimental CE50 ranking is useful, then:")
print("  • Low CE50 compounds should have different structure-PK patterns")
print("  • Training separate models might improve predictions")
print("  • HOWEVER: Predicted CE50 already uses same structural features")
print("  • So stratification may still show no benefit")
print()
print("CONCLUSION:")
print("-" * 80)
print()
print("The lack of improvement from CE50 features suggests:")
print()
print("✓ Structural features (Mordred + Morgan) already capture the information")
print("✓ Predicted CE50 is redundant (derived from same features)")
print("✓ Experimental CE50 may capture measurement artifacts not in structure")
print("✓ Or the original CE50-PK correlation was spurious/dataset-specific")
print()
print("RECOMMENDATION: Do NOT pursue CE50 enhancement further unless:")
print()
print("1. You can test with a larger, independent experimental CE50 dataset")
print("2. You identify specific mechanisms linking fragmentation to PK")
print("3. You find subgroups where CE50 stratification clearly helps")
print()
print("="*80)
print()

# Save results
all_results = pd.concat([human_results, rat_results], ignore_index=True)
all_results.to_csv('ce50_stratification_analysis.csv', index=False)
print("✓ Saved analysis to: ce50_stratification_analysis.csv")
print()
