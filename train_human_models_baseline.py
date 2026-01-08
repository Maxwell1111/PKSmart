#!/usr/bin/env python3
"""
Train Baseline Human PK Models (WITHOUT CE50 Features)

This script trains Random Forest models for human pharmacokinetic parameters
using Morgan fingerprints and Mordred descriptors ONLY (NO CE50 features).
This serves as a baseline for comparison with CE50-enhanced models.

Endpoints:
- human_VDss_L_kg (Volume of distribution)
- human_CL_mL_min_kg (Clearance)
- human_fup (Fraction unbound in plasma)
- human_mrt (Mean residence time)
- human_thalf (Half-life)

Features:
- 152 Morgan fingerprints (2048-bit, variance threshold + correlation filtered)
- 352 Mordred descriptors (variance threshold + correlation filtered)
- Total: 504 features (NO CE50)

Requirements:
- pandas, numpy, scikit-learn
- rdkit
- mordred (pip install mordred-descriptor)

Usage:
    python train_human_models_baseline.py

Author: Generated with Claude Code
Date: 2026-01-07
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from math import sqrt
from itertools import compress

# sklearn imports
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold

# RDKit imports
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Mordred imports
try:
    from mordred import Calculator, descriptors
except ImportError:
    print("ERROR: mordred package not found.")
    print("Please install it using: pip install mordred-descriptor")
    print("Or: conda install -c conda-forge mordred-descriptor")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE HUMAN MODEL TRAINING (NO CE50 FEATURES)")
print("="*80)
print()

# ============================================================================
# Helper Functions (from baseline notebook)
# ============================================================================

def fs_variance(df, threshold=0.05):
    """
    Return a list of selected variables based on the threshold.
    """
    features = list(df.columns)
    vt = VarianceThreshold(threshold=threshold)
    _ = vt.fit(df)
    feat_select = list(compress(features, vt.get_support()))
    return feat_select


def get_pairwise_correlation(population_df, method="pearson"):
    """Given a population dataframe, calculate all pairwise correlations."""
    # Get a symmetrical correlation matrix
    data_cor_df = population_df.corr(method=method)

    # Create a copy of the dataframe to generate upper triangle of zeros
    data_cor_natri_df = data_cor_df.copy()

    # Replace upper triangle in correlation matrix with NaN
    data_cor_natri_df = data_cor_natri_df.where(
        np.tril(np.ones(data_cor_natri_df.shape), k=-1).astype(bool)
    )

    # Acquire pairwise correlations in a long format
    pairwise_df = data_cor_natri_df.stack().reset_index()
    pairwise_df.columns = ["pair_a", "pair_b", "correlation"]

    return data_cor_df, pairwise_df


def determine_high_cor_pair(correlation_row, sorted_correlation_pairs):
    """
    Select highest correlated variable given a correlation row.
    For use in a pandas.apply()
    """
    pair_a = correlation_row["pair_a"]
    pair_b = correlation_row["pair_b"]

    if sorted_correlation_pairs.get_loc(pair_a) > sorted_correlation_pairs.get_loc(pair_b):
        return pair_a
    else:
        return pair_b


def count(pred, true, min_val, max_val, endpoint):
    """Calculate percentage of predictions within fold-change range."""
    if endpoint in ["human_fup"]:
        lst = [abs(a/b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100
    else:
        lst = [abs(10**a/10**b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100


def calc_gmfe(pred, true, endpoint):
    """Calculate Geometric Mean Fold Error."""
    if endpoint in ["human_fup"]:
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs


def median_fold_change_error(pred, true, endpoint):
    """Calculate Median Fold Change Error."""
    if endpoint in ["human_fup"]:
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs


def calc_bias(pred, true, endpoint):
    """Calculate prediction bias."""
    if endpoint in ["human_fup"]:
        lst = [(a - b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias
    else:
        lst = [(10**a - 10**b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias


# ============================================================================
# Step 1: Load and Process Human PK Data
# ============================================================================

print("="*80)
print("Step 1: Loading Human PK Data")
print("="*80)
human = pd.read_csv("data/Human_PK_data.csv")
print(f"  Loaded {len(human)} compounds from Human_PK_data.csv")

# Select relevant endpoints
endpoints = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]
human_filtered = human[["smiles_r"] + endpoints].dropna(
    subset=endpoints, how="all"
).reset_index(drop=True)
print(f"  Found {len(human_filtered)} compounds with human PK data")

# Log transform appropriate endpoints (VDss, CL, mrt, thalf)
for endpoint in ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_mrt", "human_thalf"]:
    human_filtered[endpoint] = np.log10(human_filtered[endpoint])
print("  Applied log10 transformation to VDss, CL, mrt, thalf")

# Group by SMILES and take median (handle duplicates)
human_filtered = human_filtered.groupby('smiles_r').median().reset_index()
print(f"  After grouping by SMILES: {len(human_filtered)} unique compounds")

# ============================================================================
# Step 2: Generate Mordred Descriptors
# ============================================================================

print("\n" + "="*80)
print("Step 2: Generating Mordred Descriptors")
print("="*80)
calc = Calculator(descriptors, ignore_3D=True)
print(f"  Total Mordred descriptors: {len(calc.descriptors)}")

Ser_Mol = human_filtered['smiles_r'].apply(Chem.MolFromSmiles)
Mordred_table = calc.pandas(Ser_Mol, nproc=1, quiet=False)

# Convert to float and handle missing values
Mordred_table = Mordred_table.astype('float')
Mordred_table['smiles_r'] = human_filtered['smiles_r']

# Drop columns with any NaN values
Mordred_table = Mordred_table.dropna(axis='columns')
print(f"  After dropping NaN columns: {Mordred_table.shape[1] - 1} descriptors")

# Variance threshold filtering
selected_Features = fs_variance(Mordred_table.iloc[:, :-1], threshold=0.05)
print(f"  After variance threshold (0.05): {len(selected_Features)} descriptors")
new_Mordred_table = Mordred_table[selected_Features]

# Correlation filtering (remove highly correlated features > 0.95)
data_cor_df, pairwise_df = get_pairwise_correlation(population_df=new_Mordred_table)
variable_cor_sum = data_cor_df.abs().sum().sort_values().index
pairwise_df = pairwise_df.query("correlation > 0.95")
excluded = pairwise_df.apply(lambda x: determine_high_cor_pair(x, variable_cor_sum), axis="columns")
excluded_features = list(set(excluded.tolist()))
print(f"  Removing {len(excluded_features)} highly correlated features (>0.95)")
new_Mordred_table = new_Mordred_table.drop(excluded_features, axis=1)
new_Mordred_table['smiles_r'] = human_filtered['smiles_r']
print(f"  Final Mordred descriptors: {new_Mordred_table.shape[1] - 1}")

# ============================================================================
# Step 3: Generate Morgan Fingerprints
# ============================================================================

print("\n" + "="*80)
print("Step 3: Generating Morgan Fingerprints")
print("="*80)
Ser_Mol = human_filtered['smiles_r'].apply(Chem.MolFromSmiles)
Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
Morganfingerprint_array = np.stack(Morgan_fingerprint)

Morgan_collection = ["Mfp" + str(x) for x in np.arange(Morganfingerprint_array.shape[1])]
Morganfingerprint_table = pd.DataFrame(Morganfingerprint_array, columns=Morgan_collection)
Morganfingerprint_table['smiles_r'] = human_filtered['smiles_r']

# Variance threshold filtering
selected_Features = fs_variance(Morganfingerprint_table.iloc[:, :-1], threshold=0.05)
print(f"  After variance threshold (0.05): {len(selected_Features)} fingerprints")
new_Morganfingerprint_table = Morganfingerprint_table[selected_Features].copy()
new_Morganfingerprint_table['smiles_r'] = human_filtered['smiles_r']

# ============================================================================
# Step 4: Merge Features (Morgan + Mordred ONLY, NO CE50)
# ============================================================================

print("\n" + "="*80)
print("Step 4: Merging Features (NO CE50)")
print("="*80)
# Merge Morgan and Mordred ONLY
human_mfp = pd.merge(human_filtered, new_Morganfingerprint_table, on='smiles_r')
human_mfp_Mordred = pd.merge(human_mfp, new_Mordred_table, on='smiles_r')
print(f"  After merging Morgan + Mordred: {human_mfp_Mordred.shape[1] - 6} features")

# Create feature list (NO CE50)
Mordred_columns = new_Mordred_table.columns[:-1].tolist()
mfp_columns = new_Morganfingerprint_table.columns[:-1].tolist()
features_mfp_mordred_columns = Mordred_columns + mfp_columns

print(f"\n  Feature breakdown:")
print(f"    Mordred descriptors: {len(Mordred_columns)}")
print(f"    Morgan fingerprints: {len(mfp_columns)}")
print(f"    CE50 features: 0 (BASELINE - NO CE50)")
print(f"    TOTAL FEATURES: {len(features_mfp_mordred_columns)} (NO CE50)")

# Save feature list
with open("features_mfp_mordred_columns_human_baseline.txt", "w") as f:
    for item in features_mfp_mordred_columns:
        f.write(item + "\n")
print(f"  Saved feature list to: features_mfp_mordred_columns_human_baseline.txt")

# ============================================================================
# Step 5: Train Models with Nested Cross-Validation
# ============================================================================

print("\n" + "="*80)
print("Step 5: Training Models with Nested 5-Fold Cross-Validation")
print("="*80)

# Hyperparameter grid (same as CE50 version)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 4, 8],
    "max_depth": [5, 10, 15],
    "bootstrap": [True, False],
    'n_jobs': [30]
}

list_of_lists = []

for endpoint in endpoints:
    print(f"\n{'='*80}")
    print(f"Training models for: {endpoint}")
    print(f"{'='*80}")

    baseline = 10.00
    data = human_mfp_Mordred
    features = features_mfp_mordred_columns

    # Filter data for this endpoint
    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    print(f"  Compounds with {endpoint} data: {len(df)}")

    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    # Save the scaler for this endpoint
    with open(f'human_baseline_{endpoint}_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Nested cross-validation with 5 random seeds
    for i in range(42, 47):
        print(f"\n  Random seed: {i}")

        # Outer CV
        outercv = KFold(n_splits=5, random_state=i, shuffle=True)

        for split, (train_index, test_index) in enumerate(outercv.split(X)):
            print(f"    Fold {split + 1}/5...", end=" ")

            X_train = X.iloc[train_index].values
            Y_train = Y.iloc[train_index].values.flatten()
            X_test = X.iloc[test_index].values
            Y_test = Y.iloc[test_index].values.flatten()

            # Inner CV for hyperparameter tuning
            inner_cv = KFold(n_splits=4, random_state=i, shuffle=True)

            # Create base model
            regressor = RandomForestRegressor(random_state=i, n_jobs=20)

            # Grid search
            gsh = GridSearchCV(
                estimator=regressor,
                param_grid=param_grid,
                cv=inner_cv,
                n_jobs=30,
                verbose=0
            )

            gsh.fit(X_train, Y_train)

            # Get best model and evaluate on test set
            classifier = gsh.best_estimator_
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)

            # Calculate metrics
            fold_2 = count(y_pred, Y_test, 0.5, 2, endpoint)
            fold_3 = count(y_pred, Y_test, 1/3, 3, endpoint)
            fold_5 = count(y_pred, Y_test, 1/5, 5, endpoint)
            gmfe = calc_gmfe(y_pred, Y_test, endpoint)
            mfe = median_fold_change_error(y_pred, Y_test, endpoint)
            bias = calc_bias(y_pred, Y_test, endpoint)
            rmse = sqrt(mean_squared_error(Y_test, y_pred))
            r2 = r2_score(Y_test, y_pred)

            print(f"GMFE={gmfe:.3f}, R2={r2:.3f}, RMSE={rmse:.3f}")

            # Save best model
            if gmfe < baseline:
                baseline = gmfe
                pickle.dump(classifier, open(f"log_human_{endpoint}_baseline.sav", 'wb'))

            # Store results
            row = ["HeldOut", endpoint, i, split, fold_2, fold_3, fold_5, gmfe, mfe, bias, rmse, r2]
            list_of_lists.append(row)

    print(f"\n  Best GMFE for {endpoint}: {baseline:.4f}")

# ============================================================================
# Step 6: Save Results
# ============================================================================

print("\n" + "="*80)
print("Step 6: Saving Results")
print("="*80)

results = pd.DataFrame(
    list_of_lists,
    columns=["HeldOut", "endpoint", "random_state", "split", "fold_2", "fold_3", "fold_5",
             "gmfe", "mfe", "bias", "rmse", "r2"]
)
results.to_csv("Prediction_human_from_mordred_morgan_baseline.csv", index=False)
print("  Saved detailed results to: Prediction_human_from_mordred_morgan_baseline.csv")

# ============================================================================
# Step 7: Train Final Models on Full Dataset
# ============================================================================

print("\n" + "="*80)
print("Step 7: Training Final Models on Full Dataset")
print("="*80)

for endpoint in endpoints:
    print(f"\n  Training final model for: {endpoint}")

    data = human_mfp_Mordred
    features = features_mfp_mordred_columns

    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    # Save scaler for this endpoint
    with open(f'human_baseline_{endpoint}_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    X_train = X.values
    Y_train = Y.values.flatten()

    # Load best hyperparameters from saved model
    loaded_rf = pickle.load(open(f"log_human_{endpoint}_baseline.sav", 'rb'))
    params = loaded_rf.get_params()

    # Train final model on all data
    classifier = RandomForestRegressor(**params)
    classifier.fit(X_train, Y_train)

    # Save final model
    pickle.dump(classifier, open(f"log_human_{endpoint}_baseline_FINAL.sav", 'wb'))
    print(f"    Saved: log_human_{endpoint}_baseline_FINAL.sav")
    print(f"    Saved scaler: human_baseline_{endpoint}_scaler.pkl")

# ============================================================================
# Step 8: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary = results.groupby("endpoint").mean()
summary.to_csv("human_baseline.csv")
print("\nMean performance across all CV folds:")
print(summary[["fold_2", "fold_3", "fold_5", "gmfe", "mfe", "rmse", "r2"]].to_string())
print("\nSaved summary to: human_baseline.csv")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. features_mfp_mordred_columns_human_baseline.txt - Feature list (504 features)")
print("  2. human_baseline_{endpoint}_scaler.pkl - Feature scalers (5 files)")
print("  3. log_human_{endpoint}_baseline_FINAL.sav - Final models (5 files)")
print("  4. Prediction_human_from_mordred_morgan_baseline.csv - Detailed CV results")
print("  5. human_baseline.csv - Summary statistics")
print(f"\nTotal features: {len(features_mfp_mordred_columns)} (Mordred + Morgan, NO CE50)")
print("BASELINE MODEL - NO CE50 features included")
print("="*80)
