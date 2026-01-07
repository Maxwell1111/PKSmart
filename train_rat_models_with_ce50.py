#!/usr/bin/env python3
"""
Train CE50-Enhanced Rat PK Models

This script trains Random Forest models for rat pharmacokinetic parameters
using Morgan fingerprints, Mordred descriptors, and CE50 features.

Endpoints:
- rat_VDss_L_kg (Volume of distribution)
- rat_CL_mL_min_kg (Clearance)
- rat_fup (Fraction unbound in plasma)

Features:
- 153 Morgan fingerprints (2048-bit, variance threshold + correlation filtered)
- 386 Mordred descriptors (variance threshold + correlation filtered)
- 3 CE50 features (ce50, pce50, confidence)
- Total: 542 features

Requirements:
- pandas, numpy, scikit-learn
- rdkit
- mordred (pip install mordred-descriptor)

Usage:
    python train_rat_models_with_ce50.py

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
print("CE50-ENHANCED RAT PK MODEL TRAINING")
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
    if endpoint == "rat_fup":
        lst = [abs(a/b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100
    else:
        lst = [abs(10**a/10**b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100


def calc_gmfe(pred, true, endpoint):
    """Calculate Geometric Mean Fold Error."""
    if endpoint == "rat_fup":
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs


def median_fold_change_error(pred, true, endpoint):
    """Calculate Median Fold Change Error."""
    if endpoint == "rat_fup":
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs


def calc_bias(pred, true, endpoint):
    """Calculate prediction bias."""
    if endpoint == "rat_fup":
        lst = [(a - b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias
    else:
        lst = [(10**a - 10**b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias


# ============================================================================
# Step 1: Load and Process Animal PK Data
# ============================================================================

print("="*80)
print("Step 1: Loading Animal PK Data")
print("="*80)
animal = pd.read_csv("data/Animal_PK_data.csv")
print(f"  Loaded {len(animal)} compounds from Animal_PK_data.csv")

# Filter for rat data
rat = animal[["smiles_r", "rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"]].dropna(
    subset=["rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"], how="all"
).reset_index(drop=True)
print(f"  Found {len(rat)} compounds with rat PK data")

# Log transform VDss and CL endpoints
for endpoint in ["rat_VDss_L_kg", "rat_CL_mL_min_kg"]:
    rat[endpoint] = np.log10(rat[endpoint])
print("  Applied log10 transformation to VDss and CL")

# Group by SMILES and take median (handle duplicates)
rat = rat.groupby('smiles_r').median().reset_index()
print(f"  After grouping by SMILES: {len(rat)} unique compounds")

# ============================================================================
# Step 2: Load CE50 Predictions
# ============================================================================

print("\n" + "="*80)
print("Step 2: Loading CE50 Predictions")
print("="*80)
ce50_data = pd.read_csv("data/rat_ce50_predictions_simple.csv")
print(f"  Loaded {len(ce50_data)} compounds with CE50 predictions")

# Select only essential CE50 columns
ce50_features = ce50_data[["smiles_r", "ce50", "pce50", "confidence"]].copy()
print(f"  CE50 features: ce50, pce50, confidence")

# ============================================================================
# Step 3: Generate Mordred Descriptors
# ============================================================================

print("\n" + "="*80)
print("Step 3: Generating Mordred Descriptors")
print("="*80)
calc = Calculator(descriptors, ignore_3D=True)
print(f"  Total Mordred descriptors: {len(calc.descriptors)}")

Ser_Mol = rat['smiles_r'].apply(Chem.MolFromSmiles)
Mordred_table = calc.pandas(Ser_Mol, nproc=1, quiet=False)

# Convert to float and handle missing values
Mordred_table = Mordred_table.astype('float')
Mordred_table['smiles_r'] = rat['smiles_r']

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
new_Mordred_table['smiles_r'] = rat['smiles_r']
print(f"  Final Mordred descriptors: {new_Mordred_table.shape[1] - 1}")

# ============================================================================
# Step 4: Generate Morgan Fingerprints
# ============================================================================

print("\n" + "="*80)
print("Step 4: Generating Morgan Fingerprints")
print("="*80)
Ser_Mol = rat['smiles_r'].apply(Chem.MolFromSmiles)
Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
Morganfingerprint_array = np.stack(Morgan_fingerprint)

Morgan_collection = ["Mfp" + str(x) for x in np.arange(Morganfingerprint_array.shape[1])]
Morganfingerprint_table = pd.DataFrame(Morganfingerprint_array, columns=Morgan_collection)
Morganfingerprint_table['smiles_r'] = rat['smiles_r']

# Variance threshold filtering
selected_Features = fs_variance(Morganfingerprint_table.iloc[:, :-1], threshold=0.05)
print(f"  After variance threshold (0.05): {len(selected_Features)} fingerprints")
new_Morganfingerprint_table = Morganfingerprint_table[selected_Features].copy()
new_Morganfingerprint_table['smiles_r'] = rat['smiles_r']

# ============================================================================
# Step 5: Merge All Features (Morgan + Mordred + CE50)
# ============================================================================

print("\n" + "="*80)
print("Step 5: Merging All Features")
print("="*80)
# Merge Morgan and Mordred
rat_mfp = pd.merge(rat, new_Morganfingerprint_table, on='smiles_r')
rat_mfp_Mordred = pd.merge(rat_mfp, new_Mordred_table, on='smiles_r')
print(f"  After merging Morgan + Mordred: {rat_mfp_Mordred.shape[1] - 4} features")

# Merge with CE50 features
rat_mfp_Mordred_ce50 = pd.merge(rat_mfp_Mordred, ce50_features, on='smiles_r')
print(f"  After merging CE50 features: {rat_mfp_Mordred_ce50.shape[1] - 4} features")

# Create feature list
Mordred_columns = new_Mordred_table.columns[:-1].tolist()
mfp_columns = new_Morganfingerprint_table.columns[:-1].tolist()
ce50_columns = ["ce50", "pce50", "confidence"]
features_mfp_mordred_ce50_columns = Mordred_columns + mfp_columns + ce50_columns

print(f"\n  Feature breakdown:")
print(f"    Mordred descriptors: {len(Mordred_columns)}")
print(f"    Morgan fingerprints: {len(mfp_columns)}")
print(f"    CE50 features: {len(ce50_columns)}")
print(f"    TOTAL FEATURES: {len(features_mfp_mordred_ce50_columns)}")

# Save feature list
with open("features_mfp_mordred_ce50_columns_rat_model.txt", "w") as f:
    for item in features_mfp_mordred_ce50_columns:
        f.write(item + "\n")
print(f"  Saved feature list to: features_mfp_mordred_ce50_columns_rat_model.txt")

# ============================================================================
# Step 6: Train Models with Nested Cross-Validation
# ============================================================================

print("\n" + "="*80)
print("Step 6: Training Models with Nested 5-Fold Cross-Validation")
print("="*80)

# Hyperparameter grid (same as baseline)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 4, 8],
    "max_depth": [5, 10, 15],
    "bootstrap": [True, False],
    'n_jobs': [30]
}

list_of_lists = []
endpoints = ["rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"]

for endpoint in endpoints:
    print(f"\n{'='*80}")
    print(f"Training models for: {endpoint}")
    print(f"{'='*80}")

    baseline = 10.00
    data = rat_mfp_Mordred_ce50
    features = features_mfp_mordred_ce50_columns

    # Filter data for this endpoint
    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    print(f"  Compounds with {endpoint} data: {len(df)}")

    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    # Save the scaler (will be overwritten for each endpoint, final one saved)
    with open('scaler_rat_ce50.pkl', 'wb') as file:
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
                pickle.dump(classifier, open(f"log_{endpoint}_model_ce50.sav", 'wb'))

            # Store results
            row = ["HeldOut", endpoint, i, split, fold_2, fold_3, fold_5, gmfe, mfe, bias, rmse, r2]
            list_of_lists.append(row)

    print(f"\n  Best GMFE for {endpoint}: {baseline:.4f}")

# ============================================================================
# Step 7: Save Results
# ============================================================================

print("\n" + "="*80)
print("Step 7: Saving Results")
print("="*80)

results = pd.DataFrame(
    list_of_lists,
    columns=["HeldOut", "endpoint", "random_state", "split", "fold_2", "fold_3", "fold_5",
             "gmfe", "mfe", "bias", "rmse", "r2"]
)
results.to_csv("Prediction_rat_from_mordred_morgan_fs_ce50.csv", index=False)
print("  Saved detailed results to: Prediction_rat_from_mordred_morgan_fs_ce50.csv")

# ============================================================================
# Step 8: Train Final Models on Full Dataset
# ============================================================================

print("\n" + "="*80)
print("Step 8: Training Final Models on Full Dataset")
print("="*80)

for endpoint in endpoints:
    print(f"\n  Training final model for: {endpoint}")

    data = rat_mfp_Mordred_ce50
    features = features_mfp_mordred_ce50_columns

    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    X_train = X.values
    Y_train = Y.values.flatten()

    # Load best hyperparameters from saved model
    loaded_rf = pickle.load(open(f"log_{endpoint}_model_ce50.sav", 'rb'))
    params = loaded_rf.get_params()

    # Train final model on all data
    classifier = RandomForestRegressor(**params)
    classifier.fit(X_train, Y_train)

    # Save final model
    pickle.dump(classifier, open(f"log_{endpoint}_model_ce50_FINAL.sav", 'wb'))
    print(f"    Saved: log_{endpoint}_model_ce50_FINAL.sav")
    print(f"    Hyperparameters: {params}")

# Save final scaler (from last endpoint)
print(f"\n  Saved final scaler: scaler_rat_ce50.pkl")

# ============================================================================
# Step 9: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary = results.groupby("endpoint").mean()
print("\nMean performance across all CV folds:")
print(summary[["fold_2", "fold_3", "fold_5", "gmfe", "mfe", "rmse", "r2"]].to_string())

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. features_mfp_mordred_ce50_columns_rat_model.txt - Feature list (542 features)")
print("  2. scaler_rat_ce50.pkl - Feature scaler")
print("  3. log_rat_VDss_L_kg_model_ce50_FINAL.sav - VDss model")
print("  4. log_rat_CL_mL_min_kg_model_ce50_FINAL.sav - CL model")
print("  5. log_rat_fup_model_ce50_FINAL.sav - fup model")
print("  6. Prediction_rat_from_mordred_morgan_fs_ce50.csv - Detailed CV results")
print("\nTotal features: 542 (386 Mordred + 153 Morgan + 3 CE50)")
print("="*80)
