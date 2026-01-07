#!/usr/bin/env python3
"""
Generate Artificial Animal Data with CE50-Enhanced Rat Models

This script generates artificial animal PK predictions for human compounds using:
- CE50-enhanced rat models (VDss, CL, fup) - 542 features including CE50
- Baseline dog models (VDss, CL, fup) - no CE50
- Baseline monkey models (VDss, CL, fup) - no CE50

The output includes:
1. Median Mordred/Morgan feature values per animal (for imputation)
2. Artificial animal PK predictions for human compounds (9 predictions per compound)

Based on: 03_MedianMordredCalculator_artificial_animal_data_mfp_mrd.ipynb
Enhanced with CE50 features for rat models

Author: Generated with Claude Code
Date: 2026-01-07
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from itertools import compress

# RDKit imports
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Mordred imports
try:
    from mordred import Calculator, descriptors
except ImportError:
    print("ERROR: mordred package not found.")
    print("Please install it using: pip install mordred-descriptor")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')

print("="*80)
print("GENERATE ARTIFICIAL ANIMAL DATA WITH CE50-ENHANCED RAT MODELS")
print("="*80)
print()

# ============================================================================
# Helper Functions
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


def find_median_tables(data, animal, ce50_data=None):
    """
    Load feature columns for a specific animal model and extract those features.
    For rat models with CE50, merge CE50 features first.
    """
    # Determine if this is a CE50-enhanced model
    if animal == "rat" and ce50_data is not None:
        feature_file = f"features_mfp_mordred_ce50_columns_{animal}_model.txt"
    else:
        feature_file = f"features_mfp_mordred_columns_{animal}_model.txt"

    # Read feature columns
    try:
        with open(feature_file, "r") as file:
            file_lines = file.read()
            features = file_lines.split("\n")
            features = [f for f in features if f]  # Remove empty strings
    except FileNotFoundError:
        print(f"ERROR: Feature file not found: {feature_file}")
        print(f"Please ensure the {animal} model has been trained and feature list saved.")
        raise

    # For rat with CE50, merge CE50 features into data first
    if animal == "rat" and ce50_data is not None:
        data = pd.merge(data, ce50_data[['smiles_r', 'ce50', 'pce50', 'confidence']],
                       on='smiles_r', how='left')

    # Extract only the required features
    X = data[features]

    return X


def predict_animal_endpoint(data, animal, endpoint, ce50_data=None):
    """
    Load animal model and make predictions on human data.

    For rat: Use CE50-enhanced models (require CE50 features)
    For dog/monkey: Use baseline models (no CE50 features)
    """
    # Determine model file names
    if animal == "rat":
        model_file = f"log_{animal}_{endpoint}_model_ce50_FINAL.sav"
        scaler_file = f"scaler_{animal}_ce50.pkl"
    else:
        model_file = f"log_{animal}_{endpoint}_model_FINAL.sav"
        scaler_file = f"scaler_{animal}.pkl"

    print(f"  Loading model: {model_file}")

    # Load model
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {model_file}")
        print(f"Please train the {animal} model first.")
        raise

    # Load scaler
    try:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Scaler file not found: {scaler_file}")
        print(f"Please ensure the {animal} model training saved the scaler.")
        raise

    # Get features for this animal
    X = find_median_tables(data, animal, ce50_data=ce50_data if animal == "rat" else None)

    # Handle missing values by filling with median from training data
    median_file = f"Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model{'_ce50' if animal == 'rat' else ''}.csv"
    try:
        animal_median = pd.read_csv(median_file)
        for col in X.columns[X.isna().any()].tolist():
            X[col].fillna(float(animal_median[col]), inplace=True)
    except FileNotFoundError:
        print(f"WARNING: Median file not found: {median_file}")
        print(f"This file should be generated as part of this script.")
        print(f"Using forward fill for missing values instead.")
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    return predictions


# ============================================================================
# Step 1: Load Human PK Data
# ============================================================================

print("="*80)
print("Step 1: Loading Human PK Data")
print("="*80)
human_lombardo = pd.read_csv("data/Human_PK_data.csv")
human_lombardo = human_lombardo[human_lombardo["smiles_r"] != "Cannot_do"].reset_index(drop=True)
print(f"  Loaded {len(human_lombardo)} compounds from human_smiles_r_v3.csv")

# Select relevant columns
human_lombardo = human_lombardo[["smiles_r", "human_VDss_L_kg", "human_CL_mL_min_kg",
                                 "human_fup", "human_mrt", "human_thalf"]]

# Log transform endpoints (except fup)
for endpoint in ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_mrt", "human_thalf"]:
    human_lombardo[endpoint] = np.log10(human_lombardo[endpoint])
print("  Applied log10 transformation to VDss, CL, MRT, and thalf")

# Group by SMILES and take median (handle duplicates)
human_lombardo = human_lombardo.groupby('smiles_r').median().reset_index()
print(f"  After grouping by SMILES: {len(human_lombardo)} unique compounds")

# ============================================================================
# Step 2: Load Human CE50 Predictions
# ============================================================================

print("\n" + "="*80)
print("Step 2: Loading Human CE50 Predictions")
print("="*80)
ce50_data = pd.read_csv("data/human_ce50_predictions_simple.csv")
print(f"  Loaded {len(ce50_data)} compounds with CE50 predictions")

# Select only essential CE50 columns
ce50_features = ce50_data[["smiles_r", "ce50", "pce50", "confidence"]].copy()

# Merge with human_lombardo
human_lombardo = pd.merge(human_lombardo, ce50_features, on='smiles_r', how='left')
print(f"  Merged CE50 features: {human_lombardo[['ce50', 'pce50', 'confidence']].notna().all(axis=1).sum()} compounds have CE50 data")

# ============================================================================
# Step 3: Generate Mordred Descriptors for Human Data
# ============================================================================

print("\n" + "="*80)
print("Step 3: Generating Mordred Descriptors")
print("="*80)
calc = Calculator(descriptors, ignore_3D=True)
print(f"  Total Mordred descriptors: {len(calc.descriptors)}")

Ser_Mol = human_lombardo['smiles_r'].apply(Chem.MolFromSmiles)
Mordred_table = calc.pandas(Ser_Mol, nproc=1, quiet=False)

# Convert to float and add SMILES
Mordred_table = Mordred_table.astype('float')
Mordred_table['smiles_r'] = human_lombardo['smiles_r']

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
new_Mordred_table['smiles_r'] = human_lombardo['smiles_r']
print(f"  Final Mordred descriptors: {new_Mordred_table.shape[1] - 1}")

# ============================================================================
# Step 4: Generate Morgan Fingerprints for Human Data
# ============================================================================

print("\n" + "="*80)
print("Step 4: Generating Morgan Fingerprints")
print("="*80)
Ser_Mol = human_lombardo['smiles_r'].apply(Chem.MolFromSmiles)
Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
Morganfingerprint_array = np.stack(Morgan_fingerprint)

Morgan_collection = ["Mfp" + str(x) for x in np.arange(Morganfingerprint_array.shape[1])]
Morganfingerprint_table = pd.DataFrame(Morganfingerprint_array, columns=Morgan_collection)
Morganfingerprint_table['smiles_r'] = human_lombardo['smiles_r']

# Variance threshold filtering
selected_Features = fs_variance(Morganfingerprint_table.iloc[:, :-1], threshold=0.05)
print(f"  After variance threshold (0.05): {len(selected_Features)} fingerprints")
new_Morganfingerprint_table = Morganfingerprint_table[selected_Features].copy()
new_Morganfingerprint_table['smiles_r'] = human_lombardo['smiles_r']

# ============================================================================
# Step 5: Merge All Features
# ============================================================================

print("\n" + "="*80)
print("Step 5: Merging All Features")
print("="*80)
human_lombardo_mfp = pd.merge(human_lombardo, new_Morganfingerprint_table, on='smiles_r')
human_lombardo_mfp_Mordred = pd.merge(human_lombardo_mfp, new_Mordred_table, on='smiles_r')
print(f"  Total features available: {human_lombardo_mfp_Mordred.shape[1] - 8} (Morgan + Mordred + CE50)")

# ============================================================================
# Step 6: Calculate and Save Median Feature Values per Animal
# ============================================================================

print("\n" + "="*80)
print("Step 6: Calculating Median Feature Values for Each Animal")
print("="*80)

animals = ['dog', 'monkey', 'rat']

for animal in animals:
    print(f"\n  Processing {animal}...")

    # For rat, we need CE50 features in the data
    if animal == 'rat':
        # Ensure CE50 features are in the data for rat
        data_for_median = human_lombardo_mfp_Mordred.copy()
        X = find_median_tables(data_for_median, animal, ce50_data=ce50_features)
        output_file = f"Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model_ce50.csv"
    else:
        X = find_median_tables(human_lombardo_mfp_Mordred, animal, ce50_data=None)
        output_file = f"Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model.csv"

    # Calculate median across all compounds
    X_median = pd.DataFrame(X.median()).T

    # Save to CSV
    X_median.to_csv(output_file, index=False)
    print(f"    Saved median values ({len(X_median.columns)} features) to: {output_file}")

# ============================================================================
# Step 7: Generate Artificial Animal Predictions
# ============================================================================

print("\n" + "="*80)
print("Step 7: Generating Artificial Animal Predictions")
print("="*80)

# Prepare data with all features
data = human_lombardo_mfp_Mordred.copy()

# Generate predictions for all 9 endpoints
predictions_dict = {}

# RAT predictions (with CE50-enhanced models)
print("\n  RAT predictions (using CE50-enhanced models):")
for endpoint in ["VDss_L_kg", "CL_mL_min_kg", "fup"]:
    col_name = f"rat_{endpoint}"
    print(f"    Predicting {col_name}...")
    predictions = predict_animal_endpoint(data, "rat", endpoint, ce50_data=ce50_features)
    predictions_dict[col_name] = predictions

# DOG predictions (baseline models)
print("\n  DOG predictions (using baseline models):")
for endpoint in ["VDss_L_kg", "CL_mL_min_kg", "fup"]:
    col_name = f"dog_{endpoint}"
    print(f"    Predicting {col_name}...")
    predictions = predict_animal_endpoint(data, "dog", endpoint, ce50_data=None)
    predictions_dict[col_name] = predictions

# MONKEY predictions (baseline models)
print("\n  MONKEY predictions (using baseline models):")
for endpoint in ["VDss_L_kg", "CL_mL_min_kg", "fup"]:
    col_name = f"monkey_{endpoint}"
    print(f"    Predicting {col_name}...")
    predictions = predict_animal_endpoint(data, "monkey", endpoint, ce50_data=None)
    predictions_dict[col_name] = predictions

# ============================================================================
# Step 8: Create Output DataFrame and Save
# ============================================================================

print("\n" + "="*80)
print("Step 8: Creating Output DataFrame")
print("="*80)

# Start with human data
output_data = human_lombardo[['smiles_r', 'human_VDss_L_kg', 'human_CL_mL_min_kg',
                               'human_fup', 'human_mrt', 'human_thalf']].copy()

# Add CE50 features
output_data = pd.merge(output_data, ce50_features, on='smiles_r', how='left')

# Add artificial animal predictions
for col_name, predictions in predictions_dict.items():
    output_data[col_name] = predictions

print(f"  Output DataFrame shape: {output_data.shape}")
print(f"  Total columns: {len(output_data.columns)}")
print(f"  Breakdown:")
print(f"    - SMILES: 1")
print(f"    - Human PK: 5 (VDss, CL, fup, MRT, thalf)")
print(f"    - CE50 features: 3 (ce50, pce50, confidence)")
print(f"    - Rat predictions (CE50-enhanced): 3")
print(f"    - Dog predictions (baseline): 3")
print(f"    - Monkey predictions (baseline): 3")

# Save to temporary file (as requested)
output_file = "artificial_animal_data_with_ce50_TEMP.csv"
output_data.to_csv(output_file, index=False)
print(f"\n  Saved to temporary file: {output_file}")
print(f"  (This will be used in the next step for human model training)")

# ============================================================================
# Step 9: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nArtificial Animal Prediction Statistics:")
for animal in ['rat', 'dog', 'monkey']:
    print(f"\n  {animal.upper()}:")
    for endpoint in ["VDss_L_kg", "CL_mL_min_kg", "fup"]:
        col_name = f"{animal}_{endpoint}"
        values = output_data[col_name]
        print(f"    {col_name:25s}: mean={values.mean():7.3f}, std={values.std():6.3f}, "
              f"min={values.min():7.3f}, max={values.max():7.3f}")

print("\nCE50 Features:")
for col in ['ce50', 'pce50', 'confidence']:
    if col in output_data.columns:
        values = output_data[col].dropna()
        print(f"  {col:15s}: mean={values.mean():7.3f}, std={values.std():6.3f}, "
              f"coverage={100*len(values)/len(output_data):5.1f}%")

# ============================================================================
# Step 10: Final Output Summary
# ============================================================================

print("\n" + "="*80)
print("GENERATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. Median_mordred_values_dog_for_artificial_animal_data_mfp_mrd_model.csv")
print("  2. Median_mordred_values_monkey_for_artificial_animal_data_mfp_mrd_model.csv")
print("  3. Median_mordred_values_rat_for_artificial_animal_data_mfp_mrd_model_ce50.csv")
print("  4. artificial_animal_data_with_ce50_TEMP.csv (main output)")
print()
print("Key Features:")
print("  - RAT: CE50-enhanced models (542 features including ce50, pce50, confidence)")
print("  - DOG: Baseline models (no CE50)")
print("  - MONKEY: Baseline models (no CE50)")
print()
print("Next Steps:")
print("  1. Use artificial_animal_data_with_ce50_TEMP.csv for human model training")
print("  2. Compare performance against baseline models without CE50")
print("="*80)
