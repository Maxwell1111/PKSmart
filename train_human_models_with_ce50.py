#!/usr/bin/env python3
"""
Train CE50-Enhanced Human PK Models

This script trains Random Forest models for human pharmacokinetic parameters
using Morgan fingerprints, Mordred descriptors, CE50 features, and
CE50-enhanced artificial animal data.

Endpoints:
- human_VDss_L_kg (Volume of distribution)
- human_CL_mL_min_kg (Clearance)
- human_fup (Fraction unbound in plasma)
- human_mrt (Mean residence time)
- human_thalf (Half-life)

Features:
- 152 Morgan fingerprints (2048-bit, variance threshold + correlation filtered)
- 352 Mordred descriptors (variance threshold + correlation filtered)
- 3 CE50 features (ce50, pce50, confidence)
- 9 Artificial animal PK predictions (rat, dog, monkey - VDss, CL, fup with CE50-enhanced rat)
- Total: 516 features

Requirements:
- pandas, numpy, scikit-learn
- rdkit
- mordred (pip install mordred-descriptor)

Usage:
    python train_human_models_with_ce50.py

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
print("CE50-ENHANCED HUMAN PK MODEL TRAINING")
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
    if endpoint == "human_fup":
        lst = [abs(a/b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100
    else:
        lst = [abs(10**a/10**b) for a, b in zip(pred, true)]
        newlist = [x for x in lst if min_val <= x <= max_val]
        return (len(newlist)/len(lst)) * 100


def calc_gmfe(pred, true, endpoint):
    """Calculate Geometric Mean Fold Error."""
    if endpoint == "human_fup":
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        mean_abs = np.mean(lst)
        return 10**mean_abs


def median_fold_change_error(pred, true, endpoint):
    """Calculate Median Fold Change Error."""
    if endpoint == "human_fup":
        lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs
    else:
        lst = [abs(np.log10(10**a/10**b)) for a, b in zip(pred, true)]
        median_abs = np.median(lst)
        return np.e**median_abs


def calc_bias(pred, true, endpoint):
    """Calculate prediction bias."""
    if endpoint == "human_fup":
        lst = [(a - b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias
    else:
        lst = [(10**a - 10**b) for a, b in zip(pred, true)]
        bias = np.median(lst)
        return bias


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
        feature_file = f"features_mfp_mordred_ce50_columns_{animal}_model.txt"
    else:
        model_file = f"log_{animal}_{endpoint}_model_FINAL.sav"
        scaler_file = f"scaler_{animal}.pkl"
        feature_file = f"features_mfp_mordred_columns_{animal}_model.txt"

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
        raise

    # Load feature columns
    try:
        with open(feature_file, "r") as file:
            file_lines = file.read()
            features = file_lines.split("\n")
            features = [f for f in features if f]  # Remove empty strings
    except FileNotFoundError:
        print(f"ERROR: Feature file not found: {feature_file}")
        raise

    # For rat with CE50, merge CE50 features into data first
    if animal == "rat" and ce50_data is not None:
        data_temp = pd.merge(data, ce50_data[['smiles_r', 'ce50', 'pce50', 'confidence']],
                            on='smiles_r', how='left')
    else:
        data_temp = data.copy()

    # Extract only the required features
    X = data_temp[features]

    # Handle missing values with median imputation
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = pd.DataFrame(imp.fit_transform(X))
    X_imputed.columns = X.columns
    X_imputed.index = X.index

    # Scale features
    X_scaled = scaler.transform(X_imputed)

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
print(f"  Loaded {len(human_lombardo)} compounds from Human_PK_data.csv")

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
# Step 3: Generate Mordred Descriptors
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
# Step 4: Generate Morgan Fingerprints
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
# Step 6: Generate Artificial Animal Predictions
# ============================================================================

print("\n" + "="*80)
print("Step 6: Generating Artificial Animal Predictions")
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

# Add predictions to dataframe
for col_name, predictions in predictions_dict.items():
    human_lombardo_mfp_Mordred[col_name] = predictions

print(f"\n  Added 9 artificial animal predictions")

# ============================================================================
# Step 7: Prepare Feature List
# ============================================================================

print("\n" + "="*80)
print("Step 7: Preparing Feature List")
print("="*80)

# Create feature list
Mordred_columns = new_Mordred_table.columns[:-1].tolist()
mfp_columns = new_Morganfingerprint_table.columns[:-1].tolist()
ce50_columns = ["ce50", "pce50", "confidence"]
animal_columns = ["dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup",
                  "monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup",
                  "rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"]

features_mfp_mordred_animal_ce50_columns = Mordred_columns + mfp_columns + ce50_columns + animal_columns

print(f"\n  Feature breakdown:")
print(f"    Mordred descriptors: {len(Mordred_columns)}")
print(f"    Morgan fingerprints: {len(mfp_columns)}")
print(f"    CE50 features: {len(ce50_columns)}")
print(f"    Artificial animal data: {len(animal_columns)}")
print(f"    TOTAL FEATURES: {len(features_mfp_mordred_animal_ce50_columns)}")

# Save feature list
with open("features_mfp_mordred_animal_ce50_columns_human.txt", "w") as f:
    for item in features_mfp_mordred_animal_ce50_columns:
        f.write(item + "\n")
print(f"  Saved feature list to: features_mfp_mordred_animal_ce50_columns_human.txt")

# ============================================================================
# Step 8: Train Models with Nested Cross-Validation
# ============================================================================

print("\n" + "="*80)
print("Step 8: Training Models with Nested 5-Fold Cross-Validation")
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
detail_list_batch = []
model_parameters_batch = []
endpoints = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]

for endpoint in endpoints:
    print(f"\n{'='*80}")
    print(f"Training models for: {endpoint}")
    print(f"{'='*80}")

    baseline = 10.00
    data = human_lombardo_mfp_Mordred
    features = features_mfp_mordred_animal_ce50_columns

    # Filter data for this endpoint
    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    print(f"  Compounds with {endpoint} data: {len(df)}")

    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

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
                pickle.dump(classifier, open(f"log_{endpoint}_withanimaldata_artificial_ce50_model.sav", 'wb'))

            # Store results
            row = ["HeldOut", endpoint, i, split, fold_2, fold_3, fold_5, gmfe, mfe, bias, rmse, r2]
            list_of_lists.append(row)

            # Store model parameters
            param_row = [endpoint, i, split, classifier.n_estimators, classifier.max_features,
                        classifier.min_samples_split, classifier.max_depth, classifier.bootstrap]
            model_parameters_batch.append(param_row)

            # Store detailed predictions
            for smiles_r, pred, true in zip(df.iloc[test_index]['smiles_r'], y_pred, Y_test):
                detail_list_batch.append([endpoint, i, split, smiles_r, pred, true])

    print(f"\n  Best GMFE for {endpoint}: {baseline:.4f}")

# ============================================================================
# Step 9: Save Results
# ============================================================================

print("\n" + "="*80)
print("Step 9: Saving Results")
print("="*80)

results = pd.DataFrame(
    list_of_lists,
    columns=["HeldOut", "endpoint", "random_state", "split", "fold_2", "fold_3", "fold_5",
             "gmfe", "mfe", "bias", "rmse", "r2"]
)
results.to_csv("Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv", index=False)
print("  Saved detailed results to: Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv")

# Save model parameters
model_parameters = pd.DataFrame(
    model_parameters_batch,
    columns=["endpoint", "random_state", "split", "n_estimators", "max_features",
             "min_samples_split", "max_depth", "bootstrap"]
)
model_parameters.to_csv("model_parameters_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv", index=False)
print("  Saved model parameters to: model_parameters_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv")

# Save detailed predictions
detail_list = pd.DataFrame(
    detail_list_batch,
    columns=["endpoint", "random_state", "split", "smiles_r", "pred", "true"]
)
detail_list.to_csv("detail_list_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv", index=False)
print("  Saved detailed predictions to: detail_list_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv")

# ============================================================================
# Step 10: Train Final Models on Full Dataset
# ============================================================================

print("\n" + "="*80)
print("Step 10: Training Final Models on Full Dataset")
print("="*80)

for endpoint in endpoints:
    print(f"\n  Training final model for: {endpoint}")

    data = human_lombardo_mfp_Mordred
    features = features_mfp_mordred_animal_ce50_columns

    df = data.dropna(subset=[endpoint]).reset_index(drop=True)
    X = df[features]
    Y = df[endpoint]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)

    # Save the scaler
    with open(f'artificial_animal_data_mfp_mrd_ce50_{endpoint}_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    print(f"    Saved scaler: artificial_animal_data_mfp_mrd_ce50_{endpoint}_scaler.pkl")

    X_train = X.values
    Y_train = Y.values.flatten()

    # Load best hyperparameters from saved model
    loaded_rf = pickle.load(open(f"log_{endpoint}_withanimaldata_artificial_ce50_model.sav", 'rb'))
    params = loaded_rf.get_params()

    # Train final model on all data
    classifier = RandomForestRegressor(**params)
    classifier.fit(X_train, Y_train)

    # Save final model
    pickle.dump(classifier, open(f"log_{endpoint}_withanimaldata_artificial_ce50_model_FINAL.sav", 'wb'))
    print(f"    Saved: log_{endpoint}_withanimaldata_artificial_ce50_model_FINAL.sav")
    print(f"    Hyperparameters: n_estimators={params['n_estimators']}, max_features={params['max_features']}, "
          f"min_samples_split={params['min_samples_split']}, max_depth={params['max_depth']}, "
          f"bootstrap={params['bootstrap']}")

# ============================================================================
# Step 11: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary = results.groupby("endpoint").mean()
summary_df = summary.reset_index()[["endpoint", "fold_2", "fold_3", "fold_5", "gmfe", "mfe", "rmse", "r2"]]
summary_df.to_csv("human_with_mfp_mordred_animal_artificial_ce50.csv", index=False)
print("\nMean performance across all CV folds:")
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. features_mfp_mordred_animal_ce50_columns_human.txt - Feature list (516 features)")
print("  2. artificial_animal_data_mfp_mrd_ce50_human_VDss_L_kg_scaler.pkl - VDss scaler")
print("  3. artificial_animal_data_mfp_mrd_ce50_human_CL_mL_min_kg_scaler.pkl - CL scaler")
print("  4. artificial_animal_data_mfp_mrd_ce50_human_fup_scaler.pkl - fup scaler")
print("  5. artificial_animal_data_mfp_mrd_ce50_human_mrt_scaler.pkl - MRT scaler")
print("  6. artificial_animal_data_mfp_mrd_ce50_human_thalf_scaler.pkl - thalf scaler")
print("  7. log_human_VDss_L_kg_withanimaldata_artificial_ce50_model_FINAL.sav - VDss model")
print("  8. log_human_CL_mL_min_kg_withanimaldata_artificial_ce50_model_FINAL.sav - CL model")
print("  9. log_human_fup_withanimaldata_artificial_ce50_model_FINAL.sav - fup model")
print(" 10. log_human_mrt_withanimaldata_artificial_ce50_model_FINAL.sav - MRT model")
print(" 11. log_human_thalf_withanimaldata_artificial_ce50_model_FINAL.sav - thalf model")
print(" 12. Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv - Detailed CV results")
print(" 13. human_with_mfp_mordred_animal_artificial_ce50.csv - Summary statistics")
print("\nTotal features: 516 (352 Mordred + 152 Morgan + 3 CE50 + 9 Animal)")
print("Feature breakdown:")
print("  - Mordred descriptors: 352")
print("  - Morgan fingerprints: 152")
print("  - CE50 features: 3 (ce50, pce50, confidence)")
print("  - Artificial animal data: 9")
print("    * rat_VDss_L_kg, rat_CL_mL_min_kg, rat_fup (CE50-enhanced)")
print("    * dog_VDss_L_kg, dog_CL_mL_min_kg, dog_fup (baseline)")
print("    * monkey_VDss_L_kg, monkey_CL_mL_min_kg, monkey_fup (baseline)")
print("="*80)
