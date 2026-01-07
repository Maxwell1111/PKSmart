"""
CE50 Prediction from SMILES Strings using Machine Learning

CE50 = Collision Energy required to fragment 50% of parent ions in mass spectrometry

This script builds regression models to predict CE50 (collision energy for 50% fragmentation)
from molecular SMILES strings. CE50 is a mass spectrometry property that depends on molecular
structure, bond strengths, and gas-phase stability.

It uses Morgan Fingerprints for molecular featurization and compares Random Forest and XGBoost regressors.

Author: Senior Bioinformatician
Date: 2026-01-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_validate_data(filepath):
    """
    Load CSV file and validate the required columns.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing SMILES and CE50 data

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with 'smiles' and 'ce50' columns
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ['smiles', 'ce50']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    print(f"Loaded {len(df)} compounds")
    return df


def validate_smiles(smiles_list):
    """
    Validate SMILES strings and return valid indices.

    Parameters:
    -----------
    smiles_list : list or pd.Series
        List of SMILES strings to validate

    Returns:
    --------
    list
        Boolean list indicating valid SMILES
    """
    valid_mask = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        valid_mask.append(mol is not None)
    return valid_mask


def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to Morgan Fingerprint.

    Parameters:
    -----------
    smiles : str
        SMILES string representation of molecule
    radius : int
        Radius for Morgan fingerprint (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)

    Returns:
    --------
    np.array
        Binary fingerprint array, or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate Morgan fingerprint (ECFP4 equivalent with radius=2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def featurize_molecules(smiles_list, radius=2, n_bits=2048):
    """
    Convert list of SMILES to feature matrix using Morgan Fingerprints.

    Parameters:
    -----------
    smiles_list : list or pd.Series
        List of SMILES strings
    radius : int
        Radius for Morgan fingerprint (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)

    Returns:
    --------
    np.array
        Feature matrix (n_samples, n_bits)
    """
    print("Generating Morgan Fingerprints (Radius 2, 2048 bits)...")
    fingerprints = []

    for smiles in smiles_list:
        fp = smiles_to_morgan_fp(smiles, radius, n_bits)
        fingerprints.append(fp)

    return np.array(fingerprints)


def convert_to_pce50(ce50_values):
    """
    Convert CE50 values to pCE50 (-log10(CE50)) for better distribution.

    Parameters:
    -----------
    ce50_values : np.array or pd.Series
        CE50 values in original units

    Returns:
    --------
    np.array
        pCE50 values
    """
    print("Converting CE50 to pCE50 (-log10(CE50)) for normalized distribution...")
    # Ensure positive values
    ce50_values = np.array(ce50_values)
    if np.any(ce50_values <= 0):
        print("Warning: Found non-positive CE50 values. Filtering them out.")
        ce50_values = np.where(ce50_values > 0, ce50_values, np.nan)

    pce50 = -np.log10(ce50_values)
    return pce50


def build_random_forest_pipeline():
    """
    Build Random Forest pipeline with StandardScaler.

    Returns:
    --------
    Pipeline
        Scikit-learn pipeline with scaler and Random Forest regressor
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    return pipeline


def build_xgboost_pipeline():
    """
    Build XGBoost pipeline with StandardScaler.

    Returns:
    --------
    Pipeline
        Scikit-learn pipeline with scaler and XGBoost regressor
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    return pipeline


def optimize_random_forest(X_train, y_train):
    """
    Optimize Random Forest hyperparameters using RandomizedSearchCV.

    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training targets

    Returns:
    --------
    GridSearchCV
        Fitted GridSearchCV object with best estimator
    """
    print("\n" + "="*60)
    print("Optimizing Random Forest Regressor...")
    print("="*60)

    pipeline = build_random_forest_pipeline()

    # Parameter grid for Random Forest
    param_grid = {
        'rf__n_estimators': [100, 200, 300, 500],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2']
    }

    # Use RandomizedSearchCV for faster optimization
    grid_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Random Forest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")

    return grid_search


def optimize_xgboost(X_train, y_train):
    """
    Optimize XGBoost hyperparameters using RandomizedSearchCV.

    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training targets

    Returns:
    --------
    GridSearchCV
        Fitted GridSearchCV object with best estimator
    """
    print("\n" + "="*60)
    print("Optimizing XGBoost Regressor...")
    print("="*60)

    pipeline = build_xgboost_pipeline()

    # Parameter grid for XGBoost
    param_grid = {
        'xgb__n_estimators': [100, 200, 300, 500],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Use RandomizedSearchCV for faster optimization
    grid_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest XGBoost parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance on test set.

    Parameters:
    -----------
    model : fitted estimator
        Trained model
    X_test : np.array
        Test features
    y_test : np.array
        Test targets
    model_name : str
        Name of the model for display

    Returns:
    --------
    dict
        Dictionary containing predictions and metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return {
        'predictions': y_pred,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


def plot_predictions(y_test, y_pred_rf, y_pred_xgb, metrics_rf, metrics_xgb):
    """
    Generate predicted vs actual scatter plots for both models.

    Parameters:
    -----------
    y_test : np.array
        Actual test values
    y_pred_rf : np.array
        Random Forest predictions
    y_pred_xgb : np.array
        XGBoost predictions
    metrics_rf : dict
        Random Forest metrics
    metrics_xgb : dict
        XGBoost metrics
    """
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Random Forest plot
    axes[0].scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual pCE50', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted pCE50', fontsize=12, fontweight='bold')
    axes[0].set_title('Random Forest: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add metrics text
    textstr_rf = f'R² = {metrics_rf["r2"]:.4f}\nMAE = {metrics_rf["mae"]:.4f}\nRMSE = {metrics_rf["rmse"]:.4f}'
    axes[0].text(0.05, 0.95, textstr_rf, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # XGBoost plot
    axes[1].scatter(y_test, y_pred_xgb, alpha=0.6, edgecolors='k', linewidths=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual pCE50', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted pCE50', fontsize=12, fontweight='bold')
    axes[1].set_title('XGBoost: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add metrics text
    textstr_xgb = f'R² = {metrics_xgb["r2"]:.4f}\nMAE = {metrics_xgb["mae"]:.4f}\nRMSE = {metrics_xgb["rmse"]:.4f}'
    axes[1].text(0.05, 0.95, textstr_xgb, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig('ce50_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ce50_predictions_comparison.png'")
    plt.show()


def main():
    """
    Main execution function for CE50 prediction pipeline.
    """
    print("\n" + "="*60)
    print("CE50 PREDICTION FROM SMILES - MACHINE LEARNING PIPELINE")
    print("="*60 + "\n")

    # 1. Load data
    df = load_and_validate_data('ce50_compounds.csv')

    # 2. Validate SMILES strings
    print("\nValidating SMILES strings...")
    valid_mask = validate_smiles(df['smiles'])
    df = df[valid_mask].copy()
    print(f"Removed {sum(~np.array(valid_mask))} invalid SMILES")
    print(f"Remaining compounds: {len(df)}")

    # 3. Convert CE50 to pCE50
    df['pce50'] = convert_to_pce50(df['ce50'].values)

    # Remove NaN values after pCE50 conversion
    df = df.dropna(subset=['pce50'])
    print(f"Compounds after removing NaN values: {len(df)}")

    # 4. Featurize molecules (Morgan Fingerprints)
    X = featurize_molecules(df['smiles'].values)
    y = df['pce50'].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # 5. Train-test split (80/20)
    print("\nPerforming 80/20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 6. Optimize and train Random Forest
    rf_grid_search = optimize_random_forest(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_

    # 7. Optimize and train XGBoost
    xgb_grid_search = optimize_xgboost(X_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_

    # 8. Evaluate models on test set
    rf_results = evaluate_model(best_rf_model, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(best_xgb_model, X_test, y_test, "XGBoost")

    # 9. Determine best model
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)

    if rf_results['r2'] > xgb_results['r2']:
        print(f"\nBest Model: Random Forest (R² = {rf_results['r2']:.4f})")
        best_model = best_rf_model
    else:
        print(f"\nBest Model: XGBoost (R² = {xgb_results['r2']:.4f})")
        best_model = best_xgb_model

    # 10. Visualize results
    plot_predictions(y_test, rf_results['predictions'], xgb_results['predictions'],
                    rf_results, xgb_results)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")

    return best_model, rf_results, xgb_results


if __name__ == "__main__":
    best_model, rf_results, xgb_results = main()
