# Training CE50-Enhanced Rat PK Models

## Overview

The `train_rat_models_with_ce50.py` script trains Random Forest models for three rat pharmacokinetic endpoints using molecular descriptors enhanced with CE50 features.

## Endpoints

1. **rat_VDss_L_kg** - Volume of distribution (L/kg)
2. **rat_CL_mL_min_kg** - Clearance (mL/min/kg)
3. **rat_fup** - Fraction unbound in plasma

## Features (542 total)

### 1. Mordred Descriptors (386 features)
- All 2D molecular descriptors from Mordred library
- Filtered with variance threshold (>0.05)
- Correlation filtered (removed pairs with r>0.95)

### 2. Morgan Fingerprints (153 features)
- Radius: 2
- Original bits: 2048
- Filtered with variance threshold (>0.05)

### 3. CE50 Features (3 features)
- **ce50**: Collision energy in eV
- **pce50**: -log10(CE50)
- **confidence**: Prediction confidence score (0-6 scale)

## Requirements

```bash
pip install pandas numpy scikit-learn rdkit mordred-descriptor
```

Or with conda:
```bash
conda install pandas numpy scikit-learn rdkit
conda install -c conda-forge mordred-descriptor
```

## Input Files

1. `data/Animal_PK_data.csv` - Animal PK dataset with rat endpoints
2. `data/rat_ce50_predictions_simple.csv` - CE50 predictions for rat compounds

## Output Files

### Models
- `log_rat_VDss_L_kg_model_ce50_FINAL.sav` - Final VDss model
- `log_rat_CL_mL_min_kg_model_ce50_FINAL.sav` - Final CL model
- `log_rat_fup_model_ce50_FINAL.sav` - Final fup model

### Supporting Files
- `scaler_rat_ce50.pkl` - StandardScaler for feature normalization
- `features_mfp_mordred_ce50_columns_rat_model.txt` - List of all 542 features
- `Prediction_rat_from_mordred_morgan_fs_ce50.csv` - Detailed CV results

## Training Method

### Nested Cross-Validation
- **Outer loop**: 5-fold cross-validation
- **Inner loop**: 4-fold cross-validation for hyperparameter tuning
- **Random seeds**: 5 seeds (42-46) for robustness
- **Total evaluations**: 5 seeds × 5 folds = 25 evaluations per endpoint

### Hyperparameter Grid
```python
{
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 4, 8],
    "max_depth": [5, 10, 15],
    "bootstrap": [True, False]
}
```

### Model Selection
- Models selected based on lowest **GMFE** (Geometric Mean Fold Error)
- Final models trained on full dataset with best hyperparameters

## Evaluation Metrics

- **fold_2**: % predictions within 2-fold of true value
- **fold_3**: % predictions within 3-fold of true value
- **fold_5**: % predictions within 5-fold of true value
- **GMFE**: Geometric Mean Fold Error
- **MFE**: Median Fold Change Error
- **bias**: Median prediction bias
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

## Usage

### Training
```bash
python train_rat_models_with_ce50.py
```

### Expected Runtime
- Approximately 30-60 minutes depending on hardware
- Uses parallel processing (n_jobs=30 for GridSearchCV)

### Loading Trained Models

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open('log_rat_VDss_L_kg_model_ce50_FINAL.sav', 'rb'))

# Load scaler
scaler = pickle.load(open('scaler_rat_ce50.pkl', 'rb'))

# Load feature names
with open('features_mfp_mordred_ce50_columns_rat_model.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Prepare your data
# X should be a DataFrame with columns matching feature_names
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict(X_scaled)

# For VDss and CL, convert from log10 scale
predictions_original_scale = 10**predictions
```

## Comparison with Baseline

### Baseline Features (539)
- 386 Mordred descriptors
- 153 Morgan fingerprints

### CE50-Enhanced Features (542)
- 386 Mordred descriptors
- 153 Morgan fingerprints
- **3 CE50 features (NEW)**

### Expected Improvements
- Better predictions for compounds with unusual collision energies
- Improved applicability domain coverage
- Additional physicochemical information from mass spectrometry

## Notes

1. **Log transformation**: VDss and CL are log10-transformed before training
2. **Duplicate handling**: Compounds with duplicate SMILES are grouped and median values used
3. **Missing data**: Each endpoint model uses only compounds with non-missing values for that endpoint
4. **Scaler**: All features are standardized using StandardScaler before training

## Citation

If you use these models, please cite:
- Maxwell1111/CE50_prediction repository for CE50 models
- Original PKSmart publication for baseline models

## Contact

Generated with Claude Code (claude-sonnet-4-5)
Date: 2026-01-07
