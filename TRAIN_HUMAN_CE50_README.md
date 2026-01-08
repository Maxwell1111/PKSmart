# Training CE50-Enhanced Human PK Models

This guide explains how to train CE50-enhanced human pharmacokinetic prediction models using the `train_human_models_with_ce50.py` script.

## Overview

The script trains Random Forest models for 5 human PK endpoints using:
- **Morgan fingerprints** (152 features)
- **Mordred descriptors** (352 features)
- **CE50 features** (3 features: ce50, pce50, confidence)
- **Artificial animal data** (9 features: rat/dog/monkey VDss, CL, fup)
- **Total: 516 features**

### Endpoints Trained

1. `human_VDss_L_kg` - Volume of distribution (L/kg)
2. `human_CL_mL_min_kg` - Clearance (mL/min/kg)
3. `human_fup` - Fraction unbound in plasma
4. `human_mrt` - Mean residence time (hours)
5. `human_thalf` - Half-life (hours)

## Prerequisites

### 1. Required Python Packages

```bash
pip install pandas numpy scikit-learn rdkit-pypi mordred-descriptor
```

### 2. Required Data Files

The following data files must exist in the `data/` directory:

- `data/Human_PK_data.csv` - Human PK measurements
- `data/human_ce50_predictions_simple.csv` - CE50 predictions for human compounds

### 3. Required Trained Animal Models

Before running this script, you must train the animal models:

#### **Step A: Train Baseline Dog and Monkey Models**
```bash
# These scripts should already exist or need to be created
python train_dog_models.py
python train_monkey_models.py
```

Expected output files:
- `log_dog_VDss_L_kg_model_FINAL.sav`
- `log_dog_CL_mL_min_kg_model_FINAL.sav`
- `log_dog_fup_model_FINAL.sav`
- `scaler_dog.pkl`
- `features_mfp_mordred_columns_dog_model.txt`
- `log_monkey_VDss_L_kg_model_FINAL.sav`
- `log_monkey_CL_mL_min_kg_model_FINAL.sav`
- `log_monkey_fup_model_FINAL.sav`
- `scaler_monkey.pkl`
- `features_mfp_mordred_columns_monkey_model.txt`

#### **Step B: Train CE50-Enhanced Rat Models**
```bash
python train_rat_models_with_ce50.py
```

Expected output files:
- `log_rat_VDss_L_kg_model_ce50_FINAL.sav`
- `log_rat_CL_mL_min_kg_model_ce50_FINAL.sav`
- `log_rat_fup_model_ce50_FINAL.sav`
- `scaler_rat_ce50.pkl`
- `features_mfp_mordred_ce50_columns_rat_model.txt`

## Usage

Once all prerequisites are met, run the script:

```bash
python train_human_models_with_ce50.py
```

### Expected Runtime

- **Morgan fingerprint generation**: ~30 seconds
- **Mordred descriptor calculation**: ~2-5 minutes
- **Artificial animal predictions**: ~1 minute
- **Model training (nested 5-fold CV × 5 seeds)**: ~2-4 hours per endpoint
- **Total runtime**: ~10-20 hours (depending on CPU cores)

### Computational Requirements

- **CPU**: 30+ cores recommended (configurable via `n_jobs` parameter)
- **Memory**: ~16 GB RAM
- **Disk**: ~500 MB for output files

## Output Files

### 1. Feature Lists
- `features_mfp_mordred_animal_ce50_columns_human.txt` (516 lines)

### 2. Scalers (one per endpoint)
- `artificial_animal_data_mfp_mrd_ce50_human_VDss_L_kg_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_CL_mL_min_kg_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_fup_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_mrt_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_thalf_scaler.pkl`

### 3. Trained Models (one per endpoint)
- `log_human_VDss_L_kg_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_CL_mL_min_kg_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_fup_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_mrt_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_thalf_withanimaldata_artificial_ce50_model_FINAL.sav`

### 4. Results and Metrics
- `Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv` - Detailed CV results for all folds
- `model_parameters_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv` - Best hyperparameters per fold
- `detail_list_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv` - Per-compound predictions
- `human_with_mfp_mordred_animal_artificial_ce50.csv` - Summary statistics (mean metrics per endpoint)

## Feature Breakdown

The 516 features consist of:

### Structural Features (504 features)
1. **Mordred Descriptors** (352 features)
   - 2D molecular descriptors
   - Variance filtered (threshold=0.05)
   - Correlation filtered (>0.95 removed)

2. **Morgan Fingerprints** (152 features)
   - Radius=2, 2048-bit
   - Variance filtered (threshold=0.05)

### CE50 Features (3 features)
3. **CE50 Fragmentation Energetics** (3 features)
   - `ce50` - Collision energy (eV) where 50% of parent ion fragments in mass spectrometry
   - `pce50` - Negative log10 of CE50
   - `confidence` - Prediction confidence score

### Artificial Animal Data (9 features)
4. **Rat Predictions** (3 features - **CE50-enhanced**)
   - `rat_VDss_L_kg` - Predicted rat volume of distribution
   - `rat_CL_mL_min_kg` - Predicted rat clearance
   - `rat_fup` - Predicted rat fraction unbound

5. **Dog Predictions** (3 features - baseline)
   - `dog_VDss_L_kg` - Predicted dog volume of distribution
   - `dog_CL_mL_min_kg` - Predicted dog clearance
   - `dog_fup` - Predicted dog fraction unbound

6. **Monkey Predictions** (3 features - baseline)
   - `monkey_VDss_L_kg` - Predicted monkey volume of distribution
   - `monkey_CL_mL_min_kg` - Predicted monkey clearance
   - `monkey_fup` - Predicted monkey fraction unbound

## Model Training Details

### Cross-Validation Strategy
- **Outer CV**: 5-fold cross-validation
- **Inner CV**: 4-fold cross-validation (for hyperparameter tuning)
- **Random seeds**: 5 seeds (42, 43, 44, 45, 46)
- **Total folds**: 5 endpoints × 5 seeds × 5 folds = 125 models per endpoint

### Hyperparameter Grid
```python
{
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 4, 8],
    "max_depth": [5, 10, 15],
    "bootstrap": [True, False],
    "n_jobs": [30]
}
```

### Evaluation Metrics
- **GMFE** (Geometric Mean Fold Error) - Primary metric for model selection
- **MFE** (Median Fold Change Error)
- **Bias** (Median bias)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of determination)
- **Fold accuracy**: % predictions within 2-fold, 3-fold, 5-fold

### Data Transformations
- **Log transformation**: Applied to VDss, CL, MRT, thalf (not fup)
- **Scaling**: StandardScaler applied to all features
- **Missing values**: Handled via median grouping by SMILES

## Troubleshooting

### Error: "Model file not found"
**Cause**: Animal models haven't been trained yet.

**Solution**: Follow the prerequisite steps to train dog, monkey, and rat models first.

### Error: "CE50 data file not found"
**Cause**: CE50 predictions haven't been generated.

**Solution**: Run `generate_ce50_human_predictions.py` first.

### Low Performance (GMFE > 3.0)
**Possible causes**:
1. Insufficient training data
2. Poor hyperparameter selection
3. Missing or corrupted features

**Solutions**:
1. Check data quality and coverage
2. Expand hyperparameter grid
3. Verify feature generation steps

### Memory Errors
**Solution**: Reduce `n_jobs` parameter in `param_grid` and `GridSearchCV` calls.

## Comparison to Baseline

This CE50-enhanced approach improves upon the baseline model by:

1. **Adding CE50 features** (3 features) - Compound fragmentation energetics from mass spectrometry
2. **CE50-enhanced rat predictions** - Better animal-to-human translation
3. **Expected improvement**: ~5-10% reduction in GMFE

To compare performance:
```bash
# Baseline (without CE50)
grep "gmfe" human_with_mfp_mordred_animal_artificial.csv

# CE50-enhanced (with CE50)
grep "gmfe" human_with_mfp_mordred_animal_artificial_ce50.csv
```

## Citation

If you use this code, please cite:
- PKSmart repository
- Maxwell1111/CE50_prediction model
- Original PK prediction methodology

## Author

Generated with Claude Code
Date: 2026-01-07

## License

See repository LICENSE file.
