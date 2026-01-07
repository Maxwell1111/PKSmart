# Human CE50-Enhanced PK Model Training (Simplified)

## Overview

The `train_human_models_ce50_simple.py` script trains CE50-enhanced Random Forest models for **5 human pharmacokinetic endpoints** without using artificial animal data. This is a simplified demonstration version to quickly show the value of CE50 features.

## Key Features

- **No Artificial Animal Data**: Uses only human PK data and CE50 predictions
- **5 PK Endpoints**: VDss, CL, fup, mrt, thalf
- **507 Total Features**: 352 Mordred + 152 Morgan + 3 CE50
- **Robust Validation**: Nested 5-fold CV with 5 random seeds
- **Endpoint-Specific Scalers**: Better practice than shared scalers

## Endpoints Trained

| Endpoint | Description | Log Transform | Count Function |
|----------|-------------|---------------|----------------|
| `human_VDss_L_kg` | Volume of distribution | Yes | Fold-change |
| `human_CL_mL_min_kg` | Clearance | Yes | Fold-change |
| `human_fup` | Fraction unbound | No | Direct ratio |
| `human_mrt` | Mean residence time | Yes | Fold-change |
| `human_thalf` | Half-life | Yes | Fold-change |

## Feature Composition

```
Total: 507 features
├── Mordred descriptors: 352
│   └── Filtered from ~1600 using variance (0.05) + correlation (>0.95)
├── Morgan fingerprints: 152
│   └── Filtered from 2048 using variance (0.05)
└── CE50 features: 3
    ├── ce50 (raw value)
    ├── pce50 (log-transformed)
    └── confidence (prediction quality)
```

## Quick Start

```bash
# Navigate to project directory
cd /Users/aardeshiri/PKSmart

# Verify input files exist
ls -lh data/Human_PK_data.csv
ls -lh data/human_ce50_predictions_simple.csv

# Run training (takes ~20-35 minutes)
python train_human_models_ce50_simple.py
```

## Output Files

After successful completion, you'll have **13 files**:

### 1. Feature List (1 file)
```
features_mfp_mordred_ce50_columns_human.txt (507 lines)
```

### 2. Scalers (5 files)
```
human_ce50_human_VDss_L_kg_scaler.pkl
human_ce50_human_CL_mL_min_kg_scaler.pkl
human_ce50_human_fup_scaler.pkl
human_ce50_human_mrt_scaler.pkl
human_ce50_human_thalf_scaler.pkl
```

### 3. Final Models (5 files)
```
log_human_human_VDss_L_kg_ce50_model_FINAL.sav
log_human_human_CL_mL_min_kg_ce50_model_FINAL.sav
log_human_human_fup_ce50_model_FINAL.sav
log_human_human_mrt_ce50_model_FINAL.sav
log_human_human_thalf_ce50_model_FINAL.sav
```

### 4. Results (2 files)
```
Prediction_human_from_mordred_morgan_ce50.csv  # Detailed CV results (125 rows)
human_ce50.csv                                 # Summary statistics
```

## Performance Metrics

Each model is evaluated using:

- **GMFE** (Geometric Mean Fold Error) - Primary metric
- **R²** (R-squared) - Variance explained
- **RMSE** (Root Mean Squared Error)
- **Fold-2/3/5** - % predictions within 2x/3x/5x of true value
- **MFE** (Median Fold Change Error)
- **Bias** - Systematic over/under-prediction

## Training Process

```
Step 1: Load Human PK Data
  ├── Read data/Human_PK_data.csv
  ├── Filter for 5 endpoints
  ├── Log-transform VDss, CL, mrt, thalf
  └── Group by SMILES (handle duplicates)

Step 2: Load CE50 Predictions
  └── Read data/human_ce50_predictions_simple.csv

Step 3: Generate Mordred Descriptors (~2-5 min)
  ├── Calculate ~1600 descriptors
  ├── Drop NaN columns
  ├── Variance threshold (0.05)
  └── Correlation filtering (>0.95)
  Result: 352 descriptors

Step 4: Generate Morgan Fingerprints
  ├── 2048-bit fingerprints
  └── Variance threshold (0.05)
  Result: 152 fingerprints

Step 5: Merge Features
  └── Morgan + Mordred + CE50 = 507 features

Step 6: Nested Cross-Validation (~15-30 min)
  For each endpoint:
    For each of 5 random seeds (42-46):
      For each of 5 outer folds:
        ├── Inner 4-fold CV for hyperparameter tuning
        ├── Train with best hyperparameters
        └── Evaluate on held-out fold
  Total: 5 endpoints × 5 seeds × 5 folds = 125 evaluations

Step 7: Save CV Results
  └── Prediction_human_from_mordred_morgan_ce50.csv

Step 8: Train Final Models
  ├── Load best hyperparameters from CV
  ├── Train on full dataset
  └── Save 5 final models + 5 scalers

Step 9: Summary Statistics
  └── human_ce50.csv
```

## Hyperparameter Grid

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

## Comparison with Rat Models

| Aspect | Rat Models | Human Models |
|--------|------------|--------------|
| Endpoints | 3 | 5 |
| Features | 542 | 507 |
| Log Transform | VDss, CL | VDss, CL, mrt, thalf |
| Scalers | 1 shared | 5 endpoint-specific |
| Training Time | ~10-20 min | ~20-35 min |
| Artificial Data | No | No |

## Verification Commands

```bash
# Count output files (should be 13)
ls -1 features_mfp_mordred_ce50_columns_human.txt \
     human_ce50_*.pkl \
     log_human_*_ce50_model_FINAL.sav \
     Prediction_human_from_mordred_morgan_ce50.csv \
     human_ce50.csv | wc -l

# Check feature count (should be 507)
wc -l features_mfp_mordred_ce50_columns_human.txt

# View summary statistics
cat human_ce50.csv
```

## Helper Functions

The script includes 7 helper functions from the baseline implementation:

1. `fs_variance()` - Variance threshold filtering
2. `get_pairwise_correlation()` - Calculate feature correlations
3. `determine_high_cor_pair()` - Select features to remove
4. `count()` - Calculate fold-change percentages
5. `calc_gmfe()` - Geometric Mean Fold Error
6. `median_fold_change_error()` - Median Fold Error
7. `calc_bias()` - Prediction bias

## Troubleshooting

### "mordred package not found"
```bash
pip install mordred-descriptor
# or
conda install -c conda-forge mordred-descriptor
```

### "File not found" errors
Ensure you're in the PKSmart directory:
```bash
cd /Users/aardeshiri/PKSmart
pwd  # Should show: /Users/aardeshiri/PKSmart
```

### Low performance (high GMFE)
This is normal for:
- Small datasets
- Complex endpoints (mrt, thalf)
- Cross-validation (more conservative than training set)

Check `human_ce50.csv` for summary statistics.

## Next Steps

After training, you can:

1. **Compare with baseline models** (without CE50)
2. **Analyze feature importance** using model.feature_importances_
3. **Make predictions** on new compounds
4. **Integrate into web application** (see CE50_INTEGRATION_GUIDE.md)

## Author

Generated with Claude Code on 2026-01-07

## Related Files

- `train_rat_models_with_ce50.py` - Rat model training
- `run_ce50_rat_predictions.py` - Generate CE50 predictions
- `CE50_INTEGRATION_GUIDE.md` - Integration documentation
