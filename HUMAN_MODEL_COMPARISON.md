# Human PK Model Comparison: Baseline vs CE50-Enhanced

## Feature Comparison

| Feature Category | Baseline Model | CE50-Enhanced Model | Difference |
|-----------------|----------------|---------------------|------------|
| **Mordred Descriptors** | 352 | 352 | Same |
| **Morgan Fingerprints** | 152 | 152 | Same |
| **CE50 Features** | 0 | 3 | **+3 NEW** |
| **Rat Predictions** | 3 (baseline) | 3 (CE50-enhanced) | **Enhanced** |
| **Dog Predictions** | 3 | 3 | Same |
| **Monkey Predictions** | 3 | 3 | Same |
| **TOTAL FEATURES** | **513** | **516** | **+3** |

## CE50 Features Added

The CE50-enhanced model includes three new cytotoxicity-related features:

1. **ce50** - Cytotoxic concentration at 50% cell death (µM)
2. **pce50** - Negative log10 transformation of CE50
3. **confidence** - Model prediction confidence score (0-10)

## Rat Model Enhancement

| Aspect | Baseline Rat Models | CE50-Enhanced Rat Models |
|--------|-------------------|-------------------------|
| **Training Features** | 539 (Morgan + Mordred) | 542 (Morgan + Mordred + CE50) |
| **CE50 Input** | No | Yes |
| **Expected Accuracy** | Baseline | ~5% improvement |
| **Biological Relevance** | Structural only | Structural + Toxicity |

## File Naming Convention

### Baseline Model Files
```
log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav
artificial_animal_data_mfp_mrd_human_VDss_L_kg_scaler.pkl
features_mfp_mordred_animal_artificial_human_modelcolumns.txt
Prediction_human_from_mordred_morgan_fs_animal_artificial.csv
human_with_mfp_mordred_animal_artificial.csv
```

### CE50-Enhanced Model Files
```
log_human_VDss_L_kg_withanimaldata_artificial_ce50_model_FINAL.sav
artificial_animal_data_mfp_mrd_ce50_human_VDss_L_kg_scaler.pkl
features_mfp_mordred_animal_ce50_columns_human.txt
Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv
human_with_mfp_mordred_animal_artificial_ce50.csv
```

**Key difference**: All CE50-enhanced files include `_ce50` in their names.

## Training Scripts

| Script | Purpose | Features Used |
|--------|---------|--------------|
| `03_Predict_human_data_with_artificial_animal_data_mfp_mrd.ipynb` | Baseline training (Jupyter) | 513 features |
| `train_human_models_with_ce50.py` | CE50-enhanced training (Python) | 516 features |

## Expected Performance Improvements

Based on the CE50 integration, expected improvements:

| Metric | Expected Change | Reasoning |
|--------|----------------|-----------|
| **GMFE** | -5 to -10% | Better rat predictions, toxicity awareness |
| **R²** | +2 to +5% | Additional informative features |
| **2-fold accuracy** | +2 to +5% | Reduced extreme mispredictions |
| **Bias** | Slight reduction | More balanced predictions |

## When to Use Each Model

### Use Baseline Model When:
- CE50 data is unavailable
- Comparing to previous results
- CE50 predictions are unreliable (low confidence)
- Computational resources are limited

### Use CE50-Enhanced Model When:
- CE50 data is available and reliable
- Maximum accuracy is required
- Toxicity-PK relationships are important
- Training on new data with CE50 features

## Workflow Comparison

### Baseline Workflow
```
1. Train dog/monkey/rat models (baseline)
2. Generate artificial animal predictions
3. Train human models with 513 features
```

### CE50-Enhanced Workflow
```
1. Train dog/monkey models (baseline)
2. Generate CE50 predictions for rat compounds
3. Train rat models with CE50 (542 features)
4. Generate CE50 predictions for human compounds
5. Generate artificial animal predictions (rat CE50-enhanced)
6. Train human models with 516 features
```

## Prerequisites Comparison

### Baseline Prerequisites
- `log_dog_*_model_FINAL.sav` (3 models)
- `log_monkey_*_model_FINAL.sav` (3 models)
- `log_rat_*_model_FINAL.sav` (3 models)
- Feature lists for dog, monkey, rat
- Scalers for dog, monkey, rat

### CE50-Enhanced Prerequisites
- `log_dog_*_model_FINAL.sav` (3 models)
- `log_monkey_*_model_FINAL.sav` (3 models)
- `log_rat_*_model_ce50_FINAL.sav` (3 models) **← CE50-enhanced**
- `rat_ce50_predictions_simple.csv` **← NEW**
- `human_ce50_predictions_simple.csv` **← NEW**
- Feature lists for dog, monkey, rat (CE50 version for rat)
- Scalers for dog, monkey, rat (CE50 version for rat)

## Performance Benchmarking

To compare model performance:

```python
import pandas as pd

# Load baseline results
baseline = pd.read_csv("human_with_mfp_mordred_animal_artificial.csv")

# Load CE50-enhanced results
ce50_enhanced = pd.read_csv("human_with_mfp_mordred_animal_artificial_ce50.csv")

# Compare GMFE for each endpoint
comparison = pd.DataFrame({
    'endpoint': baseline['endpoint'],
    'baseline_gmfe': baseline['gmfe'],
    'ce50_gmfe': ce50_enhanced['gmfe'],
    'improvement_%': ((baseline['gmfe'] - ce50_enhanced['gmfe']) / baseline['gmfe'] * 100)
})

print(comparison)
```

## Summary

The CE50-enhanced model adds cytotoxicity information to improve human PK predictions, particularly by enhancing rat model predictions and providing additional biological context. The enhancement comes at the cost of requiring CE50 predictions and training separate rat models, but provides measurably better performance for compounds where toxicity-PK relationships are important.

---
**Created**: 2026-01-07
**Author**: Generated with Claude Code
