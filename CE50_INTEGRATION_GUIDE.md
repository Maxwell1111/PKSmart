# CE50 Integration into PKSmart - Implementation Guide

## Overview

This guide describes the integration of CE50 predictions into PKSmart to enhance pharmacokinetic (PK) predictions through a two-stage approach:

1. **Stage 1**: Enhance rat PK models with CE50 features
2. **Stage 2**: Use enhanced rat predictions + CE50 to improve human PK models
3. **Stage 3**: Comprehensive comparison and evaluation

**Generated:** 2026-01-06
**Author:** Claude Code Implementation

---

## Implementation Status

### ✅ STAGE 1: Rat Model Enhancement

#### Step 1.1: Generate CE50 Predictions for Rat Dataset
**Status:** ✅ COMPLETED
**Notebook:** `02a_Generate_CE50_for_rat_data.ipynb`

**What it does:**
- Loads rat compounds from `data/Animal_PK_data.csv` (~324 compounds)
- Generates dual Morgan fingerprints (binary + count)
- Runs CE50 ensemble predictor (4 models: RF/XGB × Binary/Count)
- Outputs CE50 (eV), pCE50, and confidence scores
- Creates diagnostic visualizations

**Outputs:**
- `data/rat_ce50_predictions.csv` - Full results with all model predictions
- `data/rat_ce50_predictions_simple.csv` - Essential columns only
- `rat_ce50_predictions_diagnostics.png` - 6-panel visualization

**How to run:**
```bash
jupyter notebook 02a_Generate_CE50_for_rat_data.ipynb
# Or run all cells in Jupyter Lab
```

**Expected results:**
- ~324 CE50 predictions for rat compounds
- CE50 range: 10-50 eV (typical for small molecules)
- pCE50 range: ~-1.3 to -1.7
- Confidence distribution: High/Medium/Low

---

#### Step 1.2: Retrain Rat Models with CE50 Features
**Status:** 🚧 IN PROGRESS
**Notebook:** `02_Predict_rat_data_with_CE50.ipynb`

**What it will do:**
- Load rat PK data + CE50 predictions from Step 1.1
- Generate Morgan fingerprints (153 bits) + Mordred descriptors (386)
- Merge CE50 features: ce50, pce50, confidence = **542 total features**
- Train 3 Random Forest models with nested 5-fold CV:
  - rat_VDss_L_kg
  - rat_CL_mL_min_kg
  - rat_fup
- Compare performance vs. baseline models (without CE50)

**Outputs:**
- `log_rat_VDss_L_kg_ce50_model_FINAL.sav`
- `log_rat_CL_mL_min_kg_ce50_model_FINAL.sav`
- `log_rat_fup_ce50_model_FINAL.sav`
- `scaler_rat_ce50.pkl`
- `features_mfp_mordred_ce50_columns_rat_model.txt` (542 lines)
- `Prediction_rat_from_mordred_morgan_fs_ce50.csv`
- `rat_ce50.csv` (performance metrics)

**Key modifications from original `02_Predict_rat_data.ipynb`:**
- Merge CE50 predictions before model training
- Update feature list to include ['ce50', 'pce50', 'confidence']
- Save models with `_ce50` suffix to distinguish from baseline

---

### ⏳ STAGE 2: Human Model Enhancement

#### Step 2.1: Generate CE50 Predictions for Human Dataset
**Status:** ⏳ PENDING
**Notebook:** `03a_Generate_CE50_for_human_data.ipynb`

**What it will do:**
- Load human PK data from `data/Human_PK_data.csv` (~1,283 compounds)
- Generate CE50 predictions using same ensemble approach as rat dataset
- Create diagnostic visualizations and confidence analysis

**Outputs:**
- `data/human_ce50_predictions.csv`
- `data/human_ce50_predictions_simple.csv`
- `human_ce50_predictions_diagnostics.png`

---

#### Step 2.2: Generate Enhanced Artificial Animal Data
**Status:** ⏳ PENDING
**Notebook:** `03_MedianMordredCalculator_artificial_animal_data_mfp_mrd_ce50.ipynb`

**What it will do:**
- Load human compounds with CE50 predictions from Step 2.1
- Use **CE50-enhanced rat models** from Stage 1 to generate artificial rat predictions
- Also use baseline dog/monkey models (unchanged)
- Generate 9 artificial animal predictions per human compound:
  - rat_VDss_L_kg, rat_CL_mL_min_kg, rat_fup (using CE50-enhanced models)
  - dog_VDss_L_kg, dog_CL_mL_min_kg, dog_fup (baseline)
  - monkey_VDss_L_kg, monkey_CL_mL_min_kg, monkey_fup (baseline)

**Key modification:**
- Load CE50-enhanced rat models instead of baseline rat models
- Include CE50 features when predicting with rat models

---

#### Step 2.3: Train Final Human Models
**Status:** ⏳ PENDING
**Notebook:** `03_Predict_human_data_with_artificial_animal_data_mfp_mrd_ce50.ipynb`

**What it will do:**
- Merge all features:
  - Morgan fingerprints: 152 bits
  - Mordred descriptors: 352
  - CE50 features: 3 (ce50, pce50, confidence)
  - Enhanced artificial animal data: 9
  - **Total: 516 features**
- Train 5 Random Forest models (nested 5-fold CV):
  - human_VDss_L_kg
  - human_CL_mL_min_kg
  - human_fup
  - human_mrt
  - human_thalf

**Outputs:**
- 5 model files: `log_human_{endpoint}_withanimaldata_artificial_ce50_model_FINAL.sav`
- Scalers: `artificial_animal_data_mfp_mrd_ce50_{endpoint}_scaler.pkl`
- Features: `features_mfp_mordred_animal_ce50_columns_human.txt` (516 lines)
- Results: `Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv`
- Summary: `human_with_mfp_mordred_animal_artificial_ce50.csv`

---

### 📊 STAGE 3: Comparison & Evaluation

#### Step 3.1: Comprehensive Performance Comparison
**Status:** ⏳ PENDING
**Notebook:** `04_Compare_CE50_Enhancement.ipynb`

**What it will do:**
- Load baseline and CE50-enhanced results for both rat and human models
- Calculate performance metrics: R², RMSE, GMFE, Fold-2/3/5
- Perform statistical testing (Wilcoxon signed-rank test)
- Extract and compare feature importances
- Stratify performance by CE50 confidence levels

**Analyses:**
1. **Rat Model Comparison**
   - Baseline (539 features) vs. Enhanced (542 features)
   - 3 endpoints: VDss, CL, fup

2. **Human Model Comparison**
   - Baseline (513 features) vs. Enhanced (516 features)
   - 5 endpoints: VDss, CL, fup, MRT, thalf

3. **Feature Importance Analysis**
   - Rank of CE50 features vs. other feature types
   - Top 30 features per endpoint

4. **Confidence-Stratified Analysis**
   - Does CE50 help more for high-confidence predictions?

**Outputs:**
- `ce50_comparison_report.csv` - All metrics in tabular format
- Multiple visualization files (see below)

---

#### Step 3.2: Visualization Suite
**Status:** ⏳ PENDING

**Visualizations to generate:**

1. **performance_heatmap_ce50.png**
   - Rows: 8 endpoints (3 rat + 5 human)
   - Columns: R², RMSE, GMFE
   - Colors: Green (improvement), Red (degradation)

2. **metric_comparison_ce50.png**
   - 3 panels: R², RMSE, GMFE
   - Grouped bar charts per endpoint
   - Baseline vs. Enhanced side-by-side

3. **improvement_delta_ce50.png**
   - Percentage improvement for each endpoint
   - Separate panels for rat and human
   - Highlight statistically significant changes (p < 0.05)

4. **feature_importance_ce50.png**
   - Top 30 features per endpoint
   - Highlight CE50 features in different color
   - Show importance values

5. **prediction_comparison_ce50.png**
   - 5 subplots (one per human endpoint)
   - Scatter: Predicted vs. Actual
   - Overlay baseline and enhanced predictions
   - Color by CE50 confidence

6. **confidence_stratified_ce50.png**
   - Performance by confidence level (High/Medium/Low)
   - Show if CE50 helps more for high-confidence predictions

---

## Feature Architecture

### Rat Models (542 features total)
```
Morgan Fingerprints:           153 bits
Mordred Descriptors:           386
CE50 Features:                   3
  ├── ce50                       (collision energy in eV)
  ├── pce50                      (-log10[CE50])
  └── confidence                 (0-6 scale)
────────────────────────────────────
TOTAL:                         542 features
```

### Human Models (516 features total)
```
Morgan Fingerprints:           152 bits
Mordred Descriptors:           352
CE50 Features:                   3
  ├── ce50                       (collision energy in eV)
  ├── pce50                      (-log10[CE50])
  └── confidence                 (0-6 scale)
Enhanced Artificial Animal:      9
  ├── rat_VDss_L_kg              (from CE50-enhanced rat model)
  ├── rat_CL_mL_min_kg           (from CE50-enhanced rat model)
  ├── rat_fup                    (from CE50-enhanced rat model)
  ├── dog_VDss_L_kg              (from baseline dog model)
  ├── dog_CL_mL_min_kg           (from baseline dog model)
  ├── dog_fup                    (from baseline dog model)
  ├── monkey_VDss_L_kg           (from baseline monkey model)
  ├── monkey_CL_mL_min_kg        (from baseline monkey model)
  └── monkey_fup                 (from baseline monkey model)
────────────────────────────────────
TOTAL:                         516 features
```

---

## Expected Performance Improvements

### Conservative Estimates

**Rat Models:**
- R² improvement: +5-10%
- GMFE improvement: -3-8% (lower is better)
- Fold-2 accuracy: +2-5 percentage points

**Human Models:**
- R² improvement: +3-7% (compounded effect)
- GMFE improvement: -2-6%
- Fold-2 accuracy: +2-4 percentage points

### Success Criteria

✅ **Minimum Requirements:**
1. At least 3/5 human endpoints show statistically significant improvement (p < 0.05)
2. No endpoint shows >5% degradation
3. CE50 features rank in top 50 features by importance for at least 3 endpoints

🎯 **Stretch Goals:**
1. All 5 human endpoints improved
2. >10% R² improvement for at least 1 endpoint
3. High-confidence CE50 predictions show >2x improvement vs. low-confidence

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│              STAGE 1: RAT MODEL ENHANCEMENT                  │
└──────────────────────────────────────────────────────────────┘
                            │
      data/Animal_PK_data.csv (324 rat compounds)
                            │
                            ↓
      [02a_Generate_CE50_for_rat_data.ipynb]
                            │
                            ↓
      data/rat_ce50_predictions.csv (ce50, pce50, confidence)
                            │
                            ↓
      Merge: Morgan (153) + Mordred (386) + CE50 (3) = 542
                            │
                            ↓
      [02_Predict_rat_data_with_CE50.ipynb]
                            │
                            ↓
      3 Enhanced Rat Models:
        - log_rat_VDss_L_kg_ce50_model_FINAL.sav
        - log_rat_CL_mL_min_kg_ce50_model_FINAL.sav
        - log_rat_fup_ce50_model_FINAL.sav

┌──────────────────────────────────────────────────────────────┐
│             STAGE 2: HUMAN MODEL ENHANCEMENT                 │
└──────────────────────────────────────────────────────────────┘
                            │
      data/Human_PK_data.csv (1,283 compounds)
                            │
                            ↓
      [03a_Generate_CE50_for_human_data.ipynb]
                            │
                            ↓
      data/human_ce50_predictions.csv
                            │
                            ↓
      [03_MedianMordredCalculator_..._ce50.ipynb]
      Uses CE50-enhanced rat models ↑
                            │
                            ↓
      9 Artificial Animal Predictions
      (3 from CE50-enhanced rat models)
                            │
                            ↓
      Merge: Morgan (152) + Mordred (352) + CE50 (3) + Animal (9) = 516
                            │
                            ↓
      [03_Predict_human_data_..._ce50.ipynb]
                            │
                            ↓
      5 Enhanced Human Models:
        - log_human_VDss_L_kg_withanimaldata_artificial_ce50_FINAL.sav
        - log_human_CL_mL_min_kg_withanimaldata_artificial_ce50_FINAL.sav
        - log_human_fup_withanimaldata_artificial_ce50_FINAL.sav
        - log_human_mrt_withanimaldata_artificial_ce50_FINAL.sav
        - log_human_thalf_withanimaldata_artificial_ce50_FINAL.sav

┌──────────────────────────────────────────────────────────────┐
│               STAGE 3: EVALUATION & COMPARISON               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ↓
      [04_Compare_CE50_Enhancement.ipynb]
                            │
                            ↓
      Baseline vs. Enhanced Comparison
        - Statistical testing
        - Feature importance
        - Confidence stratification
                            │
                            ↓
      6 Visualization Files + Comparison Report
```

---

## How to Execute

### Sequential Execution (Recommended)

```bash
# STAGE 1
jupyter notebook 02a_Generate_CE50_for_rat_data.ipynb
# ↓ Creates: data/rat_ce50_predictions.csv

jupyter notebook 02_Predict_rat_data_with_CE50.ipynb
# ↓ Creates: CE50-enhanced rat models

# STAGE 2
jupyter notebook 03a_Generate_CE50_for_human_data.ipynb
# ↓ Creates: data/human_ce50_predictions.csv

jupyter notebook 03_MedianMordredCalculator_artificial_animal_data_mfp_mrd_ce50.ipynb
# ↓ Creates: Enhanced artificial animal predictions

jupyter notebook 03_Predict_human_data_with_artificial_animal_data_mfp_mrd_ce50.ipynb
# ↓ Creates: CE50-enhanced human models

# STAGE 3
jupyter notebook 04_Compare_CE50_Enhancement.ipynb
# ↓ Creates: Comparison report and visualizations
```

### Batch Execution (Advanced)

```bash
# Convert notebooks to Python scripts and run
jupyter nbconvert --to script *.ipynb
python 02a_Generate_CE50_for_rat_data.py
python 02_Predict_rat_data_with_CE50.py
# ... etc
```

---

## Key Files Reference

### Input Data
- `data/Animal_PK_data.csv` - Rat/dog/monkey PK data (372 rows)
- `data/Human_PK_data.csv` - Human PK data (1,283 rows)
- `CE50_prediction/models/*.pkl` - Pre-trained CE50 models

### Template Notebooks (Reference Only)
- `02_Predict_rat_data.ipynb` - Original rat model training
- `03_Predict_human_data_with_artificial_animal_data_mfp_mrd.ipynb` - Current best human model
- `03_MedianMordredCalculator_artificial_animal_data_mfp_mrd.ipynb` - Artificial data generation

### New Notebooks (To be Created/Run)
1. `02a_Generate_CE50_for_rat_data.ipynb` ✅ CREATED
2. `02_Predict_rat_data_with_CE50.ipynb` 🚧 IN PROGRESS
3. `03a_Generate_CE50_for_human_data.ipynb` ⏳ PENDING
4. `03_MedianMordredCalculator_artificial_animal_data_mfp_mrd_ce50.ipynb` ⏳ PENDING
5. `03_Predict_human_data_with_artificial_animal_data_mfp_mrd_ce50.ipynb` ⏳ PENDING
6. `04_Compare_CE50_Enhancement.ipynb` ⏳ PENDING

---

## Troubleshooting

### Common Issues

**Issue 1: CE50 model files not found**
- **Solution**: Ensure `CE50_prediction/models/` contains .pkl files
- Check timestamp in filenames (e.g., `rf_binary_20260105_110622.pkl`)

**Issue 2: SMILES parsing failures**
- **Solution**: Check for invalid/missing SMILES in data
- The notebooks handle this gracefully and report failures

**Issue 3: Memory errors during model training**
- **Solution**: Reduce `n_jobs` in RandomForestRegressor
- Consider running on machine with >16GB RAM

**Issue 4: Inconsistent feature counts**
- **Solution**: Verify feature list files match expected counts
- Rat: 542 features, Human: 516 features

---

## Notes

### Scientific Rationale
- Original discovery (2016): Experimental CE50 correlated with rat PK
- Current approach: Using ML-predicted CE50 (R² = 0.57)
- Hypothesis: Even imperfect CE50 predictions may capture useful structural information

### Model Architecture Decisions
- Random Forest chosen for consistency with existing PKSmart models
- Nested CV (5-fold outer, 4-fold inner) for robust hyperparameter tuning
- StandardScaler preprocessing for Mordred descriptors
- Log-transformation for PK endpoints (except fup)

### Future Enhancements
- Incorporate experimental CE50 data when available
- Test other CE50 feature engineering (e.g., CE50 bins, ratios)
- Explore non-linear interactions between CE50 and molecular descriptors
- Multi-task learning: joint prediction of PK + CE50

---

## Contact & Support

For questions or issues with this implementation:
- Review the approved plan: `/Users/aardeshiri/.claude/plans/parsed-whistling-adleman.md`
- Check notebook outputs for error messages
- Verify all intermediate files were generated correctly

---

**Last Updated:** 2026-01-06
**Implementation:** Claude Code
**Status:** STAGE 1 in progress, remaining stages pending
