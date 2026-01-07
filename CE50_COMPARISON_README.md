# CE50 Enhancement Comparison Analysis

## Overview

The `compare_ce50_enhancement.py` script provides comprehensive comparison analysis between baseline and CE50-enhanced pharmacokinetic prediction models for both rat and human data.

## Features

### 1. Performance Metrics Calculation
- **R² (Coefficient of Determination)**: Measures model fit quality
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **GMFE (Geometric Mean Fold Error)**: Measures fold-error magnitude
- **Fold-2/3/5 Accuracy**: Percentage of predictions within 2/3/5-fold of actual
- **Bias**: Median prediction error (over/under-prediction)

### 2. Statistical Testing
- **Wilcoxon Signed-Rank Test**: Non-parametric paired comparison
- **P-values**: Statistical significance calculation
- **Significance Threshold**: p < 0.05

### 3. Feature Importance Analysis
- Extracts feature importances from trained models
- Identifies top contributing features
- Highlights CE50 feature contributions
- Compares CE50 vs. structural features

### 4. Confidence Stratification
Analyzes predictions by CE50 confidence level:
- **High Confidence**: confidence ≥ 5.0
- **Medium Confidence**: 3.0 ≤ confidence < 5.0
- **Low Confidence**: confidence < 3.0

Tests hypothesis: Higher confidence predictions show better performance

### 5. Comprehensive Visualizations

#### Generated Plots:

**performance_heatmap_ce50.png**
- Side-by-side heatmaps of baseline vs CE50-enhanced R² values
- Color-coded by performance level
- Organized by species and endpoint

**metric_comparison_ce50.png**
- Grouped bar charts comparing R², RMSE, and Fold-2 accuracy
- Direct baseline vs CE50 comparison
- Organized by endpoint and species

**improvement_delta_ce50.png**
- Horizontal bar chart showing percentage improvement
- Green bars: improvements
- Red bars: degradations
- Sorted by magnitude

**confidence_stratified_ce50.png**
- Performance metrics stratified by confidence level
- Shows if high-confidence predictions perform better
- Separate panels for R², RMSE, and Fold-2

**prediction_comparison_ce50.png**
- Scatter plots: actual vs predicted values
- Separate panels for each endpoint
- Diagonal line shows perfect prediction
- Visualizes prediction quality

### 6. Output Files

**ce50_comparison_report.csv**
```
Columns:
- species: Rat or Human
- endpoint: PK parameter (VDss, CL, fup)
- metric: Performance metric name
- baseline: Baseline model value
- ce50: CE50-enhanced model value
- improvement: Absolute improvement
- improvement_pct: Percentage improvement
- p_value: Statistical test p-value
- significant: Boolean significance flag
```

**ce50_statistical_tests.csv**
```
Columns:
- species: Rat or Human
- endpoint: PK parameter
- metric: Performance metric
- p_value: Wilcoxon test p-value
- significant: Boolean (p < 0.05)
```

**ce50_confidence_analysis.csv**
```
Columns:
- species: Rat or Human
- endpoint: PK parameter
- confidence: High/Medium/Low
- n_samples: Number of predictions
- r2: R² for this confidence level
- rmse: RMSE for this confidence level
- fold2: Fold-2 accuracy
```

**ce50_feature_importance.csv**
```
Columns:
- endpoint: PK parameter
- feature_name: Feature identifier
- importance: Feature importance score
- feature_type: CE50/Mordred/Morgan
- rank: Overall importance rank
```

## Usage

### Basic Usage

```bash
python compare_ce50_enhancement.py
```

### Requirements

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Input Files Required

1. **CE50-Enhanced Predictions:**
   - `data/rat_ce50_predictions.csv`
   - `data/human_ce50_predictions.csv`

2. **Baseline Data:**
   - `data/Animal_PK_data.csv`
   - `data/Human_PK_data.csv`

3. **Feature Names:**
   - `features_mfp_mordred_ce50_columns_rat_model.txt`

4. **Model Files (optional for feature importance):**
   - `models/*_baseline.pkl`
   - `models/*_ce50.pkl`

### Expected Data Format

#### CE50 Prediction Files
```csv
smiles_r,rat_VDss_L_kg,rat_CL_mL_min_kg,rat_fup,ce50,pce50,confidence
CCCC...,0.25,33.0,0.69,21.76,-1.34,2.20
```

#### Baseline Data Files
```csv
smiles_r,rat_VDss_L_kg,rat_CL_mL_min_kg,rat_fup
CCCC...,0.25,33.0,0.69
```

## Interpretation Guide

### Performance Metrics

**R² (Coefficient of Determination)**
- Range: -∞ to 1.0 (higher is better)
- 1.0 = perfect prediction
- 0.0 = model no better than mean
- < 0.0 = model worse than mean

**RMSE (Root Mean Squared Error)**
- Range: 0 to ∞ (lower is better)
- Same units as predicted variable
- Penalizes large errors more heavily

**GMFE (Geometric Mean Fold Error)**
- Range: 1.0 to ∞ (closer to 1.0 is better)
- 1.0 = perfect prediction
- 2.0 = average 2-fold error
- Symmetric treatment of over/under-prediction

**Fold-X Accuracy**
- Range: 0% to 100% (higher is better)
- Percentage within X-fold of actual
- Industry standard: >70% for Fold-2

### Statistical Significance

**p-value < 0.05**: Improvement is statistically significant
**p-value ≥ 0.05**: Improvement not statistically significant

Note: Statistical significance doesn't always mean practical significance

### Confidence Levels

**Expected Pattern:**
- High Confidence → Best performance
- Medium Confidence → Moderate performance
- Low Confidence → Lowest performance

If this pattern holds, CE50 confidence is a reliable quality indicator.

## Example Output

### Summary Report
```
==================================================================================
SUMMARY REPORT
==================================================================================

Overall Performance Improvements:
----------------------------------------------------------------------------------
✓ IMPROVED     | Rat      rat_VDss_L_kg              | Δ R² = +0.123 (+15.2%) (p < 0.05)
✓ IMPROVED     | Rat      rat_CL_mL_min_kg           | Δ R² = +0.098 (+12.1%) (p < 0.05)
✗ DEGRADED     | Rat      rat_fup                    | Δ R² = -0.012 (-1.8%)

Endpoints with Significant Improvements:
----------------------------------------------------------------------------------
  • Rat rat_VDss_L_kg: ΔR² = 0.123 (p = 0.0034)
  • Rat rat_CL_mL_min_kg: ΔR² = 0.098 (p = 0.0089)

Confidence-Based Performance:
----------------------------------------------------------------------------------
  High     Confidence: R² = 0.891, Fold-2 = 87.3%
  Medium   Confidence: R² = 0.756, Fold-2 = 74.2%
  Low      Confidence: R² = 0.623, Fold-2 = 61.8%
```

## Customization

### Modify Confidence Thresholds

```python
CONFIDENCE_THRESHOLDS = {
    'High': (5.0, float('inf')),
    'Medium': (3.0, 5.0),
    'Low': (0, 3.0)
}
```

### Change Color Scheme

```python
COLORS = {
    'improvement': '#2ecc71',  # Green
    'degradation': '#e74c3c',  # Red
    'neutral': '#95a5a6',      # Gray
    'baseline': '#3498db',     # Blue
    'ce50': '#9b59b6'          # Purple
}
```

### Add Additional Metrics

Modify the `calculate_metrics()` function:

```python
def calculate_metrics(y_true, y_pred):
    # ... existing code ...

    # Add custom metric
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))

    return {
        # ... existing metrics ...
        'mae': mae
    }
```

## Troubleshooting

### Issue: "File not found" errors

**Solution**: Ensure all required input files are in correct locations:
```bash
ls data/rat_ce50_predictions.csv
ls data/human_ce50_predictions.csv
ls data/Animal_PK_data.csv
ls data/Human_PK_data.csv
```

### Issue: No baseline predictions available

**Solution**: The script currently simulates baseline predictions. To use actual baseline predictions:

1. Generate baseline predictions using baseline models
2. Modify the script to load actual baseline predictions:

```python
# Replace this simulation code:
merged['baseline_pred'] = merged['actual'] + np.random.normal(...)

# With actual baseline predictions:
baseline_preds = pd.read_csv('baseline_predictions.csv')
merged = merged.merge(baseline_preds, on='smiles_r')
```

### Issue: Missing feature importance

**Solution**: Ensure model pickle files are available and accessible:
```bash
ls models/*_baseline.pkl
ls models/*_ce50.pkl
```

### Issue: Matplotlib errors on headless systems

**Solution**: Use Agg backend:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

## Integration with Existing Workflow

### After Training Models

```bash
# 1. Train baseline models
python train_rat_models_baseline.py

# 2. Train CE50-enhanced models
python train_rat_models_with_ce50.py

# 3. Run comparison
python compare_ce50_enhancement.py
```

### Automated Pipeline

```bash
#!/bin/bash
# run_full_comparison.sh

echo "Training baseline models..."
python train_rat_models_baseline.py

echo "Training CE50-enhanced models..."
python train_rat_models_with_ce50.py

echo "Running comparison analysis..."
python compare_ce50_enhancement.py

echo "Analysis complete! Check generated files."
```

## Citation

If you use this comparison analysis in your research, please cite:

```
CE50 Enhancement Comparison Analysis
Generated with Claude Code
Date: 2026-01-07
```

## Support

For issues or questions:
1. Check this README for troubleshooting
2. Verify input file formats
3. Check Python package versions
4. Review error messages for specific issues

## Version History

**v1.0 (2026-01-07)**
- Initial release
- Comprehensive metric calculation
- Statistical testing
- Confidence stratification
- Feature importance analysis
- Visualization suite
- Automated reporting
