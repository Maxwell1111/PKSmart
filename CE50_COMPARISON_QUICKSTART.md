# CE50 Comparison Analysis - Quick Start Guide

## 5-Minute Quick Start

### 1. Check Prerequisites

```bash
# Test if all required files and packages are available
python test_ce50_comparison.py
```

Expected output:
```
✓ ALL TESTS PASSED - Ready to run comparison analysis!
```

### 2. Run Comparison

```bash
# Run the full comparison analysis
python compare_ce50_enhancement.py
```

This will take 2-5 minutes depending on dataset size.

### 3. Review Results

Check the generated files:
```bash
# View summary statistics
cat ce50_comparison_report.csv

# View statistical significance
cat ce50_statistical_tests.csv

# View confidence analysis
cat ce50_confidence_analysis.csv

# Open visualizations
open performance_heatmap_ce50.png
open metric_comparison_ce50.png
open improvement_delta_ce50.png
```

## Common Usage Scenarios

### Scenario 1: Quick Performance Check

**Goal**: Quickly see if CE50 features improved predictions

```bash
# Run comparison
python compare_ce50_enhancement.py

# Check improvement summary (last lines of output)
tail -20 <output>
```

Look for:
- ✓ IMPROVED markers
- Positive ΔR² values
- p < 0.05 for significance

### Scenario 2: Detailed Endpoint Analysis

**Goal**: Understand which endpoints benefited most from CE50

```python
import pandas as pd

# Load comparison results
results = pd.read_csv('ce50_comparison_report.csv')

# Filter for R² metric
r2_results = results[results['metric'] == 'r2']

# Sort by improvement
r2_results = r2_results.sort_values('improvement', ascending=False)

# Display top improvements
print(r2_results[['endpoint', 'improvement', 'improvement_pct', 'significant']])
```

### Scenario 3: Confidence Level Validation

**Goal**: Verify that high-confidence predictions perform better

```python
import pandas as pd

# Load confidence analysis
conf_analysis = pd.read_csv('ce50_confidence_analysis.csv')

# Group by endpoint and confidence
summary = conf_analysis.groupby(['endpoint', 'confidence'])['r2'].mean().unstack()

print(summary)
```

Expected pattern:
```
                    High    Medium    Low
rat_VDss_L_kg      0.89     0.75    0.62
rat_CL_mL_min_kg   0.87     0.73    0.59
rat_fup            0.85     0.71    0.61
```

### Scenario 4: Feature Importance Investigation

**Goal**: Understand which features contribute most

```python
import pandas as pd

# Load feature importance
feat_imp = pd.read_csv('ce50_feature_importance.csv')

# Get top 10 features per endpoint
for endpoint in feat_imp['endpoint'].unique():
    print(f"\n{endpoint}:")
    top_features = feat_imp[feat_imp['endpoint'] == endpoint].nlargest(10, 'importance')
    for idx, row in top_features.iterrows():
        print(f"  {row['rank']:2d}. {row['feature_name']:30s} ({row['feature_type']:8s}): {row['importance']:.4f}")
```

### Scenario 5: Statistical Significance Check

**Goal**: Determine which improvements are statistically significant

```python
import pandas as pd

# Load statistical tests
stats = pd.read_csv('ce50_statistical_tests.csv')

# Filter for R² metric
r2_stats = stats[stats['metric'] == 'r2']

# Show only significant improvements
significant = r2_stats[r2_stats['significant'] == True]

print("Statistically Significant Improvements:")
for _, row in significant.iterrows():
    print(f"  {row['species']} {row['endpoint']}: p = {row['p_value']:.6f}")
```

## Troubleshooting Quick Fixes

### Issue: Script hangs or takes too long

**Quick Fix**: Reduce dataset size for testing

```python
# Add at top of compare_ce50_enhancement.py after loading data
rat_ce50 = rat_ce50.sample(n=min(100, len(rat_ce50)))  # Test with 100 samples
```

### Issue: Memory errors

**Quick Fix**: Process one species at a time

```bash
# Comment out human processing in script, run for rat only
# Then comment out rat, run for human only
```

### Issue: Missing baseline predictions

**Quick Fix**: The script simulates baseline predictions. For real comparison, you need actual baseline model predictions.

```python
# In the script, replace:
merged['baseline_pred'] = merged['actual'] + np.random.normal(...)

# With:
baseline = pd.read_csv('path/to/baseline_predictions.csv')
merged = merged.merge(baseline[['smiles_r', 'prediction']], on='smiles_r')
merged.rename(columns={'prediction': 'baseline_pred'}, inplace=True)
```

### Issue: Visualization errors

**Quick Fix**: Use non-interactive backend

```python
# Add at top of script before importing matplotlib
import matplotlib
matplotlib.use('Agg')
```

## Interpretation Cheat Sheet

### Good Results Indicators

✓ R² improvement > 0.05 (5% relative)
✓ RMSE reduction > 10%
✓ Fold-2 accuracy > 70%
✓ p-value < 0.05
✓ High confidence predictions perform better than low

### Warning Signs

⚠ R² degradation
⚠ p-value > 0.05 (not significant)
⚠ Inconsistent confidence-performance relationship
⚠ Large improvements only on small subsets

### Red Flags

🚩 R² degradation > -0.05
🚩 RMSE increase > 20%
🚩 Fold-2 accuracy < 50%
🚩 High confidence predictions perform worse

## Quick Metrics Reference

| Metric | Good Value | Interpretation |
|--------|-----------|----------------|
| R² | > 0.7 | Strong correlation |
| R² | 0.4-0.7 | Moderate correlation |
| R² | < 0.4 | Weak correlation |
| RMSE | < 0.5 | Good accuracy (log scale) |
| RMSE | 0.5-1.0 | Moderate accuracy |
| RMSE | > 1.0 | Poor accuracy |
| GMFE | < 2.0 | Within 2-fold on average |
| GMFE | 2.0-3.0 | Within 2-3 fold |
| GMFE | > 3.0 | Poor fold-accuracy |
| Fold-2 | > 70% | Industry acceptable |
| Fold-2 | 50-70% | Moderate |
| Fold-2 | < 50% | Poor |

## One-Liners

```bash
# Quick test
python test_ce50_comparison.py | grep "PASS\|FAIL"

# Run and save output
python compare_ce50_enhancement.py 2>&1 | tee comparison_output.log

# Count significant improvements
grep "✓ IMPROVED" comparison_output.log | grep "p < 0.05" | wc -l

# Extract R² values
grep "R²:" comparison_output.log

# View heatmap
open performance_heatmap_ce50.png

# Quick CSV summary
csvlook ce50_comparison_report.csv | head -20
```

## Integration Examples

### With Jupyter Notebook

```python
# In notebook
!python compare_ce50_enhancement.py

# Load results
import pandas as pd
results = pd.read_csv('ce50_comparison_report.csv')

# Display in notebook
from IPython.display import Image, display
display(Image('performance_heatmap_ce50.png'))

# Interactive analysis
results[results['significant'] == True].sort_values('improvement', ascending=False)
```

### With Shell Script

```bash
#!/bin/bash
# automated_comparison.sh

echo "Starting CE50 comparison analysis..."
python compare_ce50_enhancement.py > comparison_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "Success! Results saved."
    ls -lh *ce50*.{csv,png}
else
    echo "Error occurred. Check log file."
    exit 1
fi
```

### With Make

```makefile
# Makefile
.PHONY: test compare clean

test:
	python test_ce50_comparison.py

compare:
	python compare_ce50_enhancement.py

clean:
	rm -f *ce50*.csv *ce50*.png

all: test compare
```

## Tips & Tricks

### Tip 1: Save Intermediate Results

Add checkpoints in the script:

```python
# After Step 2
metrics_df.to_csv('metrics_checkpoint.csv', index=False)

# After Step 3
confidence_df.to_csv('confidence_checkpoint.csv', index=False)
```

### Tip 2: Parallel Processing

For large datasets:

```python
from joblib import Parallel, delayed

# Replace loop with parallel processing
results = Parallel(n_jobs=-1)(
    delayed(calculate_metrics)(actual, predicted)
    for actual, predicted in data_pairs
)
```

### Tip 3: Custom Metrics

Add domain-specific metrics:

```python
def calculate_afr(y_true, y_pred):
    """Average Fold Ratio - geometric mean of prediction/actual"""
    return np.exp(np.mean(np.log(y_pred / y_true)))
```

### Tip 4: Export to Excel

```python
# Save all results to Excel with multiple sheets
with pd.ExcelWriter('ce50_analysis_complete.xlsx') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    confidence_df.to_excel(writer, sheet_name='Confidence', index=False)
    feat_imp_df.to_excel(writer, sheet_name='Features', index=False)
```

## Getting Help

1. **Check test output**: `python test_ce50_comparison.py`
2. **Read full documentation**: `CE50_COMPARISON_README.md`
3. **Review error messages**: Most errors are self-explanatory
4. **Check input files**: Ensure CSVs have required columns
5. **Verify data types**: SMILES should be strings, values should be numeric

## Next Steps

After running the comparison:

1. ✅ Review summary report
2. ✅ Check statistical significance
3. ✅ Analyze confidence stratification
4. ✅ Examine feature importance
5. ✅ Review visualizations
6. ✅ Document findings
7. ✅ Share results with team
8. ✅ Integrate into pipeline

## Success Criteria

Your CE50 enhancement is successful if:

- ✓ R² improved for majority of endpoints
- ✓ At least one improvement is statistically significant (p < 0.05)
- ✓ High confidence predictions perform better than low
- ✓ Fold-2 accuracy > 70% for important endpoints
- ✓ No major degradations (ΔR² < -0.05)

If criteria not met:
- Review feature engineering
- Check data quality
- Validate CE50 predictions
- Consider ensemble approaches
- Investigate outliers
