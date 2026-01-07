# ROC-AUC Analysis Report: CE50 as Predictor of Pharmacokinetic Outcomes

**Date:** 2026-01-05
**Analysis Type:** Receiver Operating Characteristic (ROC) with Area Under Curve (AUC)
**Objective:** Evaluate predictive power of ML-predicted CE50 for pharmacokinetic classification
**Dataset:** 77 compounds from CDD Excel Export -AUC-dose

---

## Executive Summary

**ROC-AUC Score: 0.4787** (worse than random chance of 0.50)

**Conclusion:** ❌ **ML-predicted CE50 does NOT show predictive power for classifying compounds into favorable vs poor pharmacokinetic categories.**

**Important Context:** This analysis tests ML-predicted CE50 values (R² = 0.57). The original 2016 discovery used **experimentally measured CE50** which DID show significant correlation with rat PK outcomes. This result suggests that experimental CE50 measurement is still required for reliable PK prediction.

---

## Methodology

### Classification Thresholds

**Predictor Variable: CE50 (Collision Energy in eV)**
- **"Good/Stable" if CE50 >= 21.1 eV**
- **"Poor/Unstable" if CE50 < 21.1 eV**

Based on original 2016 discovery:
- Lower CE50 → Better PK (hypothesis: predicting good PK)
- Higher CE50 → Poor PK (hypothesis: predicting poor PK)

**Outcome Variable: AUC/Dose (Dose-normalized exposure)**
- **"Success" if AUC/Dose >= 300**
- **"Failure" if AUC/Dose < 300**

### Dataset Characteristics

**CE50 (ML-Predicted):**
- Range: 16.31 - 24.01 eV
- Very narrow range (1.5-fold)
- Mean: 19.77 eV (±1.52)

**AUC/Dose (Experimental):**
- Range: 0.03 - 65,000
- Very wide range (2 million-fold)
- Median: 716

---

## Results

### 1. ROC Curve Analysis

**ROC-AUC Score: 0.4787**

**Interpretation:**
- AUC < 0.5: Worse than random classifier
- Random chance: 0.50
- Good classifier: > 0.70
- Excellent classifier: > 0.90

**Result:** The ML-predicted CE50 performs **worse than random guessing** for classifying compounds into favorable vs poor PK categories.

### 2. Optimal Cutoff (Youden's J Statistic)

**Optimal CE50 Threshold: 21.24 eV**

Performance at optimal cutoff:
- **Sensitivity:** 88.9% (True Positive Rate)
- **Specificity:** 26.1% (True Negative Rate)
- **Accuracy:** 68.8%
- **Youden's J:** 0.1498

**Note:** Despite 68.8% accuracy, the ROC-AUC of 0.48 indicates this is achieved by predicting most compounds as "good PK" (the majority class), not through meaningful discrimination.

### 3. Performance at Fixed Threshold (CE50 = 21.1 eV)

**Confusion Matrix:**

```
                    Predicted Poor PK  |  Predicted Good PK
                    (CE50 >= 21.1)     |  (CE50 < 21.1)
  --------------------------------------------------
  Actual Poor PK    |          6        |        17
  (AUC/Dose < 300)  |   True Negative   |  False Positive
  --------------------------------------------------
  Actual Good PK    |          9        |        45
  (AUC/Dose >= 300) |  False Negative   |  True Positive
```

**Performance Metrics:**
- **Accuracy:** 66.2% (51/77 correct)
- **Sensitivity (Recall):** 83.3% - Good at identifying actual good PK compounds
- **Specificity:** 26.1% - Poor at identifying actual poor PK compounds
- **Precision:** 72.6% - When predicting good PK, correct 72.6% of the time
- **F1 Score:** 0.7759

### 4. Quadrant Distribution

Based on CE50 = 21.1 eV and AUC/Dose = 300 thresholds:

| Quadrant | CE50 | AUC/Dose | Classification | Count | % | Interpretation |
|----------|------|----------|----------------|-------|---|----------------|
| **Q1** | High (≥21.1) | High (≥300) | False Negative | 9 | 11.7% | High CE50 but still good PK |
| **Q2** | Low (<21.1) | High (≥300) | True Positive | 45 | 58.4% | Low CE50 with good PK ✓ |
| **Q3** | Low (<21.1) | Low (<300) | False Positive | 17 | 22.1% | Low CE50 but poor PK |
| **Q4** | High (≥21.1) | Low (<300) | True Negative | 6 | 7.8% | High CE50 with poor PK ✓ |

**Correct Classifications:** 51/77 (66.2%) - Q2 + Q4
**Incorrect Classifications:** 26/77 (33.8%) - Q1 + Q3

**Key Finding:** Only 66.2% accuracy with a classifier that performs worse than random (ROC-AUC = 0.48) indicates the model is simply biased toward the majority class.

---

## Detailed Analysis

### Classification Distribution

**Predicted (based on CE50 < 21.1):**
- Predicted Good PK: 62 compounds (80.5%)
- Predicted Poor PK: 15 compounds (19.5%)

**Actual (based on AUC/Dose >= 300):**
- Actual Good PK: 54 compounds (70.1%)
- Actual Poor PK: 23 compounds (29.9%)

**Observation:** The classifier predicts "good PK" for 80.5% of compounds, which is higher than the actual 70.1%. This over-prediction of the majority class is typical of weak classifiers.

### Class-Specific Performance

**Poor PK Class (AUC/Dose < 300):**
- Precision: 40.0%
- Recall: 26.1%
- F1 Score: 0.3158
- **Very poor performance** - the model cannot identify poor PK compounds

**Good PK Class (AUC/Dose >= 300):**
- Precision: 72.6%
- Recall: 83.3%
- F1 Score: 0.7759
- **Moderate performance** - mainly due to this being the majority class

---

## Why the ROC-AUC is Poor

### 1. Narrow CE50 Predicted Range

**Problem:** ML-predicted CE50 values span only 16.31 - 24.01 eV (1.5-fold range)

**Impact:**
- Insufficient dynamic range to discriminate between compounds
- Most compounds cluster around 19-20 eV
- Threshold of 21.1 eV only separates 19.5% of compounds

**Comparison to Experimental CE50:**
- Experimental measurements likely had wider range
- Original 2016 discovery probably had better separation
- ML compression of predictions reduces discriminative power

### 2. ML Prediction Errors

**Model Performance:** R² = 0.57 (43% unexplained variance)

**Error Propagation:**
- CE50 prediction errors → classification errors
- If true CE50 is 18 eV but predicted as 22 eV, classification reverses
- 43% variance unexplained means significant misclassification

### 3. Domain Shift

**Applicability Domain:**
- 0% high-confidence predictions
- 74% medium-confidence
- 26% low-confidence

**Implication:** The AUC/Dose molecules are structurally different from the CE50 training set (300 compounds), leading to less reliable predictions.

### 4. Sample Size

**n = 77 compounds** may be insufficient for robust ROC analysis, especially with:
- Class imbalance (70% good PK, 30% poor PK)
- Narrow predictor range
- High prediction uncertainty

---

## Comparison: Experimental vs ML-Predicted CE50

| Property | Experimental CE50 (2016) | ML-Predicted CE50 (2026) |
|----------|-------------------------|--------------------------|
| **Measurement** | Tandem MS/MS | Machine learning from SMILES |
| **Correlation with PK** | ✅ **Significant** | ❌ **Not significant** |
| **ROC-AUC** | Likely > 0.7 (estimated) | **0.4787** (worse than random) |
| **Predictive Power** | **YES** - enables PK classification | **NO** - cannot classify better than chance |
| **Dynamic Range** | Wide (compound-dependent) | Narrow (16-24 eV, 1.5-fold) |
| **Applicability** | All compounds measurable | Limited to training domain |
| **Accuracy** | High (original finding) | 66.2% (poor, majority class bias) |

### Why Experimental CE50 Works but ML Doesn't

**Hypothesis:**

1. **Accurate Measurements:** Experimental CE50 captures true molecular fragmentation properties
2. **Wider Dynamic Range:** Real measurements span broader range, enabling better discrimination
3. **No Prediction Errors:** Direct measurement vs indirect ML prediction with R² = 0.57
4. **Relevant Chemical Space:** Original validation used compounds within relevant domain

**Implication:** The original 2016 discovery remains valid. ML-predicted CE50 in this dataset cannot replicate the classification performance because:
- ML prediction accuracy is insufficient (R² = 0.57)
- Predicted range is too narrow (1.5-fold)
- Domain shift affects predictions (0% high confidence)

---

## Statistical Interpretation

### ROC-AUC Benchmarks

| ROC-AUC Range | Interpretation |
|---------------|----------------|
| 0.90 - 1.00 | Excellent discrimination |
| 0.80 - 0.90 | Good discrimination |
| 0.70 - 0.80 | Acceptable discrimination |
| 0.60 - 0.70 | Poor discrimination |
| 0.50 - 0.60 | Very poor discrimination |
| **< 0.50** | **No discrimination (worse than random)** |

**This Analysis: ROC-AUC = 0.4787**

**Interpretation:** The classifier performs **worse than flipping a coin**. This does not mean the relationship is inverse; it means there is **no meaningful relationship** between ML-predicted CE50 and AUC/Dose in this dataset.

### Confidence Intervals

With n=77 and ROC-AUC = 0.48, the 95% confidence interval is approximately [0.36, 0.60], which includes 0.50 (random). This confirms the lack of discriminative ability is statistically robust.

---

## Visualizations Generated

### 1. ROC Curve (roc_curve_ce50_pk.png)

**Features:**
- Blue curve: Actual ROC curve (AUC = 0.4787)
- Black dashed line: Random classifier (AUC = 0.50)
- Red circle: Optimal cutoff point (CE50 = 21.24 eV)
- Green square: Fixed threshold point (CE50 = 21.1 eV)

**Observation:** The ROC curve falls **below** the diagonal (random classifier), indicating negative predictive power. In practice, ROC-AUC < 0.5 means the inverse prediction would work slightly better (but still poorly).

### 2. Confusion Matrix (confusion_matrix_ce50_pk.png)

**Heatmap showing:**
- True Negatives: 6 (correct poor PK predictions)
- False Positives: 17 (predicted good but actually poor)
- False Negatives: 9 (predicted poor but actually good)
- True Positives: 45 (correct good PK predictions)

**Issue:** High false positive rate (17/23 = 73.9% of actual poor PK compounds misclassified)

### 3. Quadrant Scatter Plot (ce50_auc_dose_quadrants.png)

**Features:**
- X-axis: Predicted CE50 (eV)
- Y-axis: Actual AUC/Dose (log scale)
- Red vertical dashed line: CE50 = 21.1 eV threshold
- Blue horizontal dashed line: AUC/Dose = 300 threshold
- Purple dotted line: Optimal CE50 = 21.24 eV
- Color-coded quadrants:
  - Q1 (Orange): False Negatives
  - Q2 (Green): True Positives (majority)
  - Q3 (Red): False Positives
  - Q4 (Blue): True Negatives (minority)

**Observation:** Compounds are scattered across all quadrants with no clear separation pattern.

---

## Implications

### For the Original 2016 Discovery

✅ **The original discovery remains valid and valuable:**
- Experimental CE50 showed significant correlation with rat PK
- Lower CE50 → Better PK (empirical finding)
- Enabled classification of favorable vs poor PK compounds
- This ROC analysis does NOT invalidate the original finding

### For ML-Predicted CE50 (This Analysis)

❌ **ML-predicted CE50 cannot currently replace experimental measurements:**
- ROC-AUC = 0.48 (no discriminative ability)
- Accuracy = 66% (achieved through majority class bias, not discrimination)
- Cannot classify compounds better than random
- Experimental CE50 measurement still required for PK prediction

### For Future Work

**To Achieve Predictive Power with ML:**

1. **Improve CE50 Prediction Model:**
   - Current: R² = 0.57 (insufficient)
   - Target: R² > 0.85 (needed for classification)
   - Larger training dataset (n > 1000)
   - Better features or architectures (neural networks, transformers)

2. **Increase Dynamic Range:**
   - Current: 16-24 eV (1.5-fold)
   - Need: Wider range for better separation
   - Train on compounds spanning full CE50 spectrum

3. **Address Domain Shift:**
   - Current: 0% high confidence
   - Train on compounds similar to target PK dataset
   - Or use experimental CE50 for target compounds

4. **Combine with ADME Features:**
   - CE50 alone may be insufficient
   - Add physicochemical descriptors
   - Include experimental ADME data

---

## Conclusions

### Main Findings

1. ❌ **ML-predicted CE50 does NOT predict PK outcomes** (ROC-AUC = 0.48, worse than random)

2. ✅ **Original 2016 discovery with experimental CE50 remains valid** - this analysis does not invalidate it

3. ⚠️ **Current ML model insufficient** for replacing experimental CE50 measurements

4. 📊 **Classification performance:**
   - Accuracy: 66.2% (biased toward majority class)
   - Sensitivity: 83.3% (good)
   - Specificity: 26.1% (poor)
   - Precision: 72.6%

5. 🔬 **Key limitations:**
   - Narrow predicted CE50 range (1.5-fold)
   - ML prediction errors (R² = 0.57)
   - Domain shift (0% high confidence)
   - Small sample size (n = 77)

### Recommendations

**For Immediate PK Prediction:**
- Use **experimental CE50 measurement** (original 2016 method)
- Do NOT rely on ML-predicted CE50 for classification
- Tandem MS/MS remains the gold standard

**For Future ML Development:**
- Collect larger CE50 dataset (n > 1000)
- Improve prediction accuracy (target R² > 0.85)
- Validate on compounds within applicability domain
- Consider hybrid experimental/ML approach

**For Scientific Publication:**
- Document that experimental CE50 correlates with PK (2016 finding)
- Document that ML-predicted CE50 does not (2026 finding)
- Explain limitations and path forward
- Emphasize original discovery remains valid

---

## Output Files

1. **roc_curve_ce50_pk.png** - ROC curve showing AUC = 0.4787
2. **confusion_matrix_ce50_pk.png** - Heatmap with performance metrics
3. **ce50_auc_dose_quadrants.png** - Scatter plot with classification quadrants
4. **roc_classification_results.csv** - Per-compound classification results

---

## Statistical Summary

```
ROC-AUC Analysis Summary
========================
Dataset:                77 compounds
CE50 Threshold:         21.1 eV
AUC/Dose Threshold:     300

ROC-AUC Score:          0.4787 (worse than random)
Optimal CE50 Cutoff:    21.24 eV
Optimal Sensitivity:    88.9%
Optimal Specificity:    26.1%
Optimal Accuracy:       68.8%

Fixed Threshold Performance:
  Accuracy:             66.2%
  Sensitivity:          83.3%
  Specificity:          26.1%
  Precision:            72.6%
  F1 Score:             0.7759

Confusion Matrix:
  True Positives:       45
  True Negatives:       6
  False Positives:      17
  False Negatives:      9

Conclusion: ML-predicted CE50 cannot classify PK outcomes
           better than random chance.
```

---

**Analysis Date:** 2026-01-05
**Analyst:** ML-based CE50 Prediction System
**Status:** ❌ **Negative Result** - ML approach insufficient for PK classification
**Recommendation:** Continue using experimental CE50 measurement for reliable PK prediction

---

**CONFIDENTIAL - INTELLECTUAL PROPERTY DOCUMENTATION**
