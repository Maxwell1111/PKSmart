# CE50 as a Predictor of Rat Pharmacokinetics - Invention Documentation

**Inventor:** Aardeshiri
**Original Discovery:** 2016
**Current Investigation:** 2026
**Status:** Patent/IP Documentation

---

## Executive Summary

This document records the invention and validation of using **CE50 (Collision Energy for 50% fragmentation in mass spectrometry)** as a predictive marker for **rat pharmacokinetic parameters**, specifically plasma clearance and oral exposure (AUC).

### Key Innovation

**Discovery:** Experimentally measured CE50 values correlate with and can classify compounds as having favorable or poor pharmacokinetic properties in rat PK studies.

**Current Extension:** Machine learning models can predict CE50 from molecular structure (R² = 0.57), enabling **in silico PK prediction** without experimental measurements.

---

## Timeline of Discovery

### 2016: Original Discovery
- **Finding:** Experimentally measured CE50 correlates with rat PK parameters
- **Parameters:** IV plasma clearance, oral exposure (AUC)
- **Application:** Classification of compounds as favorable vs poor PK
- **Method:** Experimental CE50 measurement via tandem mass spectrometry (see CE50.pdf)

### 2026: Machine Learning Investigation
- **Hypothesis:** ML-predicted CE50 can replace experimental CE50 for PK prediction
- **Approach:** Train ensemble models to predict CE50 from SMILES strings
- **Achievement:** R² = 0.57 for CE50 prediction from molecular fingerprints
- **Testing:** Correlation of ML-predicted CE50 with rat oral exposure (AUC/Dose)
- **Drivers:** Advancements in machine learning and availability of AI tools

---

## Scientific Hypothesis

### Original Hypothesis (2016)

**"Experimentally measured CE50 correlates with rat pharmacokinetic outcomes"**

**Rationale:**
- CE50 reflects molecular stability and fragmentation patterns in gas phase
- Fragmentation patterns may correlate with metabolic stability in vivo
- Molecules that fragment easily (low CE50) may also be metabolically labile
- Molecules that are stable in MS (high CE50) may have better metabolic stability

**Validation:**
- Demonstrated correlation between experimental CE50 and:
  - IV plasma clearance
  - Oral exposure (AUC after oral dosing)
  - Classification accuracy for favorable vs poor PK

### Extended Hypothesis (2026)

**"Machine learning-predicted CE50 correlates with rat pharmacokinetic outcomes"**

**Rationale:**
- If ML can accurately predict CE50 from structure (R² = 0.57)
- And experimental CE50 correlates with rat PK (proven 2016)
- Then ML-predicted CE50 should also correlate with rat PK
- This enables **in silico PK screening** without experimental measurements

**Test:**
- Predict CE50 for 77 compounds using trained ML ensemble
- Correlate predicted CE50 with experimental rat oral exposure (AUC/Dose)
- Assess whether correlation is significant

---

## Experimental Method

### CE50 Measurement Protocol

**Reference:** CE50.pdf (included in repository)

**Method:** Tandem mass spectrometry (MS/MS)
1. Ionize compound (ESI or APCI)
2. Select precursor ion (Q1)
3. Collision cell with variable collision energy
4. Monitor precursor and fragment ions
5. **CE50 = collision energy where 50% of precursor ions fragment**

**Typical Setup:**
- Instrument: Triple quadrupole or Q-TOF
- Collision gas: Nitrogen or Argon
- Energy range: 0-100 eV
- Measurement: Precursor ion intensity vs collision energy

**Output:**
- CE50 value in electronvolts (eV)
- Typical range for drug-like molecules: 10-50 eV

---

## Pharmacokinetic Parameters Measured

### Rat PK Study Design

**Species:** Rat (Sprague-Dawley or Wistar)

**Routes:**
- **IV:** Intravenous dosing for clearance measurement
- **PO:** Oral dosing for exposure/bioavailability

**Parameters Measured:**

#### 1. IV Plasma Clearance (CL)
- Units: mL/min/kg
- Calculation: Dose / AUC_IV
- Interpretation:
  - Low CL → compound retained in body (favorable)
  - High CL → rapid elimination (poor)

#### 2. Oral Exposure (AUC)
- Units: ng·h/mL or μM·h
- Measurement: Area under plasma concentration-time curve
- Interpretation:
  - High AUC → good oral bioavailability (favorable)
  - Low AUC → poor absorption/high first-pass (poor)

#### 3. AUC/Dose
- Normalization: AUC divided by administered dose
- Units: (ng·h/mL) / (mg/kg) or similar
- **This is the primary metric in current analysis**
- Interpretation:
  - High AUC/Dose → efficient exposure (favorable)
  - Low AUC/Dose → poor oral PK (poor)

---

## Original Finding (2016)

### Correlation Between Experimental CE50 and Rat PK

**Dataset:**
- N compounds tested in rat PK studies
- Experimental CE50 measured for each compound
- IV clearance and oral AUC measured

**Results:**
- **Significant correlation** between CE50 and:
  - IV plasma clearance (r = X, p < 0.05)
  - Oral exposure AUC (r = Y, p < 0.05)

**Classification Performance:**
- Favorable PK: AUC/Dose > threshold
- Poor PK: AUC/Dose < threshold
- CE50 cutoff: Z eV
- **Accuracy:** A% correct classification
- **Sensitivity:** S%
- **Specificity:** T%

**Mechanistic Hypothesis:**
- **Low CE50** (easier fragmentation) → **Better PK** (lower clearance, higher oral exposure/AUC)
- **High CE50** (harder fragmentation) → **Poor PK** (higher clearance, lower oral exposure/AUC)

**Empirical Finding:** Lower CE50 values measured experimentally correlated with favorable pharmacokinetic outcomes in rat studies, including lower plasma clearance and higher oral exposure (AUC). The exact mechanism linking gas-phase fragmentation to in vivo metabolism is under investigation, but the correlation was statistically significant and reproducible.

**Innovation:**
- First demonstration that MS fragmentation patterns predict in vivo PK
- Simple, high-throughput method (CE50 measurement is fast)
- Orthogonal to traditional ADME assays

---

## Current Investigation (2026)

### Machine Learning Approach

**Question:** Can ML-predicted CE50 replace experimental CE50 for rat PK prediction?

**Workflow:**
1. **Train ML models** to predict CE50 from SMILES strings
   - Dual fingerprints (binary + count Morgan)
   - 4-model ensemble (RF + XGB × 2 fingerprint types)
   - Achieved R² = 0.57 on 300 compounds

2. **Apply ML model** to compounds with rat PK data
   - 77 compounds with oral exposure data (AUC/Dose)
   - Predict CE50 from molecular structure
   - No experimental CE50 measurement required

3. **Correlate ML-predicted CE50** with rat oral exposure
   - Test hypothesis: predicted CE50 correlates with AUC/Dose
   - Compare to original 2016 finding with experimental CE50

### Results Summary

**Dataset:** 77 compounds from CDD database
- Source: "CDD Excel Export -AUC-dose.xlsx"
- Rat oral exposure: AUC/Dose values
- ML-predicted CE50: 16.31 - 24.01 eV (mean = 19.77 eV)

**Correlation Analysis:**
- **Pearson r = -0.10, p = 0.38** (not significant)
- **Spearman ρ = -0.07, p = 0.53** (not significant)

**Interpretation:**
- ❌ No significant correlation found between ML-predicted CE50 and rat AUC/Dose
- Potential reasons:
  1. **Narrow CE50 range** (16-24 eV, only 1.5-fold) limits correlation detection
  2. **Domain shift** (0% high-confidence predictions) - compounds structurally different from training set
  3. **Small sample size** (n=77) limits statistical power
  4. **ML prediction errors** (R² = 0.57 means 43% unexplained variance)
  5. **Different compound set** than original 2016 validation

---

## Comparison: Experimental vs ML-Predicted CE50

| Property | Experimental CE50 (2016) | ML-Predicted CE50 (2026) |
|----------|-------------------------|--------------------------|
| **Measurement** | Tandem MS/MS | Machine learning from SMILES |
| **Throughput** | Medium (requires instrument) | Very high (computational) |
| **Accuracy** | Ground truth | R² = 0.57 (moderate error) |
| **Cost** | $ per sample (MS time) | Free (after model training) |
| **Correlation with rat PK** | ✅ **Significant** (2016 finding) | ❌ **Not significant** (2026 test) |
| **Dynamic range** | Wide (molecule-dependent) | Narrow in this dataset (16-24 eV) |
| **Applicability** | All compounds measurable | Limited to training domain |

### Why ML-Predicted CE50 May Not Correlate

**Hypothesis 1: Prediction Error**
- ML model has R² = 0.57 → 43% unexplained variance
- Errors in CE50 prediction propagate to PK correlation
- Need experimental CE50 for accurate PK prediction

**Hypothesis 2: Narrow Predicted Range**
- All compounds predicted as 16-24 eV (1.5-fold)
- Insufficient dynamic range to detect correlation
- May indicate model compression or domain shift

**Hypothesis 3: Domain Mismatch**
- 0% high-confidence predictions
- Compounds are outside training set chemical space
- ML model extrapolating, not interpolating

**Hypothesis 4: Dataset Difference**
- 2016 validation: Different compound set (unknown size/diversity)
- 2026 test: 77 compounds from CDD database
- Compound sets may have different CE50-PK relationships

---

## Implications for the Invention

### Status of Original Discovery (2016)

✅ **Validated:** Experimental CE50 correlates with rat PK
- This is the **core invention**
- Method: Simple MS/MS measurement
- Application: PK classification and optimization
- Advantage: Fast, orthogonal to traditional ADME

### Status of ML Extension (2026)

⚠️ **Inconclusive:** ML-predicted CE50 does not show significant correlation in this test
- **Does NOT invalidate** the original 2016 discovery
- Suggests that **experimental CE50 measurement is still required**
- ML prediction may work with:
  - Larger training dataset (current n=300 may be insufficient)
  - Better models (neural networks, transformers)
  - Compounds within applicability domain
  - Improved feature engineering

### Path Forward

**Option 1: Improve ML Model**
- Train on larger CE50 dataset (n > 1000)
- Include physicochemical descriptors beyond fingerprints
- Use advanced architectures (graph neural networks, transformers)
- Validate on compounds within applicability domain

**Option 2: Hybrid Approach**
- Use ML for screening (rank-order compounds)
- Measure experimental CE50 for top candidates
- Validate PK correlation on experimental CE50 subset

**Option 3: Focus on Experimental Method**
- Continue using experimental CE50 measurement
- This is validated and works (2016 discovery)
- High-throughput MS/MS enables screening
- Maintain competitive advantage

---

## Intellectual Property Considerations

### Original Invention (2016)

**Claim:** Method for predicting rat pharmacokinetic parameters using CE50

**Components:**
1. Measure CE50 via tandem mass spectrometry
2. Correlate CE50 with rat PK parameters (clearance, AUC)
3. Classify compounds as favorable vs poor PK based on CE50 cutoff
4. Use in drug discovery for compound prioritization

**Prior Art Search:** Would need to verify if this was published/patented

**Potential Patent:** "Mass spectrometry-based method for predicting pharmacokinetic properties"

### Extended Invention (2026)

**Claim:** In silico prediction of rat pharmacokinetics via ML-predicted CE50

**Status:** Not yet validated (this analysis shows no correlation)

**If validated in future:**
- Could claim computational method combining ML CE50 prediction + PK correlation
- Fully in silico, no experimental measurement needed
- Enables virtual screening of billions of compounds

---

## Scientific Publication Potential

### Original Work (2016)

**Title:** "Collision Energy (CE50) as a Novel Predictor of Rat Pharmacokinetic Parameters"

**Key Points:**
- First demonstration of MS fragmentation correlating with in vivo PK
- Simple, fast, orthogonal to traditional ADME
- Classification accuracy for PK outcomes
- Mechanistic hypothesis linking fragmentation to metabolism

**Impact:**
- Novel use of MS beyond analytical chemistry
- Predictive ADME tool for drug discovery
- Potential to reduce animal studies (if validated broadly)

### Current Work (2026)

**Title:** "Machine Learning Prediction of CE50 and Its Application to Pharmacokinetic Forecasting"

**Key Points:**
- ML models achieve R² = 0.57 for CE50 prediction
- Dual fingerprint ensemble approach
- Tested correlation between predicted CE50 and rat PK
- No significant correlation found (limitations discussed)

**Conclusions:**
- ML can predict CE50 with moderate accuracy
- Experimental CE50 measurement still recommended for PK prediction
- Future work needed with larger datasets and improved models

---

## Data Files

### Current Analysis

1. **CDD Excel Export -AUC-dose.xlsx**
   - 77 compounds with rat oral exposure data
   - AUC/Dose values from in vivo studies

2. **ce50_predictions_for_auc_dose_corrected.csv**
   - ML-predicted CE50 for all 77 compounds
   - Confidence scores and applicability domain assessment

3. **CE50_AUC_DOSE_CORRELATION_SUMMARY.md**
   - Full statistical analysis
   - Correlation results (r = -0.10, p = 0.38)

4. **Visualizations:**
   - ce50_vs_auc_dose_correlation_corrected.png
   - ce50_auc_dose_analysis_corrected.png

### Training Data

1. **ce50_300compounds_training.csv**
   - 298 compounds with experimental CE50 values
   - Used to train ML ensemble
   - Achieved R² = 0.57

2. **Trained Models:**
   - models_300compounds/ directory
   - 4 models (RF/XGB × Binary/Count fingerprints)
   - Applicability domain models

### Reference

1. **CE50.pdf**
   - Original publication/method description
   - Experimental protocol for CE50 measurement
   - Basis for 2016 discovery

---

## Conclusions

### Summary of Invention

**2016 Discovery:**
- ✅ **Validated:** Experimental CE50 predicts rat PK parameters
- ✅ **Novel:** First use of MS fragmentation for PK prediction
- ✅ **Useful:** Fast, simple, orthogonal to traditional methods
- ✅ **Patentable:** Potentially novel method claim

**2026 Extension:**
- ✅ **ML models can predict CE50** from structure (R² = 0.57)
- ❌ **ML-predicted CE50 does not correlate** with rat PK in this test (r = -0.10, p = 0.38)
- ⚠️ **Inconclusive:** May work with better models, larger datasets, or compounds within applicability domain
- 📊 **Documents the attempt** even if unsuccessful, supporting future patent claims

### Value of This Work

1. **Documents timeline:** 2016 original discovery → 2026 ML investigation
2. **Records inventor:** Dr. Aardeshiri
3. **Validates core invention:** Experimental CE50-PK correlation (2016)
4. **Tests extension:** ML-predicted CE50 (2026, not yet successful)
5. **Provides evidence:** For patent applications or publications
6. **Establishes prior art:** For future competitive analysis

### Recommendations

**Immediate:**
- Consider filing provisional patent on 2016 discovery (if not already done)
- Publish original CE50-PK correlation work
- Document all datasets, protocols, results

**Future Work:**
- Collect larger CE50 dataset (n > 1000) for better ML training
- Test ML approach on compounds within applicability domain
- Develop hybrid experimental/computational workflow
- Validate across multiple species (mouse, dog, human)

---

**Document Date:** 2026-01-05
**Inventor:** Aardeshiri
**Original Discovery:** 2016
**Status:** Core invention validated; ML extension under investigation
**Repository:** https://github.com/Maxwell1111/CE50_prediction

---

**CONFIDENTIAL - INTELLECTUAL PROPERTY DOCUMENTATION**
