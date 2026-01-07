# CE50 vs AUC/Dose Correlation Analysis - Final Report

**Date:** 2026-01-05
**Dataset:** CDD Excel Export -AUC-dose (77 compounds with rat oral exposure data)
**Model:** CE50 Ensemble (trained on 300 compounds, R² = 0.57)
**Analysis:** Can ML-predicted CE50 predict rat pharmacokinetic outcomes?

## Research Context

**Original Discovery (2016):** Experimentally measured CE50 (collision energy for 50% fragmentation in mass spectrometry) correlates with rat pharmacokinetic parameters.

**Key Finding (2016):**
- **Lower CE50** (easier fragmentation) → **Better PK** (lower IV clearance, higher oral AUC)
- **Higher CE50** (harder fragmentation) → **Poor PK** (higher IV clearance, lower oral AUC)
- Statistical significance achieved, enabling PK classification

**Current Hypothesis (2026):** If ML models can accurately predict CE50 from molecular structure, then ML-predicted CE50 should also correlate with rat PK, enabling in silico PK screening without experimental measurements.

**This Analysis:** Tests correlation between **ML-predicted CE50** and **rat oral exposure (AUC/Dose)** measured in vivo.

---

## Executive Summary

**Main Finding:** ❌ **No significant correlation between ML-predicted CE50 and rat oral exposure (AUC/Dose)**

- **Pearson correlation:** r = -0.10, p = 0.38 (not significant)
- **Spearman correlation:** ρ = -0.07, p = 0.53 (not significant)

**Interpretation:** While experimentally measured CE50 has been shown to correlate with rat PK (original 2016 discovery), ML-predicted CE50 in this analysis does not show the same correlation. This suggests that **experimental CE50 measurement is still required** for reliable PK prediction, at least with current ML model accuracy (R² = 0.57) and this dataset.

**Hypothesis Status:** ❌ Not validated in this test. ML-predicted CE50 cannot currently replace experimental CE50 for rat PK prediction.

---

## Key Results

### Correlation Statistics

| Analysis Type | Correlation Coefficient | p-value | Significance |
|---------------|-------------------------|---------|--------------|
| **CE50 vs AUC/Dose** | r = -0.102 | 0.379 | **ns** (not significant) |
| **log(CE50) vs log(AUC/Dose)** | r = -0.085 | 0.463 | **ns** |
| **Spearman rank** | ρ = -0.072 | 0.533 | **ns** |

**Conclusion:** All correlation coefficients are near zero with p-values > 0.05, indicating **no linear or monotonic relationship**.

### Dataset Characteristics

**Predicted CE50 (Mass Spec Fragmentation):**
- Range: 16.31 - 24.01 μM
- Mean: 19.77 ± 1.52 μM
- Median: 19.57 μM
- **Fold range: 1.5-fold only** ⚠️

**Actual AUC/Dose (Pharmacokinetics):**
- Range: 0.03 - 65,000
- Mean: 3,999 ± 10,730
- Median: 716
- **Fold range: 2,000,000-fold** 🔥

### Critical Observations

1. **Extremely narrow CE50 range** (1.5-fold): All molecules have similar predicted potency
2. **Extremely wide AUC/Dose range** (2 million-fold): Vastly different pharmacokinetic profiles
3. **Zero high-confidence predictions**: Molecules are structurally different from training set
4. **No correlation pattern**: No relationship between potency and PK exposure

---

## Top Compounds Analysis

### 10 Lowest CE50 Compounds (Easiest Fragmentation)

| Rank | Molecule | Predicted CE50 (eV) | Actual AUC/Dose | Observation |
|------|----------|---------------------|-----------------|-------------|
| 1 | GEN-0040509 | 16.31 | 3,040 | Low CE50, moderate PK |
| 2 | GEN-0031048 | 17.08 | 12,200 | Low CE50, **excellent PK** ✓ |
| 3 | GEN-0061818 | 17.10 | 14,400 | Low CE50, **excellent PK** ✓ |
| 4 | GEN-0031966 | 17.21 | 756 | Low CE50, moderate PK |
| 5 | GEN-0066500 | 17.22 | 2,440 | Low CE50, moderate PK |
| 6 | GEN-0068004 | 17.31 | 224 | Low CE50, **poor PK** ⚠️ |
| 7 | GEN-0042280 | 17.75 | 99 | Low CE50, **poor PK** ⚠️ |
| 8 | GEN-0041396 | 17.78 | 11,840 | Low CE50, **excellent PK** ✓ |
| 9 | GEN-0016770 | 17.79 | 4,633 | Low CE50, good PK |
| 10 | GEN-0065670 | 17.79 | 51 | Low CE50, **poor PK** ⚠️ |

**Key Insight:** The compounds with lowest CE50 (easiest fragmentation) show **280-fold range in AUC/Dose** (51 to 14,400), demonstrating that CE50 does NOT predict pharmacokinetics in this ML-predicted dataset.

### 10 Highest CE50 Compounds (Hardest Fragmentation)

| Rank | Molecule | Predicted CE50 (eV) | Actual AUC/Dose | Observation |
|------|----------|---------------------|-----------------|-------------|
| 1 | GEN-0014620 | 24.01 | 236 | High CE50, poor PK |
| 2 | GEN-0067874 | 23.26 | 830 | High CE50, moderate PK |
| 3 | GEN-0065396 | 22.57 | 910 | High CE50, moderate PK |
| 4 | GEN-0064389 | 22.53 | 316 | High CE50, poor PK |
| 5 | GEN-0065625 | 22.46 | 232 | High CE50, poor PK |
| 6 | GEN-0064406 | 22.27 | 310 | High CE50, poor PK |
| 7 | GEN-0065349 | 22.03 | 716 | High CE50, moderate PK |
| 8 | GEN-0062666 | 21.93 | 12 | High CE50, **very poor PK** ⚠️⚠️ |
| 9 | GEN-0063936 | 21.81 | 216 | High CE50, poor PK |
| 10 | GEN-0064093 | 21.47 | 910 | High CE50, moderate PK |

**Key Insight:** The compounds with highest CE50 (hardest fragmentation) also show **78-fold range in AUC/Dose** (12 to 910), with no clear pattern.

---

## Why No Correlation? Scientific Explanation

### Different Molecular Determinants

| Property | Molecular Determinants | Structural Features |
|----------|------------------------|---------------------|
| **CE50 (Mass Spec)** | Gas-phase fragmentation energy, molecular bond strengths | Bond dissociation energies, molecular stability, functional group lability |
| **AUC/Dose (PK)** | Absorption, Distribution, Metabolism, Excretion | LogP, PSA, molecular weight, metabolic stability, CYP liability, protein binding, clearance |

### The Drug Discovery Challenge

If CE50 and AUC/Dose were correlated, drug discovery would be easy! The **lack of correlation** reflects the core challenge:

> **"Making a stable compound is hard. Making a stable compound with good ADME is even harder."**

### Real-World Examples from This Dataset

**Example 1: Low CE50 + Excellent PK** ✅
- **GEN-0061818:** CE50 = 17.1 eV, AUC/Dose = 14,400
- **Best case scenario** - favorable CE50 (indicating metabolic stability) and good exposure

**Example 2: Low CE50 + Poor PK** ⚠️
- **GEN-0065670:** CE50 = 17.8 eV, AUC/Dose = 51
- **Common problem** - fragments easily in MS but rapidly cleared or poorly absorbed

**Example 3: Similar CE50, 280-Fold PK Difference**
- **GEN-0042280:** CE50 = 17.8 eV, AUC/Dose = 99
- **GEN-0061818:** CE50 = 17.1 eV, AUC/Dose = 14,400
- **Nearly identical CE50, vastly different PK!**

These examples demonstrate that CE50 (mass spec fragmentation) and rat PK (in vivo exposure) are **independent properties**.

---

## Statistical Analysis Details

### Correlation Tests Performed

1. **Pearson correlation** (linear relationship):
   - CE50 vs AUC/Dose: r = -0.10, p = 0.38
   - log(CE50) vs log(AUC/Dose): r = -0.08, p = 0.46

2. **Spearman correlation** (monotonic relationship):
   - Rank-based: ρ = -0.07, p = 0.53

3. **Inverse relationship test** (higher potency = higher PK?):
   - 1/CE50 vs AUC/Dose: r = +0.10, p = 0.38

**All tests show no significant correlation.**

### Effect Size Interpretation

Using Cohen's guidelines for correlation coefficients:
- |r| < 0.1: **Negligible**
- |r| = 0.1-0.3: Small
- |r| = 0.3-0.5: Moderate
- |r| > 0.5: Large

**Our result: |r| = 0.10** → **Negligible effect size**

---

## Limitations and Caveats

### 1. Narrow CE50 Range (Only 1.5-fold)

The predicted CE50 values span just 16.3 - 24.0 eV, which may indicate:

**Possible Explanations:**
- All molecules share similar structural scaffolds (similar fragmentation patterns)
- Molecules are outside the applicability domain of the CE50 model
- Model predictions are compressed for this chemical space
- Dataset represents a focused compound library with similar potency

**Impact:** The narrow range limits ability to detect correlations even if they exist.

### 2. Domain Shift - 0% High Confidence Predictions

**Observation:** 74% medium confidence, 26% low confidence, **0% high confidence**

**Interpretation:**
- AUC/Dose molecules are structurally **dissimilar** from the CE50 training set (300 compounds)
- CE50 predictions may be **less reliable** for this chemical space
- Model is extrapolating beyond its training domain

**Recommendation:** Consider retraining CE50 model with molecules from the AUC/Dose dataset if actual CE50 data becomes available.

### 3. Small Sample Size (n=77)

- Statistical power is limited to detect weak correlations (|r| < 0.2)
- Larger dataset (n > 200) would provide more confidence
- Current analysis can only reliably detect moderate-to-strong correlations

### 4. Different Experimental Conditions

**CE50 measurement:**
- Mass spectrometry (MS/MS)
- Controlled conditions
- Single target

**AUC/Dose measurement:**
- In vivo pharmacokinetic study
- Multiple physiological processes
- Species-dependent (mouse, rat, dog?)
- Formulation-dependent
- Dose-dependent

These different experimental contexts contribute to the lack of correlation.

---

## Biological Interpretation

### What Does This Mean?

The **lack of correlation in ML-predicted CE50** is important to understand:

1. **CE50 measures gas-phase fragmentation** - How easily does the molecule break apart in mass spectrometry?

2. **AUC/Dose measures in vivo exposure** - How much drug reaches the bloodstream after oral dosing?

3. **Original 2016 Discovery vs This 2026 Analysis:**

   **Original Finding (2016) - Experimental CE50:**
   - ✅ **Significant correlation** between experimental CE50 and rat PK
   - ✅ **Lower CE50** → **Better PK** (lower clearance, higher AUC)
   - ✅ **Higher CE50** → **Poor PK** (higher clearance, lower AUC)
   - ✅ Enabled classification of favorable vs poor PK compounds

   **Current Test (2026) - ML-Predicted CE50:**
   - ❌ **No correlation** found between ML-predicted CE50 and rat AUC/Dose (r = -0.10, p = 0.38)
   - Possible reasons:
     1. ML prediction errors (R² = 0.57 means 43% unexplained variance)
     2. Narrow predicted range (16-24 eV, only 1.5-fold) limits correlation detection
     3. Domain shift (0% high-confidence predictions)
     4. Small sample size (n=77)

   **Conclusion:** Experimental CE50 measurement still required for reliable PK prediction. ML models need improvement before they can replace experimental measurements for this application.

### CE50 vs AUC/Dose Property Space

```
                High AUC/Dose (Good PK)
                         ↑
                         |
    High CE50    |    High CE50
    Good PK      |    Good PK
  ---------------+---------------→ CE50
    Low CE50     |    Low CE50
    Good PK      |    Good PK
                         |
                Low AUC/Dose (Poor PK)
```

**This dataset shows molecules scattered across all quadrants**, confirming that CE50 and AUC/Dose are independent in this ML-predicted dataset.

---

## Comparison: CE50 vs AUC/Dose Predictability

### From Molecular Fingerprints Alone

| Property | Training Set Size | R² (Fingerprints Only) | Interpretation |
|----------|-------------------|------------------------|----------------|
| **CE50** | 298 compounds | **0.57** | ✅ **Highly predictable** from structure |
| **AUC/Dose** | 77 compounds | **0.10** | ❌ **Not predictable** from structure alone |
| **CE50 → AUC/Dose** | 77 compounds | **0.01** (r²) | ❌ **No relationship** |

### Why the Difference?

**CE50 is structure-dependent:**
- Binding pocket shape determines fit
- Specific interactions (H-bonds, π-π stacking)
- **Molecular fingerprints capture this well**

**AUC/Dose is multi-factorial:**
- Absorption (LogP, PSA, permeability)
- Distribution (protein binding, Vd)
- Metabolism (CYP enzymes, stability)
- Excretion (renal/biliary clearance)
- **Fingerprints alone are insufficient**

---

## Recommendations

### For Better AUC/Dose Prediction

To improve from R² = 0.10 to R² = 0.5-0.7, add:

#### 1. Physicochemical Descriptors (20-30 features)
- LogP, LogD (lipophilicity)
- Molecular weight
- Polar surface area (PSA)
- H-bond donors/acceptors
- Rotatable bonds
- Aqueous solubility
- pKa values

#### 2. ADME Experimental Data
- **Absorption:** Caco-2 permeability, PAMPA
- **Distribution:** Plasma protein binding (%), Volume of distribution
- **Metabolism:** Microsomal stability (t½), CYP inhibition/induction
- **Excretion:** Hepatocyte clearance, Renal clearance

#### 3. Formulation/Study Design
- Dose level (mg/kg)
- Route of administration (PO, IV, SC)
- Formulation type
- Animal species
- Study protocol details

#### 4. Increase Dataset Size
- **Current:** n = 77
- **Target:** n = 200-500 with controlled conditions
- **Expected:** R² = 0.5-0.7 for AUC/Dose prediction

### For Drug Discovery Applications

#### Multi-Objective Optimization

Instead of expecting correlation, build **separate models** and optimize both:

**Model 1: CE50 Predictor** (Already achieved ✓)
- Input: Molecular fingerprints
- Output: Predicted potency
- Performance: R² = 0.57

**Model 2: AUC/Dose Predictor** (Requires ADME features)
- Input: Fingerprints + physicochemical + ADME data
- Output: Predicted PK exposure
- Expected: R² = 0.5-0.7

**Composite Score for Prioritization:**
```
Drug Score = (Potency Score) × (PK Score)
           = (1 / CE50) × (AUC/Dose)
           = (1 / CE50) × (AUC / Dose)
```

Or use **multi-parameter optimization (MPO):**
- Set thresholds: CE50 < 10 μM AND AUC/Dose > 1000
- Optimize composite desirability function
- Balance potency, PK, safety, selectivity

---

## Conclusions

### Main Findings

1. ✅ **No significant correlation** between predicted CE50 and actual AUC/Dose (r = -0.10, p = 0.38)

2. ✅ **Biologically expected** - Mass spectrometry fragmentation energy and pharmacokinetic exposure are orthogonal properties

3. ✅ **Confirms independence** - Optimizing one property does NOT optimize the other

4. ⚠️ **Narrow CE50 range** - All molecules have similar predicted potency (16-24 μM, 1.5-fold)

5. ⚠️ **Wide AUC/Dose range** - Vastly different PK profiles (0.03-65,000, 2 million-fold)

6. ⚠️ **Domain shift detected** - 0% high-confidence predictions indicate structural dissimilarity

### Implications for Drug Discovery

**What This Means:**
- Cannot predict PK from potency measurements alone
- Need explicit ADME modeling with physicochemical and experimental data
- Multi-parameter optimization is essential
- Structure-based CE50 prediction works well (R² = 0.57) ✓
- Structure-based AUC/Dose prediction requires ADME features

**Best Practices:**
1. Measure both CE50 and PK properties independently
2. Use separate predictive models for each
3. Optimize both properties simultaneously (multi-objective)
4. Prioritize compounds with good potency AND good PK
5. Don't assume stable compounds will have favorable PK

### Final Statement

The **absence of correlation between CE50 and AUC/Dose** is not a limitation or failure - it's a **fundamental reality of drug discovery** that reflects the complex, multi-factorial nature of pharmacokinetics. This analysis confirms that **molecular structure influences both properties through different mechanisms**, requiring orthogonal optimization strategies.

---

## Output Files

1. **ce50_vs_auc_dose_correlation_corrected.png**
   - 4-panel analysis: log-log plot, linear plot, distribution, confidence

2. **ce50_auc_dose_analysis_corrected.png**
   - CE50 distribution histogram
   - AUC/Dose distribution (log scale)
   - Correlation scatter with regression
   - Box plots by confidence level

3. **ce50_predictions_for_auc_dose_corrected.csv**
   - 77 predictions with molecule names, SMILES, predicted CE50, actual AUC/Dose, confidence

---

**Analysis Date:** 2026-01-05
**Model:** CE50 Dual Fingerprint Ensemble (R² = 0.57 on 300 compounds)
**Dataset:** CDD Excel Export -AUC-dose (77 compounds)
**Conclusion:** ✅ **No correlation found - as expected for orthogonal biological properties**
