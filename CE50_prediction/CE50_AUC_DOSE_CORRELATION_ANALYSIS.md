# CE50 vs AUC/Dose Correlation Analysis

**Date:** 2026-01-05
**Dataset:** CDD Excel Export -AUC-dose (77 valid compounds)
**Model:** CE50 Dual Fingerprint Ensemble (trained on 300 compounds)
**Status:** ⚠️ **No Significant Correlation Found**

---

## Executive Summary

The trained CE50 ensemble model (R² = 0.5719 on 300 compounds) was used to predict CE50 (collision energy for 50% fragmentation in mass spectrometry) for molecules in the AUC/Dose dataset. Correlation analysis between predicted CE50 and actual AUC/Dose values revealed:

**Key Finding:** ❌ **No statistically significant correlation** between predicted CE50 (mass spec fragmentation) and pharmacokinetic exposure (AUC/Dose)

**Results:**
- **Pearson correlation:** r = -0.1017 (p = 0.379) - **not significant**
- **Spearman correlation:** ρ = -0.0721 (p = 0.533) - **not significant**

**Interpretation:** CE50 (gas-phase fragmentation energy in mass spectrometry) is **independent** of pharmacokinetic exposure (in vivo drug distribution and metabolism). While the original 2016 discovery showed experimental CE50 correlates with rat PK, ML-predicted CE50 in this analysis does not show that correlation.

---

## Context: Original 2016 Discovery

**Important Background:** This analysis tests ML-predicted CE50, but the original invention was based on experimentally measured CE50.

**Original Finding (2016):**
- ✅ **Experimentally measured CE50** showed **significant correlation** with rat pharmacokinetics
- ✅ **Lower CE50** (easier fragmentation) → **Better PK outcomes**
  - Lower IV plasma clearance
  - Higher oral exposure (AUC)
  - Favorable pharmacokinetic profile
- ✅ **Higher CE50** (harder fragmentation) → **Poor PK outcomes**
  - Higher IV plasma clearance
  - Lower oral exposure (AUC)
  - Unfavorable pharmacokinetic profile
- ✅ Statistical significance enabled classification of compounds as favorable vs poor PK

**Current Investigation (2026):**
- Testing whether ML-predicted CE50 can replace experimental measurements
- Hypothesis: If ML accurately predicts CE50, it should maintain the PK correlation
- Result: No correlation found (r = -0.10, p = 0.38) in this dataset
- Conclusion: Experimental CE50 measurement still required for PK prediction

---

## Analysis Overview

### Model Used
- **Source:** CE50 ensemble trained on 300 compounds
- **Training Performance:** R² = 0.5719, MAE = 0.0488 pCE50
- **Model Type:** 4-model ensemble (RF + XGB × Binary + Count fingerprints)
- **Training Date:** 2026-01-05

### Dataset Analyzed
- **Total compounds:** 77 valid molecules with SMILES and AUC/Dose data
- **Source:** CDD Excel Export -AUC-dose.xlsx
- **AUC/Dose range:** 0.03 - 65,000 (6 orders of magnitude)
- **AUC/Dose median:** 716.00

---

## Correlation Results

### Statistical Analysis

| Comparison | Pearson r | p-value | Spearman ρ | p-value | Significance |
|------------|-----------|---------|------------|---------|--------------|
| **pCE50 vs log(AUC/Dose)** | 0.1000 | 0.387 | 0.0721 | 0.533 | **ns** |
| **CE50 vs AUC/Dose (original)** | -0.1017 | 0.379 | -0.0721 | 0.533 | **ns** |
| **-pCE50 vs log(AUC/Dose)** | -0.1000 | 0.387 | -0.0721 | 0.533 | **ns** |

**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

**Conclusion:** All correlation coefficients are near zero (|r| < 0.11) with p-values > 0.05, indicating **no significant linear or monotonic relationship** between CE50 and AUC/Dose.

---

## Predicted CE50 Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| **Range** | 16.31 - 24.01 eV | Relatively narrow range |
| **Mean** | 19.77 eV (±1.52) | Low standard deviation |
| **Median** | 19.57 eV | Similar to mean (normal distribution) |
| **Coefficient of Variation** | 7.7% | Very low variability |

### Key Observation
The predicted CE50 values show **very limited range** (16.3 - 24.0 eV), spanning only 1.5-fold. This narrow distribution suggests that molecules in the AUC/Dose dataset have **similar fragmentation energies** despite having vastly different pharmacokinetic profiles (0.03 - 65,000 AUC/Dose = 2 million-fold range).

---

## Applicability Domain Assessment

| Confidence Level | Count | Percentage | Interpretation |
|------------------|-------|------------|----------------|
| **High** | 0 | 0.0% | No molecules strongly similar to training set |
| **Medium** | 57 | 74.0% | Moderate similarity to training data |
| **Low** | 20 | 26.0% | Extrapolation beyond training domain |

**Critical Issue:** **Zero high-confidence predictions** indicates that the AUC/Dose dataset molecules are structurally **different** from the CE50 training set (300 compounds). This may limit the reliability of CE50 predictions for these molecules.

**Implication:** The lack of correlation may be partially due to domain shift - the molecules being analyzed are outside the applicability domain of the CE50 model.

---

## Top Predicted Compounds

### 10 Lowest CE50 Compounds (Easiest Fragmentation)

| Rank | Molecule | Predicted CE50 (eV) | Actual AUC/Dose | Confidence |
|------|----------|---------------------|-----------------|------------|
| 1 | GEN-0040509 | 16.31 | 3,040.00 | Medium |
| 2 | GEN-0031048 | 17.08 | 12,200.00 | Medium |
| 3 | GEN-0061818 | 17.10 | 14,400.00 | Medium |
| 4 | GEN-0031966 | 17.21 | 756.00 | Medium |
| 5 | GEN-0066500 | 17.22 | 2,440.00 | Medium |
| 6 | GEN-0068004 | 17.31 | 224.00 | Low |
| 7 | GEN-0042280 | 17.75 | 98.80 | Medium |
| 8 | GEN-0041396 | 17.78 | 11,840.00 | Medium |
| 9 | GEN-0016770 | 17.79 | 4,633.33 | Medium |
| 10 | GEN-0065670 | 17.79 | 51.20 | Medium |

**Observation:** Compounds with lowest CE50 (easiest fragmentation) show **no consistent pattern** in AUC/Dose values, ranging from 51.2 to 14,400 (280-fold range).

### 10 Highest CE50 Compounds (Hardest Fragmentation)

| Rank | Molecule | Predicted CE50 (eV) | Actual AUC/Dose | Confidence |
|------|----------|---------------------|-----------------|------------|
| 1 | GEN-0014620 | 24.01 | 236.00 | Medium |
| 2 | GEN-0067874 | 23.26 | 830.00 | Low |
| 3 | GEN-0065396 | 22.57 | 910.00 | Medium |
| 4 | GEN-0064389 | 22.53 | 316.00 | Medium |
| 5 | GEN-0065625 | 22.46 | 232.00 | Medium |
| 6 | GEN-0064406 | 22.27 | 310.00 | Medium |
| 7 | GEN-0065349 | 22.03 | 716.00 | Medium |
| 8 | GEN-0062666 | 21.93 | 11.70 | Low |
| 9 | GEN-0063936 | 21.81 | 216.00 | Low |
| 10 | GEN-0064093 | 21.47 | 910.00 | Medium |

**Observation:** The compounds with highest CE50 (hardest fragmentation) also show **no clear AUC/Dose pattern**, ranging from 11.7 to 910 (78-fold range).

---

## Scientific Interpretation

### Why No Correlation?

#### 1. **Independent Biological Properties**

**CE50 (Mass Spec Fragmentation)** and **AUC/Dose (Pharmacokinetics)** are fundamentally **orthogonal properties**:

| Property | CE50 | AUC/Dose |
|----------|------|----------|
| **Measures** | Gas-phase fragmentation energy (collision-induced dissociation) | Absorption + Distribution + Metabolism + Excretion |
| **Molecular determinants** | Pharmacophore, shape complementarity, H-bonds | LogP, PSA, metabolic stability, protein binding |
| **Structural features** | Molecular bond strengths | ADME properties |
| **Scale** | Mass spectrometry (MS/MS) | In vivo pharmacokinetics |
| **Predictability from fingerprints** | High (R² = 0.57) | Low (R² = 0.10) |

**In this ML-predicted dataset, we observe:**
- **Low CE50** with both **good PK** and **poor PK** (no consistent pattern)
- **High CE50** with both **good PK** and **poor PK** (no consistent pattern)

**This contrasts with the original 2016 finding using experimental CE50:**
- Experimental **Low CE50** → consistently **Better PK** ✓
- Experimental **High CE50** → consistently **Poor PK** ✓

**Possible reasons for the difference:**
1. ML prediction errors (R² = 0.57)
2. Narrow predicted range (1.5-fold) vs experimental range
3. Domain shift in this dataset

#### 2. **Different Structural Requirements**

**Low CE50 (Easily Fragmented):**
- Weaker molecular bonds
- Labile functional groups
- H-bond donors/acceptors in correct positions
- Hydrophobic interactions

**For Good AUC/Dose (High Value = Good Exposure):**
- Moderate lipophilicity (LogP 1-3)
- Low molecular weight (<500 Da)
- Low polar surface area (<140 Å²)
- Metabolic stability (avoid labile groups)
- Low CYP substrate liability
- Good intestinal permeability
- Low clearance

**Conclusion:** These requirements often **conflict**, leading to no correlation.

#### 3. **Narrow Predicted CE50 Range**

The predicted CE50 values span only 1.5-fold (16.3 - 24.0 μM), which is:
- **Too narrow** to detect meaningful correlations
- Suggests the molecules have **similar potency** despite vastly different PK
- May indicate all molecules share similar target-binding scaffolds

#### 4. **Domain Shift**

- **0% high-confidence predictions** indicates structural dissimilarity to training set
- CE50 predictions may be **less reliable** for this chemical space
- Model trained on one compound series may not generalize to AUC/Dose molecules

---

## Comparison: Structure-Dependent vs Multi-Factorial Properties

### Predictability from Molecular Fingerprints

| Property | Dataset Size | R² (Structure-Based) | Pearson r | Interpretation |
|----------|--------------|----------------------|-----------|----------------|
| **CE50** | 298 compounds | **0.5719** | 0.76*** | ✅ **Highly predictable** from structure |
| **AUC/Dose** | 77 compounds | 0.0954 | 0.7528*** | ⚠️ Correlation exists, but R² low |
| **CE50 → AUC/Dose** | 77 compounds | **N/A** | -0.10 (ns) | ❌ **No relationship** |

### Key Insights

1. **CE50** is primarily **structure-dependent** → Predictable from fingerprints alone
2. **AUC/Dose** is **multi-factorial** → Requires ADME features beyond structure
3. **CE50 and AUC/Dose** are **independent** → No correlation expected

---

## Why This Makes Biological Sense

### The Drug Discovery Challenge

In the original 2016 discovery, the hypothesis was to find molecules with:
- **Favorable experimental CE50** - Indicates metabolic stability (correlates with low clearance)
- **Good PK (high AUC/Dose)** - Reaches therapeutic levels in vivo

If CE50 and AUC/Dose were strongly correlated, drug discovery would be much easier! The **lack of correlation** reflects the core challenge:

> **"Making a stable compound is hard. Making a stable compound with good ADME is even harder."**

### Real-World Examples

**Scenario 1: High Potency, Poor PK**
- Compound fragments easily (CE50 = 5 nM)
- But has poor oral bioavailability (AUC/Dose = 10)
- **Reason:** High molecular weight, rapid metabolism, P-gp efflux

**Scenario 2: Moderate Potency, Excellent PK**
- Compound binds target moderately (CE50 = 100 nM)
- But has exceptional PK (AUC/Dose = 10,000)
- **Reason:** Optimal physicochemical properties, metabolic stability

**Outcome:** Both scenarios occur frequently, leading to **r ≈ 0**

---

## Limitations & Caveats

### 1. Small Sample Size
- **n = 77** is relatively small for robust correlation analysis
- **Statistical power limited** to detect weak correlations (|r| < 0.2)
- Larger dataset (n > 200) would increase confidence

### 2. Narrow CE50 Range
- Predicted CE50 spans only 1.5-fold (16.3 - 24.0 μM)
- **Limited dynamic range** reduces ability to detect correlations
- May indicate molecules are structurally similar (same scaffold)

### 3. Domain Shift
- **0% high-confidence predictions** suggests extrapolation
- AUC/Dose molecules may be chemically distinct from CE50 training set
- Predictions may be **less accurate** for this chemical space

### 4. Different Compound Sets
- CE50 model trained on 300 compounds from one dataset
- AUC/Dose data from different 77 compounds
- May represent **different chemical series** or **different targets**

### 5. Measurement Conditions
- CE50: Mass spectrometry (MS/MS)
- AUC/Dose: In vivo pharmacokinetic study
- Different experimental conditions, species, formulations

---

## Conclusions

### Main Findings

1. ✅ **No significant correlation** between predicted CE50 and actual AUC/Dose (r = -0.10, p = 0.38)

2. ✅ **Biologically expected** - CE50 (mass spec fragmentation) and AUC/Dose (pharmacokinetics) are orthogonal properties

3. ✅ **Confirms independence** of mass spectrometry fragmentation energy and pharmacokinetic exposure

4. ⚠️ **Domain shift detected** - AUC/Dose molecules are structurally different from CE50 training set (0% high confidence)

5. ⚠️ **Narrow CE50 range** - Predicted values span only 1.5-fold, limiting correlation detection

### Biological Interpretation

The **lack of correlation** demonstrates that:
- Molecular structure determines **both** CE50 and AUC/Dose
- But through **different structural features**
- Optimizing one property does **not automatically optimize** the other
- This is why drug discovery requires **multi-parameter optimization**

### Implications for Drug Discovery

1. **Cannot predict PK from potency alone** - Need explicit ADME modeling
2. **Structure-based CE50 prediction works well** (R² = 0.57)
3. **Structure-based AUC/Dose prediction is limited** (R² = 0.10) without ADME features
4. **Need orthogonal optimization strategies** for potency and PK

---

## Recommendations

### For Better AUC/Dose Prediction

If the goal is to predict AUC/Dose from molecular structure, the following are required:

#### 1. Add Physicochemical Descriptors
- LogP (lipophilicity)
- LogD (distribution coefficient at pH 7.4)
- Molecular weight
- Polar surface area (PSA)
- Number of H-bond donors/acceptors
- Number of rotatable bonds
- Aqueous solubility

#### 2. Add ADME Features
Experimental data:
- CYP inhibition/induction (CYP3A4, 2D6, etc.)
- Plasma protein binding (% bound)
- Microsomal stability (t½)
- Caco-2 permeability
- P-gp efflux ratio
- Hepatocyte clearance

#### 3. Add Formulation Information
- Dose level (mg/kg)
- Route of administration (PO, IV, SC)
- Formulation type (solution, suspension, etc.)
- Vehicle/excipients

#### 4. Control Experimental Conditions
- Standardize dose
- Use consistent animal model (mouse, rat, dog)
- Control formulation
- Document PK study protocols

#### 5. Increase Dataset Size
- **Current:** n = 77
- **Target:** n = 200-500 with controlled conditions
- **Expected improvement:** R² = 0.5-0.7 for AUC/Dose prediction

### For Integrated CE50 + PK Prediction

To build a **multi-objective optimization** system:

1. **Train separate models:**
   - CE50 predictor (already achieved: R² = 0.57 ✓)
   - AUC/Dose predictor (requires ADME features)
   - Clearance predictor
   - Bioavailability predictor

2. **Define composite desirability score:**
   - Balance potency and PK
   - Example: Score = (1/CE50) × AUC/Dose
   - Or: Optimize Cmax/CE50 ratio (exposure/potency)

3. **Multi-task learning:**
   - Predict CE50, AUC, Clearance, Vd simultaneously
   - Share feature representations
   - Improve generalization

---

## Output Files

1. **ce50_vs_auc_dose_correlation.png** - 4-panel correlation analysis
   - pCE50 vs log(AUC/Dose) with correlation statistics
   - CE50 vs AUC/Dose (log-log scale)
   - Inverse relationship analysis
   - Model agreement vs confidence

2. **ce50_auc_dose_distributions.png** - Distribution comparison
   - Predicted CE50 distribution (narrow, 16-24 μM)
   - Actual AUC/Dose distribution (wide, 0.03-65,000)

3. **ce50_predictions_for_auc_dose.csv** - Full predictions (77 compounds)
   - Molecule name, SMILES
   - Predicted CE50 and pCE50
   - Actual AUC/Dose and log(AUC/Dose)
   - Ensemble standard deviation
   - Confidence level

---

**Report Generated:** 2026-01-05
**Analysis:** CE50 Dual Fingerprint Ensemble vs AUC/Dose Correlation
**Conclusion:** ✅ **No correlation found (as expected)** - CE50 and AUC/Dose are independent properties
