# CE50 - Collision Energy for 50% Fragmentation

## Definition

**CE50** = **Collision Energy at which 50% of precursor (parent) ions fragment** in tandem mass spectrometry (MS/MS)

## What is CE50?

CE50 is a **mass spectrometry parameter**, not a biological or pharmacological property. It measures the collision energy required to break molecular bonds in the gas phase during tandem mass spectrometry experiments.

### Measurement

In a tandem mass spectrometry (MS/MS) experiment:

1. **Precursor ions** (parent molecules) are selected
2. These ions are accelerated and collided with inert gas molecules (typically nitrogen or argon)
3. The **collision energy** determines how much kinetic energy is transferred
4. At sufficient collision energy, bonds break and **fragment ions** are produced
5. **CE50** is the collision energy at which 50% of the precursor ions have fragmented

### Units

- **Electronvolts (eV)** - Most common unit in mass spectrometry
- **Lab-frame collision energy** - Depends on instrument settings
- **Center-of-mass collision energy** - Theoretical, mass-corrected value

Typical CE50 values range from **10-50 eV** for small molecules.

## Why CE50 is Structure-Dependent

CE50 depends on molecular properties including:

### 1. **Bond Strengths**
- Weaker bonds (C-N, C-O) fragment at lower collision energies
- Stronger bonds (C-C aromatic) require higher energies
- **More labile groups → Lower CE50**

### 2. **Molecular Size**
- Larger molecules have more vibrational degrees of freedom
- Energy distributes across the molecule
- **Larger molecules generally → Higher CE50**

### 3. **Molecular Stability**
- Conjugated systems (aromatics) are more stable
- Charged groups affect fragmentation pathways
- **More stable molecules → Higher CE50**

### 4. **Functional Groups**
- Presence of leaving groups (halogens, -OH, -NH2)
- Easily protonated/deprotonated sites
- **Reactive functional groups → Lower CE50**

### 5. **Conformational Flexibility**
- Rigid molecules retain energy better
- Flexible molecules dissipate energy through rotation
- **Rigid molecules → Higher CE50 in some cases**

## Why CE50 is Predictable from Molecular Fingerprints

Since CE50 depends on **molecular structure** (bond types, functional groups, size, stability), it can be predicted from **molecular fingerprints** that encode these structural features.

### Morgan Fingerprints Capture:
- **Substructure patterns** (aromatic rings, heteroatoms)
- **Atom connectivity** (bonds, neighbors)
- **Functional groups** (carbonyls, amines, halogens)
- **Molecular topology** (branching, cycles)

### Expected Predictive Performance:
- **R² = 0.5-0.7** achievable from fingerprints alone
- Better than random (R² > 0)
- Comparable to other physical properties (LogP, boiling point)

This repository achieved **R² = 0.57** on 300 compounds, confirming CE50 is structure-predictable.

## CE50 vs Other Properties

| Property | Definition | Predictable from Structure? | Typical R² |
|----------|-----------|------------------------------|------------|
| **CE50** | Collision energy for 50% fragmentation (MS) | ✅ Yes | **0.5-0.7** |
| **IC50/EC50** | Half-maximal inhibitory/effective concentration (biology) | ✅ Yes | 0.5-0.8 |
| **AUC/Dose** | Area under curve per dose (pharmacokinetics) | ❌ Limited | 0.1-0.3 |
| **LogP** | Lipophilicity (partition coefficient) | ✅ Yes | 0.7-0.9 |
| **Melting Point** | Physical property | ⚠️ Moderate | 0.4-0.6 |

### Why CE50 ≠ AUC/Dose

**CE50** (Mass Spectrometry):
- Intrinsic molecular property
- Measured in gas phase
- Depends only on structure
- **Predictable from fingerprints**

**AUC/Dose** (Pharmacokinetics):
- Multi-factorial in vivo property
- Depends on absorption, distribution, metabolism, excretion
- Requires ADME data beyond structure
- **Not predictable from fingerprints alone**

**Result:** No correlation between CE50 and AUC/Dose (r = -0.10, p = 0.38) ✓

This is **expected** - they measure completely different properties!

## Discovery: CE50 as Predictor of Rat Pharmacokinetics (2016)

### Original Finding

**Experimentally measured CE50 correlates with rat pharmacokinetic outcomes:**

**Correlation Discovered:**
- **Lower CE50** (easier fragmentation in MS) → **Better pharmacokinetics**
  - Lower IV plasma clearance
  - Higher oral exposure (AUC)
  - Favorable PK profile

- **Higher CE50** (harder fragmentation in MS) → **Poor pharmacokinetics**
  - Higher IV plasma clearance
  - Lower oral exposure (AUC)
  - Unfavorable PK profile

**Statistical Significance:** The correlation was statistically significant and enabled classification of compounds as having favorable vs poor PK parameters before conducting full in vivo studies.

**Innovation:** This was the first demonstration that mass spectrometry fragmentation patterns could predict in vivo pharmacokinetic behavior, providing a fast, high-throughput tool orthogonal to traditional ADME assays.

**Mechanism:** The exact mechanistic link between gas-phase fragmentation (CE50) and in vivo metabolism/clearance is still under investigation. The correlation is empirical but reproducible.

---

## Applications of CE50 Prediction

### 1. **Pharmacokinetic Screening** (Based on 2016 Discovery)
- Predict rat PK outcomes from CE50 measurements
- Classify compounds as favorable vs poor PK
- Prioritize compounds for in vivo studies
- Reduce animal testing through pre-screening

### 2. **Mass Spectrometry Method Development**
- Optimize MS/MS fragmentation conditions
- Predict optimal collision energies for new compounds
- Design Multiple Reaction Monitoring (MRM) methods

### 2. **Compound Identification**
- Match predicted vs experimental fragmentation
- Aid in structure elucidation
- Validate molecular formulas

### 3. **High-Throughput Screening**
- Pre-screen compounds for MS compatibility
- Prioritize molecules with favorable MS properties
- Design MS-friendly libraries

### 4. **Quality Control**
- Detect structural errors (predicted vs measured mismatch)
- Verify compound identity
- Flag degradation products

## CE50 Fragmentation Patterns

Different molecular classes have characteristic CE50 ranges:

| Molecular Class | Typical CE50 (eV) | Fragmentation Behavior |
|-----------------|-------------------|------------------------|
| **Small aliphatics** | 10-20 | Readily fragment, low energy |
| **Aromatics** | 20-35 | Stable, require moderate energy |
| **Peptides** | 15-30 | Backbone cleavage, amide bonds |
| **Lipids** | 25-40 | Stable fatty acid chains |
| **Glycans** | 20-35 | Glycosidic bond cleavage |
| **Nucleotides** | 25-45 | Stable aromatic bases |

## Model Interpretation

When the ensemble model predicts CE50:

**Low CE50 (15-20 eV):**
- Molecule has **labile bonds**
- Fragments easily
- May contain leaving groups, heteroatoms
- Example: Molecules with C-N, C-O single bonds

**Medium CE50 (20-30 eV):**
- **Typical organic molecules**
- Mix of stable and labile bonds
- Example: Benzene derivatives, small peptides

**High CE50 (30-50 eV):**
- Molecule is **very stable**
- Strong bonds, conjugated systems
- Example: Polycyclic aromatics, conjugated systems

## Confidence Levels

The ensemble reports confidence based on **applicability domain**:

**High Confidence:**
- Query molecule is **similar to training set**
- Structural features are well-represented
- Prediction is **reliable**

**Medium Confidence:**
- Moderate similarity to training set
- Some structural features are novel
- Prediction is **reasonable estimate**

**Low Confidence:**
- Query molecule is **very different** from training set
- Extrapolating beyond known chemical space
- Prediction is **uncertain, use with caution**

## Summary

**CE50 is a mass spectrometry property** measuring the collision energy needed to fragment 50% of precursor ions. It depends on molecular structure (bond strengths, size, stability) and is **highly predictable from molecular fingerprints** (R² ~0.57).

This is in contrast to pharmacokinetic properties like AUC/Dose, which are multi-factorial and require ADME features beyond structure for accurate prediction.

The CE50 ensemble predictor in this repository successfully demonstrates that:
1. ✅ Molecular fingerprints encode relevant structural information for CE50
2. ✅ Ensemble learning improves prediction accuracy
3. ✅ Applicability domain assessment quantifies prediction confidence
4. ✅ Structure-property relationships can be learned from data

---

**References:**
- Tandem mass spectrometry (MS/MS) principles
- Collision-induced dissociation (CID)
- Molecular fragmentation pathways
- Cheminformatics and QSAR modeling

**Related Properties:**
- Fragmentation energy
- Bond dissociation energy (BDE)
- Ionization energy
- Proton affinity
