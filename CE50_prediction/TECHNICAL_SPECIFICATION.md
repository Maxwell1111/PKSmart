# CE50 Prediction System - Technical Specification

**Version:** 1.0
**Date:** 2026-01-05
**Project Type:** Research Prototype
**Author:** Senior Bioinformatician Team

---

## Executive Summary

This document specifies a production-grade batch processing system for predicting CE50 (half-maximal effective concentration) values from molecular SMILES strings using machine learning. The system combines molecular fingerprinting, ensemble learning, and rigorous validation to support computational chemistry workflows in pharmaceutical R&D.

**Key Objectives:**
- Predict CE50 values from SMILES with quantified confidence
- Support batch processing of 100-100K compounds
- Provide robust applicability domain assessment
- Enable iterative model improvement with incremental learning
- Maintain scientific rigor suitable for publication

---

## 1. System Architecture

### 1.1 Deployment Pattern
**Selected:** Batch Processing Service with Job Queue

**Architecture Components:**
```
┌─────────────────┐
│   User Upload   │
│   (CSV Files)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Celery Queue   │
│   (Redis)       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Worker Processes│
│ (Multi-core)    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Result Storage  │
│ (Local Disk)    │
└─────────────────┘
```

**Rationale:**
- Handles asynchronous job processing for variable-sized datasets
- Simple infrastructure (Celery + Redis)
- Suitable for research prototype with internal R&D users
- Horizontally scalable by adding worker processes

---

## 2. Molecular Featurization

### 2.1 Fingerprint Strategy
**Ensemble Approach - Dual Fingerprint Types:**

1. **Binary Morgan Fingerprints**
   - Radius: 2 (ECFP4 equivalent)
   - Bits: 2048
   - Representation: Presence/absence of substructures

2. **Count-based Morgan Fingerprints**
   - Radius: 2
   - Bits: 2048
   - Representation: Frequency of substructures
   - Captures repeated motifs critical for CE50

**Implementation:**
```python
# Binary fingerprint
fp_binary = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Count-based fingerprint
fp_count = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
```

### 2.2 Configuration Parameters
**Configurable via YAML:**
```yaml
featurization:
  fingerprint_types:
    - binary_morgan
    - count_morgan
  morgan_radius: 2
  morgan_bits: 2048
  use_chirality: false
  use_features: false
```

**Rationale:**
- Binary captures structural similarity (standard in QSAR)
- Count captures substructure frequency (important for potency)
- Ensemble leverages both representations for robust prediction

---

## 3. Data Preprocessing Pipeline

### 3.1 SMILES Validation
**Level 1: RDKit Parsing**
- Accept all RDKit-valid SMILES
- No additional chemical filters (organometallics, peptides accepted)

**Level 2: Data Quality Checks**
1. **Duplicate Detection**
   - Canonicalize SMILES (RDKit canonical)
   - Remove exact duplicates
   - Log duplicates removed with original row numbers

2. **Molecular Standardization**
   ```python
   from rdkit.Chem.MolStandardize import rdMolStandardize
   - Neutralize charges
   - Remove salts/solvents
   - Select canonical tautomer
   ```

3. **Outlier Detection**
   - Statistical methods: Z-score (|z| > 3) and IQR (Q1-1.5*IQR, Q3+1.5*IQR)
   - Flag but don't auto-remove (requires user approval)

4. **Data Quality Report**
   ```json
   {
     "duplicates_removed": 23,
     "molecules_standardized": 145,
     "outliers_flagged": 8,
     "invalid_smiles": 2,
     "final_dataset_size": 1234
   }
   ```
   - Generate before model training
   - User reviews and approves/modifies

### 3.2 Unit Validation
**Automatic Detection Strategy:**
```python
def detect_unit_issues(ce50_values):
    """
    Heuristics to detect mixed units:
    - If max/min > 1000: likely mixed units
    - If std/mean > 5: likely heterogeneous
    - If values cluster in discrete decades: mixed units
    """
    ratio = np.max(ce50_values) / np.min(ce50_values)
    cv = np.std(ce50_values) / np.mean(ce50_values)

    if ratio > 1000:
        warnings.warn("Possible unit inconsistency: 1000x range detected")
    if cv > 5:
        warnings.warn("High variance: check for mixed units")
```

**User Prompt:**
- If suspicious patterns detected, prompt user to confirm units
- Provide conversion factors (nM → μM: /1000, μM → mM: /1000)

### 3.3 pCE50 Transformation
```python
pce50 = -np.log10(ce50_values)
```

**Handling Edge Cases:**
- CE50 ≤ 0: Set to NaN, report to user, exclude from training
- Extreme values (CE50 > 1M or < 0.001): Flag as potential errors

### 3.4 Weighted Regression Support
**Optional Column: `ce50_std` or `ce50_confidence`**

If provided in CSV:
```python
sample_weights = 1 / (ce50_std ** 2)  # Inverse variance weighting
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Use Case:**
- High-throughput screening data (low confidence) vs focused assays (high confidence)
- Downweight noisy measurements during training

---

## 4. Model Architecture

### 4.1 Ensemble Strategy
**Models Trained:**
1. Random Forest with Binary Fingerprints
2. Random Forest with Count Fingerprints
3. XGBoost with Binary Fingerprints
4. XGBoost with Count Fingerprints

**Total: 4 base models**

### 4.2 Feature Scaling
**Pipeline Configuration:**
```python
# Keep StandardScaler for future extensibility
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model_instance)
])
```

**Rationale:**
- Tree-based models don't require scaling, but kept for:
  - Future neural network integration
  - Future linear model baseline comparisons
  - Pipeline consistency

### 4.3 Hyperparameter Optimization

**Strategy: Bayesian Optimization with Optuna**

Replace RandomizedSearchCV:
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Progressive Halving / Hyperband:**
```python
from optuna.pruners import HyperbandPruner
pruner = HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3)
study = optuna.create_study(pruner=pruner)
```

**Warm-Start Capability:**
```yaml
hyperparameters:
  warm_start: true
  previous_best_params: "models/best_params_v1.json"
  search_radius: 0.2  # Search ±20% around previous best
```

**Dynamic CV Folds:**
```python
def get_cv_folds(n_samples):
    if n_samples < 50:
        return min(3, n_samples)  # Leave-one-out if < 3 samples
    elif n_samples < 500:
        return 3
    else:
        return 5
```

**Parameter Grids:**

*Random Forest:*
```yaml
rf_params:
  n_estimators: [50, 100, 200, 300, 500]
  max_depth: [5, 10, 20, 30, null]
  min_samples_split: [2, 5, 10]
  min_samples_leaf: [1, 2, 4]
  max_features: ['sqrt', 'log2', 0.3]
```

*XGBoost:*
```yaml
xgb_params:
  n_estimators: [50, 100, 200, 300, 500]
  max_depth: [2, 3, 5, 7, 10]
  learning_rate: [0.01, 0.03, 0.05, 0.1, 0.2]
  subsample: [0.6, 0.8, 1.0]
  colsample_bytree: [0.6, 0.8, 1.0]
  gamma: [0, 0.1, 0.5]
  reg_alpha: [0, 0.1, 1.0]
  reg_lambda: [1, 1.5, 2]
```

### 4.4 Early Stopping Strategy
**Sequential Model Training:**
```python
# 1. Train Random Forest with Binary FP
rf_binary_score = train_and_evaluate(rf_binary_pipeline)

# 2. If RF score is good (R² > 0.5), train other models
if rf_binary_score > 0.5:
    rf_count_score = train_and_evaluate(rf_count_pipeline)
    xgb_binary_score = train_and_evaluate(xgb_binary_pipeline)
    xgb_count_score = train_and_evaluate(xgb_count_pipeline)
else:
    # Poor performance - trigger diagnostics
    run_diagnostics()
    suggest_simple_baseline()
```

---

## 5. Quality Gates & Validation

### 5.1 Pre-Training Quality Gates
**Minimum Dataset Requirements:**
```python
quality_gates = {
    'min_samples': 30,
    'min_unique_smiles': 20,
    'max_missing_rate': 0.1,
    'min_target_variance': 0.01
}
```

**Checks:**
1. **Sample Size:** n ≥ 30 (alert if n < 50, error if n < 30)
2. **Target Variance:** Warn if std/mean < 0.2
3. **Feature Variance:** Check fingerprint bit usage (>5% bits should be active)

### 5.2 Post-Training Quality Gates
**Performance Thresholds:**
```python
performance_gates = {
    'min_cv_r2': 0.3,      # Halt if R² < 0.3 (worse than mean predictor)
    'min_test_r2': 0.25,
    'max_mae_pce50': 1.0   # pCE50 error should be < 1 log unit
}
```

**Actions on Failure:**
1. **R² < 0.3:** Trigger dataset diagnostics
2. **Diagnostics Include:**
   ```python
   - Sample size analysis
   - Target distribution plot
   - Feature variance heatmap
   - Duplicate check
   - Outlier detection
   ```
3. **Fallback to Simple Baselines:**
   - k-Nearest Neighbors (k=3, 5, 10)
   - Linear Regression with L2 regularization
   - Baseline: mean predictor

4. **Power Analysis:**
   ```python
   def estimate_required_samples(current_r2, current_n, target_r2=0.7):
       """
       Estimate samples needed to achieve target R²
       Based on learning curve extrapolation
       """
       # Fit power law: R² = a * n^b
       # Solve for n at target_r2
   ```

### 5.3 Cross-Validation Strategy
**Stratified K-Fold:**
```python
from sklearn.model_selection import StratifiedKFold

# Bin pCE50 into quintiles for stratification
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

skf = StratifiedKFold(n_splits=get_cv_folds(len(y)), shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y_binned):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and evaluate
```

**Metrics Reported:**
```python
{
    'cv_r2_mean': 0.65,
    'cv_r2_std': 0.08,
    'cv_mae_mean': 0.23,
    'cv_mae_std': 0.05,
    'cv_rmse_mean': 0.31,
    'cv_rmse_std': 0.06,
    'fold_scores': [0.61, 0.68, 0.63, 0.70, 0.65]
}
```

### 5.4 Scientific Validation
**Activity Cliff Analysis:**
```python
def activity_cliff_analysis(X, y, smiles, threshold=0.3):
    """
    Identify structurally similar pairs with large CE50 difference

    Parameters:
    - threshold: Tanimoto similarity threshold (default 0.3)

    Returns:
    - DataFrame of cliff pairs with similarity and CE50 difference
    """
    from rdkit import DataStructs

    cliffs = []
    for i in range(len(smiles)):
        for j in range(i+1, len(smiles)):
            tanimoto = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            pce50_diff = abs(y[i] - y[j])

            if tanimoto > threshold and pce50_diff > 1.0:
                cliffs.append({
                    'smiles1': smiles[i],
                    'smiles2': smiles[j],
                    'similarity': tanimoto,
                    'pce50_diff': pce50_diff,
                    'actual1': y[i],
                    'actual2': y[j],
                    'predicted1': y_pred[i],
                    'predicted2': y_pred[j]
                })

    return pd.DataFrame(cliffs)
```

**Chemical Series Analysis:**
```python
def scaffold_based_evaluation(smiles, y, y_pred):
    """
    Group molecules by Murcko scaffold and evaluate performance per series
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)

        if scaffold_smi not in scaffolds:
            scaffolds[scaffold_smi] = {'actual': [], 'predicted': []}

        scaffolds[scaffold_smi]['actual'].append(y[i])
        scaffolds[scaffold_smi]['predicted'].append(y_pred[i])

    # Calculate R² per scaffold
    scaffold_performance = {}
    for scaffold, data in scaffolds.items():
        if len(data['actual']) >= 3:  # Need minimum samples
            r2 = r2_score(data['actual'], data['predicted'])
            scaffold_performance[scaffold] = {
                'r2': r2,
                'n_compounds': len(data['actual']),
                'mean_actual': np.mean(data['actual']),
                'mean_predicted': np.mean(data['predicted'])
            }

    return scaffold_performance
```

---

## 6. Ensemble Combination Strategy

### 6.1 Dynamic Model Selection
**Per-Molecule Confidence-Based Selection:**

```python
def dynamic_ensemble_prediction(models, X, applicability_scores):
    """
    For each molecule, select the model with highest confidence

    Parameters:
    - models: List of (model, fingerprint_type) tuples
    - X: Feature matrices (one per fingerprint type)
    - applicability_scores: Applicability domain scores per model

    Returns:
    - predictions: Final predictions
    - selected_models: Which model was used for each molecule
    - confidences: Confidence scores
    """
    n_samples = len(X[0])
    predictions = np.zeros(n_samples)
    selected_models = []
    confidences = []

    for i in range(n_samples):
        model_predictions = []
        model_confidences = []

        for j, (model, fp_type) in enumerate(models):
            # Get prediction
            pred = model.predict(X[j][i:i+1])[0]

            # Calculate confidence based on:
            # 1. Applicability domain score
            # 2. Model uncertainty (RF: tree variance, XGB: bootstrap)
            # 3. Ensemble agreement

            conf = calculate_confidence(
                model, X[j][i:i+1],
                applicability_scores[j][i],
                model_type=type(model).__name__
            )

            model_predictions.append(pred)
            model_confidences.append(conf)

        # Select model with highest confidence
        best_idx = np.argmax(model_confidences)
        predictions[i] = model_predictions[best_idx]
        selected_models.append(models[best_idx][1])
        confidences.append(model_confidences[best_idx])

    return predictions, selected_models, confidences
```

### 6.2 Disagreement Threshold
**Trigger Applicability Domain Check:**
```python
def check_ensemble_disagreement(predictions_dict, threshold=0.5):
    """
    If model predictions differ by > threshold pCE50 units,
    flag for applicability domain review

    Returns:
    - high_disagreement_indices: List of molecule indices
    - disagreement_scores: Std dev across models
    """
    predictions_array = np.array(list(predictions_dict.values()))
    disagreement = np.std(predictions_array, axis=0)

    high_disagreement = disagreement > threshold

    return np.where(high_disagreement)[0], disagreement
```

**Applicability Domain Response:**
- Calculate Tanimoto similarity to nearest training neighbors
- Check PCA distance from training centroid
- Flag if both applicability checks fail

---

## 7. Applicability Domain Assessment

### 7.1 Multi-Method Approach
**All Four Methods Implemented:**

**Method 1: Tanimoto Similarity**
```python
def tanimoto_applicability(query_fp, training_fps, threshold=0.3):
    """
    Find nearest neighbor in training set
    Flag if max similarity < threshold
    """
    from rdkit import DataStructs

    similarities = [
        DataStructs.TanimotoSimilarity(query_fp, train_fp)
        for train_fp in training_fps
    ]
    max_sim = max(similarities)

    return {
        'max_similarity': max_sim,
        'within_domain': max_sim >= threshold,
        'nearest_neighbor_idx': np.argmax(similarities)
    }
```

**Method 2: PCA Distance**
```python
from sklearn.decomposition import PCA

def pca_applicability(X_train, X_query, n_components=50):
    """
    Project to PCA space, calculate Mahalanobis distance
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_query_pca = pca.transform(X_query)

    # Mahalanobis distance from training centroid
    centroid = np.mean(X_train_pca, axis=0)
    cov_inv = np.linalg.pinv(np.cov(X_train_pca.T))

    distances = []
    for x in X_query_pca:
        diff = x - centroid
        dist = np.sqrt(diff @ cov_inv @ diff.T)
        distances.append(dist)

    # Use 95th percentile of training distances as threshold
    training_dists = [...]  # Calculate for training set
    threshold = np.percentile(training_dists, 95)

    return {
        'distances': distances,
        'within_domain': [d < threshold for d in distances]
    }
```

**Method 3: One-Class SVM**
```python
from sklearn.svm import OneClassSVM

def train_applicability_svm(X_train):
    """
    Train one-class classifier on training fingerprints
    """
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    svm.fit(X_train)
    return svm

def svm_applicability(svm_model, X_query):
    """
    Predict if query is similar to training distribution
    Returns: 1 for inliers, -1 for outliers
    """
    predictions = svm_model.predict(X_query)
    return predictions  # 1 = within domain, -1 = out of domain
```

**Method 4: Prediction Uncertainty**
```python
def rf_uncertainty(rf_model, X_query):
    """
    Use Random Forest tree variance as uncertainty
    """
    tree_predictions = np.array([
        tree.predict(X_query) for tree in rf_model.estimators_
    ])
    std_dev = np.std(tree_predictions, axis=0)

    # High uncertainty = out of domain
    return {
        'prediction_std': std_dev,
        'high_confidence': std_dev < 0.3  # Threshold tunable
    }

def xgb_uncertainty_bootstrap(xgb_model, X_query, n_bootstrap=100):
    """
    Bootstrap uncertainty for XGBoost
    """
    # Resample training data, retrain, predict
    predictions = []
    for _ in range(n_bootstrap):
        # Bootstrap implementation
        predictions.append(bootstrap_model.predict(X_query))

    std_dev = np.std(predictions, axis=0)
    return std_dev
```

### 7.2 Aggregated Applicability Score
```python
def aggregate_applicability(tanimoto_score, pca_score, svm_score, uncertainty_score):
    """
    Combine all four methods into single confidence level

    Returns: "High", "Medium", "Low"
    """
    # Voting system
    votes_high = 0
    if tanimoto_score['within_domain']: votes_high += 1
    if pca_score['within_domain']: votes_high += 1
    if svm_score == 1: votes_high += 1
    if uncertainty_score['high_confidence']: votes_high += 1

    if votes_high >= 3:
        return "High"
    elif votes_high >= 2:
        return "Medium"
    else:
        return "Low"
```

---

## 8. Visualization Suite

### 8.1 Feature Importance

**SHAP Values for Molecular Attribution:**
```python
import shap

def generate_shap_explanations(model, X_train, X_test, feature_names):
    """
    Generate SHAP values for feature importance
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    # Force plot for individual predictions
    shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

    return shap_values
```

**Fingerprint Bit Decoding to SMARTS:**
```python
def decode_important_bits(model, fingerprint_info, top_n=20):
    """
    Map important fingerprint bits to chemical substructures
    """
    # Get feature importances from Random Forest
    importances = model.named_steps['rf'].feature_importances_
    top_bits = np.argsort(importances)[-top_n:]

    # Decode bits to SMARTS patterns
    from rdkit.Chem import AllChem

    bit_info = {}
    for bit in top_bits:
        # Find which molecules activate this bit
        activating_mols = []
        for mol in training_mols:
            fp_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048, bitInfo=fp_info
            )
            if bit in fp_info:
                activating_mols.append((mol, fp_info[bit]))

        bit_info[bit] = {
            'importance': importances[bit],
            'activating_molecules': len(activating_mols),
            'example_substructure': activating_mols[0] if activating_mols else None
        }

    return bit_info
```

### 8.2 Residual Analysis
```python
def plot_residuals(y_true, y_pred, confidence_labels):
    """
    Residual plot: (predicted - actual) vs predicted
    Colored by confidence level
    """
    residuals = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residual vs Predicted
    scatter = ax1.scatter(y_pred, residuals,
                          c=[confidence_map[c] for c in confidence_labels],
                          cmap='RdYlGn', alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_xlabel('Predicted pCE50')
    ax1.set_ylabel('Residuals (Predicted - Actual)')
    ax1.set_title('Residual Plot')
    plt.colorbar(scatter, ax=ax1, label='Confidence')

    # Histogram of residuals
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')

    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300)
```

### 8.3 Learning Curves
```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    """
    Train/validation performance vs training set size
    Diagnose if more data would help
    """
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, label='Training Score', marker='o')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes_abs, val_mean, label='Validation Score', marker='s')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curves.png', dpi=300)
```

### 8.4 Chemical Space Visualization
```python
from sklearn.manifold import TSNE
import umap

def plot_chemical_space(X, y, method='umap'):
    """
    Project fingerprints to 2D using t-SNE or UMAP
    Color by CE50 value
    """
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard')
    else:  # t-SNE
        reducer = TSNE(n_components=2, metric='jaccard', perplexity=30)

    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                          cmap='viridis', alpha=0.6, s=50, edgecolors='k')
    plt.colorbar(scatter, label='pCE50')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Chemical Space Visualization ({method.upper()})')
    plt.savefig(f'chemical_space_{method}.png', dpi=300)
```

---

## 9. Model Persistence & Versioning

### 9.1 Serialization Strategy
```python
import joblib
import json
from datetime import datetime

def save_model(model, metadata, version, output_dir='models/'):
    """
    Save model with metadata
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{output_dir}/model_v{version}_{timestamp}.pkl"
    metadata_filename = f"{output_dir}/metadata_v{version}_{timestamp}.json"

    # Save model
    joblib.dump(model, model_filename)

    # Save metadata
    metadata_full = {
        'version': version,
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'hyperparameters': model.get_params(),
        'training_data': {
            'n_samples': metadata['n_samples'],
            'n_features': metadata['n_features'],
            'data_hash': metadata['data_hash']
        },
        'performance': {
            'cv_r2': metadata['cv_r2'],
            'test_r2': metadata['test_r2'],
            'test_mae': metadata['test_mae'],
            'test_rmse': metadata['test_rmse']
        },
        'fingerprint_config': {
            'type': metadata['fp_type'],
            'radius': metadata['fp_radius'],
            'n_bits': metadata['fp_nbits']
        }
    }

    with open(metadata_filename, 'w') as f:
        json.dump(metadata_full, f, indent=2)

    return model_filename, metadata_filename
```

### 9.2 Model Registry
```python
class ModelRegistry:
    """
    Track multiple model versions
    """
    def __init__(self, registry_file='models/registry.json'):
        self.registry_file = registry_file
        self.registry = self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': []}

    def register_model(self, model_path, metadata_path, metadata):
        entry = {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'version': metadata['version'],
            'timestamp': metadata['timestamp'],
            'performance': metadata['performance'],
            'status': 'active'
        }
        self.registry['models'].append(entry)
        self.save_registry()

    def get_best_model(self, metric='test_r2'):
        active_models = [m for m in self.registry['models'] if m['status'] == 'active']
        best = max(active_models, key=lambda x: x['performance'][metric])
        return best['model_path']

    def save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
```

### 9.3 Incremental Learning
```python
def incremental_update(existing_model_path, new_X, new_y):
    """
    Update model with new data using warm_start
    """
    # Load existing model
    model = joblib.load(existing_model_path)

    # Check if model supports warm_start
    if hasattr(model.named_steps['rf'], 'warm_start'):
        model.named_steps['rf'].warm_start = True
        model.named_steps['rf'].n_estimators += 100  # Add more trees

        # Fit on combined data
        model.fit(new_X, new_y)
    else:
        # XGBoost: use xgb_model parameter
        existing_xgb = model.named_steps['xgb']
        new_model = xgb.XGBRegressor(**existing_xgb.get_params())
        new_model.fit(new_X, new_y, xgb_model=existing_xgb.get_booster())

    return model
```

### 9.4 Model Migration Tools
```python
def migrate_model_v1_to_v2(old_model_path):
    """
    Convert old model format to new format

    Example: v1 stored model directly, v2 uses pipelines
    """
    old_model = joblib.load(old_model_path)

    # Create new pipeline
    new_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', old_model)
    ])

    # Fit scaler on dummy data (or store scaler params in v1)
    # Save as v2
    version = 2
    save_model(new_pipeline, metadata={...}, version=version)
```

---

## 10. Output Formats

### 10.1 CSV Output
**Predictions with Metadata:**
```csv
smiles,actual_ce50,predicted_ce50,predicted_pce50,confidence,selected_model,tanimoto_similarity,ensemble_std
CCO,23.5,25.1,-1.40,High,count_morgan_rf,0.85,0.12
C1CCCCC1,15.2,18.3,-1.26,Medium,binary_morgan_xgb,0.42,0.34
...
```

### 10.2 JSON Output
**Full Job Results:**
```json
{
  "job_id": "job_20260105_103045",
  "status": "completed",
  "timestamp": "2026-01-05T10:35:22Z",
  "metadata": {
    "input_file": "kinase_compounds.csv",
    "n_compounds_submitted": 1234,
    "n_compounds_processed": 1189,
    "n_invalid_smiles": 23,
    "n_duplicates_removed": 22,
    "processing_time_seconds": 342.5
  },
  "model_performance": {
    "best_model": "count_morgan_rf",
    "cv_r2_mean": 0.68,
    "cv_r2_std": 0.07,
    "test_r2": 0.71,
    "test_mae": 0.24,
    "test_rmse": 0.31
  },
  "predictions": [
    {
      "smiles": "CCO",
      "actual_ce50": 23.5,
      "predicted_ce50": 25.1,
      "predicted_pce50": -1.40,
      "confidence": "High",
      "selected_model": "count_morgan_rf",
      "applicability_scores": {
        "tanimoto_similarity": 0.85,
        "pca_distance": 1.2,
        "svm_in_domain": true,
        "prediction_std": 0.12
      }
    }
  ],
  "quality_report": {
    "duplicates_removed": 22,
    "outliers_flagged": 8,
    "unit_warnings": []
  },
  "visualizations": {
    "predictions_plot": "outputs/job_20260105_103045/predictions.png",
    "residuals_plot": "outputs/job_20260105_103045/residuals.png",
    "learning_curves": "outputs/job_20260105_103045/learning_curves.png",
    "chemical_space": "outputs/job_20260105_103045/chemical_space_umap.png"
  }
}
```

---

## 11. Configuration Management

### 11.1 YAML Configuration File
**`config.yaml`:**
```yaml
# Data Configuration
data:
  input_file: "ce50_compounds.csv"
  smiles_column: "smiles"
  target_column: "ce50"
  confidence_column: null  # Optional: "ce50_std"

# Preprocessing
preprocessing:
  remove_duplicates: true
  standardize_molecules: true
  detect_outliers: true
  auto_approve_cleaning: false  # Require user approval

# Featurization
featurization:
  fingerprint_types:
    - binary_morgan
    - count_morgan
  morgan_radius: 2
  morgan_bits: 2048
  use_chirality: false
  use_features: false

# Memory Optimization
memory:
  use_sparse_matrices: true
  on_the_fly_generation: false
  chunk_size: 10000  # For streaming processing

# Model Training
models:
  algorithms:
    - random_forest
    - xgboost
  early_stopping: true
  early_stopping_threshold: 0.5  # R² threshold to train additional models

# Hyperparameter Optimization
hyperparameter_tuning:
  method: "optuna"  # "random_search", "optuna", "hyperband"
  n_trials: 50
  cv_folds: "auto"  # Or fixed: 3, 5, 10
  warm_start: false
  previous_best_params: null

# Cross-Validation
cross_validation:
  strategy: "stratified_kfold"
  n_splits: 5
  shuffle: true
  random_state: 42

# Quality Gates
quality_gates:
  min_samples: 30
  min_cv_r2: 0.3
  min_test_r2: 0.25
  max_mae_pce50: 1.0
  run_diagnostics_on_failure: true

# Applicability Domain
applicability_domain:
  methods:
    - tanimoto_similarity
    - pca_distance
    - one_class_svm
    - prediction_uncertainty
  tanimoto_threshold: 0.3
  pca_percentile: 95

# Ensemble
ensemble:
  combination_method: "dynamic_selection"  # "weighted_average", "stacking"
  disagreement_threshold: 0.5  # pCE50 units

# Visualization
visualization:
  generate_plots: true
  plots:
    - predictions_vs_actual
    - residuals
    - learning_curves
    - chemical_space
    - feature_importance
  chemical_space_method: "umap"  # "tsne"

# Output
output:
  formats:
    - csv
    - json
  output_directory: "outputs/"
  save_models: true
  model_directory: "models/"

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"  # "text", "json"
  output: "both"  # "console", "file", "both"
  log_file: "logs/ce50_prediction.log"

# Monitoring
monitoring:
  enable_mlflow: true
  mlflow_tracking_uri: "http://localhost:5000"
  experiment_name: "CE50_Prediction"

# Batch Processing
batch_processing:
  queue_backend: "celery"
  broker_url: "redis://localhost:6379/0"
  result_backend: "redis://localhost:6379/1"
  max_workers: 4
  job_timeout: 3600  # seconds

# Security
security:
  max_file_size_mb: 100
  max_rows: 1000000
  max_smiles_length: 500
  allowed_elements: ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
```

### 11.2 Configuration Loading
```python
import yaml

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key_path, default=None):
        """
        Get nested config value using dot notation
        e.g., config.get('data.input_file')
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {})
            if value == {}:
                return default
        return value if value != {} else default
```

---

## 12. Error Handling & Logging

### 12.1 Error Handling Strategy
```python
class CE50PredictionError(Exception):
    """Base exception for CE50 prediction pipeline"""
    pass

class DataValidationError(CE50PredictionError):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(CE50PredictionError):
    """Raised when model training fails"""
    pass

# Fail-fast for critical errors
try:
    df = load_and_validate_data(config.get('data.input_file'))
except FileNotFoundError:
    logger.error("Input file not found", extra={'file': input_file})
    raise DataValidationError("Input file does not exist")

# Continue with warnings for minor issues
invalid_smiles = []
for i, smi in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_smiles.append((i, smi))
        logger.warning(f"Invalid SMILES at row {i}", extra={'smiles': smi})

# Report all errors at end
if invalid_smiles:
    logger.info(f"Removed {len(invalid_smiles)} invalid SMILES")
    with open('error_log.csv', 'w') as f:
        f.write("row,smiles,error\n")
        for row, smi in invalid_smiles:
            f.write(f"{row},{smi},Invalid SMILES\n")

# Retry logic with exponential backoff (for future database/API integration)
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt+1}/{max_retries} after {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator
```

### 12.2 Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, 'job_id'):
            log_obj['job_id'] = record.job_id
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'duration_ms'):
            log_obj['duration_ms'] = record.duration_ms

        return json.dumps(log_obj)

# Setup logger
logger = logging.getLogger('ce50_prediction')
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# File handler with JSON
file_handler = logging.FileHandler('logs/ce50_prediction.log')
file_handler.setFormatter(JSONFormatter())

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Usage
logger.info("Starting model training", extra={
    'job_id': 'job_123',
    'n_samples': 1234,
    'model_type': 'random_forest'
})
```

### 12.3 Metrics Export (Prometheus)
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
job_counter = Counter('ce50_jobs_total', 'Total number of jobs processed', ['status'])
job_duration = Histogram('ce50_job_duration_seconds', 'Job processing duration')
compounds_processed = Counter('ce50_compounds_total', 'Total compounds processed')
model_r2 = Gauge('ce50_model_r2', 'Current model R² score')

# Usage
with job_duration.time():
    result = process_job(job_id)

job_counter.labels(status='success').inc()
compounds_processed.inc(len(df))
model_r2.set(best_model_r2)

# Start Prometheus metrics server
start_http_server(8000)  # Metrics available at localhost:8000/metrics
```

### 12.4 MLflow Integration
```python
import mlflow

mlflow.set_tracking_uri(config.get('monitoring.mlflow_tracking_uri'))
mlflow.set_experiment(config.get('monitoring.experiment_name'))

with mlflow.start_run(run_name=f"job_{job_id}"):
    # Log parameters
    mlflow.log_params({
        'fingerprint_type': 'binary_morgan',
        'morgan_radius': 2,
        'morgan_bits': 2048,
        'model_type': 'random_forest',
        **best_params
    })

    # Log metrics
    mlflow.log_metrics({
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    })

    # Log artifacts
    mlflow.log_artifact('outputs/predictions.csv')
    mlflow.log_artifact('outputs/predictions.png')

    # Log model
    mlflow.sklearn.log_model(best_model, "model")
```

---

## 13. Testing Strategy

### 13.1 Integration Tests
```python
import pytest
import pandas as pd

class TestCE50Pipeline:

    @pytest.fixture
    def benchmark_dataset(self):
        """Load ChEMBL kinase inhibitor benchmark"""
        return pd.read_csv('tests/data/chembl_kinase_benchmark.csv')

    def test_pipeline_on_benchmark(self, benchmark_dataset):
        """Test full pipeline on known dataset with expected performance"""
        config = Config('tests/configs/benchmark_config.yaml')

        result = run_full_pipeline(
            data=benchmark_dataset,
            config=config
        )

        # Assert minimum performance
        assert result['test_r2'] > 0.6, "R² below acceptable threshold"
        assert result['test_mae'] < 0.4, "MAE too high"

        # Assert all outputs generated
        assert os.path.exists(result['output_csv'])
        assert os.path.exists(result['output_json'])
        assert os.path.exists(result['model_path'])
```

### 13.2 Performance Regression Tests
```python
def test_training_performance():
    """Ensure training time doesn't regress"""
    import time

    X, y = load_test_dataset(n_samples=1000)

    start = time.time()
    model = train_model(X, y)
    duration = time.time() - start

    # Should complete in < 60 seconds for 1000 samples
    assert duration < 60, f"Training took {duration}s, expected < 60s"

def test_prediction_performance():
    """Ensure prediction latency is acceptable"""
    model = load_test_model()
    X_query = generate_test_fingerprints(n_samples=100)

    start = time.time()
    predictions = model.predict(X_query)
    duration = time.time() - start

    # Should predict 100 compounds in < 1 second
    assert duration < 1.0, f"Prediction took {duration}s for 100 compounds"
```

---

## 14. Prediction API (Future Extension)

### 14.1 API Design
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model at startup
registry = ModelRegistry()
best_model_path = registry.get_best_model()
model = joblib.load(best_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single SMILES prediction

    Request:
    {
        "smiles": "CCO",
        "return_confidence": true
    }

    Response:
    {
        "smiles": "CCO",
        "predicted_ce50": 25.1,
        "predicted_pce50": -1.40,
        "confidence": "High",
        "applicability_score": 0.85
    }
    """
    data = request.json
    smiles = data['smiles']

    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid SMILES'}), 400

    # Generate fingerprint
    fp = smiles_to_morgan_fp(smiles)

    # Predict
    pce50_pred = model.predict([fp])[0]
    ce50_pred = 10 ** (-pce50_pred)

    # Calculate confidence
    confidence = calculate_confidence(model, [fp])

    return jsonify({
        'smiles': smiles,
        'predicted_ce50': float(ce50_pred),
        'predicted_pce50': float(pce50_pred),
        'confidence': confidence['label'],
        'applicability_score': confidence['tanimoto_similarity']
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint

    Request (multipart/form-data):
    - file: CSV with 'smiles' column

    Response:
    - job_id for tracking
    """
    file = request.files['file']
    df = pd.read_csv(file)

    # Enqueue job
    job = celery_app.send_task('tasks.batch_prediction', args=[df.to_dict()])

    return jsonify({
        'job_id': job.id,
        'status': 'queued',
        'estimated_time': estimate_processing_time(len(df))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 15. Documentation Requirements

### 15.1 FAQ Document
**`FAQ.md`:**

```markdown
# CE50 Prediction - Frequently Asked Questions

## Common Issues

### Q: Why is my R² negative?
**A:** Negative R² means the model performs worse than predicting the mean value for all compounds. This typically happens when:
- Dataset is too small (< 50 compounds)
- CE50 values have very low variance (all similar)
- Data quality issues (mixed units, errors)
- Molecules are too diverse to learn patterns

**Solutions:**
1. Check dataset size - need minimum 50-100 compounds
2. Verify unit consistency
3. Review data quality report for outliers
4. Consider collecting more targeted data

### Q: How much data do I need?
**A:** Depends on complexity:
- **Minimum:** 50 compounds (expect R² 0.3-0.5)
- **Good:** 200-500 compounds (expect R² 0.5-0.7)
- **Excellent:** 1000+ compounds (expect R² 0.7-0.85)

More diverse chemical space requires more data.

### Q: Why are all predictions similar?
**A:** Common causes:
1. Training data has low variance (all CE50 values within narrow range)
2. Model defaults to predicting mean value
3. Out-of-domain: query molecules unlike training set

Check learning curves - if training and validation scores are both low and flat, data may not have learnable patterns.

### Q: What does "Low Confidence" mean?
**A:** Predictions marked "Low Confidence" when:
- Molecule structurally dissimilar to training set (Tanimoto < 0.3)
- Out of applicability domain (fails 3+ domain checks)
- High model disagreement (ensemble models diverge)

**Recommendation:** Experimentally validate low-confidence predictions.

### Q: Can I mix different assay types?
**A:** Not recommended. Different assays may measure different phenomena:
- Cell-based vs biochemical assays
- Different cell lines
- Different incubation times

Train separate models for each assay type.

### Q: How do I interpret pCE50?
**A:** pCE50 = -log10(CE50_in_eV)
- CE50 = 10 eV → pCE50 = -1
- CE50 = 1 eV → pCE50 = 0
- Higher pCE50 = higher fragmentation energy (more stable molecule in MS)
- 1 pCE50 unit = 10-fold difference in fragmentation energy
```

---

## 16. Development Roadmap

### Phase 1: Core Pipeline (Weeks 1-2)
- [x] Basic data loading and validation
- [x] Morgan fingerprint generation
- [x] RF and XGBoost training
- [x] Basic visualization
- [ ] Configuration file support
- [ ] Structured logging
- [ ] Error handling framework

### Phase 2: Advanced Features (Weeks 3-4)
- [ ] Ensemble with count fingerprints
- [ ] Dynamic model selection
- [ ] Applicability domain (all 4 methods)
- [ ] SHAP interpretability
- [ ] Learning curves and residual plots
- [ ] Chemical space visualization

### Phase 3: Validation & Quality (Weeks 5-6)
- [ ] Activity cliff analysis
- [ ] Chemical series evaluation
- [ ] Quality gates implementation
- [ ] Data cleaning pipeline
- [ ] Unit detection and validation
- [ ] Weighted regression support

### Phase 4: Production Features (Weeks 7-8)
- [ ] Model serialization and registry
- [ ] Incremental learning
- [ ] Celery + Redis batch processing
- [ ] CSV/JSON output formats
- [ ] Integration tests
- [ ] Performance benchmarks

### Phase 5: Monitoring & Ops (Weeks 9-10)
- [ ] MLflow integration
- [ ] Prometheus metrics
- [ ] Correlation ID tracking
- [ ] Job queue monitoring
- [ ] Model migration tools

### Phase 6: API & Documentation (Weeks 11-12)
- [ ] Prediction API endpoints
- [ ] Interactive notebook tutorials
- [ ] FAQ documentation
- [ ] API reference docs
- [ ] User guide with examples

---

## 17. Success Metrics

### Technical Metrics
- **Model Performance:** Test R² > 0.6 on benchmark datasets
- **Prediction Latency:** < 10ms per compound
- **Training Time:** < 5 minutes for 1000 compounds
- **Memory Usage:** < 2GB for 10K compounds
- **Uptime:** 99% availability for batch processing queue

### Scientific Metrics
- **Reproducibility:** 100% reproducible results with same data/config/seed
- **Applicability:** > 80% of pharma R&D compounds within domain
- **Confidence Calibration:** "High" predictions have < 0.3 RMSE
- **Activity Cliff Detection:** Identify known SAR cliffs in validation set

### User Metrics
- **Time to First Model:** < 5 minutes for experienced user
- **Documentation Coverage:** All features documented with examples
- **Error Rate:** < 1% jobs fail due to bugs (vs data issues)

---

## 18. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Small dataset yields poor models | High | High | Implement quality gates, suggest data collection |
| Mixed units corrupt training | Medium | Critical | Auto-detection + user validation |
| Out-of-domain predictions trusted | Medium | High | Mandatory applicability domain scoring |
| Model performance degrades over time | Medium | Medium | Monitor with MLflow, automated retraining |
| Security vulnerability (malicious SMILES) | Low | High | SMILES sanitization, resource limits |
| Infrastructure downtime | Low | Medium | Queue persistence, job retry logic |

---

## 19. Dependencies

### Python Packages
```requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
rdkit>=2023.9.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.43.0
umap-learn>=0.5.4
optuna>=3.4.0
celery>=5.3.0
redis>=5.0.0
flask>=3.0.0
mlflow>=2.8.0
prometheus-client>=0.19.0
pyyaml>=6.0.0
pytest>=7.4.0
joblib>=1.3.0
```

### Infrastructure
- Redis 7.0+ (job queue)
- Python 3.10+
- Multi-core CPU (4+ cores recommended)
- 8GB+ RAM for medium datasets (1K-10K compounds)
- 50GB disk space (models, logs, outputs)

---

## 20. Appendices

### Appendix A: YAML Configuration Template
See Section 11.1

### Appendix B: API Request/Response Examples
See Section 14.1

### Appendix C: Sample Data Format
```csv
smiles,ce50,compound_name,ce50_std
CCO,23.5,Ethanol,2.1
C1CCCCC1,15.2,Cyclohexane,1.8
c1ccccc1,19.8,Benzene,2.3
```

### Appendix D: Performance Benchmarks
| Dataset Size | Fingerprint Gen | Training (RF) | Training (XGB) | Prediction (1K) |
|--------------|----------------|---------------|----------------|-----------------|
| 100 compounds | 0.5s | 2s | 3s | 0.1s |
| 1K compounds | 5s | 15s | 25s | 0.5s |
| 10K compounds | 50s | 120s | 180s | 2s |
| 100K compounds | 500s | 900s | 1200s | 15s |

### Appendix E: Glossary
- **CE50:** Concentration producing 50% of maximum effect
- **pCE50:** -log10(CE50), normalized potency scale
- **Morgan Fingerprint:** Circular molecular fingerprint (ECFP equivalent)
- **Applicability Domain:** Region of chemical space where model is reliable
- **Activity Cliff:** Structurally similar molecules with large potency difference
- **Tanimoto Similarity:** Jaccard coefficient for molecular fingerprints (0-1 scale)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-05 | Senior Bioinformatician Team | Initial specification based on requirements interview |

**Approval:**
- [ ] Technical Lead
- [ ] Product Owner
- [ ] QA Lead

**Next Review Date:** 2026-02-05

---

*End of Technical Specification*
