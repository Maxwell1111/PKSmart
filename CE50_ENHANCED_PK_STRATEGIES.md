# Leveraging Predicted CE50 for PK Prediction Enhancement

**Key Insight:** Experimental CE50 is useful for **ranking/prioritization**, but predicted CE50 as linear features provides little value in **regression models**.

**Problem:** Simply adding predicted CE50 as 3 additional features (ce50, pce50, confidence) among 500+ structural descriptors doesn't capture the predictive relationship observed with experimental CE50.

---

## Why Linear Features Don't Work

### Issue 1: Information Redundancy
- Predicted CE50 is derived from the **same structural features** already in the model
- Mordred + Morgan fingerprints → Predict CE50 (R² = 0.57)
- Same Mordred + Morgan fingerprints → Predict PK
- **Adding predicted CE50 is adding derived information, not new information**

### Issue 2: Prediction Error Propagation
- Predicted CE50 has errors (R² = 0.57, so 43% unexplained variance)
- Using noisy predictions as features adds error without adding signal
- Experimental CE50 has measurement error but no prediction error

### Issue 3: Non-linear Relationship
- Experimental CE50 ranking works because it's used as a **threshold/binary classifier**
  - Low CE50 → Favorable PK (prioritize)
  - High CE50 → Poor PK (deprioritize)
- Linear regression doesn't capture this threshold behavior

---

## Strategy 1: CE50-Based Compound Stratification

**Concept:** Train separate PK models for compounds predicted to have favorable vs unfavorable CE50.

### Implementation

```python
# Step 1: Predict CE50 for all compounds
ce50_predictions = ce50_model.predict(structural_features)

# Step 2: Stratify compounds by CE50
favorable = compounds[ce50_predictions < threshold]  # e.g., 20 eV
unfavorable = compounds[ce50_predictions >= threshold]

# Step 3: Train separate PK models
pk_model_favorable = train_model(favorable_compounds, favorable_pk_values)
pk_model_unfavorable = train_model(unfavorable_compounds, unfavorable_pk_values)

# Step 4: Prediction
if predicted_ce50 < threshold:
    pk_pred = pk_model_favorable.predict(features)
else:
    pk_pred = pk_model_unfavorable.predict(features)
```

### Expected Benefit
- Captures non-linear CE50-PK relationship
- Each stratified model learns specific structure-PK patterns for its CE50 range
- Mimics how experimental CE50 is used for prioritization

### Test Plan
1. Split compounds into tertiles by predicted CE50 (low/medium/high)
2. Train separate PK models for each tertile
3. Compare R² of stratified models vs. unified baseline model
4. Hypothesis: Stratified models will show higher R² within each CE50 range

---

## Strategy 2: CE50-Structural Feature Interactions

**Concept:** Create interaction terms between CE50 and key structural descriptors.

### Implementation

```python
# Instead of: [mordred_1, mordred_2, ..., ce50, pce50, confidence]
# Use: [mordred_1, mordred_2, ..., ce50, pce50, confidence,
#       ce50 × LogP, ce50 × MW, ce50 × HBD, ce50 × PSA, ...]

# Generate interaction features
ce50_pred = ce50_model.predict(features)

interaction_features = []
for descriptor in key_descriptors:  # LogP, MW, HBD, PSA, etc.
    interaction_features.append(ce50_pred * descriptor)

enhanced_features = np.hstack([
    structural_features,
    ce50_pred.reshape(-1, 1),
    np.array(interaction_features).T
])
```

### Expected Benefit
- Captures synergistic effects (e.g., "high CE50 + high MW → poor clearance")
- Allows CE50 to modulate the effect of structural features
- More expressive than additive features

### Test Plan
1. Generate CE50 × top 20 most important structural descriptors
2. Train PK model with interaction features
3. Compare R² vs baseline + linear CE50
4. Hypothesis: Interactions will capture CE50-dependent structure-PK relationships

---

## Strategy 3: CE50 as a Residual Correction Term

**Concept:** Train baseline PK model first, then use CE50 to correct residuals.

### Implementation

```python
# Step 1: Train baseline PK model (no CE50)
baseline_model = train_model(structural_features, pk_values)
baseline_predictions = baseline_model.predict(structural_features)

# Step 2: Calculate residuals
residuals = pk_values - baseline_predictions

# Step 3: Train CE50 → residual correction model
ce50_pred = ce50_model.predict(structural_features)
residual_model = train_model(ce50_pred, residuals)

# Step 4: Enhanced prediction
final_prediction = baseline_prediction + residual_model.predict(ce50_pred)
```

### Expected Benefit
- Isolates the specific contribution of CE50 beyond structural features
- Baseline model captures bulk structure-PK relationship
- CE50 correction captures fragmentation-specific effects

### Test Plan
1. Train baseline RF model on structural features
2. Train CE50-based residual correction (linear regression or RF)
3. Compare R² of baseline vs baseline + CE50 correction
4. Hypothesis: CE50 correction will reduce systematic biases

---

## Strategy 4: Multi-Task Learning (Joint CE50 + PK Prediction)

**Concept:** Train a neural network to simultaneously predict CE50 and PK endpoints.

### Implementation

```python
# Shared encoder
input_layer = Input(shape=(n_features,))
shared_encoder = Dense(256, activation='relu')(input_layer)
shared_encoder = Dense(128, activation='relu')(shared_encoder)

# CE50 prediction head
ce50_head = Dense(64, activation='relu')(shared_encoder)
ce50_output = Dense(1, name='ce50')(ce50_head)

# PK prediction head
pk_head = Dense(64, activation='relu')(shared_encoder)
pk_output = Dense(1, name='pk')(pk_head)

# Multi-task model
model = Model(inputs=input_layer, outputs=[ce50_output, pk_output])
model.compile(
    optimizer='adam',
    loss={'ce50': 'mse', 'pk': 'mse'},
    loss_weights={'ce50': 0.3, 'pk': 0.7}  # PK is primary task
)
```

### Expected Benefit
- Shared representation learns features useful for both CE50 and PK
- CE50 task acts as regularization for PK task
- Leverages correlation without explicitly using predicted CE50

### Test Plan
1. Implement multi-task neural network
2. Compare single-task PK model vs multi-task (CE50 + PK)
3. Hypothesis: Multi-task learning will improve PK R² by learning better representations

---

## Strategy 5: Ensemble/Stacking with CE50

**Concept:** Use CE50-based models as part of an ensemble.

### Implementation

```python
# Level 0: Multiple base models
model_1 = train_rf(structural_features, pk_values)  # Baseline
model_2 = train_gbm(structural_features, pk_values)  # Different algorithm
model_3 = train_rf(structural_features + ce50_features, pk_values)  # With CE50

# Level 1: Meta-model
predictions_1 = model_1.predict(features)
predictions_2 = model_2.predict(features)
predictions_3 = model_3.predict(features)

meta_features = np.column_stack([predictions_1, predictions_2, predictions_3])
meta_model = LinearRegression()
meta_model.fit(meta_features, pk_values)

# Final prediction
final_pred = meta_model.predict(meta_features)
```

### Expected Benefit
- Combines predictions from models with/without CE50
- Meta-model learns when to weight CE50-based predictions higher
- Captures ensemble diversity

---

## Strategy 6: CE50-Based Applicability Domain

**Concept:** Use CE50 confidence to weight predictions.

### Implementation

```python
# Predict PK with uncertainty
pk_pred, pk_uncertainty = model.predict(features, return_std=True)

# Get CE50 confidence
ce50_conf = ce50_model.predict_confidence(features)

# Weight prediction by CE50 confidence
# High CE50 confidence → Trust PK prediction more
# Low CE50 confidence → Flag as uncertain
weighted_uncertainty = pk_uncertainty / ce50_conf

# Use for decision-making
if weighted_uncertainty < threshold:
    return pk_pred  # Reliable prediction
else:
    return "Uncertain - recommend experimental measurement"
```

### Expected Benefit
- CE50 confidence identifies compounds outside applicability domain
- Improves trust calibration, not necessarily R²
- Useful for prioritization and decision-making

---

## Strategy 7: CE50 Binning for Categorical Prediction

**Concept:** Treat CE50 as a categorical variable (low/medium/high fragmentation).

### Implementation

```python
# Bin predicted CE50
ce50_pred = ce50_model.predict(features)
ce50_bins = pd.cut(ce50_pred, bins=3, labels=['low', 'medium', 'high'])

# One-hot encode
ce50_categorical = pd.get_dummies(ce50_bins, prefix='ce50_bin')

# Train model with categorical CE50
enhanced_features = np.hstack([structural_features, ce50_categorical])
model = train_model(enhanced_features, pk_values)
```

### Expected Benefit
- Captures non-linear threshold effects
- More flexible than continuous CE50 values
- Aligns with how experimental CE50 is used (favorable vs unfavorable)

---

## Strategy 8: Transfer Learning from CE50 Prediction

**Concept:** Pre-train neural network on CE50 task, fine-tune on PK task.

### Implementation

```python
# Step 1: Pre-train on CE50 prediction (large dataset if available)
pretrain_model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # CE50 output
])
pretrain_model.fit(features, ce50_values, epochs=100)

# Step 2: Remove final layer, freeze early layers
base_model = Model(
    inputs=pretrain_model.input,
    outputs=pretrain_model.layers[-2].output  # 64-dim representation
)
for layer in base_model.layers[:-2]:
    layer.trainable = False  # Freeze early layers

# Step 3: Add PK prediction head and fine-tune
x = base_model.output
x = Dense(32, activation='relu')(x)
pk_output = Dense(1)(x)
finetune_model = Model(inputs=base_model.input, outputs=pk_output)
finetune_model.fit(features, pk_values, epochs=50)
```

### Expected Benefit
- Leverages large CE50 dataset for representation learning
- Learned features may capture fragmentation-related properties useful for PK
- Particularly valuable if CE50 dataset >> PK dataset

---

## Recommended Testing Order

### Phase 1: Simple Approaches (Quick Tests)
1. **Strategy 7: CE50 Binning** - Easiest to implement with current infrastructure
2. **Strategy 1: Stratification** - Test if separate models per CE50 range help
3. **Strategy 2: Interactions** - Add CE50 × key descriptor interactions

### Phase 2: Moderate Complexity
4. **Strategy 3: Residual Correction** - Test CE50 as post-hoc correction
5. **Strategy 5: Ensemble** - Combine models with/without CE50

### Phase 3: Advanced Methods (If Phase 1-2 show promise)
6. **Strategy 4: Multi-task Learning** - Requires neural network implementation
7. **Strategy 8: Transfer Learning** - Most complex, needs larger CE50 dataset

### Phase 4: Operational Use
8. **Strategy 6: Applicability Domain** - For deployment and decision support

---

## Expected Outcomes

### Best Case Scenario
- **Strategy 1 or 7** (stratification/binning): Δ R² = +0.03-0.05
- Captures threshold effects seen with experimental CE50
- Improves predictions in favorable CE50 range

### Moderate Case
- **Strategy 2 or 3** (interactions/residuals): Δ R² = +0.01-0.03
- Captures subtle CE50-dependent effects
- Modest but statistically significant improvement

### Realistic Case
- **Most strategies**: Δ R² = 0.00-0.01
- Limited improvement because structural features already capture CE50 information
- Main value is in **interpretation** and **applicability domain** assessment

---

## Key Insight: Correlation vs Causation

The fact that **experimental CE50 ranking works** but **predicted CE50 features don't** suggests:

1. **Experimental CE50 captures something beyond structure**
   - Perhaps measurement artifacts that correlate with ADME
   - Specific fragmentation pathways that aren't well-modeled

2. **Predicted CE50 is redundant with structural features**
   - CE50 model uses same descriptors as PK model
   - Adding predicted CE50 is adding a derived, noisy version of existing information

3. **The value is in the ranking/threshold, not the continuous value**
   - "Favorable CE50" vs "unfavorable CE50" is useful
   - Exact CE50 value as a continuous predictor is not

**Recommendation:** Focus on strategies that use predicted CE50 as a **categorical/threshold variable** (Strategies 1, 7) rather than continuous linear features. This better mimics how experimental CE50 is successfully used for compound prioritization.

---

## Implementation Plan

```python
# Quick test: CE50-based stratification
def test_ce50_stratification():
    # Load data
    baseline_cv = pd.read_csv('Prediction_human_from_mordred_morgan_baseline.csv')
    ce50_preds = pd.read_csv('data/human_ce50_predictions.csv')

    # Merge and stratify by predicted CE50
    merged = baseline_cv.merge(ce50_preds[['smiles_r', 'ce50']], on='smiles_r')

    # Split into tertiles
    low_ce50 = merged[merged['ce50'] < merged['ce50'].quantile(0.33)]
    mid_ce50 = merged[(merged['ce50'] >= merged['ce50'].quantile(0.33)) &
                      (merged['ce50'] < merged['ce50'].quantile(0.67))]
    high_ce50 = merged[merged['ce50'] >= merged['ce50'].quantile(0.67)]

    # Compare R² in each stratum
    print(f"Low CE50 (favorable):  R² = {low_ce50['r2'].mean():.3f}")
    print(f"Mid CE50:              R² = {mid_ce50['r2'].mean():.3f}")
    print(f"High CE50 (unfavorable): R² = {high_ce50['r2'].mean():.3f}")

    # Test if stratified models would help
    return low_ce50, mid_ce50, high_ce50
```

---

**Next Steps:**
1. Test CE50 stratification (Strategy 1) - quick diagnostic
2. If stratified models show different R² by CE50 range → implement full stratified modeling
3. If no difference → CE50 truly has no predictive value beyond structure
4. Document findings and update recommendations
