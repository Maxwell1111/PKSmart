"""
Generate CE50 Predictions for Human Dataset
Converted from notebook for command-line execution
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CE50 PREDICTION FOR HUMAN DATASET")
print("="*80)
print(f"\nWorking directory: {os.getcwd()}")
print("Starting execution...\n")

# 1. Load Human PK Data
print("STEP 1: Loading human PK data...")
df_human = pd.read_csv('data/Human_PK_data.csv')
print(f"✓ Loaded {len(df_human)} rows from Human_PK_data.csv")

# Filter to get human compounds
human_data = df_human[
    df_human['human_VDss_L_kg'].notna() |
    df_human['human_CL_mL_min_kg'].notna() |
    df_human['human_fup'].notna() |
    df_human['human_mrt'].notna() |
    df_human['human_thalf'].notna()
].copy()

print(f"✓ Found {len(human_data)} compounds with human PK data")
print(f"  - human_VDss_L_kg: {human_data['human_VDss_L_kg'].notna().sum()} compounds")
print(f"  - human_CL_mL_min_kg: {human_data['human_CL_mL_min_kg'].notna().sum()} compounds")
print(f"  - human_fup: {human_data['human_fup'].notna().sum()} compounds")
print(f"  - human_mrt: {human_data['human_mrt'].notna().sum()} compounds")
print(f"  - human_thalf: {human_data['human_thalf'].notna().sum()} compounds")

# 2. Load Pre-trained CE50 Models
print("\nSTEP 2: Loading CE50 ensemble models...")
model_dir = 'CE50_prediction/models/'

# Find available model files
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'applicability' not in f and 'metadata' not in f]
if len(model_files) == 0:
    raise ValueError(f"No model files found in {model_dir}")

# Extract timestamps
timestamps = []
for f in model_files:
    parts = f.replace('.pkl', '').split('_')
    if len(parts) >= 2:
        timestamp = '_'.join(parts[-2:])  # Get last two parts (e.g., "20260105_111730")
        timestamps.append(timestamp)

timestamps = list(set(timestamps))
timestamp = sorted(timestamps)[-1]  # Most recent
print(f"✓ Found {len(timestamps)} model timestamp(s), using: {timestamp}")

# Load all 4 ensemble models
models = {}
model_types = ['rf_binary', 'rf_count', 'xgb_binary', 'xgb_count']

for model_type in model_types:
    model_path = f"{model_dir}{model_type}_{timestamp}.pkl"
    if os.path.exists(model_path):
        try:
            models[model_type] = joblib.load(model_path)
            print(f"✓ Loaded {model_type} from {os.path.basename(model_path)}")
        except Exception as e:
            print(f"✗ Failed to load {model_type}: {e}")
    else:
        print(f"✗ Model file not found: {model_path}")

if len(models) == 0:
    raise ValueError("No models could be loaded! Check CE50_prediction/models/ directory")

print(f"✓ Total models loaded: {len(models)}")

# 3. Generate Dual Fingerprints
print("\nSTEP 3: Generating molecular fingerprints...")

def generate_binary_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def generate_count_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int32)
    for idx, val in fp.GetNonzeroElements().items():
        arr[idx] = val
    return arr

binary_fps = []
count_fps = []
valid_indices = []
invalid_smiles = []

for idx, smiles in enumerate(human_data['smiles_r'].values):
    if pd.isna(smiles):
        invalid_smiles.append((idx, smiles, "Missing SMILES"))
        continue

    binary_fp = generate_binary_fingerprint(smiles)
    count_fp = generate_count_fingerprint(smiles)

    if binary_fp is not None and count_fp is not None:
        binary_fps.append(binary_fp)
        count_fps.append(count_fp)
        valid_indices.append(idx)
    else:
        invalid_smiles.append((idx, smiles, "Invalid SMILES"))

X_binary = np.array(binary_fps)
X_count = np.array(count_fps)

print(f"✓ Generated fingerprints for {len(valid_indices)} compounds")
print(f"✗ Failed for {len(invalid_smiles)} compounds")
print(f"  Binary fingerprint shape: {X_binary.shape}")
print(f"  Count fingerprint shape: {X_count.shape}")

# 4. Make CE50 Predictions
print("\nSTEP 4: Making CE50 predictions with ensemble...")
predictions = {}

if 'rf_binary' in models:
    predictions['rf_binary'] = models['rf_binary'].predict(X_binary)
    print(f"  ✓ RF Binary: generated {len(predictions['rf_binary'])} predictions")
if 'rf_count' in models:
    predictions['rf_count'] = models['rf_count'].predict(X_count)
    print(f"  ✓ RF Count: generated {len(predictions['rf_count'])} predictions")
if 'xgb_binary' in models:
    predictions['xgb_binary'] = models['xgb_binary'].predict(X_binary)
    print(f"  ✓ XGB Binary: generated {len(predictions['xgb_binary'])} predictions")
if 'xgb_count' in models:
    predictions['xgb_count'] = models['xgb_count'].predict(X_count)
    print(f"  ✓ XGB Count: generated {len(predictions['xgb_count'])} predictions")

if len(predictions) == 0:
    raise ValueError("No predictions could be generated!")

print(f"✓ Generated predictions from {len(predictions)} models")

# Calculate ensemble statistics
all_preds = np.array(list(predictions.values())).T  # Shape: (n_compounds, n_models)
if all_preds.ndim == 1:
    all_preds = all_preds.reshape(-1, 1)

pce50_ensemble_mean = all_preds.mean(axis=1)
pce50_ensemble_std = all_preds.std(axis=1) if all_preds.shape[1] > 1 else np.zeros(len(all_preds))
pce50_predicted = pce50_ensemble_mean

print(f"  Mean pCE50: {pce50_predicted.mean():.3f} ± {pce50_predicted.std():.3f}")
print(f"  Range: [{pce50_predicted.min():.3f}, {pce50_predicted.max():.3f}]")
print(f"  Ensemble disagreement (std): {pce50_ensemble_std.mean():.3f}")

# 5. Calculate Confidence Scores
print("\nSTEP 5: Calculating confidence scores...")
std_normalized = (pce50_ensemble_std - pce50_ensemble_std.min()) / \
                 (pce50_ensemble_std.max() - pce50_ensemble_std.min() + 1e-10)
confidence_scores = 6 * (1 - std_normalized)
confidence_categories = np.where(confidence_scores >= 5, 'High',
                        np.where(confidence_scores >= 3, 'Medium', 'Low'))

print(f"✓ Confidence scores calculated (0-6 scale)")
print(f"  Mean: {confidence_scores.mean():.2f}")
for cat in ['High', 'Medium', 'Low']:
    count = (confidence_categories == cat).sum()
    pct = 100 * count / len(confidence_categories)
    print(f"  {cat}: {count} ({pct:.1f}%)")

# 6. Convert pCE50 to CE50
print("\nSTEP 6: Converting pCE50 to CE50 (eV)...")
ce50_predicted = 10 ** (-pce50_predicted)
print(f"✓ CE50 predictions:")
print(f"  Mean: {ce50_predicted.mean():.2f} ± {ce50_predicted.std():.2f} eV")
print(f"  Median: {np.median(ce50_predicted):.2f} eV")
print(f"  Range: [{ce50_predicted.min():.2f}, {ce50_predicted.max():.2f}] eV")

in_expected_range = ((ce50_predicted >= 10) & (ce50_predicted <= 50)).sum()
print(f"  In expected range (10-50 eV): {in_expected_range} / {len(ce50_predicted)} ({100*in_expected_range/len(ce50_predicted):.1f}%)")

# 7. Create Results DataFrame
print("\nSTEP 7: Creating results DataFrame...")
human_data_valid = human_data.iloc[valid_indices].copy()
human_data_valid['ce50'] = ce50_predicted
human_data_valid['pce50'] = pce50_predicted
human_data_valid['confidence'] = confidence_scores
human_data_valid['confidence_category'] = confidence_categories
human_data_valid['ensemble_std'] = pce50_ensemble_std

for model_name, preds in predictions.items():
    human_data_valid[f'{model_name}_pce50'] = preds

print(f"✓ Results DataFrame created with {len(human_data_valid)} compounds")

# 8. Save Results
print("\nSTEP 8: Saving results...")
output_file = 'data/human_ce50_predictions.csv'
human_data_valid.to_csv(output_file, index=False)
print(f"✓ Saved full results to: {output_file}")

essential_cols = ['smiles_r', 'human_VDss_L_kg', 'human_CL_mL_min_kg',
                  'human_fup', 'human_mrt', 'human_thalf',
                  'ce50', 'pce50', 'confidence', 'confidence_category', 'ensemble_std']

# Only include NAME column if it exists
if 'NAME' in human_data_valid.columns:
    essential_cols.insert(1, 'NAME')

human_ce50_simple = human_data_valid[essential_cols].copy()
human_ce50_simple.to_csv('data/human_ce50_predictions_simple.csv', index=False)
print(f"✓ Saved simplified version to: data/human_ce50_predictions_simple.csv")

# 9. Create Visualizations
print("\nSTEP 9: Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('CE50 Predictions for Human Dataset', fontsize=16, fontweight='bold')

# 1. CE50 distribution
ax = axes[0, 0]
ax.hist(ce50_predicted, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(ce50_predicted.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ce50_predicted.mean():.2f} eV')
ax.axvline(np.median(ce50_predicted), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(ce50_predicted):.2f} eV')
ax.set_xlabel('CE50 (eV)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('CE50 Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 2. pCE50 distribution
ax = axes[0, 1]
ax.hist(pce50_predicted, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax.axvline(pce50_predicted.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pce50_predicted.mean():.3f}')
ax.set_xlabel('pCE50 (-log10[CE50])', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('pCE50 Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 3. Confidence distribution
ax = axes[0, 2]
confidence_counts = pd.Series(confidence_categories).value_counts()
colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
bars = ax.bar(confidence_counts.index, confidence_counts.values,
              color=[colors.get(x, 'gray') for x in confidence_counts.index], edgecolor='black', alpha=0.7)
ax.set_xlabel('Confidence Category', fontweight='bold')
ax.set_ylabel('Number of Compounds', fontweight='bold')
ax.set_title('Confidence Distribution')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({100*height/len(confidence_categories):.1f}%)',
            ha='center', va='bottom', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 4. Ensemble disagreement
ax = axes[1, 0]
ax.hist(pce50_ensemble_std, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(pce50_ensemble_std.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {pce50_ensemble_std.mean():.3f}')
ax.set_xlabel('Ensemble Std (pCE50 units)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Ensemble Model Agreement')
ax.legend()
ax.grid(alpha=0.3)

# 5. Model comparison
ax = axes[1, 1]
model_data = [predictions[m] for m in predictions.keys()]
parts = ax.violinplot(model_data, positions=range(len(predictions)),
                      showmeans=True, showmedians=True)
ax.set_xticks(range(len(predictions)))
ax.set_xticklabels(list(predictions.keys()), rotation=45, ha='right')
ax.set_ylabel('pCE50 (-log10[CE50])', fontweight='bold')
ax.set_title('Model Predictions Comparison')
ax.grid(alpha=0.3, axis='y')

# 6. Confidence vs Ensemble Std
ax = axes[1, 2]
scatter = ax.scatter(pce50_ensemble_std, confidence_scores,
                    c=confidence_scores, cmap='RdYlGn', s=50, alpha=0.6, edgecolor='black')
ax.set_xlabel('Ensemble Std (pCE50)', fontweight='bold')
ax.set_ylabel('Confidence Score (0-6)', fontweight='bold')
ax.set_title('Confidence vs Ensemble Agreement')
plt.colorbar(scatter, ax=ax, label='Confidence')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('human_ce50_predictions_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: human_ce50_predictions_diagnostics.png")
plt.close()

# 10. Summary Statistics
print("\n" + "="*80)
print("CE50 PREDICTION SUMMARY FOR HUMAN DATASET")
print("="*80)
print(f"\n📊 DATASET STATISTICS:")
print(f"  Total compounds in Human_PK_data.csv: {len(df_human)}")
print(f"  Compounds with human PK data: {len(human_data)}")
print(f"  Successful CE50 predictions: {len(human_data_valid)}")
print(f"  Failed predictions: {len(invalid_smiles)}")

print(f"\n🎯 CE50 PREDICTIONS:")
print(f"  Mean CE50: {ce50_predicted.mean():.2f} ± {ce50_predicted.std():.2f} eV")
print(f"  Median CE50: {np.median(ce50_predicted):.2f} eV")
print(f"  Range: [{ce50_predicted.min():.2f}, {ce50_predicted.max():.2f}] eV")

print(f"\n📈 pCE50 PREDICTIONS:")
print(f"  Mean pCE50: {pce50_predicted.mean():.3f} ± {pce50_predicted.std():.3f}")
print(f"  Range: [{pce50_predicted.min():.3f}, {pce50_predicted.max():.3f}]")

print(f"\n🎓 CONFIDENCE ASSESSMENT:")
for cat in ['High', 'Medium', 'Low']:
    count = (confidence_categories == cat).sum()
    pct = 100 * count / len(confidence_categories)
    avg_conf = confidence_scores[confidence_categories == cat].mean()
    print(f"  {cat:8s}: {count:3d} compounds ({pct:5.1f}%) - Avg score: {avg_conf:.2f}")

print(f"\n💾 OUTPUT FILES GENERATED:")
print(f"  1. data/human_ce50_predictions.csv (full dataset)")
print(f"  2. data/human_ce50_predictions_simple.csv (essential columns)")
print(f"  3. human_ce50_predictions_diagnostics.png (visualizations)")

print(f"\n✅ CE50 prediction for human dataset COMPLETE!")
print("="*80)
