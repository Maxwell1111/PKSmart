# Quick Start: Training CE50-Enhanced Human PK Models

## 🚀 Quick Start (If Prerequisites Already Met)

```bash
cd /Users/aardeshiri/PKSmart
python train_human_models_with_ce50.py
```

## 📋 Step-by-Step Setup (First Time)

### Step 1: Check Required Data Files
```bash
# Should exist in data/ directory
ls -lh data/Human_PK_data.csv
ls -lh data/human_ce50_predictions_simple.csv
```

### Step 2: Train Animal Models (if not already done)

#### 2a. Train baseline dog and monkey models
```bash
# These scripts should exist from baseline training
python train_dog_models.py      # Creates 3 dog models
python train_monkey_models.py   # Creates 3 monkey models
```

#### 2b. Train CE50-enhanced rat models
```bash
python train_rat_models_with_ce50.py   # Creates 3 rat models with CE50
```

### Step 3: Train Human Models
```bash
python train_human_models_with_ce50.py
```

**Runtime**: ~10-20 hours (5 endpoints × 25 CV folds each)

## ✅ Prerequisites Checklist

Before running `train_human_models_with_ce50.py`, ensure these files exist:

### Data Files (2)
- [ ] `data/Human_PK_data.csv`
- [ ] `data/human_ce50_predictions_simple.csv`

### Dog Models (4)
- [ ] `log_dog_VDss_L_kg_model_FINAL.sav`
- [ ] `log_dog_CL_mL_min_kg_model_FINAL.sav`
- [ ] `log_dog_fup_model_FINAL.sav`
- [ ] `scaler_dog.pkl`
- [ ] `features_mfp_mordred_columns_dog_model.txt`

### Monkey Models (4)
- [ ] `log_monkey_VDss_L_kg_model_FINAL.sav`
- [ ] `log_monkey_CL_mL_min_kg_model_FINAL.sav`
- [ ] `log_monkey_fup_model_FINAL.sav`
- [ ] `scaler_monkey.pkl`
- [ ] `features_mfp_mordred_columns_monkey_model.txt`

### Rat Models - CE50 Enhanced (4)
- [ ] `log_rat_VDss_L_kg_model_ce50_FINAL.sav`
- [ ] `log_rat_CL_mL_min_kg_model_ce50_FINAL.sav`
- [ ] `log_rat_fup_model_ce50_FINAL.sav`
- [ ] `scaler_rat_ce50.pkl`
- [ ] `features_mfp_mordred_ce50_columns_rat_model.txt`

## 📊 Expected Output

After successful completion, you'll have:

### Models (5)
- `log_human_VDss_L_kg_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_CL_mL_min_kg_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_fup_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_mrt_withanimaldata_artificial_ce50_model_FINAL.sav`
- `log_human_thalf_withanimaldata_artificial_ce50_model_FINAL.sav`

### Scalers (5)
- `artificial_animal_data_mfp_mrd_ce50_human_VDss_L_kg_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_CL_mL_min_kg_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_fup_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_mrt_scaler.pkl`
- `artificial_animal_data_mfp_mrd_ce50_human_thalf_scaler.pkl`

### Results (4)
- `features_mfp_mordred_animal_ce50_columns_human.txt` (516 features)
- `Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv` (detailed results)
- `human_with_mfp_mordred_animal_artificial_ce50.csv` (summary statistics)
- `model_parameters_Prediction_human_from_mordred_morgan_fs_animal_artificial_ce50.csv`

## 🔍 Quick Verification

After training, check the summary statistics:

```bash
# View summary performance
cat human_with_mfp_mordred_animal_artificial_ce50.csv

# Expected GMFE values (approximate):
# - human_VDss_L_kg: ~2.1
# - human_CL_mL_min_kg: ~2.5
# - human_fup: ~2.7
# - human_mrt: ~2.5
# - human_thalf: ~2.4
```

## ⚠️ Common Issues

### Issue 1: "Model file not found"
**Fix**: Train the missing animal models first (see Step 2).

### Issue 2: "CE50 data file not found"
**Fix**: Run `generate_ce50_human_predictions.py` first.

### Issue 3: Script runs but produces poor results
**Check**:
- Are prerequisite models trained correctly?
- Is CE50 data complete and accurate?
- Check data quality in `data/Human_PK_data.csv`

## 💡 Tips

1. **Monitor progress**: The script prints detailed progress for each step
2. **Parallel processing**: Uses 30 CPU cores by default (adjust `n_jobs` if needed)
3. **Checkpointing**: Best models are saved after each fold
4. **Resume capability**: If interrupted, you can restart (though CV folds will re-run)

## 📖 More Information

- **Detailed documentation**: See `TRAIN_HUMAN_CE50_README.md`
- **Comparison to baseline**: See `HUMAN_MODEL_COMPARISON.md`
- **CE50 integration guide**: See `CE50_INTEGRATION_GUIDE.md`

## 🎯 Quick Comparison Command

Compare baseline vs CE50-enhanced performance:

```bash
# If you have both baseline and CE50 results
echo "=== BASELINE ==="
tail -6 human_with_mfp_mordred_animal_artificial.csv

echo "=== CE50-ENHANCED ==="
tail -6 human_with_mfp_mordred_animal_artificial_ce50.csv
```

---
**Script**: `train_human_models_with_ce50.py`
**Author**: Generated with Claude Code
**Date**: 2026-01-07
