#!/bin/bash
# Check CE50 Human Model Training Completion
# Run this script to check if training has completed

echo "========================================"
echo "CE50 Human Model Training Status Check"
echo "========================================"
echo "Current time: $(date)"
echo ""

# Check if process is still running
if ps aux | grep -q "[t]rain_human_models_ce50_simple.py"; then
    echo "Status: 🔄 TRAINING IN PROGRESS"
    echo ""
    ps aux | grep "[t]rain_human_models_ce50_simple.py" | awk '{print "  PID: "$2" | CPU: "$3"% | Memory: "$6/1024"MB | Runtime: "$10}'
    echo ""
    echo "Last log output:"
    tail -3 train_human_ce50_output.log | grep -v "UserWarning\|sklearn/utils"
    echo ""
    echo "Log file size: $(wc -l < train_human_ce50_output.log) lines"
else
    echo "Status: ✅ TRAINING COMPLETED (or stopped)"
    echo ""

    # Check for output files
    echo "Checking for generated files..."
    echo ""

    if [ -f "features_mfp_mordred_ce50_columns_human.txt" ]; then
        echo "✅ Feature list: features_mfp_mordred_ce50_columns_human.txt ($(wc -l < features_mfp_mordred_ce50_columns_human.txt) features)"
    else
        echo "❌ Feature list not found"
    fi

    echo ""
    echo "Model files:"
    find . -name "log_human_*_ce50_model_FINAL.sav" -exec ls -lh {} \; | awk '{print "  ✅ "$9" ("$5")"}'

    echo ""
    echo "Scaler files:"
    find . -name "human_ce50_*_scaler.pkl" -exec ls -lh {} \; | awk '{print "  ✅ "$9" ("$5")"}'

    echo ""
    if [ -f "Prediction_human_from_mordred_morgan_ce50.csv" ]; then
        echo "✅ Results: Prediction_human_from_mordred_morgan_ce50.csv ($(wc -l < Prediction_human_from_mordred_morgan_ce50.csv) rows)"
    fi

    if [ -f "human_ce50.csv" ]; then
        echo "✅ Summary: human_ce50.csv"
        echo ""
        echo "Performance Summary:"
        cat human_ce50.csv
    fi

    echo ""
    echo "Training log (last 20 lines):"
    tail -20 train_human_ce50_output.log | grep -v "UserWarning\|sklearn/utils"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
if ps aux | grep -q "[t]rain_human_models_ce50_simple.py"; then
    echo "  1. Wait for training to complete"
    echo "  2. Run this script again to check status"
    echo "  3. Once complete, run: python3 compare_ce50_enhancement.py"
else
    echo "  1. Verify all model files were created (13 files expected)"
    echo "  2. Run comparison: python3 compare_ce50_enhancement.py"
    echo "  3. Review visualizations and performance report"
fi
echo ""
