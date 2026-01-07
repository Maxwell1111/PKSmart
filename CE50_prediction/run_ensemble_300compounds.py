"""
Test CE50 Ensemble on 300 Compounds Dataset
"""
import sys
sys.path.append('.')

# Import the ensemble system
from ce50_ensemble_predictor import *

def main():
    """Run ensemble on 300 compounds dataset"""
    print("\n" + "="*80)
    print("CE50 ENSEMBLE - 300 COMPOUNDS DATASET TEST")
    print("="*80)

    # Load data
    df = load_and_prepare_data('ce50_300compounds_training.csv')

    # Convert to pCE50
    df['pce50'] = -np.log10(df['ce50'])

    print(f"\nData Statistics:")
    print(f"  Compounds: {len(df)}")
    print(f"  CE50 range: {df['ce50'].min():.2f} - {df['ce50'].max():.2f} μM")
    print(f"  pCE50 range: {df['pce50'].min():.4f} - {df['pce50'].max():.4f}")
    print(f"  CE50 mean: {df['ce50'].mean():.2f} μM")
    print(f"  CE50 std: {df['ce50'].std():.2f} μM")

    # Generate dual fingerprints
    print("\nGenerating dual fingerprints...")
    fp_generator = DualFingerprintGenerator()
    fps = fp_generator.generate_both(df['smiles'].values)

    # Filter valid molecules
    df_valid = df[fps['valid_mask']].copy()
    X_binary = fps['binary']
    X_count = fps['count']
    y = df_valid['pce50'].values

    print(f"Valid molecules: {len(df_valid)} / {len(df)}")
    print(f"Binary fingerprints shape: {X_binary.shape}")
    print(f"Count fingerprints shape: {X_count.shape}")

    # Create and train ensemble
    print("\nTraining ensemble (this may take 2-3 minutes)...")
    ensemble = EnsembleModel()
    models, scores = ensemble.train_ensemble(X_binary, X_count, y)

    # Evaluate ensemble
    results = ensemble.evaluate_ensemble()

    # Generate predictions for test set
    test_smiles = df_valid.iloc[len(y)-len(ensemble.y_test):]['smiles'].values
    predictions = ensemble.predict_with_confidence(test_smiles)

    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    viz = Visualizer()
    viz.plot_ensemble_comparison(ensemble.y_test, results, output_file='ensemble_300compounds_comparison.png')
    viz.plot_confidence_distribution(predictions, output_file='confidence_300compounds_distribution.png')
    viz.plot_model_selection(predictions, output_file='model_300compounds_selection.png')

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    ensemble.save_models(output_dir='models_300compounds')

    # Save predictions to CSV
    predictions_df = pd.DataFrame([{
        'compound_name': df_valid.iloc[len(y)-len(ensemble.y_test)+i]['Compound_Name'] if 'Compound_Name' in df_valid.columns else f'Compound_{i}',
        'smiles': p['smiles'],
        'actual_ce50': df_valid.iloc[len(y)-len(ensemble.y_test)+i]['ce50'],
        'actual_pce50': df_valid.iloc[len(y)-len(ensemble.y_test)+i]['pce50'],
        'predicted_ce50': p['predicted_ce50'],
        'predicted_pce50': p['predicted_pce50'],
        'confidence': p['confidence'],
        'selected_model': p['selected_model'],
        'ensemble_std': p['ensemble_std'],
        'tanimoto_binary': p['applicability_scores']['tanimoto_binary'],
        'tanimoto_count': p['applicability_scores']['tanimoto_count']
    } for i, p in enumerate(predictions)])

    predictions_df.to_csv('ensemble_300compounds_predictions.csv', index=False)
    print("\nPredictions saved to ensemble_300compounds_predictions.csv")

    # Print detailed summary
    print("\n" + "="*80)
    print("ENSEMBLE SUMMARY - 300 COMPOUNDS")
    print("="*80)
    
    print(f"\nBest performing model (by R²):")
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"  {best_model[0]}: R² = {best_model[1]['r2']:.4f}")
    print(f"  MAE = {best_model[1]['mae']:.4f}")
    print(f"  RMSE = {best_model[1]['rmse']:.4f}")

    print(f"\nAll model performance:")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"  {model_name:15s}: R² = {metrics['r2']:7.4f}, MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")

    print(f"\nConfidence distribution ({len(predictions)} test predictions):")
    for conf_level in ['High', 'Medium', 'Low']:
        count = sum(1 for p in predictions if p['confidence'] == conf_level)
        pct = 100 * count / len(predictions) if len(predictions) > 0 else 0
        print(f"  {conf_level:6s}: {count:3d} ({pct:.1f}%)")

    print(f"\nModel selection frequency:")
    model_counts = {}
    for p in predictions:
        model_name = p['selected_model']
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(predictions)
        print(f"  {model_name:15s}: {count:3d} ({pct:.1f}%)")

    # Show sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"{'Compound':<15s} {'Actual':>8s} {'Predicted':>10s} {'Error':>8s} {'Confidence':>10s} {'Model':>15s}")
    print("-" * 80)
    for i in range(min(10, len(predictions_df))):
        row = predictions_df.iloc[i]
        error = abs(row['predicted_pce50'] - row['actual_pce50'])
        print(f"{row['compound_name']:<15s} {row['actual_pce50']:8.4f} {row['predicted_pce50']:10.4f} {error:8.4f} {row['confidence']:>10s} {row['selected_model']:>15s}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")

    return ensemble, predictions, results

if __name__ == "__main__":
    ensemble, predictions, results = main()
