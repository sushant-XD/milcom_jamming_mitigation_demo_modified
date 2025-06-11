# tune_hyperparams.py
import optuna
import json
from model import run_training_and_evaluation

def objective(trial):
    params = {
        'data_dir': "srsran_csv_output",
        'sequence_length': trial.suggest_int('sequence_length', 5, 25),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs': 300, 
        'early_stopping_patience': 15,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'latent_size': trial.suggest_int('latent_size', 8, 64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'threshold_percentile': trial.suggest_int('threshold_percentile', 90, 99)
    }
    
    f1_score = run_training_and_evaluation(params, save_final_model=False, trial_number=trial.number)
    
    return f1_score

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=50) 
    
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial achieved F1-score: {best_trial.value:.4f}")
    
    print("Best parameters found:")
    best_params = best_trial.params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        
    with open("best_params_from_tuning.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("\nBest parameters saved to 'best_params_from_tuning.json'")
    print("You can now use these parameters to train the final model by updating the 'default_params' in your main script and running it.")