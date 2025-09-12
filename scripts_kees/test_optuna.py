import multiprocessing
import optuna
import time
import random
from pathlib import Path

# Dummy simulation function to mimic Arctic-XBeach
def dummy_simulation(params):
    # Simulate some computation time
    start = time.time()
    result = 0
    for _ in range(10_000_000):  # Perform a computationally intensive task
        result += random.random() * random.random()
    time.sleep(max(0, 10 - (time.time() - start)))  # Ensure it takes ~10 seconds
    return sum(params.values()) + result  # Return a dummy result

# Optuna objective function
def objective(trial):
    params = {
        "param1": trial.suggest_float("param1", 0.1, 10.0),
        "param2": trial.suggest_float("param2", 0.1, 10.0),
        "param3": trial.suggest_float("param3", 0.1, 10.0),
    }
    result = dummy_simulation(params)
    return result

# Multiprocessing test with Optuna
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Ensure Windows compatibility

    # Create a directory for the Optuna study database
    study_dir = Path("d:/Git/thermo-morphological-model/optuna_test")
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{study_dir / 'optuna_study.db'}"

    # Create or load the Optuna study
    study = optuna.create_study(
        study_name="test_study",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
    )

    # Run the optimization
    n_trials = 20
    n_parallel = 36
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel)
    except Exception as e:
        print(f"Error during optimization: {e}")

    # Print the best trial
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
