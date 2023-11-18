import tensorflow as tf
import tensorflow.keras as krs
import numpy as np
import os
import json

def load_optimal_params(results_path, loss_key):
    optimal_loss = np.inf
    optimal_params = None
    for file in os.listdir(results_path):
        if os.path.splitext(file)[-1] != ".json":
            continue
        with open(os.path.join(results_path, file), "r") as results_file:
            results = json.load(results_file)
        
        loss = results["losses"][loss_key]
        if loss < optimal_loss:
            optimal_loss = loss
            optimal_params = results["params"]
    return optimal_params, optimal_loss

def save_params_and_results(params, history, json_path):
    if not isinstance(history, dict):
        history = history.history
    
    with open(json_path, "w") as json_file:
        json.dump({
            "params": params,
            "losses": {key: values[-1] for key, values in history.items() }
        }, json_file)

# Hyperparameter tuning
def tune_hyperparameter(
    X, 
    y, 
    create_model, 
    default_params, 
    param_key, 
    param_values, 
    loss_key, 
    results_path=None, 
    X_val=None, 
    y_val=None, 
    epochs=100, 
    compile_kwargs={"optimizer": "adam"}, 
    verbose=False
):
    losses = np.zeros_like(param_values, dtype="float")
    for i, value in enumerate(param_values):
        print(f"Training model for {param_key} = {value}")
        # Set params
        params = default_params.copy()
        params[param_key] = value
        # Train model
        instance = create_model(**params)
        instance.compile(**compile_kwargs)
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        earlystopping = krs.callbacks.EarlyStopping(monitor=loss_key, patience=3, min_delta=1e-5)
        history = instance.fit(
            X, 
            y, 
            epochs=epochs, 
            shuffle=True, 
            callbacks=[earlystopping], 
            validation_data=validation_data, 
            verbose=verbose
        )
        # Get results
        history = history.history
        loss = history[loss_key][-1]
        losses[i] = loss
        # Save data
        if results_path is not None:
            save_path = os.path.join(results_path, f"{param_key}_{value}.json")
            save_params_and_results(params, history, save_path)
    print(f"{losses = }")
    if np.all(losses == np.nan):
        print("WARNING: All losses are NAN")
        return default_params[param_key], np.inf
    i_opt = np.nanargmin(losses)
    loss_opt = losses[i_opt]
    value_opt = param_values[i_opt]
    print(f"Optimal {param_key} = {value_opt} -> {loss_opt}")
    return value_opt, loss_opt

def tune_hyperparameters(
    X, 
    y, 
    create_model, 
    default_params, 
    param_keys, 
    param_ranges, 
    loss_key, 
    results_path=None, 
    X_val=None, 
    y_val=None, 
    epochs=100, 
    compile_kwargs={"optimizer": "adam"}, 
    verbose=False,
    do_random=True,
    random_attempts=50,
):
    optimal_params = default_params.copy()
    optimal_loss = np.inf

    if do_random:
        for i in range(random_attempts + 1):
            # Pick parameters
            params = default_params.copy()
            if i > 0:
                # Make random choices
                print(f"Optimizing for random choice {i} / {random_attempts}")
                for key, values in zip(param_keys, param_ranges):
                    params[key] = values[np.random.choice(len(values))]
            else:
                print("Optimizing for default parameters")
            
            # Train
            instance = create_model(**params)
            instance.compile(**compile_kwargs)
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            earlystopping = krs.callbacks.EarlyStopping(monitor=loss_key, patience=3, min_delta=1e-5)
            history = instance.fit(
                X, 
                y, 
                epochs=epochs, 
                shuffle=True, 
                callbacks=[earlystopping], 
                validation_data=validation_data, 
                verbose=verbose
            )

            # Get results
            history = history.history
            loss = history.get(loss_key)[-1]
            print(f"{loss = }")

            # Save data
            if results_path is not None:
                save_path = os.path.join(results_path, f"random_{i}.json" if i > 0 else "default.json")
                save_params_and_results(params, history, save_path)
            
            # Update optimal 
            if loss < optimal_loss:
                optimal_params = params
                optimal_loss = loss       
    else:
        for param_key, param_values in zip(param_keys, param_ranges):
            value_opt, loss_opt = tune_hyperparameter(
                X, 
                y, 
                create_model, 
                default_params, 
                param_key, 
                param_values, 
                loss_key, 
                results_path=results_path, 
                X_val=X_val, 
                y_val=y_val, 
                epochs=epochs, 
                compile_kwargs=compile_kwargs, 
                verbose=verbose
            )
            if loss_opt < optimal_loss:
                optimal_params[param_key] = value_opt
                optimal_loss = loss_opt
    
    return optimal_params, optimal_loss
