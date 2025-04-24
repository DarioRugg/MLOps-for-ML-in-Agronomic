import os
import time
import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
import clearml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from utils.clearml_reporting import log_distribution_plot, log_training_validation_loss, plot_predictions, report_model_performance_kfold, report_model_performance_test


def cal_correlation(pred, y_test, min_max_scaler):
    pred = pred.reshape((-1,1))
    target_pred = min_max_scaler.inverse_transform(pred)
    target_orig = min_max_scaler.inverse_transform(y_test)
    target_orig = target_orig[:,0]
    target_orig = pd.Series(target_orig)
    target_pred = target_pred[:,0]
    target_pred = pd.Series(target_pred)
    cor1 = target_orig.corr(target_pred, method='pearson')
    return cor1

def train_evaluate_mlp(X_train, y_train, X_test, y_test, hyper_parameters, seed, min_max_scaler):
    """
    Train and evaluate an MLPRegressor model.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - hyper_parameters: Dictionary with MLP hyperparameters.
    - task: clearml.Task object for logging.

    Returns:
    - The trained model and its test score.
    """
    mlp = MLPRegressor(max_iter=150, early_stopping=True, **hyper_parameters, random_state=seed)
    print("Training MLP...")
    strat_time = time.time()
    mlp.fit(X_train, y_train.ravel())
    print(f" - Done, training time: {time.time() - strat_time} seconds")
    
    # Evaluate the model
    test_score = mlp.score(X_test, y_test)
    
    return test_score, cal_correlation(mlp.predict(X_test), y_test, min_max_scaler), mlp.best_validation_score_, mlp

def multi_layer_perceptron_cv(X_train, y_train, X_test, y_test, hyper_parameters, task: clearml.Task, seed: int, min_max_scaler, n_splits=5):
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    cv_val_scores = []
    cv_val_corrs = []
    cv_inner_val_scores = []
    
    print("Performing k-fold cross-validation...")
    start_time = time.time()
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        val_fold_score, val_fold_corr, inner_val_score, mlp_model = train_evaluate_mlp(X_train_fold, y_train_fold, X_val_fold, y_val_fold, hyper_parameters, seed, min_max_scaler=min_max_scaler)
        
        cv_val_scores.append(val_fold_score)
        cv_val_corrs.append(val_fold_corr)
        cv_inner_val_scores.append(inner_val_score)
        
        print(f"Fold {fold}: Validation R-Square: {val_fold_score}")
        task.get_logger().report_scalar("KFold R-Square", f'Fold {fold+1}', val_fold_score, iteration=0)

        plot_predictions(y_val_fold, mlp_model.predict(X_val_fold), min_max_scaler, task, kfold=True, k=fold)
        
        log_training_validation_loss(mlp_model.loss_curve_, mlp_model.validation_scores_, task, kfold=True, k=fold)
    
    print(f" - Done, Time to perform k-fold CV: {time.time() - start_time} seconds")
    print(f"Average CV R-Square: {np.mean(cv_val_scores)}")
    
    task.get_logger().report_scalar("KFold R-Square", 'Average', np.mean(cv_val_scores), iteration=0)

    report_model_performance_kfold(cv_inner_val_scores, cv_val_scores, cv_val_corrs, n_splits, task)
    
def multi_layer_perceptron_testing(X_train, y_train, X_test, y_test, hyper_parameters, task: clearml.Task, seed: int, n_splits=5, min_max_scaler=None):
    print("Training and testing MLP on full dataset...")
    start_time = time.time()
    # After k-fold CV, retrain on the entire provided dataset
    test_score, test_corr, val_score, mlp_model = train_evaluate_mlp(X_train, y_train, 
                                                   X_test, y_test, 
                                                   hyper_parameters, seed, min_max_scaler=min_max_scaler)
    print(f" - Done, Time: {time.time() - start_time} seconds")
    
    print(f"Retrained on full dataset:\n - Test R-Square: {test_score}")
    print("    - Saving model...")
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(mlp_model, os.path.join("saved_models", "mlp_model_pretrained.pkl"), compress=True)
    
    plot_predictions(y_test, mlp_model.predict(X_test), min_max_scaler, task)
    
    log_training_validation_loss(mlp_model.loss_curve_, mlp_model.validation_scores_, task)
    
    return test_score, test_corr, val_score
    