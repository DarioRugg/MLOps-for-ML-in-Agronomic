import numpy as np
import seaborn as sns
import clearml
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def log_training_validation_loss(loss_curve, validation_scores, task, kfold=False, k=None):
    """
    Plot the training and Validation Score curves on twin Y axes due to different scales
    (e.g., MSE for training loss and R-Square for Validation Score).

    Parameters:
    - loss_curve: Numpy array of training loss values (MSE).
    - validation_scores: Numpy array of validation scores (R-Square).
    - task: clearml.Task object for logging.
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on the primary y-axis
    color = 'tab:red' if not kfold else 'tab:orange'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss (MSE)', color=color)
    ax1.plot(loss_curve, label='Training Loss (MSE)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a twin Axes sharing the x-axis for Validation Score
    ax2 = ax1.twinx()
    color = 'tab:blue' if not kfold else 'tab:green'
    ax2.set_ylabel('Validation Score (R-Square)', color=color)  # we already handled the x-label with ax1
    ax2.plot(validation_scores, label='Validation Score (R-Square)', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    plt.title('Training and Validation Score Curves')
    fig.tight_layout()  # To ensure there's no overlap in layout

    # Adding legends manually
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Log the plot to ClearML
    if not kfold:
        task.get_logger().report_matplotlib_figure("Learning curves", series="Learning curves", figure=fig)
    else:    
        task.get_logger().report_matplotlib_figure(f"KFold - Learning curves", series="KFold - Learning curves", figure=fig, iteration=k)
    plt.close(fig)
    
def log_distribution_plot(cv_scores, test_score, cv_corrs, test_corr, cv_best_vals, best_val, task: clearml.Task):
    """
    Logs a figure with three density plots for CV metrics, each with a red dot for the test metric.

    Parameters:
    - cv_scores: List of cross-validation scores.
    - test_score: The test set score.
    - cv_corrs: List of cross-validation correlations.
    - test_corr: The test set correlation.
    - cv_best_vals: List of cross-validation best inner validaation metric.
    - best_val: The best value on the test set.
    - task: clearml.Task object for logging.
    """
    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Plot density for CV scores
    # sns.kdeplot(cv_scores, ax=axes[0], shade=True, color="blue")
    sns.histplot(cv_scores, ax=axes[0], color="blue", alpha=0.5)
    axes[0].axvline(x=test_score, color="red", linestyle="--", linewidth=2, label='Test Score')
    axes[0].set_title('CV Scores Distribution')
    axes[0].legend()

    # Plot density for CV correlations
    # sns.kdeplot(cv_corrs, ax=axes[1], shade=True, color="green")
    sns.histplot(cv_corrs, ax=axes[1], color="green", alpha=0.5)
    axes[1].axvline(x=test_corr, color="red", linestyle="--", linewidth=2, label='Test Correlation')
    axes[1].set_title('CV Correlations Distribution')
    axes[1].legend()

    # Plot density for CV best values
    # sns.kdeplot(cv_best_vals, ax=axes[2], shade=True, color="orange")
    sns.histplot(cv_best_vals, ax=axes[2], color="orange", alpha=0.5)
    axes[2].axvline(x=best_val, color="red", linestyle="--", linewidth=2, label='Test Validation')
    axes[2].set_title('CV inner Validation Distribution')
    axes[2].legend()

    # Tight layout to use space efficiently
    plt.tight_layout()

    # Assuming `task.get_logger().report_matplotlib_figure` method exists and works similar to logging figures in ClearML
    # You would need to replace or adjust this line according to your actual logging implementation
    task.get_logger().report_matplotlib_figure(title="CV and Test Metrics Distribution", series="CV and Test Metrics Distribution", figure=fig)
    
    plt.close(fig)  # Close the plt object to prevent it from displaying in non-interactive environments

def inverse_scale(y_true, y_pred, min_max_scaler):
    y_pred = y_pred.reshape((-1,1))
    y_true = y_true.reshape((-1,1))
    
    target_pred = min_max_scaler.inverse_transform(y_pred)
    target_orig = min_max_scaler.inverse_transform(y_true)
    target_orig = target_orig[:,0]
    target_pred = target_pred[:,0]
    return target_orig, target_pred

def plot_predictions(y_true, y_pred, min_max_scaler, task: clearml.Task, kfold=False, k=None):
    y_true = np.squeeze(y_true)
    
    # back to the original scale
    y_true, y_pred = inverse_scale(y_true, y_pred, min_max_scaler)
    
    plot_bars(y_true, y_pred, task, kfold, k)
    plot_scatters(y_true, y_pred, task, kfold, k)

def plot_bars(y_true, y_pred, task: clearml.Task, kfold=False, k=None):
    # Create DataFrames for true and predicted values
    df_true = pd.DataFrame({'Sample Index': range(len(y_true)), 'Value': y_true, 'Type': 'True'}).head(60)
    df_pred = pd.DataFrame({'Sample Index': range(len(y_pred)), 'Value': y_pred, 'Type': 'Predicted'}).head(60)
    
    # Concatenate DataFrames
    df = pd.concat([df_true, df_pred], ignore_index=True)
    
    # Create a new figure for the bar plot
    plt.figure(figsize=(25, 8))
    
    if not kfold:
        colors = ["tab:blue", "tab:red"]
    else:
        colors = ["tab:blue", "tab:orange"]
    
    # Bar plot for the first 60 samples
    sns.barplot(data=df, x="Sample Index", y="Value", hue="Type", palette=colors)
    
    if not kfold:
        plt.title('True vs. Predicted Values for First 60 Samples\nTest Set')
    else:
        plt.title(f'True vs. Predicted Values for First 60 Samples\nValidation Fold {k+1}')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Log bar plot to ClearML
    if not kfold:
        task.get_logger().report_matplotlib_figure("True_vs_Predicted", "True_vs_Predicted_test", figure=plt.gcf(), report_image=True)
    else:
        task.get_logger().report_matplotlib_figure("True_vs_Predicted_cv", "True_vs_Predicted_cv", figure=plt.gcf(), iteration=k, report_image=True)
    plt.close()
    
def plot_scatters(y_true, y_pred, task: clearml.Task, kfold=False, k=None):
    
    # Create a new figure for the bar plot
    plt.figure(figsize=(7, 8))
    
    sns.regplot(x = y_pred, y = y_true, label="Samples", scatter_kws = {"color": "tab:red" if not kfold else "tab:orange", "alpha": 0.8}, ci=None)
    
    if not kfold:
        plt.title('True vs. Predicted Scatterplot\nTest Set')
    else:
        plt.title(f'True vs. Predicted Scatterplot\nValidation Fold {k+1}')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.legend()
    plt.tight_layout()
    
    # Log bar plot to ClearML
    if not kfold:
        task.get_logger().report_matplotlib_figure("True_vs_Predicted_scatterplot", "True_vs_Predicted_scatterplot_test", figure=plt.gcf(), report_image=True)
    else:
        task.get_logger().report_matplotlib_figure("True_vs_Predicted_scatterplot_cv", "True_vs_Predicted_scatterplot_cv", figure=plt.gcf(), iteration=k, report_image=True)
    plt.close()

def report_model_performance_test(val_best_score, test_score, test_correlation, cv_results_df, task: clearml.Task):
    """
    Reports the best training loss, best validation score, and Test Score of an MLP model.
    """
    metric_names = ["Best Validation Score", "Test Score", "Test Correlation (Authors' Accuracy)"]
    metrics_data = {
        "Metric": metric_names,
        "Test Splits": [val_best_score, test_score, test_correlation],
        "Cross-Validation Folds (average and STD)": ["{:.3f} ({:.4f})".format(mean, std) for mean, std in zip(cv_results_df[metric_names].mean().values, cv_results_df[metric_names].std().values)]
    }
    df = pd.DataFrame(metrics_data)
    
    task.get_logger().report_table(
        "Model Performance Metrics", 
        "Model Performance Metrics", 
        iteration=0, 
        table_plot=df
    )
    
    task.upload_artifact("results", df.drop(columns="Cross-Validation Folds (average and STD)"))
    
def report_model_performance_kfold(val_best_score_list, test_score_list, test_correlation_list, n_folds, task: clearml.Task):
    """
    Reports the best training loss, best validation score, and Test Score of an MLP model.
    """
    df = pd.DataFrame({
        "KFold iteration": list(range(1, n_folds+1)),
        "Best Validation Score": val_best_score_list,
        "Test Score": test_score_list,
        "Test Correlation (Authors' Accuracy)": test_correlation_list
    })
    
    task.upload_artifact("results", df)
    
    # Calculate means and standard deviations, ignoring the first column ("KFold iteration")
    means = df.iloc[:, 1:].mean()
    stds = df.iloc[:, 1:].std()

    # Format means and standard deviations into strings in the format "mean (std)"
    mean_std_rows = ["{:.3f} ({:.4f})".format(mean, std) for mean, std in zip(means, stds)]

    # Create a new row with "Average" for the first column and the formatted mean/std strings for the others
    new_row = pd.Series(["Average (STD)"] + mean_std_rows, index=df.columns)

    # Append the new row to the DataFrame
    df = df.append(new_row, ignore_index=True)
    
    task.get_logger().report_table(
        "Model Performance Metrics - Kfold", 
        "Model Performance Metrics - Kfold", 
        iteration=0, 
        table_plot=df.set_index("KFold iteration")
    )