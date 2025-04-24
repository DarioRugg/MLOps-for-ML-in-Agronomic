from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import clearml 
import json

sns.set_theme(style="whitegrid")

def main():
    task: clearml.Task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml/drafts", 
                                           task_name="aggregator task",
                                           task_type="monitor",
                                           reuse_last_task_id=False)
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image="rugg/dlwheatwithclearml:clearml",
        docker_arguments="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
            --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
            --env CLEARML_AGENT_GIT_USER={bot_name} \
            --env CLEARML_AGENT_GIT_PASS={bot_token} \
            --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/DL_Wheat_dataset/,target=/data/ \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )
    
    test_tasks_ids = ["63778b3cd37d469a8bb3484f4fdb2782", "78b12ebba04e4e2cb676d76c4f4091c3"]

    
    test_tasks_ids = task.connect({"test_tasks_to_aggregate": test_tasks_ids}, name="tasks_ids")["test_tasks_to_aggregate"]
    
    # make process as draft:
    task.execute_remotely()
    
    # Initialize lists to store kfold_results and test_results
    kfold_results_list = []
    test_results_list = []
    hyper_parameters_list = []

    # Loop through the task names and retrieve artifacts
    for i, test_task_id in enumerate(test_tasks_ids):
        print(f"Retrieving artifacts for task {i+1}/{len(test_tasks_ids)}")
        
        # get best task id from HPO task
        executed_test_task: clearml.Task = clearml.Task.get_task(task_id=test_task_id)
        cv_task_id = executed_test_task.get_parameter("previous_tasks_ids/best_task_id")
        # get best task
        executed_cv_task = clearml.Task.get_task(task_id=cv_task_id)
        
        # Retrieve kfold_results and test_results
        kfold_results = executed_cv_task.artifacts["results"].get()
        test_results = executed_test_task.artifacts["results"].get()

        # Add a column for the task name and append to lists
        kfold_results['Experiment n.'] = i
        kfold_results_list.append(kfold_results)

        test_results = test_results.set_index("Metric").T.reset_index(drop=True)
        test_results['Experiment n.'] = i
        test_results_list.append(test_results)
        
        # Retrieve hyper-parameters
        hyper_parameters = executed_test_task.get_parameters_as_dict()["hyper-parameters"]
        hyper_parameters['Experiment n.'] = i
        hyper_parameters_df = pd.DataFrame(hyper_parameters, index=[0])
        hyper_parameters_list.append(hyper_parameters_df)

    # Concatenate lists into DataFrames
    kfold_results_df = pd.concat(kfold_results_list, ignore_index=True)
    test_results_df = pd.concat(test_results_list, ignore_index=True)
    
    # Concatenate hyper_parameters into DataFrame
    hyper_parameters_df = pd.concat(hyper_parameters_list, ignore_index=True)
    hyper_parameters_df = hyper_parameters_df.set_index('Experiment n.')
    task.get_logger().report_table(title="Hyper parameters", series="Hyper parameters", iteration=0, table_plot=hyper_parameters_df)

    # Set "KFold iteration" as index for kfold_results_df
    if 'KFold iteration' in kfold_results_df.columns:
        kfold_results_df.set_index("KFold iteration", inplace=True)
        
    # Define a dictionary for color mapping based on the metric
    color_mapping = {
        "Best Validation Score": {"K-Fold": "lightcoral", "Test": "darkred"},
        "Test Score": {"K-Fold": "lightgreen", "Test": "darkgreen"},
        "Test Correlation (Authors' Accuracy)": {"K-Fold": "skyblue", "Test": "navy"}
    }

    # Plot and report results
    for i, metric in enumerate(["Best Validation Score", "Test Score", "Test Correlation (Authors' Accuracy)"]):
        bar_violinplot(task, kfold_results_df, test_results_df, metric, color_mapping, i)
        
    distribution_plot(task, kfold_results_df, test_results_df, color_mapping)

def distribution_plot(task, kfold_results_df, test_results_df, color_mapping):
    # Initialize the matplotlib figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Histogram and density for "Test Correlation (Authors' Accuracy)"
    sns.kdeplot(test_results_df["Test Correlation (Authors' Accuracy)"], color=color_mapping["Test Correlation (Authors' Accuracy)"]["Test"], fill=True, ax=axes[0])
    ax2 = axes[0].twinx()
    sns.histplot(test_results_df["Test Correlation (Authors' Accuracy)"], kde=False, color=color_mapping["Test Correlation (Authors' Accuracy)"]["Test"], ax=ax2)
    axes[0].set_title("Distribution of Test Correlation (Authors' Accuracy)")
    axes[0].set_xlabel("Test Correlation (Authors' Accuracy)")
    axes[0].set_ylabel("Frequency")
    ax2.set_ylabel("Density")

    # Add a 'Type' column to distinguish between kfold and test results
    kfold_results_df['Type'] = 'K-Fold'
    test_results_df['Type'] = 'Test'

    # Combine the dataframes
    combined_results_df = pd.concat([kfold_results_df, test_results_df], ignore_index=True)

    # Melt the combined dataframe
    melted_df = combined_results_df.melt(id_vars=['Experiment n.', 'Type'], value_vars=['Best Validation Score', 'Test Score'], var_name='Metric', value_name='Score')
    sns.barplot(x='Metric', y='Score', hue='Type', data=melted_df, palette='autumn', ax=axes[1])

    axes[1].set_title('Grouped Bar Plot of Scores by Type')
    axes[1].set_xlabel('Experiment Number')
    axes[1].set_ylabel('Score')

    plt.tight_layout()
    
    # Log the figure in ClearML
    task.get_logger().report_matplotlib_figure(title=f'Histogram and Density Plot', series=f'Histogram and Density Plot', figure=fig)

def bar_violinplot(task, kfold_results_df, test_results_df, column_name, color_mapping, iteration=0):    
    # Determine colors based on the column_name
    kfold_color = color_mapping[column_name]["K-Fold"]
    test_color = color_mapping[column_name]["Test"]
    
    # Add a 'Type' column to distinguish between kfold and test results
    kfold_results_df['Type'] = 'K-Fold'
    test_results_df['Type'] = 'Test'

    # Concatenate the results dataframes for plotting
    combined_results_df = pd.concat([kfold_results_df, test_results_df], axis=0, ignore_index=True)
    
    # Creating a figure and a set of subplots with different widths
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})

    print(combined_results_df)
    print(column_name)
    # Violin plot for the specified column
    sns.barplot(data=combined_results_df, x='Type', y=column_name, palette=[kfold_color, test_color], ax=axs[0])
    axs[0].set_title(f'{column_name} by Type')
    axs[0].set_ylabel(column_name)
    axs[0].set_xlabel('Type')

    # Violin plot with a single test result shown as a dot
    # First, plot the K-Fold violin
    sns.violinplot(data=kfold_results_df, x='Experiment n.', y=column_name, color=kfold_color, ax=axs[1])
    # Now overlay the Test results as points, centered on the violin plot
    for experiment in test_results_df['Experiment n.'].unique():
        test_value = test_results_df.loc[test_results_df['Experiment n.'] == experiment, column_name].values[0]
        axs[1].scatter(experiment, test_value, color=test_color, s=100, zorder=3)  # Adjust the position to center the dot
    axs[1].set_title(f'{column_name} by Experiment Number')
    axs[1].set_ylabel(column_name)
    axs[1].set_xlabel('Experiment Number')
    
    # Create legend
    kfold_patch = mpatches.Patch(color=kfold_color, label='K-Fold')
    test_handle = mlines.Line2D([], [], color=test_color, marker='o', linestyle='None', 
                                markersize=10, label='Test')
    axs[1].legend(handles=[kfold_patch, test_handle], loc='upper left')

    plt.tight_layout()
    # Log the figure in ClearML
    task.get_logger().report_matplotlib_figure(title=f'Violin Plots', series=f'Violin Plots', iteration=iteration, figure=plt)

    task.close()
    
    
if __name__ == "__main__":
    main()