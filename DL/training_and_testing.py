import json
import joblib
import numpy as np
import pandas as pd
import time
import clearml
import numpy as np
import seaborn as sns
from utils.utils import get_data
from utils.clearml_reporting import log_distribution_plot, report_model_performance_test
from utils.modeling import multi_layer_perceptron_testing


# Set the Seaborn style
sns.set_theme(style='whitegrid')


def main():
    task: clearml.Task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml/drafts", 
                                           task_name="base training testing task",
                                           task_type="training",
                                           reuse_last_task_id=True,
                                           auto_connect_frameworks={'matplotlib': True, 'scikit': True, 'detect_repository': True, 'joblib': True})
    
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
    
    #start time
    start = time.time()    
    
    hpo_task_id = 'f6bd167afb9444798d7915125ffc17d1'
    
    hpo_task_id = task.connect({"hpo_task_id": hpo_task_id}, "previous_tasks_ids")["hpo_task_id"]
    
    # make process as draft:
    task.execute_remotely()
    # task.execute_remotely("aai-gpu-01-cpu:2")
    
    # get best task id from HPO task
    executed_hpo_task: clearml.Task = clearml.Task.get_task(task_id=hpo_task_id)
    best_task_id = executed_hpo_task.get_parameter("best_task/task_id")
    
    best_task_id = task.connect({"best_task_id": best_task_id}, "previous_tasks_ids")["best_task_id"]
    
    # get best task
    cv_task: clearml.Task = clearml.Task.get_task(task_id=best_task_id)
    
    
    # Set the seed as the cv task
    seed = cv_task.get_parameter("seed/seed", cast=True)
    seed = task.connect({"seed": seed}, "seed")["seed"]
    
    # Set the seed as the cv task
    dataset_name = cv_task.get_parameter("dataset/name", cast=True)
    dataset_name = task.connect({"name": dataset_name}, "dataset")["name"]
    
    dataset_id = cv_task.get_parameter("Datasets/dataset_id", cast=True)
    
    # MLP hyperparameters from the cv task
    hyper_parameters = cv_task.artifacts["hyper-parameters"].get() 
    hyper_parameters = task.connect(hyper_parameters, "hyper-parameters")
    
    
    X_train, y_train, X_test, y_test, y_min_max_scaler = get_data(dataset_name=dataset_name, dataset_id=dataset_id)
    
    np.random.seed(seed)    
    
    # Multilayer perceptron
    print('training MLP:')
    test_score, test_corr, val_score = multi_layer_perceptron_testing(X_train, y_train, X_test, y_test, hyper_parameters, task, n_splits=5, seed=seed, min_max_scaler=y_min_max_scaler)
    
    # logs with cv results
    cv_results_df: pd.DataFrame = cv_task.artifacts["results"].get()
    log_distribution_plot(cv_results_df["Test Score"], test_score, cv_results_df["Test Correlation (Authors' Accuracy)"], test_corr, cv_results_df["Best Validation Score"], val_score, task)
    
    report_model_performance_test(val_score, test_score, test_corr, cv_results_df, task)
    
    #end time
    end = time.time()
    print("Time elapsed: ",end - start)
    
    task.close()

if __name__  == "__main__":
    main()
