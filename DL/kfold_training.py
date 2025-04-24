import json
import pickle
from os.path import join
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import clearml
import numpy as np
import seaborn as sns
from utils.modeling import multi_layer_perceptron_cv
from utils.utils import get_data


# Set the Seaborn style
sns.set_theme(style='whitegrid')


# calculate average of items in a list
def Average(lst): 
    return sum(lst) / len(lst)


def main():
    task: clearml.Task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml/drafts", 
                                           task_name="base kfold training task",
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
    
    # Set the seed
    seed = 43
    seed = task.connect({"seed": seed}, "seed")["seed"]
    
    # MLP hyperparameters
    hyper_parameters = {
        'hidden_layer_sizes': (120, 90, 90),
        'activation': 'relu',
        'solver': 'sgd',
        'alpha': 0.05,  # regularization
        'learning_rate': 'constant',
    }
    hyper_parameters = task.connect(hyper_parameters, "hyper-parameters")
    task.upload_artifact("hyper-parameters",hyper_parameters)
    
    dataset_name = "splitted"
    dataset_name = task.connect({"name": dataset_name}, "dataset")["name"]
    
    # make process as draft:
    task.execute_remotely()
    # task.execute_remotely("aai-gpu-01-cpu:1")
    
    #start time
    start = time.time()
    
    X_train, y_train, X_test, y_test, y_min_max_scaler = get_data(dataset_name=dataset_name)
    
    np.random.seed(seed)
    
    # Multilayer perceptron
    print('training MLP:')
    multi_layer_perceptron_cv(X_train, y_train, X_test, y_test, hyper_parameters, task, n_splits=5, seed=seed, min_max_scaler=y_min_max_scaler)
    
    #end time
    end = time.time()
    print("Time elapsed: ",end - start)
    
    task.close()

if __name__  == "__main__":
    main()
