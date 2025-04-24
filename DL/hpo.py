import json
from clearml import Task
import clearml
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformParameterRange, LogUniformParameterRange,
    UniformIntegerParameterRange)
from clearml.automation import GridSearch
import pandas as pd


def main():
    
    task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml/drafts", 
                                           task_name="HPO task",
                                           task_type="optimizer",
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
    
    # set a a seed for the child tasks
    tasks_seed = 100 
    tasks_seed = task.connect({"seed": tasks_seed}, "tasks_seed")["seed"]

    task.execute_remotely()
    
    hyper_parameters = [
        DiscreteParameterRange('hyper-parameters/hidden_layer_sizes', [
            (19,19,19), (19, 38, 38, 19), (38,38,38,19),  (50, 38, 38), (90,90,90), (120,90,90)
        ]),
        DiscreteParameterRange('hyper-parameters/activation', ['tanh', 'relu', 'identity']),
        DiscreteParameterRange('hyper-parameters/solver', ['sgd', 'adam']),
        DiscreteParameterRange('hyper-parameters/alpha', [0.001, 0.05, 0.4]), 
        DiscreteParameterRange('hyper-parameters/learning_rate', ['constant', 'adaptive']),
    ]
    
    hyper_parameters.append(
        DiscreteParameterRange('seed/seed', [tasks_seed])
    )

    # Example use case:
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=Task.get_task(project_name="e-muse/DL_Wheat_with_clearml/drafts",
                                           task_name="base kfold training task").id,
        hyper_parameters=hyper_parameters,
        
        objective_metric_title='KFold R-Square',
        objective_metric_series='Average',
        objective_metric_sign='max',

        execution_queue="aai-gpu-01-cpu:1",

        # setting optimizer 
        optimizer_class=GridSearch,

        # total_max_jobs=10,

        max_number_of_concurrent_tasks=5,
        # optimization_time_limit=20*60,

        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5,

        spawn_project="e-muse/DL_Wheat_with_clearml/HPO_tasks"
    )

    # start the optimization process
    optimizer.start()
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()

    # optimization is completed, print the top performing experiments id
    best_task = optimizer.get_top_experiments(top_k=1)[0]
    
    # make sure background optimization stopped
    optimizer.stop()

    task.connect({"task_id": best_task.id}, "best_task")
    
    task.close()


if __name__ == "__main__":
    main()
