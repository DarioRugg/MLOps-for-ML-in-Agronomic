import json
import random
from clearml import Task
import clearml
from clearml.automation import PipelineController


def save_hpo_task_id(pipeline: PipelineController, node: PipelineController.Node):
    pipeline.add_parameter("tasks_ids", default=pipeline.get_parameters()["tasks_ids"] + [node.executed])
    return

def save_test_task_id(pipeline: PipelineController, node: PipelineController.Node):
    pipeline.add_parameter("tasks_ids", default=pipeline.get_parameters()["tasks_ids"] + [node.executed])
    return


def main():
    
    task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml", 
                                           task_name="test pipeline",
                                           task_type="controller",
                                           reuse_last_task_id=False)
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image="rugg/dlwheatwithclearml:clearml",
        docker_arguments="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
            --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
            --env CLEARML_AGENT_GIT_USER={bot_name} \
            --env CLEARML_AGENT_GIT_PASS={bot_token} \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )
    
    pipe = PipelineController(name="pipeline",
        project="e-muse/DL_Wheat_with_clearml",
        target_project="e-muse/DL_Wheat_with_clearml",
        version="0.0.1",
        docker="rugg/dlwheatwithclearml:clearml",
        docker_args="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
            --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
            --env CLEARML_AGENT_GIT_USER={bot_name} \
            --env CLEARML_AGENT_GIT_PASS={bot_token} \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )
    
    global_seed = 1234
    
    # Set the seed for reproducibility
    random.seed(global_seed)


    rep_num = 2
    # Generate a list of rep_num random numbers between 0 and 10000
    children_seeds = [random.randint(0, 10000) for _ in range(rep_num)]
    
    task.connect({"seed": global_seed, "number_of_repetitions": rep_num, "children_seeds": children_seeds}, "config")

    pipe.add_parameter(name="tasks_ids", default=[])
    aggregator_parents = []

    previous_hpo = None
    for i, seed in enumerate(children_seeds):
        hpo_task_name = f"HPO {i}"
        pipe.add_step(name=hpo_task_name, 
                    base_task_project="e-muse/DL_Wheat_with_clearml/drafts", 
                    base_task_name="HPO task", 
                    parameter_override={"seed/seed": seed},
                    execution_queue="aai-gpu-01-cpu:2",
                    parents=previous_hpo,
                    post_execute_callback=save_hpo_task_id)
        previous_hpo = [hpo_task_name]
        
        task_name = f"Test {i}"
        pipe.add_step(name=task_name, 
                    base_task_project="e-muse/DL_Wheat_with_clearml/drafts", 
                    base_task_name="base training testing task", 
                    parameter_override={"previous_tasks_ids/hpo_task_id":f"${{{hpo_task_name}.id}}"},
                    execution_queue="aai-gpu-01-cpu:1",
                    parents=previous_hpo,
                    post_execute_callback=save_test_task_id)
        aggregator_parents.append(task_name)
    
    # results aggregation
    pipe.add_step(name="pipeline aggregator", 
                base_task_project="e-muse/DL_Wheat_with_clearml/drafts", 
                base_task_name="aggregator task",
                parameter_override={"tasks_ids/test_tasks_to_aggregate": "${pipeline.tasks_ids}"},
                parents=aggregator_parents,
                execution_queue="aai-gpu-01-cpu:1")

    pipe.start(queue="services")
    pipe.wait()
    pipe.stop()
    

if __name__ == "__main__":
    main()
