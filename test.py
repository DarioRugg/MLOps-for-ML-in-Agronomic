# Import these libraries
import json
import clearml


def main():
    task: clearml.Task = clearml.Task.init(project_name="e-muse/DL_Wheat_with_clearml/tests", 
                                           task_name="test task")
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image="rugg/dlwheatwithclearml:latest",
        docker_arguments="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
            --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
            --env CLEARML_AGENT_GIT_USER={bot_name} \
            --env CLEARML_AGENT_GIT_PASS={bot_token} \
            --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/DL_Wheat_dataset/,target=/data/ \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )
    
    # make process as draft:
    task.execute_remotely("aai-gpu-01-cpu:2")
    
    print("remote print")
    
    task.close()

if __name__  == "__main__":
    main()