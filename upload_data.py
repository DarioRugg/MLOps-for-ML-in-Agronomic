import json
import os
from os.path import join
import shutil
from clearml import Dataset
import clearml
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt


os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# Data processing
def data_scaling(dataset,target): 
    # Scaled data
    min_max_scaler = MinMaxScaler(feature_range = (0,1))
    np_scaled = min_max_scaler.fit_transform(dataset)
    X = pd.DataFrame(np_scaled)
    
    target_edit = pd.Series(target).values
    target_edit = target_edit.reshape(-1,1)
    np_scaled = min_max_scaler.fit_transform(target_edit)
    Y = pd.DataFrame(np_scaled)
    
    return X, Y, min_max_scaler
    
def data_split(X,Y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)
    return X_train, y_train.values, X_test, y_test.values

def data_normalizing(X_train, X_test):
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
        
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return X_train, X_test

def make_task(task_name, task_project) -> clearml.Task:
    task = clearml.Task.init(project_name=task_project, 
                                           task_name=task_name,
                                           task_type=clearml.TaskTypes.data_processing,
                                           reuse_last_task_id=False,
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
            --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/clearml_datasets/,target=/clearml_agent_cache/storage_manager/datasets/ \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )
    return task

def clear_dir_content(dir_path):
    # Remove the directory and its contents, then recreate the directory
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def plot_target_density(vector_a, task: clearml.Task, title, vector_b=None, iter=0):
    
    if vector_b is None:
        df = pd.DataFrame({'Value': vector_a, 'Split': 'All'})
    else:
        # Combining the vectors into a DataFrame
        df_a = pd.DataFrame({'Value': vector_a, 'Split': 'Train'})
        df_b = pd.DataFrame({'Value': vector_b, 'Split': 'Test'})
        df = pd.concat([df_a, df_b], ignore_index=True)
        
    # Creating the overlaid density plots
    sns.kdeplot(data=df, x="Value", hue="Split", fill=True, common_norm=False, alpha=0.5)
    plt.title('Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    
    task.get_logger().report_matplotlib_figure(title, title, figure=plt.gcf(), iteration=iter)
    plt.close()


def main():
    
    # Define the directory path
    data_dir_to_upload = './data_for_clearml/'
    original_data_dir = '/data/'

    # Remove the directory and its contents, then recreate the directory
    clear_dir_content(data_dir_to_upload)
    
    # ==============================
    # SECTION: Original data
    # ==============================
    
    task = make_task("uploading original dataset", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset before any preprocessing")
    
    # copy the original dataset to the dirictory
    shutil.copy2(join(original_data_dir, "myGD"), join(data_dir_to_upload, "myGD"))
    shutil.copy2(join(original_data_dir, "myGM.csv"), join(data_dir_to_upload, "myGM.csv"))
    shutil.copy2(join(original_data_dir, "Pheno.csv"), join(data_dir_to_upload, "Pheno.csv"))
    
    dataset = Dataset.create(dataset_name='Original Dataset', dataset_project='e-muse/DL_Wheat_with_clearml')
    
    dataset.add_files(data_dir_to_upload)

    dataset.upload()
    dataset.finalize()
    
    task.close()

    # ==============================
    # SECTION: Data filtering
    # ==============================
    
    task = make_task("uploading dataset filtered", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset after the R filtering pipeline")
    
    clear_dir_content(data_dir_to_upload)
    
    shutil.copy2(join(original_data_dir, "NAM_dat(dataset).csv"), join(data_dir_to_upload, "NAM_dat(dataset).csv"))

    data_df = pd.read_csv(join(data_dir_to_upload, 'NAM_dat(dataset).csv'), header=0, nrows=15)
    
    # Remove the directory and its contents, then recreate the directory
    dataset = Dataset.create(dataset_name='Dataset Filtered', dataset_project='e-muse/DL_Wheat_with_clearml', use_current_task=True, parent_datasets=[Dataset.get(dataset_project='e-muse/DL_Wheat_with_clearml', dataset_name='Original Dataset')])
    
    # n_files_removed = dataset.remove_files(data_dir_to_upload)
    removed_files = []
    for file in dataset.list_files():
        removed = dataset.remove_files(file, verbose=True)
        if removed:
            removed_files.append(file)
    print(f"Removed {removed_files} from the dataset")
    
    dataset.add_files(data_dir_to_upload)

    dataset.get_logger().report_table(title='Data samples',series='Data samples',iteration=0, table_plot=data_df.iloc[:15, :10])
    
    dataset.upload()
    dataset.finalize()
    
    task.get_logger().report_table(title='Data samples',series='Data samples',iteration=0, table_plot=data_df.iloc[:15, :10])
    
    task.close()
    
    # ==============================
    # SECTION: Data and target selection
    # ==============================
    
    # original data
    task = make_task("uploading dataset X and y", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset after the R filtering pipeline and divided in X and y")
    
    data_df = pd.read_csv(join(data_dir_to_upload, 'NAM_dat(dataset).csv'), header=0)
    
    x_data = data_df.iloc[:,60:]
    y_data = data_df.loc[:, "2014_Height"]
    
    # Remove the directory and its contents, then recreate the directory
    clear_dir_content(data_dir_to_upload)
    
    x_data.to_hdf(join(data_dir_to_upload, 'x_data.h5'), key="data", mode='w', index=False)
    y_data.to_csv(join(data_dir_to_upload, 'y_data.csv'), index=False)
    
    dataset = Dataset.create(dataset_name='Dataset data and target', dataset_project='e-muse/DL_Wheat_with_clearml', use_current_task=True, parent_datasets=[dataset])

    removed_files = []
    for file in dataset.list_files():
        removed = dataset.remove_files(file, verbose=True)
        if removed:
            removed_files.append(file)
    print(f"Removed {removed_files} from the dataset")

    dataset.add_files(data_dir_to_upload)
    
    dataset.upload()
    dataset.finalize()
    
    plot_target_density(y_data.values, task, "Target Density")
    
    task.get_logger().report_table(title='X Data',series='X Data',iteration=0, table_plot=x_data.iloc[:15, :10])
    task.get_logger().report_table(title='y Data',series='y Data',iteration=0, table_plot=y_data.iloc[:15].to_frame())
    
    task.close()
    
    # ==============================
    # SECTION: Data scaling
    # ==============================
    
    # original data
    task = make_task("uploading dataset scaled", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset after the R filtering pipeline and divided in X and y and min-max scaled")
    
    x_data, y_data, y_scaler = data_scaling(x_data, y_data)
    
    x_data.to_hdf(join(data_dir_to_upload, 'x_data.h5'), key="data", mode='w', index=False)
    y_data.to_csv(join(data_dir_to_upload, 'y_data.csv'), index=False)
    joblib.dump(y_scaler, join(data_dir_to_upload, 'y_min_max_scaler.pkl'))
    
    
    dataset = Dataset.create(dataset_name='Dataset scaled', dataset_project='e-muse/DL_Wheat_with_clearml', use_current_task=True, parent_datasets=[Dataset.get(dataset_project='e-muse/DL_Wheat_with_clearml', dataset_name='Dataset data and target')])

    dataset.add_files(data_dir_to_upload)
    
    dataset.upload()
    dataset.finalize()
    
    plot_target_density(np.squeeze(y_data.values), task, "Target Density")
    
    task.get_logger().report_table(title='X Data',series='X Data',iteration=0, table_plot=x_data.iloc[:15, :10])
    task.get_logger().report_table(title='y Data',series='y Data',iteration=0, table_plot=y_data.iloc[:15])
    
    task.close()
    
    # ==============================
    # SECTION: Data splitting
    # ==============================
    
    task = make_task("uploading dataset train test splits", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset after the R filtering pipeline, divided in X and y and splitted in train and test sets")
    
    x_train, y_train, x_test, y_test = data_split(x_data, y_data)
    
    # Remove the directory and its contents, then recreate the directory
    clear_dir_content(data_dir_to_upload)
    
    x_train.to_hdf(join(data_dir_to_upload, 'x_train.h5'), key="data", mode='w', index=False)
    np.save(join(data_dir_to_upload, 'y_train.npy'), y_train)
    x_test.to_hdf(join(data_dir_to_upload, 'x_test.h5'), key="data", mode='w', index=False)
    np.save(join(data_dir_to_upload, 'y_test.npy'), y_test)
    
    dataset = Dataset.create(dataset_name='Dataset split', dataset_project='e-muse/DL_Wheat_with_clearml', use_current_task=True, parent_datasets=[Dataset.get(dataset_project='e-muse/DL_Wheat_with_clearml', dataset_name='Dataset scaled')])
    
    n_files_removed = dataset.remove_files(join('x_data.h5'), verbose=True)
    n_files_removed += dataset.remove_files(join('y_data.csv'), verbose=True)
    print(f"Removed {n_files_removed} previous files from the dataset")
    
    dataset.add_files(data_dir_to_upload)
    
    dataset.upload()
    dataset.finalize()
    
    plot_target_density(np.squeeze(y_train), task, "Target Density", np.squeeze(y_test))
    
    task.get_logger().report_table(title='X Samples Data',series='Train Split',iteration=0, table_plot=x_train.iloc[:15, :10])
    task.get_logger().report_table(title='X Samples Data',series='Test Split',iteration=1, table_plot=x_test.iloc[:15, :10])
    task.get_logger().report_table(title='y Samples Data',series='Train Split',iteration=0, table_plot=pd.Series(np.squeeze(y_train)[:15]).to_frame())
    task.get_logger().report_table(title='y Samples Data',series='Test Split',iteration=1, table_plot=pd.Series(np.squeeze(y_test)[:15]).to_frame())
    
    task.close()
    
    # ==============================
    # SECTION: Data normalization
    # ==============================
    
    task = make_task("uploading dataset normalized", "e-muse/DL_Wheat_with_clearml/data_processing")
    task.set_comment("Uploading the dataset after the R filtering pipeline, divided in X and y, splitted in train and test sets and standardized")
    
    x_train, x_test = data_normalizing(x_train, x_test)
    
    x_train.to_hdf(join(data_dir_to_upload, 'x_train.h5'), key="data", mode="w", index=False)
    x_test.to_hdf(join(data_dir_to_upload, 'x_test.h5'), key="data", mode='w', index=False)
    
    dataset = Dataset.create(dataset_name='Dataset normalized', dataset_project='e-muse/DL_Wheat_with_clearml', use_current_task=True, parent_datasets=[Dataset.get(dataset_project='e-muse/DL_Wheat_with_clearml', dataset_name='Dataset split')])
    
    dataset.add_files(data_dir_to_upload)
    
    dataset.upload()
    dataset.finalize()
    
    plot_target_density(np.squeeze(y_train), task, "Target Density", np.squeeze(y_test))
    
    task.get_logger().report_table(title='X Samples Data',series='Train Split',iteration=0, table_plot=x_train.iloc[:15, :10])
    task.get_logger().report_table(title='X Samples Data',series='Test Split',iteration=1, table_plot=x_test.iloc[:15, :10])
    task.get_logger().report_table(title='y Samples Data',series='Train Split',iteration=0, table_plot=pd.Series(np.squeeze(y_train[:15])).to_frame())
    task.get_logger().report_table(title='y Samples Data',series='Test Split',iteration=1, table_plot=pd.Series(np.squeeze(y_test[:15])).to_frame())
    
    task.close()
    
    
if __name__ == "__main__":
    main()