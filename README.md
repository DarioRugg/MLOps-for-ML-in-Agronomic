# Wheat Plant Height Prediction with Explainability: MLOps Integration Showcase

This project demonstrates a full MLOps pipeline for a wheat plant height prediction model using the [ClearML](https://clear.ml) framework. The goal is to showcase how ClearML can be used for dataset versioning, experiment tracking, hyperparameter optimization (HPO), and model deployment — all integrated within a reproducible, containerized environment.

---

## 📦 Project Structure

* `Dockerfile`: Environment for Python-based preprocessing and ML experiments.
* `r.Dockerfile`: R environment used for raw data preprocessing.
* `DL/kfold_training.py`: Creates a draft task that performs k-fold cross-validation to assess model performance for HPO and model selection.
* `DL/hpo.py`: Creates a draft task that performs hyperparameter optimization using ClearML.
* `DL/training_and_testing.py`: Creates a draft task that uses the best hyperparameters from HPO to train on the full training set and evaluate on the test set.
* `DL/pipeline.py`: Runs the entire pipeline by triggering the draft tasks generated with the above scripts.
* `Filtering pipeline/Genotyping and phenotyping filtering pipeline.R`: R script for initial preprocessing of the raw dataset.
* `upload_data.py`: Executes later preprocessing steps and uploads the processed data as ClearML Datasets.
* `Data/`: Contains the original raw data (⚠️ should be excluded from version control).

---

## ⚙️ Setup Instructions

### 1. Set up ClearML Agent

Follow the [ClearML Agent Setup Guide](https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_setup) to install and configure the ClearML agent. This agent will allow the automatic execution of experiments inside Docker containers.

### 2. Configure Docker

Ensure the Docker daemon is running. You'll use Docker to build isolated environments for:

* **Python (ML pipeline)** via `Dockerfile`
* **R (data filtering)** via `r.Dockerfile`

Build the containers:

```bash
docker build -f Dockerfile -t rugg/dlwheatwithclearml:latest .
docker build -f r.Dockerfile -t rugg/dlwheatwithclearml:r .
```

---

## 📂 Dataset Handling

The dataset is located in the `data/` directory and includes raw data obtained from [Sandhu et al.](https://github.com/Sandhu-WSU/DL_Wheat/tree/master). After cloning the repository, we recommend excluding the `data/` folder from version control:

```bash
echo "data/" >> .gitignore
```

### Preprocessing Steps

1. **Run the R script** `Filtering pipeline/Genotyping and phenotyping filtering pipeline.R` inside the R container to perform initial filtering on the raw data.
2. **Upload the filtered dataset** by running:

```bash
python upload_data.py
```

This completes preprocessing and registers the dataset in ClearML for subsequent tasks, the default agronomic trait is `2014_Height` the plant height for the growing season of 2014.

---

## 🧪 Creating Draft Tasks

These scripts create draft experiments in ClearML that will be later used by the pipeline task as base tasks.

### 1. K-Fold Evaluation Task

This task runs cross-validation on the current configuration:

```bash
python DL/kfold_training.py
```

### 2. Hyperparameter Optimization Task

To generate the HPO draft:

```bash
python DL/hpo.py
```

### 3. Final Evaluation Task

To create the final evaluation task using the best parameters found during HPO:

```bash
python DL/training_and_testing.py
```

---

## 🔁 Launch the Full Pipeline

You can run the full pipeline — from training to evaluation — by executing:

```bash
python DL/pipeline.py
```

This script chains all tasks and runs them automatically using the ClearML agent.

---

## 📊 Monitoring with ClearML

All experiment tasks — cross-validation, HPO, training, testing — are logged and visualized through the ClearML Web UI. This includes:

* Real-time logs and performance metrics
* Dataset versioning and reproducibility
* HPO progress and comparison of model variants

---

## 📌 Notes

* This setup is designed to **showcase MLOps capabilities** with ClearML and Docker.
* The focus is not on replicating published results, but rather on demonstrating a robust and modular MLOps pipeline.
* Feel free to fork, extend, or adapt the pipeline for your own use cases or datasets.

