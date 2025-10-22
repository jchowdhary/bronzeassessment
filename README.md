# bronzeassessment

This repository contains a set of Jupyter notebooks and datasets used for small machine learning demonstrations (Pima diabetes prediction, housing regression, employee attrition, etc.).

Below are simple steps to download the project, create a conda virtual environment, install required Python packages from `requirements.txt`, and run the notebooks.

## 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/<your-username>/bronzeassessment.git
cd bronzeassessment
```

Replace `<your-username>` with your GitHub username.

## 2. Create (or update) a Conda environment

Recommended: create a new conda environment to avoid interfering with system packages.

```bash
# create a new environment named 'bronze' with Python 3.10 (adjust version if needed)
conda create -n bronze python=3.10 -y

# activate the environment
conda activate bronze
```

If you prefer a different name, replace `bronze` with your chosen name.

## 3. Install required packages

Install the packages listed in `requirements.txt` into the active conda environment:

```bash
pip install -r requirements.txt
```

Note: We use `pip` inside the conda env to install the exact packages listed. If you prefer `conda` packages, install them manually with `conda install`.

## 4. Run the notebooks

You can run the notebooks interactively with Jupyter Notebook or JupyterLab, or execute them non-interactively.

Interactive (recommended):

```bash
# start JupyterLab (or jupyter notebook)
jupyter lab
# or
jupyter notebook
```

Then open the notebook files located in `bronzeLayer/` (for example `bronzeLayer/neuralnetwork_with_kerastensorflow.ipynb`) and run the cells.

Non-interactive (run and save outputs):

```bash
# install nbconvert if not already present
pip install nbconvert

# run a notebook and overwrite it with the executed version
jupyter nbconvert --to notebook --execute bronzeLayer/neuralnetwork_with_kerastensorflow.ipynb --output bronzeLayer/neuralnetwork_with_kerastensorflow_executed.ipynb
```

Repeat the above `nbconvert` command for each notebook you want to execute. Adjust filenames as needed.

## 5. Notes and tips
- The dataset files are under `bronzeLayer/dataset/`.
- Some notebooks save model files (for example `diabetes_prediction_model.h5`) into the repo; be careful not to overwrite important files.
- If you run into package version issues, try creating the environment with a specific Python version and install compatible package versions.
- To export environment packages for reproducibility:

```bash
pip freeze > requirements_freeze.txt
# or for conda
conda env export > environment.yml
```

## 6. Contact / Licence
This repo is for learning/demonstration. Adapt and reuse as needed.
