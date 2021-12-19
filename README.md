# DS4400 Final Project
Repository for the DS4400 final project.

# Setup

Using conda, run the following commands:
* `conda env create -f env.yml`
* `conda activate ds4400-final-project`
* `pip install -e .`

Downloading the dataset requires usage of Kaggle's API. Follow the `API Credentials` section
of the [Kaggle API's GitHub page](https://github.com/Kaggle/kaggle-api) for information on
setting up authentication for Kaggle. Once authentication is set up, run `python ds4400_final_project/dataset/download_gtzan.py`

Alternatively, the dataset can be directly downloaded from [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
and extracted as `ds4400-final-project/Data`, with the `Data` folder at the same directory level as `setup.py`, `README.md`, etc.

# Running

The code to view the results of logistic regression, SVM, and the multilayer perceptron are all in jupyter notebooks under the `notebooks/` directory. The command
`jupyter notebook` can be used to open Jupyter and run the notebooks.

## Logistic Regression

`notebooks/logistic_regression.ipynb` contains all the code necessary to run logistic regression on the GTZAN dataset. The code can be ran
using `jupyter notebook`.

## SVM

`notebooks/svm.ipynb` contains all the code necessary to run SVM on the GTZAN dataset. The code can be ran
using `jupyter notebook`.

## Neural Network

`notebooks/neural_net.ipynb` contains all the code necessary to run multilayer perceptrons on the GTZAN dataset. The code can be ran
using `jupyter notebook`. Note that this notebook takes approximately 15 - 20 minutes to run.

## Visualization

`notebooks/data_visualization` contains code that can be used to help visualize the features used for training. The code can be ran
using `jupyter notebook`.
