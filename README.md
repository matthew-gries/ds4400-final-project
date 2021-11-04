# DS4400 Final Project
Repository for the DS4400 final project.

# Setup

Using conda, run the following commands:
* `conda env create -f env.yml`
* `conda activate ds4400-final-project`
* `pip install -e .`

Downloading the dataset requires usage of Kaggle's API. Follow the `API Credentials` section
of the [Kaggle API's GitHub page](https://github.com/Kaggle/kaggle-api) for information on
setting up authentication for Kaggle.

Alternatively, the dataset can be directly downloaded from [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
and extracted as `ds4400-final-project/Data`, with the `Data` folder at the same directory level as `setup.py`, `README.md`, etc.