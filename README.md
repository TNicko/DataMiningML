# ML Analysis of Stellar Classification & GWP Datasets

This repository contains code and notebooks for performing machine learning analysis on various datasets. The analysis is divided into different tasks, each focusing on specific aspects of data processing, model building, evaluation, and visualization.

## Datasets

The analysis in this repository utilizes the following kaggle datasets:

### Productivity Prediction of Garment Employees (GWP) Dataset

The [Productivity Prediction of Garment Employees](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees) dataset provides information about the garment manufacturing process and the productivity of employees. The dataset includes manually collected attributes and has been validated by industry experts.

### Stellar Classification Dataset (Star) Dataset

The [Stellar Classification Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) is obtained from the Sloan Digital Sky Survey (SDSS). It consists of 100,000 observations of space, including stars, galaxies, and quasars. Each observation is described by 17 feature columns and a class column indicating the object type.

## Contents

The repository consists of the following files and directories:

- `datasets/`: Directory containing the datasets used in the analysis.
- `task_3_1.ipynb`: Notebook for Task 3.1, which involves data preprocessing and feature selection.
- `task_3_2.ipynb`: Notebook for Task 3.2, which focuses on building and training machine learning models.
- `task_3_3.ipynb`: Notebook for Task 3.3, which utilizes hypothesis testing to compare model performance.
- `task_3_4.ipynb`: Notebook for Task 3.4, which explores clustering techniques and dimensionality reduction.
- `bamboo/`: Directory containing modules specific to the analysis pipeline.
  - `analysis.py`: Functions for plotting, visualizing and analyzing data.
  - `clustering.py`: Functions for working with clusters and cluster evaluation with Sklearn.
  - `model.py`: ModelManager class for managing and storing machine learning models.
  - `processing.py`: Contains functions for preprocessing different types of data in datasets.
  - `selection.py`: Contains functions for preparing processed dataset for training & testing.
  - `gwp_pipeline.py`: Contains the constants and functions used specifically for processing the GWP dataset.
  - `star_pipeline.py`: Contains the constants and functions used specifically for processing the Star dataset.

## Modules Used

The analysis in this repository relies on the following libraries:

- NumPy: A library for numerical computing in Python.
- SciPy: A library for scientific computing in Python.
- scikit-learn (sklearn): A machine learning library for Python, providing tools for data preprocessing, modeling, and evaluation.
- Matplotlib: A plotting library for creating visualizations in Python.
- Seaborn: A data visualization library based on Matplotlib, providing enhanced visualizations and statistical graphics.
