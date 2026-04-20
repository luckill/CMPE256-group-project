# CMPE256 Group Project

## Dataset Link:
[Google Drive](https://drive.google.com/drive/folders/1Z3eIdnZBQHGS5ElH6QiJ0XO2kaYDegj-?usp=sharing)

## For CMPE256_Group_Project_EDA.ipynb:

- Load ratebeer.json and beeradvocate.json to Colab runtime, or the ‘content’ folder in the working directory,
- Run all cells to view dataset analysis and train/test split creation.

## For CMPE256_Group_Project_Baseline.ipynb:

- Download basic_train_temporal.parquet and basic_test_temporal.parquet from the shared Google Drive folder and upload them to your Colab runtime (/content/),
- Update the file paths in the notebook to /content/basic_train_temporal.parquet and /content/basic_test_temporal.parquet if needed,
- Run all cells to train NormalPredictor and BaselineOnly (ALS) baselines and view RMSE, Precision@10, Recall@10 and NDCG@10 results.

## For SVD_baseline.py

- Clone the project.
- Go to the root folder for the project.
- Run python3 -m venv <#VIRTUAL ENVIRONEMNT NAME#>to create a python virtual environment.
- Run to source <#virtual environment name#>.bin/activate to activate the virtual environment.
- Run pip install -r requirements.txt to install the neccessary libraries.
- Run python svd_baseline.py to view RMSE, Precision@10, Recall@10 and NDCG@10 results.