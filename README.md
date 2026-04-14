## link for the preprocessed dataset:
[google drive folder for the preprocessed dataset](https://drive.google.com/drive/folders/1Z3eIdnZBQHGS5ElH6QiJ0XO2kaYDegj-?usp=sharing)

## For CMPE256_Group_Project_EDA.ipynb:

- Load ratebeer.json and beeradvocate.json to Colab runtime, or the ‘content’ folder in the working directory.,
- Run all cells to view dataset analysis and train/test split creation.,

## For CMPE256_Group_Project_Baseline.ipynb:

- Download basic_train_temporal.parquet and basic_test_temporal.parquet from the shared Google Drive folder and upload them to your Colab runtime (/content/),
- Update the file paths in the notebook to /content/basic_train_temporal.parquet and /content/basic_test_temporal.parquet if needed,
- Run all cells to train NormalPredictor and BaselineOnly (ALS) baselines and view RMSE, Precision@10, Recall@10 and NDCG@10 results

## For SVD_baseline.py:
- clone the project
- Go to the root folder of the project and do python3 -m venv <name of the virtual; environement>
- do source <name of your vitual environment>/bin/activate
- do pip install -r requirements.txt
- with the virtual environment activated, run python svd_baseline.py to view the RMSE, Precision@10, Recall@10 and NDCG@10 results.
