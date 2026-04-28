# CMPE256 Group Project
[link to the preprocessed dataset](https://drive.google.com/drive/folders/1Z3eIdnZBQHGS5ElH6QiJ0XO2kaYDegj-?usp=sharing)

# Phase 1 #

## For CMPE256_Group_Project_EDA.ipynb:

- Load ratebeer.json and beeradvocate.json to Colab runtime, or the ‘content’ folder in the working directory.
- Run all cells to view dataset analysis and train/test split creation.

## For CMPE256_Group_Project_Baseline.ipynb:

- Download basic_train_temporal.parquet and basic_test_temporal.parquet from the shared Google Drive folder and upload them to your Colab runtime (/content/)
- Update the file paths in the notebook to /content/basic_train_temporal.parquet and /content/basic_test_temporal.parquet if needed
- Run all cells to train NormalPredictor and BaselineOnly (ALS) baselines and view RMSE, Precision@10, Recall@10 and NDCG@10 results

## For SVD_baseline.py

- clone the project.
- go to the root folder for the project.
- run python3 -m venv <#VIRTUAL ENVIRONEMNT NAME#>to create a python virtual environment.
- run to source <#virtual environment name#>.bin/activate to activate the virtual environment.
- run pip install -r requirements.txt to install the neccessary libraries.
- run python svd_baseline.py to view RMSE, Precision@10, Recall@10 and NDCG@10 results.

# Phase 2 #
## For content based+autoencoder.ipynb ##

- Ensure you have access to the shared Google Drive folder SP2026_CMPE256_Group12_Data which contains advanced_train_temporal.parquet and advanced_test_temporal.parquet
- Mount your Google Drive when prompted
- Run all cells in order to build beer feature matrix, train the autoencoder, and evaluate using negative sampling
Final results include Precision@10, Recall@10 and NDCG@10 for all users, warm users and cold users separately

## For CMPE256_Group_Project_RK_Variant-2.ipynb##
- Load advanced_train_temporal.parquet and advanced_test_temporal.parquet to Colab runtime, or the ‘content’ folder in the working directory,,
- Run all cells to view validation experimentation and final training and test results.

## Full Hybrid Recommender System

The sections above describe the original EDA and baseline workflow. This section is an add-on for the full recommender system, which combines:

- warm-start Neural Collaborative Filtering/BPR candidate generation,
- LinUCB reranking for warm users and items,
- cold-start fallback recommenders for unseen users and/or unseen items,
- a hybrid router that chooses the correct recommendation route for each test case.

### Data and artifact files

The large dataset and generated artifacts can be downloaded from the shared Google Drive folder or regenerated locally by running the pipeline scripts.

The advanced pipeline expects these main input files in the project root:

- `advanced_train_temporal.parquet`
- `advanced_test_temporal.parquet`

The warm NCF/BPR pipeline generates the main warm-start artifacts:

- `advanced_calibration_temporal_warm.parquet`
- `top100_candidates_val_warm.parquet`
- `top100_candidates_with_scores_warm.parquet`
- `user_embeddings_val_warm.parquet`
- `item_embeddings_val_warm.parquet`
- `user_embeddings_warm.parquet`
- `item_embeddings_warm.parquet`
- `user_stats_val_warm.parquet`
- `item_stats_val_warm.parquet`
- `user_stats_warm.parquet`
- `item_stats_warm.parquet`
- `ncf_bpr_val_model_warm.pt`
- `ncf_bpr_test_model_warm.pt`

### Environment setup

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Warm NCF/BPR candidate generation

Run the warm-start NCF/BPR pipeline to train or load the warm models, evaluate the warm split, and export top-100 candidate parquet files, embeddings, and stats:

```bash
python NCF/train_ncf_warm.py
```

Main outputs include:

- `validation_export_result_warm.csv`
- `top100_candidates_val_warm.parquet`
- `top100_candidates_with_scores_warm.parquet`
- warm user/item embedding parquet files
- warm user/item stats parquet files
- warm model checkpoints

### Warm LinUCB reranking

Run the standalone warm LinUCB reranker after the warm NCF/BPR artifacts exist:

```bash
python bandit/bandit_training_final.py
```

This script loads the top-100 warm NCF candidates, combines NCF scores with contextual bandit features, and prints the final warm reranking summary.

### Cold-start recommenders

Run the cold-start fallback scripts for users and items that cannot be handled by the warm NCF route:

```bash
python cold_case1_eval_cpu_optimized.py
python cold_case2_eval_cpu_optimized.py
python cold_case3_eval_cpu_optimized.py
```

The cold-start cases are:

- Case 1: seen user, unseen item
- Case 2: unseen user, seen item
- Case 3: unseen user, unseen item

Main outputs include:

- `cold_case1_ranked_recommendations.csv`
- `cold_case1_user_metrics.csv`
- `cold_case1_summary.csv`
- `cold_case2_ranked_recommendations.csv`
- `cold_case2_user_metrics.csv`
- `cold_case2_summary.csv`
- `cold_case3_ranked_recommendations.csv`
- `cold_case3_user_metrics.csv`
- `cold_case3_summary.csv`



### Full hybrid controller

Run the final hybrid recommender router after the warm artifacts are available:

```bash
python hybrid_recommender_router.py
```

The router sends warm user-item cases through NCF top-100 candidates plus LinUCB reranking, then uses the cold-start fallback routes for unseen-user and/or unseen-item cases. Relevance metrics use rating `>= 4.0`, and the default recommendation list size is top 10.

Main outputs include:

- `hybrid_router_recommendations.csv`
- `hybrid_router_user_metrics.csv`

### Optional main-method notebook

For a step-by-step notebook version of the main project workflow, open:

- `CMPE256_Group_Project_Main_Methods.ipynb`

This notebook mirrors the main runnable scripts while keeping the original Python files available as standalone entrypoints.
