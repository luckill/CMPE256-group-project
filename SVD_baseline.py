import pandas as pd
import ast
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

def load_json(path, max_lines=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data.append(ast.literal_eval(line))
            except:
                continue
            if max_lines and i >= max_lines:
                break
    return pd.DataFrame(data)

# load raw dataframe
df = load_json("beeradvocate.json")

# keep only needed columns
df = df[[
    "review/profileName",
    "beer/beerId",
    "review/overall"
]].copy()

df.columns = ["user_id", "item_id", "rating"]

# clean types
df["user_id"] = df["user_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# optional filtering
user_counts = df["user_id"].value_counts()
item_counts = df["item_id"].value_counts()

print("DataFrame length(before):", len(df))
df = df[df["user_id"].isin(user_counts[user_counts >= 2].index)]
df = df[df["item_id"].isin(item_counts[item_counts >= 2].index)]
print("DataFrame length(after):", len(df))

# build Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# hold out a test set FIRST
train_data, testset = train_test_split(data, test_size=0.2, random_state=42)

# GridSearchCV needs a Dataset, so rebuild one from the training ratings
train_df = pd.DataFrame(train_data.build_testset(), columns=["user_id", "item_id", "rating"])
train_dataset = Dataset.load_from_df(train_df, reader)

param_grid = {
    "n_epochs": [5, 10, 20, 50],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1],
    "n_factors": [20, 50, 100]
}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures=["rmse"],
    cv=3,
    refit=False,
    return_train_measures=True,
    n_jobs=-1,
    pre_dispatch="2*n_jobs",
    joblib_verbose=5
)

gs.fit(train_dataset)

print("Best CV RMSE:", gs.best_score["rmse"])
print("Best params:", gs.best_params["rmse"])

# retrain on full training portion
best_params = gs.best_params["rmse"]
algo = SVD(**best_params)
algo.fit(train_data)

# evaluate once on held-out test set
predictions = algo.test(testset)
test_rmse = accuracy.rmse(predictions, verbose=True)

print("Final test RMSE:", test_rmse)