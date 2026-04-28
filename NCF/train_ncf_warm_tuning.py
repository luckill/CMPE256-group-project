import os

import pandas as pd
import torch
import torch.optim as optim

from tqdm.auto import tqdm

from NCF.calibration import CalibratedModel, fit_score_calibrator
from NCF.ncf_data import (
    prepare_split,
    prepare_eval_df,
    build_stats_from_raw_train,
    make_loader,
    temporal_split,
)
from NCF.ncf_export import (
    export_top_n_candidates_dataframe,
    export_top_n_candidates_parquet,
    export_user_embeddings_dataframe,
    export_item_embeddings_dataframe,
)
from NCF.ncf_io import load_model_checkpoint, save_model_checkpoint, maybe_compile_model
from dataset import RatingsDataset
from NCF.model import neural_cllaborative_filtering
from NCF.bpr_dataset import BPRDataset
from NCF.train_bpr import train_one_epoch_bpr
from evaluation import evaluate_rmse, get_topk_metrics
from utils import (
    all_exist,
    load_json,
    save_json,
    invert_map,
    same_metadata,
    build_stage_metadata,
    configure_runtime,
    get_num_workers,
    set_seed_x,
)


CANDIDATE_POOL_TOP_N = 100
VAL_RATIO = 0.125
RELEVANCE_THRESHOLD = 4.0
TOP_K = 10
MAX_EVAL_USERS = None
BEST_METRIC = "NDCG@K"

FORCE_RETRAIN = False
FORCE_REEXPORT = False

CONFIG_COLUMNS = [
    "config_name",
    "embedding_dimension",
    "lr",
    "batch_size",
    "weight_decay",
    "num_negatives",
    "epochs",
]

HYPERPARAMETER_CONFIGS = [
    {
        "config_name": "emb64_lr0_001_bs3000_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_002_bs3000_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.002,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_0005_bs3000_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.0005,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs3000_neg20_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 20,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs3000_neg10_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 10,
        "epochs": 50,
    },
    {
        "config_name": "emb32_lr0_001_bs3000_neg15_ep50",
        "embedding_dimension": 32,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb128_lr0_001_bs3000_neg15_ep50",
        "embedding_dimension": 128,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs2000_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 2000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs4000_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 4000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs3000_wd0_0001_neg15_ep50",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0001,
        "num_negatives": 15,
        "epochs": 50,
    },
    {
        "config_name": "emb64_lr0_001_bs3000_neg15_ep30",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 30,
    },
    {
        "config_name": "emb64_lr0_001_bs3000_neg15_ep70",
        "embedding_dimension": 64,
        "lr": 0.001,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 70,
    },
]

# Full Cartesian grid kept for reference. Move a smaller subset into
# HYPERPARAMETER_CONFIGS unless you are ready for a long run.
# HYPERPARAMETER_TUNING_GRID = [
#     # embedding_dimension: [32, 64, 128]
#     # lr: [0.0005, 0.001, 0.002]
#     # batch_size: [2000, 3000, 4000]
#     # weight_decay: [0.0, 0.0001]
#     # num_negatives: [10, 15, 20]
#     # epochs: [30, 50, 70]
# ]


def get_final_config():
    return normalize_config(HYPERPARAMETER_CONFIGS[0])


def normalize_config(config):
    cfg = dict(config)
    cfg["embedding_dimension"] = int(cfg["embedding_dimension"])
    cfg["lr"] = float(cfg["lr"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["num_negatives"] = int(cfg["num_negatives"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["config_name"] = str(cfg.get("config_name") or build_config_name(cfg))
    return cfg


def build_config_name(config):
    return (
        f"emb{int(config['embedding_dimension'])}"
        f"_lr{str(float(config['lr'])).replace('.', '_')}"
        f"_bs{int(config['batch_size'])}"
        f"_neg{int(config['num_negatives'])}"
        f"_ep{int(config['epochs'])}"
    )


def get_hyperparameter_configs():
    return [normalize_config(config) for config in HYPERPARAMETER_CONFIGS]


def checkpoint_safe_name(config):
    name = str(config.get("config_name") or build_config_name(config))
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def config_from_result(result):
    return normalize_config({key: result[key] for key in CONFIG_COLUMNS if key in result})


def load_selected_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Rerun validation tuning before skipping the validation stage."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    return config_from_result(df.iloc[0].to_dict())


def select_best_result(results, metric_name):
    valid_results = []
    for result in results:
        metric_value = result.get(metric_name)
        if metric_value is None or pd.isna(metric_value):
            continue
        valid_results.append(result)

    if not valid_results:
        raise ValueError(f"Cannot select a best config without valid {metric_name} values.")

    def sort_key(result):
        rmse = result.get("RMSE")
        if rmse is None or pd.isna(rmse):
            rmse = float("inf")
        return float(result[metric_name]), -float(rmse)

    return max(valid_results, key=sort_key)


def save_dataframe_formats(df, base_path):
    df.to_parquet(f"{base_path}.parquet", index=False)
    df.to_csv(f"{base_path}.csv", index=False)


def filter_warm_eval_rows(train_df, eval_df):
    """
    Keep only warm rows:
      - user appears in train_df
      - item appears in train_df
    """
    train_users = set(train_df["user_id"].astype(str).unique())
    train_items = set(train_df["item_id"].astype(str).unique())

    df = eval_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    warm_mask = df["user_id"].isin(train_users) & df["item_id"].isin(train_items)
    return df[warm_mask].copy()


def same_training_metadata(meta_path, current_meta):
    """
    Match the model-training inputs while ignoring candidate export size.
    Changing top_n should re-export candidates, not retrain the NCF checkpoint.
    """
    if not os.path.exists(meta_path):
        return False

    try:
        old_meta = load_json(meta_path)
    except Exception:
        return False

    old_training_meta = dict(old_meta)
    current_training_meta = dict(current_meta)
    old_training_meta.pop("top_n", None)
    current_training_meta.pop("top_n", None)

    return old_training_meta == current_training_meta


def evaluate_topk(
    model,
    train_df,
    eval_df,
    num_items,
    device,
    k,
    relevance_threshold,
    max_users,
):
    model.eval()
    with torch.inference_mode():
        metrics = get_topk_metrics(
            model,
            train_df=train_df,
            test_df=eval_df,
            num_items=num_items,
            device=device,
            k=k,
            relevance_threshold=relevance_threshold,
            max_users=max_users,
        )
    return metrics


def run_experiment(
    prepared_split,
    calibration_split,
    config,
    epochs,
    device,
    num_workers,
    relevance_threshold=4.0,
    k=10,
    max_users=None,
    export_top_n=None,
    export_top_n_path=None,
    export_embeddings=True,
    checkpoint_path=None,
    load_existing_model=True,
    save_trained_model=True,
):
    train_df = prepared_split["train_df"]
    eval_df = prepared_split["eval_df"]
    num_users = prepared_split["num_users"]
    num_items = prepared_split["num_items"]
    calibration_df = calibration_split["eval_df"]

    if len(eval_df) == 0:
        return {
            **config,
            "epochs": epochs,
            "final_bpr_loss": None,
            "RMSE": 0.0,
            "Precision@K": 0.0,
            "Recall@K": 0.0,
            "NDCG@K": 0.0,
            "num_eval_users": 0,
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_dataset = BPRDataset(
        train_df,
        num_items=num_items,
        positive_threshold=relevance_threshold,
        num_negatives=config["num_negatives"],
    )
    calibration_dataset = RatingsDataset(calibration_df)
    eval_dataset = RatingsDataset(eval_df)

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        device=device,
        num_workers=num_workers,
    )
    calibration_loader = make_loader(
        dataset=calibration_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )
    eval_loader = make_loader(
        dataset=eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        device=device,
        num_workers=num_workers,
    )

    optimizer = None
    last_loss = None
    loaded_from_checkpoint = False

    if checkpoint_path is not None and load_existing_model and os.path.exists(checkpoint_path):
        print(f"Loading saved model from: {checkpoint_path}")
        model, _ = load_model_checkpoint(
            path=checkpoint_path,
            num_users=num_users,
            num_items=num_items,
            device=device,
        )
        loaded_from_checkpoint = True
    else:
        model = neural_cllaborative_filtering(
            num_users=num_users,
            num_items=num_items,
            embedding_dimension=config["embedding_dimension"],
            use_sigmoid=False,
        ).to(device)

        model = maybe_compile_model(model)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        epoch_bar = tqdm(
            range(epochs),
            desc="epochs",
            leave=False,
        )

        for epoch in epoch_bar:
            last_loss = train_one_epoch_bpr(model, train_loader, optimizer, device)
            epoch_bar.set_postfix(loss=f"{last_loss:.6f}")

        if checkpoint_path is not None and save_trained_model:
            save_model_checkpoint(model, config, checkpoint_path)
            print(f"Saved trained model to: {checkpoint_path}")

    if loaded_from_checkpoint:
        model = maybe_compile_model(model)

    calibrator, calibration_scale, calibration_bias = fit_score_calibrator(
        model,
        calibration_loader,
        device,
    )
    calibrated_model = CalibratedModel(model, calibrator)

    rmse = evaluate_rmse(calibrated_model, eval_loader, device)
    metrics = evaluate_topk(
        model=model,
        train_df=train_df,
        eval_df=eval_df,
        num_items=num_items,
        device=device,
        k=k,
        relevance_threshold=relevance_threshold,
        max_users=max_users,
    )

    result = {
        **config,
        "epochs": epochs,
        "final_bpr_loss": last_loss,
        "RMSE": rmse,
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
        **metrics,
    }

    candidate_df = pd.DataFrame()
    user_emb_df = pd.DataFrame()
    item_emb_df = pd.DataFrame()

    if export_top_n is not None and export_top_n > 0:
        user_inv_map = invert_map(prepared_split["user_map"])
        item_inv_map = invert_map(prepared_split["item_map"])

        if export_top_n_path is not None:
            export_candidate_rows = export_top_n_candidates_parquet(
                model=model,
                train_df=train_df,
                eval_df=eval_df,
                num_items=num_items,
                device=device,
                output_path=export_top_n_path,
                top_n=export_top_n,
                user_inv_map=user_inv_map,
                item_inv_map=item_inv_map,
                rating_model=calibrated_model,
            )
        else:
            candidate_df = export_top_n_candidates_dataframe(
                model=model,
                train_df=train_df,
                eval_df=eval_df,
                num_items=num_items,
                device=device,
                top_n=export_top_n,
                user_inv_map=user_inv_map,
                item_inv_map=item_inv_map,
                rating_model=calibrated_model,
            )
            export_candidate_rows = len(candidate_df)

        result["export_top_n"] = int(export_top_n)
        result["export_candidate_rows"] = int(export_candidate_rows)

    if export_embeddings:
        user_emb_df = export_user_embeddings_dataframe(
            model=model,
            user_map=prepared_split["user_map"],
        )
        item_emb_df = export_item_embeddings_dataframe(
            model=model,
            item_map=prepared_split["item_map"],
        )

        result["user_embedding_dim"] = int(
            len([c for c in user_emb_df.columns if c.startswith("user_emb_")])
        )
        result["beer_embedding_dim"] = int(
            len([c for c in item_emb_df.columns if c.startswith("beer_emb_")])
        )
        result["num_user_embeddings"] = int(len(user_emb_df))
        result["num_item_embeddings"] = int(len(item_emb_df))

    try:
        del model
    except NameError:
        pass

    try:
        del calibrator
    except NameError:
        pass

    try:
        del calibrated_model
    except NameError:
        pass

    try:
        del optimizer
    except NameError:
        pass

    del train_loader
    del calibration_loader
    del eval_loader
    del train_dataset
    del calibration_dataset
    del eval_dataset

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, candidate_df, user_emb_df, item_emb_df


def main():
    set_seed_x(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)
    num_workers = get_num_workers()

    print("Device:", device)
    print("DataLoader workers:", num_workers)

    full_train_raw = pd.read_parquet("advanced_train_temporal.parquet")
    test_raw = pd.read_parquet("advanced_test_temporal.parquet")

    hyperparameter_configs = get_hyperparameter_configs()
    val_ratio = VAL_RATIO
    top_n = CANDIDATE_POOL_TOP_N

    tune_train_raw, calibration_raw = temporal_split(
        full_train_raw,
        val_ratio_within_train=val_ratio,
    )

    # Warm-only filtering
    warm_calibration_raw = filter_warm_eval_rows(tune_train_raw, calibration_raw)
    warm_test_raw = filter_warm_eval_rows(full_train_raw, test_raw)

    print("Full train rows:", len(full_train_raw))
    print("Tune-train rows:", len(tune_train_raw))
    print("Original calibration rows:", len(calibration_raw))
    print("Warm calibration rows:", len(warm_calibration_raw))
    print("Original held-out test rows:", len(test_raw))
    print("Warm held-out test rows:", len(warm_test_raw))

    VAL_MODEL_TEMPLATE = "ncf_bpr_val_model_warm_{config_name}.pt"
    TEST_MODEL_PATH = "ncf_bpr_test_model_warm.pt"

    VAL_META_PATH = "val_artifacts_meta_warm.json"
    TEST_META_PATH = "test_artifacts_meta_warm.json"
    BEST_CONFIG_PATH = "best_batch_size_config_warm.csv"
    VALIDATION_RESULT_PATH = "validation_export_result_warm.csv"
    FINAL_TEST_RESULT_PATH = "best_hidden_layer_final_test_result_warm.csv"

    VAL_FILES = [
        "advanced_calibration_temporal_warm.parquet",
        "top100_candidates_val_warm.parquet",
        "user_embeddings_val_warm.parquet",
        "user_embeddings_val_warm.csv",
        "item_embeddings_val_warm.parquet",
        "item_embeddings_val_warm.csv",
        "user_stats_val_warm.parquet",
        "user_stats_val_warm.csv",
        "item_stats_val_warm.parquet",
        "item_stats_val_warm.csv",
        BEST_CONFIG_PATH,
        VALIDATION_RESULT_PATH,
    ]

    TEST_FILES = [
        "top100_candidates_with_scores_warm.parquet",
        "user_embeddings_warm.parquet",
        "user_embeddings_warm.csv",
        "item_embeddings_warm.parquet",
        "item_embeddings_warm.csv",
        "user_stats_warm.parquet",
        "user_stats_warm.csv",
        "item_stats_warm.parquet",
        "item_stats_warm.csv",
        FINAL_TEST_RESULT_PATH,
    ]

    val_meta = build_stage_metadata(
        stage_name="val_warm",
        config={"configs": hyperparameter_configs, "best_metric": BEST_METRIC},
        train_file="advanced_train_temporal.parquet",
        val_ratio=val_ratio,
        top_n=top_n,
    )

    val_meta_matches = same_metadata(VAL_META_PATH, val_meta)
    val_training_meta_matches = same_training_metadata(VAL_META_PATH, val_meta)
    need_val = FORCE_RETRAIN or FORCE_REEXPORT or (not all_exist(VAL_FILES)) or (not val_meta_matches)

    print("\nWarm BPR hyperparameter configs:")
    for config in hyperparameter_configs:
        print(config)
    print(f"Candidate pool top_n: {top_n}")
    print(f"Best validation metric: {BEST_METRIC}")

    if not need_val:
        print("\nWarm validation artifacts already up to date. Skipping validation stage.")
        final_config = load_selected_config(BEST_CONFIG_PATH)
    else:
        print("\nTuning warm validation artifacts...")

        prepared_val_split = prepare_split(tune_train_raw, warm_calibration_raw)
        prepared_val_calibration_split = {
            "eval_df": prepare_eval_df(
                warm_calibration_raw,
                prepared_val_split["user_map"],
                prepared_val_split["item_map"],
            )
        }

        val_results = []
        for config_idx, config in enumerate(hyperparameter_configs, start=1):
            print(
                f"\nRunning validation config {config_idx}/{len(hyperparameter_configs)}: "
                f"{config['config_name']}"
            )
            val_model_path = VAL_MODEL_TEMPLATE.format(
                config_name=checkpoint_safe_name(config),
            )

            val_result, _, _, _ = run_experiment(
                prepared_split=prepared_val_split,
                calibration_split=prepared_val_calibration_split,
                config=config,
                epochs=config["epochs"],
                device=device,
                num_workers=num_workers,
                relevance_threshold=RELEVANCE_THRESHOLD,
                k=TOP_K,
                max_users=MAX_EVAL_USERS,
                export_top_n=None,
                export_embeddings=False,
                checkpoint_path=val_model_path,
                load_existing_model=(
                    (not FORCE_RETRAIN)
                    and val_training_meta_matches
                    and os.path.exists(val_model_path)
                ),
                save_trained_model=True,
            )
            val_result["selection_stage"] = "validation"
            val_result["selection_metric"] = BEST_METRIC
            val_results.append(val_result)
            print(val_result)

        validation_best_result = select_best_result(val_results, BEST_METRIC)
        final_config = config_from_result(validation_best_result)

        print(
            f"\nBest validation config by {BEST_METRIC}: "
            f"{final_config['config_name']}"
        )
        print(final_config)

        best_val_model_path = VAL_MODEL_TEMPLATE.format(
            config_name=checkpoint_safe_name(final_config),
        )
        val_result, val_top100_df, val_user_emb_df, val_item_emb_df = run_experiment(
            prepared_split=prepared_val_split,
            calibration_split=prepared_val_calibration_split,
            config=final_config,
            epochs=final_config["epochs"],
            device=device,
            num_workers=num_workers,
            relevance_threshold=RELEVANCE_THRESHOLD,
            k=TOP_K,
            max_users=MAX_EVAL_USERS,
            export_top_n=top_n,
            export_top_n_path="top100_candidates_val_warm.parquet",
            export_embeddings=True,
            checkpoint_path=best_val_model_path,
            load_existing_model=(not FORCE_RETRAIN) and os.path.exists(best_val_model_path),
            save_trained_model=True,
        )
        val_result["selection_stage"] = "validation_best_export"
        val_result["selection_metric"] = BEST_METRIC

        val_user_stats_df, val_item_stats_df = build_stats_from_raw_train(tune_train_raw)

        # Save warm-specific filenames
        warm_calibration_raw.to_parquet("advanced_calibration_temporal_warm.parquet", index=False)
        save_dataframe_formats(val_user_emb_df, "user_embeddings_val_warm")
        save_dataframe_formats(val_item_emb_df, "item_embeddings_val_warm")
        save_dataframe_formats(val_user_stats_df, "user_stats_val_warm")
        save_dataframe_formats(val_item_stats_df, "item_stats_val_warm")

        pd.DataFrame(val_results).to_csv(VALIDATION_RESULT_PATH, index=False)
        pd.DataFrame([final_config])[CONFIG_COLUMNS].to_csv(BEST_CONFIG_PATH, index=False)
        save_json(VAL_META_PATH, val_meta)

        print("Saved warm validation artifacts.")

    test_meta = build_stage_metadata(
        stage_name="test_warm",
        config=final_config,
        train_file="advanced_train_temporal.parquet",
        test_file="advanced_test_temporal.parquet",
        val_ratio=val_ratio,
        top_n=top_n,
    )

    test_meta_matches = same_metadata(TEST_META_PATH, test_meta)
    test_training_meta_matches = same_training_metadata(TEST_META_PATH, test_meta)
    need_test = FORCE_RETRAIN or FORCE_REEXPORT or (not all_exist(TEST_FILES)) or (not test_meta_matches)

    print("\nSelected final BPR config:")
    print(final_config)

    if not need_test:
        print("\nWarm test artifacts already up to date. Skipping test stage.")
    else:
        print("\nBuilding warm test artifacts...")

        prepared_final_split = prepare_split(full_train_raw, warm_test_raw)
        prepared_calibration_split = {
            "eval_df": prepare_eval_df(
                calibration_raw,
                prepared_final_split["user_map"],
                prepared_final_split["item_map"],
            )
        }

        final_result, top100_df, user_emb_df, item_emb_df = run_experiment(
            prepared_split=prepared_final_split,
            calibration_split=prepared_calibration_split,
            config=final_config,
            epochs=final_config["epochs"],
            device=device,
            num_workers=num_workers,
            relevance_threshold=RELEVANCE_THRESHOLD,
            k=TOP_K,
            max_users=MAX_EVAL_USERS,
            export_top_n=top_n,
            export_top_n_path="top100_candidates_with_scores_warm.parquet",
            export_embeddings=True,
            checkpoint_path=TEST_MODEL_PATH,
            load_existing_model=(not FORCE_RETRAIN) and test_training_meta_matches,
            save_trained_model=True,
        )
        print(final_result)
        test_user_stats_df, test_item_stats_df = build_stats_from_raw_train(full_train_raw)

        # Save warm-specific filenames
        save_dataframe_formats(user_emb_df, "user_embeddings_warm")
        save_dataframe_formats(item_emb_df, "item_embeddings_warm")
        save_dataframe_formats(test_user_stats_df, "user_stats_warm")
        save_dataframe_formats(test_item_stats_df, "item_stats_warm")

        pd.DataFrame([final_result]).to_csv(FINAL_TEST_RESULT_PATH, index=False)
        save_json(TEST_META_PATH, test_meta)

        print("Saved warm test artifacts.")

    print("\nDone.")


if __name__ == "__main__":
    main()
