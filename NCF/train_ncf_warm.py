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
    set_seed,
    configure_runtime,
    get_num_workers,
    set_seed_x,
)


CANDIDATE_POOL_TOP_N = 100


def get_final_config():
    return {
        "embedding_dimension": 64,
        "lr": 0.002,
        "batch_size": 3000,
        "weight_decay": 0.0,
        "num_negatives": 15,
        "epochs": 50,
    }


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

    final_config = get_final_config()
    val_ratio = 0.125
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

    FORCE_RETRAIN = False
    FORCE_REEXPORT = False

    VAL_MODEL_PATH = "ncf_bpr_val_model_warm.pt"
    TEST_MODEL_PATH = "ncf_bpr_test_model_warm.pt"

    VAL_META_PATH = "val_artifacts_meta_warm.json"
    TEST_META_PATH = "test_artifacts_meta_warm.json"

    VAL_FILES = [
        "advanced_calibration_temporal_warm.parquet",
        "top100_candidates_val_warm.parquet",
        "user_embeddings_val_warm.parquet",
        "item_embeddings_val_warm.parquet",
        "user_stats_val_warm.parquet",
        "item_stats_val_warm.parquet",
    ]

    TEST_FILES = [
        "top100_candidates_with_scores_warm.parquet",
        "user_embeddings_warm.parquet",
        "item_embeddings_warm.parquet",
        "user_stats_warm.parquet",
        "item_stats_warm.parquet",
    ]

    val_meta = build_stage_metadata(
        stage_name="val_warm",
        config=final_config,
        train_file="advanced_train_temporal.parquet",
        val_ratio=val_ratio,
        top_n=top_n,
    )

    test_meta = build_stage_metadata(
        stage_name="test_warm",
        config=final_config,
        train_file="advanced_train_temporal.parquet",
        test_file="advanced_test_temporal.parquet",
        val_ratio=val_ratio,
        top_n=top_n,
    )

    val_meta_matches = same_metadata(VAL_META_PATH, val_meta)
    test_meta_matches = same_metadata(TEST_META_PATH, test_meta)
    val_training_meta_matches = same_training_metadata(VAL_META_PATH, val_meta)
    test_training_meta_matches = same_training_metadata(TEST_META_PATH, test_meta)

    need_val = FORCE_RETRAIN or FORCE_REEXPORT or (not all_exist(VAL_FILES)) or (not val_meta_matches)
    need_test = FORCE_RETRAIN or FORCE_REEXPORT or (not all_exist(TEST_FILES)) or (not test_meta_matches)

    print("\nUsing final BPR config:")
    print(final_config)
    print(f"Candidate pool top_n: {top_n}")

    if not need_val:
        print("\nWarm validation artifacts already up to date. Skipping validation stage.")
    else:
        print("\nBuilding warm validation artifacts...")

        prepared_val_split = prepare_split(tune_train_raw, warm_calibration_raw)
        prepared_val_calibration_split = {
            "eval_df": prepare_eval_df(
                warm_calibration_raw,
                prepared_val_split["user_map"],
                prepared_val_split["item_map"],
            )
        }

        val_result, val_top100_df, val_user_emb_df, val_item_emb_df = run_experiment(
            prepared_split=prepared_val_split,
            calibration_split=prepared_val_calibration_split,
            config=final_config,
            epochs=final_config["epochs"],
            device=device,
            num_workers=num_workers,
            relevance_threshold=4.0,
            k=10,
            max_users=None,
            export_top_n=top_n,
            export_top_n_path="top100_candidates_val_warm.parquet",
            export_embeddings=True,
            checkpoint_path=VAL_MODEL_PATH,
            load_existing_model=(not FORCE_RETRAIN) and val_training_meta_matches,
            save_trained_model=True,
        )
        print(val_result)
        val_user_stats_df, val_item_stats_df = build_stats_from_raw_train(tune_train_raw)

        # Save warm-specific filenames
        warm_calibration_raw.to_parquet("advanced_calibration_temporal_warm.parquet", index=False)
        val_user_emb_df.to_parquet("user_embeddings_val_warm.parquet", index=False)
        val_item_emb_df.to_parquet("item_embeddings_val_warm.parquet", index=False)
        val_user_stats_df.to_parquet("user_stats_val_warm.parquet", index=False)
        val_item_stats_df.to_parquet("item_stats_val_warm.parquet", index=False)

        pd.DataFrame([val_result]).to_csv("validation_export_result_warm.csv", index=False)
        save_json(VAL_META_PATH, val_meta)

        print("Saved warm validation artifacts.")


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
            relevance_threshold=4.0,
            k=10,
            max_users=None,
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
        user_emb_df.to_parquet("user_embeddings_warm.parquet", index=False)
        item_emb_df.to_parquet("item_embeddings_warm.parquet", index=False)
        test_user_stats_df.to_parquet("user_stats_warm.parquet", index=False)
        test_item_stats_df.to_parquet("item_stats_warm.parquet", index=False)

        save_json(TEST_META_PATH, test_meta)

        print("Saved warm test artifacts.")

    pd.DataFrame([final_config]).to_csv("best_batch_size_config_warm.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
