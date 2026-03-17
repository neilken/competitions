from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

from src.datasets.soundscape_dataset import SoundscapeSegmentDataset
from src.models.model_factory import create_model
from src.utils.experiment_registry import append_row
from src.utils.metrics import macro_roc_auc_skip_empty


def print_step(message: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{message}\n{line}")


def load_config(path: Path) -> Dict:
    # Centralized YAML loading keeps all run settings reproducible and explicit.
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    # Deterministic seeds reduce run-to-run variance and simplify debugging.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_workers(requested_workers: int) -> int:
    """
    Keep DataLoader workers Colab-safe:
    - In Colab, aggressive worker counts can be unstable depending on runtime memory/process limits.
    - For portability we clamp to a small, safe value.
    """
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    if in_colab:
        return max(0, min(requested_workers, 2))
    return max(0, requested_workers)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []

    # TQDM progress bars show batch-level movement for longer epochs.
    for batch in tqdm(loader, desc="train" if is_train else "valid", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_true = []
    all_pred = []
    all_row_ids: List[str] = []

    # Predict loop keeps its own progress bar so validation/inference progress is visible.
    for batch in tqdm(loader, desc="predict", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_true.append(y)
        all_pred.append(probs)
        all_row_ids.extend(batch["row_id"])

    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0, 0), dtype=np.float32)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 0), dtype=np.float32)
    return y_true, y_pred, all_row_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline CNN on labeled soundscape segments.")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "baseline_colab.yaml"),
        help="Path to baseline training config.",
    )
    args = parser.parse_args()

    print_step("[STEP 1/8] Loading run configuration")
    cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    tr_cfg = cfg["training"]
    proj = cfg["project"]
    repro = cfg["reproducibility"]

    print_step("[STEP 2/8] Seeding and device selection")
    set_seed(int(proj["seed"]))
    device = choose_device(str(tr_cfg.get("device", "auto")).lower())
    print(f"[INFO] Training device: {device}")

    print_step("[STEP 3/8] Resolving input/output paths")
    data_root = Path(paths["data_root"])
    output_root = Path(paths["output_root"])
    folds_csv = Path(paths["folds_csv"])
    registry_csv = Path(paths["experiment_registry_csv"])

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] folds_csv={folds_csv}")
    print(f"[INFO] registry_csv={registry_csv}")

    taxonomy_csv = data_root / data_cfg["taxonomy_csv"]
    soundscape_dir = data_root / data_cfg["train_soundscapes_dir"]

    required = [taxonomy_csv, folds_csv, soundscape_dir]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[ERROR] Missing required training inputs:")
        for p in missing:
            print(f"  - {p}")
        return 1

    print_step("[STEP 4/8] Loading metadata and fold definitions")
    taxonomy = pd.read_csv(taxonomy_csv)
    folds = pd.read_csv(folds_csv)
    classes = taxonomy["primary_label"].astype(str).tolist()
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    if "fold" not in folds.columns:
        print("[ERROR] folds CSV must contain 'fold'.")
        return 1

    print(f"[INFO] taxonomy rows={len(taxonomy)}")
    print(f"[INFO] fold rows={len(folds)}")
    print(f"[INFO] num classes={len(classes)}")

    max_rows = tr_cfg.get("max_rows")
    if max_rows:
        folds = folds.iloc[: int(max_rows)].copy()
        print(f"[INFO] max_rows applied: {len(folds)} rows")

    n_folds = int(tr_cfg["folds"])
    epochs = int(tr_cfg["epochs"])
    batch_size = int(tr_cfg["batch_size"])
    lr = float(tr_cfg["learning_rate"])
    requested_workers = int(tr_cfg.get("num_workers", 0))
    num_workers = resolve_num_workers(requested_workers)

    sample_rate = int(feat_cfg["sample_rate"])
    clip_seconds = float(feat_cfg["window_seconds"])
    n_mels = int(feat_cfg["n_mels"])
    n_fft = int(feat_cfg["fft_size"])
    hop_length = int(feat_cfg["hop_length"])

    print("[INFO] training config summary:")
    print(f"  folds={n_folds}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  requested_num_workers={requested_workers}, resolved_num_workers={num_workers}")
    print(f"  sample_rate={sample_rate}, clip_seconds={clip_seconds}, n_mels={n_mels}, n_fft={n_fft}, hop={hop_length}")

    oof_preds = []
    fold_scores = []
    ckpt_dir = output_root / "checkpoints"
    oof_dir = output_root / "oof"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    print_step("[STEP 5/8] Fold training loop")
    start_time = time.time()
    for fold in tqdm(range(n_folds), desc="folds", unit="fold"):
        train_df = folds[folds["fold"] != fold].reset_index(drop=True)
        valid_df = folds[folds["fold"] == fold].reset_index(drop=True)
        if len(valid_df) == 0:
            print(f"[WARN] Fold {fold} has zero validation rows. Skipping.")
            continue

        print(f"[INFO] fold={fold} train_rows={len(train_df)} valid_rows={len(valid_df)}")
        train_ds = SoundscapeSegmentDataset(
            frame=train_df,
            soundscape_dir=soundscape_dir,
            class_to_idx=class_to_idx,
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        valid_ds = SoundscapeSegmentDataset(
            frame=valid_df,
            soundscape_dir=soundscape_dir,
            class_to_idx=class_to_idx,
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = create_model(str(tr_cfg["model_family"]), num_classes=len(classes)).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[INFO] fold={fold} model_family={tr_cfg['model_family']} params={param_count:,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = -1.0
        best_state = None
        print(f"[INFO] Starting epochs for fold={fold}")
        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device, criterion)
            valid_loss = run_epoch(model, valid_loader, None, device, criterion)
            y_true, y_pred, _ = predict_loader(model, valid_loader, device)
            val_auc = macro_roc_auc_skip_empty(y_true, y_pred)
            print(
                f"[FOLD {fold}] epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} val_macro_auc={val_auc:.6f}"
            )
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is None:
            print(f"[WARN] Fold {fold} produced no valid AUC. Using last model state.")
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_auc = float("nan")

        model.load_state_dict(best_state)
        y_true, y_pred, row_ids = predict_loader(model, valid_loader, device)
        fold_auc = macro_roc_auc_skip_empty(y_true, y_pred)
        fold_scores.append((fold, fold_auc))
        print(f"[OK] Fold {fold} best macro AUC: {fold_auc:.6f}")

        fold_oof = pd.DataFrame({"row_id": row_ids})
        for idx, col in enumerate(classes):
            fold_oof[col] = y_pred[:, idx]
        fold_oof["fold"] = fold
        oof_preds.append(fold_oof)

        ckpt_path = ckpt_dir / f"{repro['config_id']}_fold{fold}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "classes": classes,
                "config_id": repro["config_id"],
                "model_family": str(tr_cfg["model_family"]),
                "feature_config": feat_cfg,
                "fold": fold,
            },
            ckpt_path,
        )
        print(f"[OK] Saved checkpoint: {ckpt_path}")

    if not oof_preds:
        print("[ERROR] No fold outputs were generated.")
        return 1

    print_step("[STEP 6/8] Writing OOF artifacts")
    oof_df = pd.concat(oof_preds, ignore_index=True)
    oof_path = oof_dir / f"{repro['config_id']}_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[OK] Saved OOF predictions: {oof_path}")

    print_step("[STEP 7/8] Logging experiment registry")
    total_minutes = (time.time() - start_time) / 60.0
    fold_summary = "; ".join([f"fold{f}={s:.6f}" for f, s in fold_scores])
    avg_auc = float(np.nanmean([s for _, s in fold_scores])) if fold_scores else float("nan")
    run_id = f"{proj['run_name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    append_row(
        registry_csv,
        {
            "run_id": run_id,
            "phase": "phase2_baseline_training",
            "config_id": repro["config_id"],
            "seed": str(proj["seed"]),
            "fold_strategy": "groupkfold_site_date",
            "model_family": str(tr_cfg["model_family"]),
            "feature_version": "logmel_v1",
            "checkpoint_id": str(repro["checkpoint_id"]),
            "oof_metric_public": f"{avg_auc:.6f}" if np.isfinite(avg_auc) else "",
            "runtime_minutes": f"{total_minutes:.2f}",
            "status": "completed",
            "notes": fold_summary,
        },
    )
    print(f"[OK] Experiment log updated: {registry_csv}")
    print_step("[STEP 8/8] Baseline training complete")
    print(f"[INFO] avg_macro_auc={avg_auc:.6f}" if np.isfinite(avg_auc) else "[INFO] avg_macro_auc=nan")
    print(f"[INFO] total_runtime_minutes={total_minutes:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
