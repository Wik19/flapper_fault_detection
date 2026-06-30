"""Leave-one-recording-out cross-validation for the binary fault detector.

Protocol (see README): there are 9 usable recordings -- 1 healthy (Healthy1)
and 8 damaged -- after excluding fixed_all_tape.

  * Damaged recordings are rotated one-at-a-time as the held-out test wing
    (true "unseen damage" generalization). -> 8 folds.
  * The single healthy recording is split ONCE in time (with a guard gap so no
    train window overlaps a test window). Its held-out portion is added to every
    fold's test set so each fold can measure specificity.

Each fold trains a fresh model; nothing from the test fold is ever seen in
training. We report sensitivity (damaged recall, strong), specificity (healthy
recall, same-recording / weak), balanced accuracy and ROC-AUC, aggregated as
mean +/- std across folds, plus a pooled confusion matrix.
"""
import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score, confusion_matrix

from data_tools.binary_data import (
    BinaryMultimodalDataset, gather_recordings, records_for, HEALTHY, DAMAGED)
from architectures.binary_net import MODELS

DATA_DIR = "data/raw"
MODEL_DIR = "models/binary"     # final deployable models (matches models/<arch>/ layout)
RESULTS_DIR = "results/binary"  # CV report (matches results/<arch>/ layout)
# This environment ships a mismatched cuDNN; disabling it avoids a
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH crash (same workaround as the 5-class scripts).
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_folds(recordings, window_sec, hop_sec, healthy_split, healthy_hop=None):
    """Return (healthy_train_records, healthy_test_records, [damaged_recording,...]).

    There is only ONE healthy recording, so it is split once in time: the first
    `healthy_split` fraction trains, the tail tests, with a guard gap between so
    no train window overlaps a test window.

    `healthy_hop` densifies ONLY the healthy *training* windows. With a single
    recording the model otherwise sees very few healthy examples; a hop finer
    than `hop_sec` slides the window in smaller steps, yielding more (overlapping)
    training windows. The test tail keeps the standard hop + guard, so this never
    creates train/test overlap -- it just gives the model more varied healthy
    views of the same flight (paired with the per-window augmentation).
    """
    healthy = [r for r in recordings if r["label"] == HEALTHY]
    damaged = [r for r in recordings if r["label"] == DAMAGED]
    if len(healthy) != 1:
        raise RuntimeError(f"Expected exactly 1 healthy recording, found {len(healthy)}.")

    rec = healthy[0]
    total = rec["total_chunks"]
    guard = math.ceil(window_sec / hop_sec)  # chunks needed for non-overlap
    split_point = int(total * healthy_split)

    if healthy_hop is None or healthy_hop >= hop_sec:
        train_idx = list(range(0, split_point))            # one window per hop
    else:
        stride = healthy_hop / hop_sec                     # < 1.0 -> overlap
        train_idx = list(np.arange(0, split_point, stride))
    healthy_train = records_for(rec, train_idx)
    healthy_test = records_for(rec, range(split_point + guard, total))
    return healthy_train, healthy_test, damaged


def make_loader(records, is_train, window_sec, hop_sec, batch_size):
    ds = BinaryMultimodalDataset(records, is_train=is_train,
                                 window_sec=window_sec, hop_sec=hop_sec)
    if is_train:
        # Balance the 2 classes per batch (damaged outnumbers healthy ~6:1).
        counts = ds.label_counts()
        weights = [1.0 / counts[r["label"]] for r in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def train_one(model, loader, epochs, lr):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        for mel, imu, labels in loader:
            mel, imu, labels = mel.to(device), imu.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(mel, imu), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, preds, trues = [], [], []
    for mel, imu, labels in loader:
        mel, imu = mel.to(device), imu.to(device)
        logits = model(mel, imu)
        p = torch.softmax(logits, dim=1)[:, DAMAGED]
        probs.extend(p.cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(labels.numpy())
    return np.array(trues), np.array(preds), np.array(probs)


def run(arch, args, folds):
    healthy_train, healthy_test, damaged = folds
    rows, all_true, all_pred = [], [], []

    for held_out in damaged:
        train_recs = list(healthy_train)
        for d in damaged:
            if d["base_name"] == held_out["base_name"]:
                continue
            train_recs += records_for(d, range(d["total_chunks"]))
        test_recs = list(healthy_test) + records_for(held_out, range(held_out["total_chunks"]))

        train_loader = make_loader(train_recs, True, args.window, args.hop, args.batch)
        test_loader = make_loader(test_recs, False, args.window, args.hop, args.batch)

        torch.manual_seed(0)
        model = MODELS[arch]().to(device)
        train_one(model, train_loader, args.epochs, args.lr)
        y_true, y_pred, y_prob = evaluate(model, test_loader)

        sens = recall_score(y_true, y_pred, pos_label=DAMAGED, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=HEALTHY, zero_division=0)
        bal = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        rows.append((held_out["base_name"], sens, spec, bal, auc))
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    return rows, np.array(all_true), np.array(all_pred)


def fit_final_and_save(arch, args, folds):
    """Train ONE model on every available window and persist it for deployment.

    The cross-validation above only *estimates* generalization (each fold's
    model is discarded). This is the model you would actually ship: it has seen
    all 1 healthy + 8 damaged recordings. Saved alongside its window/hop config
    so inference can reproduce the exact framing.
    """
    healthy_train, healthy_test, damaged = folds
    all_recs = list(healthy_train) + list(healthy_test)
    for d in damaged:
        all_recs += records_for(d, range(d["total_chunks"]))

    loader = make_loader(all_recs, True, args.window, args.hop, args.batch)
    torch.manual_seed(0)
    model = MODELS[arch]().to(device)
    train_one(model, loader, args.epochs, args.lr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"best_{arch}_binary_model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "window_sec": args.window,
        "hop_sec": args.hop,
        "class_names": ["Healthy", "Damaged"],
    }, save_path)
    print(f"💾 Saved final {arch} model (trained on all {len(all_recs)} windows) "
          f"-> {save_path}")
    return save_path


def print_report(arch, rows, all_true, all_pred):
    print("\n" + "=" * 78)
    print(f"  {arch.upper()} MODEL  -- leave-one-damaged-recording-out CV ({len(rows)} folds)")
    print("=" * 78)
    print(f"{'held-out damaged wing':<34}{'sens':>7}{'spec':>7}{'bal-acc':>9}{'auc':>7}")
    for name, sens, spec, bal, auc in rows:
        print(f"{name:<34}{sens:>7.2f}{spec:>7.2f}{bal:>9.2f}{auc:>7.2f}")
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    print("-" * 78)
    print(f"{'MEAN +/- STD':<34}"
          f"{mean[0]:>7.2f}{mean[1]:>7.2f}{mean[2]:>9.2f}{mean[3]:>7.2f}")
    print(f"{'':<34}{std[0]:>6.2f}{std[1]:>7.2f}{std[2]:>9.2f}{std[3]:>7.2f}  (std)")
    cm = confusion_matrix(all_true, all_pred, labels=[HEALTHY, DAMAGED])
    print("\nPooled confusion matrix (rows=true, cols=pred) [Healthy, Damaged]:")
    print(f"  Healthy: {cm[0]}")
    print(f"  Damaged: {cm[1]}")
    return mean, std, cm


def write_markdown_report(results, args, md_path):
    """Render the CV results as a GitHub-friendly Markdown table."""
    L = []
    L.append("# Binary Fault Detection — Cross-Validation Report\n")
    L.append(f"_Leave-one-damaged-recording-out CV · window {args.window}s · "
             f"damaged hop {args.hop}s · healthy train hop {args.healthy_hop}s · "
             f"{args.epochs} epochs._\n")

    if len(results) > 1:
        L.append("## Summary (mean across folds)\n")
        L.append("| Model | Sensitivity | Specificity | Balanced acc | ROC-AUC |")
        L.append("|---|:--:|:--:|:--:|:--:|")
        for arch, r in results.items():
            m = r["mean"]
            L.append(f"| `{arch}` | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | **{m[3]:.2f}** |")
        L.append("")

    for arch, r in results.items():
        m, s, cm = r["mean"], r["std"], r["cm"]
        L.append(f"## `{arch}` model — per-fold detail\n")
        L.append("| Held-out damaged wing | Sens | Spec | Bal-acc | AUC |")
        L.append("|---|:--:|:--:|:--:|:--:|")
        for name, sens, spec, bal, auc in r["rows"]:
            L.append(f"| {name} | {sens:.2f} | {spec:.2f} | {bal:.2f} | {auc:.2f} |")
        L.append(f"| **Mean ± std** | {m[0]:.2f} ± {s[0]:.2f} | {m[1]:.2f} ± {s[1]:.2f} "
                 f"| {m[2]:.2f} ± {s[2]:.2f} | {m[3]:.2f} ± {s[3]:.2f} |")
        L.append("")
        L.append("Pooled confusion matrix:\n")
        L.append("| true ⧵ pred | Healthy | Damaged |")
        L.append("|---|:--:|:--:|")
        L.append(f"| **Healthy** | {int(cm[0][0])} | {int(cm[0][1])} |")
        L.append(f"| **Damaged** | {int(cm[1][0])} | {int(cm[1][1])} |")
        L.append("")

    with open(md_path, "w") as fh:
        fh.write("\n".join(L))
    print(f"📝 Saved Markdown report -> {md_path}")


def save_confusion_pngs(results, out_dir):
    """Save a pooled confusion-matrix heatmap per arch (matches results/*.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["Healthy", "Damaged"]
    for arch, r in results.items():
        plt.figure(figsize=(4.6, 4.0))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Purples", cbar=False,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=1, linecolor="black")
        plt.title(f"{arch} model — pooled confusion (LORO-CV)")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"confusion_{arch}.png")
        plt.savefig(png_path, dpi=200); plt.close()
        print(f"🖼️  Saved confusion matrix -> {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Binary LORO cross-validation.")
    parser.add_argument("--arch", choices=["audio", "late", "both"], default="both")
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--healthy-split", type=float, default=0.7)
    parser.add_argument("--healthy-hop", type=float, default=0.25,
                        help="Hop (s) for healthy TRAINING windows only; finer than "
                             "--hop adds overlapping healthy views. Use --hop's value "
                             "(e.g. 1.0) to disable densification.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Device: {device}")
    recordings = gather_recordings(DATA_DIR, window_sec=args.window, hop_sec=args.hop)
    folds = build_folds(recordings, args.window, args.hop, args.healthy_split, args.healthy_hop)
    healthy_train, healthy_test, damaged = folds
    print(f"Recordings: 1 healthy (Healthy1) + {len(damaged)} damaged (tape excluded)")
    print(f"Healthy windows: {len(healthy_train)} train (hop {args.healthy_hop}s) / "
          f"{len(healthy_test)} test (hop {args.hop}s, guard-gapped)")

    archs = ["audio", "late"] if args.arch == "both" else [args.arch]
    results = {}
    for arch in archs:
        rows, all_true, all_pred = run(arch, args, folds)
        mean, std, cm = print_report(arch, rows, all_true, all_pred)
        results[arch] = {"rows": rows, "mean": mean, "std": std, "cm": cm}
        # CV done for this arch -> fit the deployable model on everything.
        fit_final_and_save(arch, args, folds)

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("  COMPARISON (mean across folds)")
        print("=" * 78)
        print(f"{'model':<10}{'sens':>8}{'spec':>8}{'bal-acc':>10}{'auc':>8}")
        for arch, r in results.items():
            m = r["mean"]
            print(f"{arch:<10}{m[0]:>8.2f}{m[1]:>8.2f}{m[2]:>10.2f}{m[3]:>8.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_markdown_report(results, args, os.path.join(RESULTS_DIR, "cv_report.md"))
    try:
        save_confusion_pngs(results, RESULTS_DIR)
    except Exception as exc:  # plotting is a nicety; never fail the run over it
        print(f"(skipped confusion-matrix PNGs: {exc})")


if __name__ == "__main__":
    main()
