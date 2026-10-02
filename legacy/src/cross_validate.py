"""Stratified group k-fold cross-validation for the binary fault detector.

Protocol (see README): after excluding fixed_all_tape there are 23 usable
recordings -- 11 healthy (Healthy1 + H1S1..H1S10) and 12 damaged. The recording
is the cross-validation unit: every window of a recording stays together, so a
recording is either entirely in train or entirely in test for a given fold (this
is what makes overlapping-window leakage impossible).

  * Recordings are partitioned into K folds (default 5), stratified by class so
    each fold's held-out test set contains a MIX of unseen healthy AND damaged
    recordings. Every fold therefore yields honest sensitivity, specificity,
    balanced accuracy and ROC-AUC -- all on recordings never seen in training.
    (This replaces the old single-healthy temporal split: with 11 healthy
    recordings, specificity is now a real generalization estimate.)

Each fold trains a fresh model; no window from a test recording is ever seen in
training. We report per-fold metrics plus mean +/- std across folds and a pooled
confusion matrix. The decision threshold is calibrated leak-safely (on each
fold's own training predictions); the deployable model uses a threshold from the
pooled out-of-fold predictions.
"""
import os
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


def make_folds(recordings, k, seed=0):
    """Partition recordings into K stratified group folds.

    The recording is the unit: each recording lands wholly in exactly one fold's
    test set (and in every other fold's training set). Healthy and damaged
    recordings are shuffled and round-robin-assigned to folds SEPARATELY, so
    every fold's test set holds a mix of both classes -- guaranteeing each fold
    can compute sensitivity, specificity and ROC-AUC. Round-robin also keeps fold
    sizes as even as possible (e.g. 11 healthy over K=5 -> 3,2,2,2,2).

    Returns a list of (train_recordings, test_recordings) tuples.
    """
    rng = np.random.default_rng(seed)
    healthy = [r for r in recordings if r["label"] == HEALTHY]
    damaged = [r for r in recordings if r["label"] == DAMAGED]
    if len(healthy) < k or len(damaged) < k:
        raise RuntimeError(
            f"Need at least K={k} recordings per class for stratified {k}-fold "
            f"(have {len(healthy)} healthy, {len(damaged)} damaged).")

    def assign(recs):
        groups = [[] for _ in range(k)]
        for pos, i in enumerate(rng.permutation(len(recs))):
            groups[pos % k].append(recs[i])
        return groups

    hgroups, dgroups = assign(healthy), assign(damaged)
    folds = []
    for i in range(k):
        test = hgroups[i] + dgroups[i]
        train = [r for j in range(k) if j != i for r in hgroups[j] + dgroups[j]]
        folds.append((train, test))
    return folds


def _expand(recordings):
    """Flatten a list of recordings into all their per-window dataset records."""
    records = []
    for rec in recordings:
        records += records_for(rec, range(rec["total_chunks"]))
    return records


def make_loader(records, is_train, window_sec, hop_sec, batch_size, augment=True):
    ds = BinaryMultimodalDataset(records, is_train=is_train, augment=augment,
                                 window_sec=window_sec, hop_sec=hop_sec,
                                 imu_has_header=True)
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


def pick_threshold(y_true, y_prob):
    """Decision threshold maximizing Youden's J (= balanced accuracy).

    LEAK-SAFE: this is only ever called on a model's *training* predictions, so
    the held-out test fold never influences the chosen threshold. When several
    thresholds tie for best J (common when the model separates its training data
    cleanly), the median of that plateau is returned for stability.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_prob)) < 2:
        return 0.5
    # Candidate thresholds = midpoints between observed scores. This avoids
    # landing exactly on a saturated 0.0/1.0 value (overconfident models pile
    # probabilities there on their own training data), which would be knife-edge.
    vals = np.unique(y_prob)
    cands = (vals[:-1] + vals[1:]) / 2.0
    js = []
    for t in cands:
        pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, pred, pos_label=DAMAGED, zero_division=0)
        spec = recall_score(y_true, pred, pos_label=HEALTHY, zero_division=0)
        js.append(sens + spec - 1.0)
    js = np.array(js)
    plateau = cands[js >= js.max() - 1e-9]
    return float(np.median(plateau))


def _pooled_metrics(y_true, y_pred):
    """Micro (pooled-over-folds) sensitivity, specificity, balanced acc, accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    sens = recall_score(y_true, y_pred, pos_label=DAMAGED, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=HEALTHY, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    return sens, spec, bal, acc


def run(arch, args, folds):
    rows, all_true, all_pred05, all_predcal, all_prob = [], [], [], [], []

    for i, (train_recs_list, test_recs_list) in enumerate(folds):
        train_recs = _expand(train_recs_list)
        test_recs = _expand(test_recs_list)
        n_h = sum(1 for r in test_recs_list if r["label"] == HEALTHY)
        n_d = len(test_recs_list) - n_h

        train_loader = make_loader(train_recs, True, args.window, args.hop, args.batch,
                                   augment=not args.no_augment)
        test_loader = make_loader(test_recs, False, args.window, args.hop, args.batch)
        # Plain pass (no augmentation, no sampler) over the SAME training
        # recordings, used only to choose the threshold -> never sees the test fold.
        calib_loader = make_loader(train_recs, False, args.window, args.hop, args.batch)

        torch.manual_seed(0)
        model = MODELS[arch]().to(device)
        train_one(model, train_loader, args.epochs, args.lr)

        tr_true, _, tr_prob = evaluate(model, calib_loader)
        tau = pick_threshold(tr_true, tr_prob)

        y_true, _, y_prob = evaluate(model, test_loader)
        pred05 = (y_prob >= 0.5).astype(int)
        predcal = (y_prob >= tau).astype(int)

        sens = recall_score(y_true, predcal, pos_label=DAMAGED, zero_division=0)
        spec = recall_score(y_true, predcal, pos_label=HEALTHY, zero_division=0)
        bal = balanced_accuracy_score(y_true, predcal)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        rows.append((f"fold{i+1} ({n_h}H/{n_d}D)", tau, sens, spec, bal, auc))
        all_true.extend(y_true)
        all_pred05.extend(pred05)
        all_predcal.extend(predcal)
        all_prob.extend(y_prob)

    return (rows, np.array(all_true), np.array(all_pred05),
            np.array(all_predcal), np.array(all_prob))


def fit_final_and_save(arch, args, recordings, threshold):
    """Train ONE model on every available window and persist it for deployment.

    The cross-validation above only *estimates* generalization (each fold's
    model is discarded). This is the model you would actually ship: it has seen
    all 11 healthy + 12 damaged recordings. The decision `threshold` is passed in
    from the cross-validated (out-of-fold) predictions -- an honest, non-saturated
    operating point rather than one read off this model's overconfident training
    scores. Saved with the window/hop config so inference reproduces the framing.
    """
    all_recs = _expand(recordings)

    loader = make_loader(all_recs, True, args.window, args.hop, args.batch,
                         augment=not args.no_augment)
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
        "threshold": float(threshold),
        "class_names": ["Healthy", "Damaged"],
    }, save_path)
    print(f"💾 Saved final {arch} model (all {len(all_recs)} windows, "
          f"deploy threshold={threshold:.2f} from CV predictions) -> {save_path}")
    return save_path


def print_report(arch, rows, all_true, all_pred05, all_predcal):
    print("\n" + "=" * 78)
    print(f"  {arch.upper()} MODEL  -- stratified group {len(rows)}-fold CV")
    print("=" * 78)
    print(f"{'fold (test H/D recs)':<30}{'thr':>6}{'sens':>7}{'spec':>7}{'bal-acc':>9}{'auc':>7}")
    for name, tau, sens, spec, bal, auc in rows:
        print(f"{name:<30}{tau:>6.2f}{sens:>7.2f}{spec:>7.2f}{bal:>9.2f}{auc:>7.2f}")
    arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    mean_tau = float(np.mean([r[1] for r in rows]))
    print("-" * 78)
    print(f"{'MEAN (calibrated)':<30}{mean_tau:>6.2f}"
          f"{mean[0]:>7.2f}{mean[1]:>7.2f}{mean[2]:>9.2f}{mean[3]:>7.2f}")

    pooled05 = _pooled_metrics(all_true, all_pred05)
    pooled_cal = _pooled_metrics(all_true, all_predcal)
    print("\n  Decision-threshold effect (pooled over all folds):")
    print(f"  {'':<22}{'sens':>7}{'spec':>7}{'bal-acc':>9}{'accuracy':>10}")
    print(f"  {'@ 0.50 (default)':<22}{pooled05[0]:>7.2f}{pooled05[1]:>7.2f}"
          f"{pooled05[2]:>9.2f}{pooled05[3]:>10.2f}")
    print(f"  {'@ calibrated':<22}{pooled_cal[0]:>7.2f}{pooled_cal[1]:>7.2f}"
          f"{pooled_cal[2]:>9.2f}{pooled_cal[3]:>10.2f}")

    cm = confusion_matrix(all_true, all_predcal, labels=[HEALTHY, DAMAGED])
    print("\n  Pooled confusion @ calibrated (rows=true, cols=pred) [Healthy, Damaged]:")
    print(f"    Healthy: {cm[0]}")
    print(f"    Damaged: {cm[1]}")
    return {"rows": rows, "mean": mean, "std": std, "cm": cm,
            "pooled05": pooled05, "pooled_cal": pooled_cal, "mean_tau": mean_tau}


def write_markdown_report(results, args, md_path):
    """Render the CV results as a GitHub-friendly Markdown table."""
    L = []
    L.append("# Binary Fault Detection — Cross-Validation Report\n")
    L.append(f"_Stratified group {args.folds}-fold CV (recording-level) · "
             f"window {args.window}s · hop {args.hop}s · "
             f"{args.epochs} epochs · augmentation "
             f"{'off' if args.no_augment else 'on'}._\n")
    L.append("_Hard-decision metrics use a **leak-safe calibrated threshold** "
             "(per fold, Youden's J on that fold's own training data). "
             "ROC-AUC is threshold-free._\n")

    if len(results) > 1:
        L.append("## Summary — calibrated threshold (mean across folds)\n")
        L.append("| Model | Sensitivity | Specificity | Balanced acc | ROC-AUC |")
        L.append("|---|:--:|:--:|:--:|:--:|")
        for arch, r in results.items():
            m = r["mean"]
            L.append(f"| `{arch}` | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | **{m[3]:.2f}** |")
        L.append("")

    for arch, r in results.items():
        m, s, cm = r["mean"], r["std"], r["cm"]
        p0, pc = r["pooled05"], r["pooled_cal"]
        L.append(f"## `{arch}` model\n")
        L.append("**Decision-threshold effect (pooled over folds):**\n")
        L.append("| Threshold | Sensitivity | Specificity | Balanced acc | Accuracy |")
        L.append("|---|:--:|:--:|:--:|:--:|")
        L.append(f"| 0.50 (default) | {p0[0]:.2f} | {p0[1]:.2f} | {p0[2]:.2f} | {p0[3]:.2f} |")
        L.append(f"| calibrated (mean τ={r['mean_tau']:.2f}) | {pc[0]:.2f} | {pc[1]:.2f} "
                 f"| {pc[2]:.2f} | {pc[3]:.2f} |")
        L.append("")
        L.append("**Per-fold detail (at calibrated threshold):**\n")
        L.append("| Fold (test H/D recs) | τ | Sens | Spec | Bal-acc | AUC |")
        L.append("|---|:--:|:--:|:--:|:--:|:--:|")
        for name, tau, sens, spec, bal, auc in r["rows"]:
            L.append(f"| {name} | {tau:.2f} | {sens:.2f} | {spec:.2f} | {bal:.2f} | {auc:.2f} |")
        L.append(f"| **Mean ± std** | {r['mean_tau']:.2f} | {m[0]:.2f} ± {s[0]:.2f} "
                 f"| {m[1]:.2f} ± {s[1]:.2f} | {m[2]:.2f} ± {s[2]:.2f} | {m[3]:.2f} ± {s[3]:.2f} |")
        L.append("")
        L.append("**Pooled confusion matrix (calibrated threshold):**\n")
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
        plt.title(f"{arch} model — pooled confusion (group k-fold, calibrated τ)")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"confusion_{arch}.png")
        plt.savefig(png_path, dpi=200); plt.close()
        print(f"🖼️  Saved confusion matrix -> {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Binary stratified group k-fold CV.")
    parser.add_argument("--arch", choices=["audio", "late", "both"], default="both")
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of stratified recording-level CV folds.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for assigning recordings to folds.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable ALL train-time augmentation (random gain, additive "
                             "noise, time-shift, SpecAugment). Trains on the raw lab data; "
                             "RMS/standardization preprocessing still applies.")
    args = parser.parse_args()

    print(f"Device: {device}")
    recordings = gather_recordings(DATA_DIR, window_sec=args.window, hop_sec=args.hop)
    n_healthy = sum(1 for r in recordings if r["label"] == HEALTHY)
    n_damaged = sum(1 for r in recordings if r["label"] == DAMAGED)
    folds = make_folds(recordings, args.folds, args.seed)
    print(f"Recordings: {n_healthy} healthy + {n_damaged} damaged (tape excluded)")
    print(f"Stratified group {args.folds}-fold CV (seed {args.seed}). "
          f"Fold composition (held-out test recordings):")
    for i, (_, test_recs) in enumerate(folds):
        h = [r["base_name"] for r in test_recs if r["label"] == HEALTHY]
        d = [r["base_name"] for r in test_recs if r["label"] == DAMAGED]
        print(f"  fold{i+1}: healthy={h}  damaged={d}")
    print(f"Augmentation: {'OFF (raw lab data)' if args.no_augment else 'ON'}")

    archs = ["audio", "late"] if args.arch == "both" else [args.arch]
    results = {}
    for arch in archs:
        rows, all_true, all_pred05, all_predcal, all_prob = run(arch, args, folds)
        results[arch] = print_report(arch, rows, all_true, all_pred05, all_predcal)
        # Deploy threshold from the cross-validated (out-of-fold) predictions.
        deploy_tau = pick_threshold(all_true, all_prob)
        # CV done for this arch -> fit the deployable model on everything.
        fit_final_and_save(arch, args, recordings, deploy_tau)

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("  COMPARISON (calibrated threshold, mean across folds)")
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
