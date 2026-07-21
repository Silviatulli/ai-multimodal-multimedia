"""
cluster_distillation_comparison.py
------------------------------------
Compares 6 distillation strategies that integrate participant clustering into
the cross-modal KD pipeline.  All conditions share the same 5-fold KFold CV,
features, and teacher/student architecture imported from cross_modal_distillation.py.

Conditions
----------
  0. Baseline          — Student only (no distillation, no clustering)
  1. Global KD         — Student + global teacher (existing pipeline, no clustering)
  A. Cluster Teachers  — One teacher per cluster; participant gets KD from own cluster's
                         teacher (fallback to global if cluster has <5 physio-labelled samples)
  B. Cluster Feature   — 4-dim one-hot cluster ID appended to student input; clusters
                         fitted on training fold only (no data leakage)
  C. Cluster Alpha     — Per-sample distillation weight α proportional to cluster
                         consistency (lower within-cluster label variance → higher α)
  D. Relabelled KD     — Self-report labels shrunk toward cluster centroids via
                         cluster_relabel.build_cluster_relabel_candidate, then student
                         trained on shrunk labels + global teacher KD

t-SNE + KMeans (k=4, perplexity=15) clustering on the per-participant mean physio
feature matrix, scaled with StandardScaler — matches the t-SNE analysis that identified
4 arousal response profiles with mean arousal 0.89–1.52 across clusters.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from cross_modal_distillation import (
    MODALITY_DELAYS,   # noqa: F401 — re-exported for downstream callers
    PHYSIO_COLS,       # noqa: F401
    StudentNet,
    TeacherNet,
    load_data,
    train_teacher,
)
from cluster_relabel import build_cluster_relabel_candidate

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
N_CLUSTERS = 4
TSNE_PERPLEXITY = 15.0
ALPHA = 0.5
TEMPERATURE = 3.0
N_EPOCHS_TEACHER = 80
N_EPOCHS_STUDENT = 100
N_FOLDS = 5
SEED = 42
MIN_PHYSIO_CLUSTER = 5   # Strategy A: fall back to global teacher below this count


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _participant_matrix(
    data: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Return (participant_frame, X_scaled, fitted_scaler) — one row per participant."""
    pmat = data.groupby("participant")[feat_cols].mean().fillna(0)
    sc = StandardScaler()
    X_sc = sc.fit_transform(pmat.values.astype(np.float32))
    return pmat, X_sc, sc


def _tsne_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    perplexity: float = TSNE_PERPLEXITY,
    seed: int = SEED,
) -> tuple[np.ndarray, KMeans]:
    """Run TSNE + KMeans; return (cluster_labels, fitted_KMeans)."""
    perp = float(min(perplexity, max(5.0, (X_scaled.shape[0] - 1) / 3.0)))
    emb = TSNE(
        n_components=2, perplexity=perp, init="pca",
        random_state=seed, learning_rate="auto",
    ).fit_transform(X_scaled)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = km.fit_predict(emb)
    return labels, km


def compute_global_clusters(data: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Assign a cluster ID to every row in *data* (strategies A, C, D).

    Builds participant × feature matrix → StandardScaler → TSNE → KMeans(k=4).
    """
    pmat, X_sc, _ = _participant_matrix(data, feat_cols)
    labels, _ = _tsne_kmeans(X_sc)
    pid_to_cluster = dict(zip(pmat.index, labels))
    return data["participant"].map(pid_to_cluster).values.astype(int)


def compute_fold_clusters(
    data_tr: pd.DataFrame,
    data_full: pd.DataFrame,
    feat_cols: list[str],
) -> np.ndarray:
    """Strategy B — compute clusters on *training* participants only, no leakage.

    Training participants: TSNE + KMeans fitted solely on training fold.
    Test participants: assigned to nearest centroid via NearestCentroid (in scaled
    feature space) so no future data leaks into the assignment.

    Returns cluster labels indexed over all rows of *data_full*.
    """
    pmat_tr, X_tr_sc, sc = _participant_matrix(data_tr, feat_cols)
    tr_labels, _ = _tsne_kmeans(X_tr_sc)

    # Fit a NearestCentroid classifier in scaled feature space
    nc = NearestCentroid()
    nc.fit(X_tr_sc, tr_labels)

    # Apply to all participants (train participants get their exact labels back
    # since NearestCentroid will reproduce them for in-sample data)
    pmat_all = data_full.groupby("participant")[feat_cols].mean().fillna(0)
    X_all_sc = sc.transform(pmat_all.values.astype(np.float32))
    all_labels = nc.predict(X_all_sc)

    # Override training participants with their exact TSNE-based labels
    pid_to_exact = dict(zip(pmat_tr.index, tr_labels))
    for i, pid in enumerate(pmat_all.index):
        if pid in pid_to_exact:
            all_labels[i] = pid_to_exact[pid]

    pid_to_cluster = dict(zip(pmat_all.index, all_labels))
    return data_full["participant"].map(pid_to_cluster).values.astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Training utilities (extended beyond cross_modal_distillation.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _teacher_probs(
    teacher: TeacherNet,
    X: np.ndarray,
    temperature: float,
    device: str,
) -> np.ndarray:
    """Return temperature-scaled soft targets (N, n_classes) from *teacher*."""
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = F.softmax(teacher(Xt) / temperature, dim=1)
    return probs.cpu().numpy()


def _train_cluster_teacher(
    X: np.ndarray,
    y_class: np.ndarray,
    n_epochs: int = N_EPOCHS_TEACHER,
    lr: float = 1e-3,
    batch: int = 128,
    device: str = "cpu",
) -> TeacherNet:
    """Like train_teacher but skips batches with <2 valid samples (handles small clusters)."""
    model = TeacherNet(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y_class, dtype=torch.long, device=device)
    valid_cpu = (yt >= 0).cpu().numpy()

    model.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            idx_v = idx[valid_cpu[idx]]
            if len(idx_v) < 2:   # BatchNorm1d requires >1 sample during training
                continue
            loss = F.cross_entropy(model(Xt[idx_v]), yt[idx_v])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_student_from_probs(
    X: np.ndarray,
    y_reg: np.ndarray,
    teacher_probs: np.ndarray | None,   # (N, n_classes), or None for baseline
    n_epochs: int = N_EPOCHS_STUDENT,
    lr: float = 1e-3,
    batch: int = 128,
    alpha: float = ALPHA,
    temperature: float = TEMPERATURE,
    device: str = "cpu",
) -> StudentNet:
    """Train student with pre-computed (optionally per-sample) teacher soft targets."""
    model = StudentNet(X.shape[1], n_targets=y_reg.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yr = torch.tensor(y_reg, dtype=torch.float32, device=device)
    tp = (
        torch.tensor(teacher_probs, dtype=torch.float32, device=device)
        if teacher_probs is not None else None
    )

    model.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            if len(idx) < 2:   # BatchNorm1d requires >1 sample during training
                continue
            reg_pred, soft_pred = model(Xt[idx])
            task_loss = F.mse_loss(reg_pred, yr[idx])
            if tp is not None:
                sl = F.log_softmax(soft_pred / temperature, dim=1)
                kd = F.kl_div(sl, tp[idx], reduction="batchmean") * temperature**2
                loss = (1 - alpha) * task_loss + alpha * kd
            else:
                loss = task_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_student_per_sample_alpha(
    X: np.ndarray,
    y_reg: np.ndarray,
    teacher_probs: np.ndarray,       # (N, n_classes)
    sample_alphas: np.ndarray,       # (N,) per-sample α values
    n_epochs: int = N_EPOCHS_STUDENT,
    lr: float = 1e-3,
    batch: int = 128,
    temperature: float = TEMPERATURE,
    device: str = "cpu",
) -> StudentNet:
    """Train student with per-sample distillation weight α (Strategy C)."""
    model = StudentNet(X.shape[1], n_targets=y_reg.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yr = torch.tensor(y_reg, dtype=torch.float32, device=device)
    tp = torch.tensor(teacher_probs, dtype=torch.float32, device=device)
    sa = torch.tensor(sample_alphas, dtype=torch.float32, device=device)

    model.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            if len(idx) < 2:   # BatchNorm1d requires >1 sample during training
                continue
            reg_pred, soft_pred = model(Xt[idx])
            # Per-sample task loss
            task_per = F.mse_loss(reg_pred, yr[idx], reduction="none").mean(dim=1)
            # Per-sample KD loss
            sl = F.log_softmax(soft_pred / temperature, dim=1)
            kd_per = F.kl_div(sl, tp[idx], reduction="none").sum(dim=1) * temperature**2
            a = sa[idx]
            loss = ((1 - a) * task_per + a * kd_per).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _eval(model: StudentNet, X_test: np.ndarray, y_test: np.ndarray, device: str) -> dict:
    """Return valence/arousal R² and Pearson r on the test fold."""
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred, _ = model(Xt)
    p = pred.cpu().numpy()
    out = {}
    for i, tgt in enumerate(("valence", "arousal")):
        yt, yp = y_test[:, i], p[:, i]
        out[f"{tgt}_r2"] = float(r2_score(yt, yp))
        out[f"{tgt}_pearson"] = float(pearsonr(yt, yp)[0])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Per-condition runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(X_tr, X_te, y_tr, y_te, **_) -> dict:
    """Condition 0 — Student only, no distillation."""
    device = _.get("device", "cpu")
    student = _train_student_from_probs(X_tr, y_tr, teacher_probs=None, device=device)
    return _eval(student, X_te, y_te, device)


def run_global_kd(X_tr, X_te, y_tr, y_te, y_tr_class, device, **_) -> dict:
    """Condition 1 — Global teacher KD (no clustering)."""
    alpha       = _.get("alpha",       ALPHA)
    temperature = _.get("temperature", TEMPERATURE)
    teacher = train_teacher(X_tr, y_tr_class, n_epochs=N_EPOCHS_TEACHER, device=device)
    tp = _teacher_probs(teacher, X_tr, temperature, device)
    student = _train_student_from_probs(X_tr, y_tr, tp, alpha=alpha,
                                        temperature=temperature, device=device)
    return _eval(student, X_te, y_te, device)


def run_strategy_a(
    X_tr, X_te, y_tr, y_te, y_tr_class, device,
    cluster_tr, cluster_te, **_
) -> dict:
    """Strategy A — One teacher per cluster; global fallback if <5 physio samples."""
    alpha       = _.get("alpha",       ALPHA)
    temperature = _.get("temperature", TEMPERATURE)
    global_teacher = train_teacher(X_tr, y_tr_class, n_epochs=N_EPOCHS_TEACHER, device=device)
    global_tp = _teacher_probs(global_teacher, X_tr, temperature, device)

    assembled_tp = global_tp.copy()   # start with global, override per cluster

    for c in range(N_CLUSTERS):
        mask = cluster_tr == c
        n_physio = int((y_tr_class[mask] >= 0).sum())
        if mask.sum() < MIN_PHYSIO_CLUSTER or n_physio < MIN_PHYSIO_CLUSTER:
            log.debug("Strategy A: cluster %d — %d physio samples → global fallback", c, n_physio)
            continue
        teacher_c = _train_cluster_teacher(X_tr[mask], y_tr_class[mask],
                                       n_epochs=N_EPOCHS_TEACHER, device=device)
        assembled_tp[mask] = _teacher_probs(teacher_c, X_tr[mask], temperature, device)

    student = _train_student_from_probs(X_tr, y_tr, assembled_tp, alpha=alpha,
                                        temperature=temperature, device=device)
    return _eval(student, X_te, y_te, device)


def run_strategy_b(
    X_tr, X_te, y_tr, y_te, y_tr_class, device,
    cluster_b_tr, cluster_b_te, **_
) -> dict:
    """Strategy B — 4-dim one-hot cluster ID appended to student input (no leakage)."""
    alpha       = _.get("alpha",       ALPHA)
    temperature = _.get("temperature", TEMPERATURE)

    def one_hot(clusters: np.ndarray, n: int = N_CLUSTERS) -> np.ndarray:
        oh = np.zeros((len(clusters), n), dtype=np.float32)
        oh[np.arange(len(clusters)), clusters] = 1.0
        return oh

    X_tr_aug = np.concatenate([X_tr, one_hot(cluster_b_tr)], axis=1)
    X_te_aug = np.concatenate([X_te, one_hot(cluster_b_te)], axis=1)

    teacher = train_teacher(X_tr, y_tr_class, n_epochs=N_EPOCHS_TEACHER, device=device)
    tp = _teacher_probs(teacher, X_tr, temperature, device)

    student = _train_student_from_probs(X_tr_aug, y_tr, tp, alpha=alpha,
                                        temperature=temperature, device=device)
    return _eval(student, X_te_aug, y_te, device)


def run_strategy_c(
    X_tr, X_te, y_tr, y_te, y_tr_class, device,
    cluster_tr, cluster_te, **_
) -> dict:
    """Strategy C — Per-sample α proportional to cluster consistency (low within-var → high α)."""
    alpha       = _.get("alpha",       ALPHA)
    temperature = _.get("temperature", TEMPERATURE)

    cluster_std = np.ones(N_CLUSTERS, dtype=np.float32)
    for c in range(N_CLUSTERS):
        mask = cluster_tr == c
        if mask.sum() > 1:
            cluster_std[c] = float(y_tr[mask].std()) or 1.0

    consistency = 1.0 / np.where(cluster_std > 1e-6, cluster_std, 1e-6)
    cluster_alpha = np.clip(alpha * consistency / consistency.mean(), 0.05, 0.95)
    log.debug("Strategy C cluster alphas: %s", np.round(cluster_alpha, 3))

    sample_alphas = cluster_alpha[cluster_tr].astype(np.float32)

    teacher = train_teacher(X_tr, y_tr_class, n_epochs=N_EPOCHS_TEACHER, device=device)
    tp = _teacher_probs(teacher, X_tr, temperature, device)

    student = _train_student_per_sample_alpha(X_tr, y_tr, tp, sample_alphas,
                                              temperature=temperature, device=device)
    return _eval(student, X_te, y_te, device)


def run_strategy_d(
    X_tr, X_te, y_tr, y_te, y_tr_class, device,
    data_tr: pd.DataFrame, feat_cols: list[str], **_
) -> dict:
    """Strategy D — Cluster-shrunk self-report labels + global teacher KD."""
    alpha       = _.get("alpha",       ALPHA)
    temperature = _.get("temperature", TEMPERATURE)
    try:
        relabeled, _ = build_cluster_relabel_candidate(
            data_tr.copy(),
            id_column="participant",
            axes=["rating_valence", "rating_arousal"],
            feature_prefixes=None,
            feature_columns=feat_cols,
            reducer="tsne",
            n_clusters=N_CLUSTERS,
            tsne_perplexity=TSNE_PERPLEXITY,
            kmeans_n_init=10,
            seed=SEED,
        )
        rv = relabeled["relabel_rating_valence"].fillna(relabeled.get("raw_rating_valence", pd.Series(np.nan))).values
        ra = relabeled["relabel_rating_arousal"].fillna(relabeled.get("raw_rating_arousal", pd.Series(np.nan))).values

        if len(rv) == len(y_tr):
            y_tr_relabeled = np.stack([rv, ra], axis=1).astype(np.float32)
            # Final NaN fallback to original labels
            nan_mask = ~np.isfinite(y_tr_relabeled).all(axis=1)
            y_tr_relabeled[nan_mask] = y_tr[nan_mask]
        else:
            log.warning("Strategy D: row count mismatch (%d vs %d), using raw labels",
                        len(rv), len(y_tr))
            y_tr_relabeled = y_tr
    except Exception as exc:
        log.warning("Strategy D: relabeling failed (%s) — using raw labels", exc)
        y_tr_relabeled = y_tr

    teacher = train_teacher(X_tr, y_tr_class, n_epochs=N_EPOCHS_TEACHER, device=device)
    tp = _teacher_probs(teacher, X_tr, temperature, device)
    student = _train_student_from_probs(X_tr, y_tr_relabeled, tp, alpha=alpha,
                                        temperature=temperature, device=device)
    return _eval(student, X_te, y_te, device)


# ═══════════════════════════════════════════════════════════════════════════════
# Results table
# ═══════════════════════════════════════════════════════════════════════════════

_CONDITION_DISPLAY = {
    "0_Baseline":         "0  Baseline (no KD)",
    "1_Global_KD":        "1  Global KD",
    "A_Cluster_Teachers": "A  Cluster-Specific Teachers",
    "B_Cluster_Feature":  "B  Cluster ID Feature",
    "C_Cluster_Alpha":    "C  Cluster-Aware Alpha",
    "D_Relabelled_KD":    "D  Relabelled Targets + KD",
}


def _print_table(all_metrics: dict[str, list[dict]], conditions: list[str]) -> None:
    W = 82
    sep = "═" * W
    thin = "─" * W
    print()
    print(sep)
    print(f"  Cluster-Distillation Comparison — {N_FOLDS}-fold CV  "
          f"(α={ALPHA:.2f}, T={TEMPERATURE:.1f}, k={N_CLUSTERS} clusters)")
    print(sep)
    hdr = f"  {'Condition':<30}  {'Val R²':>8}  {'Aro R²':>8}  {'Val r':>8}  {'Aro r':>8}"
    print(hdr)
    print(thin)

    rows: list[tuple[str, float, float, float, float]] = []
    for cond in conditions:
        folds = all_metrics[cond]
        vr2  = float(np.mean([f["valence_r2"]      for f in folds]))
        ar2  = float(np.mean([f["arousal_r2"]       for f in folds]))
        vpr  = float(np.mean([f["valence_pearson"]  for f in folds]))
        apr  = float(np.mean([f["arousal_pearson"]  for f in folds]))
        rows.append((cond, vr2, ar2, vpr, apr))
        label = _CONDITION_DISPLAY.get(cond, cond)
        print(f"  {label:<30}  {vr2:>8.4f}  {ar2:>8.4f}  {vpr:>8.4f}  {apr:>8.4f}")

    print(sep)

    # Best per metric
    metric_keys = [
        ("Val R²",       1),
        ("Aro R²",       2),
        ("Val Pearson r", 3),
        ("Aro Pearson r", 4),
    ]
    print()
    print("  Best per metric:")
    for label, col_idx in metric_keys:
        best = max(rows, key=lambda r: r[col_idx])
        disp = _CONDITION_DISPLAY.get(best[0], best[0])
        print(f"    {label:<16} → {disp:<30}  ({best[col_idx]:+.4f})")

    # KD gain vs baseline
    print()
    print("  Δ vs Baseline (Condition 0):")
    base_row = next(r for r in rows if r[0] == "0_Baseline")
    for cond, vr2, ar2, vpr, apr in rows[1:]:
        disp = _CONDITION_DISPLAY.get(cond, cond)
        dvr2 = vr2 - base_row[1]
        dar2 = ar2 - base_row[2]
        print(f"    {disp:<30}  ΔValR²={dvr2:+.4f}  ΔAroR²={dar2:+.4f}")

    print(sep)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--protocol",    default="protocolaudio",
                   help="NEURO sub-protocol folder (default: protocolaudio). "
                        "Use 'all' to load all 5 protocols simultaneously "
                        "(protocolimage, protocolaudio, protocolbel, protocoldanone, protocolNRJ).")
    p.add_argument("--folds",       type=int,   default=N_FOLDS,
                   help=f"Number of KFold CV splits (default: {N_FOLDS})")
    p.add_argument("--alpha",       type=float, default=ALPHA,
                   help=f"Base KD loss weight α (default: {ALPHA})")
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help=f"Softmax temperature for soft targets (default: {TEMPERATURE})")
    p.add_argument("--output-dir",  default=None,
                   help="If set, write results JSON to <output-dir>/cluster_distillation_results.json")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    log.info("Device: %s | N_clusters=%d | perplexity=%.0f | α=%.2f | T=%.1f",
             device, N_CLUSTERS, TSNE_PERPLEXITY, args.alpha, args.temperature)

    # ── Load data ─────────────────────────────────────────────────────────────
    data, feat_cols = load_data(args.protocol, delay_sec=0)

    X_raw  = data[feat_cols].fillna(0).values.astype(np.float32)
    y_reg  = data[["rating_valence", "rating_arousal"]].values.astype(np.float32)
    y_class = np.where(
        data["arousal_three_all"].notna(),
        data["arousal_three_all"].fillna(-1).values.astype(int),
        -1,
    )
    log.info("Physio label coverage: %d%%  |  total rows: %d  |  participants: %d",
             int(100 * (y_class >= 0).mean()), len(data), data["participant"].nunique())

    # ── Global clusters (strategies A, C, D) ──────────────────────────────────
    log.info("Computing global t-SNE + KMeans clusters (k=%d, perplexity=%.0f)…",
             N_CLUSTERS, TSNE_PERPLEXITY)
    global_clusters = compute_global_clusters(data, feat_cols)
    for c in range(N_CLUSTERS):
        m = global_clusters == c
        log.info("  Cluster %d: n=%d  mean_arousal=%.3f  mean_valence=%.3f",
                 c, m.sum(),
                 float(y_reg[m, 1].mean()) if m.sum() else float("nan"),
                 float(y_reg[m, 0].mean()) if m.sum() else float("nan"))

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    conditions = [
        "0_Baseline",
        "1_Global_KD",
        "A_Cluster_Teachers",
        "B_Cluster_Feature",
        "C_Cluster_Alpha",
        "D_Relabelled_KD",
    ]
    all_metrics: dict[str, list[dict]] = {c: [] for c in conditions}

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_raw)):
        log.info("── Fold %d / %d ──────────────────────────────────────────────", fold + 1, N_FOLDS)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[tr_idx])
        X_te = sc.transform(X_raw[te_idx])
        y_tr      = y_reg[tr_idx]
        y_te      = y_reg[te_idx]
        y_tr_cl   = y_class[tr_idx]

        cluster_tr = global_clusters[tr_idx]
        cluster_te = global_clusters[te_idx]

        # Strategy B: per-fold cluster (no leakage)
        log.info("  [B] Computing fold-local clusters for Strategy B…")
        fold_clusters = compute_fold_clusters(data.iloc[tr_idx].reset_index(drop=True),
                                              data, feat_cols)
        cluster_b_tr = fold_clusters[tr_idx]
        cluster_b_te = fold_clusters[te_idx]

        data_tr = data.iloc[tr_idx].reset_index(drop=True)

        shared = dict(
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            y_tr_class=y_tr_cl, device=device,
            cluster_tr=cluster_tr, cluster_te=cluster_te,
            cluster_b_tr=cluster_b_tr, cluster_b_te=cluster_b_te,
            data_tr=data_tr, feat_cols=feat_cols,
            alpha=args.alpha, temperature=args.temperature,
        )

        log.info("  [0] Baseline…")
        all_metrics["0_Baseline"].append(run_baseline(**shared))

        log.info("  [1] Global KD…")
        all_metrics["1_Global_KD"].append(run_global_kd(**shared))

        log.info("  [A] Cluster-specific teachers…")
        all_metrics["A_Cluster_Teachers"].append(run_strategy_a(**shared))

        log.info("  [B] Cluster-ID feature…")
        all_metrics["B_Cluster_Feature"].append(run_strategy_b(**shared))

        log.info("  [C] Cluster-aware alpha…")
        all_metrics["C_Cluster_Alpha"].append(run_strategy_c(**shared))

        log.info("  [D] Cluster-relabelled targets + KD…")
        all_metrics["D_Relabelled_KD"].append(run_strategy_d(**shared))

        # Free GPU/MPS memory between folds
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        log.info("  Fold %d complete.", fold + 1)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_table(all_metrics, conditions)

    # ── Save results JSON ─────────────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict = {
            "config": {
                "protocol":    args.protocol,
                "folds":       args.folds,
                "alpha":       args.alpha,
                "temperature": args.temperature,
                "n_clusters":  N_CLUSTERS,
                "tsne_perplexity": TSNE_PERPLEXITY,
            },
            "results": {},
        }
        metrics_keys = ("valence_r2", "arousal_r2", "valence_pearson", "arousal_pearson")
        for cond in conditions:
            folds_data = all_metrics[cond]
            summary["results"][cond] = {
                k: round(float(np.mean([f[k] for f in folds_data])), 4)
                for k in metrics_keys
            }
        out_path = out_dir / "cluster_distillation_results.json"
        out_path.write_text(json.dumps(summary, indent=2))
        log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
