"""
cluster_relabel.py
-------------------
Cluster-then-relabel: unsupervised-cluster-informed shrinkage for continuous
affect labels (valence / arousal / any other numeric rating axis).

Motivation
----------
`empirical_validation.py` already compares each rating to a *predefined*
subgroup mean — crowd-wisdom-reliable vs. unreliable rows in
`crowd_wisdom_analysis`, or per-rater means in `annotator_bias_analysis` —
to flag label-quality issues. This module borrows that "compare a unit to
its group's central tendency" spirit but takes it one step further in two
ways: the group is *learned* from the sample's own feature vector (audio /
physio / video features, whatever the caller passes in) via PCA → t-SNE →
KMeans, rather than a predefined subgroup; and instead of only reporting
the group-mean gap, that gap is used to actually *shrink* the raw label
toward the learned cluster's trimmed mean.

The shrink is gated so a cluster can only move a label if the clustering
demonstrably explains variance on that specific axis — an ANOVA-style
eta-squared (between-cluster / total variance) — and only in proportion to
how many samples support the cluster (`n / (n + shrink_tau)`, the same
"more support → more trust" shape used for the sigmoid calibration floors
in `fusion_methods.CALIBRATION`). Samples whose robust (MAD-based)
deviation from their own cluster's centroid is large are flagged as
outliers rather than silently altered further.

This is diagnostic/experimental by construction: it never overwrites an
input rating column — every derived column is written under a
``cluster_*`` / ``relabel_*`` / ``is_outlier_*`` prefix alongside the raw
values, and the manifest carries ``promotion_status:
"unpromoted_experimental_candidate"`` plus explicit per-axis
informativeness gates, so a human has to decide whether (and how) any of
this should ever feed a production label pipeline.

How to run
----------
    python cluster_relabel.py \\
        --input path/to/features_and_ratings.parquet \\
        --feature_prefixes clip_,vmae_ \\
        --n_clusters 8 \\
        --reducer tsne \\
        --out runs/cluster_relabel_v1

    # physiology-feature variant — target the kind of wide physio columns
    # this repo already extracts in pipeline_valence_arousal.py's
    # _PHYSIO_COLS (meanRR_/RMSSD_/SDNN_/SCL_/SCR_/saccade_/pupil_ prefixes)
    python cluster_relabel.py \\
        --input path/to/physio_features.parquet \\
        --feature_prefixes SCL_,SCR_,RMSSD_,SDNN_ \\
        --id_column participant \\
        --n_clusters 6 \\
        --out runs/cluster_relabel_physio_v1

Requires scikit-learn (``sklearn.decomposition.PCA``, ``sklearn.manifold.
TSNE``, ``sklearn.cluster.KMeans``, ``sklearn.metrics``); imported lazily
inside the functions that need them, the same way this repo already
treats optional heavy dependencies (see ``features_multimedia.
cluster_and_label_layout``, ``features_physiology.bandpass_filter``), so
``--help`` and static import checks work even when scikit-learn isn't
installed in the current environment.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_AXES = ("valence", "arousal")


# ---------------------------------------------------------------------------
# Small IO / stats helpers
# ---------------------------------------------------------------------------

def _read_table(path: str | Path) -> pd.DataFrame:
    """Load a feature/label table from a parquet or csv file."""
    value = str(path)
    if value.lower().endswith(".parquet"):
        return pd.read_parquet(value)
    if value.lower().endswith(".csv"):
        return pd.read_csv(value, sep=None, engine="python")
    raise ValueError(f"unsupported table format: {path}")


def _first_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"{label} requires one of {list(candidates)}, found {list(frame.columns)}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def trimmed_mean(values: Iterable[float], trim_fraction: float = 0.1) -> float:
    """Mean of *values* after dropping the top/bottom *trim_fraction* of sorted values."""
    arr = np.asarray(list(values), dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    if not len(arr):
        return float("nan")
    count = int(np.floor(len(arr) * min(max(trim_fraction, 0.0), 0.45)))
    if len(arr) - 2 * count < 3:
        count = 0
    return float(arr[count: len(arr) - count].mean())


def _robust_mad(values: np.ndarray) -> tuple[float, float]:
    """Median and 1.4826*MAD (a normal-consistent robust scale)."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return center, scale


# ---------------------------------------------------------------------------
# Feature selection + standardization
# ---------------------------------------------------------------------------

def select_feature_columns(
    frame: pd.DataFrame,
    feature_prefixes: Iterable[str] | None,
    feature_columns: Iterable[str] | None,
) -> list[str]:
    """Resolve the numeric columns to cluster on, from an explicit list or a set of prefixes."""
    if feature_columns:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"requested feature columns not found: {missing}")
        return list(feature_columns)
    prefixes = tuple(feature_prefixes or ())
    if not prefixes:
        raise ValueError("either feature_columns or feature_prefixes must select at least one column")
    selected = [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not selected:
        raise ValueError(f"no numeric columns matched feature prefixes {prefixes}")
    return selected


def standardize_features(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Z-score each feature column; missing values are imputed with the column mean (-> 0 after scaling)."""
    raw = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    column_mean = np.nanmean(raw, axis=0)
    column_mean = np.where(np.isfinite(column_mean), column_mean, 0.0)
    filled = np.where(np.isfinite(raw), raw, column_mean)
    std = np.nanstd(filled, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    standardized = (filled - column_mean) / std
    coverage = float(np.isfinite(raw).mean())
    return standardized, {"n_features": len(feature_columns), "finite_value_coverage": coverage}


# ---------------------------------------------------------------------------
# Dimensionality reduction + clustering (sklearn imported lazily)
# ---------------------------------------------------------------------------

def reduce_dimensionality(
    X: np.ndarray,
    method: str,
    embed_dims: int,
    pca_components: int,
    tsne_perplexity: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reduce *X* with PCA pre-whitening followed by t-SNE, plain PCA, or a no-op."""
    info: dict[str, object] = {"method": method, "embed_dims": embed_dims}
    if method == "none":
        return X, info

    from sklearn.decomposition import PCA

    n_components = int(min(pca_components, max(1, min(X.shape) - 1)))
    pca = PCA(n_components=n_components, random_state=seed)
    pre = pca.fit_transform(X)
    info["pca_components_used"] = n_components
    info["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))

    if method == "pca":
        return pre[:, :embed_dims], info

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(tsne_perplexity, max(5.0, (X.shape[0] - 1) / 3.0)))
        tsne = TSNE(
            n_components=embed_dims,
            perplexity=perplexity,
            init="pca",
            random_state=seed,
            learning_rate="auto",
        )
        embedded = tsne.fit_transform(pre)
        info["tsne_perplexity_used"] = perplexity
        return embedded, info

    raise ValueError(f"unknown reducer method: {method}")


def cluster_embedding(
    embedding: np.ndarray,
    n_clusters: int,
    n_init: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run KMeans on *embedding* and report cluster counts + silhouette diagnostics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    k = int(min(n_clusters, max(1, embedding.shape[0] - 1)))
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = kmeans.fit_predict(embedding)
    info: dict[str, object] = {"n_clusters_requested": n_clusters, "n_clusters_used": k}
    if k > 1 and embedding.shape[0] > k:
        info["silhouette_score"] = float(silhouette_score(embedding, labels))
        per_sample_silhouette = silhouette_samples(embedding, labels)
    else:
        info["silhouette_score"] = float("nan")
        per_sample_silhouette = np.full(embedding.shape[0], np.nan)
    return labels, {**info, "_silhouette_samples": per_sample_silhouette}


# ---------------------------------------------------------------------------
# Relabeling: shrink each sample toward its cluster centroid
# ---------------------------------------------------------------------------

def _eta_squared(values: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Fraction of an axis's variance explained by cluster membership (ANOVA eta^2)."""
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return float("nan")
    v = values[valid]
    c = cluster_ids[valid]
    grand_mean = float(np.mean(v))
    total_ss = float(np.sum((v - grand_mean) ** 2))
    if total_ss <= 1e-12:
        return float("nan")
    between_ss = 0.0
    for cluster in np.unique(c):
        members = v[c == cluster]
        between_ss += len(members) * (float(np.mean(members)) - grand_mean) ** 2
    return float(np.clip(between_ss / total_ss, 0.0, 1.0))


def relabel_with_clusters(
    frame: pd.DataFrame,
    axes: list[str],
    cluster_ids: np.ndarray,
    trim_fraction: float,
    shrink_tau: float,
    outlier_z_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute per-cluster centroids and shrink each sample's label toward its cluster's mean.

    ``relabel_<axis> = (1 - shrink_weight) * raw_<axis> + shrink_weight * cluster_trimmed_mean_<axis>``
    where ``shrink_weight = eta_squared_<axis> * cluster_n / (cluster_n + shrink_tau)`` — a
    ``support / (support + tau)`` reliability-shrinkage shape, scaled down further when the
    clustering explains little of that axis's variance (``eta_squared`` near 0 => almost no
    shrink is applied, so the mechanism cannot manufacture structure absent from the features).
    """
    out = frame.copy()
    out["cluster_id"] = cluster_ids.astype(int)
    cluster_sizes = out.groupby("cluster_id")["cluster_id"].transform("size")
    out["cluster_n"] = cluster_sizes.astype(int)

    axis_diagnostics: dict[str, dict[str, object]] = {}
    for axis in axes:
        raw = pd.to_numeric(out[axis], errors="coerce").to_numpy(dtype=float)
        out[f"raw_{axis}"] = raw

        cluster_mean = np.full(len(out), np.nan, dtype=float)
        cluster_median = np.full(len(out), np.nan, dtype=float)
        residual = np.full(len(out), np.nan, dtype=float)
        robust_z = np.full(len(out), np.nan, dtype=float)
        for cluster in np.unique(cluster_ids):
            member_mask = cluster_ids == cluster
            member_values = raw[member_mask]
            centroid = trimmed_mean(member_values, trim_fraction)
            median = float(np.nanmedian(member_values)) if np.isfinite(member_values).any() else float("nan")
            cluster_mean[member_mask] = centroid
            cluster_median[member_mask] = median
            residual[member_mask] = member_values - centroid
            _, mad_scale = _robust_mad(member_values)
            scale = mad_scale if np.isfinite(mad_scale) and mad_scale > 1e-6 else float(np.nanstd(member_values) or 1.0)
            robust_z[member_mask] = (member_values - centroid) / max(scale, 1e-6)

        eta_squared = _eta_squared(raw, cluster_ids)
        support_shrink = cluster_sizes.to_numpy(dtype=float) / (cluster_sizes.to_numpy(dtype=float) + shrink_tau)
        shrink_weight = np.clip((eta_squared if np.isfinite(eta_squared) else 0.0) * support_shrink, 0.0, 1.0)
        relabel = np.where(
            np.isfinite(raw) & np.isfinite(cluster_mean),
            (1.0 - shrink_weight) * raw + shrink_weight * cluster_mean,
            raw,
        )
        is_outlier = np.isfinite(robust_z) & (np.abs(robust_z) > outlier_z_threshold)

        out[f"cluster_mean_{axis}"] = cluster_mean
        out[f"cluster_median_{axis}"] = cluster_median
        out[f"residual_{axis}"] = residual
        out[f"robust_z_{axis}"] = robust_z
        out[f"shrink_weight_{axis}"] = shrink_weight
        out[f"relabel_{axis}"] = relabel
        out[f"is_outlier_{axis}"] = is_outlier

        axis_diagnostics[axis] = {
            "eta_squared_cluster_vs_axis": None if not np.isfinite(eta_squared) else eta_squared,
            "mean_shrink_weight": float(np.nanmean(shrink_weight)),
            "n_outliers": int(np.nansum(is_outlier)),
            "pct_outliers": float(np.nanmean(is_outlier)) if len(is_outlier) else float("nan"),
            "clusters_informative": bool(np.isfinite(eta_squared) and eta_squared >= 0.05),
        }
    return out, axis_diagnostics


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_cluster_relabel_candidate(
    frame: pd.DataFrame,
    id_column: str,
    axes: list[str],
    feature_prefixes: list[str] | None,
    feature_columns: list[str] | None,
    reducer: str = "tsne",
    embed_dims: int = 2,
    pca_components: int = 50,
    tsne_perplexity: float = 30.0,
    n_clusters: int = 8,
    kmeans_n_init: int = 20,
    trim_fraction: float = 0.1,
    shrink_tau: float = 20.0,
    outlier_z_threshold: float = 3.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cluster samples on their own features, then shrink each label toward its cluster's mean.

    Args:
        frame: Per-sample feature + label table (e.g. one row per video or per
            participant/stimulus pair).
        id_column: Row identifier column, e.g. ``"video"`` or ``"participant"``.
        axes: Label columns to relabel, e.g. ``["valence", "arousal"]``.
        feature_prefixes: Column-name prefixes selecting numeric clustering features.
        feature_columns: Optional explicit feature column list (overrides *feature_prefixes*).
        reducer: ``"tsne"``, ``"pca"``, or ``"none"``.
        embed_dims: Dimensionality of the reduced embedding used for KMeans.
        pca_components: PCA pre-whitening dimensionality applied before t-SNE.
        tsne_perplexity: t-SNE perplexity (clipped to a safe range for small samples).
        n_clusters: Requested KMeans cluster count (clipped to ``n_samples - 1``).
        kmeans_n_init: Number of KMeans initializations.
        trim_fraction: Trim fraction for each cluster's trimmed-mean centroid.
        shrink_tau: Support-shrinkage tau; larger values shrink small clusters less.
        outlier_z_threshold: Robust (MAD) z-score threshold flagging within-cluster outliers.
        seed: Random seed for PCA/t-SNE/KMeans.

    Returns:
        ``(relabeled_frame, manifest)`` — the input frame augmented with
        ``cluster_*`` / ``relabel_*`` / ``is_outlier_*`` columns, and a manifest
        dict describing the reduction, clustering, and per-axis gates.
    """
    work = frame.copy()
    present_axes = [axis for axis in axes if axis in work.columns]
    if not present_axes:
        raise ValueError(f"none of the requested axes {axes} are present in the input table")
    work = work.dropna(subset=[id_column]).reset_index(drop=True)

    feature_cols = select_feature_columns(work, feature_prefixes, feature_columns)
    X, feature_report = standardize_features(work, feature_cols)
    embedding, reduce_report = reduce_dimensionality(
        X, method=reducer, embed_dims=embed_dims, pca_components=pca_components,
        tsne_perplexity=tsne_perplexity, seed=seed,
    )
    cluster_ids, cluster_report = cluster_embedding(embedding, n_clusters=n_clusters, n_init=kmeans_n_init, seed=seed)
    per_sample_silhouette = cluster_report.pop("_silhouette_samples")

    relabeled, axis_diagnostics = relabel_with_clusters(
        work, present_axes, cluster_ids, trim_fraction=trim_fraction,
        shrink_tau=shrink_tau, outlier_z_threshold=outlier_z_threshold,
    )
    relabeled["silhouette_sample"] = per_sample_silhouette
    if embedding.shape[1] >= 1:
        for dim in range(min(embedding.shape[1], embed_dims)):
            relabeled[f"embed_{dim + 1}"] = embedding[:, dim]

    manifest: dict[str, object] = {
        "module": "cluster_relabel.py",
        "promotion_status": "unpromoted_experimental_candidate",
        "method": "pca_prewhiten -> t-sne/pca embedding -> kmeans clustering -> reliability-shrunk relabel toward cluster centroid",
        "n_rows_in": int(len(frame)),
        "n_rows_used": int(len(work)),
        "id_column": id_column,
        "axes_present": present_axes,
        "axes_requested_but_missing": sorted(set(axes) - set(present_axes)),
        "feature_columns_selected": feature_cols,
        "feature_report": feature_report,
        "reduction": reduce_report,
        "clustering": cluster_report,
        "axis_diagnostics": axis_diagnostics,
        "hyperparameters": {
            "reducer": reducer,
            "embed_dims": embed_dims,
            "pca_components": pca_components,
            "tsne_perplexity": tsne_perplexity,
            "n_clusters_requested": n_clusters,
            "kmeans_n_init": kmeans_n_init,
            "trim_fraction": trim_fraction,
            "shrink_tau": shrink_tau,
            "outlier_z_threshold": outlier_z_threshold,
            "seed": seed,
        },
        "gates": {
            f"clusters_informative_{axis}": axis_diagnostics[axis]["clusters_informative"]
            for axis in present_axes
        },
        "warning": (
            "This is a diagnostic/experimental relabeling candidate only. It must never be treated "
            "as a finalized label and must never overwrite an existing production rating column. "
            "Promotion, if ever warranted, requires an explicit separate review evaluated against "
            "held-out, sample-disjoint folds."
        ),
    }
    manifest["gates"]["any_axis_cluster_informative"] = any(manifest["gates"].values())
    return relabeled, manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Per-sample feature+label table (parquet or csv).")
    p.add_argument("--out", required=True, help="Output directory for the candidate artifacts.")
    p.add_argument("--id_column", default=None, help="Row identifier column (default: auto-detect video/participant/id).")
    p.add_argument("--axes", default=",".join(DEFAULT_AXES), help="Comma-separated label columns to relabel.")
    p.add_argument("--feature_prefixes", default=None, help="Comma-separated column-name prefixes used as clustering features.")
    p.add_argument("--feature_columns", default=None, help="Explicit comma-separated feature column list (overrides --feature_prefixes).")
    p.add_argument("--reducer", choices=("tsne", "pca", "none"), default="tsne")
    p.add_argument("--embed_dims", type=int, default=2)
    p.add_argument("--pca_components", type=int, default=50, help="PCA pre-whitening dimensionality before t-SNE.")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--kmeans_n_init", type=int, default=20)
    p.add_argument("--trim_fraction", type=float, default=0.1, help="Trim fraction for the per-cluster trimmed-mean centroid.")
    p.add_argument("--shrink_tau", type=float, default=20.0, help="Support-shrinkage tau; larger = less shrink for small clusters.")
    p.add_argument("--outlier_z_threshold", type=float, default=3.0, help="Robust (MAD) z-score threshold flagging within-cluster label outliers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing candidate directory.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    out_dir = Path(args.out)
    candidate_path = out_dir / "cluster_relabel_candidate.parquet"
    if candidate_path.exists() and not args.force:
        logger.info("SKIP — %s exists (use --force)", candidate_path)
        return

    frame = _read_table(args.input)
    id_column = args.id_column or _first_column(frame, ("video", "participant", "media_id", "id"), "input table")
    axes = [axis.strip() for axis in args.axes.split(",") if axis.strip()]
    feature_columns = (
        [column.strip() for column in args.feature_columns.split(",") if column.strip()]
        if args.feature_columns
        else None
    )
    feature_prefixes = (
        None
        if feature_columns
        else [prefix.strip() for prefix in (args.feature_prefixes or "").split(",") if prefix.strip()]
    )

    relabeled, manifest = build_cluster_relabel_candidate(
        frame,
        id_column=id_column,
        axes=axes,
        feature_prefixes=feature_prefixes,
        feature_columns=feature_columns,
        reducer=args.reducer,
        embed_dims=args.embed_dims,
        pca_components=args.pca_components,
        tsne_perplexity=args.tsne_perplexity,
        n_clusters=args.n_clusters,
        kmeans_n_init=args.kmeans_n_init,
        trim_fraction=args.trim_fraction,
        shrink_tau=args.shrink_tau,
        outlier_z_threshold=args.outlier_z_threshold,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    relabeled.to_parquet(candidate_path, index=False)
    relabeled.to_csv(out_dir / "cluster_relabel_candidate.csv", index=False)
    _write_json(out_dir / "cluster_relabel_manifest.json", manifest)
    informative_axes = [a for a, ok in manifest["gates"].items() if ok and a != "any_axis_cluster_informative"]
    logger.info(
        "rows=%d clusters=%s informative_axes=%s out=%s",
        len(relabeled), manifest["clustering"].get("n_clusters_used"), informative_axes, out_dir,
    )


if __name__ == "__main__":
    main()
