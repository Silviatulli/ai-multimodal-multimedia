"""
empirical_validation.py
-----------------------
Empirical validation of aggregated labels and dataset-level quality checks.

Contains methods for:

    - Cross-field semantic overlap between annotation fields (CLIP-based)
    - Person presence and count statistics across a dataset
    - Aggregation of pairwise distance JSONs into dataset-level statistics
    - Label quality and inter-rater agreement utilities

Sources
-------
- cross_field_overlap, clip_topic_overlap  : ai-fresco-replicated/fresco/utils/validation/validation.py
- count_person_presence                    : ai-fresco-replicated/fresco/utils/validation/validation_count.py
- aggregate_distance_stats                 : ai-fresco-replicated/fresco/distance/dataset_level_distances.py
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Person-related terms used across object detection and panoptic vocabularies
_PERSON_SYNSET: frozenset[str] = frozenset({
    "person", "man", "woman", "girl", "boy", "child", "men", "women",
    "girls", "boys", "lady", "ladies", "people", "baby", "kid", "kids",
    "male", "female", "mother", "father", "daughter", "son", "family",
    "player", "actor", "soldiers", "students", "student", "bride",
    "graduates", "clown", "astronauts", "officer", "nun", "model",
})


# ---------------------------------------------------------------------------
# CLIP-based cross-field semantic overlap
# ---------------------------------------------------------------------------

def clip_topic_overlap(
    topics_a: list[str],
    topics_b: list[str],
    threshold: float = 0.9,
    device: str | None = None,
) -> dict[str, int]:
    """Compute topic overlap between two tag lists using CLIP text similarity.

    For each topic in *topics_a* we check whether any topic in *topics_b* is
    semantically similar (cosine similarity ≥ *threshold*).  Returns counts
    of topics exclusive to A, shared, and exclusive to B.

    Args:
        topics_a:   List of topic strings from the first annotation field.
        topics_b:   List of topic strings from the second annotation field.
        threshold:  Cosine similarity cutoff for two topics to be considered
                    equivalent (default 0.9).
        device:     Torch device string (``"cuda"`` / ``"cpu"`` / ``"mps"``).
                    Auto-detected when ``None``.

    Returns:
        ``{"only_a": int, "common": int, "only_b": int}``

    Requires ``clip`` (``pip install git+https://github.com/openai/CLIP.git``).
    """
    import torch
    import clip as _clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = _clip.load("ViT-L/14", device=device)

    def _is_common(t1: str, t2: str) -> bool:
        toks  = _clip.tokenize([t1, t2]).to(device)
        with torch.no_grad():
            feats = model.encode_text(toks)
        cos = torch.nn.CosineSimilarity(dim=0)(feats[0], feats[1])
        return float(cos) >= threshold

    set_a, set_b = set(topics_a), set(topics_b)
    only_a = common = only_b = 0

    for ta in set_a:
        if any(_is_common(ta, tb) for tb in set_b):
            common += 1
        else:
            only_a += 1

    for tb in set_b:
        if not any(_is_common(ta, tb) for ta in set_a):
            only_b += 1

    return {"only_a": only_a, "common": common, "only_b": only_b}


def cross_field_overlap(
    identikit_records: list[dict],
    field_a_path: str,
    field_b_path: str,
    threshold: float = 0.9,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure CLIP-based semantic overlap between two annotation fields across a dataset.

    *field_a_path* and *field_b_path* are dot-separated key paths into each
    identikit record dict, e.g.
    ``"figurative_level.general.main_topic_tags"`` (list field) or
    ``"plastic_level.topological_categories.semantic_palette.categories"``.

    Args:
        identikit_records: List of identikit dicts (FRESCO output).
        field_a_path:      Dot-separated path to the first list field.
        field_b_path:      Dot-separated path to the second list field.
        threshold:         CLIP cosine similarity threshold (default 0.9).
        device:            Torch device string.

    Returns:
        Aggregated ``{"only_a", "common", "only_b", "total", "pct_common"}`` dict.
    """
    def _get(record: dict, path: str) -> list[str]:
        val = record
        for key in path.split("."):
            val = val.get(key, {}) if isinstance(val, dict) else {}
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            return [t.strip() for t in val.split("|")]
        return []

    totals = {"only_a": 0, "common": 0, "only_b": 0}
    for rec in tqdm(identikit_records, desc="cross_field_overlap"):
        a = _get(rec, field_a_path)
        b = _get(rec, field_b_path)
        if not a or not b:
            continue
        counts = clip_topic_overlap(a, b, threshold=threshold, device=device)
        for k in totals:
            totals[k] += counts[k]

    total = sum(totals.values())
    pct_common = round(100 * totals["common"] / total, 2) if total else 0.0
    return {**totals, "total": total, "pct_common": pct_common}


# ---------------------------------------------------------------------------
# Person presence and count statistics
# ---------------------------------------------------------------------------

def count_person_presence(
    identikit_records: list[dict],
) -> dict[str, Any]:
    """Compute person presence and count statistics across an identikit dataset.

    Analyses face boxes, object categories, panoptic labels, semantic
    categories, topic tags, and captions — the same six signals used in the
    FRESCO validation script.

    Args:
        identikit_records: List of FRESCO identikit dicts.

    Returns:
        Dict with presence counts, mean counts, and face-count group histograms.
    """
    face_count = obj_count = pan_count = 0
    face_pres = obj_pres = pan_pres = sem_pres = tag_pres = cap_pres = 0
    n = len(identikit_records)
    face_groups = {"0_1": 0, "2": 0, "3_6": 0, "7_12": 0, "13_30": 0, "31_inf": 0}

    def _group(faces: int) -> str:
        if faces <= 1:  return "0_1"
        if faces == 2:  return "2"
        if faces <= 6:  return "3_6"
        if faces <= 12: return "7_12"
        if faces <= 30: return "13_30"
        return "31_inf"

    for rec in tqdm(identikit_records, desc="count_person_presence"):
        try:
            faces = len(
                rec["figurative_level"]["content_participants"]
                    ["single_person_face_attributes"]["face_boxes"]
            )
        except (KeyError, TypeError):
            faces = 0

        try:
            objs = rec["figurative_level"]["content_participants"]["objects_goals"]["objects_categories"]
            objs_count = sum(1 for o in objs if o in _PERSON_SYNSET)
        except (KeyError, TypeError):
            objs_count = 0

        try:
            pan = rec["plastic_level"]["topological_categories"]["MC_avg_depth_vs_background"]["main_characters_labels"]
            pan_count_rec = sum(1 for o in pan if o in _PERSON_SYNSET)
        except (KeyError, TypeError):
            pan_count_rec = 0

        try:
            sem = rec["plastic_level"]["topological_categories"]["semantic_palette"]["categories"]
            sem_pres_rec = sum(1 for o in sem if o in _PERSON_SYNSET)
        except (KeyError, TypeError):
            sem_pres_rec = 0

        try:
            tags = [t.strip() for t in
                    rec["figurative_level"]["general"]["main_topic_tags"].split("|")]
            tags_pres_rec = sum(1 for p in _PERSON_SYNSET for t in tags if p in t)
        except (KeyError, TypeError):
            tags_pres_rec = 0

        try:
            cap_words = rec["figurative_level"]["action"]["single_action_caption"].split()
            cap_pres_rec = sum(1 for p in _PERSON_SYNSET for w in cap_words if p in w)
        except (KeyError, TypeError):
            cap_pres_rec = 0

        face_count += faces
        obj_count  += objs_count
        pan_count  += pan_count_rec
        face_pres  += 1 if faces        > 0 else 0
        obj_pres   += 1 if objs_count   > 0 else 0
        pan_pres   += 1 if pan_count_rec > 0 else 0
        sem_pres   += 1 if sem_pres_rec > 0 else 0
        tag_pres   += 1 if tags_pres_rec > 0 else 0
        cap_pres   += 1 if cap_pres_rec > 0 else 0
        face_groups[_group(faces)] += 1

    return {
        "n_records":          n,
        "face_count":         face_count,
        "obj_person_count":   obj_count,
        "pan_person_count":   pan_count,
        "face_presence":      face_pres,
        "obj_person_presence":obj_pres,
        "pan_person_presence":pan_pres,
        "sem_person_presence":sem_pres,
        "tags_person_presence":tag_pres,
        "caption_person_presence": cap_pres,
        "face_groups":        face_groups,
    }


# ---------------------------------------------------------------------------
# Dataset-level distance aggregation
# ---------------------------------------------------------------------------

def aggregate_distance_stats(distance_dir: str) -> dict[str, dict[str, float]]:
    """Aggregate per-pair pairwise distance JSON files into dataset-level stats.

    Reads all ``*.json`` files under *distance_dir* (recursively), accumulates
    per-metric lists, and returns mean / std / min / max for each metric.

    The expected JSON schema mirrors FRESCO's ``distance_evaluator.py`` output,
    with keys like ``"Overall_distance"``, ``"plastic_level"``,
    ``"figurative_level"``, and ``"narrative_level"``.

    Args:
        distance_dir: Directory containing per-pair distance JSON files.

    Returns:
        ``{metric_name: {"mean": float, "std": float, "min": float, "max": float}}``
    """
    json_files = glob.glob(
        str(Path(distance_dir) / "**" / "*.json"), recursive=True
    )
    if not json_files:
        logger.warning("aggregate_distance_stats: no JSON files found in %s", distance_dir)
        return {}

    buckets: dict[str, list[float]] = {}

    def _add(key: str, val) -> None:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            buckets.setdefault(key, []).append(float(val))

    for fpath in tqdm(json_files, desc="aggregate_distance_stats"):
        try:
            with open(fpath) as fh:
                d = json.load(fh)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        _add("overall_similarity", d.get("Overall_distance"))

        pl = d.get("plastic_level", {})
        _add("plastic_level",          pl.get("mean_plastic_level"))
        ch = pl.get("chromatic_categories", {})
        _add("chromatic_categories",   ch.get("mean_chromatic_distance"))
        col = ch.get("colors", {})
        _add("palette",                col.get("palette"))
        _add("color_distribution",     col.get("color_distribution"))
        _add("brightness",             ch.get("brightness"))
        _add("saturation",             ch.get("saturation"))
        tp = pl.get("topological_categories", {})
        _add("topological_categories", tp.get("mean_topological_categories"))
        pos = tp.get("obj_positions", {})
        _add("obj_position_v_ratio",   (pos.get("vertical_ratio")   or [None, None])[1])
        _add("obj_position_h_ratio",   (pos.get("horizontal_ratio") or [None, None])[1])
        _add("semantic_palette",       tp.get("semantic_palette"))

        fl = d.get("figurative_level", {})
        _add("figurative_level",       fl.get("mean_figurative_level"))
        gen = fl.get("general", {})
        _add("main_topic_tags",        gen.get("main_topic_tags"))
        cp  = fl.get("content_participants", {})
        _add("content_participants",   cp.get("mean_content_participants"))
        spc = cp.get("single_person_characteristics", {})
        _add("age",                    (spc.get("age")            or [None, None])[1])
        _add("gender",                 (spc.get("gender_scores")  or [None, None])[1])
        _add("ethnicity",              (spc.get("ethnicity_scores") or [None, None])[1])
        sfa = cp.get("single_person_face_attributes", {})
        _add("face_attributes",        sfa.get("overall_mean_attrs"))
        _add("inner_face_attributes",  sfa.get("overall_mean_inner_attrs"))
        _add("outer_face_attributes",  sfa.get("overall_mean_outer_attrs"))
        _add("objects_categories",     (cp.get("objects_goals") or {}).get("objects_categories"))
        se  = cp.get("settings_events", {})
        _add("indoor_outdoor",         se.get("indoor-outdoor"))
        _add("place",                  se.get("place"))
        ac  = fl.get("action", {})
        _add("single_action_caption",  ac.get("single_action_caption"))
        em  = fl.get("emotion", {})
        _add("emotions",               em.get("mean_emotion"))
        ei  = em.get("intensity", {})
        _add("valence",                ei.get("mean_valence"))
        _add("arousal",                ei.get("mean_arousal"))
        _add("mean_intensity",         ei.get("mean_intensity"))

        nl = d.get("narrative_level", {})
        _add("narrative_level",        nl.get("mean_narrative_level"))
        bw = nl.get("basic_watcher_looked_system", {})
        _add("basic_watcher_looked",   bw.get("mean_basic_watcher_looked_system"))
        _add("face_background_ratio",  bw.get("portrait_or_scene"))
        fw = nl.get("first_grade_secondary_watcher_looked_system", {})
        _add("first_grade_secondary",  fw.get("mean_first_grade_secondary_watcher_looked_system"))
        hp = fw.get("head_pose", {})
        _add("head_yaw",   hp.get("mean_yaw"))
        _add("head_pitch", hp.get("mean_pitch"))
        _add("head_roll",  hp.get("mean_roll"))
        _add("mean_head_pose", hp.get("mean_head_pose"))
        gz = fw.get("gaze_direction", {})
        _add("gaze_yaw",   gz.get("mean_yaw"))
        _add("gaze_pitch", gz.get("mean_pitch"))
        _add("mean_gaze",  gz.get("mean_gaze"))

    stats: dict[str, dict[str, float]] = {}
    for metric, vals in buckets.items():
        arr = np.array([v for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            continue
        stats[metric] = {
            "mean":  round(float(np.mean(arr)),  4),
            "std":   round(float(np.std(arr)),   4),
            "min":   round(float(np.min(arr)),   4),
            "max":   round(float(np.max(arr)),   4),
            "count": int(arr.size),
        }

    return stats


# ===========================================================================
# Affective label validation  (valence / arousal ratings)
# ===========================================================================

def compute_icc(
    df: "pd.DataFrame",
    target: str,
    rater_col: str = "ID",
    item_col: str  = "video",
) -> dict:
    """Compute ICC(2,1) — two-way random, single measures — for continuous ratings.

    ICC(2,1) measures absolute agreement between raters across items. It is
    the standard metric for inter-rater reliability with continuous affective
    labels (valence, arousal).

    Uses ``pingouin`` when available; falls back to a manual ANOVA-based
    implementation otherwise.

    Returns a dict with: ``icc``, ``F``, ``df1``, ``df2``, ``pval``,
    ``ci95_lower``, ``ci95_upper``, ``interpretation``.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats as _stats

    sub = df[[rater_col, item_col, target]].dropna()

    try:
        import pingouin as pg  # type: ignore[import]
        res = pg.intraclass_corr(
            data=sub, targets=item_col, raters=rater_col, ratings=target
        )
        row = res[res["Type"] == "ICC2"].iloc[0]
        icc_val = float(row["ICC"])
        return {
            "icc":           round(icc_val, 4),
            "F":             round(float(row["F"]), 4),
            "df1":           int(row["df1"]),
            "df2":           int(row["df2"]),
            "pval":          float(row["pval"]),
            "ci95_lower":    round(float(row["CI95%"][0]), 4),
            "ci95_upper":    round(float(row["CI95%"][1]), 4),
            "interpretation": _icc_label(icc_val),
        }
    except ImportError:
        pass  # fallback below

    # Manual one-way ANOVA decomposition (ICC(1,1) approximation)
    pivot  = sub.pivot_table(index=item_col, columns=rater_col, values=target)
    k      = pivot.shape[1]          # number of raters
    n      = pivot.shape[0]          # number of items
    grand  = pivot.values
    ms_row = np.nanvar(np.nanmean(grand, axis=1)) * k
    ms_err = np.nanmean(np.nanvar(grand, axis=0, ddof=1))
    icc_val = (ms_row - ms_err) / (ms_row + (k - 1) * ms_err)
    icc_val = float(np.clip(icc_val, -1, 1))
    F_val  = ms_row / ms_err if ms_err > 0 else float("nan")
    return {
        "icc":           round(icc_val, 4),
        "F":             round(F_val, 4),
        "df1":           int(n - 1),
        "df2":           int(n * (k - 1)),
        "pval":          float("nan"),
        "ci95_lower":    float("nan"),
        "ci95_upper":    float("nan"),
        "interpretation": _icc_label(icc_val),
    }


def _icc_label(icc: float) -> str:
    if icc < 0.50: return "poor"
    if icc < 0.75: return "moderate"
    if icc < 0.90: return "good"
    return "excellent"


def rating_distribution_stats(
    df: "pd.DataFrame",
    targets: list[str] = ("rating_valence", "rating_arousal"),
    rater_col: str = "ID",
    item_col: str  = "video",
) -> dict:
    """Descriptive statistics for each rating target.

    Returns per-target dicts with global stats, per-rater means, and
    per-stimulus variance (as a measure of crowd agreement).
    """
    import pandas as pd
    from scipy.stats import skew, kurtosis

    result = {}
    for tgt in targets:
        sub = df[[rater_col, item_col, tgt]].dropna()
        vals = sub[tgt].values

        per_rater   = sub.groupby(rater_col)[tgt].mean().to_dict()
        per_stim_sd = sub.groupby(item_col)[tgt].std()
        mean_disagreement = float(per_stim_sd.mean())
        most_contested    = per_stim_sd.idxmax() if len(per_stim_sd) else None
        most_agreed       = per_stim_sd.idxmin() if len(per_stim_sd) else None

        result[tgt] = {
            "n_ratings":           int(len(vals)),
            "n_raters":            int(sub[rater_col].nunique()),
            "n_stimuli":           int(sub[item_col].nunique()),
            "mean":                round(float(vals.mean()), 4),
            "std":                 round(float(vals.std()), 4),
            "min":                 round(float(vals.min()), 4),
            "max":                 round(float(vals.max()), 4),
            "skewness":            round(float(skew(vals)), 4),
            "kurtosis":            round(float(kurtosis(vals)), 4),
            "mean_inter_rater_sd": round(mean_disagreement, 4),
            "most_contested_stim": most_contested,
            "most_agreed_stim":    most_agreed,
            "per_rater_mean":      {k: round(v, 4) for k, v in per_rater.items()},
        }
    return result


def annotator_bias_analysis(
    df: "pd.DataFrame",
    targets: list[str] = ("rating_valence", "rating_arousal"),
    rater_col: str = "ID",
) -> "pd.DataFrame":
    """Per-rater statistics to detect scale-usage bias.

    Returns a DataFrame with one row per rater and columns:
    mean, std, min, max, range_used (= max - min) for each target.
    Raters using a narrow range or consistently offset from the group
    mean are flagged as potentially biased.
    """
    import pandas as pd

    rows = []
    group_means = {t: df[t].mean() for t in targets if t in df.columns}

    for rater, gdf in df.groupby(rater_col):
        row = {"rater": rater, "n_ratings": len(gdf)}
        for tgt in targets:
            if tgt not in gdf.columns:
                continue
            vals     = gdf[tgt].dropna()
            r_mean   = float(vals.mean())
            r_std    = float(vals.std())
            r_range  = float(vals.max() - vals.min())
            bias     = r_mean - group_means.get(tgt, r_mean)
            row[f"{tgt}_mean"]      = round(r_mean, 3)
            row[f"{tgt}_std"]       = round(r_std, 3)
            row[f"{tgt}_range"]     = round(r_range, 3)
            row[f"{tgt}_bias"]      = round(bias, 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index("rater")


def crowd_wisdom_analysis(
    df: "pd.DataFrame",
    reliable_col: str = "crowd_wisdom_reliable",
    targets: list[str] = ("rating_valence", "rating_arousal"),
) -> dict:
    """Compare rating statistics for crowd-wisdom-reliable vs unreliable rows.

    Returns counts and per-target mean/std split by reliability flag.
    """
    if reliable_col not in df.columns:
        return {"error": f"Column '{reliable_col}' not found"}

    n_total    = len(df)
    n_reliable = int(df[reliable_col].sum())
    result     = {
        "n_total":         n_total,
        "n_reliable":      n_reliable,
        "n_unreliable":    n_total - n_reliable,
        "pct_reliable":    round(100 * n_reliable / n_total, 1),
    }
    for tgt in targets:
        if tgt not in df.columns:
            continue
        rel   = df[df[reliable_col]  == True][tgt].dropna()
        unrel = df[df[reliable_col]  == False][tgt].dropna()
        result[f"{tgt}_reliable_mean"]   = round(float(rel.mean()),   4) if len(rel)   else None
        result[f"{tgt}_reliable_std"]    = round(float(rel.std()),    4) if len(rel)   else None
        result[f"{tgt}_unreliable_mean"] = round(float(unrel.mean()), 4) if len(unrel) else None
        result[f"{tgt}_unreliable_std"]  = round(float(unrel.std()),  4) if len(unrel) else None
    return result


def validate_affective_labels(df: "pd.DataFrame") -> dict:
    """Run the full empirical validation suite on an affective ratings DataFrame.

    Expects columns: ID (rater), video (item), rating_valence, rating_arousal,
    crowd_wisdom_reliable.

    Returns a nested dict with all validation results.
    """
    targets = [c for c in ("rating_valence", "rating_arousal") if c in df.columns]

    report: dict = {
        "n_rows":      len(df),
        "n_raters":    int(df["ID"].nunique()) if "ID" in df.columns else None,
        "n_stimuli":   int(df["video"].nunique()) if "video" in df.columns else None,
    }

    logger.info("Computing rating distribution stats…")
    report["distribution"] = rating_distribution_stats(df, targets)

    logger.info("Computing inter-rater reliability (ICC)…")
    report["icc"] = {}
    for tgt in targets:
        report["icc"][tgt] = compute_icc(df, tgt)

    logger.info("Computing annotator bias…")
    bias_df = annotator_bias_analysis(df, targets)
    report["annotator_bias"] = bias_df.to_dict(orient="index")

    logger.info("Crowd wisdom reliability analysis…")
    report["crowd_wisdom"] = crowd_wisdom_analysis(df)

    return report


# ===========================================================================
# Script entry point  — run validation against Azure
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import io
    import json
    import sys
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s  %(levelname)-8s  %(message)s",
                         datefmt="%H:%M:%S")
    log = _logging.getLogger(__name__)

    p = argparse.ArgumentParser(
        description="Empirical validation of affective labels from Azure."
    )
    p.add_argument("--protocol", default="protocolaudio",
                   help="NEURO sub-protocol (default: protocolaudio)")
    p.add_argument("--blob", default="auxiliary_signals/occulo_event_level_features_all.csv",
                   help="Blob path relative to the protocol folder")
    p.add_argument("--out", default=None,
                   help="Optional JSON file to save the report")
    args = p.parse_args()

    try:
        from azure_blob import build_client
        from config import AZURE_CONTAINER_NAME
    except ImportError:
        from .azure_blob import build_client  # type: ignore[no-redef]
        from .config import AZURE_CONTAINER_NAME  # type: ignore[no-redef]

    import pandas as pd

    client = build_client()
    cc     = client.get_container_client(AZURE_CONTAINER_NAME)
    blob   = f"NEURO/{args.protocol}/{args.blob}"

    log.info("Loading %s …", blob)
    data = cc.download_blob(blob).readall()
    df   = pd.read_csv(io.BytesIO(data))
    log.info("Loaded %d rows × %d columns", *df.shape)

    report = validate_affective_labels(df)

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Affective Label Validation  —  {args.protocol}")
    print(f"{'═'*60}")
    print(f"  Ratings:   {report['n_rows']}  "
          f"({report['n_raters']} raters × {report['n_stimuli']} stimuli)")

    for tgt, stats in report["distribution"].items():
        label = tgt.replace("rating_", "").capitalize()
        icc   = report["icc"].get(tgt, {})
        print(f"\n  {label}")
        print(f"    Mean ± SD       : {stats['mean']:.3f} ± {stats['std']:.3f}  "
              f"[{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"    Skewness        : {stats['skewness']:.3f}  "
              f"Kurtosis: {stats['kurtosis']:.3f}")
        print(f"    Mean inter-rater SD : {stats['mean_inter_rater_sd']:.3f}")
        print(f"    ICC(2,1)        : {icc.get('icc', 'n/a')}  "
              f"({icc.get('interpretation', '')})  "
              f"95% CI [{icc.get('ci95_lower','?')}, {icc.get('ci95_upper','?')}]")
        print(f"    Most contested  : {stats['most_contested_stim']}")
        print(f"    Most agreed     : {stats['most_agreed_stim']}")

    cw = report["crowd_wisdom"]
    print(f"\n  Crowd Wisdom Reliability")
    print(f"    Reliable rows   : {cw['n_reliable']} / {cw['n_total']} "
          f"({cw['pct_reliable']}%)")

    print(f"\n  Annotator Bias  (mean ± range per rater)")
    bias_df = pd.DataFrame(report["annotator_bias"]).T
    for tgt in ("rating_valence", "rating_arousal"):
        mc = f"{tgt}_mean"
        rc = f"{tgt}_range"
        bc = f"{tgt}_bias"
        if mc in bias_df.columns:
            label = tgt.replace("rating_", "").capitalize()
            print(f"    {label}: mean range = {bias_df[rc].mean():.3f}  "
                  f"max bias = {bias_df[bc].abs().max():.3f}")
    print(f"{'═'*60}\n")

    if args.out:
        safe = json.loads(json.dumps(report, default=str))
        with open(args.out, "w") as fh:
            json.dump(safe, fh, indent=2)
        log.info("Report saved → %s", args.out)

    log.info("Done.")
