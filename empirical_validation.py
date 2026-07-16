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
