"""
fusion_methods.py
-----------------
Fusion methods for multimodal affective computing.

Implements early (feature-level), late (decision-level), and hybrid fusion
strategies for combining signals from heterogeneous modalities such as
physiology, audio, video, text, and web content, including learned fusion
via attention mechanisms and cross-modal transformers.

Audio insight blending
    compute_insight_scores   CLAP + LLM → 0-100 per-dimension scores
    SCORING_WEIGHTS          per-dimension blend weights (extensible)
    CALIBRATION              per-dimension sigmoid calibration floors/ceilings

Semiotic pairwise similarity  (FRESCO DistanceEvaluator)
    DistanceEvaluator        full pairwise semiotic similarity across all metrics
    evaluate_pair            convenience wrapper: two identikit dicts → distance dict

Simple modality fusion
    weighted_average         weighted mean of a score list
    late_fusion_majority     majority vote over per-modality class predictions
    early_fusion_concat      concatenate feature vectors from multiple modalities
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Audio insight blending  (from ai-habs_audio_insights/src/clap_rubric.py)
# ===========================================================================

SCORING_WEIGHTS: dict[str, tuple[float, float]] = {
    "attention":         (0.5, 0.5),
    "attractiveness":    (0.7, 0.3),
    "engagement":        (0.5, 0.5),
    "cognitive_clarity": (0.3, 0.7),
    "memorization":      (0.5, 0.5),
    "differentiation":   (0.5, 0.5),
    "comfort":           (0.7, 0.3),
}
"""Per-dimension blend weights ``(clap_weight, llm_weight)``.

Content-heavy dimensions (e.g. ``cognitive_clarity``) weight the LLM more;
purely acoustic dimensions (e.g. ``attractiveness``) weight CLAP more.
Extend or override per project.
"""

CALIBRATION: dict[str, tuple[float, float]] = {
    k: (0.35, 0.65) for k in SCORING_WEIGHTS
}
"""Per-dimension sigmoid calibration: ``(floor, ceiling)`` in [0, 1].

The default stretches the typical sigmoid output range to the full scale.
"""

_CLAP_TO_INSIGHT: dict[str, str] = {
    "attention":         "attention_impact",
    "attractiveness":    "attractiveness_pleasure",
    "engagement":        "engagement_desire",
    "cognitive_clarity": "cognitive_clarity",
    "memorization":      "memorization",
    "differentiation":   "differentiation",
    "comfort":           "rejection",   # inverted: high comfort → low rejection
}

_PRODUCTION_QUALITY: dict[str, list[tuple]] = {
    "attention":         [("onset_density_per_second", 0.4, 5.0,   False),
                          ("initial_energy_ratio",     0.4, 3.0,   False),
                          ("loudness_dynamic_range_db", 0.2, 40.0,  False)],
    "attractiveness":    [("energy_variance",           0.5, 0.005, True),
                          ("loudness_dynamic_range_db", 0.5, 40.0,  False)],
    "engagement":        [("onset_density_per_second", 0.4, 5.0,   False),
                          ("loudness_dynamic_range_db", 0.4, 40.0,  False),
                          ("energy_variance",           0.2, 0.005, False)],
    "cognitive_clarity": [("pause_ratio",               0.5, 0.4,   False),
                          ("energy_variance",           0.5, 0.005, True)],
    "comfort":           [("energy_variance",           0.5, 0.005, True),
                          ("loudness_dynamic_range_db", 0.5, 40.0,  True)],
    "memorization":      [("onset_density_per_second", 0.5, 5.0,   False),
                          ("initial_energy_ratio",     0.3, 3.0,   False),
                          ("loudness_dynamic_range_db", 0.2, 40.0,  False)],
    "differentiation":   [("spectral_centroid_mean_hz", 0.4, 4000.0, False),
                          ("energy_variance",           0.3, 0.005,  False),
                          ("onset_density_per_second",  0.3, 5.0,    False)],
}


def _prod_mult(clap_dim: str, features) -> float:
    signal = sum(
        w * (1.0 - min(max(getattr(features, f, 0.0), 0.0) / mv, 1.0) if inv
             else w * min(max(getattr(features, f, 0.0), 0.0) / mv, 1.0))
        for f, w, mv, inv in _PRODUCTION_QUALITY.get(clap_dim, [])
    )
    return 0.90 + 0.20 * signal


def _level(score: float) -> str:
    if score < 25: return "low"
    if score < 50: return "moderate"
    if score < 75: return "high"
    return "very_high"


def compute_insight_scores(
    similarities,
    features,
    transcript_scores=None,
    scoring_weights: dict[str, tuple[float, float]] | None = None,
    calibration: dict[str, tuple[float, float]] | None = None,
) -> Any:
    """Blend CLAP + LLM transcript scores into 7 insight scores (0-100).

    This is the cross-modal fusion step for the audio modality.  It combines
    purely acoustic CLAP similarity scores with text-content LLM scores using
    configurable per-dimension weights and a production-quality multiplier
    derived from acoustic features.

    Formula per dimension::

        clap_cal = clamp(0,1, (clap_raw - floor) / (ceiling - floor))
        blended  = clap_w * clap_cal + llm_w * (llm_score / 100)
        score    = clamp(0, 100, blended * production_multiplier * 100)

    When *transcript_scores* is ``None``, CLAP takes full weight (acoustic-only
    mode).  The ``rejection`` dimension inverts the ``comfort`` score.

    Args:
        similarities:      ``ClapRubricSimilarities`` from ``features_multimedia``.
        features:          ``AcousticFeatures`` from ``features_multimedia``.
        transcript_scores: Optional ``TranscriptScores`` from LLM scoring.
        scoring_weights:   Override ``SCORING_WEIGHTS`` (default ``None``).
        calibration:       Override ``CALIBRATION`` (default ``None``).

    Returns:
        ``InsightScores`` with seven 0-100 fields.
    """
    try:
        from schema import InsightDetail, InsightScores  # type: ignore[import]
    except ImportError:
        from .schema import InsightDetail, InsightScores  # type: ignore[import]

    sw   = scoring_weights or SCORING_WEIGHTS
    cal  = calibration     or CALIBRATION

    insights: dict[str, Any] = {}
    for clap_dim, field in _CLAP_TO_INSIGHT.items():
        clap_raw       = getattr(similarities, clap_dim).score
        floor, ceiling = cal.get(clap_dim, (0.35, 0.65))
        calibrated     = max(0.0, min(1.0, (clap_raw - floor) / (ceiling - floor)))
        if clap_dim == "comfort":
            calibrated = 1.0 - calibrated

        mult          = _prod_mult(clap_dim, features)
        clap_w, llm_w = sw.get(clap_dim, (1.0, 0.0))
        llm_raw: float | None = None

        if transcript_scores is not None:
            llm_val  = getattr(transcript_scores, clap_dim)
            if clap_dim == "comfort":
                llm_val = 100.0 - llm_val
            llm_norm = max(0.0, min(1.0, llm_val / 100.0))
            llm_raw  = round(llm_val, 1)
            blended  = clap_w * calibrated + llm_w * llm_norm
        else:
            blended = calibrated

        final = max(0.0, min(100.0, blended * mult * 100.0))
        comps: dict[str, float] = {
            "clap_raw":              round(clap_raw,  4),
            "calibrated":            round(calibrated, 4),
            "production_multiplier": round(mult,       4),
            "clap_weight":           clap_w if transcript_scores is not None else 1.0,
            "llm_weight":            llm_w  if transcript_scores is not None else 0.0,
        }
        if llm_raw is not None:
            comps["llm_raw"] = llm_raw
        insights[field] = InsightDetail(
            score=round(final, 1), level=_level(final),
            explanation=None, components=comps,
        )

    return InsightScores(**insights)


# ===========================================================================
# Semiotic pairwise distance  (FRESCO DistanceEvaluator)
# ===========================================================================

class DistanceEvaluator:
    """Compute pairwise semiotic similarity between two FRESCO identikit records.

    Mirrors the interface of ``fresco/distance/distance_evaluator.py``.
    Provides metric methods for each semiotic dimension (plastic, figurative,
    narrative level) and aggregates them into an overall distance dict.

    Requires ``scikit-image``, ``scipy``, ``clip``, and ``torch``.
    """

    def __init__(
        self,
        obj_matching: str = "hungarian",
        include_unpaired: bool = True,
        conf_distance: str = "cosine",
        decimals: int = 3,
    ):
        import torch
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
        self.include_unpaired = include_unpaired
        self.conf_dst        = conf_distance
        self.decimals        = decimals
        self._img1: dict     = {}
        self._img2: dict     = {}

        # Lazy-load CLIP text scorer
        self._clip_txt = None

        if obj_matching == "hungarian":
            from features_multimedia import percentage_common_objects as _match  # type: ignore[import]
            self._obj_match = _match
        else:
            self._obj_match = None

    @property
    def clip_txt(self):
        if self._clip_txt is None:
            try:
                from features_multimedia import ClipTextScore  # type: ignore[import]
            except ImportError:
                from .features_multimedia import ClipTextScore
            self._clip_txt = ClipTextScore(device=self.device)
        return self._clip_txt

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, identikit_a: dict, identikit_b: dict) -> dict:
        """Compute all semiotic distances between two identikit records.

        Args:
            identikit_a: First FRESCO identikit dict.
            identikit_b: Second FRESCO identikit dict.

        Returns:
            Nested distance dict with keys matching FRESCO's output schema.
        """
        self._img1 = identikit_a
        self._img2 = identikit_b
        return self._compute_all()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _safe(self, fn, *args, default=0.0):
        try:
            return fn(*args)
        except Exception as exc:
            logger.debug("DistanceEvaluator metric failed: %s", exc)
            return default

    def _mean(self, values: list[float]) -> float:
        valid = [v for v in values if v is not None and math.isfinite(v)]
        return round(float(np.mean(valid)), self.decimals) if valid else 0.0

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _binary(self, v1, v2) -> float:
        return 0.0 if v1 == v2 else 1.0

    def _cosine(self, vec1, vec2) -> float:
        a = np.asarray(vec1, dtype=float)
        b = np.asarray(vec2, dtype=float)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return round(float(np.dot(a, b) / d), self.decimals) if d else 0.0

    def _jaccard_list(self, l1: list, l2: list) -> float:
        s1, s2 = set(l1), set(l2)
        if not s1 and not s2: return 1.0
        return round(len(s1 & s2) / len(s1 | s2), self.decimals)

    def _scalar_dist(self, v1: float, v2: float, scale: float = 1.0) -> float:
        return round(1.0 - abs(v1 - v2) / scale, self.decimals)

    # ------------------------------------------------------------------
    # Full metric computation
    # ------------------------------------------------------------------

    def _compute_all(self) -> dict:
        d1, d2 = self._img1, self._img2
        pl1    = d1.get("plastic_level",    {})
        pl2    = d2.get("plastic_level",    {})
        fl1    = d1.get("figurative_level", {})
        fl2    = d2.get("figurative_level", {})
        nl1    = d1.get("narrative_level",  {})
        nl2    = d2.get("narrative_level",  {})

        # -- Plastic level ------------------------------------------------
        ch1, ch2 = pl1.get("chromatic_categories", {}), pl2.get("chromatic_categories", {})
        c1,  c2  = ch1.get("colors", {}), ch2.get("colors", {})
        grayscale  = self._safe(lambda: self._binary(c1.get("is_grayscale"), c2.get("is_grayscale")))
        brightness = self._safe(lambda: self._scalar_dist(ch1.get("brightness", 0),
                                                           ch2.get("brightness", 0), 255.0))
        saturation = self._safe(lambda: self._scalar_dist(ch1.get("saturation", 0),
                                                           ch2.get("saturation", 0), 255.0))
        col_dist   = self._safe(lambda: self._cosine(c1.get("color_distribution", []),
                                                      c2.get("color_distribution", [])))
        chromatic_mean = self._mean([grayscale, brightness, saturation, col_dist])

        tp1, tp2 = (pl1.get("topological_categories", {}),
                    pl2.get("topological_categories", {}))
        semantic_pal = self._safe(lambda: self._jaccard_list(
            tp1.get("semantic_palette", {}).get("categories", []),
            tp2.get("semantic_palette", {}).get("categories", [])))
        topological_mean = self._mean([semantic_pal])

        plastic_mean = self._mean([chromatic_mean, topological_mean])

        # -- Figurative level --------------------------------------------
        gen1, gen2 = fl1.get("general", {}), fl2.get("general", {})
        tags1 = [t.strip() for t in gen1.get("main_topic_tags", "").split("|")]
        tags2 = [t.strip() for t in gen2.get("main_topic_tags", "").split("|")]
        topic_sim = self._safe(lambda: self._clip_txt(" ".join(tags1), " ".join(tags2)))

        cp1, cp2 = (fl1.get("content_participants", {}),
                    fl2.get("content_participants", {}))
        obj1 = cp1.get("objects_goals", {}).get("objects_categories", [])
        obj2 = cp2.get("objects_goals", {}).get("objects_categories", [])
        obj_sim  = self._safe(lambda: self._jaccard_list(obj1, obj2))

        se1, se2 = cp1.get("settings_events", {}), cp2.get("settings_events", {})
        io_sim   = self._safe(lambda: self._binary(se1.get("indoor-outdoor"),
                                                   se2.get("indoor-outdoor")))

        ac1, ac2 = fl1.get("action", {}), fl2.get("action", {})
        cap_sim  = self._safe(lambda: self._clip_txt(
            ac1.get("single_action_caption", ""),
            ac2.get("single_action_caption", ""),
        ))

        em1, em2 = fl1.get("emotion", {}), fl2.get("emotion", {})
        ei1, ei2 = em1.get("intensity", {}), em2.get("intensity", {})
        val_sim  = self._safe(lambda: self._scalar_dist(
            ei1.get("mean_valence", 0), ei2.get("mean_valence", 0), 2.0))
        aro_sim  = self._safe(lambda: self._scalar_dist(
            ei1.get("mean_arousal", 0), ei2.get("mean_arousal", 0), 2.0))
        figurative_mean = self._mean([topic_sim, obj_sim, cap_sim, val_sim, aro_sim])

        # -- Narrative level ---------------------------------------------
        bw1, bw2 = (nl1.get("basic_watcher_looked_system", {}),
                    nl2.get("basic_watcher_looked_system", {}))
        scene_sim = self._safe(lambda: self._binary(
            bw1.get("portrait_or_scene"), bw2.get("portrait_or_scene")))

        fw1, fw2 = (nl1.get("first_grade_secondary_watcher_looked_system", {}),
                    nl2.get("first_grade_secondary_watcher_looked_system", {}))
        hp1, hp2 = fw1.get("head_pose", {}), fw2.get("head_pose", {})
        yaw_sim  = self._safe(lambda: self._scalar_dist(
            hp1.get("mean_yaw", 0), hp2.get("mean_yaw", 0), 180.0))
        narrative_mean = self._mean([scene_sim, yaw_sim])

        overall = self._mean([plastic_mean, figurative_mean, narrative_mean])

        return {
            "Overall_distance": overall,
            "plastic_level": {
                "mean_plastic_level":     plastic_mean,
                "chromatic_categories":   {"mean_chromatic_distance": chromatic_mean,
                                           "colors": {"is_grayscale": grayscale,
                                                      "color_distribution": col_dist},
                                           "brightness": brightness,
                                           "saturation": saturation},
                "topological_categories": {"mean_topological_categories": topological_mean,
                                           "semantic_palette": semantic_pal},
            },
            "figurative_level": {
                "mean_figurative_level":  figurative_mean,
                "general":                {"main_topic_tags": topic_sim},
                "content_participants":   {"objects_goals": {"objects_categories": obj_sim},
                                           "settings_events": {"indoor-outdoor": io_sim}},
                "action":                 {"single_action_caption": cap_sim},
                "emotion":                {"mean_emotion": self._mean([val_sim, aro_sim]),
                                           "intensity":    {"mean_valence": val_sim,
                                                            "mean_arousal": aro_sim}},
            },
            "narrative_level": {
                "mean_narrative_level":   narrative_mean,
                "basic_watcher_looked_system": {"portrait_or_scene": scene_sim},
                "first_grade_secondary_watcher_looked_system": {"head_pose": {"mean_yaw": yaw_sim}},
            },
        }


def evaluate_pair(identikit_a: dict, identikit_b: dict, **kwargs) -> dict:
    """Convenience wrapper: compute all semiotic distances between two identikits.

    Args:
        identikit_a: First FRESCO identikit dict.
        identikit_b: Second FRESCO identikit dict.
        **kwargs:    Forwarded to ``DistanceEvaluator.__init__``.

    Returns:
        Nested distance dict (see ``DistanceEvaluator.evaluate``).
    """
    return DistanceEvaluator(**kwargs).evaluate(identikit_a, identikit_b)


# ===========================================================================
# Simple modality fusion utilities
# ===========================================================================

def weighted_average(scores: list[float], weights: list[float] | None = None) -> float:
    """Weighted mean of *scores*.  Uniform weights when *weights* is ``None``."""
    arr = np.asarray(scores, dtype=float)
    w   = (np.asarray(weights, dtype=float) if weights is not None
           else np.ones(len(arr)))
    return float(np.average(arr, weights=w))


def late_fusion_majority(predictions: list[int]) -> int:
    """Return the majority-voted class from a list of per-modality predictions."""
    from collections import Counter
    return Counter(predictions).most_common(1)[0][0]


def early_fusion_concat(feature_vectors: list[np.ndarray]) -> np.ndarray:
    """Concatenate feature vectors from multiple modalities into one flat vector."""
    return np.concatenate([v.ravel() for v in feature_vectors])
