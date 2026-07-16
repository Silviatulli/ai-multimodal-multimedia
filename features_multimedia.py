"""
features_multimedia.py
----------------------
Feature extraction methods for multimedia content.

Implements extraction of audio, video, visual, audiovisual, textual, and
web-based features from heterogeneous media sources, including:

Audio
    extract_acoustic_features   13 librosa-based descriptors
    score_clap_rubric           7-dimension CLAP audio-text cosine scores
    transcribe_audio            Whisper large-v3 transcription
    compute_insight_scores      CLAP + LLM → 0-100 per-dimension blend

Text / image layout
    extract_line_features       spatial + embedding summaries per OCR line
    build_hybrid_feature_matrix KMeans-ready feature matrix
    cluster_and_label_layout    assign Header/Subheader/Body/Caption roles
    analyze_sentiment           LLM-based ad sentiment (Ollama / DeepSeek)

Video
    VideoEmotionClassifier      VideoMAE / TimeSformer emotion classifier
    build_model / load_model    model factory helpers
    VideoEmotionDataset         live-decode PyTorch Dataset
    CachedVideoDataset          .npy-backed fast Dataset
    VideoEmotionWindowDataset   windowed Dataset
    TemporalLabelDataset        per-second Dataset for oculometry labels
    load_videos_from_predictions CSV prediction loader

Distance / similarity metrics
    jaccard_similarity          set-based Jaccard
    continuous_jaccard_similarity weighted Jaccard for frequency dicts
    cosine_similarity           cosine similarity between vectors
    bhattacharyya_distance      Bhattacharyya distance between histograms
    percentage_common_objects   fraction of shared objects
    ClipTextScore               CLIP text-text cosine similarity scorer
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .config import CATEGORIES_DIR, DATA_DIR, NUM_FRAMES, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC
    from .multimedia_data_preprocessing import (
        augment_frames,
        build_video_index,
        get_video_info,
        get_video_windows,
        load_participant_labels_from_predictions,
        sample_frames,
        sample_frames_from_segment,
    )
    from .schema import (
        AcousticFeatures,
        ClapRubricSimilarities,
        ClapSimilarity,
        InsightDetail,
        InsightScores,
        LayoutItem,
        LineSummary,
        TextRegion,
        Transcript,
        TranscriptScores,
    )
except ImportError:
    from config import CATEGORIES_DIR, DATA_DIR, NUM_FRAMES, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC  # type: ignore[no-redef]
    from multimedia_data_preprocessing import (  # type: ignore[no-redef]
        augment_frames,
        build_video_index,
        get_video_info,
        get_video_windows,
        load_participant_labels_from_predictions,
        sample_frames,
        sample_frames_from_segment,
    )
    from schema import (  # type: ignore[no-redef]
        AcousticFeatures,
        ClapRubricSimilarities,
        ClapSimilarity,
        InsightDetail,
        InsightScores,
        LayoutItem,
        LineSummary,
        TextRegion,
        Transcript,
        TranscriptScores,
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# Audio — acoustic feature extraction
# ===========================================================================

_HOP_LENGTH    = 512
_FRAME_LENGTH  = 2048
_EPS           = 1e-10
_INITIAL_WIN_S = 3.0
_PAUSE_THR_DB  = -30.0


def _finite(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _rms(waveform: np.ndarray) -> np.ndarray:
    import librosa
    rms = librosa.feature.rms(
        y=waveform, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
    )[0]
    return rms if rms.size else np.array([0.0], dtype=np.float32)


def extract_acoustic_features(waveform: np.ndarray, sample_rate: int) -> AcousticFeatures:
    """Extract 13 librosa-based acoustic features from a mono float32 waveform.

    Returns an ``AcousticFeatures`` instance with all fields finite.
    Requires ``librosa``.
    """
    import librosa

    waveform   = np.asarray(waveform, dtype=np.float32)
    duration_s = len(waveform) / sample_rate if sample_rate else 0.0
    rms        = _rms(waveform)
    rms_mean   = float(np.mean(rms))
    rms_max    = float(np.max(rms))

    n_init           = int(_INITIAL_WIN_S * sample_rate)
    init             = waveform[:n_init] if n_init > 0 else waveform
    init_rms         = float(np.sqrt(np.mean(np.square(init)))) if init.size else 0.0
    initial_en_ratio = init_rms / (rms_mean + _EPS)

    energy_slope    = float(np.polyfit(np.arange(rms.size), rms, 1)[0]) if rms.size >= 2 else 0.0
    peak_pos_ratio  = float(np.argmax(rms) / (rms.size - 1)) if rms.size >= 2 else 0.0

    high        = float(np.percentile(rms, 95))
    low         = float(np.percentile(rms, 5))
    dyn_rng_db  = 20.0 * float(np.log10((high + _EPS) / (low + _EPS)))

    shape_5seg  = [float(np.mean(s)) if s.size else 0.0 for s in np.array_split(rms, 5)]

    zcr         = librosa.feature.zero_crossing_rate(
        waveform, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]
    zcr_mean    = float(np.mean(zcr)) if zcr.size else 0.0
    zcr_var     = float(np.var(zcr))  if zcr.size else 0.0

    centroid      = librosa.feature.spectral_centroid(
        y=waveform, sr=sample_rate, hop_length=_HOP_LENGTH)[0]
    centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0

    if duration_s > 0:
        onsets        = librosa.onset.onset_detect(
            y=waveform, sr=sample_rate, hop_length=_HOP_LENGTH, units="time")
        onset_density = float(len(onsets) / duration_s)
    else:
        onset_density = 0.0

    peak = float(np.max(rms))
    if peak > _EPS:
        thr         = peak * (10.0 ** (_PAUSE_THR_DB / 20.0))
        pause_ratio = float(np.mean(rms < thr))
    else:
        pause_ratio = 0.0

    return AcousticFeatures(
        rms_energy_mean=_finite(rms_mean),
        rms_energy_max=_finite(rms_max),
        initial_energy_ratio=_finite(initial_en_ratio),
        energy_slope=_finite(energy_slope),
        peak_position_ratio=_finite(peak_pos_ratio),
        energy_variance=_finite(float(np.var(rms))),
        loudness_dynamic_range_db=_finite(dyn_rng_db),
        loudness_shape_5seg=[_finite(v) for v in shape_5seg],
        zero_crossing_rate_mean=_finite(zcr_mean),
        zero_crossing_rate_var=_finite(zcr_var),
        spectral_centroid_mean_hz=_finite(centroid_mean),
        onset_density_per_second=_finite(onset_density),
        pause_ratio=_finite(pause_ratio),
    )


# ---------------------------------------------------------------------------
# CLAP rubric scoring
# ---------------------------------------------------------------------------

RUBRIC_PROMPTS_V1: dict[str, dict[str, str]] = {
    "attention":         {"positive": "audio that immediately grabs your attention",
                          "negative": "boring monotonous background audio that is easy to ignore"},
    "attractiveness":    {"positive": "pleasant appealing audio that creates positive feelings",
                          "negative": "annoying grating unpleasant audio"},
    "engagement":        {"positive": "exciting compelling audio that makes you want to take action",
                          "negative": "dull uninteresting audio you want to skip"},
    "cognitive_clarity": {"positive": "clear simple audio with an easy to understand message",
                          "negative": "confusing cluttered audio that is hard to follow"},
    "memorization":      {"positive": "catchy memorable audio with a distinctive hook",
                          "negative": "forgettable generic audio"},
    "differentiation":   {"positive": "unique distinctive original audio that stands out",
                          "negative": "generic typical conventional audio"},
    "comfort":           {"positive": "smooth comfortable relaxing audio",
                          "negative": "harsh loud jarring uncomfortable audio"},
}

_CLAP_SR       = 48_000
_CLAP_MAX_SAMP = 10 * _CLAP_SR
_SIGMOID_K     = 10


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def _sigmoid(pos: float, neg: float) -> float:
    return 1.0 / (1.0 + math.exp(-(pos - neg) * _SIGMOID_K))


def score_clap_rubric(waveform: np.ndarray, sample_rate: int) -> ClapRubricSimilarities:
    """Score audio against 7 CLAP rubric dimensions (positive vs. negative prompts).

    Audio is chunked into ≤10 s segments, embedded, and averaged before
    cosine similarity is computed.  Returns neutral 0.5 scores on failure.
    Requires ``msclap`` (``pip install msclap``).
    """
    try:
        import librosa
        from msclap import CLAP  # type: ignore[import]

        model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
        if sample_rate != _CLAP_SR:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=_CLAP_SR)
        waveform = waveform.astype(np.float32)

        chunks   = [waveform[s:s + _CLAP_MAX_SAMP]
                    for s in range(0, len(waveform), _CLAP_MAX_SAMP)]
        embs     = [model.get_audio_embedding_from_data(x=c.reshape(1, -1))[0] for c in chunks]
        audio_v  = np.mean(embs, axis=0)

        all_texts = [p for dim in RUBRIC_PROMPTS_V1
                     for p in (RUBRIC_PROMPTS_V1[dim]["positive"],
                                RUBRIC_PROMPTS_V1[dim]["negative"])]
        text_embs = model.get_text_embedding(all_texts)

        sims: dict[str, ClapSimilarity] = {}
        for i, dim in enumerate(RUBRIC_PROMPTS_V1):
            sims[dim] = ClapSimilarity(score=_sigmoid(
                _cos_sim(audio_v, text_embs[i * 2]),
                _cos_sim(audio_v, text_embs[i * 2 + 1]),
            ))
        return ClapRubricSimilarities(**sims)

    except Exception:
        logger.exception("score_clap_rubric: failed, returning neutral defaults")
        neutral = ClapSimilarity(score=0.5)
        return ClapRubricSimilarities(**{d: neutral for d in RUBRIC_PROMPTS_V1})


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(waveform: np.ndarray, sample_rate: int) -> tuple[Transcript, str | None]:
    """Transcribe a waveform with Whisper large-v3 via faster-whisper.

    Returns ``(Transcript, detected_language)``; on failure returns an empty
    transcript and ``None``.  Requires ``faster-whisper``.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
        from schema import TranscriptSegment

        model          = WhisperModel("large-v3",
                                      compute_type="float16" if torch.cuda.is_available() else "int8")
        segs_iter, info = model.transcribe(
            waveform.astype(np.float32), word_timestamps=True, beam_size=5, temperature=0.0
        )
        segments, texts = [], []
        for seg in segs_iter:
            txt = seg.text.strip()
            if txt:
                segments.append(TranscriptSegment(start_s=seg.start, end_s=seg.end, text=txt))
                texts.append(txt)

        if not segments:
            return Transcript(full_text="", segments=[], word_count=0,
                              estimated_speech_rate_wpm=None), None

        full     = " ".join(texts)
        wc       = len(full.split())
        dur_s    = segments[-1].end_s - segments[0].start_s
        rate_wpm = round(wc / (dur_s / 60.0), 1) if dur_s > 0 and wc > 0 else None
        return Transcript(full_text=full, segments=segments,
                          word_count=wc, estimated_speech_rate_wpm=rate_wpm), info.language
    except Exception:
        logger.exception("transcribe_audio: failed")
        return Transcript(full_text="", segments=[], word_count=0,
                          estimated_speech_rate_wpm=None), None


# ---------------------------------------------------------------------------
# Insight score blending  (CLAP + LLM → 0-100)
# ---------------------------------------------------------------------------

_DEFAULT_FLOOR   = 0.35
_DEFAULT_CEILING = 0.65
_CALIBRATION: dict[str, tuple[float, float]] = {
    k: (_DEFAULT_FLOOR, _DEFAULT_CEILING) for k in RUBRIC_PROMPTS_V1
}
_SCORING_WEIGHTS: dict[str, tuple[float, float]] = {
    "attention":         (0.5, 0.5),
    "attractiveness":    (0.7, 0.3),
    "engagement":        (0.5, 0.5),
    "cognitive_clarity": (0.3, 0.7),
    "memorization":      (0.5, 0.5),
    "differentiation":   (0.5, 0.5),
    "comfort":           (0.7, 0.3),
}
_CLAP_TO_INSIGHT: dict[str, str] = {
    "attention":         "attention_impact",
    "attractiveness":    "attractiveness_pleasure",
    "engagement":        "engagement_desire",
    "cognitive_clarity": "cognitive_clarity",
    "memorization":      "memorization",
    "differentiation":   "differentiation",
    "comfort":           "rejection",
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


def _prod_multiplier(clap_dim: str, features: AcousticFeatures) -> float:
    specs  = _PRODUCTION_QUALITY.get(clap_dim, [])
    signal = sum(
        w * (1.0 - min(max(getattr(features, f, 0.0), 0.0) / mv, 1.0) if inv
             else w * min(max(getattr(features, f, 0.0), 0.0) / mv, 1.0))
        for f, w, mv, inv in specs
    )
    return 0.90 + 0.20 * signal


def _level(score: float) -> str:
    if score < 25: return "low"
    if score < 50: return "moderate"
    if score < 75: return "high"
    return "very_high"


def compute_insight_scores(
    similarities: ClapRubricSimilarities,
    features: AcousticFeatures,
    transcript_scores: TranscriptScores | None = None,
) -> InsightScores:
    """Blend CLAP + LLM transcript scores into 7 insight scores (0-100).

    When *transcript_scores* is None, CLAP takes full weight (acoustic-only).
    The ``rejection`` dimension inverts the ``comfort`` CLAP/LLM score.
    """
    insights: dict[str, InsightDetail] = {}
    for clap_dim, field in _CLAP_TO_INSIGHT.items():
        clap_raw       = getattr(similarities, clap_dim).score
        floor, ceiling = _CALIBRATION[clap_dim]
        calibrated     = max(0.0, min(1.0, (clap_raw - floor) / (ceiling - floor)))
        if clap_dim == "comfort":
            calibrated = 1.0 - calibrated
        mult          = _prod_multiplier(clap_dim, features)
        clap_w, llm_w = _SCORING_WEIGHTS[clap_dim]
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
# Text / image layout — feature extraction and sentiment
# ===========================================================================

def _text_bg_contrast(image_gray, bbox, neutral: float = 0.5) -> float:
    try:
        h, w   = image_gray.shape[:2]
        x1, y1 = max(int(bbox.x_min), 0), max(int(bbox.y_min), 0)
        x2, y2 = min(int(bbox.x_max), w), min(int(bbox.y_max), h)
        if x2 <= x1 or y2 <= y1:
            return neutral
        inner      = image_gray[y1:y2, x1:x2]
        inner_mean = float(np.mean(inner))
        mx = max(int(bbox.width  * 0.3), 5)
        my = max(int(bbox.height * 0.3), 5)
        ox1, oy1   = max(x1 - mx, 0), max(y1 - my, 0)
        ox2, oy2   = min(x2 + mx, w), min(y2 + my, h)
        outer      = image_gray[oy1:oy2, ox1:ox2].copy()
        mask       = np.ones(outer.shape, dtype=bool)
        mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
        bg         = outer[mask]
        if bg.size == 0:
            return neutral
        bg_mean = float(np.mean(bg.astype(np.float32)))
        lo, hi  = min(inner_mean, bg_mean), max(inner_mean, bg_mean)
        return float(np.clip((hi - lo) / (hi + lo + 1e-6), 0.0, 1.0))
    except Exception:
        return neutral


def extract_line_features(
    region_lines: list[list],
    image_width: int,
    image_height: int,
    image_gray: np.ndarray | None = None,
    contrast_neutral: float = 0.5,
) -> list[dict]:
    """Aggregate word-level TextRegions into one summary dict per line.

    Args:
        region_lines:    List of lines; each line is a list of TextRegion words.
        image_width:     Source image width in pixels.
        image_height:    Source image height in pixels.
        image_gray:      Optional grayscale image for contrast computation.
        contrast_neutral: Fallback contrast when image is unavailable.
    """
    from schema import BBox

    summaries: list[dict] = []
    for line in region_lines:
        if not line:
            continue
        text  = " ".join(r.text for r in line)
        stack = np.stack([r.features for r in line])
        emb   = np.mean(stack, axis=0)

        centers = [r.get_center() for r in line if r.get_center() is not None]
        bboxes  = [r.bbox for r in line if r.bbox is not None]
        cx      = float(np.mean([c[0] for c in centers])) if centers else image_width  / 2
        cy      = float(np.mean([c[1] for c in centers])) if centers else image_height / 2

        if bboxes:
            x_min = min(b[0] for b in bboxes)
            y_min = min(b[1] for b in bboxes)
            x_max = max(b[2] for b in bboxes)
            y_max = max(b[3] for b in bboxes)
        else:
            x_min, y_min, x_max, y_max = cx - 20, cy - 20, cx + 20, cy + 20

        confs   = [float(getattr(r, "confidence", 0.5)) for r in line]
        bb      = BBox(x_min, y_min, x_max, y_max)
        contrast = (_text_bg_contrast(image_gray, bb, neutral=contrast_neutral)
                    if image_gray is not None else contrast_neutral)

        summaries.append({
            "text":            text,
            "embedding":       emb,
            "center":          (cx, cy),
            "bbox":            bb,
            "word_count":      len(text.split()),
            "has_features":    bool(np.any(stack != 0)),
            "image_width":     image_width,
            "image_height":    image_height,
            "min_confidence":  min(confs),
            "mean_confidence": sum(confs) / len(confs),
            "bg_contrast":     contrast,
        })
    return summaries


def build_hybrid_feature_matrix(
    summaries: list[dict],
    spatial_weight: float = 1.0,
    semantic_weight: float = 1.0,
    use_semantic: bool = True,
) -> np.ndarray:
    """Build the KMeans input matrix from line summaries.

    Combines z-scored spatial features with z-scored semantic embeddings.
    """
    from sklearn.preprocessing import StandardScaler

    img_w = summaries[0]["image_width"]
    img_h = summaries[0]["image_height"]

    def _spatial(s: dict) -> list[float]:
        bb   = s["bbox"]
        area = max(img_w * img_h, 1.0)
        nc   = len(s["text"].replace(" ", ""))
        return [
            bb.area / area,
            s["center"][1] / max(img_h, 1.0),
            s["center"][0] / max(img_w, 1.0),
            bb.width / bb.height,
            nc / bb.area * 1e4,
            1.0 / max(s["word_count"], 1),
            float(s.get("bg_contrast", 0.5)),
        ]

    def _z(m: np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(m).astype(np.float32) if m.shape[0] >= 2 else m

    spatial_m  = _z(np.array([_spatial(s) for s in summaries], dtype=np.float32))
    semantic_m = _z(np.array([s["embedding"] for s in summaries], dtype=np.float32))

    return np.hstack([spatial_m * spatial_weight, semantic_m * semantic_weight]) \
        if use_semantic else spatial_m * spatial_weight


def cluster_and_label_layout(
    region_lines: list[list],
    n_clusters: int = 2,
    image_width: int = 1000,
    image_height: int = 1000,
    image_gray: np.ndarray | None = None,
    use_semantic: bool = True,
) -> list[dict]:
    """Cluster OCR text lines into visual-hierarchy roles (Header/Body/Caption).

    Runs KMeans on a hybrid spatial + semantic feature matrix, then ranks
    clusters by visual prominence (size, contrast, position, brevity).
    """
    from sklearn.cluster import KMeans

    _ROLE_NAMES = {
        2: ["Header", "Body"],
        3: ["Header", "Body", "Caption"],
        4: ["Header", "Subheader", "Body", "Caption"],
    }

    summaries  = extract_line_features(region_lines, image_width, image_height, image_gray)
    if not summaries:
        return []

    n_clusters  = min(n_clusters, len(summaries))
    feat_mat    = build_hybrid_feature_matrix(summaries, use_semantic=use_semantic)
    labels      = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(feat_mat)

    stats: dict[int, dict] = {}
    for cid in range(n_clusters):
        members = [summaries[i] for i, l in enumerate(labels) if l == cid]
        if not members:
            stats[cid] = {"size": 0.0, "contrast": 0.0, "rel_y": 0.0, "wc": 0.0}
            continue
        stats[cid] = {
            "size":    float(np.mean([max(s["bbox"].y_max - s["bbox"].y_min, 1.0) for s in members])),
            "contrast":float(np.mean([s.get("bg_contrast", 0.5) for s in members])),
            "rel_y":   float(np.mean([s["center"][1] for s in members])) / max(image_height, 1.0),
            "wc":      float(np.mean([s["word_count"] for s in members])),
        }

    def _mm(vals: list) -> list:
        lo, hi = min(vals), max(vals)
        return [0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in vals]

    ids   = list(range(n_clusters))
    s_sc  = _mm([stats[i]["size"]     for i in ids])
    c_sc  = _mm([stats[i]["contrast"] for i in ids])
    p_sc  = _mm([-stats[i]["rel_y"]   for i in ids])
    b_sc  = _mm([-stats[i]["wc"]      for i in ids])
    prom  = {ids[i]: 0.4*s_sc[i] + 0.3*c_sc[i] + 0.2*p_sc[i] + 0.1*b_sc[i]
             for i in range(n_clusters)}
    ranked   = sorted(ids, key=lambda x: prom[x], reverse=True)
    roles    = _ROLE_NAMES.get(n_clusters, ["Header"] + [f"Body_{i}" for i in range(1, n_clusters)])
    id2role  = {ranked[i]: roles[i] for i in range(n_clusters)}

    result: list[dict] = []
    for summary, label in zip(summaries, labels):
        bb = summary["bbox"]
        result.append({
            "text":                  summary["text"],
            "visual_role":           id2role[int(label)],
            "position_x":            round(summary["center"][0], 1),
            "position_y":            round(summary["center"][1], 1),
            "bbox":                  [round(v, 1) for v in bb.as_tuple()],
            "word_count":            summary["word_count"],
            "rel_y":                 round(summary["center"][1] / max(image_height, 1.0), 3),
            "has_feature_embeddings":summary["has_features"],
            "ocr_min_confidence":    round(summary.get("min_confidence",  1.0), 3),
            "ocr_mean_confidence":   round(summary.get("mean_confidence", 1.0), 3),
        })
    result.sort(key=lambda x: (x["position_y"], x["position_x"]))
    return result


def analyze_sentiment(
    layout_elements: list[dict],
    provider: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
    ollama_url: str = "http://localhost:11434",
    deepseek_url: str = "https://api.deepseek.com",
) -> dict:
    """Layout-aware LLM sentiment analysis for structured ad text.

    Sends Phase-2 layout elements (with visual roles) to an LLM and parses
    the response into five fields: ``cleaned_text``, ``emotional_tone``,
    ``target_audience``, ``structural_analysis``, ``overall_feedback``.

    Providers: ``"ollama"`` (local) or ``"deepseek"`` (API, needs
    ``DEEPSEEK_API_KEY`` env var or *api_key* argument).
    """
    import textwrap, urllib.request

    ROLE_W  = {"Header": 3.0, "Subheader": 2.0, "Body": 1.5, "Caption": 1.0}
    r_order = {"Header": 0, "Subheader": 1, "Body": 2, "Caption": 3}
    elems   = sorted(layout_elements, key=lambda x: r_order.get(x.get("visual_role", ""), 4))

    lines = []
    for item in elems:
        role   = item.get("visual_role", "Unknown")
        weight = ROLE_W.get(role, 1.0)
        flag   = " [OCR?]" if item.get("ocr_min_confidence", 1.0) < 0.75 else ""
        txt    = item.get("text", "").replace("\n", " ")
        lines.append(f'[{role:<12}] (x{weight:.0f}){flag}  "{txt}"')

    sys_p = ("You are an expert brand analyst. Analyse OCR text from an ad poster. "
             "Output STRICTLY as a valid JSON object, no markdown.")
    usr_p = textwrap.dedent(f"""\
        Advertisement text by visual role:
        {chr(10).join(lines)}

        Return exactly:
        {{"cleaned_text":"...","emotional_tone":"...","target_audience":"...",
          "structural_analysis":"...","overall_feedback":"..."}}
    """)

    def _call() -> str:
        if provider.lower() == "deepseek":
            key     = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            payload = json.dumps({"model": model or "deepseek-chat",
                                  "messages": [{"role": "system", "content": sys_p},
                                               {"role": "user",   "content": usr_p}],
                                  "temperature": 0.2, "stream": False}).encode()
            req = urllib.request.Request(
                f"{deepseek_url}/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"}, method="POST")
        else:
            payload = json.dumps({"model": model or "llama3.2",
                                  "messages": [{"role": "system", "content": sys_p},
                                               {"role": "user",   "content": usr_p}],
                                  "format": "json", "stream": False}).encode()
            req = urllib.request.Request(
                f"{ollama_url}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            if provider.lower() == "deepseek":
                return body["choices"][0]["message"]["content"] or ""
            return body.get("message", {}).get("content", "")

    result = {k: None for k in ("cleaned_text", "emotional_tone",
                                 "target_audience", "structural_analysis", "overall_feedback")}
    try:
        raw = _call()
        result["raw"] = raw
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            for k, v in json.loads(m.group()).items():
                if k in result:
                    result[k] = v
    except Exception as exc:
        logger.warning("analyze_sentiment: %s", exc)
        result["raw"] = ""

    result["provider"] = provider
    result["model"]    = model or ("llama3.2" if provider == "ollama" else "deepseek-chat")
    return result


# ===========================================================================
# Video — emotion classification model and datasets
# ===========================================================================

class VideoEmotionClassifier(nn.Module):
    """VideoMAE / TimeSformer backbone + N-class arousal quadrant classifier.

    The backbone is frozen except for its last *unfreeze_last_n* encoder
    layers.  A small MLP head maps the pooled hidden state to *num_classes*.
    """

    def __init__(self, model_name: str = "MCG-NJU/videomae-base",
                 num_classes: int = 4, unfreeze_last_n: int = 2):
        super().__init__()
        from transformers import AutoConfig, TimesformerModel, VideoMAEModel

        cfg  = AutoConfig.from_pretrained(model_name)
        arch = (cfg.architectures or [""])[0]
        self.backbone = (TimesformerModel.from_pretrained(model_name)
                         if ("Timesformer" in arch or "timesformer" in model_name.lower())
                         else VideoMAEModel.from_pretrained(model_name))
        self.model_name = model_name

        for p in self.backbone.parameters():
            p.requires_grad = False
        enc = getattr(self.backbone.encoder, "layer", None) \
           or getattr(self.backbone.encoder, "layers", None)
        if enc is not None:
            for layer in enc[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        hidden = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.5),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

    def count_trainable_params(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Total: %d  Trainable: %d (%.1f%%)", total, trainable,
                    100 * trainable / total)


def build_feature_extractor(model_name: str = "MCG-NJU/videomae-base"):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_name)


def build_model(model_name: str = "MCG-NJU/videomae-base",
                num_classes: int = 4, unfreeze_last_n: int = 2) -> VideoEmotionClassifier:
    return VideoEmotionClassifier(model_name, num_classes, unfreeze_last_n)


def load_model(checkpoint_path: str, model_name: str = "MCG-NJU/videomae-base",
               device: str | None = None) -> VideoEmotionClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoEmotionClassifier(model_name=model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_videos_from_predictions(categories_dir: str = CATEGORIES_DIR,
                                  data_dir: str = DATA_DIR):
    """Return ``(paths, labels, participants, time_offsets)`` from prediction CSVs."""
    samples     = load_participant_labels_from_predictions(data_dir)
    if not samples:
        raise ValueError(f"No prediction CSVs found under '{data_dir}'.")
    video_index = build_video_index(categories_dir)
    if not video_index:
        raise ValueError(f"No video files found under '{categories_dir}'.")

    paths, labels_int, participants, time_offsets = [], [], [], []
    missing = []
    for pid, event_id, t, label in samples:
        vpath = video_index.get(event_id.lower())
        if vpath is None:
            missing.append(event_id)
            continue
        paths.append(vpath); labels_int.append(label)
        participants.append(pid); time_offsets.append(t)

    if missing:
        logger.warning("%d event_ids not matched to video files", len(missing))

    has_time     = any(t is not None for t in time_offsets)
    time_arr     = (np.array([t if t is not None else 0 for t in time_offsets], dtype=np.int64)
                    if has_time else None)
    is_consensus = all(p == "__consensus__" for p in participants)
    return paths, np.array(labels_int, dtype=np.int64), \
           (None if is_consensus else participants), time_arr


class VideoEmotionDataset(Dataset):
    """Live-decode video dataset with optional in-RAM frame caching."""

    def __init__(self, video_paths, labels_int, feature_extractor,
                 num_frames: int = NUM_FRAMES, augment: bool = False,
                 cache_frames: bool = False):
        self.video_paths = video_paths
        self.labels      = np.asarray(labels_int, dtype=np.int64)
        self.fe          = feature_extractor
        self.num_frames  = num_frames
        self.augment     = augment
        self._cache      = ([sample_frames(p, num_frames)
                             for p in tqdm(video_paths, unit="video")]
                            if cache_frames else None)

    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        frames = (self._cache[idx] if self._cache is not None
                  else sample_frames(self.video_paths[idx], self.num_frames))
        if self.augment: frames = augment_frames(frames)
        pv = self.fe(frames, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, torch.tensor(self.labels[idx], dtype=torch.long)


class CachedVideoDataset(Dataset):
    """Fast dataset backed by pre-extracted .npy frame arrays."""

    def __init__(self, video_paths, labels_int, feature_extractor,
                 frames_dir: str, num_frames: int = NUM_FRAMES, augment: bool = False):
        self.video_paths = list(video_paths)
        self.labels      = np.asarray(labels_int, dtype=np.int64)
        self.fe          = feature_extractor
        self.num_frames  = num_frames
        self.augment     = augment
        self._cache: dict[str, str] = {}
        for root, _, files in os.walk(frames_dir):
            for f in files:
                if f.endswith(".npy"):
                    self._cache[Path(f).stem] = os.path.join(root, f)

    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        vpath  = self.video_paths[idx]
        npy    = self._cache.get(Path(vpath).stem)
        frames = list(np.load(npy)) if npy else sample_frames(vpath, self.num_frames)
        if self.augment: frames = augment_frames(frames)
        pv = self.fe(frames, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, torch.tensor(self.labels[idx], dtype=torch.long)


class VideoEmotionWindowDataset(Dataset):
    """Each video expanded into N sliding windows; all share the clip label."""

    def __init__(self, video_paths, labels_int, feature_extractor,
                 window_size_sec: float = WINDOW_SIZE_SEC,
                 stride_sec: float = WINDOW_STRIDE_SEC,
                 num_frames: int = NUM_FRAMES, augment: bool = False):
        self.fe         = feature_extractor
        self.num_frames = num_frames
        self.augment    = augment
        self.samples: list[tuple] = []
        for path, label in zip(video_paths, labels_int):
            for w in get_video_windows(path, window_size_sec, stride_sec, num_frames):
                self.samples.append((path, w["start_frame"], w["end_frame"], int(label)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, sf, ef, label = self.samples[idx]
        frames = sample_frames_from_segment(path, sf, ef, self.num_frames)
        if self.augment: frames = augment_frames(frames)
        pv = self.fe(frames, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, torch.tensor(label, dtype=torch.long)


class TemporalLabelDataset(Dataset):
    """Per-second dataset for oculometry-labelled videos.

    Reads from a per-second .npy cache (``<stem>_sec{N:04d}.npy``) when
    available, falling back to pts-based video seeking.
    """

    def __init__(self, video_paths, time_offsets_sec, labels_int, feature_extractor,
                 window_size_sec: float = 1.0, num_frames: int = NUM_FRAMES,
                 augment: bool = False, frames_dir: str = ""):
        self.video_paths    = list(video_paths)
        self.time_offsets   = np.asarray(time_offsets_sec, dtype=np.int64)
        self.labels         = np.asarray(labels_int, dtype=np.int64)
        self.fe             = feature_extractor
        self.window_sec     = window_size_sec
        self.num_frames     = num_frames
        self.augment        = augment
        self._fps: dict[str, float] = {}
        for p in dict.fromkeys(video_paths):
            fps, _, _ = get_video_info(p)
            self._fps[p] = fps if fps > 0 else 25.0
        self._sec: dict[tuple, str] = {}
        if frames_dir:
            for root, _, files in os.walk(frames_dir):
                for f in files:
                    if f.endswith(".npy") and "_sec" in f:
                        parts = f[:-4].rsplit("_sec", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            self._sec[(parts[0], int(parts[1]))] = os.path.join(root, f)

    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        path  = self.video_paths[idx]
        t_sec = int(self.time_offsets[idx])
        label = self.labels[idx]
        npy   = self._sec.get((Path(path).stem, t_sec))
        if npy:
            frames = list(np.load(npy))
        else:
            fps    = self._fps[path]
            frames = sample_frames_from_segment(
                path, int(round(t_sec * fps)),
                int(round((t_sec + self.window_sec) * fps)), self.num_frames)
        if self.augment: frames = augment_frames(frames)
        pv = self.fe(frames, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, torch.tensor(label, dtype=torch.long)


# ===========================================================================
# Distance / similarity metrics
# ===========================================================================

def jaccard_similarity(set1, set2) -> float:
    """Set-based Jaccard: |intersection| / |union|."""
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2: return 1.0
    return len(s1 & s2) / len(s1 | s2)


def continuous_jaccard_similarity(dict1: dict, dict2: dict) -> float:
    """Weighted Jaccard for frequency dicts: sum(min) / sum(max)."""
    keys = set(dict1) | set(dict2)
    return (sum(min(dict1.get(k, 0), dict2.get(k, 0)) for k in keys) /
            sum(max(dict1.get(k, 0), dict2.get(k, 0)) for k in keys) or 0.0)


def cosine_similarity(vec1, vec2) -> float:
    a, b  = np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def bhattacharyya_distance(hist1, hist2) -> float:
    """Bhattacharyya distance between two normalised histograms."""
    h1, h2 = np.asarray(hist1, float), np.asarray(hist2, float)
    s1, s2 = h1.sum(), h2.sum()
    if s1 > 0: h1 /= s1
    if s2 > 0: h2 /= s2
    return -math.log(float(np.sum(np.sqrt(h1 * h2))) + 1e-10)


def percentage_common_objects(list1, list2) -> float:
    c1    = Counter(list1)
    c2    = Counter(list2)
    common = sum(min(v, c2.get(k, 0)) for k, v in c1.items())
    total  = len(list1) + len(list2) - common
    return common / total if total else 0.0


class ClipTextScore:
    """CLIP text-text cosine similarity scorer.

    Requires ``clip`` (``pip install git+https://github.com/openai/CLIP.git``).
    """

    def __init__(self, device: str | None = None):
        import clip as _clip
        device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._m, _ = _clip.load("ViT-L/14", device=device)
        self._clip  = _clip
        self.device = device

    def __call__(self, text1: str, text2: str) -> float:
        toks  = self._clip.tokenize([text1, text2]).to(self.device)
        with torch.no_grad():
            feats = self._m.encode_text(toks)
        return float(nn.CosineSimilarity(dim=0, eps=1e-6)(feats[0], feats[1]))
