"""
schema.py
---------
Shared data models for the multimodal affective analysis pipeline.

Consolidates output schemas from all modalities so downstream modules
(features_multimedia.py, fusion_methods.py, …) share a single type
system without circular imports.

Audio schemas ported from ai-habs_audio_insights/src/schema.py.
OCR / layout schemas ported from ai-text_insights/src/models/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import numpy as np
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Audio — metadata and transcript
# ---------------------------------------------------------------------------

class ModelsInfo(BaseModel):
    whisper: str
    clap: str
    llm: str | None = None


class Metadata(BaseModel):
    file_name: str
    duration_seconds: float
    sample_rate_hz: int
    detected_language: str | None
    service_version: str
    processing_time_seconds: float
    models: ModelsInfo


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str


class Transcript(BaseModel):
    full_text: str
    segments: list[TranscriptSegment]
    word_count: int
    estimated_speech_rate_wpm: float | None


# ---------------------------------------------------------------------------
# Audio — CLAP rubric scores
# ---------------------------------------------------------------------------

class ClapSimilarity(BaseModel):
    """Sigmoid-combined cosine similarity in [0, 1] for one rubric dimension."""
    score: float


class ClapRubricSimilarities(BaseModel):
    """Seven perceptual dimensions scored by CLAP audio-text embeddings."""
    attention: ClapSimilarity
    attractiveness: ClapSimilarity
    engagement: ClapSimilarity
    cognitive_clarity: ClapSimilarity
    memorization: ClapSimilarity
    differentiation: ClapSimilarity
    comfort: ClapSimilarity


# ---------------------------------------------------------------------------
# Audio — transcript LLM scores and acoustic features
# ---------------------------------------------------------------------------

class TranscriptScores(BaseModel):
    """Per-dimension LLM scores in [0, 100] derived from transcript text."""
    attention: float
    attractiveness: float
    engagement: float
    cognitive_clarity: float
    memorization: float
    differentiation: float
    comfort: float


class AcousticFeatures(BaseModel):
    """13 librosa-based acoustic descriptors extracted from a waveform."""
    rms_energy_mean: float
    rms_energy_max: float
    initial_energy_ratio: float
    energy_slope: float
    peak_position_ratio: float
    energy_variance: float
    loudness_dynamic_range_db: float
    loudness_shape_5seg: list[float]
    zero_crossing_rate_mean: float
    zero_crossing_rate_var: float
    spectral_centroid_mean_hz: float
    onset_density_per_second: float
    pause_ratio: float


# ---------------------------------------------------------------------------
# Audio — blended insight scores
# ---------------------------------------------------------------------------

class InsightDetail(BaseModel):
    score: float
    level: Literal["low", "moderate", "high", "very_high"]
    explanation: str | None
    components: dict[str, float]


class InsightScores(BaseModel):
    """Seven 0-100 insight scores blended from CLAP and LLM signals."""
    attention_impact: InsightDetail
    attractiveness_pleasure: InsightDetail
    engagement_desire: InsightDetail
    cognitive_clarity: InsightDetail
    rejection: InsightDetail
    memorization: InsightDetail
    differentiation: InsightDetail


class AudioAnalysisResult(BaseModel):
    """Top-level result object returned by the audio analysis pipeline."""
    metadata: Metadata
    transcript: Transcript
    clap_rubric_similarities: ClapRubricSimilarities
    transcript_scores: TranscriptScores | None
    acoustic_features: AcousticFeatures
    insight_scores: InsightScores


# ---------------------------------------------------------------------------
# OCR / text — bounding box geometry
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(self.x_max - self.x_min, 1.0)

    @property
    def height(self) -> float:
        return max(self.y_max - self.y_min, 1.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    def vertical_gap(self, other: BBox) -> float:
        """Vertical distance from bottom of self to top of other (0 if overlapping)."""
        return max(0.0, other.y_min - self.y_max)

    def horizontal_overlap_ratio(self, other: BBox) -> float:
        overlap = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        return overlap / min(self.width, other.width)

    def center_x_distance(self, other: BBox) -> float:
        return abs(self.center[0] - other.center[0])

    @classmethod
    def union(cls, boxes: list[BBox]) -> BBox:
        return cls(
            min(b.x_min for b in boxes),
            min(b.y_min for b in boxes),
            max(b.x_max for b in boxes),
            max(b.y_max for b in boxes),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


# ---------------------------------------------------------------------------
# OCR / text — detection and region models
# ---------------------------------------------------------------------------

@dataclass
class OcrDetection:
    """Single word detected by an OCR engine with polygon and confidence."""
    text: str
    polygon: list[list[float]]                      # 4 corner points [[x,y], …]
    confidence: float
    center: tuple[float, float]                     # midpoint (cx, cy)
    bbox: tuple[float, float, float, float]         # (min_x, min_y, max_x, max_y)
    features: Optional[np.ndarray] = None
    fresco_text: Optional[str] = None
    feature_source: Optional[str] = None
    source: str = "base"


@dataclass
class TextRegion:
    """One OCR-detected word with its visual embedding and optional location."""
    text: str
    features: np.ndarray          # shape (embedding_dim,)
    bbox: Optional[tuple[float, float, float, float]] = None   # (x1, y1, x2, y2)
    center: Optional[tuple[float, float]] = None               # (cx, cy)
    confidence: float = 1.0

    def get_center(self) -> Optional[tuple[float, float]]:
        if self.center is not None:
            return self.center
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return None


# ---------------------------------------------------------------------------
# OCR / text — layout typed dicts
# ---------------------------------------------------------------------------

class LineSummary(TypedDict):
    """One reconstructed OCR line summarised for feature extraction."""
    text: str
    embedding: np.ndarray
    center: tuple[float, float]
    bbox: BBox
    word_count: int
    has_features: bool
    image_width: int
    image_height: int
    min_confidence: float
    mean_confidence: float
    bg_contrast: float


class BlockSummary(TypedDict):
    """A paragraph-like group of adjacent LineSummary entries."""
    text: str
    center: tuple[float, float]
    bbox: BBox
    word_count: int
    line_count: int
    mean_line_height: float
    embedding: np.ndarray
    has_features: bool
    rel_area: float
    rel_y: float
    rel_x: float
    aspect_ratio: float
    char_density: float
    inv_word_count: float
    bg_contrast: float
    image_width: int
    image_height: int
    source_indices: list[int]


class LayoutItem(TypedDict):
    """One line's final clustering result: text + assigned visual role."""
    text: str
    visual_role: str
    position_x: float
    position_y: float
    bbox: list[float]
    word_count: int
    rel_y: float
    has_feature_embeddings: bool
    ocr_min_confidence: float
    ocr_mean_confidence: float
