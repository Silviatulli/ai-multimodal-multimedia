"""
multimedia_data_preprocessing.py
---------------------------------
Data loading and preprocessing for all multimedia modalities used in
multimodal affective analysis.

Centralises raw-media I/O so that downstream feature-extraction modules
(`features_multimedia.py`, `fusion_methods.py`, …) share a single,
non-redundant interface.

Modalities
----------
Video
    get_video_info            metadata (fps, frames, duration)
    sample_frames             uniform frame sampling from full clip
    sample_frames_from_segment  pts-based segment sampling
    get_video_windows         sliding-window segmentation
    augment_frames            flip + brightness jitter (training only)
    build_video_index         {stem: path} index for local datasets

Audio
    # TODO: get_audio_info, load_audio, get_audio_windows, augment_audio

Images
    # TODO: load_image, load_image_from_bytes, resize_image,
    #        normalize_image, augment_image, build_image_index

Text
    # TODO: clean_text, chunk_text, tokenize_text

Web / HTML
    # TODO: parse_html, extract_text_from_url

All functions that consume raw media live here.  Feature extraction
(optical flow, spectral descriptors, embeddings, …) belongs in
`features_multimedia.py`, which imports the loaders defined below.

Run directly to preprocess videos streamed from Azure Blob Storage:
    python multimedia_data_preprocessing.py [--n N] [--frames F]
                                            [--window-sec W] [--stride-sec S]
"""

from __future__ import annotations

import glob
import logging
import os

import av
import numpy as np

# Support both package import (relative) and direct script execution (absolute)
try:
    from .config import NUM_FRAMES, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC
except ImportError:
    from config import NUM_FRAMES, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> tuple[float, int, float]:
    """Return ``(fps, total_frames, duration_sec)`` for a video file."""
    container    = av.open(video_path)
    stream       = container.streams.video[0]
    fps          = float(stream.average_rate) if stream.average_rate else 25.0
    total_frames = stream.frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
    container.close()
    duration_sec = total_frames / fps if fps > 0 else 0.0
    return fps, total_frames, duration_sec


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    """Sample *num_frames* frames uniformly from the full video clip."""
    container    = av.open(video_path)
    stream       = container.streams.video[0]
    total_frames = stream.frames

    if total_frames == 0:
        all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
        return [all_frames[i] for i in indices]

    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break
    container.close()

    while len(frames) < num_frames:          # pad with last frame if needed
        frames.append(frames[-1])
    return frames


def sample_frames_from_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_frames: int = NUM_FRAMES,
) -> list[np.ndarray]:
    """Sample *num_frames* uniformly from the ``[start_frame, end_frame)`` range.

    Uses pts-based seeking so only the relevant portion of the video is decoded.
    """
    container = av.open(video_path)
    stream    = container.streams.video[0]
    fps       = float(stream.average_rate) if stream.average_rate else 25.0

    if start_frame > 0 and fps > 0:
        container.seek(int(start_frame / fps * 1_000_000))

    target = set(np.linspace(start_frame, max(start_frame, end_frame - 1), num_frames, dtype=int))
    frames: list[np.ndarray] = []
    for raw in container.decode(video=0):
        if raw.pts is not None:
            fn = int(round(float(raw.pts) * float(stream.time_base) * fps))
        else:
            fn = start_frame
        if fn >= end_frame:
            break
        if fn in target:
            frames.append(raw.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break

    container.close()
    filler = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
    while len(frames) < num_frames:
        frames.append(filler)
    return frames


# ---------------------------------------------------------------------------
# Sliding-window segmentation
# ---------------------------------------------------------------------------

def get_video_windows(
    video_path: str,
    window_size_sec: float = WINDOW_SIZE_SEC,
    stride_sec: float      = WINDOW_STRIDE_SEC,
    num_frames: int        = NUM_FRAMES,
) -> list[dict]:
    """Slide a temporal window over a video and return one metadata dict per window.

    Each dict contains: ``start_sec``, ``end_sec``, ``start_frame``, ``end_frame``.
    When the video is shorter than one window the whole clip is returned as a
    single window.
    """
    fps, total_frames, duration_sec = get_video_info(video_path)
    if fps <= 0 or total_frames == 0:
        return [{"start_sec": 0.0, "end_sec": duration_sec,
                 "start_frame": 0, "end_frame": max(1, total_frames)}]

    win_frames    = max(num_frames, int(round(window_size_sec * fps)))
    stride_frames = max(1, int(round(stride_sec * fps)))

    windows: list[dict] = []
    start = 0
    while start < total_frames:
        end = min(start + win_frames, total_frames)
        windows.append({
            "start_sec":   start / fps,
            "end_sec":     end   / fps,
            "start_frame": start,
            "end_frame":   end,
        })
        if end >= total_frames:
            break
        start += stride_frames
    return windows


# ---------------------------------------------------------------------------
# Augmentation  (training only — label-preserving transforms)
# ---------------------------------------------------------------------------

def augment_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply random horizontal flip and brightness jitter to a frame list.

    Horizontal flipping is label-preserving and effectively doubles the
    training set without extra data collection.  Brightness jitter reduces
    the luminance confound (spurious correlation between clip brightness and
    affective label).
    """
    if np.random.rand() > 0.5:
        frames = [np.fliplr(f).copy() for f in frames]
    factor = np.random.uniform(0.8, 1.2)
    return [np.clip(f * factor, 0, 255).astype(np.uint8) for f in frames]


# ---------------------------------------------------------------------------
# Dataset-level indexing
# ---------------------------------------------------------------------------

def build_video_index(categories_dir: str) -> dict[str, str]:
    """Return ``{stem_lower: full_path}`` for every video file under *categories_dir*.

    Supports ``.mp4``, ``.avi``, ``.mov``, and ``.mkv`` extensions.
    Used by dataset classes to resolve video filenames to absolute paths without
    hardcoding directory layouts.
    """
    index: dict[str, str] = {}
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for p in glob.glob(os.path.join(categories_dir, "**", ext), recursive=True):
            index[os.path.splitext(os.path.basename(p))[0].lower()] = p
    return index


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

# Validation error hierarchy — maps to spec error codes
class AudioValidationError(Exception):
    """Base class for audio input validation failures."""
    code = "PROCESSING_ERROR"

class InvalidFormatError(AudioValidationError):
    """Unsupported extension or file that cannot be decoded as audio."""
    code = "INVALID_FORMAT"

class EmptyFileError(AudioValidationError):
    """File is missing or has zero bytes."""
    code = "EMPTY_FILE"

class FileTooLargeError(AudioValidationError):
    """File exceeds the maximum allowed size."""
    code = "FILE_TOO_LARGE"

class DurationExceededError(AudioValidationError):
    """Decoded audio is longer than the maximum allowed duration."""
    code = "DURATION_EXCEEDED"


# Audio loading constants
AUDIO_TARGET_SR: int   = 16_000
AUDIO_MAX_DURATION_S   = 600.0          # 10 minutes
AUDIO_MAX_FILE_BYTES   = 500 * 1024 * 1024   # 500 MB
AUDIO_TARGET_LUFS      = -23.0
AUDIO_ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac",
    ".ogg", ".opus", ".wma", ".webm",
}


def load_audio(
    path: str,
    target_sr: int = AUDIO_TARGET_SR,
    mono: bool = True,
    lufs_normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """Load and validate an audio file, returning a normalised waveform.

    Validates extension, file size, and decoded duration before returning.
    When *lufs_normalize* is True, the waveform is LUFS-normalised to
    ``AUDIO_TARGET_LUFS`` (skipped for clips too short to measure or silence).

    Args:
        path:           Path to the audio file.
        target_sr:      Target sample rate in Hz (default 16 000).
        mono:           Mix down to mono (default True).
        lufs_normalize: Apply LUFS loudness normalisation (default True).

    Returns:
        ``(waveform, sample_rate)`` — float32 numpy array, always *target_sr*.

    Raises:
        InvalidFormatError, EmptyFileError, FileTooLargeError,
        DurationExceededError.
    """
    import librosa
    from pathlib import Path as _Path

    p = _Path(path)
    if p.suffix.lower() not in AUDIO_ALLOWED_EXTENSIONS:
        raise InvalidFormatError(
            f"unsupported extension {p.suffix!r}; "
            f"allowed: {sorted(AUDIO_ALLOWED_EXTENSIONS)}"
        )
    if not p.is_file() or p.stat().st_size == 0:
        raise EmptyFileError(f"file is missing or empty: {p}")
    if p.stat().st_size > AUDIO_MAX_FILE_BYTES:
        raise FileTooLargeError(
            f"file size {p.stat().st_size} bytes exceeds "
            f"limit of {AUDIO_MAX_FILE_BYTES} bytes"
        )

    try:
        waveform, sr = librosa.load(path, sr=target_sr, mono=mono)
    except Exception as exc:
        logger.exception("load_audio: failed to decode %s", p.name)
        raise InvalidFormatError(f"could not decode audio: {p.name}") from exc

    waveform = np.asarray(waveform, dtype=np.float32)
    duration_s = len(waveform) / sr
    if duration_s > AUDIO_MAX_DURATION_S:
        raise DurationExceededError(
            f"duration {duration_s:.2f}s exceeds limit of {AUDIO_MAX_DURATION_S}s"
        )

    if lufs_normalize:
        waveform = _lufs_normalize(waveform, sr)
    return waveform.astype(np.float32), sr


def _lufs_normalize(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """LUFS-normalise to ``AUDIO_TARGET_LUFS``; return unchanged if not measurable."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sample_rate)
        if len(waveform) < meter.block_size * sample_rate:
            logger.warning("_lufs_normalize: clip too short, skipping")
            return waveform
        loudness = meter.integrated_loudness(waveform)
        if not np.isfinite(loudness):
            logger.warning("_lufs_normalize: non-finite loudness, skipping")
            return waveform
        return pyln.normalize.loudness(waveform, loudness, AUDIO_TARGET_LUFS)
    except ImportError:
        logger.warning("_lufs_normalize: pyloudnorm not installed, skipping")
        return waveform


def get_audio_info(audio_path: str) -> tuple[int, int, float]:
    """Return ``(sample_rate, n_channels, duration_sec)`` for an audio file."""
    container    = av.open(audio_path)
    stream       = container.streams.audio[0]
    sample_rate  = stream.sample_rate or 0
    n_channels   = stream.channels or 1
    duration_sec = (
        float(stream.duration * stream.time_base)
        if stream.duration else 0.0
    )
    container.close()
    return sample_rate, n_channels, duration_sec


def get_audio_windows(
    audio_path: str,
    window_size_sec: float = WINDOW_SIZE_SEC,
    stride_sec: float = WINDOW_STRIDE_SEC,
    target_sr: int = AUDIO_TARGET_SR,
    mono: bool = True,
) -> list[dict]:
    """Slide a window over an audio file and return one dict per segment.

    Each dict contains: ``start_sec``, ``end_sec``, ``waveform`` (float32).
    The last window is zero-padded to *window_size_sec* if the clip is shorter.
    """
    waveform, _ = load_audio(audio_path, target_sr=target_sr, mono=mono)
    total_samples  = waveform.shape[-1]
    win_samples    = int(window_size_sec * target_sr)
    stride_samples = max(1, int(stride_sec * target_sr))

    windows: list[dict] = []
    start = 0
    while start < total_samples:
        end     = min(start + win_samples, total_samples)
        segment = waveform[..., start:end]
        if segment.shape[-1] < win_samples:
            pad     = win_samples - segment.shape[-1]
            segment = np.pad(segment, (0, pad))
        windows.append({
            "start_sec": start / target_sr,
            "end_sec":   end   / target_sr,
            "waveform":  segment,
        })
        if end >= total_samples:
            break
        start += stride_samples
    return windows


def augment_audio(
    waveform: np.ndarray,
    gain_range: tuple = (0.8, 1.2),
    noise_std: float = 0.005,
) -> np.ndarray:
    """Apply random gain jitter and additive Gaussian noise to a waveform.

    Both transforms are label-preserving and help reduce recording-level
    loudness as a confound in affective analysis.
    """
    waveform = waveform * np.random.uniform(*gain_range)
    waveform = waveform + np.random.normal(0, noise_std, waveform.shape)
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# TODO: add image preprocessing functions
# Suggested: load_image, load_image_from_bytes, resize_image,
#            normalize_image, augment_image, build_image_index


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise Unicode (NFC), collapse whitespace, and strip control characters.

    Handles typical OCR noise: arbitrary line breaks, missing punctuation gaps,
    and non-printable control characters are all removed or collapsed.
    """
    import re
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    tokenizer=None,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* tokens.

    When *tokenizer* is ``None`` (default), splits by whitespace-separated
    words.  Pass a HuggingFace tokenizer to split by subword tokens instead,
    which respects model vocabulary boundaries.

    Args:
        text:       Input text string.
        chunk_size: Maximum number of tokens (words or subwords) per chunk.
        overlap:    Number of tokens shared between consecutive chunks.
        tokenizer:  Optional HuggingFace tokenizer.

    Returns:
        List of text strings, each at most *chunk_size* tokens long.
    """
    text = clean_text(text)
    if tokenizer is not None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunks.append(tokenizer.decode(token_ids[start:end]))
            if end == len(token_ids):
                break
            start += chunk_size - overlap
        return chunks

    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
) -> dict:
    """Run a HuggingFace *tokenizer* on *text* and return the encoding dict.

    Returns ``input_ids``, ``attention_mask``, and any other fields the
    tokenizer produces, all as plain Python lists (no framework tensors).
    """
    text = clean_text(text)
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,
    )
    return dict(encoding)


# ---------------------------------------------------------------------------
# Web / HTML
# ---------------------------------------------------------------------------
# TODO: add web/HTML preprocessing functions
# Suggested: parse_html, extract_text_from_url


# ---------------------------------------------------------------------------
# Video — pre-extraction modes (for Azure ML parallel jobs)
# ---------------------------------------------------------------------------

def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize *frame* (H×W×3) to *size*×*size* using bilinear interpolation."""
    from PIL import Image
    return np.array(
        Image.fromarray(frame).resize((size, size), Image.BILINEAR),
        dtype=np.uint8,
    )


def preprocess_video_whole(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    size: int = 224,
) -> np.ndarray:
    """Decode *num_frames* uniformly-sampled frames from the full clip.

    Returns a uint8 array of shape ``[T, H, W, C]`` (T = num_frames,
    H = W = size).  This is the format expected by ``CachedVideoDataset``
    when frames are pre-extracted for the training loop.
    """
    raw_frames = sample_frames(video_path, num_frames=num_frames)
    return np.stack([_resize_frame(f, size) for f in raw_frames], axis=0)


def preprocess_video_per_second(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    size: int = 224,
) -> list[dict]:
    """Decode *num_frames* from each non-overlapping 1-second window of the clip.

    Returns a list of dicts, one per second, each with keys:
    - ``second``      — 0-based integer second index
    - ``start_frame`` — first frame index of the window
    - ``end_frame``   — last frame index (exclusive)
    - ``frames``      — uint8 array ``[T, H, W, C]``

    Used by ``TemporalLabelDataset`` when oculometry labels are at per-second
    granularity.  Mirrors the ``per_second`` mode of
    ``scripts/preprocess_videos.py`` in ai-occulo-video-insights.
    """
    fps, total_frames, _ = get_video_info(video_path)
    if fps <= 0 or total_frames == 0:
        return []

    frames_per_sec = max(num_frames, int(round(fps)))
    n_seconds      = max(1, int(total_frames / fps))

    results: list[dict] = []
    for sec in range(n_seconds):
        start_frame = int(round(sec * fps))
        end_frame   = min(int(round((sec + 1) * fps)), total_frames)
        raw_frames  = sample_frames_from_segment(
            video_path, start_frame, end_frame, num_frames=num_frames
        )
        arr = np.stack([_resize_frame(f, size) for f in raw_frames], axis=0)
        results.append({
            "second":      sec,
            "start_frame": start_frame,
            "end_frame":   end_frame,
            "frames":      arr,
        })
    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import io
    import json
    import sys
    from pathlib import Path

    from azure_blob import build_client, list_video_blobs, output_prefix_for, upload_blob
    from config import AZURE_CONTAINER_NAME, AZURE_OUTPUT_PREFIX, AZURE_VIDEO_PREFIX  # type: ignore[no-redef]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Preprocess Azure videos and write outputs back to silver.",
    )
    p.add_argument("--n",          type=int,   default=3,
                   help="number of videos to process (default: 3)")
    p.add_argument("--frames",     type=int,   default=NUM_FRAMES,
                   help=f"frames sampled per clip (default: {NUM_FRAMES})")
    p.add_argument("--window-sec", type=float, default=WINDOW_SIZE_SEC,
                   help=f"sliding window size in seconds (default: {WINDOW_SIZE_SEC})")
    p.add_argument("--stride-sec", type=float, default=WINDOW_STRIDE_SEC,
                   help=f"sliding window stride in seconds (default: {WINDOW_STRIDE_SEC})")
    args = p.parse_args()

    client = build_client()
    logger.info("Input     : %s/%s", AZURE_CONTAINER_NAME, AZURE_VIDEO_PREFIX)
    logger.info("Output    : %s/%s", AZURE_CONTAINER_NAME, AZURE_OUTPUT_PREFIX)
    logger.info("Videos    : %d", args.n)

    videos = list_video_blobs(client, args.n)
    if not videos:
        logger.error("No video blobs found under '%s/%s'.",
                     AZURE_CONTAINER_NAME, AZURE_VIDEO_PREFIX)
        sys.exit(1)

    for blob_name, url in videos:
        short  = Path(blob_name).name
        prefix = output_prefix_for(blob_name)   # silver/NEURO/.../preprocessed/<cat>/<stem>
        logger.info("── %s  →  %s/", short, prefix)

        # ── metadata ────────────────────────────────────────────────────────
        fps, total_frames, duration = get_video_info(url)
        logger.info("   info       fps=%.2f  frames=%d  duration=%.2fs",
                    fps, total_frames, duration)

        # ── sampled frames  →  frames.npy ───────────────────────────────────
        frames = sample_frames(url, num_frames=args.frames)
        buf = io.BytesIO()
        np.save(buf, np.stack(frames))           # shape [N, H, W, 3]
        upload_blob(client, f"{prefix}/frames.npy", buf.getvalue())
        logger.info("   frames.npy      %d frames  shape=%s", len(frames), frames[0].shape)

        # ── augmented frames  →  frames_aug.npy ─────────────────────────────
        augmented = augment_frames(frames)
        buf = io.BytesIO()
        np.save(buf, np.stack(augmented))
        upload_blob(client, f"{prefix}/frames_aug.npy", buf.getvalue())
        logger.info("   frames_aug.npy  %d frames", len(augmented))

        # ── sliding windows  →  windows.json ────────────────────────────────
        windows = get_video_windows(url, window_size_sec=args.window_sec,
                                    stride_sec=args.stride_sec, num_frames=args.frames)
        windows_bytes = json.dumps(windows, indent=2).encode()
        upload_blob(client, f"{prefix}/windows.json", windows_bytes,
                    content_type="application/json")
        logger.info("   windows.json    %d windows  (%.1fs / stride %.1fs)",
                    len(windows), args.window_sec, args.stride_sec)

    logger.info("Done.")
