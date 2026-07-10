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
# TODO: add audio preprocessing functions
# Suggested: get_audio_info, load_audio, get_audio_windows, augment_audio


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# TODO: add image preprocessing functions
# Suggested: load_image, load_image_from_bytes, resize_image,
#            normalize_image, augment_image, build_image_index


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
# TODO: add text preprocessing functions
# Suggested: clean_text, chunk_text, tokenize_text


# ---------------------------------------------------------------------------
# Web / HTML
# ---------------------------------------------------------------------------
# TODO: add web/HTML preprocessing functions
# Suggested: parse_html, extract_text_from_url


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
