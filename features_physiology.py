"""
features_physiology.py
----------------------
Feature extraction methods for physiological signals.

Implements feature extraction for biosignals such as EEG, ECG, EDA, EMG,
PPG, and respiration, intended for affective computing and psychophysiology
research.

EEG  (deep features via BENDR)
    load_bendr_encoder          load the pretrained BENDR contextual encoder
    extract_eeg_bendr_features  encode raw EEG epochs → fixed-length embeddings
    load_eeg_from_bytes         decode an EEG CSV blob → (n_channels, n_samples)

Non-EEG physiological (precomputed, loaded from Azure)
    load_precomputed_physio     load a precomputed feature row (pupil, ECG, EDA …)
                                from a CSV file path or raw bytes

Signal utilities
    bandpass_filter             zero-phase Butterworth bandpass
    zscore_signal               z-score normalisation
    epoch_signal                cut a continuous signal into fixed-length epochs
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal utilities  (shared across modalities)
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to *signal*.

    Args:
        signal:  1-D or 2-D ``(channels, samples)`` float array.
        lowcut:  Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs:      Sampling rate in Hz.
        order:   Filter order (default 4).

    Returns:
        Filtered signal of the same shape as input.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    lo  = max(lowcut  / nyq, 1e-6)
    hi  = min(highcut / nyq, 1.0 - 1e-6)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(np.float32)
    return np.stack(
        [sosfiltfilt(sos, signal[ch]) for ch in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


def zscore_signal(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z-score normalise *signal* along *axis*."""
    mean = np.mean(signal, axis=axis, keepdims=True)
    std  = np.std(signal,  axis=axis, keepdims=True)
    return ((signal - mean) / (std + 1e-10)).astype(np.float32)


def epoch_signal(
    signal: np.ndarray,
    fs: float,
    epoch_sec: float,
    overlap_sec: float = 0.0,
) -> np.ndarray:
    """Slice *signal* into fixed-length overlapping epochs.

    Args:
        signal:      1-D ``(samples,)`` or 2-D ``(channels, samples)`` array.
        fs:          Sampling rate in Hz.
        epoch_sec:   Epoch length in seconds.
        overlap_sec: Overlap between consecutive epochs in seconds.

    Returns:
        3-D ``(n_epochs, channels, samples_per_epoch)`` for 2-D input, or
        2-D ``(n_epochs, samples_per_epoch)`` for 1-D input.
    """
    samples_per_epoch = int(epoch_sec * fs)
    stride            = max(1, int((epoch_sec - overlap_sec) * fs))
    is_1d             = signal.ndim == 1
    data              = signal[np.newaxis, :] if is_1d else signal
    n_ch, n_samp      = data.shape
    starts            = range(0, n_samp - samples_per_epoch + 1, stride)
    epochs            = np.stack(
        [data[:, s:s + samples_per_epoch] for s in starts], axis=0
    )   # (n_epochs, n_ch, samples_per_epoch)
    return epochs[:, 0, :] if is_1d else epochs


# ---------------------------------------------------------------------------
# EEG  —  BENDR pretrained encoder
# ---------------------------------------------------------------------------

def load_bendr_encoder(
    pretrained: bool = True,
    device: str | None = None,
    n_channels: int = 64,
):
    """Load the BENDR contextual encoder from braindecode.

    BENDR (Kostas & Bhatt, 2022) is a contrastive self-supervised model
    pre-trained on large EEG corpora.  The encoder maps raw EEG epochs
    ``(batch, channels, time)`` to fixed-length contextual embeddings.

    Args:
        pretrained:  Download and load the pretrained weights (default True).
        device:      Torch device string.  Auto-detected when ``None``.
        n_channels:  Number of EEG channels expected by the model (default 64).

    Returns:
        A frozen ``torch.nn.Module`` BENDR encoder in eval mode.

    Requires ``braindecode >= 0.8`` (``pip install braindecode``).
    """
    import torch

    if device is None:
        device = "mps" if torch.backends.mps.is_available() \
            else "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # braindecode 0.8+ exposes BENDR directly
        from braindecode.models import BENDR as _BENDR  # type: ignore[import]

        encoder = _BENDR(
            n_channels=n_channels,
            encoder_h=512,
            contextualizer_hidden=3076,
            projection_head=False,
            pretrained=pretrained,
        )
    except (ImportError, TypeError):
        # Fallback: load encoder-only component for older braindecode versions
        try:
            from braindecode.models.bendr import (  # type: ignore[import]
                ConvEncoderBENDR,
            )
            encoder = ConvEncoderBENDR(in_features=n_channels, encoder_h=512)
            if pretrained:
                logger.warning(
                    "load_bendr_encoder: automatic pretrained weights not "
                    "available for this braindecode version — encoder is "
                    "randomly initialised.  Download weights manually and "
                    "call encoder.load_state_dict()."
                )
        except ImportError:
            raise ImportError(
                "braindecode is required for BENDR feature extraction. "
                "Install with: pip install braindecode"
            )

    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    logger.info("Loaded BENDR encoder on %s (pretrained=%s)", device, pretrained)
    return encoder, device


def extract_eeg_bendr_features(
    eeg: np.ndarray,
    fs: float,
    encoder,
    device: str = "cpu",
    epoch_sec: float = 4.0,
    overlap_sec: float = 0.0,
    batch_size: int = 16,
    pool: str = "mean",
) -> np.ndarray:
    """Encode raw EEG into contextual embeddings using a frozen BENDR encoder.

    The signal is first epoched, then passed through the encoder in batches.
    Per-epoch embeddings are pooled (mean or max) across the time dimension
    to produce one fixed-length vector per epoch.

    Args:
        eeg:         2-D ``(n_channels, n_samples)`` float32 array in µV.
        fs:          Sampling rate in Hz.
        encoder:     Frozen BENDR encoder (from ``load_bendr_encoder``).
        device:      Device string matching the encoder.
        epoch_sec:   Epoch length in seconds (default 4 s).
        overlap_sec: Epoch overlap in seconds (default 0).
        batch_size:  GPU/CPU batch size for encoding (default 16).
        pool:        Temporal pooling strategy — ``"mean"`` or ``"max"``.

    Returns:
        2-D float32 array ``(n_epochs, embedding_dim)``.
    """
    import torch

    # Epoch the signal: (n_epochs, n_channels, samples_per_epoch)
    epochs = epoch_signal(eeg, fs, epoch_sec=epoch_sec, overlap_sec=overlap_sec)
    if epochs.ndim == 2:               # single-channel fallback
        epochs = epochs[:, np.newaxis, :]

    n_epochs = epochs.shape[0]
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_epochs, batch_size):
            batch = torch.tensor(
                epochs[start:start + batch_size], dtype=torch.float32
            ).to(device)               # (B, C, T)

            out = encoder(batch)       # shape depends on BENDR version:
                                       # (B, D, T') or (B, D) or tuple

            # Unwrap tuple outputs (some versions return (z, c))
            if isinstance(out, (tuple, list)):
                out = out[0]

            # Pool temporal dimension if present
            if out.ndim == 3:
                out = out.mean(dim=-1) if pool == "mean" else out.max(dim=-1).values

            embeddings.append(out.cpu().numpy().astype(np.float32))

    return np.concatenate(embeddings, axis=0)   # (n_epochs, D)


# ---------------------------------------------------------------------------
# EEG  —  data loading
# ---------------------------------------------------------------------------

def load_eeg_from_bytes(data: bytes, fs: float) -> np.ndarray:
    """Decode a raw EEG CSV blob into a ``(n_channels, n_samples)`` array.

    Expected CSV format:
        - One column named ``time`` or ``timestamp`` (optional, dropped).
        - Remaining columns are EEG channels in µV (any name).
        - Rows are time samples.

    Args:
        data: Raw bytes of the CSV file (e.g. from a blob download).
        fs:   Sampling rate in Hz (used only for logging; not resampled here).

    Returns:
        Float32 array ``(n_channels, n_samples)``.
    """
    df = pd.read_csv(io.BytesIO(data))
    df.columns = df.columns.str.strip().str.lower()

    # Drop time column if present
    for tcol in ("time", "timestamp", "t"):
        if tcol in df.columns:
            df = df.drop(columns=[tcol])

    eeg = df.values.T.astype(np.float32)   # (n_channels, n_samples)
    logger.debug("load_eeg_from_bytes: %d channels × %d samples @ %.1f Hz",
                 eeg.shape[0], eeg.shape[1], fs)
    return eeg


def load_eeg_from_path(csv_path: str) -> np.ndarray:
    """Load a raw EEG CSV from the local filesystem.

    Same column convention as ``load_eeg_from_bytes``.
    """
    with open(csv_path, "rb") as fh:
        return load_eeg_from_bytes(fh.read(), fs=0)


# ---------------------------------------------------------------------------
# Non-EEG physiological  —  precomputed feature loader
# ---------------------------------------------------------------------------

def load_precomputed_physio(source) -> np.ndarray:
    """Load a row of precomputed non-EEG physiological features.

    *source* can be:
    - A file path string (local CSV or NPY)
    - Raw bytes of a CSV or NPY file (from a blob download)
    - A ``pandas.DataFrame`` row (already loaded)
    - A 1-D ``numpy.ndarray`` (already a feature vector)

    The CSV is expected to have a single data row (or the first non-header row
    is used) with numeric feature columns.  An ``stimulus_id`` / ``participant``
    / ``time`` column is silently dropped.

    Returns:
        1-D float32 numpy array of feature values.
    """
    _ID_COLS = {"stimulus_id", "participant_id", "participant", "time", "timestamp", "id"}

    # Already a numpy array
    if isinstance(source, np.ndarray):
        return source.ravel().astype(np.float32)

    # DataFrame row
    if isinstance(source, (pd.Series, pd.DataFrame)):
        df = source if isinstance(source, pd.DataFrame) else source.to_frame().T
        df = df.drop(columns=[c for c in df.columns if c.lower() in _ID_COLS],
                     errors="ignore")
        return df.iloc[0].values.astype(np.float32)

    # Bytes (CSV or NPY)
    if isinstance(source, (bytes, bytearray)):
        # Try NPY first
        try:
            arr = np.load(io.BytesIO(source))
            return arr.ravel().astype(np.float32)
        except Exception:
            pass
        df = pd.read_csv(io.BytesIO(source))
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=[c for c in df.columns if c in _ID_COLS], errors="ignore")
        return df.iloc[0].values.astype(np.float32)

    # File path string
    path = str(source)
    if path.endswith(".npy"):
        return np.load(path).ravel().astype(np.float32)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(columns=[c for c in df.columns if c in _ID_COLS], errors="ignore")
    return df.iloc[0].values.astype(np.float32)
