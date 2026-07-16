"""
pipeline_valence_arousal.py
----------------------------
End-to-end multimodal pipeline for continuous valence and arousal prediction.

Modalities
----------
    Audio         → acoustic features (librosa, 13 descriptors)
    EEG           → deep embeddings (BENDR pretrained contextual encoder)
    Non-EEG physio→ precomputed feature vectors (pupil, ECG, EDA, …)

All data is streamed from Azure Blob Storage (silver container) via
DefaultAzureCredential or a connection-string.

Azure data layout (silver container)
-------------------------------------
    NEURO/protocolimage/categories/<category>/<stimulus>.wav       ← audio
    NEURO/physiological/eeg/<participant>/<session>/<stimulus>.csv ← raw EEG
    NEURO/physiological/precomputed/<participant>/<stimulus>.csv   ← physio
    NEURO/labels/valence_arousal.csv                               ← labels

Labels CSV columns
------------------
    participant_id, stimulus_id, valence, arousal
    (valence and arousal in [-1, 1] or [1, 9] — either scale is accepted)

Usage
-----
    python pipeline_valence_arousal.py                    # default (RF, 5-fold)
    python pipeline_valence_arousal.py --model mlp        # MLP regressor
    python pipeline_valence_arousal.py --n 50             # limit to 50 participants
    python pipeline_valence_arousal.py --eeg-fs 250       # EEG sampling rate
    python pipeline_valence_arousal.py --epoch-sec 4      # EEG epoch length (s)
    python pipeline_valence_arousal.py --no-eeg           # audio + physio only
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy local imports
# ---------------------------------------------------------------------------

def _import_features_multimedia():
    try:
        from features_multimedia import extract_acoustic_features
        from multimedia_data_preprocessing import load_audio
    except ImportError:
        from .features_multimedia import extract_acoustic_features  # type: ignore[no-redef]
        from .multimedia_data_preprocessing import load_audio  # type: ignore[no-redef]
    return extract_acoustic_features, load_audio


def _import_features_physiology():
    try:
        from features_physiology import (
            extract_eeg_bendr_features,
            load_bendr_encoder,
            load_eeg_from_bytes,
            load_precomputed_physio,
        )
    except ImportError:
        from .features_physiology import (  # type: ignore[no-redef]
            extract_eeg_bendr_features,
            load_bendr_encoder,
            load_eeg_from_bytes,
            load_precomputed_physio,
        )
    return load_bendr_encoder, extract_eeg_bendr_features, load_eeg_from_bytes, load_precomputed_physio


def _import_fusion():
    try:
        from fusion_methods import early_fusion_concat
    except ImportError:
        from .fusion_methods import early_fusion_concat  # type: ignore[no-redef]
    return early_fusion_concat


def _import_azure():
    try:
        from azure_blob import build_client, make_sas_url
        from config import (
            AZURE_CONTAINER_NAME, AZURE_EEG_PREFIX, AZURE_LABELS_PREFIX,
            AZURE_PHYSIO_PREFIX, AZURE_VIDEO_PREFIX,
        )
    except ImportError:
        from .azure_blob import build_client, make_sas_url  # type: ignore[no-redef]
        from .config import (  # type: ignore[no-redef]
            AZURE_CONTAINER_NAME, AZURE_EEG_PREFIX, AZURE_LABELS_PREFIX,
            AZURE_PHYSIO_PREFIX, AZURE_VIDEO_PREFIX,
        )
    return (build_client, make_sas_url, AZURE_CONTAINER_NAME,
            AZURE_VIDEO_PREFIX, AZURE_EEG_PREFIX,
            AZURE_PHYSIO_PREFIX, AZURE_LABELS_PREFIX)


# ===========================================================================
# Azure data loaders
# ===========================================================================

def _load_labels(container_client, labels_prefix: str) -> pd.DataFrame:
    """Download the valence/arousal labels CSV from Azure.

    Returns a DataFrame with columns:
        participant_id, stimulus_id, valence, arousal
    """
    blobs = [b.name for b in container_client.list_blobs(name_starts_with=labels_prefix)
             if b.name.endswith(".csv")]
    if not blobs:
        raise FileNotFoundError(f"No label CSV found under '{labels_prefix}'")

    data = container_client.download_blob(blobs[0]).readall()
    df   = pd.read_csv(io.BytesIO(data))
    df.columns = df.columns.str.strip().str.lower()

    for col in ("participant_id", "stimulus_id", "valence", "arousal"):
        if col not in df.columns:
            raise ValueError(f"Labels CSV missing required column '{col}'")

    for col in ("valence", "arousal"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].max() > 1.5:
            df[col] = (df[col] - 5.0) / 4.0   # [1,9] → [-1,1]

    df = df.dropna(subset=["valence", "arousal"])
    log.info("Loaded %d label rows from '%s'", len(df), blobs[0])
    return df


def _load_audio_features(
    container_client,
    video_prefix: str,
    stimulus_ids: list[str],
    make_sas_url_fn,
    client,
    container_name: str,
) -> dict[str, np.ndarray]:
    """Stream audio blobs and extract 13 librosa-based acoustic features.

    Returns ``{stimulus_id: feature_vector}`` (length-13 float32 arrays).
    """
    extract_acoustic_features, load_audio = _import_features_multimedia()

    exts  = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    blobs = {
        Path(b.name).stem.lower(): b.name
        for b in container_client.list_blobs(name_starts_with=video_prefix)
        if Path(b.name).suffix.lower() in exts
    }

    result: dict[str, np.ndarray] = {}
    for sid in stimulus_ids:
        blob_name = blobs.get(sid.lower())
        if blob_name is None:
            log.warning("Audio not found for stimulus '%s'", sid)
            continue
        try:
            url          = make_sas_url_fn(client, blob_name)
            waveform, sr = load_audio(url)
            feats        = extract_acoustic_features(waveform, sr)
            result[sid]  = _acoustic_to_vector(feats)
        except Exception as exc:
            log.warning("Audio feature extraction failed for '%s': %s", sid, exc)

    log.info("Audio features extracted for %d / %d stimuli",
             len(result), len(stimulus_ids))
    return result


def _acoustic_to_vector(feats) -> np.ndarray:
    """Flatten an AcousticFeatures object to a 1-D float32 vector."""
    return np.array([
        feats.rms_energy_mean, feats.rms_energy_max, feats.initial_energy_ratio,
        feats.energy_slope, feats.peak_position_ratio, feats.energy_variance,
        feats.loudness_dynamic_range_db, *feats.loudness_shape_5seg,
        feats.zero_crossing_rate_mean, feats.zero_crossing_rate_var,
        feats.spectral_centroid_mean_hz, feats.onset_density_per_second,
        feats.pause_ratio,
    ], dtype=np.float32)


def _load_eeg_features(
    container_client,
    eeg_prefix: str,
    stimulus_ids: list[str],
    encoder,
    device: str,
    eeg_fs: float,
    epoch_sec: float,
) -> dict[str, np.ndarray]:
    """Download raw EEG CSVs, encode with BENDR, return mean embedding per stimulus.

    Returns ``{stimulus_id: embedding_vector}`` (mean-pooled across epochs).
    """
    _, extract_eeg_bendr_features, load_eeg_from_bytes, _ = _import_features_physiology()

    blobs: dict[str, str] = {
        Path(b.name).stem.lower(): b.name
        for b in container_client.list_blobs(name_starts_with=eeg_prefix)
        if b.name.endswith(".csv")
    }

    result: dict[str, np.ndarray] = {}
    for sid in stimulus_ids:
        blob_name = blobs.get(sid.lower())
        if blob_name is None:
            log.warning("EEG not found for stimulus '%s'", sid)
            continue
        try:
            raw        = container_client.download_blob(blob_name).readall()
            eeg        = load_eeg_from_bytes(raw, eeg_fs)
            eeg        = _preprocess_eeg(eeg, eeg_fs)
            embs       = extract_eeg_bendr_features(eeg, eeg_fs, encoder, device,
                                                    epoch_sec=epoch_sec)
            result[sid] = embs.mean(axis=0)   # mean across epochs → (D,)
        except Exception as exc:
            log.warning("EEG feature extraction failed for '%s': %s", sid, exc)

    log.info("EEG features extracted for %d / %d stimuli",
             len(result), len(stimulus_ids))
    return result


def _preprocess_eeg(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Standard EEG preprocessing: bandpass 0.5-45 Hz + z-score per channel."""
    try:
        from features_physiology import bandpass_filter, zscore_signal
    except ImportError:
        from .features_physiology import bandpass_filter, zscore_signal  # type: ignore[no-redef]
    return zscore_signal(bandpass_filter(eeg, 0.5, 45.0, fs))


def _load_physio_features(
    container_client,
    physio_prefix: str,
    stimulus_ids: list[str],
) -> dict[str, np.ndarray]:
    """Download precomputed non-EEG physio feature CSVs/NPYs from Azure.

    Returns ``{stimulus_id: feature_vector}``.
    """
    _, _, _, load_precomputed_physio = _import_features_physiology()

    blobs: dict[str, str] = {
        Path(b.name).stem.lower(): b.name
        for b in container_client.list_blobs(name_starts_with=physio_prefix)
        if b.name.endswith((".csv", ".npy"))
    }

    result: dict[str, np.ndarray] = {}
    for sid in stimulus_ids:
        blob_name = blobs.get(sid.lower())
        if blob_name is None:
            log.warning("Precomputed physio not found for stimulus '%s'", sid)
            continue
        try:
            raw         = container_client.download_blob(blob_name).readall()
            result[sid] = load_precomputed_physio(raw)
        except Exception as exc:
            log.warning("Physio load failed for '%s': %s", sid, exc)

    log.info("Precomputed physio loaded for %d / %d stimuli",
             len(result), len(stimulus_ids))
    return result


# ===========================================================================
# Dataset assembly
# ===========================================================================

def _build_dataset(
    labels: pd.DataFrame,
    audio_feats: dict[str, np.ndarray],
    eeg_feats: dict[str, np.ndarray],
    physio_feats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join all feature dicts with the labels table.

    Returns:
        X     — float32 ``(n_samples, n_features)``
        y     — float32 ``(n_samples, 2)``  [valence, arousal]
        sids  — list of stimulus IDs (same order as rows)
    """
    early_fusion_concat = _import_fusion()

    X_rows, y_rows, sids = [], [], []
    for _, row in labels.iterrows():
        sid   = str(row["stimulus_id"]).strip().lower()
        a_vec = audio_feats.get(sid)
        e_vec = eeg_feats.get(sid)
        p_vec = physio_feats.get(sid)

        missing = [m for m, v in [("audio", a_vec), ("EEG", e_vec), ("physio", p_vec)]
                   if v is None]
        if missing:
            log.debug("Skipping '%s' — missing: %s", sid, missing)
            continue

        X_rows.append(early_fusion_concat([a_vec, e_vec, p_vec]))
        y_rows.append([float(row["valence"]), float(row["arousal"])])
        sids.append(sid)

    if not X_rows:
        raise ValueError(
            "No samples could be assembled — verify that stimulus_id values in "
            "the labels CSV match the blob file stems."
        )

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.float32)
    log.info("Dataset: %d samples × %d features  (valence + arousal targets)",
             X.shape[0], X.shape[1])
    return X, y, sids


# ===========================================================================
# Model training and evaluation
# ===========================================================================

def _build_model(model_type: str = "rf"):
    """Return a scikit-learn MultiOutputRegressor pipeline.

    ``"rf"``  — Random Forest (robust baseline)
    ``"mlp"`` — Multi-layer Perceptron
    """
    if model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        base = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        base = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(base)),
    ])


def _train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    n_folds: int = 5,
) -> dict:
    """Cross-validate the regressor and report per-target metrics.

    Uses ``cross_val_predict`` so every sample appears exactly once in the
    test set.  Finally fits the model on the full dataset and returns it.
    """
    model  = _build_model(model_type)
    cv     = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    log.info("\n── Cross-validated results (%d-fold, model=%s) ──", n_folds, model_type)
    log.info("%-10s  %8s  %8s  %10s", "Target", "MSE", "R²", "Pearson r")
    log.info("%-10s  %8s  %8s  %10s", "─" * 10, "─" * 8, "─" * 8, "─" * 10)

    metrics: dict[str, dict] = {}
    for i, tgt in enumerate(("valence", "arousal")):
        mse = float(mean_squared_error(y[:, i], y_pred[:, i]))
        r2  = float(r2_score(y[:, i], y_pred[:, i]))
        pr  = float(pearsonr(y[:, i], y_pred[:, i])[0])
        metrics[tgt] = {"mse": round(mse, 4), "r2": round(r2, 4), "pearson_r": round(pr, 4)}
        log.info("%-10s  %8.4f  %8.4f  %10.4f", tgt, mse, r2, pr)

    model.fit(X, y)
    return {"model": model, "metrics": metrics}


# ===========================================================================
# CLI and main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal valence/arousal pipeline (audio + EEG/BENDR + physio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",     default="rf", choices=["rf", "mlp"],
                   help="Regressor: 'rf' (Random Forest) or 'mlp' (default: rf)")
    p.add_argument("--n",         type=int, default=None,
                   help="Limit to N participants")
    p.add_argument("--folds",     type=int, default=5,
                   help="Cross-validation folds (default: 5)")
    p.add_argument("--eeg-fs",    type=float, default=250.0,
                   help="EEG sampling rate in Hz (default: 250)")
    p.add_argument("--epoch-sec", type=float, default=4.0,
                   help="EEG epoch length in seconds (default: 4)")
    p.add_argument("--no-eeg",    action="store_true",
                   help="Skip EEG — use audio + precomputed physio only")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    (build_client, make_sas_url, CONTAINER, AUDIO_PFX,
     EEG_PFX, PHYSIO_PFX, LABELS_PFX) = _import_azure()

    log.info("Connecting to Azure (container: %s)…", CONTAINER)
    client = build_client()
    cc     = client.get_container_client(CONTAINER)

    # 1. Labels
    labels = _load_labels(cc, LABELS_PFX)
    if args.n:
        parts  = labels["participant_id"].unique()[:args.n]
        labels = labels[labels["participant_id"].isin(parts)]
        log.info("Restricted to %d participants (%d stimuli)",
                 len(parts), len(labels))

    stimulus_ids = labels["stimulus_id"].astype(str).str.strip().str.lower().unique().tolist()
    log.info("Unique stimuli to process: %d", len(stimulus_ids))

    # 2. Audio features
    log.info("Extracting audio features…")
    audio_feats = _load_audio_features(cc, AUDIO_PFX, stimulus_ids,
                                       make_sas_url, client, CONTAINER)

    # 3. EEG features (BENDR)
    if args.no_eeg:
        log.info("EEG modality skipped (--no-eeg).")
        eeg_feats = {sid: np.zeros(512, dtype=np.float32) for sid in audio_feats}
    else:
        log.info("Loading BENDR encoder…")
        encoder, device = _import_features_physiology()[0](pretrained=True)
        log.info("Extracting EEG features (fs=%.0f Hz, epoch=%.1f s)…",
                 args.eeg_fs, args.epoch_sec)
        eeg_feats = _load_eeg_features(cc, EEG_PFX, stimulus_ids,
                                       encoder, device,
                                       args.eeg_fs, args.epoch_sec)

    # 4. Precomputed non-EEG physio
    log.info("Loading precomputed physiological features…")
    physio_feats = _load_physio_features(cc, PHYSIO_PFX, stimulus_ids)

    # 5. Assemble multimodal feature matrix
    log.info("Assembling multimodal feature matrix…")
    X, y, _ = _build_dataset(labels, audio_feats, eeg_feats, physio_feats)

    # 6. Train and evaluate
    log.info("Training %s regressor (%d-fold CV)…", args.model.upper(), args.folds)
    result = _train_and_evaluate(X, y, model_type=args.model, n_folds=args.folds)

    log.info("Done.  Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
