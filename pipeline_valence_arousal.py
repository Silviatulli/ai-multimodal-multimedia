"""
pipeline_valence_arousal.py
----------------------------
End-to-end multimodal pipeline for continuous valence and arousal prediction.

Modalities
----------
    Audio         → acoustic features from video stimuli (librosa, 13 descriptors)
    EEG           → deep embeddings (BENDR pretrained contextual encoder)
    Physio        → precomputed features from features_arousal CSV
                    (HRV, EDA/GSR, pupil dilation, saccades, respiration)

Data layout (silver container, protocol = protocolaudio)
---------------------------------------------------------
    NEURO/<protocol>/auxiliary_signals/<participant>/features_arousal_<participant>_<day>.csv
        → time-series rows; rows with non-NaN `video` are stimulus windows
        → columns: HRV, EDA, pupil, saccade, respiration, video, rating_valence, rating_arousal

    NEURO/protocolimage/categories/<category>/<video_name>
        → the actual video/audio stimulus files

    NEURO/<protocol>/deepclean/<participant>/<file>.csv
        → clean EEG recordings aligned to the session

Labels come from rating_valence / rating_arousal in the features_arousal files.
Physio features are the other numeric columns in those same files, averaged
over the stimulus window per (participant, video).

Usage
-----
    python pipeline_valence_arousal.py                          # default RF, 5-fold
    python pipeline_valence_arousal.py --model mlp              # MLP regressor
    python pipeline_valence_arousal.py --n 10                   # limit participants
    python pipeline_valence_arousal.py --no-eeg                 # audio + physio only
    python pipeline_valence_arousal.py --protocol AUDIO2        # switch protocol
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

# Physio feature columns extracted from features_arousal CSV
# (excludes time, video, start, end, markers, ratings, predicted label, reliability)
_PHYSIO_COLS = [
    "meanRR_30", "RMSSD_30", "SDNN_30",
    "meanRR_10", "RMSSD_10", "SDNN_10",
    "SCL_mean", "SCL_rate",
    "SCR_n", "SCR_amplitude_sum", "SCR_amplitude_mean", "SCR_amplitude_max",
    "saccade_rate_hz", "peak_saccade_velocity_deg_s", "mean_saccade_velocity_deg_s",
    "mean_saccade_velocity_deg_s_z", "peak_saccade_velocity_deg_s_z", "saccade_rate_hz_z",
    "pupil_diam_L_interp_mean", "pupil_diam_L_interp_z_mean", "pupil_diam_L_interp_std",
    "RR", "Ti", "Te", "Ttot", "IE_ratio",
]


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
        from features_physiology import extract_eeg_bendr_features, load_bendr_encoder, load_eeg_from_bytes
    except ImportError:
        from .features_physiology import extract_eeg_bendr_features, load_bendr_encoder, load_eeg_from_bytes  # type: ignore[no-redef]
    return load_bendr_encoder, extract_eeg_bendr_features, load_eeg_from_bytes


def _import_fusion():
    try:
        from fusion_methods import early_fusion_concat
    except ImportError:
        from .fusion_methods import early_fusion_concat  # type: ignore[no-redef]
    return early_fusion_concat


def _import_azure():
    try:
        from azure_blob import build_client, make_sas_url
        from config import AZURE_CONTAINER_NAME, AZURE_VIDEO_PREFIX
    except ImportError:
        from .azure_blob import build_client, make_sas_url  # type: ignore[no-redef]
        from .config import AZURE_CONTAINER_NAME, AZURE_VIDEO_PREFIX  # type: ignore[no-redef]
    return build_client, make_sas_url, AZURE_CONTAINER_NAME, AZURE_VIDEO_PREFIX


# ===========================================================================
# Step 1 — Load physio features + labels from features_arousal files
# ===========================================================================

def _load_stimulus_physio_labels(
    container_client,
    protocol: str,
    n_participants: int | None = None,
) -> pd.DataFrame:
    """Load all features_arousal CSVs and return one row per (participant, stimulus).

    Each row contains:
        participant  — participant ID (e.g. P0290)
        video        — stimulus filename (e.g. vomit_1.mp4)
        valence      — mean rating_valence for the stimulus window
        arousal      — mean rating_arousal for the stimulus window
        <physio cols>— mean of each physio feature over the stimulus window
    """
    prefix  = f"NEURO/{protocol}/auxiliary_signals"
    records = []

    # Collect participant blob groups
    participant_blobs: dict[str, list[str]] = {}
    for b in container_client.list_blobs(name_starts_with=prefix):
        if "features_arousal" not in b.name or not b.name.endswith(".csv"):
            continue
        parts       = b.name[len(prefix):].lstrip("/").split("/")
        participant = parts[0] if parts else "unknown"
        participant_blobs.setdefault(participant, []).append(b.name)

    participants = list(participant_blobs.keys())
    if n_participants:
        participants = participants[:n_participants]

    log.info("Loading physio/labels from %d participants…", len(participants))

    for pid in participants:
        for blob_name in participant_blobs[pid]:
            try:
                raw = container_client.download_blob(blob_name).readall()
                df  = pd.read_csv(io.BytesIO(raw))
                df.columns = df.columns.str.strip()

                # Keep only rows where the participant is watching a stimulus
                stim = df[df["video"].notna()].copy()
                if stim.empty:
                    continue

                # Average physio features + labels per stimulus (video)
                available_physio = [c for c in _PHYSIO_COLS if c in stim.columns]
                agg_cols         = available_physio + ["rating_valence", "rating_arousal"]
                grouped          = stim.groupby("video")[agg_cols].mean().reset_index()
                grouped.insert(0, "participant", pid)
                records.append(grouped)

            except Exception as exc:
                log.warning("Failed to load %s: %s", blob_name, exc)

    if not records:
        raise ValueError(
            f"No stimulus rows found under '{prefix}'. "
            "Check that features_arousal CSVs have non-NaN 'video' rows."
        )

    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["rating_valence", "rating_arousal"])

    # Normalise ratings from [1,9] to [-1,1] if needed
    for col in ("rating_valence", "rating_arousal"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if result[col].max() > 1.5:
            result[col] = (result[col] - 5.0) / 4.0

    log.info("Physio/label table: %d rows × %d physio features  "
             "(%d unique stimuli, %d unique participants)",
             len(result), len([c for c in _PHYSIO_COLS if c in result.columns]),
             result["video"].nunique(), result["participant"].nunique())
    return result


# ===========================================================================
# Step 2 — Audio features  (extracted from video stimulus files)
# ===========================================================================

def _load_audio_features(
    container_client,
    video_prefix: str,
    stimulus_names: list[str],
    make_sas_url_fn,
    client,
) -> dict[str, np.ndarray]:
    """Extract acoustic features from the video stimulus files.

    Stimulus names (e.g. ``vomit_1.mp4``) are matched against blobs under
    ``video_prefix`` by stem.  PyAV extracts the audio track from the video.

    Returns ``{stimulus_name: feature_vector}`` (float32, length 13).
    """
    extract_acoustic_features, load_audio = _import_features_multimedia()

    # Index: stem_lower → blob name
    blob_index: dict[str, str] = {}
    for b in container_client.list_blobs(name_starts_with=video_prefix):
        blob_index[Path(b.name).stem.lower()] = b.name

    result: dict[str, np.ndarray] = {}
    missing = []

    for sname in stimulus_names:
        stem      = Path(sname).stem.lower()
        blob_name = blob_index.get(stem)
        if blob_name is None:
            missing.append(sname)
            continue
        try:
            url          = make_sas_url_fn(client, blob_name)
            waveform, sr = load_audio(url)
            feats        = extract_acoustic_features(waveform, sr)
            result[sname] = _acoustic_to_vector(feats)
        except Exception as exc:
            log.warning("Audio extraction failed for '%s': %s", sname, exc)

    if missing:
        log.warning("Audio blobs not found for %d stimuli: %s…",
                    len(missing), missing[:5])
    log.info("Audio features: %d / %d stimuli", len(result), len(stimulus_names))
    return result


def _acoustic_to_vector(feats) -> np.ndarray:
    return np.array([
        feats.rms_energy_mean, feats.rms_energy_max, feats.initial_energy_ratio,
        feats.energy_slope, feats.peak_position_ratio, feats.energy_variance,
        feats.loudness_dynamic_range_db, *feats.loudness_shape_5seg,
        feats.zero_crossing_rate_mean, feats.zero_crossing_rate_var,
        feats.spectral_centroid_mean_hz, feats.onset_density_per_second,
        feats.pause_ratio,
    ], dtype=np.float32)


# ===========================================================================
# Step 3 — EEG features  (BENDR per participant)
# ===========================================================================

def _load_eeg_features(
    container_client,
    protocol: str,
    participants: list[str],
    encoder,
    device: str,
    eeg_fs: float,
    epoch_sec: float,
) -> dict[str, np.ndarray]:
    """Load deepclean EEG per participant and encode with BENDR.

    Returns ``{participant_id: mean_embedding_vector}``.  The whole-session
    mean is used here; per-stimulus alignment can be added when marker data
    is parsed.
    """
    load_bendr_encoder, extract_eeg_bendr_features, load_eeg_from_bytes = _import_features_physiology()
    prefix = f"NEURO/{protocol}/deepclean"

    # Index: participant → list of EEG blob names
    part_blobs: dict[str, list[str]] = {}
    for b in container_client.list_blobs(name_starts_with=prefix):
        if not b.name.endswith(".csv"):
            continue
        rel  = b.name[len(prefix):].lstrip("/")
        pid  = rel.split("/")[0]
        part_blobs.setdefault(pid, []).append(b.name)

    result: dict[str, np.ndarray] = {}
    for pid in participants:
        blobs = part_blobs.get(pid)
        if not blobs:
            log.warning("EEG not found for participant '%s'", pid)
            continue
        embeddings: list[np.ndarray] = []
        for blob_name in blobs[:1]:   # use first recording per participant
            try:
                raw    = container_client.download_blob(blob_name).readall()
                eeg    = load_eeg_from_bytes(raw, eeg_fs)
                eeg    = _preprocess_eeg(eeg, eeg_fs)
                embs   = extract_eeg_bendr_features(eeg, eeg_fs, encoder, device,
                                                    epoch_sec=epoch_sec)
                embeddings.append(embs.mean(axis=0))
            except Exception as exc:
                log.warning("EEG extraction failed for '%s': %s", pid, exc)
        if embeddings:
            result[pid] = np.mean(embeddings, axis=0)

    log.info("EEG features: %d / %d participants", len(result), len(participants))
    return result


def _preprocess_eeg(eeg: np.ndarray, fs: float) -> np.ndarray:
    try:
        from features_physiology import bandpass_filter, zscore_signal
    except ImportError:
        from .features_physiology import bandpass_filter, zscore_signal  # type: ignore[no-redef]
    return zscore_signal(bandpass_filter(eeg, 0.5, 45.0, fs))


# ===========================================================================
# Step 4 — Build fused feature matrix
# ===========================================================================

def _build_dataset(
    stimulus_df: pd.DataFrame,
    audio_feats: dict[str, np.ndarray],
    eeg_feats: dict[str, np.ndarray],
    use_eeg: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Join physio, audio, and EEG features into a single matrix.

    Returns:
        X     — float32 ``(n_samples, n_features)``
        y     — float32 ``(n_samples, 2)``  [valence, arousal]
        meta  — DataFrame with participant / video columns (same row order)
    """
    early_fusion_concat = _import_fusion()
    physio_cols         = [c for c in _PHYSIO_COLS if c in stimulus_df.columns]

    X_rows, y_rows, meta_rows = [], [], []

    for _, row in stimulus_df.iterrows():
        pid   = row["participant"]
        sname = row["video"]

        a_vec = audio_feats.get(sname)
        if a_vec is None:
            log.debug("Skipping %s/%s — no audio", pid, sname)
            continue

        p_vec = row[physio_cols].values.astype(np.float32)
        if np.isnan(p_vec).all():
            log.debug("Skipping %s/%s — all physio NaN", pid, sname)
            continue
        p_vec = np.nan_to_num(p_vec, nan=0.0)

        if use_eeg:
            e_vec = eeg_feats.get(pid)
            if e_vec is None:
                log.debug("Skipping %s/%s — no EEG", pid, sname)
                continue
            fused = early_fusion_concat([a_vec, p_vec, e_vec])
        else:
            fused = early_fusion_concat([a_vec, p_vec])

        X_rows.append(fused)
        y_rows.append([float(row["rating_valence"]), float(row["rating_arousal"])])
        meta_rows.append({"participant": pid, "video": sname})

    if not X_rows:
        raise ValueError(
            "No samples assembled — check that audio blobs match the video "
            "names in the features_arousal CSVs."
        )

    X    = np.stack(X_rows).astype(np.float32)
    y    = np.array(y_rows, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    log.info("Dataset: %d samples × %d features  (valence + arousal targets)",
             X.shape[0], X.shape[1])
    return X, y, meta


# ===========================================================================
# Step 5 — Train and evaluate
# ===========================================================================

def _build_model(model_type: str = "rf"):
    if model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        base = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                            max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1)
    else:
        from sklearn.ensemble import RandomForestRegressor
        base = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     random_state=42, n_jobs=-1)
    return Pipeline([("scaler", StandardScaler()),
                     ("model",  MultiOutputRegressor(base))])


def _train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    n_folds: int = 5,
) -> dict:
    model  = _build_model(model_type)
    cv     = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    log.info("\n── Cross-validated results (%d-fold, model=%s) ──", n_folds, model_type)
    log.info("%-10s  %8s  %8s  %10s", "Target", "MSE", "R²", "Pearson r")
    log.info("%-10s  %8s  %8s  %10s", "─"*10, "─"*8, "─"*8, "─"*10)

    metrics: dict[str, dict] = {}
    for i, tgt in enumerate(("valence", "arousal")):
        mse = float(mean_squared_error(y[:, i], y_pred[:, i]))
        r2  = float(r2_score(y[:, i], y_pred[:, i]))
        pr  = float(pearsonr(y[:, i], y_pred[:, i])[0])
        metrics[tgt] = {"mse": round(mse,4), "r2": round(r2,4), "pearson_r": round(pr,4)}
        log.info("%-10s  %8.4f  %8.4f  %10.4f", tgt, mse, r2, pr)

    model.fit(X, y)
    return {"model": model, "metrics": metrics}


# ===========================================================================
# CLI and main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal valence/arousal pipeline (audio + physio + EEG/BENDR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--protocol",  default="protocolaudio",
                   help="NEURO sub-protocol folder (default: protocolaudio)")
    p.add_argument("--model",     default="rf", choices=["rf", "mlp"])
    p.add_argument("--n",         type=int, default=None,
                   help="Limit to N participants")
    p.add_argument("--folds",     type=int, default=5)
    p.add_argument("--eeg-fs",    type=float, default=250.0)
    p.add_argument("--epoch-sec", type=float, default=4.0)
    p.add_argument("--no-eeg",    action="store_true",
                   help="Use audio + physio only (skip EEG/BENDR)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_client, make_sas_url, CONTAINER, VIDEO_PREFIX = _import_azure()

    log.info("Protocol : %s  |  container : %s", args.protocol, CONTAINER)
    client = build_client()
    cc     = client.get_container_client(CONTAINER)

    # 1. Physio features + labels from features_arousal CSVs
    log.info("Loading physio features and ratings from auxiliary_signals…")
    stim_df = _load_stimulus_physio_labels(cc, args.protocol, n_participants=args.n)

    stimulus_names = stim_df["video"].unique().tolist()
    participants   = stim_df["participant"].unique().tolist()

    # 2. Audio features from video stimulus files
    log.info("Extracting audio features from %d unique stimuli…", len(stimulus_names))
    audio_feats = _load_audio_features(
        cc, VIDEO_PREFIX, stimulus_names, make_sas_url, client
    )

    # 3. EEG features (optional)
    eeg_feats: dict[str, np.ndarray] = {}
    if not args.no_eeg:
        log.info("Loading BENDR encoder…")
        encoder, device = _import_features_physiology()[0](pretrained=True)
        log.info("Extracting EEG features for %d participants…", len(participants))
        eeg_feats = _load_eeg_features(
            cc, args.protocol, participants, encoder, device,
            args.eeg_fs, args.epoch_sec
        )

    # 4. Assemble fused dataset
    log.info("Assembling feature matrix…")
    X, y, meta = _build_dataset(stim_df, audio_feats, eeg_feats,
                                 use_eeg=not args.no_eeg)

    # 5. Train and evaluate
    log.info("Training %s regressor (%d-fold CV)…", args.model.upper(), args.folds)
    result = _train_and_evaluate(X, y, model_type=args.model, n_folds=args.folds)
    log.info("Done.  Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
