"""
cross_modal_distillation.py
-----------------------------
Bridging the Modality Gap: Cross-Modal Knowledge Distillation for
Valence / Arousal Prediction.

Two parallel models are trained on the same physiological features but
with different supervision signals:

    Teacher  — supervised by physio-derived 3-class labels
               (arousal_three_all / valence_three_all from
                ai-physiological-labeling PCA pipeline).
               Learns to map raw physio signals to a consistent
               ordinal arousal/valence space.

    Student  — supervised by self-reported continuous ratings
               (rating_valence / rating_arousal).
               Simultaneously distilled from the Teacher via a
               temperature-scaled KL-divergence loss so it can
               leverage the Teacher's richer physio-label signal
               even where self-report data are noisy or sparse.

Combined student loss (Hinton et al., 2015 KD formula):
    L = (1 - α) · MSE(ŷ_regression, y_self_report)
      + α · T² · KL(P_teacher(T), P_student(T))

where T is the softmax temperature and α controls the distillation
weight.  The student's regression head is evaluated on held-out folds
without the teacher's guidance, so test-time performance is independent
of whether the teacher is available.

Usage
-----
    python cross_modal_distillation.py          # default settings
    python cross_modal_distillation.py --alpha 0.3 --temperature 4
    python cross_modal_distillation.py --no-distill  # student only (baseline)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature + label columns
# ---------------------------------------------------------------------------

PHYSIO_COLS = [
    "meanRR_30", "RMSSD_30", "SDNN_30", "meanRR_10", "RMSSD_10", "SDNN_10",
    "SCL_mean", "SCL_rate", "SCR_n", "SCR_amplitude_sum",
    "SCR_amplitude_mean", "SCR_amplitude_max",
    "saccade_rate_hz", "peak_saccade_velocity_deg_s",
    "mean_saccade_velocity_deg_s", "mean_saccade_velocity_deg_s_z",
    "peak_saccade_velocity_deg_s_z", "saccade_rate_hz_z",
    "pupil_diam_L_interp_mean", "pupil_diam_L_interp_z_mean",
    "pupil_diam_L_interp_std",
    "RR", "Ti", "Te", "Ttot", "IE_ratio",
]

# Per-modality physiological response latencies (seconds from stimulus onset).
# Features are only reliable AFTER these delays — earlier windows still carry
# the response to the previous stimulus.
# Sources: Boucsein (2012) for GSR; Berntson et al. (1997) for ECG;
#          Mathôt (2018) for pupil; Benedek & Kaernbach (2010) for respiration.
MODALITY_DELAYS: dict[str, int] = {
    # Oculometry — saccades are reflexive, near-real-time
    "saccade_rate_hz":                    0,
    "peak_saccade_velocity_deg_s":        0,
    "mean_saccade_velocity_deg_s":        0,
    "mean_saccade_velocity_deg_s_z":      0,
    "peak_saccade_velocity_deg_s_z":      0,
    "saccade_rate_hz_z":                  0,
    # Pupil dilation — parasympathetic/sympathetic onset ~0.5–2 s
    "pupil_diam_L_interp_mean":           1,
    "pupil_diam_L_interp_z_mean":         1,
    "pupil_diam_L_interp_std":            1,
    # ECG / HRV — vagal response 1–3 s; 30 s window averages absorb earlier noise
    "meanRR_30":                          2,
    "RMSSD_30":                           2,
    "SDNN_30":                            2,
    "meanRR_10":                          1,
    "RMSSD_10":                           1,
    "SDNN_10":                            1,
    # GSR — SCR onset 1–4 s, peak ~2–4 s; SCL is even slower
    "SCL_mean":                           4,
    "SCL_rate":                           4,
    "SCR_n":                              2,
    "SCR_amplitude_sum":                  3,
    "SCR_amplitude_mean":                 3,
    "SCR_amplitude_max":                  3,
    # Respiration — cycle-level changes emerge over 2–5 s
    "RR":                                 3,
    "Ti":                                 3,
    "Te":                                 3,
    "Ttot":                               3,
    "IE_ratio":                           3,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ALL_PROTOCOLS = [
    "protocolimage",
    "protocolaudio",
    "protocolbel",
    "protocoldanone",
    "protocolNRJ",
]


def _load_protocol_rows(cc, protocol: str) -> list["pd.DataFrame"]:
    """Return a list of per-participant trial DataFrames for one NEURO protocol folder."""
    prefix = f"NEURO/{protocol}/auxiliary_signals/"
    rows: list[pd.DataFrame] = []
    for blob in cc.list_blobs(name_starts_with=prefix):
        fname = blob.name.split("/")[-1]
        m = re.match(r"features_arousal_(P\d+)_(day\d+)\.csv", fname)
        if not m:
            continue
        pid, day = m.group(1), m.group(2)
        try:
            df = pd.read_csv(io.BytesIO(cc.download_blob(blob.name).readall()))
            stim = df[df["video"].notna()].copy()
            if stim.empty:
                continue

            # Group by (video, start_marker) — each unique (video, start) = one stimulus trial.
            # For each group, apply per-feature response-latency delay:
            # only average time windows where win_idx_in_trial >= that feature's delay.
            has_win_idx = "win_idx_in_trial" in stim.columns
            agg_rows: list[dict] = []
            for (video, start_marker), trial in stim.groupby(["video", "start"]):
                row: dict = {"video": video, "start": start_marker}
                for col in PHYSIO_COLS:
                    if col not in trial.columns:
                        continue
                    delay = MODALITY_DELAYS.get(col, 0)
                    if has_win_idx and delay > 0:
                        valid = trial[trial["win_idx_in_trial"] >= delay]
                    else:
                        valid = trial
                    row[col] = float(valid[col].mean()) if len(valid) and valid[col].notna().any() else float("nan")
                for rc in ("rating_valence", "rating_arousal"):
                    if rc in trial.columns:
                        row[rc] = float(trial[rc].mean())
                agg_rows.append(row)

            if not agg_rows:
                continue
            trial_df = pd.DataFrame(agg_rows)
            trial_df["participant"] = pid
            trial_df["session"] = day
            trial_df["protocol"] = protocol
            rows.append(trial_df)
        except Exception as exc:
            log.debug("Skip %s/%s: %s", protocol, fname, exc)
    return rows


def load_data(protocol: str = "protocolaudio",
              delay_sec: int = 0) -> tuple["pd.DataFrame", list[str]]:
    """Load and merge physio features, self-report labels, and physio-derived labels.

    Args:
        protocol:  NEURO sub-protocol folder (e.g. ``protocolaudio``), or ``"all"``
                   to load all 5 protocols simultaneously (protocolimage, protocolaudio,
                   protocolbel, protocoldanone, protocolNRJ).  Rows are deduplicated on
                   (participant, video) after concatenation so cross-protocol duplicates
                   are collapsed.
        delay_sec: Additional global minimum ``win_idx_in_trial`` for physio-derived
                   *teacher labels* (applied on top of per-feature ``MODALITY_DELAYS``
                   which already handle latency in the feature aggregation).
                   Set to 0 (default) to rely entirely on the per-feature delays.
    """
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    logging.disable(logging.CRITICAL)
    client = BlobServiceClient(
        "https://sahabsdatalakeprodweu.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    cc = client.get_container_client("silver")
    logging.disable(logging.NOTSET)

    # ── Self-report + physio features ─────────────────────────────────────
    protocols_to_load = ALL_PROTOCOLS if protocol == "all" else [protocol]
    rows: list[pd.DataFrame] = []
    for proto in protocols_to_load:
        log.info("Loading features_arousal CSVs from %s…", proto)
        rows.extend(_load_protocol_rows(cc, proto))

    base = pd.concat(rows, ignore_index=True)
    base = base.dropna(subset=["rating_valence", "rating_arousal"]).reset_index(drop=True)

    # Deduplicate on (participant, video) — keep first occurrence when multiple
    # protocols recorded the same participant/stimulus pair.
    before_dedup = len(base)
    base = base.drop_duplicates(subset=["participant", "video"]).reset_index(drop=True)
    if before_dedup != len(base):
        log.info("Deduplication: %d → %d rows (removed %d cross-protocol duplicates)",
                 before_dedup, len(base), before_dedup - len(base))

    log.info("Self-report data: %d rows, %d stimuli, %d participants",
             len(base), base["video"].nunique(), base["participant"].nunique())

    # ── Physio-derived labels — direct merge on (participant, video) ──────
    # The all_labels file has video column already populated for stimulus rows.
    log.info("Loading physio-derived labels from TARGETS/all_labels/…")
    try:
        lab_raw = pd.read_csv(io.BytesIO(
            cc.download_blob("TARGETS/all_labels/all_arousal_emotion_video.csv").readall()
        ))
        # Keep only stimulus rows (video not NaN) and average per (participant, video)
        # Apply delay: skip windows < delay_sec to avoid inter-stimulus carry-over
        # in slow modalities (GSR peak at ~2-4s, respiration ~3-5s, ECG ~1-3s).
        lab_stim = lab_raw[lab_raw["video"].notna()].copy()
        if delay_sec > 0:
            before = len(lab_stim)
            lab_stim = lab_stim[lab_stim["win_idx_in_trial"] >= delay_sec]
            log.info("  Delay filter (>=%ds): %d → %d rows (%.0f%% retained)",
                     delay_sec, before, len(lab_stim),
                     100 * len(lab_stim) / max(before, 1))
        merged_lab = (
            lab_stim.groupby(["subject_id", "video"])
            [["arousal_three_all", "valence_three_all"]]
            .mean()
            .reset_index()
            .rename(columns={"subject_id": "participant"})
        )
    except Exception as exc:
        log.warning("Could not load physio labels: %s — using NaN", exc)
        merged_lab = pd.DataFrame(columns=["participant", "video",
                                            "arousal_three_all", "valence_three_all"])

    # ── Join ───────────────────────────────────────────────────────────────
    data = base.merge(merged_lab, on=["participant", "video"], how="left")
    feat_cols = [c for c in PHYSIO_COLS if c in data.columns]
    log.info("Features available: %d / %d  |  physio labels merged: %d%%",
             len(feat_cols), len(PHYSIO_COLS),
             int(100 * data["arousal_three_all"].notna().mean()))
    return data, feat_cols


# ---------------------------------------------------------------------------
# Neural network models
# ---------------------------------------------------------------------------

class TeacherNet(nn.Module):
    """MLP: physio features → 3-class softmax (Low / Med / High).

    Trained on physio-derived labels.  Produces calibrated soft
    probability vectors used as distillation targets for the Student.
    """

    def __init__(self, n_features: int, n_classes: int = 3,
                 hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # logits


class StudentNet(nn.Module):
    """MLP: physio features → (continuous regression, 3-class soft head).

    The regression head predicts continuous valence/arousal (self-report).
    The classification head produces soft probabilities distilled from
    the Teacher.  At inference only the regression head is used.
    """

    def __init__(self, n_features: int, n_targets: int = 2, n_classes: int = 3,
                 hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            in_dim = h
        self.backbone  = nn.Sequential(*layers)
        self.reg_head  = nn.Linear(in_dim, n_targets)    # → continuous ratings
        self.soft_head = nn.Linear(in_dim, n_classes)    # → 3-class logits (distillation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat  = self.backbone(x)
        reg   = self.reg_head(feat)
        soft  = self.soft_head(feat)
        return reg, soft   # (continuous, logits)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def soft_labels(y_class: torch.Tensor, temperature: float,
                n_classes: int = 3) -> torch.Tensor:
    """Convert integer class labels → soft probability vectors (uniform if missing)."""
    logits = torch.zeros(len(y_class), n_classes, device=y_class.device)
    valid  = y_class >= 0
    if valid.any():
        oh = F.one_hot(y_class[valid].long().clamp(0, n_classes - 1), n_classes).float()
        logits[valid] = oh * 10.0   # sharp before temperature
    return F.softmax(logits / temperature, dim=1)


def train_teacher(
    X: np.ndarray,
    y_class: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch: int = 128,
    device: str = "cpu",
) -> TeacherNet:
    """Train Teacher on physio-derived 3-class labels."""
    model = TeacherNet(X.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt    = torch.tensor(X, dtype=torch.float32, device=device)
    yt    = torch.tensor(y_class, dtype=torch.long, device=device)
    valid = yt >= 0

    model.train()
    valid_cpu = valid.cpu().numpy()   # boolean mask on CPU for indexing
    for epoch in range(n_epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            idx   = perm[i:i + batch]
            idx_v = idx[valid_cpu[idx]]
            if not len(idx_v):
                continue
            loss = F.cross_entropy(model(Xt[idx_v]), yt[idx_v])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_student(
    X: np.ndarray,
    y_reg: np.ndarray,          # self-report ratings  (N, 2)
    teacher: TeacherNet | None,
    y_class: np.ndarray,        # physio labels for arousal (N,) — used to produce teacher soft targets
    n_epochs: int = 150,
    lr: float = 1e-3,
    batch: int = 128,
    alpha: float = 0.5,         # distillation weight
    temperature: float = 3.0,
    device: str = "cpu",
) -> StudentNet:
    """Train Student with optional KD loss from Teacher."""
    model = StudentNet(X.shape[1], n_targets=y_reg.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt    = torch.tensor(X, dtype=torch.float32, device=device)
    yr    = torch.tensor(y_reg, dtype=torch.float32, device=device)

    # Pre-compute teacher soft targets (fixed, no gradient)
    if teacher is not None:
        teacher.eval()
        with torch.no_grad():
            teacher_logits = teacher(Xt)
            teacher_probs  = F.softmax(teacher_logits / temperature, dim=1)
    else:
        teacher_probs = None

    model.train()
    for epoch in range(n_epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            reg_pred, soft_pred = model(Xt[idx])

            # Task loss: MSE on self-report ratings
            task_loss = F.mse_loss(reg_pred, yr[idx])

            if teacher_probs is not None:
                # Distillation loss: KL(teacher || student) scaled by T²
                student_log_probs = F.log_softmax(soft_pred / temperature, dim=1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs[idx],
                                   reduction="batchmean") * (temperature ** 2)
                loss = (1 - alpha) * task_loss + alpha * kd_loss
            else:
                loss = task_loss

            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_student(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train_reg: np.ndarray, y_test_reg: np.ndarray,
    y_train_class: np.ndarray,
    alpha: float, temperature: float,
    n_epochs_teacher: int, n_epochs_student: int,
    device: str,
) -> dict:
    """Train teacher + student on train fold, evaluate on test fold."""
    teacher = train_teacher(X_train, y_train_class, n_epochs=n_epochs_teacher, device=device)

    # Student WITH distillation
    student_kd = train_student(X_train, y_train_reg, teacher, y_train_class,
                               n_epochs=n_epochs_student, alpha=alpha,
                               temperature=temperature, device=device)

    # Student WITHOUT distillation (baseline)
    student_base = train_student(X_train, y_train_reg, None, y_train_class,
                                 n_epochs=n_epochs_student, alpha=0.0,
                                 temperature=temperature, device=device)

    Xtest_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    results = {}
    for name, model in [("Student (baseline)", student_base),
                         ("Student + KD",       student_kd)]:
        with torch.no_grad():
            pred, _ = model(Xtest_t)
        pred_np = pred.cpu().numpy()
        results[name] = {}
        for i, tgt in enumerate(("valence", "arousal")):
            yt = y_test_reg[:, i]
            yp = pred_np[:, i]
            results[name][tgt] = dict(
                mse=round(float(mean_squared_error(yt, yp)), 4),
                r2 =round(float(r2_score(yt, yp)), 4),
                pearson_r=round(float(pearsonr(yt, yp)[0]), 4),
            )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol",    default="protocolaudio")
    p.add_argument("--folds",       type=int,   default=5)
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="KD loss weight (0=no distillation, 1=only KD)")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="Softmax temperature for soft targets")
    p.add_argument("--delay-sec",   type=int,   default=0,
                   help="Skip first N seconds of each trial for physio labels "
                        "(accounts for GSR/respiration response latency). "
                        "Recommended: 3 for GSR-dominated 'all' labels.")
    p.add_argument("--epochs-teacher", type=int, default=100)
    p.add_argument("--epochs-student", type=int, default=150)
    p.add_argument("--no-distill", action="store_true",
                   help="Run student-only baseline (ignores alpha/temperature)")
    p.add_argument("--output-dir", default=None,
                   help="If set, write results JSON to <output-dir>/cross_modal_results.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = "mps" if torch.backends.mps.is_available() \
        else "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    data, feat_cols = load_data(args.protocol, delay_sec=args.delay_sec)

    X_raw = data[feat_cols].fillna(0).values.astype(np.float32)
    y_reg = data[["rating_valence", "rating_arousal"]].values.astype(np.float32)

    # Physio-derived arousal label (use as teacher supervision)
    y_class = np.where(
        data["arousal_three_all"].notna(),
        data["arousal_three_all"].fillna(-1).values.astype(int),
        -1,                  # -1 = missing → teacher skips those samples
    )
    pct_physio = int(100 * (y_class >= 0).mean())
    log.info("Physio labels available for %d%% of samples", pct_physio)

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results: list[dict] = []

    for fold, (tr, te) in enumerate(cv.split(X_raw)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[tr])
        X_te = sc.transform(X_raw[te])

        res = evaluate_student(
            X_tr, X_te,
            y_reg[tr], y_reg[te],
            y_class[tr],
            alpha       = 0.0 if args.no_distill else args.alpha,
            temperature = args.temperature,
            n_epochs_teacher = args.epochs_teacher,
            n_epochs_student = args.epochs_student,
            device      = device,
        )
        fold_results.append(res)
        log.info("Fold %d done", fold + 1)

    # ── Aggregate across folds ────────────────────────────────────────────
    models = list(fold_results[0].keys())
    targets = ("valence", "arousal")
    metrics = ("mse", "r2", "pearson_r")

    agg: dict[str, dict] = {m: {t: {k: [] for k in metrics} for t in targets} for m in models}
    for fr in fold_results:
        for m in models:
            for t in targets:
                for k in metrics:
                    agg[m][t][k].append(fr[m][t][k])

    log.info("\n" + "═" * 68)
    log.info("  Cross-Modal Knowledge Distillation  (%d-fold CV)", args.folds)
    log.info("  alpha=%.2f  temperature=%.1f  physio_labels=%d%%  delay=%ds",
             args.alpha, args.temperature, pct_physio, args.delay_sec)
    log.info("═" * 68)
    log.info("%-24s  %-10s  %8s  %8s  %10s",
             "Model", "Target", "MSE", "R²", "Pearson r")
    log.info("%-24s  %-10s  %8s  %8s  %10s",
             "─" * 24, "─" * 10, "─" * 8, "─" * 8, "─" * 10)

    for m in models:
        for t in targets:
            mse = float(np.mean(agg[m][t]["mse"]))
            r2  = float(np.mean(agg[m][t]["r2"]))
            pr  = float(np.mean(agg[m][t]["pearson_r"]))
            log.info("%-24s  %-10s  %8.4f  %8.4f  %10.4f", m, t, mse, r2, pr)

    log.info("═" * 68)
    log.info("")
    for t in targets:
        dr2 = (float(np.mean(agg["Student + KD"][t]["r2"])) -
               float(np.mean(agg["Student (baseline)"][t]["r2"])))
        dpr = (float(np.mean(agg["Student + KD"][t]["pearson_r"])) -
               float(np.mean(agg["Student (baseline)"][t]["pearson_r"])))
        log.info("  KD gain — %s: ΔR²=%+.4f  ΔPearson=%+.4f  → %s",
                 t, dr2, dpr, "IMPROVED" if dr2 > 0 else "no gain")

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict = {
            "config": {
                "protocol":    args.protocol,
                "folds":       args.folds,
                "alpha":       args.alpha,
                "temperature": args.temperature,
                "delay_sec":   args.delay_sec,
                "no_distill":  args.no_distill,
            },
            "results": {},
        }
        for m in models:
            summary["results"][m] = {}
            for t in targets:
                summary["results"][m][t] = {
                    k: round(float(np.mean(agg[m][t][k])), 4)
                    for k in metrics
                }
        out_path = out_dir / "cross_modal_results.json"
        out_path.write_text(json.dumps(summary, indent=2))
        log.info("Results saved to %s", out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
