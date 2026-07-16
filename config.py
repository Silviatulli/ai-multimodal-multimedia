"""
config.py
---------
Project-wide constants and path configuration.

Override any value by setting the corresponding environment variable before
importing this module, or edit the defaults here.
"""

import os

# ── Video sampling ──────────────────────────────────────────────────────────
NUM_FRAMES: int        = int(os.getenv("NUM_FRAMES", "8"))
WINDOW_SIZE_SEC: float = float(os.getenv("WINDOW_SIZE_SEC", "4.0"))
WINDOW_STRIDE_SEC: float = float(os.getenv("WINDOW_STRIDE_SEC", "2.0"))

# ── Local dataset paths ─────────────────────────────────────────────────────
DATA_DIR: str       = os.getenv("DATA_DIR", "data")
CATEGORIES_DIR: str = os.getenv("CATEGORIES_DIR", os.path.join(DATA_DIR, "categories"))

# ── Azure Blob Storage ──────────────────────────────────────────────────────
# Input  layout: silver/NEURO/protocolimage/categories/<category>/<file>
# Output layout: silver/AI/protocolimage/preprocessed/<category>/<stem>/
#                    frames.npy        sampled frames  [N, H, W, 3]  uint8
#                    frames_aug.npy    augmented frames
#                    windows.json      sliding-window metadata
AZURE_CONNECTION_STRING: str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME: str     = os.getenv("AZURE_CONTAINER_NAME", "silver")
AZURE_VIDEO_PREFIX: str       = os.getenv("AZURE_VIDEO_PREFIX",  "NEURO/protocolimage/categories")
AZURE_OUTPUT_PREFIX: str      = os.getenv("AZURE_OUTPUT_PREFIX", "AI/protocolimage/preprocessed")

# ── Physiological data paths ─────────────────────────────────────────────────
# Auxiliary (non-EEG) signals live inside each protocol folder:
#   silver/NEURO/<protocol>/auxiliary_signals/<participant>/<file>
# Set AZURE_PROTOCOL to select the protocol (default: protocolaudio).
#
# EEG raw CSVs:             silver/NEURO/<protocol>/eeg/<participant>/<file>.csv
# Labels:                   silver/NEURO/<protocol>/labels/valence_arousal.csv
AZURE_PROTOCOL: str           = os.getenv("AZURE_PROTOCOL",       "protocolaudio")
AZURE_EEG_PREFIX: str         = os.getenv("AZURE_EEG_PREFIX",
                                           f"NEURO/{os.getenv('AZURE_PROTOCOL', 'protocolaudio')}/eeg")
AZURE_PHYSIO_PREFIX: str      = os.getenv("AZURE_PHYSIO_PREFIX",
                                           f"NEURO/{os.getenv('AZURE_PROTOCOL', 'protocolaudio')}/auxiliary_signals")
AZURE_LABELS_PREFIX: str      = os.getenv("AZURE_LABELS_PREFIX",
                                           f"NEURO/{os.getenv('AZURE_PROTOCOL', 'protocolaudio')}/labels")
