"""
config.py
---------
Project-wide constants and Azure path configuration.

All Azure paths follow the HABS data-lake layout:
    silver/NEURO/<protocol>/...   ← raw and processed physiological + stimulus data
    silver/TARGETS/...            ← arousal/valence labels (from ai-physiological-labeling)
    silver/AI/<system>/...        ← AI pipeline outputs
    bronze/...                    ← raw recordings and device data

Override any value by setting the corresponding environment variable.

Datastore → container mapping (from adlsgen2_*_weu.yml):
    adlsgen2_silver_weu  →  sahabsdatalakeprodweu / silver
    adlsgen2_bronze_weu  →  sahabsdatalakeprodweu / bronze
    adlsgen2_gold_weu    →  sahabsdatalakeprodweu / gold
"""

import os

# ── Video / audio sampling ────────────────────────────────────────────────────
NUM_FRAMES: int          = int(os.getenv("NUM_FRAMES", "8"))
WINDOW_SIZE_SEC: float   = float(os.getenv("WINDOW_SIZE_SEC", "4.0"))
WINDOW_STRIDE_SEC: float = float(os.getenv("WINDOW_STRIDE_SEC", "2.0"))

# ── Local dataset paths ───────────────────────────────────────────────────────
DATA_DIR: str       = os.getenv("DATA_DIR", "data")
CATEGORIES_DIR: str = os.getenv("CATEGORIES_DIR", os.path.join(DATA_DIR, "categories"))

# ── Azure storage account (shared across all containers) ──────────────────────
AZURE_STORAGE_ACCOUNT: str    = os.getenv("AZURE_STORAGE_ACCOUNT", "sahabsdatalakeprodweu")
AZURE_CONNECTION_STRING: str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# ── Container names ───────────────────────────────────────────────────────────
AZURE_CONTAINER_NAME: str    = os.getenv("AZURE_CONTAINER_NAME",    "silver")
AZURE_BRONZE_CONTAINER: str  = os.getenv("AZURE_BRONZE_CONTAINER",  "bronze")
AZURE_GOLD_CONTAINER: str    = os.getenv("AZURE_GOLD_CONTAINER",    "gold")

# ── Protocol selection ────────────────────────────────────────────────────────
# Controls the NEURO subfolder used for physio data.
# Available: protocolimage | protocolaudio | protocolbel | protocoldanone | protocolNRJ
AZURE_PROTOCOL: str = os.getenv("AZURE_PROTOCOL", "protocolaudio")

# ── Silver — stimulus input paths ─────────────────────────────────────────────
# Video/audio stimuli:   silver/NEURO/protocolimage/categories/<category>/<file>
AZURE_VIDEO_PREFIX: str = os.getenv(
    "AZURE_VIDEO_PREFIX", "NEURO/protocolimage/categories"
)

# ── Silver — physiological data paths ────────────────────────────────────────
# Auxiliary signals (ECG, GSR, pupil, respiration, all-modalities features):
#   silver/NEURO/<protocol>/auxiliary_signals/<participant>/<file>.csv
#
# Clean EEG recordings (deepclean pipeline output):
#   silver/NEURO/<protocol>/deepclean/<participant>/<file>.csv
#   (folder is named 'deepclean2' for protocolimage, 'deepclean' for others)
#
# Arousal/valence labels (ai-physiological-labeling pipeline output):
#   silver/TARGETS/all_labels/all_arousal_emotion_video.csv
#   silver/TARGETS/<protocol>.csv                          ← marker→stimulus map
#   silver/TARGETS/gsr_labels/gsr_arousal_emotion_video.csv
#   silver/TARGETS/occulo_labels/occulo_arousal_emotion_video.csv
#   silver/TARGETS/respiration_labels/respiration_arousal_emotion_video.csv

_proto = os.getenv("AZURE_PROTOCOL", "protocolaudio")

AZURE_PHYSIO_PREFIX: str  = os.getenv(
    "AZURE_PHYSIO_PREFIX", f"NEURO/{_proto}/auxiliary_signals"
)
AZURE_EEG_PREFIX: str     = os.getenv(
    "AZURE_EEG_PREFIX", f"NEURO/{_proto}/deepclean"
)
AZURE_LABELS_PREFIX: str  = os.getenv(
    "AZURE_LABELS_PREFIX", "TARGETS"          # labels live under silver/TARGETS/
)

# ── Silver — AI pipeline output paths ────────────────────────────────────────
# Preprocessed video frames (multimedia_data_preprocessing.py output):
#   silver/AI/protocolimage/preprocessed/<category>/<stem>/
#       frames.npy, frames_aug.npy, windows.json
AZURE_OUTPUT_PREFIX: str  = os.getenv(
    "AZURE_OUTPUT_PREFIX", "AI/protocolimage/preprocessed"
)

# context-team dossier pipeline I/O (ai-context-team job):
#   silver/AI/ai-context-team/input/   ← request.json + media asset
#   silver/AI/ai-context-team/output/  ← sourced Markdown dossier
AZURE_CONTEXT_TEAM_INPUT: str  = os.getenv(
    "AZURE_CONTEXT_TEAM_INPUT",  "AI/ai-context-team/input"
)
AZURE_CONTEXT_TEAM_OUTPUT: str = os.getenv(
    "AZURE_CONTEXT_TEAM_OUTPUT", "AI/ai-context-team/output"
)

# ── Bronze — raw recordings ───────────────────────────────────────────────────
# Raw video recordings from eye-tracker / recording device:
#   bronze/protocolimage/RAW/<participant>/<session>/<file>
AZURE_BRONZE_RAW_PREFIX: str = os.getenv(
    "AZURE_BRONZE_RAW_PREFIX", "protocolimage/RAW"
)

# ── Azure ML workspace ────────────────────────────────────────────────────────
AML_SUBSCRIPTION_ID: str    = os.getenv("AML_SUBSCRIPTION_ID",   "72db8851-7710-43df-ad69-c0c0712252a3")
AML_RESOURCE_GROUP: str     = os.getenv("AML_RESOURCE_GROUP",    "rg-habs-ml-ai-prod-euw")
AML_WORKSPACE_NAME: str     = os.getenv("AML_WORKSPACE_NAME",    "habs-ml-ai-prod-euw")
AML_COMPUTE_GPU: str        = os.getenv("AML_COMPUTE_GPU",       "H100-EUW-INFERENCE")
AML_COMPUTE_CPU: str        = os.getenv("AML_COMPUTE_CPU",       "cpu-cluster")
