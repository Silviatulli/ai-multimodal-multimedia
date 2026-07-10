"""
features_multimedia.py
----------------------
Feature extraction methods for multimedia content.

Implements extraction of audio, video,visual, audiovisual, textual, and web-based
features from heterogeneous media sources, including spectral descriptors,
visual saliency, motion, natural language embeddings, sentiment and readability
measures from text, and structural or semantic features from web pages —
all intended for multimodal affective analysis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import CATEGORIES_DIR, DATA_DIR, NUM_FRAMES
from .multimedia_data_preprocessing import (
    augment_frames,
    build_video_index,
    get_video_info,
    get_video_windows,
    sample_frames,
    sample_frames_from_segment,
)

logger = logging.getLogger(__name__)