"""
processing_auxiliary.py
-----------------------
Auxiliary processing utilities shared across the pipeline.

Provides helper functions and classes for device detection, database
connectivity, image path resolution, image cropping, and the compound
identifier used across the FRESCO-derived semiotic analysis pipeline.

Sources
-------
- Device helpers  : ai-fresco-replicated/fresco/db_utils/device.py
- ExtendedID      : ai-fresco-replicated/fresco/db_utils/ExtendedID.py
- DB / logging    : ai-fresco-replicated/fresco/db_utils/misc.py
"""

from __future__ import annotations

import configparser
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def resolve_device(requested: str = "auto") -> str:
    """Return a torch device string based on *requested* and available hardware.

    Resolution order when ``requested == "auto"``: MPS → CUDA → CPU.
    Any explicit device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``)
    is returned as-is without probing hardware.
    """
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device(fallback: str = "auto") -> str:
    """Read device from the ``FRESCO_DEVICE`` env var, falling back to auto-detection."""
    return resolve_device(os.environ.get("FRESCO_DEVICE", fallback))


# ---------------------------------------------------------------------------
# ExtendedID — compound subject/image identifier
# ---------------------------------------------------------------------------

class ExtendedID:
    """Compound identifier combining a subject ID and an image ID.

    String representation: ``"<subjectId>_<imageId>"``.

    Attributes:
        subjectId: String identifier of a participant or subject.
        imageId:   String identifier of an image (unique within subjectId).
    """

    def __init__(self, subjectId: str, imageId: Union[str, int]):
        self.subjectId = str(subjectId)
        self.imageId   = str(imageId)

    def __str__(self) -> str:
        return f"{self.subjectId}_{self.imageId}"

    def __repr__(self) -> str:
        return f"ExtendedID(subjectId={self.subjectId!r}, imageId={self.imageId!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExtendedID) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def from_string(cls, input_str: Union[str, List[str]]) -> Union[ExtendedID, List[ExtendedID]]:
        """Parse ``"subjectId_imageId"`` string(s) into ExtendedID object(s)."""
        if isinstance(input_str, list):
            return [cls.from_string(k) for k in input_str]  # type: ignore[return-value]
        if isinstance(input_str, ExtendedID):
            return input_str
        if isinstance(input_str, str):
            parts = str(input_str).split("_", 1)
            return cls(parts[0], parts[1] if len(parts) > 1 else "")
        raise TypeError("input must be a string or list of strings")

    @classmethod
    def from_db_document(cls, document: dict) -> Optional[ExtendedID]:
        """Extract an ExtendedID from a MongoDB image document.

        Accepts documents with either an ``extendedId`` field or both
        ``subjectId`` and ``imageId`` fields.  Returns ``None`` when
        neither pattern is present.
        """
        if not isinstance(document, dict):
            raise TypeError("document must be a dict")
        if "extendedId" in document:
            return cls.from_string(document["extendedId"])  # type: ignore[return-value]
        if "subjectId" in document and "imageId" in document:
            return cls(document["subjectId"], document["imageId"])
        return None


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

# Module-level base directory for resolving relative image paths.
_input_base_dir: str | None = None


def start_database_connection(
    db_name: str,
    db_ip: str,
    log: logging.Logger,
    db_port: int = 27017,
    db_uri: str | None = None,
):
    """Connect to a MongoDB database and return a ``pymongo.Database`` object.

    When *db_uri* is provided it is used directly; otherwise *db_ip* and
    *db_port* are used to build the connection.

    Raises:
        TimeoutError:    If the server is unreachable.
        AttributeError:  If the expected ``images`` collection is missing.
    """
    from pymongo import MongoClient
    from pymongo import errors as mongo_errors

    try:
        client = MongoClient(db_uri) if db_uri else MongoClient(db_ip, db_port)
        log.info("Connected to MongoDB (%s)", db_uri or f"{db_ip}:{db_port}")
    except mongo_errors.ServerSelectionTimeoutError as exc:
        raise TimeoutError(str(exc)) from exc

    if db_name not in client.list_database_names():
        raise AttributeError(f"Database '{db_name}' not found")
    db = client[db_name]
    if "images" not in db.list_collection_names():
        raise AttributeError("Database missing required 'images' collection")
    return db


def start_db_config_and_logger(config_path: str, tool_name: str):
    """Initialise MongoDB, config, and logger from an INI config file.

    Args:
        config_path: Path to the ``.ini`` configuration file.
        tool_name:   Name used for the log file and logger instance.

    Returns:
        ``(db, config, logger)`` — pymongo Database, ConfigParser, Logger.
    """
    global _input_base_dir

    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise IOError(f"Cannot read config file: {config_path}")

    stage = os.getenv("PROJECT_STAGE", "dep")
    config["db"]["db_name"] = config["db"][f"db_name_{stage}"]
    config["db"]["db_port"] = config["db"][f"db_port_{stage}"]

    log = logging.getLogger(tool_name)
    log.setLevel(int(config["log"]["level"]))
    log_folder = config["log"]["log_folder"]
    os.makedirs(log_folder, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_folder, f"{tool_name}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s"))
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(int(config["log"]["level"]))
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    _input_base_dir = config["filesystem"].get("input_dir", "./input")

    db_uri  = os.getenv("FRESCO_MONGO_URI") or config["db"].get("db_uri", "").strip() or None
    db_port = int(config["db"].get("db_port", "27017"))
    db      = start_database_connection(
        config["db"]["db_name"], config["db"]["db_ip"], log,
        db_port=db_port, db_uri=db_uri,
    )
    return db, config, log


def push_result_to_db(db, extended_id: Union[ExtendedID, str], software_id: str, result) -> None:
    """Upsert an analysis result into the MongoDB collection named *software_id*.

    Args:
        db:           pymongo Database object.
        extended_id:  Image identifier (``ExtendedID`` or ``"subjectId_imageId"`` string).
        software_id:  Name of the analysis tool (used as collection name).
        result:       Python-primitive result object to store.

    Raises:
        TimeoutError: If the database is unreachable.
    """
    from pymongo import errors as mongo_errors

    try:
        db[software_id].update_one(
            filter={"extendedId": str(extended_id)},
            update={"$set": {
                "results":   result,
                "timestamp": datetime.now().strftime("%Y/%m/%d_%H:%M:%S"),
            }},
            upsert=True,
        )
    except mongo_errors.ServerSelectionTimeoutError as exc:
        raise TimeoutError(str(exc)) from exc


def fetch_image_document(db, extended_id: Union[ExtendedID, str]) -> Optional[dict]:
    """Return the ``images`` collection document for *extended_id*, or ``None``."""
    return db.images.find_one({"extendedId": str(extended_id)})


# ---------------------------------------------------------------------------
# Image path helpers
# ---------------------------------------------------------------------------

def get_image_path(db, extended_id: Union[ExtendedID, str]) -> Optional[str]:
    """Resolve the filesystem path for an image stored in the database.

    Relative paths stored in MongoDB are resolved against *_input_base_dir*
    (set by ``start_db_config_and_logger``).
    """
    doc = fetch_image_document(db, extended_id)
    if doc is None:
        return None
    raw = doc.get("path", "")
    if os.path.isabs(raw):
        return raw
    if _input_base_dir:
        return os.path.join(_input_base_dir, raw)
    return raw


def crop_image(
    image,
    bbox: List[Union[int, float]],
    padding_pct: float = 0.0,
    min_size: int = 0,
    squared: bool = False,
) -> Tuple[List[int], object]:
    """Crop *image* to the region defined by *bbox*, with optional padding.

    Args:
        image:       NumPy HxWxC uint8 image array.
        bbox:        Bounding box in ``[x, y, w, h]`` format (top-left corner).
        padding_pct: Percentage by which to expand the box on each side.
        min_size:    Minimum edge length of the cropped region (pixels).
        squared:     If True, force a square crop.

    Returns:
        ``(actual_bbox, cropped_image)`` — the applied bbox and the cropped array.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h   = bbox
    min_half      = min_size // 2
    hx            = int(max(w / 2 * (1 + padding_pct / 100), min_half))
    hy            = int(max(h / 2 * (1 + padding_pct / 100), min_half))

    if squared:
        hx = hy = max(hx, hy)

    cx   = min(max(hx, int(x + w / 2)), img_w - hx)
    cy   = min(max(hy, int(y + h / 2)), img_h - hy)
    left = max(cx - hx, 0)
    right = min(cx + hx, img_w - 1)
    top   = max(cy - hy, 0)
    bot   = min(cy + hy, img_h - 1)

    return [left, right, right - left, bot - top], image.copy()[top:bot, left:right, :]
