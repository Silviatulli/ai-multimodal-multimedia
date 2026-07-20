"""
azure_blob.py
-------------
Azure Blob Storage helpers for the multimodal preprocessing pipeline.

Provides authentication, SAS URL generation, blob listing, and upload so that
all other modules remain free of Azure-specific dependencies.

    build_client()         → BlobServiceClient
    make_sas_url(...)      → HTTPS URL with read-only SAS token
    list_video_blobs(...)  → [(blob_name, sas_url), …]
    upload_blob(...)       → upload raw bytes to a blob path
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    UserDelegationKey,
)

try:
    from .config import (
        AZURE_CONNECTION_STRING,
        AZURE_CONTAINER_NAME,
        AZURE_OUTPUT_PREFIX,
        AZURE_STORAGE_ACCOUNT,
        AZURE_VIDEO_PREFIX,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        AZURE_CONNECTION_STRING,
        AZURE_CONTAINER_NAME,
        AZURE_OUTPUT_PREFIX,
        AZURE_STORAGE_ACCOUNT,
        AZURE_VIDEO_PREFIX,
    )

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# AZURE_STORAGE_ACCOUNT is now defined in config.py and imported above


def build_client() -> BlobServiceClient:
    """Return a ``BlobServiceClient``.

    Priority:
      1. ``AZURE_STORAGE_CONNECTION_STRING`` env var (account key auth)
      2. ``DefaultAzureCredential`` via the Azure CLI session (``az login``)
    """
    if AZURE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    logger.info("No connection string found — using DefaultAzureCredential (az login).")
    account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


def make_sas_url(
    client: BlobServiceClient,
    blob_name: str,
    expiry_hours: int = 1,
) -> str:
    """Return a read-only HTTPS SAS URL for *blob_name*, valid for *expiry_hours*.

    Three strategies in priority order:
      1. Account-key SAS   — when credential has ``account_key``
      2. User-delegation SAS — when logged in via ``az login`` / managed identity
      3. Bare blob URL     — when credential is already a SAS token
    """
    blob_client = client.get_blob_client(
        container=AZURE_CONTAINER_NAME, blob=blob_name
    )
    expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

    # Strategy 1: account key
    try:
        account_key = client.credential.account_key
        sas = generate_blob_sas(
            account_name=client.account_name,
            container_name=AZURE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        return f"{blob_client.url}?{sas}"
    except AttributeError:
        pass

    # Strategy 2: user delegation key (DefaultAzureCredential / az login)
    try:
        udk: UserDelegationKey = client.get_user_delegation_key(
            key_start_time=datetime.now(timezone.utc),
            key_expiry_time=expiry,
        )
        sas = generate_blob_sas(
            account_name=client.account_name,
            container_name=AZURE_CONTAINER_NAME,
            blob_name=blob_name,
            user_delegation_key=udk,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        return f"{blob_client.url}?{sas}"
    except Exception as exc:
        logger.warning("User-delegation SAS failed (%s); using bare URL.", exc)

    # Strategy 3: bare URL (connection string already embeds SAS)
    return blob_client.url


def upload_blob(
    client: BlobServiceClient,
    blob_path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    overwrite: bool = True,
) -> None:
    """Upload *data* to *blob_path* inside ``AZURE_CONTAINER_NAME``.

    *blob_path* is relative to the container root, e.g.
    ``"NEURO/protocolimage/preprocessed/cat/clip/frames.npy"``.
    """
    blob_client = client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_path)
    blob_client.upload_blob(
        data,
        overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type),
    )
    logger.debug("Uploaded %d bytes → %s/%s", len(data), AZURE_CONTAINER_NAME, blob_path)


def output_prefix_for(blob_name: str, video_prefix: str = AZURE_VIDEO_PREFIX,
                      output_prefix: str = AZURE_OUTPUT_PREFIX) -> str:
    """Derive the output blob prefix for a given input *blob_name*.

    Maps ``<video_prefix>/<category>/<stem>.<ext>``
    →   ``<output_prefix>/<category>/<stem>/``
    """
    relative = blob_name[len(video_prefix):].lstrip("/")   # <category>/<stem>.<ext>
    stem_path = "/".join(Path(relative).with_suffix("").parts)
    return f"{output_prefix}/{stem_path}"


def list_video_blobs(client: BlobServiceClient, n: int) -> list[tuple[str, str]]:
    """Return up to *n* ``(blob_name, sas_url)`` pairs under ``AZURE_VIDEO_PREFIX``.

    Only blobs with a recognised video extension are included.
    """
    container = client.get_container_client(AZURE_CONTAINER_NAME)
    results: list[tuple[str, str]] = []
    for blob in container.list_blobs(name_starts_with=AZURE_VIDEO_PREFIX or None):
        if len(results) >= n:
            break
        if Path(blob.name).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        results.append((blob.name, make_sas_url(client, blob.name)))
    return results
