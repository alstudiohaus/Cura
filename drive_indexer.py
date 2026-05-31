"""
Google Drive photo indexer for Cura.
Lists images in a folder, downloads thumbnails, runs Claude vision analysis.
"""

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
import httpx
from typing import List, Dict
import asyncio


SUPPORTED_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
]


def get_drive_service(access_token: str):
    creds = Credentials(token=access_token)
    return build("drive", "v3", credentials=creds)


def list_images_in_folder(service, folder_id: str, page_size: int = 200) -> List[Dict]:
    """List all image files in a Google Drive folder."""
    mime_filter = " or ".join([f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES])
    query = f"'{folder_id}' in parents and ({mime_filter}) and trashed=false"

    results = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            pageSize=page_size,
            fields="nextPageToken, files(id, name, mimeType, thumbnailLink, size)",
            pageToken=page_token,
        ).execute()

        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


async def download_thumbnail(thumbnail_url: str) -> str:
    """Download a Drive thumbnail and return as base64."""
    async with httpx.AsyncClient() as client:
        # Use higher resolution thumbnail
        url = thumbnail_url.replace("=s220", "=s400")
        resp = await client.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")


async def index_folder(service, folder_id: str, analyze_fn, batch_size: int = 10) -> List[Dict]:
    """
    Full indexing pipeline for a Drive folder.
    Returns list of analyzed photo metadata.
    """
    files = list_images_in_folder(service, folder_id)
    analyzed = []

    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        tasks = []
        for f in batch:
            if f.get("thumbnailLink"):
                tasks.append(process_file(f, analyze_fn))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, dict):
                analyzed.append(r)

        # Progress: (i + batch_size) / len(files) * 100
        progress = min(100, int((i + batch_size) / len(files) * 100))
        yield progress, analyzed

    return analyzed


async def process_file(file_meta: Dict, analyze_fn) -> Dict:
    """Download thumbnail and run Claude vision analysis."""
    try:
        image_b64 = await download_thumbnail(file_meta["thumbnailLink"])
        mime = file_meta.get("mimeType", "image/jpeg")
        # Normalize heic to jpeg for Claude
        if mime == "image/heic":
            mime = "image/jpeg"

        analysis = await analyze_fn(image_b64, mime)
        return {
            "drive_id": file_meta["id"],
            "filename": file_meta["name"],
            "mime_type": mime,
            "thumbnail_b64": image_b64,
            **analysis,
        }
    except Exception as e:
        return {"drive_id": file_meta["id"], "error": str(e)}
