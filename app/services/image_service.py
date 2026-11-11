from __future__ import annotations

import io
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..config import Settings

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_FORMATS = {"PNG", "JPEG"}


async def store_image(upload: UploadFile, settings: Settings) -> Image.Image:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG images are accepted")
    data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the upload size limit")
    if not data:
        raise HTTPException(status_code=400, detail="Image upload is empty")
    try:
        with Image.open(io.BytesIO(data)) as checked:
            checked.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format")
            converted = image.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from None

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return converted
