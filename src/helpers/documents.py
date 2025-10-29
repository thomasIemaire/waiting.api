import base64
import mimetypes
import os
import re

MIME_MAP_EXT = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "text/plain": ".txt",
}

def _ext_from(mime: str | None, filename: str | None) -> str:
    if mime and mime in MIME_MAP_EXT:
        return MIME_MAP_EXT[mime]
    if filename:
        _, ext = os.path.splitext(filename)
        if ext:
            return ext.lower()
    if mime:
        ext = mimetypes.guess_extension(mime)
        if ext:
            return ext
    return ""

DATAURL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)

def _decode_base64_any(data_str: str) -> tuple[bytes, str | None]:
    """
    Accepte base64 pur OU data URL 'data:mime;base64,...'
    Retourne (raw_bytes, mime_detecte_ou_None)
    """
    m = DATAURL_RE.match(data_str or "")
    if m:
        return base64.b64decode(m.group("data")), m.group("mime").strip()
    return base64.b64decode(data_str), None
