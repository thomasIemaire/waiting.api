from __future__ import annotations

from typing import Any, Tuple
from flask import jsonify
import hashlib, base64, uuid, hmac
from datetime import datetime, timezone

def json_error(message: str, status: int = 400) -> Tuple[Any, int]:
    return jsonify({"error": message}), status

def bump_version(version: str, bump: str) -> str:
    major, minor = map(int, version.split("."))
    if bump == "major":
        return f"{major + 1}.0"
    return f"{major}.{minor + 1}"

def generate_apikey() -> str:
    return str(uuid.uuid4())

def b64url(bytes: bytes) -> bytes:
    return base64.urlsafe_b64encode(bytes)

def hash_password(password: str, apikey: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), b64url(uuid.UUID(apikey).bytes), 269_874)
    return base64.b64encode(dk).decode("ascii")

def verify_password(password: str, hashed_password: str, apikey: str) -> bool:
    return hmac.compare_digest(hash_password(password, apikey), hashed_password)

def get_current_time() -> datetime:
    return datetime.now(timezone.utc)

def is_integer(value: any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False