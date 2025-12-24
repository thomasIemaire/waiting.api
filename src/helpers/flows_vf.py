import base64
import io
import os
import time
import json
import re
import ast
import operator as op
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from datetime import datetime
from typing import Any, Callable, Dict, Literal, Optional, Union
from collections.abc import Mapping, Sequence


import warnings

# ========= Hugging Face cache / warnings =========
def configure_hf_cache(cache_dir: str | None = None, *, debug: bool = False) -> str:
    """Configure un cache Hugging Face stable (évite les re-téléchargements entre runs).

    Priorité:
      1) param cache_dir
      2) env SARDINE_HF_CACHE
      3) env HF_HOME (si déjà défini)
      4) défaut: ~/.cache/huggingface
    """
    if cache_dir is None:
        cache_dir = (
            os.getenv("SARDINE_HF_CACHE")
            or os.getenv("HF_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )

    cache_dir = os.path.abspath(cache_dir)

    # Ne remplace pas une config existante, sauf si SARDINE_HF_CACHE est explicitement fourni
    if os.getenv("SARDINE_HF_CACHE"):
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    else:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))

    # Optionnel: éviter la télémétrie
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if debug:
        print(f"[INFO] HF cache dir (requested): {cache_dir}")
        try:
            from huggingface_hub import constants as hf_consts  # import après config env
            print(f"[INFO] Resolved HF_HOME     : {hf_consts.HF_HOME}")
            print(f"[INFO] Resolved HF_HUB_CACHE: {hf_consts.HF_HUB_CACHE}")
        except Exception as e:
            print(f"[INFO] huggingface_hub not available yet: {e}")

    return cache_dir


def configure_hf_warnings(*, silence: bool = False) -> None:
    """Masque 2 warnings très fréquents (optionnel)."""
    if not silence:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*`resume_download` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*byte fallback option.*not implemented.*fast tokenizers.*",
        category=UserWarning,
    )

# IMPORTANT: configurer AVANT tout import de transformers/gliner
configure_hf_cache()
configure_hf_warnings(silence=os.getenv("SARDINE_SILENCE_HF_WARNINGS", "0") == "1")

from PIL import Image, ImageFilter, ImageOps

start_import_time = time.time()
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:
    from gliner import GLiNER
except Exception:  # pragma: no cover
    GLiNER = None  # type: ignore
end_import_time = time.time()


# ========= CONSTANTS =========
FLOW_MAX_WORKERS = 8
OCR_MAX_WORKERS = 24

_YOLO_CACHE: dict[str, Any] = {}
_YOLO_LOCK = threading.Lock()

_AGENT_CACHE: dict[str, dict[str, Any]] = {}
_AGENT_LOCK = threading.Lock()

def _require(dep: Any, name: str) -> None:
    if dep is None:
        raise RuntimeError(
            f"Dépendance optionnelle manquante: {name}. "
            f"Installez-la pour utiliser les fonctionnalités associées."
        )


# ========= Patterns =========
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

_BRACED_PATH_RE = re.compile(r"\$\{([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\}")
_DOLLAR_PATH_RE = re.compile(r"\$([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
_DATAURL_RE = re.compile(r"data:(?P<mime>[\w/-]+)?(;base64)?,(?P<data>.+)", re.IGNORECASE)


# ========= Database =========
def get_db_connection() -> None:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/sardine")
    if not mongo_uri:
        print("MONGODB_URI environment variable is not set. Database connection will not be established.")
        return
    
    try:
        from pymongo import MongoClient

        client = MongoClient(mongo_uri)
        return client.get_default_database()
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

DB = get_db_connection()
DB_AGENTS_COLLECTION = "agents"

# ========= Helpers =========
def print_debug(
    printable: Any,
    debug: bool,
    *,
    tags: Union[list[str], str, None] = None,
) -> None:
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if isinstance(tags, str):
        tags = [tags]
    tag_part = f"[{'.'.join(tags)}]" if tags else ""
    if debug:
        print(f"[DEBUG]{tag_part} {date} - {printable}")


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Deep merge dict2 into dict1, returning a new dict."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_value_from_path(data: dict, path: str) -> Any:
    keys = path.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def set_value_at_path(data: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    current: Any = data
    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Cannot set into non-dict at '{key}' for path '{path}'")
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    if not isinstance(current, dict):
        raise TypeError(f"Cannot set into non-dict for path '{path}'")
    current[keys[-1]] = value


def add_value_at_path(data: dict, path: str, value: dict) -> None:
    for k, v in value.items():
        set_value_at_path(data, f"{path}.{k}", v)


# ========= DSL (expressions) =========
def _to_python_expr(expr: str) -> str:
    """Convert DSL to restricted python expression.

      ${a.b} -> $a.b
      $a.b   -> get("a.b")

    Indexing/slicing kept, e.g.:
      $seller.vat.number[-9:] -> get("seller.vat.number")[-9:]
    """
    expr = _BRACED_PATH_RE.sub(lambda m: f"${m.group(1)}", expr)
    expr = _DOLLAR_PATH_RE.sub(lambda m: f'get("{m.group(1)}")', expr)
    return expr


def _safe_eval(expr: str, funcs: dict[str, Callable[..., Any]]) -> Any:
    """Evaluate an expression with a restricted AST."""
    node = ast.parse(expr, mode="eval").body

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Constant):
            return n.value

        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](ev(n.left), ev(n.right))

        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(n.op)](ev(n.operand))

        if isinstance(n, ast.Subscript):
            base = ev(n.value)
            sl = n.slice
            if isinstance(sl, ast.Slice):
                lower = ev(sl.lower) if sl.lower else None
                upper = ev(sl.upper) if sl.upper else None
                step = ev(sl.step) if sl.step else None
                return base[slice(lower, upper, step)]
            return base[ev(sl)]

        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = n.func.id
            if name not in funcs:
                raise ValueError(f"Fonction non autorisée: {name}")
            args = [ev(a) for a in n.args]
            kwargs = {kw.arg: ev(kw.value) for kw in n.keywords}
            return funcs[name](*args, **kwargs)

        if isinstance(n, ast.Name) and n.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[n.id]

        raise ValueError(f"Expression non autorisée: {ast.dump(n)}")

    return ev(node)


def resolve_value(data: Optional[dict], value: Any) -> Any:
    """Resolve a DSL expression string using data."""
    if not isinstance(value, str):
        return value

    expr = value.strip()
    if "$" not in expr:
        return value

    ctx = data or {}

    def get(path: str) -> Any:
        return get_value_from_path(ctx, path)

    funcs: dict[str, Callable[..., Any]] = {
        "get": get,
        "str": str,
        "int": int,
        "float": float,
        "len": len,
        "round": round,
        "abs": abs,
        "max": max,
        "min": min,
    }

    pyexpr = _to_python_expr(expr)
    return _safe_eval(pyexpr, funcs)


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# ========= Sardine / Images utilities =========
def extract_images_from_base64(
    base64_data: Any,
    *,
    debug: bool = False,
    page_dpi: int = 200,
    page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
) -> list[Image.Image]:
    images = load_images_from_base64(
        base64_data,
        page_mode=page_mode,
        page_dpi=page_dpi,
        debug=debug,
    )

    if not images:
        print_debug("[SARDINE] No images extracted from base64 data.", debug, tags=["SARDINE", "EXTRACT_IMAGES"])
        return []

    return images


def load_images_from_base64(
    base64_data: Any,
    page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
    page_dpi: int = 200,
    *,
    debug: bool = False,
) -> list[Image.Image]:
    raw_bytes, mime = _try_decode_base64(base64_data)

    if raw_bytes is None:
        print_debug("[SARDINE] No valid base64 data found.", debug, tags=["SARDINE", "LOAD_IMAGES"])
        return []

    try:
        is_pdf = (mime == "application/pdf") or (mime is None and raw_bytes[:4] == b"%PDF")
        return (
            _load_pil_from_pdf(raw_bytes, page_mode=page_mode, dpi=page_dpi, debug=debug)
            if is_pdf
            else [_load_pil_from_bytes(raw_bytes)]
        )
    except Exception as e:
        print_debug(f"[SARDINE] Error loading images: {e}", debug, tags=["SARDINE", "LOAD_IMAGES"])
        return []


def _load_pil_from_pdf(
    raw_bytes: bytes,
    page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
    dpi: int = 200,
    *,
    debug: bool = False,
) -> list[Image.Image]:
    try:
        import fitz  # type: ignore
    except ImportError:
        print_debug("[SARDINE] fitz (PyMuPDF) library is not installed.", debug, tags=["SARDINE", "LOAD_PDF"])
        return []

    images: list[Image.Image] = []
    try:
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes()
                img = _load_pil_from_bytes(img_bytes)
                images.append(img)
                print_debug(f"[SARDINE] Loaded page {page_num + 1}/{len(doc)}", debug, tags=["SARDINE", "LOAD_PDF"])
                if page_mode == "first_page_only":
                    break
    except Exception as e:
        print_debug(f"[SARDINE] Error loading PDF pages: {e}", debug, tags=["SARDINE", "LOAD_PDF"])
        return []

    return images


def _load_pil_from_bytes(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _try_decode_base64(s: Any) -> tuple[Optional[bytes], Optional[str]]:
    if not isinstance(s, str) or not s.strip():
        return None, None

    m = _DATAURL_RE.match(s.strip())
    if m:
        try:
            return base64.b64decode(m.group("data"), validate=True), (m.group("mime") or "").lower()
        except Exception:
            return None, None

    try:
        return base64.b64decode(s), None
    except Exception:
        return None, None


# ========= YOLO / OCR =========
def get_cached_yolo_model(model_path: str, *, debug: bool = False) -> Any:
    _require(YOLO, "ultralytics")

    abs_path = os.path.abspath(model_path)
    print_debug(f"[SARDINE] Loading YOLO model from: {abs_path}", debug, tags=["SARDINE", "YOLO"])

    if abs_path in _YOLO_CACHE:
        print_debug(f"[SARDINE] Using cached YOLO model for: {abs_path}", debug, tags=["SARDINE", "YOLO"])
        return _YOLO_CACHE[abs_path]

    with _YOLO_LOCK:
        if abs_path not in _YOLO_CACHE:
            print_debug(f"[SARDINE] Loading YOLO model into cache: {abs_path}", debug, tags=["SARDINE", "YOLO"])
            _YOLO_CACHE[abs_path] = YOLO(model_path)  # type: ignore[misc]

    return _YOLO_CACHE[abs_path]


def classify_pages(model_cls: Any, sources: list[Image.Image], device: str = "cpu", *, debug: bool = False) -> list[str]:
    cls = model_cls.predict(source=sources, device=device, save=False, verbose=False)
    results: list[str] = []
    for r in cls:
        if hasattr(r, "probs") and r.probs is not None:
            top1 = r.probs.top1
            name = r.names[top1]
        else:
            print_debug("[SARDINE] No classification probabilities found.", debug, tags=["SARDINE", "CLASSIFY"])
            name = "unknown"
        results.append(name)
    return results


def _predict_document_class(images: list[Image.Image], model_path: str, device: str = "cpu", *, debug: bool = False) -> list[str]:
    try:
        model = get_cached_yolo_model(model_path, debug=debug)
        return classify_pages(model, images, device=device, debug=debug)
    except Exception as e:
        print_debug(f"[SARDINE] Document classification failed: {e}", debug, tags=["SARDINE", "CLASSIFY"])
        return []


def _ocr_pil_image(image: Image.Image, *, lang: str = "fra+eng", debug: bool = False) -> str:
    _require(pytesseract, "pytesseract")

    image = image.convert("RGB")
    image = _deskew_and_orient(image, debug=debug)
    image = _preprocess_for_ocr(image, debug=debug)

    text = pytesseract.image_to_string(image, lang=lang, config="--oem 3 --psm 6")  # type: ignore[union-attr]
    return (text or "").strip()


def _ocr_many_pil(images: list[Image.Image], *, debug: bool = False) -> list[str]:
    max_workers = min(OCR_MAX_WORKERS, len(images))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_ocr_pil_image, c, debug=debug) for c in images]
        return [f.result() for f in futures]


def _deskew_and_orient(image: Image.Image, *, debug: bool = False) -> Image.Image:
    _require(pytesseract, "pytesseract")

    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)  # type: ignore[union-attr]
        rotation = osd.get("rotate", 0) or 0
        if rotation and rotation % 360 != 0:
            image = image.rotate(360 - rotation, expand=True)
    except Exception as e:
        print_debug(f"[SARDINE] Deskew and orient failed: {e}", debug, tags=["SARDINE", "OCR"])

    return image


def _preprocess_for_ocr(image: Image.Image, *, debug: bool = False) -> Image.Image:
    _require(np, "numpy")
    _require(cv2, "opencv-python")

    arr = np.array(image, dtype=np.uint8)  # type: ignore[union-attr]
    ink = 255 - np.max(arr, axis=2).astype(np.uint8)  # type: ignore[union-attr]
    gray = (255 - ink).astype(np.uint8)
    gray = cv2.medianBlur(gray, 3)  # type: ignore[union-attr]

    gpil = Image.fromarray(gray)
    gpil = ImageOps.autocontrast(gpil, cutoff=1)
    gpil = gpil.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))

    W, H = gpil.size
    if H < 120:
        scale = max(2.0, 120 / max(1, H))
        new_w = int(W * scale)
        new_h = int(H * scale)
        gpil = gpil.resize((new_w, new_h), resample=Image.BICUBIC)
    return gpil


def _predict_document_detection(
    images: list[Image.Image],
    model_path: str,
    device: str = "cpu",
    conf: float = 0.25,
    *,
    debug: bool = False,
    padding: int = 8,
    fallback_full_page_ocr: bool = True,
) -> tuple[list[list[dict[str, Any]]], float, float]:
    _require(np, "numpy")

    model_det = get_cached_yolo_model(model_path, debug=debug)

    try:
        start_detection = time.time()
        det_results = model_det.predict(
            source=images, save=False, conf=conf, device=device, verbose=False
        )
        end_detection = time.time()
        duration_detection = end_detection - start_detection
        print_debug(f"[SARDINE] Document detection completed in {duration_detection:.2f}s", debug, tags=["SARDINE", "DETECT"])
    except Exception as e:
        print_debug(f"[SARDINE] Document detection failed: {e}", debug, tags=["SARDINE", "DETECT"])
        return []

    start_ocr = time.time()

    output: list[list[dict[str, Any]]] = []

    default_names = {0: "text", 1: "text-column", 2: "table", 3: "logo", 4: "signature"}

    for img, res in zip(images, det_results):
        page_items: list[dict[str, Any]] = []
        W, H = img.size

        names = getattr(res, "names", None) or getattr(model_det, "names", None) or default_names

        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy
            cls = boxes.cls
            confs = getattr(boxes, "conf", None)

            # to numpy
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy.numpy()
            cls = cls.cpu().numpy() if hasattr(cls, "cpu") else cls.numpy()
            confs_np = None
            if confs is not None:
                confs_np = confs.cpu().numpy() if hasattr(confs, "cpu") else confs.numpy()

            xyxy = xyxy.astype(int)
            cls = cls.astype(int)

            order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))
            xyxy = xyxy[order]
            cls = cls[order]
            if confs_np is not None:
                confs_np = confs_np[order]

            crops: list[Image.Image] = []
            meta: list[tuple[int, int, int, int, int, float | None]] = []
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
                x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
                if x2p > x1p and y2p > y1p:
                    crops.append(img.crop((x1p, y1p, x2p, y2p)))
                    c = int(cls[i])
                    cconf = float(confs_np[i]) if confs_np is not None else None
                    meta.append((x1p, y1p, x2p, y2p, c, cconf))

            if crops:
                texts = _ocr_many_pil(crops, debug=debug)
                for (x1p, y1p, x2p, y2p, c, cconf), t in zip(meta, texts):
                    zone_type = names.get(c, str(c)) if isinstance(names, dict) else str(c)
                    page_items.append(
                        {
                            "type": zone_type,
                            "class_id": c,
                            "conf": cconf,
                            "bbox": [x1p, y1p, x2p, y2p],
                            "text": (t or "").strip(),
                        }
                    )

        if fallback_full_page_ocr and not page_items:
            full_text = _ocr_pil_image(img, debug=debug)
            if full_text:
                page_items.append(
                    {
                        "type": "page",
                        "class_id": None,
                        "conf": None,
                        "bbox": [0, 0, W, H],
                        "text": full_text.strip(),
                    }
                )

        output.append(page_items)

    end_ocr = time.time()
    duration_ocr = end_ocr - start_ocr
    print_debug(f"[SARDINE] OCR on detected zones completed in {duration_ocr:.2f}s", debug, tags=["SARDINE", "DETECT"])

    return output, duration_detection, duration_ocr


def _clean_detection_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ========= Agent Utilities =========
def get_cached_agent_model(
        reference: str,
        version: str = "latest",
        *,
        debug: bool = False
) -> Any:
    print_debug(f"Loading agent model: {reference} (version: {version})", debug, tags=["AGENT", "MODEL"])
    
    key_cache = f"{reference}::{version}"
    if key_cache in _AGENT_CACHE:
        print_debug(f"Using cached agent model for: {key_cache}", debug, tags=["AGENT", "MODEL"])
        model, agent = _AGENT_CACHE[key_cache]["model"], _AGENT_CACHE[key_cache]["agent"]
        return model, agent
    
    agent = get_agent_config(reference, version=version, debug=debug)
    if not agent or not isinstance(agent, dict):
        raise ValueError(f"Agent model not found or invalid: {reference} (version: {version})")
    
    version = agent.get("version") if version == "latest" else version
    key_cache = f"{reference}::{version}"

    model_path = agent.get("path", "")
    if not model_path:
        raise ValueError(f"Agent model path missing for: {reference} (version: {version})")

    model_mapper = agent.get("mapper", None)
    if model_mapper is None:
        raise ValueError(f"Agent model mapper missing for: {reference} (version: {version})")

    model_mapper = transform_mapper(model_mapper, debug=debug)
    agent["transformed_mapper"] = model_mapper
    agent["labels"] = list(model_mapper.keys())
    
    with _AGENT_LOCK:
        if key_cache not in _AGENT_CACHE:
            print_debug(f"Loading agent model into cache: {key_cache}", debug, tags=["AGENT", "MODEL"])
            _AGENT_CACHE[key_cache] = {"model": GLiNER.from_pretrained(model_path), "agent": agent }

    model, agent = _AGENT_CACHE[key_cache]["model"], _AGENT_CACHE[key_cache]["agent"]
    return model, agent


def get_agent_config(
    reference: str,
    version: str = "latest",
    *,
    debug: bool = False
) -> Any:
    if DB is None:
        print_debug("Database connection is not available.", debug, tags=["AGENT", "CONFIG", "ERROR"])
        return None

    col = DB[DB_AGENTS_COLLECTION]

    try:
        if version != "latest":
            return col.find_one({"reference": reference, "version": version})

        cursor = col.find({"reference": reference}).sort("version", -1).limit(1)
        return next(cursor, None)

    except Exception as e:
        print_debug(f"Failed to fetch agent config: {e}", debug, tags=["AGENT", "CONFIG", "ERROR"])
        return None


def transform_mapper(
        mapper: dict,
        *,
        debug: bool = False
) -> dict:
    output: dict[str, str] = {}

    def walk(obj, path: str):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                k_str = str(k)
                new_path = f"{path}.{k_str}" if path else k_str
                walk(v, new_path)
            return

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
            return

        if not isinstance(obj, str):
            print_debug(f"Mapper value at path '{path}' is not a string: {obj}", debug, tags=["AGENT", "MAPPER", "WARNING"])
            return

        if obj in output:
            print_debug(f"Duplicate mapper label found: {obj} (existing path: {output[obj]}, new path: {path})", debug, tags=["AGENT", "MAPPER", "WARNING"])
            return

        output[obj] = path

    walk(mapper, "")
    return output


def _predict_agent(
        text: str,
        model: Any,
        agent: dict,
        *,
        conf: float = 0.5,
        debug: bool = False
) -> tuple[bool, list[Dict[str, Any]]]:
    model_labels = agent.get("labels", [])
    if not model_labels:
        print_debug(f"No labels defined in agent config", debug, tags=["AGENT", "PREDICT", "ERROR"])
        return False, []
    
    output: list[Dict[str, Any]] = []

    try:
        entities = model.predict_entities(text, labels=model_labels, threshold=conf)
        for ent in entities:
            start, end = ent["start"], ent["end"]
            label, score = ent["label"], ent["score"] or 0.0 
            output.append({
                "label": label,
                "start": start,
                "end": end,
                "score": score,
                "value": text[start:end],
            })
    except Exception as e:
        print_debug(f"Agent prediction failed: {e}", debug, tags=["AGENT", "PREDICT", "ERROR"])
        return False, []
    
    return True, output


def best_entities_by_label(
    agent_output: list[list[Dict[str, Any]]],
    *,
    debug: bool = False
) -> dict[str, Dict[str, Any]]:
    best_entities: dict[str, Dict[str, Any]] = {}
    for page in agent_output:
        for ent in page:
            label = ent.get("label", "")
            score = ent.get("score", 0.0)
            if label not in best_entities or score > best_entities[label].get("score", 0.0):
                best_entities[label] = ent
    return best_entities


def map_agent_output(
    agent_output: list[Dict[str, Any]],
    model_mapper: dict[str, str],
    *,
    debug: bool = False
) -> dict:
    mapped_output: dict = {}
    for item in agent_output:
        label = item.get("label", "")
        value = item.get("value", "")
        if label in model_mapper:
            key_path = model_mapper[label]
            set_value_at_path(mapped_output, key_path, value)
    return mapped_output


# ========= Node Utilities =========
def get_node_by_id(flow: dict, node_id: str) -> Optional[dict]:
    return flow.get(node_id)


def get_node_children(node: dict) -> Any:
    return node.get("outputs", [])


def get_node_parents(node: dict) -> list[str]:
    return node.get("inputs", [])


def get_node_config(node: dict) -> dict:
    return node.get("config", {})


def add_user_approval(
    input_ctx: dict,
    message: str,
    *,
    key_path: Optional[str] = None,
    debug: bool = False,
) -> None:
    print_debug(f"User approval requested: {message}", debug, tags=["NODE", "APPROVAL"])
    approval_id = uuid4().hex if not key_path else key_path.replace(".", "_")
    set_value_at_path(
        input_ctx,
        f"user_approvals.{approval_id}",
        {
            "message": message,
            "approved": False,
        },
    )


# ========= Nodes Processing =========
# Convention: une node retourne toujours
#   (status: bool, children_ids: list[str], output: dict, images: list[Image.Image], texts: list[list[str]])

def process_node(
    node: dict,
    input_ctx: dict,
    base64_data: Any,
    *,
    node_id: Optional[str] = None,
    debug: bool = False,
    images: Optional[list[Image.Image]] = None,
    texts: Optional[list[list[str]]] = None,
) -> tuple[bool, list[str], dict, list[Image.Image], list[list[str]]]:
    start = time.time()
    print_debug(f"START Processing {node_id} (Type: {node.get('type')})", debug, tags=["NODE", "INFO"])

    node_config = get_node_config(node)
    node_root = node_config.get("root")

    images = list(images or [])
    texts = list(texts or [])

    status, children_ids, output, images, texts = execute_node_by_type(
        node,
        input_ctx,
        base64_data,
        debug=debug,
        images=images,
        texts=texts,
        node_id=node_id
    )

    wrapped_output = output if not node_root else {node_root: output}

    end = time.time()
    set_value_at_path(input_ctx, f"_debug.nodes.{node_id}.duration", end - start)
    print_debug(f"END Processing {node_id} (Duration: {end - start:.2f}s)", debug, tags=["NODE", "INFO"])

    return status, children_ids, wrapped_output, images, texts


def execute_node_by_type(
    node: dict,
    input_ctx: dict,
    base64_data: Any,
    *,
    debug: bool = False,
    node_id: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    texts: Optional[list[list[str]]] = None,
) -> tuple[bool, list[str], dict, list[Image.Image], list[list[str]]]:
    node_type = node.get("type")
    if not node_type:
        print_debug("Node type is missing.", debug, tags=["NODE", "ERROR"])
        raise ValueError("Node type is missing")

    images = images or []
    texts = texts or []

    status = True
    output: dict = {}

    # Par défaut : outputs = liste de node_ids
    children_ids: list[str] = []

    match node_type:
        case "edit":
            output = node_edit(node, input_ctx, debug=debug)
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case "switch":
            status, branch = node_switch(node, input_ctx, debug=debug)
            outputs = get_node_children(node)
            # outputs attendu: {"match": ["..."], "default": ["..."]}
            if isinstance(outputs, dict):
                children_ids = list(outputs.get(branch, outputs.get("default", [])))
            else:
                children_ids = []

        case "if":
            status, branch = node_if(node, input_ctx, debug=debug)
            outputs = get_node_children(node)
            if isinstance(outputs, dict):
                children_ids = list(outputs.get(branch, outputs.get("false", [])))
            else:
                children_ids = []

        case "http":
            status, output = node_http(node, input_ctx, debug=debug)
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case "final":
            status, output, children_ids = True, input_ctx, []

        case "sardine":
            status, images, output = node_sardine(node, input_ctx, base64_data, debug=debug, images=images)
            children_ids = get_node_children(node)["valid"] if images else get_node_children(node)["invalid"]

        case "zone-detection":
            status, images, detections = node_detection(node, input_ctx, base64_data, node_id=node_id, debug=debug, images=images)
            texts = detections
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case "agent":
            status, output = node_agent(
                node,
                texts,
                debug=debug,
            )
            children_ids = get_node_children(node)

        case "debug":
            import random

            output = {"result": random.randint(1, 100)}
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case "start":
            # Normalement non exécuté (utilisé uniquement pour trouver les premiers nodes)
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case _:
            print_debug(f"Unknown node type: {node_type}", debug, tags=["NODE", "WARNING"])
            status, output, children_ids = True, {}, []

    return status, children_ids, output, images, texts


def node_edit(node: dict, input_ctx: dict, *, debug: bool = False) -> dict:
    """Edit node: applique config.fields sur un delta dict."""
    output: dict = {}
    node_config = get_node_config(node)

    fields = node_config.get("fields", [])
    fields_to_edit = [fields] if isinstance(fields, dict) else list(fields)

    if not fields_to_edit:
        print_debug("No fields to edit specified.", debug, tags=["NODE", "EDIT", "WARNING"])
        return output

    for field in fields_to_edit:
        key_path = field.get("key", "")
        if not key_path:
            print_debug("Field key path is missing.", debug, tags=["NODE", "EDIT", "WARNING"])
            continue

        raw_value = field.get("value", "")
        try:
            resolved = resolve_value(input_ctx, raw_value)
        except Exception as e:
            print_debug(f"Failed to resolve '{raw_value}' for '{key_path}': {e}", debug, tags=["NODE", "EDIT", "ERROR"])
            resolved = raw_value

        if get_value_from_path(input_ctx, key_path) == resolved:
            continue

        set_value_at_path(output, key_path, resolved)
        set_value_at_path(output, f"_traceback.{key_path}", {"raw": raw_value, "resolved": resolved})

        if field.get("user_approval", True):
            add_user_approval(
                output,
                message=f"Please approve the change to '{key_path}': {resolved}",
                key_path=key_path,
                debug=debug,
            )

    return output


def node_switch(node: dict, input_ctx: dict, *, debug: bool = False) -> tuple[bool, str]:
    node_config = get_node_config(node)
    key = node_config.get("key")
    output_key = "default"

    if key is None:
        print_debug("Switch key is missing.", debug, tags=["NODE", "SWITCH", "ERROR"])
        return False, output_key

    value = get_value_from_path(input_ctx, key)
    value = str(value) if value is not None else ""

    for case in node_config.get("cases", []) or []:
        case_value = resolve_value(input_ctx, case.get("value", ""))
        if str(case_value) == value:
            output_key = case.get("name", "default")
            break

    return True, output_key


def node_if(node: dict, input_ctx: dict, *, debug: bool = False) -> tuple[bool, str]:
    """If node: évalue des groupes de règles.

    Convention de sortie (outputs):
      outputs = {"true": [...], "false": [...]}

    Config attendu:
      conditions: [ { rules: [ {left, operator, right, link?}, ... ] }, ... ]

    Les groupes (conditions[i]) sont OR entre eux.
    Les règles d'un même groupe sont combinées via le champ `link` (AND/OR) du rule précédent.
    """

    def eval_rule(r: dict) -> bool:
        l_value = resolve_value(input_ctx, r.get("left", ""))
        r_value = resolve_value(input_ctx, r.get("right", ""))

        op_name = (r.get("operator") or "==").strip()
        l_str, r_str = normalize_value(l_value), normalize_value(r_value)

        try:
            l_num, r_num = float(l_str), float(r_str)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        match op_name:
            case "==":
                return (l_num == r_num) if is_numeric else (l_str == r_str)
            case "!=":
                return (l_num != r_num) if is_numeric else (l_str != r_str)
            case "<":
                return (l_num < r_num) if is_numeric else (l_str < r_str)
            case "<=":
                return (l_num <= r_num) if is_numeric else (l_str <= r_str)
            case ">":
                return (l_num > r_num) if is_numeric else (l_str > r_str)
            case ">=":
                return (l_num >= r_num) if is_numeric else (l_str >= r_str)
            case "contains":
                return r_str in l_str
            case "not_contains":
                return r_str not in l_str
            case _:
                print_debug(f"Unknown operator in rule: {op_name}", debug, tags=["NODE", "IF", "WARNING"])
                return False

    conditions = get_node_config(node).get("conditions", []) or []

    for cond in conditions:
        rules = cond.get("rules", []) or []
        if not rules:
            continue

        current = eval_rule(rules[0])
        for i in range(1, len(rules)):
            prev_rule = rules[i - 1]
            link = (prev_rule.get("link") or "AND").upper()
            nxt = eval_rule(rules[i])
            current = (current or nxt) if link == "OR" else (current and nxt)

        if current:
            return True, "true"

    return True, "false"


def node_http(node: dict, input_ctx: dict, *, debug: bool = False) -> tuple[bool, dict]:
    import requests

    node_config = get_node_config(node)

    url = resolve_value(input_ctx, node_config.get("url", ""))
    method = str(node_config.get("method", "GET")).upper()
    headers = node_config.get("headers", {})
    body = resolve_value(input_ctx, node_config.get("body", ""))

    try:
        response = requests.request(method, url, headers=headers, data=body)
        response.raise_for_status()
        output = {"status_code": response.status_code, "response_body": response.text}
        return True, output
    except Exception as e:
        print_debug(f"HTTP request failed: {e}", debug, tags=["NODE", "HTTP", "ERROR"])
        return False, {}


def node_sardine(
    node: dict,
    input_ctx: dict,
    base64_data: Any,
    *,
    debug: bool = False,
    images: Optional[list[Image.Image]] = None,
) -> tuple[bool, list[Image.Image], list[str]]:
    node_config = get_node_config(node)

    model_path = node_config.get("model_path", "")
    if not model_path:
        print_debug("No model path specified for Sardine node.", debug, tags=["NODE", "SARDINE", "ERROR"])
        return False, [], []

    accepted_files = node_config.get("accepted_files", [])
    if not accepted_files:
        print_debug("No accepted files specified for Sardine node.", debug, tags=["NODE", "SARDINE", "ERROR"])
        return False, [], []

    images = images or []
    if not images:
        page_mode: Literal["first_page_only", "all_pages"] = node_config.get("page_mode", "first_page_only")
        page_dpi: int = int(node_config.get("page_dpi", 200))
        images = extract_images_from_base64(base64_data, page_mode=page_mode, page_dpi=page_dpi, debug=debug)

    device = node_config.get("device", "cpu")

    classes = _predict_document_class(images, model_path=model_path, device=device, debug=debug)

    accepted_images: list[Image.Image] = []
    for i, cls in enumerate(classes):
        if cls in accepted_files:
            accepted_images.append(images[i])

    return True, accepted_images, { "document": { "details": { "classes": classes } } }


def node_detection(
    node: dict,
    input_ctx: dict,
    base64_data: Any,
    *,
    node_id: Optional[str] = None,
    debug: bool = False,
    images: Optional[list[Image.Image]] = None,
) -> tuple[bool, list[Image.Image], list[list[str]]]:
    node_config = get_node_config(node)

    model_path = node_config.get("model_path", "")
    if not model_path:
        print_debug("No model path specified for Detection node.", debug, tags=["NODE", "DETECTION", "ERROR"])
        return False, images or [], []

    images = images or []
    if not images:
        page_mode: Literal["first_page_only", "all_pages"] = node_config.get("page_mode", "first_page_only")
        page_dpi: int = int(node_config.get("page_dpi", 200))
        images = extract_images_from_base64(base64_data, page_mode=page_mode, page_dpi=page_dpi, debug=debug)

    zone_padding: int = int(node_config.get("zone_padding", 8))
    device = node_config.get("device", "cpu")
    conf: float = float(node_config.get("confidence_threshold", 0.25))

    try:
        detections, det_dur, ocr_dur = _predict_document_detection(
            images,
            model_path=model_path,
            device=device,
            conf=conf,
            debug=debug,
            padding=zone_padding,
        )

        set_value_at_path(input_ctx, f"_debug.nodes.{node_id}.ocr_duration", ocr_dur)
        set_value_at_path(input_ctx, f"_debug.nodes.{node_id}.detection_duration", det_dur)

        exclusion_types = ["logo", "signature"]
        detections_clean = [
            [
                _clean_detection_text(text.get("text")) 
                    for text in page_detections 
                    if text.get("type") not in exclusion_types
            ]
            for page_detections in detections
        ]

        return True, images, detections_clean
    except Exception as e:
        print_debug(f"Document detection failed: {e}", debug, tags=["NODE", "DETECTION", "ERROR"])
        return False, images, []


def node_agent(
    node: dict,
    texts: list[list[str]],
    *,
    debug: bool = False,
) -> tuple[bool, dict]:
    if not texts:
        print_debug("No texts provided to Agent node.", debug, tags=["NODE", "AGENT", "ERROR"])
        return False, {}
    
    node_config = get_node_config(node)

    model = node_config.get("model", None)
    if not model:
        print_debug("No model specified for Agent node.", debug, tags=["NODE", "AGENT", "ERROR"])
        return False, {}

    version = node_config.get("version", "latest")
    processing_mode = node_config.get("processing_mode", "per_zone") # "all_at_once" or "per_zone"

    if processing_mode == "all_at_once":
        texts = [[" \n ".join(page_texts)] for page_texts in texts]

    try:
        model, agent = get_cached_agent_model(model, version=version, debug=debug)
    except Exception as e:
        print_debug(f"Failed to load agent model: {e}", debug, tags=["NODE", "AGENT", "ERROR"])
        return False, {}

    entities_accumulated: list[list[Dict[str, Any]]] = []

    for p in texts:
        for text in p:
            status, agent_output = _predict_agent(
                text,
                model,
                agent,
                conf=float(node_config.get("confidence_threshold", 0.5)),
                debug=debug,
            )

            if not status or not agent_output:
                continue

            entities_accumulated.append(agent_output)

    transformed_mapper = agent.get("transformed_mapper")
    best_entities = best_entities_by_label(entities_accumulated, debug=debug)
    best_entities_list = list(best_entities.values())
    mapped_output = map_agent_output(best_entities_list, transformed_mapper, debug=debug)

    print_debug(f"Agent node completed successfully. {mapped_output}", debug, tags=["NODE", "AGENT", "INFO"])

    return True, mapped_output


# ========= Flow Utilities =========
def get_flow_details(result: dict) -> dict:
    total_node_duration = 0.0
    node_count = 0

    for _, v in result.get("_debug", {}).get("nodes", {}).items():
        duration = v.get("duration", 0.0)
        total_node_duration += duration
        node_count += 1

    average_node_duration = (total_node_duration / node_count) if node_count > 0 else 0.0

    return {
        "node_count": node_count,
        "total_node_duration": total_node_duration,
        "average_node_duration": average_node_duration,
    }


# ========= Flows Engine =========
def process_flow(flow: dict, base64_data: Any, *, debug: bool = False) -> dict:
    start_node_id = next((k for k, v in flow.items() if v.get("type") == "start"), None)
    if not start_node_id:
        print_debug("No start node found in the flow.", debug, tags=["FLOW"])
        raise ValueError("No start node found in the flow.")

    start_node = get_node_by_id(flow, start_node_id)
    if not start_node:
        raise ValueError("Start node id not found in flow dict.")

    start_children = get_node_children(start_node)
    if not isinstance(start_children, list):
        raise ValueError("Start node outputs must be a list of node ids.")

    # Contexte initial
    result: dict = dict(base64_data) if isinstance(base64_data, dict) else {}

    images: list[Image.Image] = []
    texts: list[list[str]] = []

    nodes_completed: set[str] = {start_node_id}
    nodes_queued: set[str] = set(start_children)
    nodes_running: set[str] = set()

    future_to_node: dict[concurrent.futures.Future, str] = {}

    with ThreadPoolExecutor(max_workers=FLOW_MAX_WORKERS) as executor:
        while nodes_queued or nodes_running:
            for node_id in list(nodes_queued):
                node = get_node_by_id(flow, node_id)
                if node:
                    future = executor.submit(
                        process_node,
                        node,
                        result,
                        base64_data,
                        node_id=node_id,
                        debug=debug,
                        images=images,
                        texts=texts,
                    )
                    future_to_node[future] = node_id
                    nodes_running.add(node_id)
                nodes_queued.remove(node_id)

            if not nodes_running:
                break

            done, _ = concurrent.futures.wait(
                list(future_to_node.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                node_id = future_to_node.pop(future)
                nodes_running.remove(node_id)

                try:
                    status, children_ids, output, images, texts = future.result()

                    if status:
                        nodes_completed.add(node_id)

                        if output:
                            result = merge_dicts(result, output)

                        for child_id in children_ids:
                            if child_id in nodes_completed or child_id in nodes_running or child_id in nodes_queued:
                                continue

                            child = get_node_by_id(flow, child_id)
                            if not child:
                                continue

                            parents = get_node_parents(child)
                            if all(p in nodes_completed for p in parents):
                                nodes_queued.add(child_id)

                    else:
                        print_debug(f"Node {node_id} failed.", debug, tags=["FLOW", "ERROR"])

                except Exception as e:
                    print_debug(f"Exception in node {node_id}: {e}", debug, tags=["FLOW", "CRITICAL"])

    # Flow debug details
    set_value_at_path(result, "_debug.flow", get_flow_details(result))
    print_debug("Flow processing completed.", debug, tags=["FLOW"])

    return result


def run(flow: dict, base64_data: Any, *, debug: bool = False) -> dict:
    start = time.time()
    print_debug("Flow started", debug)

    result = process_flow(flow, base64_data, debug=debug)

    end = time.time()
    total_duration = end - start
    print_debug(f"Flow ended in {total_duration:.2f} seconds", debug)

    details = {
        "started_at": datetime.fromtimestamp(start).isoformat(),
        "ended_at": datetime.fromtimestamp(end).isoformat(),
        "duration": total_duration,
    }
    add_value_at_path(result, "_debug.flow", details)

    return result


if __name__ == "__main__":
    print(f"[INFO] Sardine Flow Engine Module Test; Time to import: {end_import_time - start_import_time:.2f}s")

    test_flow = {
        "start": {"id": "start", "type": "start", "outputs": ["node_A", "node_B"]},

        # debug (random)
        "node_A": {
            "id": "node_A",
            "type": "debug",
            "config": {"root": "node_A"},
            "inputs": ["start"],
            "outputs": ["node_C"],
        },
        "node_B": {
            "id": "node_B",
            "type": "debug",
            "config": {"root": "node_B"},
            "inputs": ["start"],
            "outputs": ["node_C"],
        },

        # edit (arithmétique + slicing)
        "node_C": {
            "id": "node_C",
            "type": "edit",
            "config": {
                "fields": [
                    {"key": "node_C.value", "value": "${node_A.result} + ${node_B.result}", "user_approval": True},
                    {"key": "seller.siren", "value": "${seller.vat.number}[-9:]"},
                ]
            },
            "inputs": ["node_A", "node_B"],
            "outputs": ["node_D"],
        },

        # switch (outputs DOIT être un dict)
        "node_D": {
            "id": "node_D",
            "type": "switch",
            "config": {
                "key": "seller.siren",
                "cases": [
                    {"name": "match", "value": "456789000"},
                ],
            },
            "inputs": ["node_C"],
            "outputs": {
                "match": ["node_MATCH"],
                "default": ["node_DEFAULT"],
            },
        },

        # debug sur chaque branche
        "node_MATCH": {
            "id": "node_MATCH",
            "type": "debug",
            "config": {"root": "branch.match"},
            "inputs": ["node_D"],
            "outputs": ["node_END_MATCH"],
        },
        "node_DEFAULT": {
            "id": "node_DEFAULT",
            "type": "debug",
            "config": {"root": "branch.default"},
            "inputs": ["node_D"],
            "outputs": ["node_END_DEFAULT"],
        },

        # final (un final par branche, sinon le join bloque)
        "node_END_MATCH": {"id": "node_END_MATCH", "type": "final", "inputs": ["node_MATCH"], "outputs": []},
        "node_END_DEFAULT": {"id": "node_END_DEFAULT", "type": "final", "inputs": ["node_DEFAULT"], "outputs": []},
    }

    # === Sardine -> Detection gating flow template ===
    dpi = 768
    sardine_detection_flow = {
        "start": {"id": "start", "type": "start", "outputs": ["node_SARDINE"]},

        "node_SARDINE": {
            "id": "node_SARDINE",
            "type": "sardine",
            "inputs": ["start"],
            "outputs": {
                "valid": ["node_DETECTION"],
                "invalid": ["node_END"],
            },
            "config": {
                "model_path": "C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\sardine.agents\\sard-cls\\best.pt",
                "accepted_files": ["facture"],
                "stop_if_no_match": True,
                "page_mode": "first_page_only",
                "page_dpi": 512,
                "device": "cpu",
            },
        },

        "node_DETECTION": {
            "id": "node_DETECTION",
            "type": "zone-detection",
            "inputs": ["node_SARDINE"],
            "outputs": ["node_SIREN", "node_ADDRESS", "node_VAT"],
            "config": {
                "model_path": "C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\sardine.agents\\sard-det\\best.pt",
                "confidence_threshold": 0.75,
                "zone_padding": 8,
                "page_mode": "first_page_only",
                "page_dpi": 512,
                "device": "cpu",
            },
        },

        "node_SIREN": {
            "id": "node_SIREN",
            "type": "agent",
            "inputs": ["node_DETECTION"],
            "outputs": ["node_END"],
            "config": {
                "model": "siren",
                "version": "latest",
                "processing_mode": "per_zone",
            },
        },

        "node_ADDRESS": {
            "id": "node_ADDRESS",
            "type": "agent",
            "inputs": ["node_DETECTION"],
            "outputs": ["node_END"],
            "config": {
                "model": "address",
                "version": "latest",
                "processing_mode": "per_zone",
            },
        },

        "node_VAT": {
            "id": "node_VAT",
            "type": "agent",
            "inputs": ["node_DETECTION"],
            "outputs": ["node_END"],
            "config": {
                "model": "vat-number",
                "version": "latest",
                "processing_mode": "per_zone",
            },
        },

        "node_END": {"id": "node_END", "type": "final", "inputs": ["node_SARDINE", "node_SIREN", "node_ADDRESS", "node_VAT"], "outputs": []},
    }
    print("=== Sardine -> Detection flow template is available in variable: sardine_detection_flow ===")

    files_to_test_ok = [
        # "facture_001-1",
        # "facture_002-1",
        # "facture_003-2",
        "FACTDVX_01_TERRESDUSUD",
        "9472_Facture_TransportsCombemale_001",
        # "8787_Facture_SocieteNouvelleDeMateriaux_001",
        # "2024071184",
        # "F2024-09-30",
        # "DIPRES_20241004",
        # "FACT33481-1"
    ]

    files_to_test_nok = [
        "bulletin_de_paie",
        "5e7f689f-051c-41ca-b067-bb2a387cdf8f"
    ]

    BASE_DIR_OK = r"C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\invoices"
    BASE_DIR_NOK = r"C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\invoices"

    def _get_flow_duration(res: dict) -> float:
        flow_dbg = res.get("_debug", {}).get("flow", {})
        return float(flow_dbg.get("total_node_duration", flow_dbg.get("duration", 0.0)) or 0.0)

    def _get_node_duration(res: dict, node_id: str) -> float | None:
        nodes_dbg = res.get("_debug", {}).get("nodes", {})
        d = nodes_dbg.get(node_id, {}).get("duration", None)
        ocr_d = nodes_dbg.get(node_id, {}).get("ocr_duration", None)
        det_d = nodes_dbg.get(node_id, {}).get("detection_duration", None)
        return float(d) if d is not None else None, float(ocr_d) if ocr_d is not None else None, float(det_d) if det_d is not None else None

    # --- stats cumulées
    stats = {
        "node_count": len(files_to_test_ok) + len(files_to_test_nok),
        "node_count_ok": len(files_to_test_ok),
        "node_count_nok": len(files_to_test_nok),

        "total_node_duration": 0.0,
        "total_node_duration_ok": 0.0,
        "total_node_duration_nok": 0.0,

        "average_node_duration": 0.0,
        "average_node_duration_ok": 0.0,
        "average_node_duration_nok": 0.0,
    }

    node_stats = {
        "node_SARDINE": {"sum": 0.0, "count": 0, "avg": 0.0},
        "node_DETECTION": {"sum": 0.0, "count": 0, "avg": 0.0},
    }

    node_substats = {
        "node_DETECTION_OCR": {"sum": 0.0, "count": 0, "avg": 0.0},
        "node_DETECTION_DET": {"sum": 0.0, "count": 0, "avg": 0.0},
    }

    # --- historiques pour graphes (x = numéro du test)
    x_all, y_all, y_all_avg = [], [], []
    x_ok, y_ok, y_ok_avg = [], [], []
    x_nok, y_nok, y_nok_avg = [], [], []

    history = []  # optionnel : log structuré par test

    test_idx = 0
    ok_seen = 0
    nok_seen = 0

    def _update_avgs():
        if test_idx > 0:
            stats["average_node_duration"] = stats["total_node_duration"] / test_idx
        if ok_seen > 0:
            stats["average_node_duration_ok"] = stats["total_node_duration_ok"] / ok_seen
        if nok_seen > 0:
            stats["average_node_duration_nok"] = stats["total_node_duration_nok"] / nok_seen

    def _run_one(file_name: str, label: str, base_dir: str):
        global test_idx, ok_seen, nok_seen

        print(f"\n--- Testing {label.upper()} file: {file_name} ---")
        with open(fr"{base_dir}\{file_name}.txt", "r", encoding="utf-8") as f:
            file_content = f.read()

        res = run(sardine_detection_flow, file_content, debug=True)

        flow_dur = _get_flow_duration(res)
        sard_dur, _, _ = _get_node_duration(res, "node_SARDINE")
        det_g_dur, ocr_dur, det_dur = _get_node_duration(res, "node_DETECTION")

        # --- totals global
        test_idx += 1
        stats["total_node_duration"] += flow_dur

        if label == "ok":
            ok_seen += 1
            stats["total_node_duration_ok"] += flow_dur
        else:
            nok_seen += 1
            stats["total_node_duration_nok"] += flow_dur

        _update_avgs()

        # --- séries globales pour graphes
        x_all.append(test_idx)
        y_all.append(flow_dur)
        y_all_avg.append(stats["average_node_duration"])

        if label == "ok":
            x_ok.append(ok_seen)
            y_ok.append(flow_dur)
            y_ok_avg.append(stats["average_node_duration_ok"])
        else:
            x_nok.append(nok_seen)
            y_nok.append(flow_dur)
            y_nok_avg.append(stats["average_node_duration_nok"])

        # --- node averages cumulées (on n’incrémente que si la node a tourné)
        def upd_node(node_id: str, dur: float | None):
            if dur is None:
                return None
            node_stats[node_id]["sum"] += dur
            node_stats[node_id]["count"] += 1
            node_stats[node_id]["avg"] = node_stats[node_id]["sum"] / node_stats[node_id]["count"]
            return node_stats[node_id]["avg"]
        
        def upd_sub(node_key: str, dur: float | None):
            if dur is None:
                return None
            node_substats[node_key]["sum"] += dur
            node_substats[node_key]["count"] += 1
            node_substats[node_key]["avg"] = node_substats[node_key]["sum"] / node_substats[node_key]["count"]
            return node_substats[node_key]["avg"]

        sard_avg = upd_node("node_SARDINE", sard_dur)
        det_avg  = upd_node("node_DETECTION", det_g_dur)
        det_ocr_avg = upd_sub("node_DETECTION_OCR", ocr_dur)
        det_det_avg = upd_sub("node_DETECTION_DET", det_dur)

        history.append({
            "test_idx": test_idx,
            "file": file_name,
            "label": label,

            "flow_duration": flow_dur,
            "flow_avg_total": stats["average_node_duration"],
            "flow_avg_ok": stats["average_node_duration_ok"],
            "flow_avg_nok": stats["average_node_duration_nok"],

            "node_SARDINE_duration": sard_dur,
            "node_SARDINE_avg": sard_avg,        # moyenne cumulée sur les runs où la node existe
            "node_DETECTION_duration": det_g_dur,
            "node_DETECTION_avg": det_avg,       # idem (souvent absent sur NOK)

            "tot_total": stats["total_node_duration"],
            "tot_ok": stats["total_node_duration_ok"],
            "tot_nok": stats["total_node_duration_nok"],

            "node_DETECTION_ocr_duration": ocr_dur,
            "node_DETECTION_ocr_avg": det_ocr_avg,

            "node_DETECTION_det_duration": det_dur,
            "node_DETECTION_det_avg": det_det_avg,
        })

        return res

    # --- exécution OK puis NOK (ou mélange si tu préfères)
    for fn in files_to_test_ok:
        print(json.dumps(_run_one(fn, "ok", BASE_DIR_OK), indent=2))

    for fn in files_to_test_nok:
        _run_one(fn, "nok", BASE_DIR_NOK)

    print("\n=== Final stats ===")
    print(json.dumps(stats, indent=2))
    print("\n=== Node stats ===")
    print(json.dumps(merge_dicts(node_stats, node_substats), indent=2))

    import matplotlib.pyplot as plt

    def plot_bench_with_nodes(history, out_dir="."):
        ok_hist = [h for h in history if h.get("label") == "ok"]

        # Helper: 1 figure = 2 courbes (durée + moyenne cumulée)
        def plot_two_curves(x1, y1, x2, y2, title, xlabel, ylabel, filename,
                            label1="Durée", label2="Moyenne cumulée"):
            plt.figure()
            if x1 and y1:
                plt.plot(x1, y1, marker="o", label=label1)
            if x2 and y2:
                plt.plot(x2, y2, marker="o", label=label2)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{filename}", dpi=160)
            plt.close()

        # ---------------- Flow (OK only)
        x_ok_idx    = list(range(1, len(ok_hist) + 1))  # 1..N_OK
        flow_y_ok   = [h["flow_duration"] for h in ok_hist]
        flow_avg_ok = [h["flow_avg_ok"] for h in ok_hist]

        plot_two_curves(
            x_ok_idx, flow_y_ok,
            x_ok_idx, flow_avg_ok,
            title="Flow (OK only) - Durée & moyenne cumulée",
            xlabel="Test OK #",
            ylabel="Temps (s)",
            filename="flow_duration_and_cumavg_OK.png",
            label1="Durée par test",
            label2="Moyenne cumulée"
        )

        # ---------------- node_SARDINE (OK only quand dispo)
        sard_x = [h["test_idx"] for h in ok_hist if h.get("node_SARDINE_duration") is not None]
        sard_y = [h["node_SARDINE_duration"] for h in ok_hist if h.get("node_SARDINE_duration") is not None]
        sard_avg_x = [h["test_idx"] for h in ok_hist if h.get("node_SARDINE_avg") is not None]
        sard_avg_y = [h["node_SARDINE_avg"] for h in ok_hist if h.get("node_SARDINE_avg") is not None]

        plot_two_curves(
            sard_x, sard_y,
            sard_avg_x, sard_avg_y,
            title="node_SARDINE - Durée & moyenne cumulée (OK only)",
            xlabel="Test #",
            ylabel="Temps (s)",
            filename="node_sardine_duration_and_cumavg_OK.png",
            label1="Durée",
            label2="Moyenne cumulée"
        )

        # ---------------- node_DETECTION (runs où exécuté)
        det_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_duration") is not None]
        det_y = [h["node_DETECTION_duration"] for h in ok_hist if h.get("node_DETECTION_duration") is not None]
        det_avg_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_avg") is not None]
        det_avg_y = [h["node_DETECTION_avg"] for h in ok_hist if h.get("node_DETECTION_avg") is not None]

        plot_two_curves(
            det_x, det_y,
            det_avg_x, det_avg_y,
            title="node_DETECTION - Durée & moyenne cumulée (runs exécutés)",
            xlabel="Test #",
            ylabel="Temps (s)",
            filename="node_detection_duration_and_cumavg.png",
            label1="Durée",
            label2="Moyenne cumulée"
        )

        # ---------------- DETECTION / OCR (OK only)
        ocr_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_ocr_duration") is not None]
        ocr_y = [h["node_DETECTION_ocr_duration"] for h in ok_hist if h.get("node_DETECTION_ocr_duration") is not None]
        ocr_avg_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_ocr_avg") is not None]
        ocr_avg_y = [h["node_DETECTION_ocr_avg"] for h in ok_hist if h.get("node_DETECTION_ocr_avg") is not None]

        plot_two_curves(
            ocr_x, ocr_y,
            ocr_avg_x, ocr_avg_y,
            title="node_DETECTION/OCR - Durée & moyenne cumulée (OK only)",
            xlabel="Test #",
            ylabel="Temps (s)",
            filename="node_detection_ocr_duration_and_cumavg_OK.png",
            label1="OCR durée",
            label2="OCR moyenne cumulée"
        )

        # ---------------- DETECTION / DET (OK only)
        det2_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_det_duration") is not None]
        det2_y = [h["node_DETECTION_det_duration"] for h in ok_hist if h.get("node_DETECTION_det_duration") is not None]
        det2_avg_x = [h["test_idx"] for h in ok_hist if h.get("node_DETECTION_det_avg") is not None]
        det2_avg_y = [h["node_DETECTION_det_avg"] for h in ok_hist if h.get("node_DETECTION_det_avg") is not None]

        plot_two_curves(
            det2_x, det2_y,
            det2_avg_x, det2_avg_y,
            title="node_DETECTION/DET - Durée & moyenne cumulée (OK only)",
            xlabel="Test #",
            ylabel="Temps (s)",
            filename="node_detection_det_duration_and_cumavg_OK.png",
            label1="DET durée",
            label2="DET moyenne cumulée"
        )

        plt.show()

    # Utilisation :
    plot_bench_with_nodes(history, out_dir=".")
    print("Graphs saved: duration_per_test.png, cumulative_average.png, ok_vs_nok.png, sorted_durations.png")