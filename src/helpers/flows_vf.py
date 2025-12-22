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
from typing import Any, Callable, Literal, Optional, Union

from PIL import Image, ImageFilter, ImageOps

# Optional / heavy deps (le flow de démo ne les utilise pas)
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


# ========= CONSTANTS =========
FLOW_MAX_WORKERS = 4
OCR_MAX_WORKERS = 4

_YOLO_CACHE: dict[str, Any] = {}
_YOLO_LOCK = threading.Lock()


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
    with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as ex:
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
) -> list[list[str]]:
    _require(np, "numpy")

    model_det = get_cached_yolo_model(model_path, debug=debug)

    try:
        det_results = model_det.predict(source=images, save=False, conf=conf, device=device, verbose=False)
    except Exception as e:
        print_debug(f"[SARDINE] Document detection failed: {e}", debug, tags=["SARDINE", "DETECT"])
        return []

    output: list[list[str]] = []

    for img, res in zip(images, det_results):
        page_output: list[str] = []
        W, H = img.size

        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy.numpy()
            xyxy = xyxy.astype(int)
            order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))  # type: ignore[union-attr]
            xyxy = xyxy[order]

            crops: list[Image.Image] = []
            for (x1, y1, x2, y2) in xyxy:
                x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
                x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
                if x2p > x1p and y2p > y1p:
                    crops.append(img.crop((x1p, y1p, x2p, y2p)))

            if crops:
                texts = _ocr_many_pil(crops, debug=debug)
                page_output = [t.strip() for t in texts if t is not None]

        if not page_output:
            full_text = _ocr_pil_image(img, debug=debug)
            if full_text:
                page_output.append(full_text)

        output.append(page_output)

    return output


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
    )

    # Root-wrapping (optionnel)
    wrapped_output = output if not node_root else {node_root: output}

    time.sleep(0.1)

    # Debug timings (dans le contexte global)
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
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

        case "zone-detection":
            status, images, detections = node_detection(node, input_ctx, base64_data, debug=debug, images=images)
            texts = detections
            children_raw = get_node_children(node)
            children_ids = list(children_raw) if isinstance(children_raw, list) else []

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
        detections = _predict_document_detection(
            images,
            model_path=model_path,
            device=device,
            conf=conf,
            debug=debug,
            padding=zone_padding,
        )
        return True, images, detections
    except Exception as e:
        print_debug(f"Document detection failed: {e}", debug, tags=["NODE", "DETECTION", "ERROR"])
        return False, images, []


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

    # # 1) Force la branche "match" (vat.number -> siren = 456789000)
    # ctx_match = {"seller": {"vat": {"number": "FR000456789000"}}}
    # print("=== MATCH ===")
    # print(json.dumps(run(test_flow, ctx_match, debug=True), indent=2))

    # # 2) Force la branche "default" (vat.number -> siren != 456789000)
    # ctx_default = {"seller": {"vat": {"number": "FR000123456789"}}}
    # print("=== DEFAULT ===")
    # print(json.dumps(run(test_flow, ctx_default, debug=True), indent=2))

    # === Sardine -> Detection gating flow template ===
    sardine_detection_flow = {
        "start": {"id": "start", "type": "start", "outputs": ["node_SARDINE"]},

        # 1) Sardine: classify pages; if invoice => continue; else stop.
        "node_SARDINE": {
            "id": "node_SARDINE",
            "type": "sardine",
            "inputs": ["start"],
            "outputs": ["node_DETECTION"],
            "config": {
                "model_path": "C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\sardine.agents\\sard-cls\\best.pt",
                # Classes that should be treated as "invoice"
                "accepted_files": ["invoice"],
                "stop_if_no_match": True,
                "page_mode": "first_page_only",
                "page_dpi": 512,
                "device": "cpu",
            },
        },

        # 2) Detection: runs only if Sardine decided to continue
        "node_DETECTION": {
            "id": "node_DETECTION",
            "type": "zone-detection",
            "inputs": ["node_SARDINE"],
            "outputs": ["node_END"],
            "config": {
                "model_path": "C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\sardine.agents\\sard-det\\best.pt",
                "confidence_threshold": 0.25,
                "zone_padding": 8,
                "page_mode": "first_page_only",
                "page_dpi": 512,
                "device": "cpu",
            },
        },

        "node_END": {"id": "node_END", "type": "final", "inputs": ["node_DETECTION"], "outputs": []},
    }
    print("=== Sardine -> Detection flow template is available in variable: sardine_detection_flow ===")
    with open("C:\\Users\\Utilisateur\\Documents\\workspace.sardine.v2\\invoices\\facture_002-1.txt", "r") as file:
        file_content = file.read()
    print(json.dumps(run(sardine_detection_flow, file_content, debug=True), indent=2))
    print(json.dumps(run(sardine_detection_flow, file_content, debug=True), indent=2))