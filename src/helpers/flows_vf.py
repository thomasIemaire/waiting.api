import base64
import time
import json
import re
import ast
from uuid import uuid4
from datetime import datetime
import operator as op
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, List, Literal, Optional, Union
import numpy as np
from scipy import io

# ========= CONSTANTS =========
FLOW_MAX_WORKERS = 4


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

_DATAURL_RE = re.compile(r'data:(?P<mime>[\w/-]+)?(;base64)?,(?P<data>.+)', re.IGNORECASE)


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


def _to_python_expr(expr: str) -> str:
    """
    Convert a DSL expression into a restricted python expression:

      ${a.b}    -> $a.b
      $a.b      -> get("a.b")

    Slicing/indexing stays intact, e.g.:
      $seller.vat.number[-9:] -> get("seller.vat.number")[-9:]
    """
    expr = _BRACED_PATH_RE.sub(lambda m: f"${m.group(1)}", expr)
    expr = _DOLLAR_PATH_RE.sub(lambda m: f'get("{m.group(1)}")', expr)
    return expr


def _safe_eval(expr: str, funcs: dict[str, Callable[..., Any]]) -> Any:
    """
    Evaluate a python expression with a restricted AST.
    Allowed:
      - constants (int/float/str/bool/None)
      - + - * / // % **, unary +/-
      - indexing and slicing
      - calls to whitelisted funcs only (by name)
    """
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


def resolve_value(data: dict, value: Any) -> Any:
    """
    Resolve a DSL expression string using data:
      - supports ${path} and $path
      - supports arithmetic, string operations, indexing/slicing, and limited builtins
    """
    if not isinstance(value, str):
        return value

    expr = value.strip()
    if "$" not in expr:
        return value

    def get(path: str) -> Any:
        return get_value_from_path(data, path)

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


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    return str(value)


# ========= Sardine Utilities =========
from PIL import Image

def extract_images_from_base64(
        base64: Any,
        *,
        debug: bool = False,
        page_dpi: int = 200,
        page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
) -> List[Image.Image]:
    images = load_images_from_base64(
        base64,
        page_mode=page_mode,
        page_dpi=page_dpi,
        debug=debug
    )

    if not images:
        print_debug("[SARDINE] No images extracted from base64 data.", debug, tags=["SARDINE", "EXTRACT_IMAGES"])
        return []
    
    return images


def load_images_from_base64(
        base64: Any,
        page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
        page_dpi: int = 200,
        *,
        debug: bool = False
) -> List[Image.Image]:
    raw_bytes, mime = _try_decode_base64(base64)

    if raw_bytes is None:
        print_debug("[SARDINE] No valid base64 data found.", debug, tags=["SARDINE", "LOAD_IMAGES"])
        return []

    try:
        is_pdf = (mime == "application/pdf") or (mime is None and raw_bytes[:4] == b"%PDF")
        return _load_pil_from_pdf(raw_bytes, page_mode=page_mode, dpi=page_dpi) if is_pdf else [_load_pil_from_bytes(raw_bytes)]
    except Exception as e:
        print_debug(f"[SARDINE] Error loading images: {e}", debug, tags=["SARDINE", "LOAD_IMAGES"])
        return []


def _load_pil_from_pdf(
        raw_bytes: bytes,
        page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
        dpi: int = 200,
        *,
        debug: bool = False
) -> List[Image.Image]:
    try:
        import fitz
    except ImportError:
        print_debug("[SARDINE] fitz (PyMuPDF) library is not installed.", debug, tags=["SARDINE", "LOAD_PDF"])
        return []
    
    images: List[Image.Image] = []
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


def _try_decode_base64(s: str) -> tuple[Optional[bytes], Optional[str]]:
    if not isinstance(s, str) or not s.strip(): return None, None
    m = _DATAURL_RE.match(s.strip())
    if m:
        try:
            return base64.b64decode(m.group("data"), validate=True), (m.group("mime") or "").lower()
        except: 
            return None, None
    try:
        return base64.b64decode(s), None
    except: 
        return None, None


def _predict_document_class(
        images: List[Image.Image],
        model_path: str,
        device: str = "cpu",
        *,
        debug: bool = False
) -> List[str]:
    try:
        return classify_pages(
            get_cached_yolo_model(model_path),
            images,
            device=device,
            debug=debug
        )
    except Exception as e:
        print_debug(f"[SARDINE] Document classification failed: {e}", debug, tags=["SARDINE", "CLASSIFY"])
        return []


def classify_pages(
        model_cls: YOLO,
        sources: List[Image.Image],
        device: str = "cpu",
        *,
        debug: bool = False
) -> List[str]:
    cls = model_cls.predict(source=sources, device=device, save=False, verbose=False)
    results = []
    for r in cls:
        if hasattr(r, 'probs') and r.probs is not None:
            top1 = r.probs.top1
            name = r.names[top1]
        else:
            print_debug("[SARDINE] No classification probabilities found.", debug, tags=["SARDINE", "CLASSIFY"])
            name = "unknown"
        results.append(name)
    return results
    

def _predict_document_detection(
        images: List[Image.Image],
        model_path: str,
        device: str = "cpu",
        conf: float = 0.25,
        *,
        debug: bool = False,
        padding: int = 8,
) -> List[dict]:
    model_det = get_cached_yolo_model(model_path)

    try:
        det_results = model_det.predict(
            source=images,
            save=False,
            conf=conf,
            device=device,
            verbose=False
        )
    except Exception as e:
        print_debug(f"[SARDINE] Document detection failed: {e}", debug, tags=["SARDINE", "DETECT"])
        return []

    output: List[List[str]] = []

    for img, res in zip(images, det_results):
        page_output = []
        W, H = img.size

        if res.boxes and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy.numpy()
            xyxy = xyxy.astype(int)
            order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))
            xyxy = xyxy[order]

            crops = []
            for (x1, y1, x2, y2) in xyxy:
                x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
                x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
                if x2p > x1p and y2p > y1p:
                    crops.append(img.crop((x1p, y1p, x2p, y2p)))
            
            if crops:
                texts = _ocr_many_pil(crops)
                page_output = [t.strip() for t in texts]

        if not page_output:
            full_text = _ocr_pil_image(img)
            if full_text:
                page_output.append(full_text)
        
        output.append(page_output)
    
    return output


# ========= Node Utilities =========
def get_node_by_id(flow: dict, node_id: str) -> Optional[dict]:
    return flow.get(node_id, None)


def get_node_children(node: dict) -> list:
    return node.get("outputs", [])


def get_node_parents(node: dict) -> list:
    return node.get("inputs", [])


def get_node_config(node: dict) -> dict:
    return node.get("config", {})


def add_user_approval(
        input: dict,
        message: str,
        *,
        key_path: str | None = None,
        debug: bool = False,
) -> None:
    print_debug(f"User approval requested: {message}", debug, tags=["NODE", "APPROVAL"])
    approval_id = uuid4().hex if not key_path else key_path.replace(".", "_")
    set_value_at_path(input, f"user_approvals.{approval_id}", {
        "message": message,
        "approved": False,
    })


# ========= Nodes Processing =========
def process_node(
    node: dict,
    input: Optional[dict] = None,
    *,
    node_id: Optional[str] = None,
    debug: bool = False,
) -> tuple[bool, dict]:
    start = time.time()
    print_debug(f"START Processing {node_id} (Type: {node.get('type')})", debug, tags=["NODE", "INFO"])

    node_config = node.get("config", {})
    node_root = node_config.get("root", None)

    status, output, children = execute_node_by_type(node, input, debug=debug)

    time.sleep(0.1)

    end = time.time()
    print_debug(f"END Processing {node_id} (Duration: {end - start:.2f}s)", debug, tags=["NODE", "INFO"])
    set_value_at_path(input, f"_debug.nodes.{node_id}.duration", end - start)

    return status, output if not node_root else {node_root: output}, children


def execute_node_by_type(
    node: dict,
    input: Optional[dict] = None,
    *,
    debug: bool = False,
) -> tuple[bool, dict]:
    node_type = node.get("type", None)
    if not node_type:
        print_debug("Node type is missing.", debug, tags=["NODE", "ERROR"])
        raise ValueError("Node type is missing")
    
    status = True
    output: dict = {}
    children: list | None = None

    match node_type:
        case "edit":
            status, output = True, node_edit(node, input, debug=debug)
        case "switch":
            status, output_key = node_switch(node, input, debug=debug)
            children = get_node_children(node)[output_key]
            output = {}
        case "if":
            status, output_key = node_if(node, input, debug=debug)
            children = get_node_children(node)[output_key]
            output = {}
        case "http":
            status, output = node_http(node, input, debug=debug)
        case "final":
            status, output, children = True, input or {}, []

        case "debug":
            import random
            status, output =  True, {"result": random.randint(1, 100)}
        case _:
            print_debug(f"Unknown node type: {node_type}", debug, tags=["NODE", "WARNING"])
            status, output = True, {}
    
    return status, output, get_node_children(node) if children is None else children


def node_edit(
    node: dict,
    input: Optional[dict] = None,
    *,
    debug: bool = False,
) -> dict:
    """
    Edit node: applies config.fields onto a copy of input context.
    fields can be:
      - a list of {"key": "...", "value": ...}
      - a single dict {"key": "...", "value": ...}
    Values are resolved via resolve_value() against the current context.
    """
    output = {}
    node_config = get_node_config(node)

    fields = node_config.get("fields", [])
    if isinstance(fields, dict):
        fields_to_edit = [fields]
    else:
        fields_to_edit = fields

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
            resolved = resolve_value(input, raw_value)
        except Exception as e:
            print_debug(f"Failed to resolve '{raw_value}' for '{key_path}': {e}", debug, tags=["NODE", "EDIT", "ERROR"])
            resolved = raw_value

        if get_value_from_path(input, key_path) == resolved:
            continue

        set_value_at_path(output, key_path, resolved)

        set_value_at_path(output, f"_traceback.{key_path}", {
            "raw": raw_value, "resolved": resolved
        })

        if field.get("user_approval", True):
            add_user_approval(
                output,
                message=f"Please approve the change to '{key_path}': {resolved}",
                key_path=key_path,
                debug=debug
            )

    return output


def node_switch(
        node: dict,
        input: Optional[dict] = None,
        *,
        debug: bool = False,
) -> tuple[bool, str]:
    """
    Switch node: routes to one of several branches based on conditions.
    Config should have:
      - "cases": list of {"condition": "...", "output": "node_id"}
      - "default": "node_id" (optional)
    Conditions are evaluated against input context.
    """
    node_config = get_node_config(node)
    key = node_config.get("key", None)
    output_key = "default"

    if key is None:
        print_debug("Switch key is missing.", debug, tags=["NODE", "SWITCH", "ERROR"])
        return False, output_key
    
    value = get_value_from_path(input, key)
    value = str(value) if value is not None else ""

    cases = node_config.get("cases", [])
    for case in cases:
        case_value = resolve_value(input, case.get("value", ""))
        if str(case_value) == value:
            output_key = case.get("name", "default")
            break
    
    return True, output_key


def node_if(
        node: dict,
        input: Optional[dict] = None,
        *,
        debug: bool = False,
) -> tuple[bool, str]:
    """
    If node: evaluates a condition to determine which branches to follow.
    Config should have:
      - "condition": "..." (expression)
      - "true_branch": list of node_ids
      - "false_branch": list of node_ids
    Condition is evaluated against input context.
    """
    def eval_rule(r):
        l_value, r_value = resolve_value(input, r.get("left", "")), resolve_value(input, r.get("right", ""))

        op = r.get("operator", "==")
        l_str, r_str = normalize_value(l_value), normalize_value(r_value)

        try:
            l_num, r_num = float(l_str), float(r_str)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        match op:
            case "==":
                return l_str == r_str if not is_numeric else l_num == r_num
            case "!=":
                return l_str != r_str if not is_numeric else l_num != r_num
            case "<":
                return l_str < r_str if not is_numeric else l_num < r_num
            case "<=":
                return l_str <= r_str if not is_numeric else l_num <= r_num
            case ">":
                return l_str > r_str if not is_numeric else l_num > r_num
            case ">=":
                return l_str >= r_str if not is_numeric else l_num >= r_num
            case "contains":
                return r_str in l_str
            case "not_contains":
                return r_str not in l_str
            case _:
                print_debug(f"Unknown operator in rule: {op}", debug, tags=["NODE", "IF", "WARNING"])
                return False
    
    node_config = get_node_config(node)
    conditions = node_config.get("conditions", "")
    matched_index = -1

    for i, cond in enumerate(conditions):
        rules = cond.get("rules", [])
        
        if not rules:
            continue

        current_result = eval_rule(rules[0])

        for r in range(1, len(rules)):
            l_rule = rules[r - 1]
            r_rule = rules[r]
            logical_op = l_rule.get("link", "AND").upper()

            next_result = eval_rule(r_rule)

            if logical_op == "OR":
                current_result = current_result or next_result
            else:
                current_result = current_result and next_result

        if current_result:
            matched_index = i
            break
    
    node_children = get_node_children(node)
    output_key = node_children[matched_index].get("name", "false") if matched_index >= 0 else "false"

    return True, output_key


def node_http(
        node: dict,
        input: Optional[dict] = None,
        *,
        debug: bool = False,
) -> tuple[bool, dict]:
    """
    HTTP node: makes an HTTP request based on config and input context.
    Config should have:
      - "url": "..."
      - "method": "GET" | "POST" | ...
      - "headers": dict
      - "body": "..." (for POST/PUT)
    """
    import requests

    node_config = get_node_config(node)

    url = resolve_value(input, node_config.get("url", ""))
    method = node_config.get("method", "GET").upper()
    headers = node_config.get("headers", {})
    body = resolve_value(input, node_config.get("body", ""))

    try:
        response = requests.request(method, url, headers=headers, data=body)
        response.raise_for_status()
        output = {"status_code": response.status_code, "response_body": response.text}
        return True, output
    except Exception as e:
        print_debug(f"HTTP request failed: {e}", debug, tags=["NODE", "HTTP", "ERROR"])
        return False, {}


def node_db(
        node: dict,
        input: Optional[dict] = None,
        *,
        debug: bool = False,
) -> tuple[bool, dict]:
    # Placeholder for database node implementation
    return False, {}


def node_sardine(
        node: dict,
        input: Optional[dict] = None,
        base64: Any = None,
        *,
        debug: bool = False,
        images: List[Image.Image] = []
) -> tuple[bool, List[Image.Image], List[str]]:
    node_config = get_node_config(node)

    model_path = node_config.get("model_path", "")
    if not model_path:
        print_debug("No model path specified for Sardine node.", debug, tags=["NODE", "SARDINE", "ERROR"])
        return False, [], []

    accepted_files = node_config.get("accepted_files", [])
    if not accepted_files:
        print_debug("No accepted files specified for Sardine node.", debug, tags=["NODE", "SARDINE", "ERROR"])
        return False, [], []
    
    if not images:
        page_mode: Literal["first_page_only", "all_pages"] = node_config.get("page_mode", "first_page_only")
        page_dpi: int = node_config.get("page_dpi", 200)

        images = extract_images_from_base64(
            base64,
            page_mode=page_mode,
            page_dpi=page_dpi,
            debug=debug
        )

    device = node_config.get("device", "cpu")

    try:
        classes = _predict_document_class(
            images,
            model_path=model_path,
            device=device,
            debug=debug
        )
    except Exception as e:
        print_debug(f"Document classification failed: {e}", debug, tags=["NODE", "SARDINE", "ERROR"])
        return False, [], []

    accepted_images = []

    for i, cls in enumerate(classes):
        if cls in accepted_files:
            accepted_images.append(images[i])

    return True, accepted_images, classes


def node_detection(
        node: dict,
        input: Optional[dict] = None,
        *,
        debug: bool = False,
        images: List[Image.Image] = []
) -> tuple[bool, dict]:
    node_config = get_node_config(node)

    model_path = node_config.get("model_path", "")
    if not model_path:
        print_debug("No model path specified for Detection node.", debug, tags=["NODE", "DETECTION", "ERROR"])
        return False, {}

    if not images:
        page_mode: Literal["first_page_only", "all_pages"] = node_config.get("page_mode", "first_page_only")
        page_dpi: int = node_config.get("page_dpi", 200)

        images = extract_images_from_base64(
            base64,
            page_mode=page_mode,
            page_dpi=page_dpi,
            debug=debug
        )

    zone_padding: int = node_config.get("zone_padding", 8)
    device = node_config.get("device", "cpu")
    conf: float = node_config.get("confidence_threshold", 0.25)

    try:
        detections = _predict_document_detection(
            images,
            model_path=model_path,
            device=device,
            conf=conf,
            debug=debug,
            padding=zone_padding
        )
    except Exception as e:
        print_debug(f"Document detection failed: {e}", debug, tags=["NODE", "DETECTION", "ERROR"])


    return False, {}


# ========= Flow Utilities =========
def get_flow_details(flow: dict, result: dict) -> dict:
    total_node_duration = 0.0
    node_count = 0

    for k, v in result.get("_debug", {}).get("nodes", {}).items():
        duration = v.get("duration", 0.0)
        total_node_duration += duration
        node_count += 1

    average_node_duration = (total_node_duration / node_count) if node_count > 0 else 0.0

    flow_details = {
        "node_count": node_count,
        "total_node_duration": total_node_duration,
        "average_node_duration": average_node_duration,
    }

    return flow_details


# ========= Flows Engine =========
def process_flow(
    flow: dict,
    base64: Any,
    *,
    debug: bool = False,
) -> dict:
    start_node_id = next((k for k, v in flow.items() if v.get("type") == "start"), None)
    if not start_node_id:
        print_debug("No start node found in the flow.", debug, tags=["FLOW"])
        raise ValueError("No start node found in the flow.")

    start_node = get_node_by_id(flow, start_node_id)
    start_children = get_node_children(start_node)

    result: dict = dict(base64) if isinstance(base64, dict) else {}

    nodes_completed = {start_node_id}
    nodes_queued = set(start_children)
    nodes_running: set[str] = set()

    future_to_node: dict[concurrent.futures.Future, str] = {}

    with ThreadPoolExecutor(max_workers=FLOW_MAX_WORKERS) as executor:
        while nodes_queued or nodes_running:
            for node_id in list(nodes_queued):
                node = get_node_by_id(flow, node_id)
                if node:
                    future = executor.submit(process_node, node, result, node_id=node_id, debug=debug)
                    future_to_node[future] = node_id
                    nodes_running.add(node_id)
                nodes_queued.remove(node_id)

            if not nodes_running:
                break

            done, _ = concurrent.futures.wait(
                future_to_node.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                node_id = future_to_node.pop(future)
                nodes_running.remove(node_id)

                try:
                    status, output, children_ids = future.result()

                    if status:
                        nodes_completed.add(node_id)
                        if output:
                            for k, v in output.items():
                                result = merge_dicts(result, {k: v})

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
                        print_debug(f"Node {node_id} failed logic.", debug, tags="ERROR")

                except Exception as e:
                    print_debug(f"Exception in node {node_id}: {e}", debug, tags="CRITICAL")

    flow_details = get_flow_details(flow, result)
    set_value_at_path(result, "_debug.flow", flow_details)
    print_debug("Flow processing completed.", debug, tags=["FLOW"])
    
    return result


def run(
    flow: dict,
    base64: Any,
    *,
    debug: bool = False,
) -> dict:
    start = time.time()
    print_debug("Flow started", debug)

    result = process_flow(flow, base64, debug=debug)

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

    # 1) Force la branche "match" (vat.number -> siren = 456789000)
    ctx_match = {"seller": {"vat": {"number": "FR000456789000"}}}
    print("=== MATCH ===")
    print(json.dumps(run(test_flow, ctx_match, debug=True), indent=2))

    # 2) Force la branche "default" (vat.number -> siren != 456789000)
    ctx_default = {"seller": {"vat": {"number": "FR000123456789"}}}
    print("=== DEFAULT ===")
    print(json.dumps(run(test_flow, ctx_default, debug=True), indent=2))