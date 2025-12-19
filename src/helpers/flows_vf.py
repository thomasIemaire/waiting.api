import time
import json
import re
import ast
import operator as op
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Union

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


# ========= Node Utilities =========
def get_node_by_id(flow: dict, node_id: str) -> Optional[dict]:
    return flow.get(node_id, None)


def get_node_children(node: dict) -> list:
    return node.get("outputs", [])


def get_node_parents(node: dict) -> list:
    return node.get("inputs", [])


# ========= Nodes Processing =========
def process_node(
    node: dict,
    input: Optional[dict] = None,
    *,
    node_id: Optional[str] = None,
    debug: bool = False,
) -> tuple[bool, dict]:
    print_debug(
        f"START Processing {node_id} (Type: {node.get('type')})",
        debug,
        tags=["NODE", "THREAD"],
    )
    node_config = node.get("config", {})
    node_root = node_config.get("root", None)

    status, output = execute_node_by_type(node, input, debug=debug)

    print_debug(f"END Processing {node_id}", debug, tags=["NODE", "THREAD"])
    return status, output if not node_root else {node_root: output}


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

    match node_type:
        case "start":
            return True, (input or {})
        case "edit":
            return True, node_edit(node, input, debug=debug)
        case "final":
            return True, (input or {})
        case _:
            import random
            output = {"result": random.randint(1, 100)}
            print_debug(f"Unknown node type: {node_type}", debug, tags=["NODE", "WARNING"])
            return True, output


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
    node_config = node.get("config", {})

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

        set_value_at_path(output, key_path, resolved)

    return output


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
                    status, output = future.result()

                    if status:
                        nodes_completed.add(node_id)
                        if output:
                            for k, v in output.items():
                                result = merge_dicts(result, {k: v})

                        current_node = get_node_by_id(flow, node_id)
                        children_ids = get_node_children(current_node)

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
    print_debug(f"Flow ended in {end - start:.2f} seconds", debug)

    return result


if __name__ == "__main__":
    test_flow = {
        "start": {"id": "start", "type": "start", "outputs": ["node_A", "node_B"]},
        "node_A": {"id": "node_A", "type": "action", "config": {"root": "node_A"}, "inputs": ["start"], "outputs": ["node_C"]},
        "node_B": {"id": "node_B", "type": "action", "config": {"root": "node_B"}, "inputs": ["start"], "outputs": ["node_C"]},
        "node_C": {
            "id": "node_C",
            "type": "edit",
            "config": {
                "fields": [
                    {"key": "node_C.value", "value": "${node_A.result} + ${node_B.result}"},
                    {"key": "seller.siren", "value": "$seller.vat.number[-9:]"},
                ]
            },
            "inputs": ["node_A", "node_B"],
            "outputs": ["node_D"],
        },
        "node_D": {"id": "node_D", "type": "final", "inputs": ["node_C"], "outputs": []},
    }

    initial_context = {"seller": {"vat": {"number": "FR12345678900012"}}}
    print(json.dumps(run(test_flow, initial_context, debug=True), indent=2))