import threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.helpers.agents import run as run_agent
from src.helpers.sardine import inference as run_sardine

import re, json

STATUS_LOCK = threading.Lock()
NODE_EVENTS: dict[str, threading.Event] = {}

def _event_for(nid: str) -> threading.Event:
    ev = NODE_EVENTS.get(nid)
    if ev is None:
        ev = threading.Event()
        NODE_EVENTS[nid] = ev
    return ev

# ============ Utils ============
def bool2str(value: bool) -> str:
    return "true" if value else "false"

def print_debug(printable, debug: bool):
    if debug:
        print(printable)

def _parse_money_fr(s) -> float:
    if s is None: return 0.0
    if isinstance(s, (int, float)): return float(s)
    s = str(s).replace("€", "").replace("\u00a0", " ").strip()
    # garder chiffres + , . -
    s = "".join(ch for ch in s if ch.isdigit() or ch in ",.-")
    if not s: return 0.0
    # virgule = décimal FR
    if "," in s and "." not in s:
        s = s.replace(".", "")  # points de milliers éventuels
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

def _parse_percent_fr(s) -> float:
    """ '20,00' -> 0.20 ; '' -> 0.0 """
    v = _parse_money_fr(s)
    return v/100.0 if v > 0 else 0.0

def _iter_lines(line_dict: dict):
    """
    line_dict ressemble à:
      { "label":[...], "quantity":[...], "unitprice":[...], "totalprice":[...], "tva":[...] }
    On itère ligne par ligne en renvoyant un tuple de valeurs par index.
    """
    if not isinstance(line_dict, dict):
        return
    keys = ["label", "quantity", "unitprice", "totalprice", "tva", "reference"]
    cols = {k: line_dict.get(k, []) for k in keys}
    n = max((len(v) for v in cols.values() if isinstance(v, list)), default=0)
    for i in range(n):
        yield {
            "label":      (cols["label"][i] if i < len(cols["label"]) else ""),
            "quantity":   (cols["quantity"][i] if i < len(cols["quantity"]) else ""),
            "unitprice":  (cols["unitprice"][i] if i < len(cols["unitprice"]) else ""),
            "totalprice": (cols["totalprice"][i] if i < len(cols["totalprice"]) else ""),
            "tva":        (cols["tva"][i] if i < len(cols["tva"]) else ""),
            "reference":  (cols["reference"][i] if i < len(cols["reference"]) else ""),
        }

def _amounts_from_lines(line_dict: dict) -> dict:
    """
    Retourne un dict: {"ht": float, "tva": float, "ttc": float}
    en calculant par ligne: ht, tva=ht*rate, ttc=ht+tva.
    """
    total_ht = total_tva = 0.0
    for row in _iter_lines(line_dict):
        q  = _parse_money_fr(row["quantity"])
        pu = _parse_money_fr(row["unitprice"])
        tp = _parse_money_fr(row["totalprice"])
        rate = _parse_percent_fr(row["tva"])
        # HT de la ligne
        ht_line = tp if tp > 0 else (q * pu if q > 0 and pu > 0 else 0.0)
        # TVA de la ligne
        tva_line = ht_line * rate
        total_ht  += ht_line
        total_tva += tva_line
    return {"ht": total_ht, "tva": total_tva, "ttc": total_ht + total_tva}

def _sum_list_money_fr(values) -> float:
    total = 0.0
    if isinstance(values, dict):
        # si c'est la colonne 'totalprice' sous forme dict/colonne => prendre valeurs
        # mais chez toi c'est souvent une list déjà
        values = list(values.values())
    if not isinstance(values, list):
        return _parse_money_fr(values)
    for v in values:
        total += _parse_money_fr(v)
    return total

def _text_from_pages(data) -> str:
    """
    Concatène tous les textes simples présents dans data['pages'] (profondément),
    utile pour regex globales ("Total HT ...", etc.)
    """
    pages = data.get("pages", [])
    chunks = []

    def walk(x):
        if isinstance(x, str):
            chunks.append(x)
        elif isinstance(x, list):
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            # tables: header/columns
            if x.get("type") == "table":
                hdr = x.get("header", [])
                cols = x.get("columns", [])
                for h in hdr: 
                    if isinstance(h, str): chunks.append(h)
                for col in cols:
                    if isinstance(col, list):
                        for cell in col:
                            if isinstance(cell, str):
                                chunks.append(cell)
            else:
                for v in x.values():
                    walk(v)

    walk(pages)
    return "\n".join(chunks)


# ============ Node Actions ============
def node_sardine(config, *, base64=None, debug=False):
    accepted_files = config.get("accepted_files", [])

    if not base64:
        print_debug("[SARDINE] No base64 provided", debug)
        return False, "unknown", []

    dir_to_del = "../../workspace.sardine"
    cls, pages = run_sardine(
        model_detect_path=f"{dir_to_del}/sardine.train/runs/detect/sardine-layout-l14/weights/best.pt",
        model_class_path=f"{dir_to_del}/sardine.train/runs/classify/sardine-type-l3/weights/best.pt",
        model_table_path=f"{dir_to_del}/sardine.train/runs/table/sardine-table-l4/weights/last.pt",
        img_b64=base64,
        device="cpu",
        conf=.7,
        pdf_dpi=834
    )

    print_debug(f"[SARDINE] Classified as: {cls}", debug)

    return cls in accepted_files, cls, pages

def node_agent(config, text, *, debug=False):
    model = config.get("model", "")
    version = config.get("version", "")

    def process_chunk(chunk):
        """Traite un sous-ensemble de textes (chunk) séquentiellement."""
        best_result, max_score, mapper = {}, 0, {}
        for t in chunk:

            t, table = (" ".join(t.get("header", [])), t) if isinstance(t, dict) and t.get("type") == "table" else (t, None)
            print_debug(f"[AGENT] Processed text chunk: {t}", debug and table is not None)

            current, mapper = run_agent(t, reference=model, version=version)
            temp = sum(v["score"] for v in current.values())
            res = {k: v["word"] for k, v in current.items()}
            if temp > max_score:
                max_score, best_result = temp, res
            
            if table is not None:
                headers = table.get("header", [])
                columns = table.get("columns", [])

                for k, v in best_result.items():
                    if not isinstance(v, str): continue
                    for i, col in enumerate(headers):
                        if isinstance(col, str) and (v in col or v == col or v.replace(" ", "") == col.replace(" ", "")):
                            best_result[k] = columns[i]
                            break

                print_debug(f"[AGENT] Table mapping applied: {best_result}", debug)

        mapper_str = json.dumps(mapper, ensure_ascii=False)

        for k, v in best_result.items():
            mapper_str = mapper_str.replace(f'"{k}"', json.dumps(v))
        mapper = json.loads(mapper_str)

        return max_score, best_result, mapper

    if isinstance(text, list):
        best_result, max_score, mapper = {}, 0, {}

        max_workers = 1
        if len(text) <= max_workers:
            text_chunks = [[t] for t in text]
        else:
            chunk_size = (len(text) + max_workers - 1) // max_workers
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in text_chunks}

            for future in as_completed(futures):
                try:
                    temp, res, map = future.result()
                    if temp > max_score:
                        max_score, best_result, mapper = temp, res, map
                except Exception as e:
                    print_debug(f"[AGENT] Error on chunk {futures[future]}: {e}", debug)

        result = mapper

    else:
        _, best, result = run_agent(text, reference=model, version=version)

    print_debug(f"[AGENT] Final Result: {result}", debug)
    return result

def node_agent_group(config, text, *, debug=False):
    agents = config.get("agents", [])

    def run_all_agents_on_text(t):
        """
        Fait passer tous les agents sur un texte t.
        - total_score: somme des scores sur tous les agents (et toutes les clés).
        - merged_result: dict {clé: word} où, par clé, on garde le word de l'agent avec le score le plus élevé.
        """
        total_score = 0
        merged_result = {}
        best_score_per_key = {}
        mapper_combined = []

        for agent in agents:
            model = agent.get("model", "")
            version = agent.get("version", "")

            if isinstance(t, dict) and t.get("type") == "table":
                t = " ".join(t.get("header", []))

            current, mapper = run_agent(t, reference=model, version=version)

            # cumul des scores au niveau "agent"
            total_score += sum(v["score"] for v in current.values())

            mapper_combined.append(mapper)

            # fusion par clé: on garde le "word" avec le meilleur score
            for k, v in current.items():
                s = v["score"]
                if k not in best_score_per_key or s > best_score_per_key[k]:
                    best_score_per_key[k] = s
                    merged_result[k] = v["word"]
                    t = t.replace(v["word"], "\<" + v["word"] + "\>")  # pour debug
        
        mapper_str = json.dumps(mapper_combined, ensure_ascii=False)

        for k, v in merged_result.items():
            safe_v = json.dumps(v, ensure_ascii=False)[1:-1]
            mapper_str = mapper_str.replace(k, safe_v)
        
        mapper_combined = json.loads(mapper_str)

        return total_score, merged_result, mapper_combined

    # ----------- Cas liste de textes : même logique de chunking que node_agent -----------
    if isinstance(text, list):
        best_result, max_score, mapper = {}, 0, {}

        # même paramétrage que node_agent
        max_workers = 1
        if len(text) <= max_workers:
            text_chunks = [[t] for t in text]
        else:
            chunk_size = (len(text) + max_workers - 1) // max_workers
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        def process_chunk(chunk):
            """Évalue séquentiellement les textes d'un chunk avec tous les agents, retourne le meilleur."""
            chunk_best_result, chunk_max_score = {}, 0
            for t in chunk:
                score, merged, map = run_all_agents_on_text(t)
                if score > chunk_max_score:
                    chunk_max_score, chunk_best_result, mapper = score, merged, map
            return chunk_max_score, chunk_best_result, mapper

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in text_chunks}

            for future in as_completed(futures):
                try:
                    temp, res, map = future.result()
                    if temp > max_score:
                        max_score, best_result, mapper = temp, res, map
                except Exception as e:
                    print_debug(f"[AGENT-GROUP] Error on chunk {futures[future]}: {e}", debug)
                
                print(mapper)

        result = mapper

    # ----------- Cas texte unique -----------
    else:
        _, merged, mapper = run_all_agents_on_text(text)
        result = mapper

    print_debug(f"[AGENT-GROUP] Final Result: {result}", debug)
    return result

def get_by_path(d, path):
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def set_by_path(d, path, value):
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

_slice_re = re.compile(r'^([A-Za-z0-9_.]+)(?:\[(\-?\d*):(\-?\d*)\])?$')

_expr_sum   = re.compile(r'^sum\(([^)]+)\)$')
_expr_calc  = re.compile(r'^calc_ttc\(\s*([A-Za-z0-9_.]+)\s*,\s*([A-Za-z0-9_.]+)\s*\)$')
_expr_lines_ht  = re.compile(r'^lines_ht\(\s*([A-Za-z0-9_.]+)\s*\)$')
_expr_lines_tva = re.compile(r'^lines_tva\(\s*([A-Za-z0-9_.]+)\s*\)$')
_expr_lines_ttc = re.compile(r'^lines_ttc\(\s*([A-Za-z0-9_.]+)\s*\)$')
_expr_lines_all = re.compile(r'^lines_amounts\(\s*([A-Za-z0-9_.]+)\s*\)$')

def _get_by_path_any(d, path):
    return get_by_path(d, path)

def resolve_value(value, data):
    if isinstance(value, str):
        v = value.strip()

        # nouvelles fonctions basées sur les lignes
        m = _expr_lines_all.match(v)
        if m:
            path = m.group(1)
            line_obj = _get_by_path_any(data, path)
            return _amounts_from_lines(line_obj)

        m = _expr_lines_ht.match(v)
        if m:
            line_obj = _get_by_path_any(data, m.group(1))
            return _amounts_from_lines(line_obj)["ht"]

        m = _expr_lines_tva.match(v)
        if m:
            line_obj = _get_by_path_any(data, m.group(1))
            return _amounts_from_lines(line_obj)["tva"]

        m = _expr_lines_ttc.match(v)
        if m:
            line_obj = _get_by_path_any(data, m.group(1))
            return _amounts_from_lines(line_obj)["ttc"]

        # existant
        m = _expr_sum.match(v)
        if m:
            path = m.group(1).strip()
            arr = _get_by_path_any(data, path)
            return _parse_money_fr(arr) if not isinstance(arr, list) else sum(_parse_money_fr(x) for x in arr)

        m = _expr_calc.match(v)
        if m:
            ht = _get_by_path_any(data, m.group(1).strip())
            tv = _get_by_path_any(data, m.group(2).strip())
            return (_parse_money_fr(ht) + _parse_money_fr(tv))

        m = _slice_re.match(v)
        if m:
            path, s, e = m.group(1), m.group(2), m.group(3)
            src = get_by_path(data, path)
            if isinstance(src, (str, bytes)):
                s_idx = int(s) if s not in (None, "") else None
                e_idx = int(e) if e not in (None, "") else None
                return src[s_idx:e_idx]
            return src
    return value

def node_edit(config, data):
    key = config.get("key")
    raw_value = config.get("value")

    # calcule la valeur finale si c'est une référence
    value = resolve_value(raw_value, data)
    if value == 0 or value == "0" or value == "":
        return data

    # support des chemins pointés pour la clé
    set_by_path(data, key, value)
    return data
# ============ Flow Processing ============
def ignored_node(flow, o2i, ignored_by: str):
    for nid in o2i:
        node = find_node_by_id(flow, nid)
        with STATUS_LOCK:
            if node.get("ignored_by") is None:
                node["ignored_by"] = [ignored_by]
            elif ignored_by not in node["ignored_by"]:
                node["ignored_by"].append(ignored_by)
        if get_outputs_len(flow, nid) > 0:
            outs = get_all_outputs(flow, nid)
            ignored_node(flow, outs, nid)

def process_outputs(node, *, output_keys: list[str] = []):
    outs = node.get("outputs", {})
    o2v, o2i = [], []
    ks = ["base"] + output_keys

    for out in outs:
        if out in ks:
            o2v.extend(outs[out])
        else:
            o2i.extend(outs[out])

    return o2v, o2i

def process_type(flow, node, *, data={}, nid=None, debug=False):
    node_type = node.get("type")
    node_config = node.get("config", {})

    print_debug(f"[INFO] Start processing type: {node_type}", debug)
    start = time.time()

    node_outputs = None
    node_result = data

    current_type = data.get("type")
    
    match node_type:
        case "start":
            time.sleep(0)
        case "end":
            time.sleep(0)
        case "if":
            result = True
            output_state = bool2str(result)
            node_outputs, o2i = process_outputs(node, output_keys=[output_state])
        case "switch":
            time.sleep(0)
        case "merge":
            time.sleep(0)
        case "edit":
            if current_type:
                result = node_edit(node_config, data[current_type])
                data[current_type] = data[current_type] | result
        case "sardine":
            valid, doc_type, pages = node_sardine(node_config, base64=flow['base64'], debug=debug)
            output_state = "valid" if valid else "invalid"
            node_outputs, o2i = process_outputs(node, output_keys=[output_state])
            data["type"] = doc_type
            data["pages"] = pages
            data[doc_type] = {}
        case "agent":
            result = node_agent(node_config, data.get("pages", [])[0], debug=debug)
            if current_type and isinstance(result, dict):
                data[current_type] = data[current_type] | result
        case "agent-group":
            result = node_agent_group(node_config, data.get("pages", [])[0], debug=debug)
            if current_type:
                if isinstance(result, dict):
                    data[current_type] = data[current_type] | result
                elif isinstance(result, list):
                    k = list(result[0].keys())[0]
                    data[current_type][k] = {}
                    for r in result[1:]:
                        data[current_type][k] = data[current_type][k] | r
        case _:
            print_debug(f"[WARN] Unknown type: {node_type}", debug)

    if not node_outputs:
        node_outputs, o2i = process_outputs(node)

    if o2i and len(o2i) > 0:
        ignored_node(flow, o2i, nid)

    end = time.time()
    print_debug(f"[INFO] Finished processing type: {node_type} in {end - start:.2f}s", debug)

    return node_outputs, node_result

def process_node(flow, node, *, id=None, debug=False, data=None):
    if data is None:
        data = {}

    parents = node.get("inputs", []) or []
    while parents:
        ignored_by = set(node.get("ignored_by") or [])
        needed_parents = [p for p in parents if p not in ignored_by]
        if all(flow[p].get("status") == "processed" for p in needed_parents):
            break
        time.sleep(.05)

    node["status"] = "processing"
    nos, data = process_type(flow, node, nid=id, debug=debug, data=data)
    node["status"] = "processed"

    threads = []
    for no in nos:
        t = threading.Thread(target=process_node, args=(flow, flow[no]), kwargs={"id": no, "debug": debug, "data": data})
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return data

def find_start_node(flow):
    for k, v in flow.items():
        if v.get("type") == "start":
            return v
    return None

def find_node_by_id(flow, id):
    return flow.get(id, None)

def get_outputs_len(flow, id):
    node = find_node_by_id(flow, id)
    if node:
        outputs = node.get("outputs", {})
        return sum(len(v) for v in outputs.values())
    return 0

def get_all_outputs(flow, id):
    node = find_node_by_id(flow, id)
    if node:
        outputs = node.get("outputs", {})
        all_outputs = []
        for v in outputs.values():
            all_outputs.extend(v)
        return all_outputs
    return []

def get_inputs_len(flow, id):
    node = find_node_by_id(flow, id)
    if node:
        inputs = node.get("inputs", [])
        return len(inputs)
    return 0

def run(flow, *, base64=None, debug=False):
    def reduce_str(s: str, max_len=50) -> str:
        return f"{s[:max_len]}..." if len(s) > max_len else s

    start = time.time()
    start_node = find_start_node(flow)

    if not start_node:
        print("No start node found")
        return
    
    flow['base64'] = base64

    results = process_node(flow, start_node, debug=debug)

    end = time.time()
    print_debug(f"[INFO] Flow executed in {end - start:.2f}s", debug)

    flow_pages = results.get("pages", [])
    for i in range(len(flow_pages)):
        if isinstance(flow_pages[i], list):
            flow_pages[i] = [ (reduce_str(z, 50) if isinstance(z, str) else z) for z in flow_pages[i] ]
        else:
            flow_pages[i] = reduce_str(flow_pages[i], 50)

    print_debug(f"[INFO] Final results: {results}", debug)
    return results

flow = {
    "001": {
        "type": "start",
        "outputs": { "base": ["002"] },
        "inputs": []
    },
    "002": {
        "type": "sardine",
        "config": {
            "accepted_files": ["facture"],
            "result_sardine": "..."
        },
        "outputs": { "valid": ["003", "004", "005", "006", "007", "008", "009", "010", "011"], "invalid": ["400"] },
        "inputs": ["001"]
    },

    "003": {
        "type": "agent-group",
        "config": {
            "agents": [
                {
                    "model": "addressto",
                    "version": "1.13",
                },
                {
                    "model": "address",
                    "version": "2.1",
                }
            ],
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "004": {
        "type": "agent-group",
        "config": {
            "agents": [
                {
                    "model": "addressfrom",
                    "version": "1.12",
                },
                {
                    "model": "address",
                    "version": "2.1",
                }
            ],
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "005": {
        "type": "agent-group",
        "config": {
            "agents": [
                {
                    "model": "addressship",
                    "version": "1.4"
                },
                {
                    "model": "address",
                    "version": "2.1",
                }
            ],
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "006": {
        "type": "agent",
        "config": {
            "model": "amounts",
            "version": "1.1"
        },
        "outputs": { "base": ["011calcHT"] },
        "inputs": ["002"]
    },
    "007": {
        "type": "agent",
        "config": {
            "model": "vatsiren",
            "version": "1.5",
        },
        "outputs": { "base": ["007bis"] },
        "inputs": ["002"]
    },
    "007bis": {
        "type": "edit",
        "config": {
            "key": "siren",
            "value": "vat.number[4:15]"
        },
        "outputs": { "base": ["200"] },
        "inputs": ["007"]
    },
    "008": {
        "type": "agent",
        "config": {
            "model": "invoicenumber",
            "version": "2.3",
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "009": {
        "type": "agent",
        "config": {
            "model": "invoicecategory",
            "version": "1.0",
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "010": {
        "type": "agent",
        "config": {
            "model": "invoicecurrency",
            "version": "1.0",
        },
        "outputs": { "base": ["200"] },
        "inputs": ["002"]
    },
    "011": {
        "type": "agent",
        "config": {
            "model": "lines",
            "version": "1.0",
        },
        "outputs": { "base": ["011calcHT"] },
        "inputs": ["002"]
    },

    "011calcHT": {
        "type": "edit",
        "config": {
            "key": "amount.ht",
            "value": "lines_ht(line)"
        },
        "outputs": { "base": ["011calcTVA"] },
        "inputs": ["011", "006"]
    },
    "011calcTVA": {
        "type": "edit",
        "config": {
            "key": "amount.tva",
            "value": "lines_tva(line)"
        },
        "outputs": { "base": ["011calcTTC"] },
        "inputs": ["011calcHT"]
    },
    "011calcTTC": {
        "type": "edit",
        "config": {
            "key": "amount.ttc",
            "value": "lines_ttc(line)"
        },
        "outputs": { "base": ["200"] },
        "inputs": ["011calcTVA"]
    },
    
    "400": {
        "type": "end",
        "outputs": { "base": [] },
        "inputs": ["002"]
    },
    "200": {
        "type": "end",
        "outputs": { "base": [] },
        "inputs": ["003", "004", "005", "007bis", "008", "009", "010", "011calcTTC"]
    }
}

if __name__ == "__main__":
    run(flow, debug=True)