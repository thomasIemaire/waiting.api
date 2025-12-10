from platform import node
import threading, time, re, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import src
from src.app import data
from src.helpers.agents import run as run_agent
# On importe les fonctions séparées depuis le nouveau sardine.py
from src.helpers.sardine import load_images, predict_classification, predict_detection

STATUS_LOCK = threading.Lock()

# Verrou spécifique pour empêcher l'exécution multiple d'un même nœud (ex: Merge)
NODE_EXEC_LOCKS = {} 
def _get_node_lock(nid):
    if nid not in NODE_EXEC_LOCKS:
        NODE_EXEC_LOCKS[nid] = threading.Lock()
    return NODE_EXEC_LOCKS[nid]

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
    s = "".join(ch for ch in s if ch.isdigit() or ch in ",.-")
    if not s: return 0.0
    if "," in s and "." not in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

def deep_merge(target: dict, source: dict) -> dict:
    """Fusionne récursivement source dans target sans écraser les sous-clés existantes."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target

def _sum_list_money_fr(values) -> float:
    total = 0.0
    if isinstance(values, dict):
        values = list(values.values())
    if not isinstance(values, list):
        return _parse_money_fr(values)
    for v in values:
        total += _parse_money_fr(v)
    return total

# ============ Node Actions ============

def node_sardine(config, *, base64=None, debug=False, data=None):
    print(f"[FLOW-DEBUG] >>> Node Sardine STARTED")
    accepted_files = config.get("accepted_files", [])
    print(f"[FLOW-DEBUG] Sardine accepted files: {accepted_files}")
    
    # 1. Chargement des images (mise en cache dans data pour éviter rechargement PDF)
    if data is not None and data.get("_pil_images") is None and base64:
        data["_pil_images"] = load_images(base64, pdf_dpi=768)
    
    images = data.get("_pil_images", []) if data else []
    
    # 2. Classification uniquement
    dir_agents = "../sardine.agents"
    # On suppose que le modèle de classification est dans sardine.agents/sard-cls
    cls = predict_classification(
        images, 
        model_path=f"{dir_agents}/sard-cls/best.pt", 
        device="cpu"
    )
    
    # Pages vide ici, sera rempli par zone-detection
    pages = [] 

    print(f"[FLOW-DEBUG] [SARDINE] Classified as: {cls}")

    is_valid = cls in accepted_files
    print(f"[FLOW-DEBUG] <<< Node Sardine FINISHED. Valid? {is_valid}")
    return is_valid, cls, pages

def node_zone_detection(config, *, base64=None, debug=False, data=None):
    print(f"[FLOW-DEBUG] >>> Node Zone Detection STARTED")
    
    # 1. Récupération des images du cache
    if data is not None and data.get("_pil_images") is None and base64:
        data["_pil_images"] = load_images(base64, pdf_dpi=768)
    
    images = data.get("_pil_images", []) if data else []

    # 2. Détection et OCR uniquement
    dir_agents = "../sardine.agents"
    # On suppose que le modèle de détection est dans sardine.agents/sard-det
    pages = predict_detection(
        images, 
        model_path=f"{dir_agents}/sard-det/best.pt",
        device="cpu",
        conf=0.3
    )
    
    # On récupère le type déjà classifié s'il existe (via Sardine), sinon unknown
    cls = data.get("type", "unknown") if data else "unknown"

    print(f"[FLOW-DEBUG] [ZONE-DETECTION] Extracted zones for type: {cls}, Pages count: {len(pages)}")
    print(f"[FLOW-DEBUG] <<< Node Zone Detection FINISHED")
    return True, cls, pages

def clean_tokens(mapper_str: str, model: str) -> str:
    model_upper = model.upper()
    pattern = rf'(?<="){re.escape(model_upper)}(?:_[A-Za-z]+)*(?=")'
    cleaned = re.sub(pattern, '', mapper_str)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

def node_agent(config, text, *, debug=False):
    model = config.get("model", "")
    version = config.get("version", "")
    type = config.get("type", "single")
    processing_mode = config.get("processing_mode", "per_zone")
    max_zones_raw = config.get("max_zones")
    try:
        max_zones = int(max_zones_raw) if max_zones_raw is not None else None
    except (TypeError, ValueError):
        max_zones = None

    print(f"[FLOW-DEBUG] >>> Node Agent '{model}' (v{version}) STARTED")

    if not text:
        print("[FLOW-DEBUG] [AGENT] WARNING: Input text is empty or None!")
    elif isinstance(text, list):
        print(f"[FLOW-DEBUG] [AGENT] Input text is a list of {len(text)} elements.")
    else:
        print(f"[FLOW-DEBUG] [AGENT] Input text length: {len(str(text))}")

    # Conversion en liste de zones indépendantes
    if not isinstance(text, list):
        text = [text]

    processed_text_list = []
    for t in text:
        if isinstance(t, list):
            # t peut représenter toutes les zones d'une page -> on traite chaque zone indépendamment
            for zone in t:
                if isinstance(zone, list):
                    processed_text_list.append("\n".join([str(line) for line in zone]))
                elif isinstance(zone, dict) and zone.get("type") == "table":
                    processed_text_list.append(" ".join(zone.get("header", [])))
                else:
                    processed_text_list.append(str(zone))
        elif isinstance(t, dict) and t.get("type") == "table":
            processed_text_list.append(" ".join(t.get("header", [])))
        else:
            processed_text_list.append(str(t))

    # Option de limitation du nombre de zones à traiter
    if isinstance(max_zones, int) and max_zones > 0:
        processed_text_list = processed_text_list[:max_zones]
        print(f"[FLOW-DEBUG] [AGENT] Limiting processing to first {max_zones} zones.")

    # Option pour traiter tout le texte en une seule fois
    if str(processing_mode).lower() == "all_at_once":
        combined_text = "\n\n".join(processed_text_list)
        processed_text_list = [combined_text]
        print(f"[FLOW-DEBUG] [AGENT] Processing all text at once. Combined length: {len(combined_text)}")

    # --- AGREGATION ---
    # On parcourt TOUTES les zones et on garde le meilleur résultat pour chaque champ
    aggregated_entities = {} # { "LABEL": {"word": "...", "score": ...} }
    mapper_template = {}

    for zone_text in processed_text_list:
        current_entities, mapper = run_agent(zone_text, reference=model, version=version)

        # On sauvegarde le template de mapper s'il n'est pas encore défini
        if mapper and not mapper_template:
            mapper_template = mapper

        # Fusion intelligente: on garde le meilleur score pour chaque label
        for label, entity in current_entities.items():
            score = entity.get("score", 0)
            if label not in aggregated_entities or score > aggregated_entities[label].get("score", 0):
                aggregated_entities[label] = entity

    # Construction du résultat final à partir des meilleures entités trouvées
    final_mapper = {}
    if mapper_template:
        mapper_str = json.dumps(mapper_template, ensure_ascii=False)
        for label, entity in aggregated_entities.items():
            val = entity.get("word", "")
            # Remplacement intelligent
            mapper_str = mapper_str.replace(f'"{label}"', json.dumps(val)) # Clé directe
            token = f"{model.upper()}_{label.upper()}"
            mapper_str = mapper_str.replace(f'"{token}"', json.dumps(val)) # Token complet

        mapper_str = clean_tokens(mapper_str, model)
        try:
            final_mapper = json.loads(mapper_str)
        except:
            print("[FLOW-DEBUG] [AGENT] Error parsing JSON mapper result.")
            final_mapper = {}
    elif aggregated_entities:
        # Fallback si pas de mapper (ex: mode sans config)
        final_mapper = {k: v["word"] for k, v in aggregated_entities.items()}

    print(f"[FLOW-DEBUG] <<< Agent '{model}' Result keys (Aggregated): {list(final_mapper.keys())}")
    return final_mapper

def node_agent_group(config, text, *, debug=False):
    agents = config.get("agents", [])
    print(f"[FLOW-DEBUG] >>> Node Agent Group ({len(agents)} agents) STARTED")

    # Conversion en liste de strings
    if not isinstance(text, list): text = [text]
    processed_text_list = []
    for t in text:
        if isinstance(t, list):
            processed_text_list.append("\n".join([str(line) for line in t]))
        elif isinstance(t, dict) and t.get("type") == "table":
            processed_text_list.append(" ".join(t.get("header", [])))
        else:
            processed_text_list.append(str(t))

    # Agrégation globale pour le groupe
    aggregated_entities_by_agent = {} # { agent_model: { LABEL: best_entity } }
    mappers_by_agent = {}

    for page_text in processed_text_list:
        for agent in agents:
            model = agent.get("model", "")
            version = agent.get("version", "")
            
            if model not in aggregated_entities_by_agent:
                aggregated_entities_by_agent[model] = {}

            current, mapper = run_agent(page_text, reference=model, version=version)
            
            if mapper and model not in mappers_by_agent:
                mappers_by_agent[model] = mapper

            for label, entity in current.items():
                score = entity.get("score", 0)
                best_so_far = aggregated_entities_by_agent[model].get(label)
                if not best_so_far or score > best_so_far.get("score", 0):
                    aggregated_entities_by_agent[model][label] = entity

    # Construction du résultat final combiné
    final_combined_mapper = {}
    
    # On itère sur chaque agent pour reconstruire son morceau de JSON
    for agent in agents:
        model = agent.get("model", "")
        mapper_template = mappers_by_agent.get(model, {})
        best_entities = aggregated_entities_by_agent.get(model, {})
        
        if mapper_template:
            m_str = json.dumps(mapper_template, ensure_ascii=False)
            for label, ent in best_entities.items():
                val = ent.get("word", "")
                m_str = m_str.replace(f'"{label}"', json.dumps(val))
                token = f"{model.upper()}_{label.upper()}"
                m_str = m_str.replace(f'"{token}"', json.dumps(val))
            
            m_str = clean_tokens(m_str, model)
            try:
                agent_res = json.loads(m_str)
                # Merge dans le résultat final (attention aux écrasements si clés identiques)
                for key, value in agent_res.items():
                    if key in final_combined_mapper:
                        existing = final_combined_mapper[key]
                        if isinstance(existing, dict) and isinstance(value, dict):
                            deep_merge(existing, value)
                        elif isinstance(existing, list):
                            if value not in existing:
                                existing.append(value)
                        elif existing != value:
                            final_combined_mapper[key] = [existing, value]
                    else:
                        final_combined_mapper[key] = value
            except:
                pass
        else:
            # Fallback
            for k, v in best_entities.items():
                val = v.get("word")
                if k in final_combined_mapper:
                    existing = final_combined_mapper[k]
                    if isinstance(existing, list):
                        if val not in existing:
                            existing.append(val)
                    elif existing != val:
                        final_combined_mapper[k] = [existing, val]
                else:
                    final_combined_mapper[k] = val

    print(f"[FLOW-DEBUG] <<< Node Agent Group FINISHED. Result keys: {list(final_combined_mapper.keys())}")
    return final_combined_mapper

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
_template_var_re = re.compile(r'\{\{([A-Za-z0-9_.]+)\}\}')

def _get_by_path_any(d, path):
    return get_by_path(d, path)

def smart_get(data, path):
    """Cherche une valeur à la racine, dans 'analysis', ou dans le sous-objet typé"""
    # 1. Recherche exacte (ex: "facture.vat.number")
    val = get_by_path(data, path)
    if val is not None: return val

    # 2. Recherche dans 'analysis' (ex: "vat.number" -> "analysis.vat.number")
    val = get_by_path(data, f"analysis.{path}")
    if val is not None: return val

    # 3. Recherche dans le type de document (ex: "vat.number" -> "facture.vat.number")
    doc_type = data.get("type")
    if doc_type and isinstance(doc_type, str):
        val = get_by_path(data, f"{doc_type}.{path}")
        if val is not None: return val
    
    return None

def resolve_value(value, data):
    if isinstance(value, str):
        v = value.strip()
        full_match = _template_var_re.fullmatch(v)
        if full_match:
            return smart_get(data, full_match.group(1))
        
        if "{{" in v:
            def replace_var(m):
                val = get_by_path(data, m.group(1))
                return str(val) if val is not None else ""
            v = _template_var_re.sub(replace_var, v)
            if re.match(r'^[\d\.\s\+\-\*\/\(\)]+$', v):
                try:
                    return eval(v)
                except Exception:
                    pass
            return v

        m = _expr_sum.match(v)
        if m:
            arr = _get_by_path_any(data, m.group(1).strip())
            return _sum_list_money_fr(arr)
        m = _expr_calc.match(v)
        if m:
            ht = _get_by_path_any(data, m.group(1).strip())
            tv = _get_by_path_any(data, m.group(2).strip())
            return (_parse_money_fr(ht) + _parse_money_fr(tv))
        m = _slice_re.match(v)
        if m:
            path, s, e = m.group(1), m.group(2), m.group(3)
            src = smart_get(data, path)
            if isinstance(src, (str, bytes)):
                # Gestion sécurisée des index vides (ex: [-9:] ou [:5])
                s_idx = int(s) if s not in (None, "") else None
                e_idx = int(e) if e not in (None, "") else None
                return src[s_idx:e_idx]
            return src
    return value

def node_edit(config, data):
    key = config.get("key")
    raw_value = config.get("value")
    if not key: return data
    value = resolve_value(raw_value, data)

    if "analysis" not in data:
        data["analysis"] = {}

    if get_by_path(data, f"analysis.{key}") is not None:
            set_by_path(data, f"analysis.{key}", value)
            return data
    
    if not key.startswith("analysis."):
            set_by_path(data, f"analysis.{key}", value)
            return data

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

    # Log pour voir où ça part
    print(f"[FLOW-DEBUG] Resolving outputs for node {node.get('type')}. Keys active: {ks}")

    for out in outs:
        if out in ks:
            o2v.extend(outs[out])
        else:
            o2i.extend(outs[out])

    print(f"[FLOW-DEBUG] Valid Outputs: {o2v} | Ignored Outputs: {o2i}")
    return o2v, o2i

def normalize_val(v):
    """Convertit None en '' et force le string pour la comparaison"""
    if v is None: return ""
    return str(v)

def process_type(flow, node, *, data={}, nid=None, debug=False):
    node_type = node.get("type")
    node_config = node.get("config", {})

    print(f"[FLOW-DEBUG] START Processing Node: {node_type} (ID: {nid})")
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
            conditions = node_config.get("conditions", [])
            matched_index = -1
            
            for i, cond in enumerate(conditions):
                rules = cond.get("rules", [])
                if not rules:
                    # Fallback compatibilité
                    rules = [{"left": cond.get("left"), "operator": cond.get("operator", "=="), "right": cond.get("right"), "link": "AND"}]
                
                # Évaluation séquentielle des règles
                # On évalue la première règle
                current_result = False
                
                # Helper pour évaluer une règle unique
                def eval_rule(r):
                    left_raw = resolve_value(r.get("left"), data)
                    # Gestion du mode "valeur brute" ou "clé dynamique" pour la partie droite
                    if r.get("rightIsKey") is True:
                        right_raw = resolve_value(r.get("right"), data)
                    else:
                        # Si ce n'est pas une clé, c'est une valeur brute (déjà résolue ou string)
                        right_raw = resolve_value(r.get("right"), data)

                    op = r.get("operator", "==")
                    
                    # Normalisation pour comparaison vide/null
                    l_str, r_str = normalize_val(left_raw), normalize_val(right_raw)

                    try:
                        # Tentative comparaison numérique
                        l_num, r_num = float(left_raw), float(right_raw)
                        is_num = True
                    except (ValueError, TypeError):
                        is_num = False

                    if op == "==": return l_str == r_str
                    elif op == "!=": return l_str != r_str
                    elif op == "contains": return r_str in l_str
                    elif is_num:
                        if op == ">": return l_num > r_num
                        elif op == ">=": return l_num >= r_num
                        elif op == "<": return l_num < r_num
                        elif op == "<=": return l_num <= r_num
                    else:
                        # Comparaison alphabétique si pas numérique
                        if op == ">": return l_str > r_str
                        elif op == ">=": return l_str >= r_str
                        elif op == "<": return l_str < r_str
                        elif op == "<=": return l_str <= r_str
                    return False

                # Initialisation avec la première règle
                if len(rules) > 0:
                    current_result = eval_rule(rules[0])

                # Chaînage des règles suivantes
                for k in range(1, len(rules)):
                    prev_rule = rules[k-1]
                    curr_rule = rules[k]
                    logic_link = prev_rule.get("link", "AND") # Le lien est défini sur la règle PRÉCÉDENTE
                    
                    next_res = eval_rule(curr_rule)
                    
                    if logic_link == "OR":
                        current_result = current_result or next_res
                    else:
                        current_result = current_result and next_res

                if current_result:
                    matched_index = i
                    break
            
            if matched_index >= 0:
                print(f"[FLOW-DEBUG] IF matched index: {matched_index}")
                # Logique de sortie inchangée
                if len(node.get("outputs", {})) == 2 and "true" in node.get("outputs", {}):
                     node_outputs, o2i = process_outputs(node, output_keys=["true" if matched_index == 0 else "false"])
                else:
                     available_keys = list(node.get("outputs", {}).keys())
                     if matched_index < len(available_keys):
                         node_outputs, o2i = process_outputs(node, output_keys=[available_keys[matched_index]])

        case "switch":
            key = node_config.get("key")
            val = get_by_path(data, key)
            val_str = str(val) if val is not None else ""
            
            active_key = "default"
            cases = node_config.get("cases", [])
            for c in cases:
                c_val = resolve_value(c.get("value"), data)
                if str(c_val) == val_str:
                    active_key = c.get("name")
                    break
            print(f"[FLOW-DEBUG] SWITCH active key: {active_key}")
            node_outputs, o2i = process_outputs(node, output_keys=[active_key])

        case "merge":
            time.sleep(0)
        case "edit":
            fields = node_config.get("fields", [])
            if fields:
                for f in fields:
                    data = node_edit(f, data)
            else:
                data = node_edit(node_config, data)
                
        case "sardine":
            print(f"[FLOW-DEBUG] Running Sardine node...")
            # Passe 'data' pour le cache d'images
            valid, doc_type, pages = node_sardine(
                node_config, 
                base64=flow.get('base64'), 
                debug=debug, 
                data=data
            )
            
            available_outs = node.get("outputs", {})
            
            if valid:
                if "valide" in available_outs: output_state = "valide"
                else: output_state = "valid"
            else:
                if "invalide" in available_outs: output_state = "invalide"
                else: output_state = "invalid"

            print(f"[FLOW-DEBUG] Sardine Output State: {output_state} (Based on available: {list(available_outs.keys())})")
            
            node_outputs, o2i = process_outputs(node, output_keys=[output_state])
            
            data["type"] = doc_type
            if not data.get("pages"):
                data["pages"] = pages 

            if "analysis" not in data:
                data["analysis"] = {}

        case "zone-detection":
            print(f"[FLOW-DEBUG] Running Zone Detection node...")
            # Passe 'data' pour le cache d'images
            _, doc_type, pages = node_zone_detection(
                node_config, 
                base64=flow.get('base64'), 
                debug=debug, 
                data=data
            )
            
            # Ici on met à jour le type et les pages (vrais résultats OCR)
            if doc_type != "unknown":
                data["type"] = doc_type
            data["pages"] = pages
            
            if "analysis" not in data:
                data["analysis"] = {}
            
            node_outputs, o2i = process_outputs(node)

        case "agent":
            pages_text = data.get("pages", [])
            if not pages_text:
                print("[FLOW-DEBUG] [WARN] Agent node executed but NO PAGES found in data!")
            
            result = node_agent(node_config, pages_text, debug=debug)
            
            if "analysis" not in data:
                data["analysis"] = {}

            if isinstance(result, dict):
                deep_merge(data["analysis"], result)

        case "agent-group":
            pages_text = data.get("pages", [])
            result = node_agent_group(node_config, pages_text, debug=debug)
            
            if "analysis" not in data:
                data["analysis"] = {}

            if isinstance(result, dict):
                deep_merge(data["analysis"], result)
            elif isinstance(result, list):
                pass

        case _:
            print(f"[FLOW-DEBUG] [WARN] Unknown node type encountered: {node_type}")

    if not node_outputs:
        node_outputs, o2i = process_outputs(node)

    if o2i and len(o2i) > 0:
        print(f"[FLOW-DEBUG] Ignoring downstream nodes: {o2i}")
        ignored_node(flow, o2i, nid)

    end = time.time()
    print(f"[FLOW-DEBUG] END Processing Node: {node_type} (Duration: {end - start:.2f}s)")

    return node_outputs, node_result

def process_node(flow, node, *, id=None, debug=False, data=None):
    if data is None: data = {}

    parents = node.get("inputs", []) or []
    
    if parents:
        print(f"[FLOW-DEBUG] Node {node.get('type')} ({id}) waiting for parents: {parents}")

    while parents:
        ignored_by = set(node.get("ignored_by") or [])
        needed_parents = [p for p in parents if p not in ignored_by]

        # Branche morte
        if not needed_parents and parents:
             print(f"[FLOW-DEBUG] Node {node.get('type')} ({id}) is in an ignored branch. Stopping.")
             return data

        # Si tous les parents nécessaires sont "processed", on avance
        if all(flow[p].get("status") == "processed" for p in needed_parents):
            print(f"[FLOW-DEBUG] All parents processed for node {node.get('type')} ({id}). Proceeding.")
            break

        time.sleep(0.05)

    # --- FIX MERGE : VERROUILLAGE ---
    # On utilise un verrou par nœud pour s'assurer qu'un seul thread parent déclenche l'exécution
    lock = _get_node_lock(id)
    event = _event_for(id)
    with lock:
        status = node.get("status")
        if status == "processing":
            print(f"[FLOW-DEBUG] Node {node.get('type')} ({id}) already processing. Waiting for completion to avoid duplicates.")
            # Libère le verrou et attend la fin d'exécution existante
            pass
        elif status == "processed":
            print(f"[FLOW-DEBUG] Node {node.get('type')} ({id}) already processed. Reusing existing result.")
            return data
        else:
            node["status"] = "processing"
            event.clear()
            status = "start-processing"

    if status == "processing":
        event.wait()
        return data

    # Exécution du nœud
    try:
        nos, data = process_type(flow, node, nid=id, debug=debug, data=data)
    finally:
        # Marquer comme terminé (même en cas d'exception) pour libérer les éventuels appels concurrents
        node["status"] = "processed"
        event.set()

    # Lancement des enfants en parallèle
    threads = []
    for no in nos:
        print(f"[FLOW-DEBUG] Spawning thread for next node: {no}")
        t = threading.Thread(target=process_node, args=(flow, flow[no]), kwargs={"id": no, "debug": debug, "data": data})
        threads.append(t)
        t.start()

    # On attend la fin des branches enfants
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
        return len(node.get("inputs", []))
    return 0

def run(flow, *, base64=None, debug=False):
    def reduce_str(s: str, max_len=50) -> str:
        return f"{s[:max_len]}..." if len(s) > max_len else s

    start = time.time()
    
    # Reset status & events to avoid stale synchronization between runs
    NODE_EVENTS.clear()
    for nid, n in flow.items():
        n["status"] = "pending"
        # Prépare les events utilisés pour synchroniser les nœuds
        NODE_EVENTS[nid] = threading.Event()
    
    start_node = find_start_node(flow)
    if not start_node:
        print("[FLOW-ERROR] No start node found")
        return {}
    
    flow['base64'] = base64
    start_node_id = next((k for k, v in flow.items() if v.get("type") == "start"), None)

    print(f"[FLOW-DEBUG] Flow Run Started. Start Node ID: {start_node_id}")

    results = process_node(flow, start_node, id=start_node_id, debug=debug)

    # Nettoyage des objets non-sérialisables (PIL images)
    if results:
        results.pop("_pil_images", None)

    end = time.time()
    print(f"[FLOW-DEBUG] Flow execution finished in {end - start:.2f}s")

    flow_pages = results.get("pages", [])
    for i in range(len(flow_pages)):
        if isinstance(flow_pages[i], list):
            flow_pages[i] = [ (reduce_str(z, 50) if isinstance(z, str) else z) for z in flow_pages[i] ]
        else:
            flow_pages[i] = reduce_str(flow_pages[i], 50)

    print_debug(f"[INFO] Final results: {results}", debug)
    return results

# ============ Transformations ============
def transform_graph(data: dict) -> dict:
    raw_nodes = data.get("nodes", [])
    raw_links = data.get("links", [])

    nodes_map = {n['id']: n for n in raw_nodes}
    engine_flow = {}

    for nid, n in nodes_map.items():
        node_type = n.get("type", "unknown")
        config = n.get("config", {}).copy()
        
        if node_type == "sardine":
            if "documentTypes" in config:
                config["accepted_files"] = config["documentTypes"]
        
        elif node_type == "agent":
            if "agentName" in config:
                config["model"] = config["agentName"]
        
        elif node_type == "agent-group":
            child_ids = config.get("ids", [])
            agents_list = []
            for child_id in child_ids:
                child_node = nodes_map.get(child_id)
                if child_node:
                    child_cfg = child_node.get("config", {})
                    agents_list.append({
                        "model": child_cfg.get("agentName", ""),
                        "version": child_cfg.get("version", "")
                    })
            config["agents"] = agents_list

        engine_flow[nid] = {
            "type": node_type,
            "config": config,
            "outputs": {},
            "inputs": [],
            "status": "pending"
        }

    for link in raw_links:
        src_id = link['src']['nodeId']
        dst_id = link['dst']['nodeId']
        
        if src_id not in engine_flow or dst_id not in engine_flow:
            continue

        if src_id not in engine_flow[dst_id]["inputs"]:
            engine_flow[dst_id]["inputs"].append(src_id)

        src_raw = nodes_map[src_id]
        port_index = link['src'].get('portIndex', 0)
        output_name = "base"

        # 1. Priorité au nom défini dans le nœud (dynamique)
        if "outputs" in src_raw and isinstance(src_raw["outputs"], list):
            if port_index < len(src_raw["outputs"]):
                p_name = src_raw["outputs"][port_index].get("name")
                if p_name:
                    output_name = p_name
        
        # 2. Fallbacks statiques si pas de nom explicite
        src_type = src_raw.get("type")
        if output_name == "base": 
            if src_type == "sardine":
                output_name = "valide" if port_index == 0 else "invalid"
            elif src_type == "if":
                output_name = "true" if port_index == 0 else "false"
            elif src_type == "zone-detection":
                output_name = "base"

        if output_name not in engine_flow[src_id]["outputs"]:
            engine_flow[src_id]["outputs"][output_name] = []
        
        engine_flow[src_id]["outputs"][output_name].append(dst_id)

    return engine_flow

if __name__ == "__main__":
    pass