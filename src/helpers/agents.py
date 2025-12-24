import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pymongo import MongoClient

from src.helpers import utils

# HF (compat)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)

_MODEL_LOAD_LOCK = threading.Lock()

# ---------- DB ----------
def get_db():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("[WARN] MONGO_URI manquant.")
        return None
    try:
        client = MongoClient(mongo_uri)
        return client.get_default_database()
    except Exception:
        return None


db = get_db()

# ---------- CONFIG CACHE ----------
def _get_agent_config(reference: str, version: str) -> dict | None:
    if db is None:
        return None

    if version == "latest":
        cursor = db["agents"].find({"reference": reference}).sort("version", -1).limit(1)
        lst = list(cursor)
        if lst:
            print(f"[INFO] 'latest' resolved to v{lst[0].get('version')} for agent {reference}")
            return lst[0]
        return None

    return db["agents"].find_one({"reference": reference, "version": version})


# ---------- Utils ----------
def clean_text(text: str) -> str:
    return (text or "").replace("\r", " ").replace("\n", " ").strip()


def _as_path(model_dir: str) -> Path:
    return Path(model_dir).expanduser().resolve()


def _detect_gliner(model_dir: Path, agent: Optional[dict]) -> bool:
    if (model_dir / "gliner_config.json").exists():
        return True
    base = (agent or {}).get("base_model")
    if isinstance(base, str) and base.lower().startswith("gliner"):
        return True
    # optionnel: un champ explicite si tu l’ajoutes
    engine = (agent or {}).get("engine")
    if isinstance(engine, str) and engine.lower() == "gliner":
        return True
    return False


def _get_device_str() -> str:
    """
    CPU par défaut.
    Override possible via env: AGENTS_DEVICE=cpu|cuda|mps
    """
    dev = (os.getenv("AGENTS_DEVICE") or "cpu").strip().lower()
    if dev not in ("cpu", "cuda", "mps"):
        dev = "cpu"
    return dev


def _infer_labels(agent: Optional[dict]) -> List[str]:
    """
    GLiNER a BESOIN d'une liste de labels à l'inférence.
    On essaie plusieurs sources côté DB (agent).
    """
    if not agent:
        return []

    # 1) champ direct "labels" (recommandé)
    labels = agent.get("labels")
    if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
        return [x.strip() for x in labels if x and x.strip()]

    # 2) champ "vocabulary" (si tu t’en sers pour ça)
    vocab = agent.get("vocabulary")
    if isinstance(vocab, list):
        out = [str(x).strip() for x in vocab if str(x).strip()]
        if out:
            return out

    # 3) keys des requirements si c'est un dict {LABEL: [...]}
    reqs = agent.get("requirements")
    if isinstance(reqs, dict):
        keys = [str(k).strip() for k in reqs.keys() if str(k).strip()]
        if keys:
            return keys

    # 4) keys du mapper (souvent proche des champs attendus)
    mapper = agent.get("mapper")
    if isinstance(mapper, dict):
        keys = [str(k).strip() for k in mapper.keys() if str(k).strip()]
        if keys:
            return keys

    return []


def _get_threshold(agent: Optional[dict]) -> float:
    thr = None
    if agent:
        thr = agent.get("confidence_threshold")
        if thr is None:
            # fallback si tu stockes "threshold" ailleurs
            thr = agent.get("threshold")
    try:
        thr_f = float(thr) if thr is not None else 0.5
    except Exception:
        thr_f = 0.5
    return max(0.0, min(1.0, thr_f))


# ---------- HF Token-classification (compat) ----------
@lru_cache(maxsize=32)
def get_token_classifier(model_dir: str, device_id: int) -> Pipeline:
    md = _as_path(model_dir)
    if not md.is_dir():
        raise FileNotFoundError(f"Répertoire modèle introuvable : {md}")

    with _MODEL_LOAD_LOCK:
        tokenizer = AutoTokenizer.from_pretrained(str(md), local_files_only=True, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            str(md),
            local_files_only=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )

    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_id,
    )


def _infer_hf(model_dir: str, text: str, threshold: float) -> List[Dict[str, Any]]:
    try:
        device_id = -1  # CPU par défaut
        # (optionnel) si tu veux autoriser cuda pour HF aussi :
        # if _get_device_str() == "cuda": device_id = 0

        nlp = get_token_classifier(model_dir, device_id=device_id)
        raw_entities = nlp(text)

        ents = utils._pyify(raw_entities)
        out: List[Dict[str, Any]] = []
        for e in ents:
            score = float(e.get("score", 0.0))
            if score < threshold:
                continue
            start, end = e.get("start"), e.get("end")
            if start is None or end is None:
                continue
            # normalisation
            out.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "entity_group": e.get("entity_group") or e.get("entity") or "UNK",
                    "score": score,
                    "word": text[int(start) : int(end)],
                }
            )
        return out
    except Exception as e:
        print(f"[ERROR] Erreur inférence HF {model_dir}: {e}")
        return []


# ---------- GLiNER ----------
@lru_cache(maxsize=16)
def get_gliner_model(model_dir: str, device_str: str):
    """
    Cache GLiNER par (model_dir, device_str)
    """
    md = _as_path(model_dir)
    if not md.is_dir():
        raise FileNotFoundError(f"Répertoire modèle introuvable : {md}")

    try:
        from gliner import GLiNER
    except ImportError as e:
        raise RuntimeError("GLiNER non installé. Fais: pip install gliner") from e

    with _MODEL_LOAD_LOCK:
        model = GLiNER.from_pretrained(str(md))

    # to(device) accepte souvent "cpu"/"cuda"/"mps" sous forme str
    try:
        model.to(device_str)
    except Exception:
        model.to("cpu")

    return model


def _infer_gliner(model_dir: str, text: str, labels: List[str], threshold: float) -> List[Dict[str, Any]]:
    if not labels:
        print("[WARN] GLiNER: liste de labels vide => aucune extraction.")
        return []

    try:
        device_str = _get_device_str()
        model = get_gliner_model(model_dir, device_str=device_str)

        ents = model.predict_entities(text, labels, threshold=threshold)
        # ents: [{start, end, text, label, score}, ...]
        out: List[Dict[str, Any]] = []
        for e in ents or []:
            try:
                start = int(e["start"])
                end = int(e["end"])
                score = float(e.get("score", 0.0))
                out.append(
                    {
                        "start": start,
                        "end": end,
                        "entity_group": e.get("label") or "UNK",
                        "score": score,
                        "word": text[start:end],
                    }
                )
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[ERROR] Erreur inférence GLiNER {model_dir}: {e}")
        return []


# ---------- Public API ----------
def run(text: str, *, reference: str, version: str):
    agent = _get_agent_config(reference, version)

    # Détermination du chemin
    if agent and agent.get("path"):
        model_dir = str(_as_path(agent["path"]))
    else:
        real_ver = agent.get("version", version) if agent else version
        model_dir = str((Path("..") / f"sardine.agents/{reference}/{real_ver}").resolve())

    text_clean = clean_text(text)
    if not text_clean:
        return {}, {}

    threshold = _get_threshold(agent)

    # Inference dispatch
    md = _as_path(model_dir)
    is_gliner = _detect_gliner(md, agent)

    if is_gliner:
        labels = _infer_labels(agent)
        entities = _infer_gliner(model_dir, text_clean, labels=labels, threshold=threshold)
        print(f"[INFO] Agent '{reference}' v{version} (GLiNER) - thr={threshold} labels={labels}")
    else:
        entities = _infer_hf(model_dir, text_clean, threshold=threshold)
        print(f"[INFO] Agent '{reference}' v{version} (HF token-cls) - thr={threshold}")

    mapper = agent.get("mapper", {}) if agent else {}
    reqs = agent.get("requirements", []) if agent else []
    best_entities_result = best_entities(entities, reqs)

    if best_entities_result:
        print(f"[INFO] Agent '{reference}' v{version} - Entities found: {list(best_entities_result.keys())}")

    return best_entities_result, mapper


def best_entities(entities: List[Dict[str, Any]], reqs: Any) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    requirements_map = reqs if isinstance(reqs, dict) else {}

    # Conserve la meilleure entité globale par label, même si les requirements échouent
    best_overall: Dict[str, Any] = {}

    for ent in entities:
        label = ent.get("entity_group")
        score = float(ent.get("score", 0.0))
        word = ent.get("word", "")

        specific_reqs = requirements_map.get(label, []) if requirements_map else []
        respect, value = check_requirements(word, specific_reqs)

        if label not in best_overall or score > float(best_overall[label].get("score", 0.0)):
            ent_any = ent.copy()
            ent_any["word"] = value if respect else word
            best_overall[label] = ent_any

        if respect:
            if label not in best or score > float(best[label].get("score", 0.0)):
                ent_valid = ent.copy()
                ent_valid["word"] = value
                best[label] = ent_valid

    for label, fallback_ent in best_overall.items():
        if label not in best:
            best[label] = fallback_ent

    return best


def check_requirements(value: Any, requirements: Iterable[Mapping[str, Any]]) -> Tuple[bool, Any]:
    if not requirements:
        return True, value

    str_val = str(value)

    for r in requirements:
        rule = r.get("rule")
        constraint = r.get("constraint")
        try:
            if rule == "regex":
                if not re.search(str(constraint), str_val):
                    return False, value
            elif rule == "eq" and str_val != str(constraint):
                return False, value
            elif rule == "neq" and str_val == str(constraint):
                return False, value
            elif rule in ["gt", "lt", "gte", "lte"]:
                f_val = float(value)
                f_const = float(constraint)
                if rule == "gt" and f_val <= f_const:
                    return False, value
                if rule == "lt" and f_val >= f_const:
                    return False, value
                if rule == "gte" and f_val < f_const:
                    return False, value
                if rule == "lte" and f_val > f_const:
                    return False, value
            elif rule == "in" and str_val not in split_constraint(constraint):
                return False, value
            elif rule == "nin" and str_val in split_constraint(constraint):
                return False, value
            elif rule == "len" and len(str_val) != int(constraint):
                return False, value
        except Exception:
            return False, value

    return True, value


def split_constraint(constraint: Any) -> list[str]:
    if isinstance(constraint, str):
        return [part.strip() for part in constraint.split(",") if part.strip()]
    if isinstance(constraint, Iterable):
        return [str(item) for item in constraint]
    return [str(constraint)]
