import os
import re
import threading
from functools import lru_cache
from typing import Dict, Any, Iterable, List, Mapping

from pymongo import MongoClient
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)

from src.helpers import utils

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
@lru_cache(maxsize=32)
def _get_agent_config(reference: str, version: str) -> dict | None:
    if db is None: return None
    if version == "latest":
        cursor = db["agents"].find({"reference": reference}).sort("version", -1).limit(1)
        lst = list(cursor)
        if lst:
            print(f"[INFO] 'latest' resolved to v{lst[0].get('version')} for agent {reference}")
            return lst[0]
    else:
        return db["agents"].find_one({"reference": reference, "version": version})
    return None

# ---------- NLP ----------
@lru_cache(maxsize=32)
def get_token_classifier(model_dir: str) -> Pipeline:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Répertoire modèle introuvable : {model_dir}")

    with _MODEL_LOAD_LOCK:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir, local_files_only=True, low_cpu_mem_usage=False, device_map=None
        )

    return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)

# ---------- Utils ----------
def clean_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").strip()

# ---------- Public API ----------
def run(text: str, *, reference: str, version: str):
    agent = _get_agent_config(reference, version)
    
    # Détermination du chemin
    if agent and agent.get("path"):
        model_dir = agent["path"]
    else:
        # Fallback relatif
        real_ver = agent.get("version", version) if agent else version
        model_dir = os.path.normpath(os.path.join("..", f"sardine.agents/{reference}/{real_ver}"))

    # Chargement
    try:
        nlp = get_token_classifier(model_dir)
    except Exception as e:
        print(f"[ERROR] Erreur modèle {reference}: {e}")
        return {}, {}

    text_clean = clean_text(text)
    if not text_clean: return {}, {}

    try:
        raw_entities = nlp(text_clean)
    except Exception as e:
        print(f"[ERROR] Erreur inférence {reference}: {e}")
        return {}, {}

    mapper = agent.get("mapper", {}) if agent else {}
    print(f"[INFO] Agent '{reference}' v{version} - mapper: {mapper}")

    entities = utils._pyify(raw_entities)

    print(f"[DEBUG] text_clean: {text_clean}")
    
    valid_entities = []
    for ent in entities:
        if ent.get("score", 0) >= 0.5: # Seuil légèrement baissé par sécurité
            start, end = ent.get("start"), ent.get("end")
            ent["word"] = text_clean[start:end]
            valid_entities.append(ent)

    reqs = agent.get("requirements", []) if agent else []
    best_entities_result = best_entities(valid_entities, reqs)

    if len(best_entities_result) > 0:
        print(f"[INFO] Agent '{reference}' v{version} - Entities found: {list(best_entities_result.keys())}")

    return best_entities_result, mapper

def best_entities(entities: List[Dict[str, Any]], reqs: Any) -> Dict[str, Any]:
    best = {}
    requirements_map = reqs if isinstance(reqs, dict) else {}

    # Conserve la meilleure entité globale par label, même si les requirements échouent
    best_overall = {}

    for ent in entities:
        label = ent.get("entity_group")
        score = ent.get("score", 0)
        word = ent.get("word", "")

        specific_reqs = requirements_map.get(label, []) if requirements_map else []
        respect, value = check_requirements(word, specific_reqs)

        # Mise à jour du meilleur score brut (fallback si aucune entité ne passe les contraintes)
        if label not in best_overall or score > best_overall[label]["score"]:
            ent_any = ent.copy()
            ent_any["word"] = value if respect else word
            best_overall[label] = ent_any

        if respect:
            if label not in best or score > best[label]["score"]:
                ent_valid = ent.copy()
                ent_valid["word"] = value
                best[label] = ent_valid

    # Si aucune entité ne satisfait les requirements, on renvoie la plus pertinente trouvée
    for label, fallback_ent in best_overall.items():
        if label not in best:
            best[label] = fallback_ent

    return best

def check_requirements(value: Any, requirements: Iterable[Mapping[str, Any]]) -> bool:
    if not requirements: return True, value
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
                if rule == "gt" and f_val <= f_const: return False, value
                if rule == "lt" and f_val >= f_const: return False, value
                if rule == "gte" and f_val < f_const: return False, value
                if rule == "lte" and f_val > f_const: return False, value
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