import os
import re
import threading
import json
from functools import lru_cache
from typing import Dict, Any, Iterable, List, Mapping

from pymongo import MongoClient
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)

from src.helpers import utils  # On garde utils pour _pyify

# ===================== CACHE MANUEL =====================
# Dictionnaire pour stocker les pipelines chargés en mémoire vive
_PIPELINE_CACHE: Dict[str, Pipeline] = {}
# Verrou pour empêcher le chargement simultané du même modèle par plusieurs threads
_MODEL_LOAD_LOCK = threading.Lock()

# ---------- DB ----------
def get_db():
    """
    Retourne l'instance de base Mongo à partir des variables d'environnement.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        # On lève une erreur comme dans le code original pour ne pas masquer le problème
        raise RuntimeError("MONGO_URI est manquant dans l'environnement.")

    client = MongoClient(mongo_uri)

    try:
        return client.get_default_database()
    except Exception:
        return None

db = get_db()

# ---------- NLP (AVEC CACHE FIXE) ----------
def get_token_classifier(model_dir: str) -> Pipeline:
    """
    Charge un pipeline token-classification en CPU, en utilisant un cache dictionnaire global.
    """
    # 1. Vérification rapide dans le cache
    if model_dir in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_dir]

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Répertoire modèle introuvable : {model_dir}")

    # 2. Chargement avec verrou
    with _MODEL_LOAD_LOCK:
        # On vérifie à nouveau une fois le verrou acquis (double-check locking)
        if model_dir in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[model_dir]

        print(f"[INFO] Chargement du modèle en mémoire : {model_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            low_cpu_mem_usage=False,  # Force tenseurs réels
            device_map=None,          # Évite 'meta' device
            trust_remote_code=False,
        )

        nlp_pipe = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,  # CPU
        )

        # Enregistrement dans le cache
        _PIPELINE_CACHE[model_dir] = nlp_pipe
        return nlp_pipe

# ---------- Utils ----------
def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    return text.strip()

# ---------- Public API ----------
def run(text: str, *, reference: str, version: str):
    """
    Exécute l'agent NER pour (reference, version).
    """
    # 1) Vérifier que l'agent existe
    # Note: On utilise db[...] directement comme dans l'original pour lever l'erreur si db est None
    agent = db["agents"].find_one({"reference": reference, "version": version})
    if not agent:
        print(f"[WARN] Agent introuvable : {reference} v{version}")

    # 2) Construire le chemin modèle
    # Attention : le ".." dépend d'où le script est lancé. 
    # Si le code original fonctionnait, on garde cette logique relative.
    path = f"sardine.agents/{reference}/{version}"
    model_dir = os.path.normpath(os.path.join("..", path))

    # 3) Charger le pipeline (via notre nouvelle fonction avec Cache Dictionnaire)
    nlp = get_token_classifier(model_dir)

    # 4) NER
    text_clean = clean_text(text)
    raw_entities = nlp(text_clean)
    mapper = agent.get("mapper", {}) if agent else {}

    # 5) Post-traitements
    entities = utils._pyify(raw_entities)

    # Filtrage et nettoyage (logique originale)
    entities_to_keep = []
    for ent in entities:
        start, end = ent.get("start"), ent.get("end")
        ent["word"] = text_clean[start:end]
        if ent.get("score", 0) >= 0.8:  # Correction: >= au lieu de < avec remove (plus sûr)
            entities_to_keep.append(ent)
    
    entities = entities_to_keep

    # 6) Agrégation "best"
    # On s'assure de passer 'requirements' correctement
    reqs = agent.get("requirements", []) if agent else {}
    return best_entities(entities, reqs=reqs), mapper

def best_entities(entities: List[Dict[str, Any]], reqs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Logique d'agrégation.
    """
    best = {}

    # Conversion de reqs en dict si ce n'est pas déjà le cas (sécurité)
    # Le code original faisait reqs[label], supposant un dict { "LABEL": [rules] }
    reqs_map = reqs
    if not isinstance(reqs, dict):
        # Si reqs est une liste, on ne pourra pas faire reqs[label] facilement sans transformation
        # On suppose ici que la structure en DB est bien un Dict/Map
        pass

    for ent in entities:
        label = ent.get("entity_group")
        score = ent.get("score", 0)
        word = ent.get("word", "")

        # Récupération safe des règles pour ce label
        # Utilisation de .get() pour éviter KeyError si le label n'est pas dans les requirements
        label_reqs = reqs_map.get(label, []) if isinstance(reqs_map, dict) else []

        respect, value = check_requirements(word, label_reqs)
        
        if respect:
            if (label not in best or best[label]["score"] < score):
                ent["word"] = value
                best[label] = ent

    return best

def check_requirements(value: Any, requirements: Iterable[Mapping[str, Any]]) -> tuple[bool, Any]:
    for requirement in requirements or []:
        rule = requirement.get("rule", "")
        constraint = requirement.get("constraint", "")

        if rule == "regex":
            match = re.search(str(constraint), str(value))
            if match:
                value = match.group(0)
            # Si regex ne match pas, faut-il rejeter ? 
            # Le code original continuait. Ajoutons le check 'match' dans le try/except suivant si besoin.

        try:
            val_str = str(value)
            con_str = str(constraint)
            
            # Regex check négatif explicite
            if rule == "regex" and not re.search(str(constraint), str(value)):
                return False, value

            if rule == "eq" and val_str != con_str:
                return False, value
            elif rule == "neq" and val_str == con_str:
                return False, value
            
            # Comparaisons numériques
            if rule in ["gt", "lt", "gte", "lte"]:
                v_f = float(value)
                c_f = float(constraint)
                if rule == "gt" and v_f <= c_f: return False, value
                if rule == "lt" and v_f >= c_f: return False, value
                if rule == "gte" and v_f < c_f: return False, value
                if rule == "lte" and v_f > c_f: return False, value

            elif rule == "in" and val_str not in split_constraint(constraint):
                return False, value
            elif rule == "nin" and val_str in split_constraint(constraint):
                return False, value
            elif rule == "contains" and con_str not in val_str:
                return False, value
            elif rule == "ncontains" and con_str in val_str:
                return False, value
        except Exception:
            # En cas d'erreur de conversion (ex: float sur du texte), on rejette
            return False, value
        
    return True, value

def split_constraint(constraint: Any) -> list[str]:
    if isinstance(constraint, str):
        return [part.strip() for part in constraint.split(",") if part.strip()]
    if isinstance(constraint, Iterable):
        return [str(item) for item in constraint]
    return [str(constraint)]