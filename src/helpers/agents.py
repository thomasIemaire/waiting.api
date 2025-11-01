import os
import re
import threading
from bson import ObjectId
from functools import lru_cache
from typing import Dict, Any, Iterable, List, Mapping

from pymongo import MongoClient
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)

from src.helpers import utils  # on garde utils pour _pyify (et autres utilitaires éventuels)
# from your_module import best_entities  # <-- décommentez si vous l’avez ailleurs

_MODEL_LOAD_LOCK = threading.Lock()

# ---------- DB ----------
def get_db():
    """
    Retourne l'instance de base Mongo à partir des variables d'environnement :
    - MONGO_URI (ex: mongodb://localhost:27017)
    - MONGO_DB  (ex: my_database)
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI est manquant dans l'environnement.")

    client = MongoClient(mongo_uri)

    try:
        return client.get_default_database()
    except Exception:
        return None

db = get_db()

# ---------- NLP ----------
@lru_cache(maxsize=32)
def get_token_classifier(model_dir: str) -> Pipeline:
    """
    Charge un pipeline token-classification en CPU, en évitant le device 'meta'.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Répertoire modèle introuvable : {model_dir}")

    with _MODEL_LOAD_LOCK:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            low_cpu_mem_usage=False,  # <-- force des tenseurs réels (pas 'meta')
            device_map=None,          # <-- évite accelerate/'auto' qui place sur 'meta'
            trust_remote_code=False,  # mets True si ton modèle custom le requiert
        )

    return pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU
    )

# ---------- Utils ----------
def clean_text(text: str) -> str:
    # Remplace \r/\n par espaces et compacte les espaces multiples
    text = text.replace("\r", " ").replace("\n", " ")
    # text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- Public API ----------
def run(text: str, *, reference: str, version: str):
    """
    Exécute l'agent NER pour (reference, version) :
    - récupère la config 'agent' en base (optionnel, mais utile pour vérifier l'existence)
    - charge le pipeline depuis le répertoire sardine.trainer/sardine.agents/{reference}/{version}
    - nettoie le texte, passe le NER, puis _pyify le résultat
    - applique best_entities si disponible, sinon renvoie les entités telles quelles
    """
    # 1) Vérifier que l'agent existe (si vous stockez la config en base)
    agent = db["agents"].find_one({"reference": reference, "version": version})
    if not agent:
        print(f"[WARN] Agent introuvable : {reference} v{version}")

    # 2) Construire le chemin modèle
    path = f"sardine.trainer/sardine.agents/{reference}/{version}"
    model_dir = os.path.normpath(os.path.join("../../workspace.sardine", path))

    # 3) Charger le pipeline (caché)
    nlp = get_token_classifier(model_dir)

    # 4) NER
    text_clean = clean_text(text)
    raw_entities = nlp(text_clean)
    mapper = agent.get("mapper", {}) if agent else {}

    # 5) Post-traitements
    entities = utils._pyify(raw_entities)

    for ent in entities:
        start, end = ent.get("start"), ent.get("end")
        ent["word"] = text_clean[start:end]
        if ent["score"] < .8:
            entities.remove(ent) 

    # 6) Agrégation "best"
    return best_entities(entities, reqs=agent.get("requirements", [])) if agent else entities, mapper

def best_entities(entities: List[Dict[str, Any]], reqs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Exemple de fonction d'agrégation des entités.
    Vous pouvez la modifier ou la remplacer par votre propre logique.
    Ici, on garde pour chaque type d'entité la plus longue (en caractères).
    """
    best = {}

    for ent in entities:
        label, score, word = ent.get("entity_group"), ent.get("score", 0), ent.get("word", "")
        respect, value = check_requirements(word, reqs[label])
        if (label not in best or best[label]["score"] < score) and respect:
            ent["word"] = value
            best[label] = ent

    return best

def check_requirements(value: Any, requirements: Iterable[Mapping[str, Any]]) -> bool:
    for requirement in requirements or []:
        rule = requirement.get("rule", "")
        constraint = requirement.get("constraint", "")

        if rule == "regex":
            match = re.search(str(constraint), str(value))
            if match:
                value = match.group(0)

        try:
            if rule == "regex" and not re.match(str(constraint), str(value)):
                return False, value
            elif rule == "eq" and str(value) != str(constraint):
                return False, value
            elif rule == "neq" and str(value) == str(constraint):
                return False, value
            elif rule == "gt" and float(value) <= float(constraint):
                return False, value
            elif rule == "lt" and float(value) >= float(constraint):
                return False, value
            elif rule == "gte" and float(value) < float(constraint):
                return False, value
            elif rule == "lte" and float(value) > float(constraint):
                return False, value
            elif rule == "in" and str(value) not in split_constraint(constraint):
                return False, value
            elif rule == "nin" and str(value) in split_constraint(constraint):
                return False, value
            elif rule == "contains" and str(constraint) not in str(value):
                return False, value
            elif rule == "ncontains" and str(constraint) in str(value):
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