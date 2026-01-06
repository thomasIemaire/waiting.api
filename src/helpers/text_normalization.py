import re
import unicodedata
from collections import Counter
from urllib.parse import urlsplit

# Stopwords FR de base (à compléter)
FRENCH_STOPWORDS = {
    "le","la","les","de","des","du","d","l","un","une","et","a","au","aux","en","dans","sur","pour","par",
    "avec","sans","ce","cet","cette","ces","se","sa","son","ses","leur","leurs","nos","notre","vos","votre",
    "qui","que","quoi","dont","ou","où","ne","pas","plus","moins","si","comme","mais","or","ni","car",
    "t","n","no","n°","nr","numero","numéro","page","fr"
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _normalize_amount(raw: str, decimal: str = ".") -> str:
    s = raw.replace("\u00A0", " ").strip()
    m = re.search(r"([.,])(\d{1,6})$", s)
    if not m:
        return re.sub(r"\D", "", s)

    dec_part = m.group(2)
    int_part = s[:m.start(1)]
    int_digits = re.sub(r"\D", "", int_part)
    return f"{int_digits}{decimal}{dec_part}"

def _normalize_identifier(raw: str) -> str:
    s = raw.upper().replace("\u00A0", " ")
    return re.sub(r"[^A-Z0-9]", "", s)

def _normalize_email(raw: str) -> str:
    return raw.strip().lower()

def _normalize_url(raw: str) -> str:
    """
    Normalisation "généraliste":
    - ignore http/https
    - supprime www.
    - lowercase host
    - enlève query/fragment
    - enlève trailing slash
    Retour: host + path (si path != '/')
    """
    s = raw.strip()

    # enlève ponctuation terminale courante (factures / PDF)
    s = s.rstrip(".,;:!?)]]}\"'")

    # urlsplit a besoin d'un schéma pour bien remplir netloc
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        s_for_parse = "http://" + s
    else:
        s_for_parse = s

    parts = urlsplit(s_for_parse)
    host = parts.netloc.lower()

    # supprime userinfo éventuel (rare)
    if "@" in host:
        host = host.split("@", 1)[1]

    # supprime www.
    if host.startswith("www."):
        host = host[4:]

    # enlève ports par défaut
    host = re.sub(r":(80|443)$", "", host)

    path = parts.path or ""
    # normalise path
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""

    return f"{host}{path}"

def _flatten(x):
    if isinstance(x, (list, tuple)):
        for it in x:
            yield from _flatten(it)
    else:
        yield str(x)

def _mask_spans(text: str, spans):
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + (" " * (end - start)) + text[end:]
    return text

# Montants: 2 décimales, tolère séparateurs de milliers
_AMOUNT_PAT = re.compile(r"(?<![\d.])(?:\d{1,3}(?:[ \u00A0]\d{3})+|\d+)[.,]\d{2}(?!\.\d)")

# Emails
_EMAIL_PAT = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# URLs:
# 1) avec schéma ou www.
_URL_SCHEME_OR_WWW_PAT = re.compile(r"\b(?:https?://|www\.)[^\s<>()\"]+", re.IGNORECASE)

# 2) domaines "nus" (subdomain.tld[/...]) - évite les emails via lookbehind @
_DOMAIN_PAT = re.compile(
    r"(?<!@)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,})(?:/[^\s<>()\"]*)?\b",
    re.IGNORECASE
)

# IDs "préfixe lettres + chiffres" (TVA, IBAN FR..., etc.) avec séparateurs tolérés
_PREFIX_DIGIT_PAT = re.compile(r"\b[A-Z]{2,4}(?:[ \u00A0.\-]?\d){6,}\b")

# Tokens alphanum collés contenant lettres+chiffres (ex: AGRIFRPP833)
_ALPHANUM_TOKEN_PAT = re.compile(r"\b(?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]+\b")

# Longues séquences de chiffres (SIREN/SIRET/tél/doc/etc.), avec ou sans séparateurs, >= 8 chiffres
_DIGIT_SEQ_PAT = re.compile(r"(?<!\d)(?:\d[\d\s.\-]{6,}\d)(?!\d)")

_WORD_PAT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", re.UNICODE)

def count_entities(texts, stopwords=None, decimal=".", strip_accents=True):
    """
    texts: str ou liste (même imbriquée) de strings
    Retour: dict avec Counter: words / identifiers / amounts / urls / emails
    """
    stop = set(FRENCH_STOPWORDS if stopwords is None else stopwords)

    corpus = "\n".join(_flatten(texts)).replace("\u00A0", " ")

    # 0) Emails & URLs (on les extrait tôt + on masque pour éviter de polluer words/identifiers)
    emails = Counter()
    urls = Counter()
    spans_to_mask = []

    for m in _EMAIL_PAT.finditer(corpus):
        spans_to_mask.append(m.span())
        emails[_normalize_email(m.group(0))] += 1

    # URLs avec schéma / www
    for m in _URL_SCHEME_OR_WWW_PAT.finditer(corpus):
        spans_to_mask.append(m.span())
        urls[_normalize_url(m.group(0))] += 1

    # Domaines nus (ex: groupe-terresdusud.fr / sub.domain.tld/path)
    # On évite de recompter ceux déjà matchés par l'autre regex en masquant d'abord
    corpus_for_domains = _mask_spans(corpus, spans_to_mask)
    for m in _DOMAIN_PAT.finditer(corpus_for_domains):
        spans_to_mask.append(m.span())
        urls[_normalize_url(m.group(0))] += 1

    # 1) Montants
    amounts = Counter()
    amount_spans = []
    for m in _AMOUNT_PAT.finditer(corpus):
        amount_spans.append(m.span())
        amounts[_normalize_amount(m.group(0), decimal=decimal)] += 1

    # 2) Identifiants
    identifiers = Counter()
    id_spans = []

    for m in _PREFIX_DIGIT_PAT.finditer(corpus):
        id_spans.append(m.span())
        identifiers[_normalize_identifier(m.group(0))] += 1

    for m in _ALPHANUM_TOKEN_PAT.finditer(corpus):
        identifiers[_normalize_identifier(m.group(0))] += 1

    # On masque montants + IDs + urls/emails pour éviter de recompter leurs digits à part
    masked = _mask_spans(corpus, amount_spans + id_spans + spans_to_mask)

    for m in _DIGIT_SEQ_PAT.finditer(masked):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= 8:
            identifiers[digits] += 1

    # 3) Mots (sans stopwords) — sur un texte masqué urls/emails pour éviter "www", "http", etc.
    masked_for_words = _mask_spans(corpus, spans_to_mask)
    words = Counter()
    for m in _WORD_PAT.finditer(masked_for_words):
        w = m.group(0).lower()
        if strip_accents:
            w = _strip_accents(w)
        if w in stop:
            continue
        words[w] += 1

    return {
        "words": words,
        "identifiers": identifiers,
        "amounts": amounts,
        "urls": urls,
        "emails": emails,
    }
