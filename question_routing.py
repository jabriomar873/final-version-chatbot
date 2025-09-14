"""Question classification and generalized handling utilities.

This module centralizes heuristics so the main app logic stays cleaner and
future extensions (adding new question types) only require edits here.

We keep everything lightweight (regex / string patterns) to avoid adding
external dependencies while still being more general than ad‑hoc inline code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class QuestionClassification:
    raw: str
    lang: str  # 'fr' or 'en'
    # Primary intent flags
    is_summary: bool = False
    is_phase_count: bool = False
    is_phase_list: bool = False
    is_subject: bool = False
    is_subject_all: bool = False
    is_actors: bool = False
    is_critical_phase: bool = False
    is_generic_count: bool = False  # e.g. "how many X" where X not phases
    is_risks: bool = False
    is_kpis: bool = False
    generic_count_target: Optional[str] = None
    # Fallback generic doc analysis
    is_document_analysis: bool = True
    # Rationale / matched patterns for debugging
    debug_matches: Dict[str, List[str]] = field(default_factory=dict)

    def add_debug(self, key: str, pattern: str):
        self.debug_matches.setdefault(key, []).append(pattern)


# ---------------------------------------------------------------------------
# Language aware helper patterns
# ---------------------------------------------------------------------------
SUMMARY_PATTERNS = [
    r"\bsummar(y|ise|ize)\b", r"\brésum[ée]r?\b", r"\brésume\b", r"\bsynth[èe]se\b"
]

PHASE_COUNT_PATTERNS = [
    r"how many\s+phases?", r"number of\s+phases?", r"combien de\s+phases?",
    r"nombre de\s+phases?", r"nbr de\s+phases?", r"combien d[' ]?étapes?", r"how many\s+stages?",
]

PHASE_LIST_PATTERNS = [
    r"list(e)?\s+of\s+phases", r"toutes?\s+les\s+phases", r"liste\s+des\s+phases",
    r"all\s+phases", r"phases?\s+principales?"
]

SUBJECT_PATTERNS = [
    r"\bsubject\b",
    r"\bobjet\b",
    r"\bsujet\b",
    r"\bth[eè]me\b",
    r"\btitre\b",
    r"what is the subject",
    r"objet du document",
    r"quel est le sujet",
    r"quel est l'objet",
    r"quel est le theme",
    r"quel est le th[eè]me",
    r"\bscope\b",
    r"\bpurpose\b",
    r"\btopic\b",
    r"intitul[ée]"
]

SUBJECT_ALL_HINTS = [
    r"all documents", r"tous les documents", r"tous les pdf", r"each document",
    r"every document", r"pour chaque document"
]

ACTORS_PATTERNS = [
    r"\bactors?\b", r"\bacteurs?\b", r"intervenants", r"stakeholders", r"team", r"équipe"
]

CRITICAL_PHASE_PATTERNS = [
    r"most important phase", r"critical phase", r"phase critique",
    r"étape la plus importante"
]

RISKS_PATTERNS = [
    r"\brisks?\b", r"\brisques?\b", r"\bmitigation[s]?\b", r"\batténuation\b", r"\bmesures?\b"
]

KPIS_PATTERNS = [
    r"\bkpis?\b", r"\bindicateurs?\b", r"\bmetrics?\b", r"\bperformance indicator\b", r"\bindicator[s]?\b"
]

GENERIC_COUNT_PREFIX = [
    r"how many\s+([a-zA-Zéèàùç0-9 _\-]{2,50})",  # English
    r"combien de\s+([a-zA-Zéèàùç0-9 _\-]{2,50})",  # French
    r"nombre de\s+([a-zA-Zéèàùç0-9 _\-]{2,50})",   # French variant
]


def _lang(text: str) -> str:
    low = f" {text.lower()} "
    fr_score = sum(1 for t in [" le ", " la ", " les ", " des ", " et ", " que ", " pour "] if t in low)
    en_score = sum(1 for t in [" the ", " and ", " is ", " of ", " for ", " with "] if t in low)
    return 'fr' if fr_score >= en_score else 'en'


def classify_question(question: str) -> QuestionClassification:
    q = question.strip()
    low = q.lower()
    lang = _lang(q)
    qc = QuestionClassification(raw=q, lang=lang)

    # Summary detection
    for pat in SUMMARY_PATTERNS:
        if re.search(pat, low):
            qc.is_summary = True
            qc.add_debug('summary', pat)
            break

    # Phase count
    for pat in PHASE_COUNT_PATTERNS:
        if re.search(pat, low):
            qc.is_phase_count = True
            qc.add_debug('phase_count', pat)
            break

    # Phase list (explicit listing request)
    for pat in PHASE_LIST_PATTERNS:
        if re.search(pat, low):
            qc.is_phase_list = True
            qc.add_debug('phase_list', pat)
            break

    # Subject
    for pat in SUBJECT_PATTERNS:
        if re.search(pat, low):
            qc.is_subject = True
            qc.add_debug('subject', pat)
            break
    if qc.is_subject:
        for hint in SUBJECT_ALL_HINTS:
            if hint in low:
                qc.is_subject_all = True
                qc.add_debug('subject_all', hint)
                break

    # Actors
    for pat in ACTORS_PATTERNS:
        if re.search(pat, low):
            qc.is_actors = True
            qc.add_debug('actors', pat)
            break

    # Critical phase
    for pat in CRITICAL_PHASE_PATTERNS:
        if re.search(pat, low):
            qc.is_critical_phase = True
            qc.add_debug('critical_phase', pat)
            break

    # Risks
    for pat in RISKS_PATTERNS:
        if re.search(pat, low):
            qc.is_risks = True
            qc.add_debug('risks', pat)
            break

    # KPIs / Metrics
    for pat in KPIS_PATTERNS:
        if re.search(pat, low):
            qc.is_kpis = True
            qc.add_debug('kpis', pat)
            break

    # Generic count (only if not already a phase count)
    if not qc.is_phase_count:
        for pat in GENERIC_COUNT_PREFIX:
            m = re.search(pat, low)
            if m:
                target = m.group(1).strip()
                # Remove trailing question marks/punctuation
                target = re.sub(r"[?.,;:]+$", "", target)
                # Ignore if target refers to phases/etapes already
                if not re.search(r"phase|étape|etape|stage", target):
                    qc.is_generic_count = True
                    qc.generic_count_target = target
                    qc.add_debug('generic_count', pat)
                break

    # If no explicit flags matched, attempt semantic enrichment to recognize intent beyond regex
    if not any([qc.is_summary, qc.is_phase_count, qc.is_phase_list, qc.is_subject, qc.is_actors, qc.is_critical_phase, qc.is_risks, qc.is_kpis]):
        try:
            cat_map = {
                'summary': [
                    'provide a summary', 'résume le document', 'give overall synopsis', 'synthèse globale'
                ],
                'phase_count': [
                    'total phases project', 'how many distinct phases', 'nombre exact de phases'
                ],
                'subject': [
                    'main subject project', 'core topic document', 'objet principal'
                ],
                'actors': [
                    'stakeholder list', 'team members involved', 'principaux acteurs'
                ],
                'critical_phase': [
                    'most important project step', 'phase critique', 'critical stage'
                ]
            }
            # Lightweight embedding using simple bag-of-words hashing (no external model) as fallback heuristic
            def tokenize(t: str) -> List[str]:
                return [w for w in re.split(r"[^a-zA-Zéèàùç0-9]+", t.lower()) if w]
            q_tokens = tokenize(low)
            if q_tokens:
                def jaccard(a: List[str], b: List[str]) -> float:
                    sa, sb = set(a), set(b)
                    if not sa or not sb: return 0.0
                    inter = len(sa & sb); uni = len(sa | sb)
                    return inter / uni
                scores: List[Tuple[str, float]] = []
                for label, samples in cat_map.items():
                    local_best = 0.0
                    for s in samples:
                        sc = jaccard(q_tokens, tokenize(s))
                        if sc > local_best:
                            local_best = sc
                    if local_best > 0:
                        scores.append((label, local_best))
                if scores:
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top_label, top_score = scores[0]
                    # Threshold tuned heuristically; accept if moderate overlap
                    if top_score >= 0.28:  # moderate semantic similarity threshold
                        if top_label == 'summary':
                            qc.is_summary = True; qc.add_debug('semantic_summary', f"score={top_score:.2f}")
                        elif top_label == 'phase_count':
                            qc.is_phase_count = True; qc.add_debug('semantic_phase_count', f"score={top_score:.2f}")
                        elif top_label == 'subject':
                            qc.is_subject = True; qc.add_debug('semantic_subject', f"score={top_score:.2f}")
                        elif top_label == 'actors':
                            qc.is_actors = True; qc.add_debug('semantic_actors', f"score={top_score:.2f}")
                        elif top_label == 'critical_phase':
                            qc.is_critical_phase = True; qc.add_debug('semantic_critical_phase', f"score={top_score:.2f}")
        except Exception:
            pass

    # Document analysis scope: any of the flags above implies doc analysis; keep default True
    return qc


# ---------------------------------------------------------------------------
# Generic counting logic
# ---------------------------------------------------------------------------
def generic_count_occurrences(target: str, all_docs: List[Any], max_docs: int = 60) -> Dict[str, Any]:
    """Count approximate occurrences of a target noun phrase across all documents.

    Returns a dict with keys: count, docs_considered, sample_evidence (list[str]).
    This is heuristic (simple substring / token based) but provides a general answer
    for many 'how many X' questions without bespoke code each time.
    """
    if not target or not all_docs:
        return {"count": 0, "docs_considered": 0, "sample_evidence": []}
    phrase = target.lower().strip()
    # Basic normalization (singular/plural naive strip of trailing s/es)
    singular = re.sub(r"(es|s)$", "", phrase)
    evidence = []
    total = 0
    for d in all_docs[:max_docs]:
        text = d.page_content
        low = text.lower()
        # Count occurrences of phrase or singular form
        hits = low.count(phrase)
        if hits == 0 and singular != phrase:
            hits = low.count(singular)
        if hits > 0:
            total += hits
            # Capture first evidence line
            for line in text.splitlines():
                ll = line.lower()
                if phrase in ll or (singular != phrase and singular in ll):
                    snippet = line.strip()
                    if len(snippet) > 280:
                        snippet = snippet[:277] + '...'
                    evidence.append(snippet)
                    break
        if len(evidence) >= 6:  # enough evidence samples
            break
    return {"count": total, "docs_considered": min(len(all_docs), max_docs), "sample_evidence": evidence}


__all__ = [
    'QuestionClassification', 'classify_question', 'generic_count_occurrences'
]
