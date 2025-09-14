"""Lightweight knowledge graph extraction utilities.

Extracts structured elements from document chunks (LangChain Document objects):
- Phases (already extracted elsewhere, we map into KG)
- Actors (reuse existing extractor if present)
- Risks & Mitigations (pattern-based)
- KPIs / Metrics (numbers with units & labels)

Designed to be dependency-light and fast.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any
from langchain_core.documents import Document

PHASE_LINE = re.compile(r"(?i)phase\s*0*(\d+)\s*[:\-–]?\s*([^\n\r]{2,120})")
RISK_HEADINGS = ["risk", "risque", "risks", "risques", "mitigation", "mitigation", "mitigations", "mitigations"]
RISK_INLINE = re.compile(r"(?i)(risque|risk)\s*[:\-]\s*([^\n]{3,140})")
MITIGATION_INLINE = re.compile(r"(?i)(mitigation|mesure|action)\s*[:\-]\s*([^\n]{3,140})")
# KPI pattern: capture label + numeric+unit; exclude isolated single letters or code fragments
KPI_PATTERN = re.compile(r"(?i)\b([A-Z][A-Za-z0-9 _\-/]{3,60})\b[^\n]{0,50}?\b(\d{1,4}(?:[\.,]\d+)?\s?(?:%|jours|day[s]?|semaine[s]?|mois|€|eur|euros|k€|k|m|units?))")
NUM_UNIT = re.compile(r"(?i)\b(\d{1,4}(?:[\.,]\d+)?)\s?(%|jours?|days?|semaine[s]?|mois|€|eur|euros|k€|k|m|units?)")


def _clean(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text.strip())


def extract_risks_and_mitigations(docs: List[Document], max_docs: int = 80) -> List[Dict[str, Any]]:
    results = []
    for d in docs[:max_docs]:
        meta = getattr(d, 'metadata', {}) or {}
        lines = d.page_content.splitlines()
        # Heading proximity approach
        capture_block = False
        current_risk = None
        for ln in lines:
            low = ln.lower().strip()
            if any(h in low for h in RISK_HEADINGS):
                capture_block = True
                continue
            if capture_block and not low:
                capture_block = False
                current_risk = None
                continue
            if capture_block:
                # Inline risk/mitigation detection
                rm = RISK_INLINE.search(ln)
                if rm:
                    current_risk = _clean(rm.group(2))
                    results.append({
                        'type': 'risk', 'risk': current_risk, 'mitigation': None,
                        'source': meta.get('source','doc'), 'page': meta.get('page','?'), 'evidence': _clean(ln)
                    })
                    continue
                mm = MITIGATION_INLINE.search(ln)
                if mm and current_risk:
                    # attach mitigation to latest risk without one
                    for r in reversed(results):
                        if r.get('type') == 'risk' and r.get('risk') == current_risk and r.get('mitigation') is None:
                            r['mitigation'] = _clean(mm.group(2))
                            r.setdefault('evidence_mitigation', _clean(ln))
                            break
        # Also look for single-line risk patterns outside blocks
        for m in RISK_INLINE.finditer(d.page_content):
            meta2 = getattr(d, 'metadata', {}) or {}
            candidate = _clean(m.group(2))
            if not any(r.get('risk') == candidate for r in results):
                results.append({'type':'risk','risk':candidate,'mitigation':None,'source':meta2.get('source','doc'),'page':meta2.get('page','?'),'evidence':candidate})
    # Deduplicate by (risk, source, page)
    seen = set(); uniq=[]
    for r in results:
        key=(r.get('risk'), r.get('source'), r.get('page'))
        if key in seen: continue
        seen.add(key); uniq.append(r)
    return uniq[:60]


def extract_kpis(docs: List[Document], max_docs: int = 80) -> List[Dict[str, Any]]:
    kpis = []
    for d in docs[:max_docs]:
        meta = getattr(d, 'metadata', {}) or {}
        for m in KPI_PATTERN.finditer(d.page_content):
            label = _clean(m.group(1))
            value = _clean(m.group(2))
            # Filter noise: very short labels or labels that look like codes (e.g., 'MBST 31/:' or lone 'M')
            if len(label) < 4:
                continue
            if re.fullmatch(r"[A-Z]{1,3}\d{0,3}", label):  # e.g., MB31, ABC2
                continue
            kpis.append({
                'label': label,
                'value': value,
                'source': meta.get('source','doc'),
                'page': meta.get('page','?'),
                'evidence': _clean(m.group(0)[:180])
            })
        # Generic numeric+unit lines
        for ln in d.page_content.splitlines():
            if NUM_UNIT.search(ln):
                snippet = _clean(ln)[:180]
                # Skip lines that are mostly punctuation or a single code token
                if len(snippet.split()) < 2:
                    continue
                kpis.append({'label': None, 'value': snippet, 'source': meta.get('source','doc'), 'page': meta.get('page','?' ), 'evidence': snippet})
    # Deduplicate by full value+source+page
    seen=set(); final=[]
    for k in kpis:
        key=(k.get('label'), k.get('value'), k.get('source'), k.get('page'))
        if key in seen: continue
        seen.add(key); final.append(k)
    return final[:80]


def build_knowledge_graph(docs: List[Document]) -> Dict[str, Any]:
    from enhanced_retrieval import extract_phases_from_docs, extract_actors_from_docs
    phases = extract_phases_from_docs(docs)
    actors = extract_actors_from_docs(docs)
    risks = extract_risks_and_mitigations(docs)
    kpis = extract_kpis(docs)
    graph = {
        'phases': phases,              # phase_number -> {...}
        'actors': actors,              # list of actor dicts
        'risks': risks,                # list of risk dicts
        'kpis': kpis,                  # list of KPI dicts
    }
    return graph

__all__ = [
    'build_knowledge_graph', 'extract_risks_and_mitigations', 'extract_kpis'
]
