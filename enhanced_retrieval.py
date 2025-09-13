# Enhanced Retrieval Functions for Better Accuracy and Precision

from config import QUERY_ENHANCEMENT, VALIDATION_CONFIG, PROMPT_TEMPLATES
import re
from typing import List, Dict, Any
from langchain_core.documents import Document

def enhance_query(query):
    """Enhance user query to improve retrieval accuracy"""
    
    enhanced_query = query.lower()
    
    # Add related terms for better matching
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["phase_keywords"]):
        enhanced_query += " phase étape processus développement phase00 phase01 phase02 phase03"
    
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["process_keywords"]):
        enhanced_query += " phase procédure workflow méthodologie"
    
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["completeness_keywords"]):
        enhanced_query += " nombre total liste complète toutes phases étapes phase00 phase01 phase02 phase03 étude chiffrage conception développement réalisation production"
    
    # Special enhancement for "how many phases" type questions
    if any(phrase in enhanced_query for phrase in ["how many", "combien", "number of"]):
        enhanced_query += " phase00 phase01 phase02 phase03 étude chiffrage conception réalisation production toutes phases complète liste"
    
    return enhanced_query

def rerank_retrieved_docs(docs, query):
    """Re-rank retrieved documents based on query relevance"""
    
    query_lower = query.lower()
    
    # Keywords that indicate completeness requests
    completeness_indicators = ["combien", "toutes", "liste", "how many", "all", "complete"]
    
    scored_docs = []
    for doc in docs:
        score = 0
        content = doc.page_content.lower()
        
        # Boost score for documents containing phase information
        if "phase" in content:
            score += 10
        
        # Moderate boost for Phase 00 (reduce overweighting)
        if any(indicator in content for indicator in ["phase 00", "phase 0", "étude et chiffrage", "study and estimation"]):
            score += 10
        
        # Boost score for specific phases
        phase_mentions = 0
        for i in range(0, 6):  # Check phases 0-5
            if f"phase {i:02d}" in content or f"phase {i}" in content:
                phase_mentions += 1
                score += 8
        
        # Boost score for numbered lists or sequences
        if any(str(i) in content for i in range(0, 10)):
            score += 5
        
        # Boost score for completeness if query asks for it
        if any(indicator in query_lower for indicator in completeness_indicators):
            if "phase" in content and phase_mentions >= 2:
                score += 12
            # Extra boost if document contains Phase 00
            if any(p00 in content for p00 in ["phase 00", "phase 0", "étude et chiffrage"]):
                score += 12
        
        # Boost score for document structure indicators
        if any(marker in content for marker in [":", "•", "-", "1.", "2.", "3."]):
            score += 3
        
        # Boost for comprehensive content
        if len(content.split()) > 100:  # Longer documents likely more comprehensive
            score += 2
        
        scored_docs.append((score, doc))
    
    # Sort by score (descending) and return documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

def create_enhanced_context(retrieved_docs, query):
    """Create enhanced context from retrieved documents"""
    
    query_lower = query.lower()
    
    # Check if query is asking for phases/steps
    is_phase_query = any(keyword in query_lower for keyword in 
                        ["phase", "étape", "combien", "how many", "liste", "all"])
    
    if is_phase_query:
        # For phase queries, prioritize documents with phase information
        phase_docs = []
        other_docs = []
        
        for doc in retrieved_docs:
            if "phase" in doc.page_content.lower():
                phase_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Combine with phase docs first
        ordered_docs = phase_docs + other_docs
    else:
        ordered_docs = retrieved_docs
    
    # Create enhanced context string
    context_parts = []
    for i, doc in enumerate(ordered_docs[:8]):  # Limit to top 8 docs
        context_parts.append(f"[DOCUMENT {i+1}]\n{doc.page_content}\n")
    
    return "\n".join(context_parts)

def validate_response_completeness(response, query):
    """Validate if response is complete and accurate for phase-related queries"""
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Check for phase completeness
    if any(keyword in query_lower for keyword in ["combien", "how many", "toutes", "all", "phases"]):
        if "phase" in query_lower or "étape" in query_lower:
            # Look for all possible phase formats
            phases_found = set()
            
            # Check for Phase 00, Phase 0, Phase 01, etc.
            phase_patterns = [
                r'phase\s*0*0',   # Phase 00, Phase 0
                r'phase\s*0*1',   # Phase 01, Phase 1  
                r'phase\s*0*2',   # Phase 02, Phase 2
                r'phase\s*0*3',   # Phase 03, Phase 3
                r'phase\s*0*4',   # Phase 04, Phase 4
                r'phase\s*0*5'    # Phase 05, Phase 5
            ]
            
            import re
            for pattern in phase_patterns:
                matches = re.findall(pattern, response_lower)
                for match in matches:
                    # Extract the phase number
                    phase_num = re.search(r'\d+', match)
                    if phase_num:
                        phases_found.add(int(phase_num.group()))
            
            # Special check for Phase 00 keywords
            if not any(p == 0 for p in phases_found):
                phase_00_indicators = [
                    "étude et chiffrage", "study and estimation", 
                    "feasibility", "faisabilité", "estimation"
                ]
                
                if any(indicator in response_lower for indicator in phase_00_indicators):
                    phases_found.add(0)
            
            # Check for hallucination warning - phases mentioned but not in typical project documents
            suspicious_phases = [p for p in phases_found if p > 3]
            if suspicious_phases and any(phrase in response_lower for phrase in 
                ["non mentionnée", "non trouvée", "pas dans les documents", "documents fournis"]):
                return False, f"⚠️ ERREUR: La réponse mentionne des phases inexistantes (Phase {', '.join(map(str, suspicious_phases))}). Basez-vous uniquement sur ce qui existe dans les documents."
            
            # Check if response seems incomplete (missing important phases)
            if len(phases_found) < 3:
                return False, f"La réponse semble incomplète. Phases détectées: {sorted(phases_found)}. Vérifiez si toutes les phases sont mentionnées."
            
            # Specific check for Phase 00
            if 0 not in phases_found:
                return False, "⚠️ ATTENTION: La Phase 00 (Étude et chiffrage) semble manquer dans la réponse."
            
            # Check for counting accuracy
            total_claimed = None
            count_patterns = [
                r'il y a (\d+) phases',
                r'(\d+) phases principales',
                r'(\d+) phases dans',
                r'total[^0-9]*(\d+)[^0-9]*phases'
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    total_claimed = int(match.group(1))
                    break
            
            if total_claimed and total_claimed != len(phases_found):
                return False, f"⚠️ INCOHÉRENCE: Le nombre annoncé ({total_claimed}) ne correspond pas aux phases listées ({len(phases_found)}). Vérifiez le décompte."
    
    return True, ""


def extract_phases_from_docs(retrieved_docs):
    """Extract phases and names from the retrieved documents using regex.

    Returns a dict: {phase_number: {"name": name, "evidence": [snippets...]}}
    """
    phase_map = {}
    # Common patterns: "Phase 03 : Production série", "Phase 1- ...", etc.
    # Capture number and name to end-of-line.
    patterns = [
        r"phase\s*0*(\d+)\s*[:\-–]\s*([^\n\r]+)",
        r"phase\s*0*(\d+)\s+([^:\n\r]{3,50})"  # fallback without colon
    ]

    for doc in retrieved_docs:
        text = doc.page_content
        # Work line by line for cleaner names
        for line in text.splitlines():
            line_l = line.lower()
            if "phase" not in line_l:
                continue
            for pat in patterns:
                for m in re.finditer(pat, line_l, flags=re.IGNORECASE):
                    try:
                        num = int(m.group(1))
                    except Exception:
                        continue
                    # Extract name from original line preserving case
                    start, end = m.span()
                    # Try to map name from original casing using indices
                    name = line[m.start(2):m.end(2)].strip() if m.lastindex and m.lastindex >= 2 else line.strip()
                    # Clean name (remove trailing bullets or numbers)
                    name = re.sub(r"^[\-•\d\.\s]+", "", name).strip()
                    name = re.sub(r"\s{2,}", " ", name)

                    entry = phase_map.get(num, {"name": None, "evidence": [], "citations": []})
                    # Prefer the longest, more descriptive name
                    if not entry["name"] or (name and len(name) > len(entry["name"])):
                        entry["name"] = name
                    snippet = line.strip()
                    if snippet not in entry["evidence"]:
                        entry["evidence"].append(snippet)
                    # Add a citation anchor from this doc
                    meta = getattr(doc, 'metadata', {}) or {}
                    cite = {
                        'source': meta.get('source', 'doc'),
                        'page': meta.get('page', '?'),
                        'line': snippet[:200]
                    }
                    if cite not in entry.get('citations', []):
                        entry['citations'].append(cite)
                    phase_map[num] = entry

    return phase_map


def extract_subject_from_docs(retrieved_docs):
    """Extract a concise 'subject' or 'purpose' from early pages.

    Looks for headings like 'Objet', 'Subject', 'Purpose', 'Scope', or a bold title line.
    Returns dict { 'subject': str, 'source': file, 'page': int } or None.
    """
    subject_patterns = [
        r"\bobjet\b[:\-]?\s*(.+)",
        r"\bsubject\b[:\-]?\s*(.+)",
        r"\bpurpose\b[:\-]?\s*(.+)",
        r"\bscope\b[:\-]?\s*(.+)",
        r"^\s*[A-Z][A-Z \-/]{8,}$"  # ALL CAPS title line
    ]
    for doc in retrieved_docs[:6]:  # favor early, most relevant docs
        lines = doc.page_content.splitlines()
        for ln in lines[:80]:  # limit scan per doc
            line = ln.strip()
            low = line.lower()
            for pat in subject_patterns:
                m = re.search(pat, low, flags=re.IGNORECASE)
                if m:
                    text = line if m.lastindex is None else line[m.start(1):m.end(1)].strip() if m.lastindex else line
                    text = re.sub(r"\s{2,}", " ", text)
                    meta = getattr(doc, 'metadata', {}) or {}
                    return {
                        'subject': text[:300],
                        'source': meta.get('source', 'doc'),
                        'page': meta.get('page', '?')
                    }
    return None


def build_subject_index(all_docs: List[Document], pdf_meta: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Dict[str, Any]]:
    """Build a per-file subject index using robust heuristics.

    Inputs:
    - all_docs: chunked LC Documents preserving metadata {source, page}
    - pdf_meta: optional dict {source: {Title, Subject, ...}}

    Returns: {source: {subject: str, page: int, confidence: float, method: str}}

    Heuristics order (highest priority first):
    1) Explicit heading lines on early pages: Objet/Subject/Purpose/Scope/Title
    2) Clean, prominent title-like lines (ALL CAPS or Title Case) near top of page 1, excluding section headers like "Phase 0X"
    3) PDF metadata Title/Subject if non-generic
    """
    pdf_meta = pdf_meta or {}
    results: Dict[str, Dict[str, Any]] = {}

    # Group early-page text by source
    per_source: Dict[str, Dict[str, Any]] = {}
    for d in all_docs:
        meta = getattr(d, 'metadata', {}) or {}
        src = meta.get('source', 'doc')
        page = int(meta.get('page', 9999)) if str(meta.get('page', '')).isdigit() else 9999
        if page > 3:
            continue
        per = per_source.setdefault(src, {'lines': [], 'pages': set()})
        per['pages'].add(page)
        # Keep only the first ~1200 chars per page chunk
        for ln in (d.page_content[:1200].splitlines()):
            if ln.strip():
                per['lines'].append((page, ln.strip()))

    # Helpers
    heading_re = re.compile(r"(?i)\b(objet|subject|purpose|scope|title)\b\s*[:\-]\s*(.+)")
    bad_tokens = [
        'phase 0', 'phase 1', 'phase 2', 'phase 3', 'sommaire', 'table of contents',
        'version', 'référence', 'reference', 'date', 'logigramme', 'données', 'donnees'
    ]

    def looks_like_section_header(s: str) -> bool:
        sl = s.lower()
        if any(bt in sl for bt in bad_tokens):
            return True
        # too long or mostly numeric/punct
        if len(s) > 140 or sum(c.isalpha() for c in s) < 4:
            return True
        return False

    def is_title_case_or_caps(s: str) -> bool:
        # Accept ALL CAPS or Title Case with limited punctuation
        if s.isupper() and 8 <= len(s) <= 120:
            return True
        words = s.split()
        cap_words = sum(1 for w in words if w[:1].isupper())
        return len(words) >= 2 and cap_words / max(1, len(words)) >= 0.6 and 8 <= len(s) <= 120

    # Build result per source
    for src, info in per_source.items():
        lines = info['lines']
        best = None
        # 1) Explicit headings
        for page, ln in lines[:200]:
            m = heading_re.search(ln)
            if m:
                cand = m.group(2).strip()
                if not looks_like_section_header(cand):
                    best = {'subject': cand, 'page': page, 'confidence': 0.95, 'method': 'heading'}
                    break

        # 2) Prominent title-like lines on first page
        if not best:
            for page, ln in lines[:120]:
                if page != 1:
                    continue
                if is_title_case_or_caps(ln) and not looks_like_section_header(ln):
                    best = {'subject': ln, 'page': page, 'confidence': 0.85, 'method': 'titleline'}
                    break

        # 3) PDF metadata fallback
        if not best and src in pdf_meta:
            meta = pdf_meta.get(src) or {}
            for key in ('Subject', 'Title', 'title', 'subject'):
                val = (meta.get(key) or '').strip() if isinstance(meta.get(key), str) else ''
                if val and not looks_like_section_header(val) and len(val) >= 6:
                    best = {'subject': val, 'page': 1, 'confidence': 0.7, 'method': f'meta:{key}'}
                    break

        if best:
            results[src] = best

    return results


def extract_actors_from_docs(retrieved_docs):
    """Extract project actors/stakeholders from documents.

    Returns list of dicts: [{ 'name': str, 'source': file, 'page': int }]
    """
    actors = []
    actor_headings = [
        'intervenants', 'acteurs', 'actors', 'stakeholders', 'responsables',
        'equipe projet', "équipe projet", 'roles', 'rôles'
    ]
    bullet_re = re.compile(r"^\s*(?:[-•\u2022]|\d+\.|\*)\s*(.+)")
    for doc in retrieved_docs:
        text = doc.page_content
        meta = getattr(doc, 'metadata', {}) or {}
        lines = text.splitlines()
        near_heading = False
        for ln in lines:
            low = ln.strip().lower()
            # Detect heading zone
            if any(h in low for h in actor_headings):
                near_heading = True
                continue
            if near_heading:
                if not low or len(low) < 2:
                    near_heading = False
                    continue
                m = bullet_re.match(ln.strip())
                item = (m.group(1).strip() if m else ln.strip())
                # Stop if the item looks like a new section header
                if len(item) > 0 and item.isupper() and len(item) > 40:
                    near_heading = False
                    continue
                # Filter overly long lines
                if 1 <= len(item) <= 120:
                    actors.append({
                        'name': re.sub(r"\s{2,}", " ", item),
                        'source': meta.get('source', 'doc'),
                        'page': meta.get('page', '?')
                    })
        # Also capture inline comma-separated roles
        joined = " ".join(lines)
        if any(h in joined.lower() for h in actor_headings):
            parts = re.split(r",|;", joined)
            for p in parts:
                token = p.strip()
                if 2 < len(token) <= 80 and any(k in token.lower() for k in ['chef', 'ingénieur', 'engineer', 'leader', 'qualité', 'logistique', 'commercial', 'process', 'projet', 'project']):
                    actors.append({
                        'name': token,
                        'source': meta.get('source', 'doc'),
                        'page': meta.get('page', '?')
                    })
    # Deduplicate by name
    seen = set()
    uniq = []
    for a in actors:
        key = a['name'].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq[:20]
