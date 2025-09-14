"""Simple evaluation harness skeleton.

Usage:
    Streamlit not required. This script can be run to batch-query the local
    retrieval + LLM pipeline for regression testing.

Planned extensions (future):
    - Precision / coverage metrics for phase detection
    - Latency aggregation
    - Language consistency checks
"""
from __future__ import annotations
import json, time, os, sys
from typing import List, Dict

try:
    from question_routing import classify_question
except ImportError:
    print("Run from project root so imports resolve.")
    sys.exit(1)

SAMPLE_QUERIES = [
    "Combien de phases y a-t-il dans ce projet?",
    "Liste de toutes les phases",
    "Quel est le sujet du document?",
    "Quels sont les acteurs principaux?",
    "Give me a concise summary of the documents",
]


def main():
    rows = []
    for q in SAMPLE_QUERIES:
        qc = classify_question(q)
        rows.append({
            'question': q,
            'flags': {k: getattr(qc, k) for k in ['is_summary','is_phase_count','is_subject','is_actors','is_generic_count']},
            'generic_target': qc.generic_count_target,
            'debug_matches': qc.debug_matches
        })
    print(json.dumps(rows, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
