# Configuration for Enhanced PDF Chat Accuracy

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    # Number of documents to retrieve initially
    "initial_k": 10,
    
    # Number of documents to consider in MMR
    "fetch_k": 20,
    
    # Lambda parameter for MMR (balance between relevance and diversity)
    "lambda_mult": 0.6,
    
    # Number of documents for extended search when answer seems incomplete
    "extended_k": 15,
    
    # Search type: "similarity", "mmr", or "similarity_score_threshold"
    "search_type": "mmr",

    # Hybrid retrieval: combine lexical (BM25) + vector search
    "use_hybrid": True,
    "bm25_k": 30,
    "hybrid_alpha": 0.6  # Weight for dense score in hybrid fusion (0..1)
}

# Text Processing Configuration
TEXT_CONFIG = {
    # Chunk size for text splitting
    "chunk_size": 800,
    
    # Overlap between chunks
    "chunk_overlap": 300,
    
    # Text separators for better splitting
    "separators": ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    # Maximum number of features
    "max_features": 1024,
    
    # N-gram range
    "ngram_range": (1, 3),
    
    # Minimum document frequency
    "min_df": 1,
    
    # Maximum document frequency
    "max_df": 0.85,
    
    # Use sublinear term frequency
    "sublinear_tf": True,
    
    # Stop words
    "stop_words": "english"
}

# FastEmbed Embeddings (local, CPU, ONNX)
FASTEMBED_CONFIG = {
    # Upgraded model: multi-task, multi-lingual (search, cloze, classification) good local tradeoff
    "model_name": "BAAI/bge-m3",
    "batch_size": 64,
    "normalize": True,
    "offline_only": False,
    "local_model_dir": None,
    "cache_dir": ".emb_cache"
}

# Neural re-ranking (local, ONNX via fastembed)
RERANK_CONFIG = {
    "use_neural_reranker": True,
    "model_name": "BAAI/bge-reranker-small",
    "top_n": 12
}

# Confidence/"no-answer" guardrails
GUARDRAIL_CONFIG = {
    # Enable confidence gate to avoid hallucinations when evidence is weak
    "enable_confidence_gate": True,
    # Minimum evidence score (0..1) required to answer directly
    "min_evidence_score": 0.35,
    # Minimum number of decent matches before answering
    "min_docs": 2
}

# Vector store persistence (local-only)
PERSISTENCE_CONFIG = {
    "persist_faiss": True,
    # Directory to save FAISS + docstore on disk
    "path": ".faiss_index"
}

# Conversation Memory
MEMORY_CONFIG = {
    "summarize_after": 10,   # summarize when messages exceed this count
    "keep_last": 6,          # keep last N messages verbatim alongside summary
    "summary_prompt": (
        "Tu es un assistant. Résume brièvement l'échange ci-dessous en 5-8 puces max, "
        "en conservant les faits clés (phases, nombres, décisions). "
        "Évite les généralités et ne rajoute aucune information non présente.\n\n"
        "Résumé actuel (si présent):\n{current_summary}\n\n"
        "Nouveaux messages (du plus ancien au plus récent):\n{messages}\n\n"
        "Résumé mis à jour:" 
    )
}

# LLM Configuration
LLM_CONFIG = {
    # Temperature for response generation
    "temperature": 0.2,
    
    # Top-p for nucleus sampling
    "top_p": 0.9,
    
    # Top-k for top-k sampling
    "top_k": 40,
    # Preferred local models to try (order matters). Must be available in Ollama.
    # We'll pick the first installed one.
    "preferred_models": [
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.2:3b",
        "phi3:3.8b",
        "qwen2.5:7b-instruct-q4_K_M",
        "mistral:7b-instruct-q4_K_M"
    ]
}

# Embedding / vector caching configuration
CACHING_CONFIG = {
    "enable": True,
    # Directory to store cached embedding arrays / doc metadata
    "path": ".cache/embeddings",
    # Recompute if chunk params change (hash includes chunk_size & overlap)
    "respect_chunk_params": True,
    # File extension for serialized vectors
    "ext": ".npz"
}

# Metrics collection
METRICS_CONFIG = {
    "enable": True,
    "collect_retrieval_latency": True,
    "collect_rerank_latency": True,
    "collect_embedding_build_time": True,
    # Keep only last N sessions metrics
    "max_history": 20
}

# UI adjustments
UI_CONFIG = {
    "show_metrics_panel": False,  # disabled to simplify UI
    "show_confidence_badge": True,
    "max_source_snippet_chars": 320,
    # Highlight citations like [source: file p.X]
    "highlight_citations": True
}

# Query Enhancement Keywords
QUERY_ENHANCEMENT = {
    "phase_keywords": ["phase", "étape", "step", "stage", "niveau"],
    "process_keywords": ["processus", "process", "procédure", "workflow"],
    "project_keywords": ["projet", "project", "développement", "development"],
    "completeness_keywords": ["combien", "how many", "toutes", "all", "liste", "complete"]
}

# Response Validation
VALIDATION_CONFIG = {
    # Minimum number of phases expected in complete answers
    "min_phases_expected": 2,
    
    # Keywords that indicate the user wants complete information
    "completeness_indicators": ["combien", "toutes", "liste", "how many", "all", "complete", "entier"],
    
    # Keywords that boost document relevance scores
    "relevance_boosters": {
        "phase_mentions": 10,
        "numbered_lists": 5,
        "structure_indicators": 3,
        "completeness_bonus": 15
    }
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "main_template": """Tu es un assistant expert en analyse de documents techniques. Tu dois répondre aux questions en te basant STRICTEMENT sur les documents fournis.

INSTRUCTIONS:
1. Si la question porte sur des éléments spécifiques des documents (phases, processus, etc.), analyse le contenu en détail
2. Si c'est une question générale, donne une réponse appropriée et propose d'analyser les documents si nécessaire
3. Sois précis et utilise les informations exactes des documents; ne fabrique rien
4. Si des informations semblent manquer, indique-le clairement et dis "Je ne trouve pas dans les documents fournis"
5. Lorsque tu affirmes un fait, ajoute une courte citation entre crochets: [source: <fichier> p.<page>]

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE:""",

    "phase_counting_template": """Tu es un assistant expert qui doit compter avec PRÉCISION toutes les phases dans les documents.

INSTRUCTION ABSOLUE: 
- Examine TOUT le contexte fourni
- Compte UNIQUEMENT les phases qui existent réellement dans les documents
- N'invente JAMAIS de phases qui n'existent pas
- Inclus OBLIGATOIREMENT la Phase 00 si elle existe

MÉTHODE STRICTE:
1. Scan tout le texte pour "Phase" suivi d'un numéro (00, 01, 02, 03, etc.)
2. Identifie les descriptions de chaque phase
3. Compte le nombre total EXACT
4. Liste chaque phase avec sa description exacte

INTERDICTIONS:
- N'invente pas de phases inexistantes (Phase 04, 05, etc.)
- Ne spécule pas sur des phases possibles
- Ne compte que les phases explicitement mentionnées

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE PRÉCISE (compte exact basé sur les documents):
Voici le nombre EXACT et la liste COMPLÈTE des phases trouvées dans les documents:""",

    "document_analysis_template": """Tu es un assistant expert en analyse de documents techniques. Tu dois répondre aux questions en te basant UNIQUEMENT sur les documents fournis.

INSTRUCTIONS IMPORTANTES:
1. Lis attentivement TOUT le contexte fourni
2. Identifie TOUTES les phases, étapes ou sections mentionnées
3. Si une question porte sur les phases d'un projet, liste-les TOUTES sans exception
4. Sois précis et complet dans tes réponses
5. Si des informations semblent manquer, indique-le clairement
6. Utilise les numéros et noms exacts des phases/sections trouvés dans les documents

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE DÉTAILLÉE:
Basé sur l'analyse complète des documents fournis:""",

    "completeness_template": """Tu es un assistant expert qui doit donner une réponse COMPLÈTE et EXHAUSTIVE.

INSTRUCTION SPÉCIALE: Cette question demande une réponse complète. Tu DOIS inclure TOUTES les phases, étapes ou éléments mentionnés dans les documents.

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE COMPLÈTE ET EXHAUSTIVE:
Voici la liste COMPLÈTE basée sur les documents:"""
}

# Error Messages
ERROR_MESSAGES = {
    "incomplete_response": "La réponse semble incomplète. Vérifiez si toutes les phases sont mentionnées.",
    "no_context": "Aucun contexte pertinent trouvé dans les documents.",
    "retrieval_error": "Erreur lors de la récupération des informations.",
    "llm_error": "Erreur lors de la génération de la réponse."
}
