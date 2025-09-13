# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import shutil

# Suppress Streamlit torch warnings
import os
import sys

# Set environment variables before any torch-related imports
os.environ["PYTORCH_DISABLE_PER_OP_PROFILING"] = "1"
os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Any
from pydantic import PrivateAttr, ConfigDict, Field
from htmlTemplates import css, bot_template, user_template
from enhanced_retrieval import (
    enhance_query,
    rerank_retrieved_docs,
    create_enhanced_context,
    validate_response_completeness,
    extract_phases_from_docs,
    extract_subject_from_docs,
    extract_actors_from_docs,
    build_subject_index,
)
from config import (
    RETRIEVAL_CONFIG, TEXT_CONFIG, TFIDF_CONFIG, FASTEMBED_CONFIG, RERANK_CONFIG,
    LLM_CONFIG, PROMPT_TEMPLATES, MEMORY_CONFIG, GUARDRAIL_CONFIG, PERSISTENCE_CONFIG,
    CACHING_CONFIG, METRICS_CONFIG, UI_CONFIG
)
from cache_utils import cache_key, load_cache, save_cache
from intent_detection import detect_intent, generate_greeting_response, generate_simple_chat_response, should_use_enhanced_retrieval
import warnings
import subprocess
import re
from datetime import datetime
import time

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.*")

# Redirect torch warnings to suppress them completely
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Application constants / default model
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"  # Force use of this model as per user request

# ---------------------------------------------------------------------------
# Lightweight language detection (English vs French) without external deps
# ---------------------------------------------------------------------------
def detect_language_simple(text: str) -> str:
    t = text.lower()
    # Common French tokens and diacritics
    fr_hits = sum(1 for w in [" le ", " la ", " les ", " des ", " une ", " et ", " est ", " que ", " pour ", " avec ", " sur ", " phase ", " étapes ", " projet ", " document ", " résumé "] if w in f" {t} ")
    # English tokens
    en_hits = sum(1 for w in [" the ", " and ", " is ", " are ", " for ", " with ", " about ", " project ", " document ", " summary ", " phase "] if w in f" {t} ")
    # Heuristic: presence of accented characters => French
    if any(c in t for c in "éèàùçôîïÊÉÀÇ"):
        fr_hits += 2
    if fr_hits == en_hits:
        # fallback: default to language of previous interaction if available
        return st.session_state.get('last_lang', 'en')
    return 'fr' if fr_hits > en_hits else 'en'

def localize(fr: str, en: str, lang: str) -> str:
    return fr if lang == 'fr' else en

def build_global_phase_map() -> dict:
    """Scan all processed chunks to deterministically extract phases with citations.

    Caches result in st.session_state.phase_cache.
    """
    try:
        if st.session_state.get('phase_cache'):
            return st.session_state['phase_cache']
        chunks = st.session_state.get('all_chunks') or []
        if not chunks:
            return {}
        # Reuse extractor on all chunks
        from enhanced_retrieval import extract_phases_from_docs
        phase_map = extract_phases_from_docs(chunks)
        st.session_state['phase_cache'] = phase_map
        return phase_map
    except Exception:
        return {}

def get_pdf_text(pdf_docs):
    """Extract text and metadata from PDFs with OCR fallback for scanned pages.

    Returns a list of LangChain Document objects with page metadata for better citations.
    """
    docs = []
    # Keep lightweight PDF-level metadata for subject detection
    if 'pdf_meta' not in st.session_state:
        st.session_state.pdf_meta = {}

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            file_name = getattr(pdf, 'name', 'uploaded.pdf')
            full_text_found = False
            # Capture top-level PDF metadata
            try:
                raw_meta = pdf_reader.metadata or {}
                st.session_state.pdf_meta[file_name] = {k: (str(v) if v is not None else '') for k, v in raw_meta.items()}
            except Exception:
                st.session_state.pdf_meta[file_name] = {}
            
            # First try normal text extraction
            for page_index, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text_found = True
                    docs.append(Document(
                        page_content=page_text,
                        metadata={"source": file_name, "page": page_index + 1}
                    ))
            
            # If no text found, try OCR with PyMuPDF
            if not full_text_found:
                try:
                    import fitz  # PyMuPDF
                    import pytesseract
                    from PIL import Image
                    import io
                    
                    # Check if Tesseract is available
                    try:
                        # Set Tesseract path for Windows
                        import platform
                        if platform.system() == 'Windows':
                            possible_paths = [
                                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Microsoft\WinGet\Packages\UB-Mannheim.TesseractOCR_Microsoft.Winget.Source_8wekyb3d8bbwe\tesseract.exe",
                                r"C:\tools\tesseract\tesseract.exe"
                            ]
                            tesseract_found = False
                            for path in possible_paths:
                                if os.path.exists(path):
                                    pytesseract.pytesseract.tesseract_cmd = path
                                    tesseract_found = True
                                    st.info(f"🔍 Found Tesseract at: {path}")
                                    break
                            
                            if not tesseract_found:
                                # Try to find tesseract in PATH
                                try:
                                    import shutil
                                    tesseract_path = shutil.which("tesseract")
                                    if tesseract_path:
                                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                                        tesseract_found = True
                                        st.info(f"🔍 Found Tesseract in PATH: {tesseract_path}")
                                except:
                                    pass
                        
                        # Test if Tesseract works
                        pytesseract.get_tesseract_version()
                        
                    except Exception as e:
                        st.error(f"❌ {pdf.name} is a scanned PDF but Tesseract OCR is not installed.")
                        st.info("🔧 **Install Tesseract OCR:**")
                        st.code("winget install UB-Mannheim.TesseractOCR")
                        st.info("Or download from: https://github.com/UB-Mannheim/tesseract/wiki")
                        st.info("After installation, restart the app.")
                        continue
                    
                    # Reset file pointer
                    pdf.seek(0)
                    pdf_bytes = pdf.read()
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    st.info(f"🔍 Using OCR for {file_name} (scanned document)")
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR
                        try:
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            if ocr_text.strip():
                                docs.append(Document(
                                    page_content=ocr_text,
                                    metadata={"source": file_name, "page": page_num + 1, "ocr": True}
                                ))
                        except Exception as ocr_error:
                            st.warning(f"⚠️ OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    
                    pdf_document.close()
                    
                except ImportError:
                    st.error(f"❌ {pdf.name} appears to be a scanned PDF but OCR libraries are missing.")
                    st.info("Run: pip install pymupdf pytesseract pillow")
                    continue
                except Exception as e:
                    st.error(f"❌ OCR failed for {pdf.name}: {str(e)}")
                    continue
            
            if docs:
                st.success(f"✅ Processed {file_name}")
            else:
                st.error(f"❌ No text found in {file_name}")
                
        except Exception as e:
            st.error(f"❌ Error reading {getattr(pdf, 'name', 'uploaded.pdf')}: {str(e)}")
    
    # Determine dominant document language (very lightweight heuristic)
    try:
        if docs:
            sample_text = " \n".join(d.page_content[:1500] for d in docs[:8])[:8000].lower()
            fr_tokens = [" le ", " la ", " les ", " des ", " une ", " et ", " est ", " que ", " projet ", " phase ", " document ", " étapes "]
            en_tokens = [" the ", " and ", " is ", " are ", " project ", " phase ", " document ", " development ", " step ", " process "]
            fr_score = sum(sample_text.count(t) for t in fr_tokens)
            en_score = sum(sample_text.count(t) for t in en_tokens)
            # Accented chars bonus
            if any(c in sample_text for c in "éèàùçôîïÊÉÀÇ"):
                fr_score += 3
            dominant = 'fr' if fr_score > en_score else 'en'
            st.session_state['doc_language'] = dominant
    except Exception:
        pass
    return docs

def get_text_chunks(documents):
    """Split LC Documents into chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CONFIG["chunk_size"],
        chunk_overlap=TEXT_CONFIG["chunk_overlap"],
        length_function=len,
        separators=TEXT_CONFIG["separators"]
    )
    return text_splitter.split_documents(documents)

def get_vectorstore(text_chunks):
    """Create vector store from text chunks using FastEmbed (bge-m3) with optional cache; TF-IDF fallback."""
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings

        cache_dir = os.path.join(os.getcwd(), ".emb_cache")
        local_hint = FASTEMBED_CONFIG.get("local_model_dir")
        offline_only = FASTEMBED_CONFIG.get("offline_only", False)

        # Local availability detection
        have_local = False
        try:
            if local_hint and os.path.exists(local_hint):
                have_local = True
            elif os.path.isdir(cache_dir):
                for name in os.listdir(cache_dir):
                    if "onnx" in name.lower() or "model" in name.lower() or name.startswith("models--"):
                        have_local = True
                        break
        except Exception:
            have_local = False

        if offline_only and not have_local:
            raise RuntimeError("FastEmbed offline_only is enabled and no local model/cache is present")

        st.info("🧠 Using FastEmbed embeddings (bge-m3)")
        embeddings = FastEmbedEmbeddings(
            model_name=FASTEMBED_CONFIG["model_name"],
            cache_dir=(local_hint if local_hint else cache_dir),
            normalize_embeddings=FASTEMBED_CONFIG.get("normalize", True)
        )

        use_cache = CACHING_CONFIG.get('enable', False)
        key = None
        if use_cache:
            try:
                key = cache_key(text_chunks)
                cached = load_cache(key)
                if cached:
                    import numpy as np
                    import faiss
                    from langchain_community.docstore.in_memory import InMemoryDocstore
                    arr = cached['vectors']
                    # Rebuild documents from cached serialized form if present
                    cached_docs = []
                    for d in cached.get('docs', []):
                        try:
                            cached_docs.append(Document(page_content=d['page_content'], metadata=d['metadata']))
                        except Exception:
                            pass
                    if len(cached_docs) == len(text_chunks) and arr.shape[0] == len(text_chunks):
                        dim = arr.shape[1]
                        index = faiss.IndexFlatL2(dim)
                        index.add(arr.astype('float32'))
                        # Build docstore
                        docstore = InMemoryDocstore({str(i): cached_docs[i] for i in range(len(cached_docs))})
                        index_map = {i: str(i) for i in range(len(cached_docs))}
                        vs = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_map)
                        st.info("✅ Loaded embeddings from cache")
                        return vs
            except Exception:
                pass

        # Load existing persisted FAISS index and merge new docs
        if PERSISTENCE_CONFIG.get("persist_faiss") and os.path.isdir(PERSISTENCE_CONFIG.get("path", ".faiss_index")):
            try:
                vs = FAISS.load_local(
                    PERSISTENCE_CONFIG["path"],
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                vs.add_documents(text_chunks)
                return vs
            except Exception:
                pass

        # Fresh build
        t0 = time.time()
        vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        build_time = round(time.time() - t0, 3)
        if METRICS_CONFIG.get('enable') and METRICS_CONFIG.get('collect_embedding_build_time'):
            st.session_state.setdefault('metrics', {})['embedding_build_s'] = build_time

        # Save cache
        if use_cache and key:
            try:
                embs = embeddings.embed_documents(text_chunks)
                save_cache(key, text_chunks, embs)
            except Exception:
                pass

        # Persist to disk if requested
        try:
            if PERSISTENCE_CONFIG.get("persist_faiss"):
                vectorstore.save_local(PERSISTENCE_CONFIG.get("path", ".faiss_index"))
        except Exception:
            pass
        return vectorstore

    except Exception:
        st.info("FastEmbed not ready (using TF-IDF fallback)...")
        try:
            st.info("🔄 Falling back to TF-IDF embeddings (torch-free)...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from langchain.embeddings.base import Embeddings
            import numpy as np

            class TorchFreeTFIDFEmbeddings(Embeddings):
                def __init__(self):
                    self.vectorizer = TfidfVectorizer(
                        max_features=TFIDF_CONFIG["max_features"],
                        stop_words=TFIDF_CONFIG["stop_words"],
                        ngram_range=TFIDF_CONFIG["ngram_range"],
                        min_df=TFIDF_CONFIG["min_df"],
                        max_df=TFIDF_CONFIG["max_df"],
                        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
                        analyzer='word'
                    )
                    self.fitted = False
                    self.feature_dim = TFIDF_CONFIG["max_features"]

                def embed_documents(self, texts):
                    raw = [t.page_content if isinstance(t, Document) else str(t) for t in texts]
                    if not self.fitted:
                        vectors = self.vectorizer.fit_transform(raw)
                        self.fitted = True
                    else:
                        vectors = self.vectorizer.transform(raw)
                    dense = vectors.toarray()
                    result = []
                    for v in dense:
                        if len(v) < self.feature_dim:
                            padded = np.pad(v, (0, self.feature_dim - len(v)), 'constant')
                        else:
                            padded = v[:self.feature_dim]
                        result.append(padded.tolist())
                    return result

                def embed_query(self, text):
                    if not self.fitted:
                        return [0.0] * self.feature_dim
                    vector = self.vectorizer.transform([text])
                    dense = vector.toarray()[0]
                    if len(dense) < self.feature_dim:
                        padded = np.pad(dense, (0, self.feature_dim - len(dense)), 'constant')
                    else:
                        padded = dense[:self.feature_dim]
                    return padded.tolist()

            embeddings = TorchFreeTFIDFEmbeddings()
            vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
            st.info("Using TF-IDF embeddings (lexical precision mode)")
            return vectorstore
        except Exception as e2:
            st.error(f"❌ All embedding methods failed: {e2}")
            return None

def build_hybrid_retriever(vectorstore, raw_docs):
    """Optionally build a hybrid retriever combining BM25 and vector search.

    raw_docs: list[Document] (original chunked documents)
    Returns a retriever-like object with get_relevant_documents().
    """
    base_retriever = vectorstore.as_retriever(
        search_type=RETRIEVAL_CONFIG["search_type"],
        search_kwargs={
            "k": RETRIEVAL_CONFIG["initial_k"],
            "lambda_mult": RETRIEVAL_CONFIG["lambda_mult"],
            "fetch_k": RETRIEVAL_CONFIG["fetch_k"]
        }
    )

    if not RETRIEVAL_CONFIG.get("use_hybrid", False):
        return base_retriever

    try:
        from rank_bm25 import BM25Okapi

        corpus = [d.page_content for d in raw_docs]
        tokenized_corpus = [c.split() for c in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        class HybridRetriever(BaseRetriever):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            # Public field so pydantic is aware of it
            search_kwargs: dict = Field(default_factory=dict)

            # Private attributes (not validated by pydantic)
            _dense_retriever: Any = PrivateAttr()
            _bm25: Any = PrivateAttr()
            _docs: List[Document] = PrivateAttr()
            _reranker: Any = PrivateAttr(default=None)

            def __init__(self, dense_retriever, bm25, docs):
                # Initialize BaseModel with declared fields only
                super().__init__(search_kwargs=dict(getattr(dense_retriever, 'search_kwargs', {})))
                # Set private attrs
                self._dense_retriever = dense_retriever
                self._bm25 = bm25
                self._docs = docs
                # optional neural reranker
                if RERANK_CONFIG.get("use_neural_reranker", False):
                    try:
                        from fastembed import Rerank
                        self._reranker = Rerank(model=RERANK_CONFIG.get("model_name", "BAAI/bge-reranker-small"))
                    except Exception:
                        self._reranker = None

            def _doc_key(self, doc: Document) -> str:
                meta = getattr(doc, 'metadata', {}) or {}
                return f"{meta.get('source','')}|{meta.get('page','')}|{doc.page_content[:80]}"

            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                # Dense results
                dense_docs: List[Document] = self._dense_retriever.get_relevant_documents(query)
                dense_scores = {self._doc_key(doc): (len(dense_docs) - i) for i, doc in enumerate(dense_docs)}

                # BM25 results
                scores = self._bm25.get_scores(query.split())
                bm25_ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                bm_k = RETRIEVAL_CONFIG.get("bm25_k", 30)
                bm25_docs: List[Document] = [self._docs[i] for i in bm25_ranked_idx[:bm_k]]
                bm25_scores = {self._doc_key(doc): scores[i] for i, doc in zip(bm25_ranked_idx[:bm_k], bm25_docs)}

                # Fuse scores per key
                alpha = RETRIEVAL_CONFIG.get("hybrid_alpha", 0.6)
                combined = {}
                for key in set(list(dense_scores.keys()) + list(bm25_scores.keys())):
                    combined[key] = alpha * dense_scores.get(key, 0.0) + (1 - alpha) * bm25_scores.get(key, 0.0)

                ranked_keys = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)
                # Map keys back to a document instance (prefer dense, then bm25)
                by_key = {}
                for d in dense_docs:
                    k = self._doc_key(d)
                    if k not in by_key:
                        by_key[k] = d
                for d in bm25_docs:
                    k = self._doc_key(d)
                    if k not in by_key:
                        by_key[k] = d

                # Optional neural reranker on top-N
                prelim = [by_key[k] for k in ranked_keys if k in by_key]
                if self._reranker:
                    top_n = RERANK_CONFIG.get("top_n", 12)
                    pairs = [(query, d.page_content) for d in prelim[:max(top_n, self.search_kwargs.get("k", 10))]]
                    try:
                        reranked = self._reranker.rerank(pairs)
                        order = [idx for idx, _ in sorted(enumerate(reranked), key=lambda x: -x[1].score)]
                        prelim = [prelim[i] for i in order]
                    except Exception:
                        pass

                k_res = self.search_kwargs.get("k", 10)
                return prelim[:k_res]

        return HybridRetriever(base_retriever, bm25, raw_docs)
    except Exception as e:
        st.warning(f"Hybrid retrieval disabled due to: {e}")
        return base_retriever

def get_conversation_chain(retriever, model_name):
    """Create conversation chain from a retriever and selected local model."""
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.0,  # deterministic for consistency
            top_p=LLM_CONFIG["top_p"],
            top_k=LLM_CONFIG["top_k"]
        )

        PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATES["main_template"],
            input_variables=["context", "question"]
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        return conversation_chain
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {e}")
        return None

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except:
        return []

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def handle_user_input(user_question):
    """Handle user input and generate response with enhanced accuracy"""
    try:
        # Detect user language first
        lang_user = detect_language_simple(user_question)
        # If we have a dominant document language, prefer it for answers
        doc_lang = st.session_state.get('doc_language')
        lang = doc_lang if doc_lang else lang_user
        st.session_state['last_lang'] = lang
        # Detect user intent first
        intent = detect_intent(user_question)
        
        # Handle greetings and simple chat without document retrieval
        if intent == "greeting":
            response_text = generate_greeting_response(lang)
            
            # Update chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append(f"Human: {user_question}")
            st.session_state.chat_history.append(f"Assistant: {response_text}")
            return
            
        elif intent == "simple_chat":
            response_text = generate_simple_chat_response(user_question, lang)
            
            # Update chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append(f"Human: {user_question}")
            st.session_state.chat_history.append(f"Assistant: {response_text}")
            return
        
        # Helper to update memory after each exchange
        def update_memory(user_q: str, bot_a: str):
            try:
                # Append to chat history as plain strings
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append(f"Human: {user_q}")
                st.session_state.chat_history.append(f"Assistant: {bot_a}")

                # Summarize if too long
                total_msgs = len(st.session_state.chat_history)
                if total_msgs > MEMORY_CONFIG.get("summarize_after", 10):
                    # Build summary prompt
                    last_k = MEMORY_CONFIG.get("keep_last", 6)
                    keep = st.session_state.chat_history[-last_k:]
                    earlier = st.session_state.chat_history[:-last_k]
                    messages_str = "\n".join(earlier)
                    prompt = MEMORY_CONFIG.get("summary_prompt", "").format(
                        current_summary=st.session_state.get('memory_summary', ""),
                        messages=messages_str
                    )
                    # Use current model (Ollama) to summarize locally
                    model = st.session_state.model_info.get('model') if hasattr(st.session_state, 'model_info') else None
                    if model and hasattr(st.session_state, 'conversation') and st.session_state.conversation:
                        # Reuse LLM via combine_docs_chain without docs
                        try:
                            llm_only = OllamaLLM(model=model, temperature=0.0, top_p=0.9, top_k=40)
                            summary = llm_only.invoke(prompt)
                            st.session_state.memory_summary = summary.strip()
                            # Keep only last_k recent messages plus summary anchor
                            st.session_state.chat_history = keep
                        except Exception:
                            pass
            except Exception:
                pass

        # For document analysis questions, use enhanced retrieval
        if intent == "document_analysis":
            with st.spinner("🔍 Analyzing documents thoroughly..."):
                # Detect summarization request early
                lower_q = user_question.lower().strip()
                wants_summary = any(k in lower_q for k in ["summarize", "summary", "résume", "resumer", "résumer", "resume", "synthèse", "synthese"])
                if wants_summary:
                    chunks = st.session_state.get('all_chunks') or []
                    if not chunks:
                        st.warning(localize("Aucun document traité pour résumer.", "No processed document available to summarize.", lang))
                        return
                    # Concatenate limited text
                    max_chars = 8000
                    acc = []
                    total = 0
                    for c in chunks:
                        txt = c.page_content.strip()
                        if not txt:
                            continue
                        if total + len(txt) > max_chars:
                            acc.append(txt[: max_chars - total])
                            break
                        acc.append(txt)
                        total += len(txt)
                    corpus = "\n\n".join(acc)
                    # Build a simple summarization prompt
                    if lang == 'fr':
                        sum_prompt = ("Résume de façon structurée le contenu suivant en 6 à 10 puces concises, "
                                      "sans ajouter d'informations non présentes. Utilise des phrases brèves.\n\nTEXTE:\n" + corpus + "\n\nRÉSUMÉ:")
                    else:
                        sum_prompt = ("Provide a structured summary of the following content in 6-10 concise bullet points, "
                                      "without adding any information that is not present. Use brief factual sentences.\n\nTEXT:\n" + corpus + "\n\nSUMMARY:")
                    try:
                        model_name = st.session_state.model_info.get('model') if hasattr(st.session_state, 'model_info') else DEFAULT_MODEL
                        llm_tmp = OllamaLLM(model=model_name, temperature=0.0, top_p=0.9, top_k=40)
                        summary = llm_tmp.invoke(sum_prompt)
                        # Update memory & display
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        st.session_state.chat_history.append(f"Human: {user_question}")
                        st.session_state.chat_history.append(f"Assistant: {summary.strip()}")
                        st.write(bot_template.replace("{{MSG}}", summary.strip()), unsafe_allow_html=True)
                        return
                    except Exception as se:
                        st.error(localize(f"Erreur de résumé: {se}", f"Summarization error: {se}", lang))
                        return
                # Enhance the user query for better retrieval
                enhanced_question = enhance_query(user_question)
                
                # Get existing chat history from session state
                chat_history = st.session_state.get('chat_history', [])
                
                # Convert string format to tuples for LangChain compatibility
                formatted_history = []
                for i in range(0, len(chat_history), 2):
                    if i + 1 < len(chat_history):
                        human_msg = chat_history[i].replace("Human: ", "")
                        ai_msg = chat_history[i + 1].replace("Assistant: ", "")
                        formatted_history.append((human_msg, ai_msg))
                
                # Simple question-type routing
                ql = user_question.lower()
                is_phase_counting = any(p in ql for p in ["how many phases", "combien de phases", "number of phases", "phases are", "phases dans"]) or ("how many" in ql and "phase" in ql)
                is_subject = any(p in ql for p in ["subject", "objet", "what is the subject", "objet du document", "scope", "purpose"])
                # Detect plural/all-docs variant
                is_subject_all = is_subject and any(p in ql for p in [
                    "all documents", "tous les documents", "toutes les documents", "tous les pdf",
                    "each document", "every document", "pour chaque document", "de tous les documents"
                ])
                is_actors = any(p in ql for p in ["actor", "acteurs", "intervenants", "stakeholders", "team", "équipe"])
                is_critical = (any(p in ql for p in ["most important", "critical", "critique", "étape la plus importante"]) 
                               and ("phase" in ql or "step" in ql))
                
                # Get retriever from session state
                retriever = st.session_state.retriever
                
                # For phase counting, use more aggressive retrieval
                if is_phase_counting and hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs["k"] = max(20, RETRIEVAL_CONFIG.get("initial_k", 10))
                    retriever.search_kwargs["fetch_k"] = max(40, RETRIEVAL_CONFIG.get("fetch_k", 20))
                
                # Retrieve documents with timing
                rt0 = time.time()
                retrieved_docs = retriever.get_relevant_documents(enhanced_question)
                rtime = time.time() - rt0
                if METRICS_CONFIG.get('enable') and METRICS_CONFIG.get('collect_retrieval_latency'):
                    st.session_state.setdefault('metrics', {})['retrieval_ms'] = int(rtime * 1000)
                
                # Re-rank documents for better relevance (timed)
                rr0 = time.time()
                reranked_docs = rerank_retrieved_docs(retrieved_docs, user_question)
                rrerank = time.time() - rr0
                if METRICS_CONFIG.get('enable') and METRICS_CONFIG.get('collect_rerank_latency'):
                    st.session_state.setdefault('metrics', {})['rerank_ms'] = int(rrerank * 1000)

                # Confidence gate: estimate evidence score from top docs
                def estimate_evidence_score(docs):
                    if not docs:
                        return 0.0
                    text = "\n".join(d.page_content.lower() for d in docs[:6])
                    cues = ["phase 00", "phase 01", "phase 02", "phase 03", "source", "procedure", "processus", "section"]
                    hits = sum(1 for c in cues if c in text)
                    density = min(1.0, hits / max(4, len(cues)))
                    # Light length prior
                    length = sum(len(d.page_content) for d in docs[:6])
                    len_factor = 1.0 if length > 2000 else 0.3 if length < 600 else 0.6
                    return 0.5 * density + 0.5 * len_factor

                evidence_score = estimate_evidence_score(reranked_docs)
                
                # Create enhanced context
                enhanced_context = create_enhanced_context(reranked_docs, user_question)
                
                # Use specialized handling
                if is_subject:
                    # List subjects for all documents if the question implies it
                    if is_subject_all:
                        idx = st.session_state.get('subject_index') or {}
                        if idx:
                            lines = [localize("Sujets par document:", "Subjects per document:", lang)]
                            for src in sorted(idx.keys()):
                                entry = idx[src]
                                subj = entry.get('subject')
                                pg = entry.get('page')
                                label = localize("source", "source", lang)
                                lines.append(f"- {src}: {subj} [{label}: {src} p.{pg}]")
                            response = {"answer": "\n".join(lines), "source_documents": reranked_docs[:3]}
                            # Update memory and skip single-doc subject flow
                            update_memory(user_question, response['answer'])
                            if 'source_documents' in response and response['source_documents']:
                                with st.expander("📚 Sources utilisées"):
                                    for i, doc in enumerate(response['source_documents'][:3]):
                                        meta = doc.metadata if hasattr(doc, 'metadata') else {}
                                        src = meta.get('source', 'unknown')
                                        page = meta.get('page', '?')
                                        st.text(f"Source {i+1} — {src} (page {page}):\n{doc.page_content[:400]}...")
                            return
                    # Prefer global subject index built at processing time
                    idx = st.session_state.get('subject_index') or {}
                    chosen_src = None
                    chosen = None
                    if idx:
                        counts = {}
                        for d in reranked_docs[:6]:
                            m = getattr(d, 'metadata', {}) or {}
                            src = m.get('source')
                            if src in idx:
                                counts[src] = counts.get(src, 0) + 1
                        if counts:
                            chosen_src = max(counts, key=counts.get)
                            chosen = idx.get(chosen_src)
                        else:
                            # pick any
                            chosen_src, chosen = next(iter(idx.items())) if idx else (None, None)
                    if chosen:
                        ans = localize("Sujet", "Subject", lang) + f": {chosen.get('subject')} [source: {chosen_src} p.{chosen.get('page')}]"
                        response = {"answer": ans, "source_documents": reranked_docs[:3]}
                    else:
                        # fallback: local extraction from retrieved docs
                        sub = extract_subject_from_docs(reranked_docs)
                        if sub:
                            ans = localize("Sujet", "Subject", lang) + f": {sub['subject']} [source: {sub['source']} p.{sub['page']}]"
                        else:
                            ans = localize("Je ne trouve pas le sujet dans les documents fournis.", "I can't find the subject in the provided documents.", lang)
                        response = {"answer": ans, "source_documents": reranked_docs[:3]}

                elif is_actors:
                    acts = extract_actors_from_docs(reranked_docs)
                    if acts:
                        lines = [localize("Acteurs/intervenants principaux:", "Main actors/stakeholders:", lang)]
                        for i, a in enumerate(acts[:12], 1):
                            lines.append(f"{i}. {a['name']} [source: {a['source']} p.{a['page']}]")
                        ans = "\n".join(lines)
                    else:
                        ans = localize("Je ne trouve pas la liste des acteurs dans les documents fournis.", "I can't find the list of actors in the provided documents.", lang)
                    response = {"answer": ans, "source_documents": reranked_docs[:3]}

                elif is_critical:
                    # If docs do not define a single critical phase, state that clearly
                    phase_map = extract_phases_from_docs(reranked_docs)
                    if not phase_map:
                        ans = localize("Les documents ne définissent pas d'étape critique unique.", "The documents do not define a single critical step.", lang)
                    else:
                        ans = localize(
                            ("Les documents ne définissent pas explicitement une phase 'critique'. "
                             "La production/validation (Phases 02–03) apparaissent comme sensibles selon le contexte, "
                             "mais aucune mention formelle n'établit une 'phase la plus importante'."),
                            ("The documents do not explicitly define a 'critical' phase. "
                             "Production/validation (Phases 02–03) appear sensitive in context, "
                             "but no formal source declares a 'most important' phase."),
                            lang)
                    response = {"answer": ans, "source_documents": reranked_docs[:4]}

                elif is_phase_counting:
                    # Create a temporary conversation chain with phase counting prompt (kept for reference)
                    # First, extract phases deterministically from the whole corpus
                    phase_map = build_global_phase_map()
                    # If empty (unlikely), fall back to retrieved docs
                    if not phase_map:
                        phase_map = extract_phases_from_docs(reranked_docs)
                    # If likely incomplete (e.g., missing 03 while 01/02 exist), expand search once
                    if 1 in phase_map and 2 in phase_map and 3 not in phase_map:
                        more_docs = retriever.get_relevant_documents(enhanced_question + " phase 03 production série")
                        more_docs = rerank_retrieved_docs(more_docs, user_question)
                        merge_map = extract_phases_from_docs(more_docs)
                        for k, v in merge_map.items():
                            if k not in phase_map or (v.get("name") and len(v.get("name","")) > len(phase_map[k].get("name",""))):
                                phase_map[k] = v

                    # Build a crisp, grounded answer from the extracted phases
                    if phase_map:
                        sorted_nums = sorted(phase_map.keys())
                        if lang == 'fr':
                            lines = [f"Il y a {len(sorted_nums)} phases principales dans ce projet :"]
                        else:
                            lines = [f"There are {len(sorted_nums)} main phases in this project:"]
                        for i, n in enumerate(sorted_nums, 1):
                            name = phase_map[n].get("name") or ""
                            # Preferred citation from extractor map
                            cite = ""
                            cites = phase_map[n].get('citations', [])
                            if cites:
                                c0 = cites[0]
                                cite = f" [source: {c0.get('source','doc')} p.{c0.get('page','?')}]"
                            if not cite:
                                # Derive a short citation from retrieved docs as fallback
                                for d in reranked_docs:
                                    if f"phase {n:02d}" in d.page_content.lower() or f"phase {n}" in d.page_content.lower():
                                        m = getattr(d, 'metadata', {}) or {}
                                        src = m.get('source', 'doc')
                                        page = m.get('page', '?')
                                        cite = f" [source: {src} p.{page}]"
                                        break
                            if name:
                                if lang == 'fr':
                                    lines.append(f"{i}.  Phase {n:02d} : {name}{cite}")
                                else:
                                    lines.append(f"{i}.  Phase {n:02d}: {name}{cite}")
                            else:
                                lines.append(f"{i}.  Phase {n:02d}{cite}")
                        response = {"answer": "\n".join(lines), "source_documents": reranked_docs[:5]}
                    else:
                        # Fall back to LLM generation with strict instruction
                        instruction = ("CONSIGNE STRICTE: Basez-vous UNIQUEMENT sur les documents fournis. Ne mentionnez que les phases qui existent vraiment. Question: " if lang=='fr' else 
                                       "STRICT INSTRUCTION: Base ONLY on the provided documents. Mention only phases that actually exist. Question: ")
                        response_txt = st.session_state.conversation.combine_docs_chain.run(
                            input_documents=reranked_docs[:8], 
                            question=instruction + user_question + ("\nRéponds en français." if lang=='fr' else "\nAnswer in English.")
                        )
                        response = {"answer": response_txt, "source_documents": reranked_docs[:5]}
                else:
                    # Include memory summary and last turns in the question to improve continuity
                    memory_header = ""
                    if st.session_state.get('memory_summary'):
                        memory_header += ("Contexte de la conversation (résumé): " if lang=='fr' else "Conversation context (summary): ") + f"{st.session_state.memory_summary}\n"
                    # Also include last few turns verbatim
                    tail = st.session_state.chat_history[-MEMORY_CONFIG.get('keep_last', 6):]
                    if tail:
                        memory_header += ("Derniers échanges:\n" if lang=='fr' else "Recent turns:\n") + "\n".join(tail) + "\n"

                    # Light language mirroring hint (prefix instruction)
                    def _detect_lang(txt: str) -> str:
                        fr_hits = sum(1 for w in [" le ", " la ", " les ", " des ", " de ", " et ", " phase ", " combien ", " projet "] if w in (" " + txt.lower() + " "))
                        en_hits = sum(1 for w in [" the ", " and ", " phase ", " how many ", " project "] if w in (" " + txt.lower() + " "))
                        return "fr" if fr_hits >= en_hits else "en"

                    if st.session_state.get('mirror_lang', True):
                        _lang = _detect_lang(user_question)
                        if _lang == "en":
                            memory_header = "Please answer in English.\n" + memory_header
                        else:
                            memory_header = "Réponds en français.\n" + memory_header

                    # Confidence/"no-answer" behavior
                    if GUARDRAIL_CONFIG.get("enable_confidence_gate", True):
                        min_docs = GUARDRAIL_CONFIG.get("min_docs", 2)
                        min_score = GUARDRAIL_CONFIG.get("min_evidence_score", 0.35)
                        if len(reranked_docs) < min_docs or evidence_score < min_score:
                            # Provide helpful fallback with top snippets
                            preview = []
                            for d in reranked_docs[:3]:
                                m = getattr(d, 'metadata', {}) or {}
                                src = m.get('source', 'doc')
                                page = m.get('page', '?')
                                snippet = d.page_content.strip().replace("\n", " ")[:280]
                                preview.append(f"- {src} p.{page}: {snippet}...")
                            msg = "Je ne trouve pas de réponse sûre dans les documents fournis. Voici les passages les plus proches:\n" + "\n".join(preview)
                            response = {"answer": msg, "source_documents": reranked_docs[:3]}
                        else:
                            response = st.session_state.conversation.invoke({
                                'question': memory_header + user_question,
                                'chat_history': formatted_history
                            })
                    else:
                        response = st.session_state.conversation.invoke({
                            'question': memory_header + user_question,
                            'chat_history': formatted_history
                        })
                
                # Validate response completeness (this will catch missing Phase 00)
                is_complete, warning_msg = validate_response_completeness(response['answer'], user_question)
                
                # Display warning if response seems incomplete
                if not is_complete:
                    st.warning(f"⚠️ {warning_msg}")
                    
                    # Try to get more context and regenerate with specific emphasis on Phase 00
                    with st.spinner("🔄 Searching for ALL phases including Phase 00..."):
                        # Enhanced search specifically for Phase 00
                        phase_00_query = f"{enhanced_question} phase00 phase 00 étude chiffrage study estimation"
                        extended_docs = retriever.get_relevant_documents(phase_00_query)
                        extended_context = create_enhanced_context(extended_docs, user_question)
                        
                        # Regenerate with explicit Phase 00 instruction
                        enhanced_response = st.session_state.conversation.invoke({
                            'question': f"IMPORTANT: Liste TOUTES les phases y compris la Phase 00 (Étude et chiffrage). Question: {user_question}",
                            'chat_history': formatted_history
                        })
                        
                        response = enhanced_response
                
                # Reset retriever parameters to normal
                if is_phase_counting and hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs["k"] = RETRIEVAL_CONFIG["initial_k"]
                    retriever.search_kwargs["fetch_k"] = RETRIEVAL_CONFIG["fetch_k"]
                
                # Update memory and chat history
                update_memory(user_question, response['answer'])
                
                # Show source documents for transparency
                if 'source_documents' in response and response['source_documents']:
                    with st.expander("📚 Sources utilisées"):
                        for i, doc in enumerate(response['source_documents'][:3]):
                            meta = doc.metadata if hasattr(doc, 'metadata') else {}
                            src = meta.get('source', 'unknown')
                            page = meta.get('page', '?')
                            st.text(f"Source {i+1} — {src} (page {page}):\n{doc.page_content[:400]}...")
                            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Try fallback approach
        try:
            st.info("🔄 Trying alternative approach...")
            
            # Check if we have a conversation chain available
            if hasattr(st.session_state, 'conversation') and st.session_state.conversation:
                simple_response = st.session_state.conversation.invoke({
                    'question': user_question,
                    'chat_history': []
                })
                
                update_memory(user_question, simple_response['answer'])
            else:
                # If no conversation chain, provide a helpful message
                fallback_msg = "Je suis désolé, mais je n'ai pas encore de documents à analyser. Veuillez d'abord uploader et traiter vos documents PDF."
                update_memory(user_question, fallback_msg)
            
        except Exception as e2:
            st.error(f"❌ Fallback also failed: {str(e2)}")

def main():
    load_dotenv()

    st.set_page_config(page_title="CAVEO Chatbot", page_icon="💬", layout="wide")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory_summary" not in st.session_state:
        st.session_state.memory_summary = ""

    # Custom header (centered)
    st.markdown("""
        <div style='text-align:center; margin-top:-30px; margin-bottom:10px;'>
            <h1 style='font-weight:700; letter-spacing:2px; margin-bottom:4px;'>CAVEO Chatbot</h1>
        </div>
    """, unsafe_allow_html=True)

    # INPUT BAR AT VERY TOP (under title)
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5,1])
            with col1:
                user_question = st.text_input(
                    "Ask a question:",
                    placeholder="Type your question...",
                    label_visibility="collapsed"
                )
            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)
    
    # Check Ollama installation
    if not check_ollama_installation():
        st.error("Ollama not detected. Please install Ollama first.")
        st.info("Download from: https://ollama.ai")
        if st.button("Check Again"):
            st.rerun()
        st.stop()

    # Helper to test if required model is present
    def _ensure_default_model():
        models = get_available_ollama_models()
        if DEFAULT_MODEL not in models:
            st.warning(f"Required model '{DEFAULT_MODEL}' not found. Pull it in a terminal: ollama pull {DEFAULT_MODEL}")
            return False
        return True
    
    # Handle user input (after form submission)
    if 'user_question' not in locals():
        user_question = None
    if submit_button and user_question:
        if st.session_state.conversation:
            handle_user_input(user_question)
            st.rerun()
        else:
            st.warning("Please process documents first.")
    
    # **SIMPLE SIDEBAR FOR DOCUMENTS**
    with st.sidebar:
        st.header("Documents")
        # Uniform width buttons: Reset, Browse, Process
        if st.button("Reset", use_container_width=True):
            for key in ["memory_summary", "chat_history", "conversation", "retriever", "all_chunks", "subject_index", "phase_cache"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.memory_summary = ""
            st.session_state.chat_history = []
            st.success("Reset done")

        pdf_docs = st.file_uploader(
            "Browse PDFs",
            accept_multiple_files=True,
            type="pdf",
            label_visibility="collapsed"
        )

        if st.button("Process", use_container_width=True, disabled=not _ensure_default_model()):
            if not _ensure_default_model():
                st.stop()
            if not pdf_docs:
                st.warning("Select PDFs first")
            else:
                with st.spinner("Processing..."):
                    try:
                        docs = get_pdf_text(pdf_docs)
                        if not docs:
                            st.error("No text found")
                        else:
                            text_chunks = get_text_chunks(docs)
                            st.session_state.all_chunks = text_chunks
                            try:
                                st.session_state.subject_index = build_subject_index(text_chunks, st.session_state.get('pdf_meta', {}))
                            except Exception:
                                st.session_state.subject_index = {}
                            vectorstore = get_vectorstore(text_chunks)
                            if not vectorstore:
                                st.error("Processing failed")
                                return
                            retriever = build_hybrid_retriever(vectorstore, text_chunks)
                            if not _ensure_default_model():
                                return
                            conversation_chain = get_conversation_chain(retriever, DEFAULT_MODEL)
                            if conversation_chain:
                                st.session_state.conversation = conversation_chain
                                st.session_state.retriever = retriever
                                st.session_state.model_info = {'model': DEFAULT_MODEL}
                                st.session_state.phase_cache = None
                                try:
                                    if PERSISTENCE_CONFIG.get("persist_faiss"):
                                        vectorstore.save_local(PERSISTENCE_CONFIG.get("path", ".faiss_index"))
                                except Exception:
                                    pass
                                st.success("Ready.")
                            else:
                                st.error("Setup failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)[:50]}...")
        
    # Show status (outside sidebar)
    if st.session_state.conversation:
        st.caption(f"Model: {DEFAULT_MODEL}")
    
    # **CHAT AREA - MAIN CONTENT**
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    if 'chat_history' in st.session_state and st.session_state.chat_history:
        # Display conversation (most recent first)
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            message = st.session_state.chat_history[i]
            if i % 2 == 0:  # User messages
                user_msg = message.replace("Human: ", "")
                st.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
            else:  # Bot messages
                bot_msg = message.replace("Assistant: ", "")
                st.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
