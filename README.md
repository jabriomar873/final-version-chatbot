# 🧠 CAVEO Chatbot – Local PDF Intelligence

CAVEO Chatbot lets you load multiple PDF documents (text or scanned) and interrogate them using a single local LLM (locked: `llama3.1:8b-instruct-q4_K_M`).
It combines hybrid retrieval (FastEmbed + BM25 + optional neural re‑ranking) with deterministic phase / subject / actor extraction, caching, guardrails, and a minimalist dark UI.

Everything runs **100% locally** – no external API calls; documents, embeddings, and answers stay on your machine.

## ✨ Core Features

- **Multi‑PDF ingestion** – drag & browse multiple files; handles text & scanned pages (OCR fallback)  
- **Hybrid retrieval** – FastEmbed dense vectors + BM25 fusion + optional neural re‑rank + citation context  
- **Embedding cache** – Deterministic content hashing → skip recomputation on re‑processing same docs  
- **Deterministic extractors** – Subjects, phases (00..), actors, critical context mapping  
- **Conversation memory + summarization** – Rolling summary + on‑demand structured document summary  
- **Confidence guardrails** – Low‑evidence fallback with top snippets instead of hallucination  
- **Language enforcement** – Answers in dominant document language (with question language fallback)  
- **Watermark branding** – Subtle centered translucent “CAVEO” background  
- **Minimal dark UI** – Only three controls: Reset · Browse · Process  
- **Metrics instrumentation** – Tracks embedding build time + retrieval & rerank latency (UI panel WIP)  
- **Fully offline** – No external APIs; CPU friendly (TF‑IDF fallback if embeddings unavailable)  

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+** (tested on 3.12)  
2. **Ollama** installed and running  
3. **Tesseract OCR** (only needed for scanned PDFs)  

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd CAVEO-chat-main
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (if not already installed)
   - Download from: https://ollama.ai
   - Follow the installation instructions for your OS

4. **Pull Required Model (mandatory)**
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```
   (The app is locked to this model – adjust `DEFAULT_MODEL` in `app.py` if you want another.)

5. **Install Tesseract OCR** (for scanned PDF support)
   
   **Windows:**
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

### Running the Application

Use Streamlit directly:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. **Browse** – Select one or more PDF files (scanned or text).  
2. **Process** – Builds embeddings, hybrid index, subject & phase maps.  
3. **Chat** – Ask domain or structural questions (e.g., “How many phases?”, “List actors”, “Subject of all documents”) or just freeform explore.  
4. **Reset** – Clears memory + indexes (safe to re‑process new sets).  

## 🏗️ Architecture

```
```text
PDFs -> Extract (pypdf / OCR fallback) -> Chunk -> FastEmbed + TF-IDF/BM25 Hybrid -> Re-rank -> Structured context builder
        |                                                                                |
        +--> Subject / Phase / Actor extraction (deterministic)  ←-----------------------+

User Query -> Intent detection -> (Enhanced query) -> Hybrid Retrieval -> Guardrails -> LLM answer + citations
```
```

## 📁 Project Structure

```
CAVEO-chat-main/
├── app.py                # Main application (retrieval pipeline + UI + memory)
├── config.py             # All configuration dictionaries (retrieval, LLM, guardrails, caching, metrics)
├── enhanced_retrieval.py # Query enhancement, rerank, context builders, structured extractors
├── intent_detection.py   # Intent classification + localized small‑talk responses
├── cache_utils.py        # Embedding/document cache helpers (hashing + serialize vectors)
├── htmlTemplates.py      # Chat UI templates, watermark, CSS (incl. citation highlight class)
├── requirements.txt      # Dependencies (core + optional)
├── start_app.bat         # Windows launcher convenience script
├── .streamlit/config.toml# Dark theme enforcement
└── scripts/              # Utility scripts / future evaluation tooling
```

## 🔧 Configuration

### Model

Locked to: `llama3.1:8b-instruct-q4_K_M` (change `DEFAULT_MODEL` constant in `app.py` if needed).  
Reason: balanced quality vs. quantized memory footprint for local CPU/GPU laptops.

### OCR Configuration

Tesseract OCR is automatically detected in these locations (Windows):
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- System PATH

### Environment Tweaks
Set in code to suppress noisy logs / speed up imports (torch profiling off, warning filters). No manual action required.

## 🛠️ Troubleshooting

### Common Issues

**1. "Ollama not detected"**
- Ensure Ollama is installed and running
- Try running `ollama list` in terminal to verify installation

**2. "No Ollama models found"**
- Install a model: `ollama pull llama3.2:1b`
- Restart the application

**3. "Tesseract OCR is not installed" (for scanned PDFs)**
- Install Tesseract OCR using the instructions above
- Restart the application

**4. "No text found in PDF"**
- The PDF might be password-protected
- Try a different PDF file
- For scanned PDFs, ensure Tesseract is installed

### Performance Notes

- First processing run creates embeddings; subsequent queries are fast.
- If FastEmbed fails offline, automatic TF-IDF fallback engages.
- Hybrid retrieval may re-rank top-N; adjust constants in `config.py` if you need speed over relevance.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Dependencies

### Core
- Streamlit, LangChain, Ollama, FAISS, FastEmbed, scikit-learn, rank-bm25

### PDF / OCR
- pypdf, PyMuPDF (fallback OCR image generation), pytesseract, Pillow

### Utilities
Optional (planned / experimental): `qdrant-client`, `tqdm`, `diskcache`
- python-dotenv, numpy, pydantic

## 📋 System Requirements

### Minimum
- OS: Windows 10+, macOS 12+, or modern Linux
- RAM: 6GB (8GB+ recommended)
- Storage: ~2GB + model size
- Python: 3.10+

### Recommended
- RAM: 16GB for large document sets
- CPU: 6+ cores
- Storage: 5GB+ (multiple models / indexes)

## 🔐 Privacy & Security

- **100% Local Processing**: No data is sent to external servers
- **Offline Capable**: Works completely offline once set up
- **No Data Storage**: Documents are processed in memory only
- **Open Source**: Full transparency of code and functionality

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify Ollama and Tesseract installations
4. Check the terminal/console for error messages

## 📜 License

This project is open source. Please check the license file for details.

---

## 🗺️ Roadmap (Short‑Term)

- [ ] Sidebar metrics panel (embed build, retrieval, rerank timing)
- [ ] Inline citation highlighting logic (function wrapping answer segments)
- [ ] Evaluation script (`scripts/eval_sample.py`) for batch Q&A scoring
- [ ] Download/export chat with source citations (Markdown)
- [ ] Confidence badge + answer provenance block

## ✅ Completed Highlights
- Hybrid retrieval + rerank
- Deterministic phase / subject / actor extraction
- Embedding cache (content hash)
- Language enforcement (document dominant)
- Summarization path (explicit “summarize” intent)
- Guardrail fallback with evidence snippets
- Watermark + minimalist dark UI

**Made with ❤️ by/for CAVEO – secure, local document intelligence.**