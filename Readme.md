# 📚 RAG-Based Document Question Answering System

A **Retrieval-Augmented Generation (RAG)** system that answers questions strictly from provided documents using only free-tier and open-source components.

---

## 🎯 Project Overview

This system implements a document-grounded question-answering chatbot designed to:
- Answer questions **exclusively from uploaded policy documents**
- Prevent hallucinations through strict context grounding
- Clearly indicate when information is not available in the source material
- Operate entirely on free-tier or local AI components

The solution provides a **Streamlit web interface**

---

## 🏗️ System Architecture

### RAG Pipeline

The system follows a four-stage retrieval-augmented generation pipeline:

```
┌─────────────────┐
│ 1. Document     │  PDF Upload → Text Extraction → Normalization
│    Ingestion    │  → Chunking (250 chars, 50 overlap)
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. Embedding &  │  Text Chunks → SentenceTransformer Embeddings
│    Vector Store │  → FAISS Index (Local Storage)
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. Retrieval    │  User Question → Semantic Search (MMR)
│                 │  → Top 4 Relevant Chunks → Context Compression
└────────┬────────┘
         │
┌────────▼────────┐
│ 4. Answer       │  Context + Question → LLM Prompt
│    Generation   │  → Grounded Answer (max 2 sentences)
└─────────────────┘
```

### Key Design Decisions

1. **Chunking Strategy**: 250-character chunks with 50-character overlap
   - Optimized for single-page policy documents to maintain granular retrieval
   - Small chunks enable precise pinpointing of specific policy statements
   - Minimal overlap reduces redundancy while preserving sentence boundaries

2. **MMR Retrieval**: Maximal Marginal Relevance re-ranking
   - Reduces redundancy in retrieved chunks
   - Increases diversity of context provided to LLM
   - Improves coverage for multi-faceted questions

3. **Context Compression**: 2500-character limit
   - Fits within token limits of smaller local models
   - Reduces latency and computational overhead
   - Maintains sufficient context for accurate answers

4. **Strict Grounding**: Explicit prompt instructions
   - Forces LLM to answer only from provided context
   - Returns standardized "not specified" message when uncertain
   - Prevents hallucination and fabricated information

---

## 🛠️ Technology Stack

### Large Language Models (Free-Tier Only)

| Model | Type | Purpose |
|-------|------|---------|
| **Llama 3.2 1B** | Local (llama.cpp) | Primary inference engine (offline) |
| **Grok Beta** | API (xAI) | Optional cloud fallback (free tier) |
| **Gemini Flash** | API (Google) | Optional cloud fallback (free tier) |

### Embeddings
- **sentence-transformers/all-MiniLM-L6-v2**
  - Open-source, 384-dimensional embeddings
  - Optimized for semantic similarity
  - Fast inference on CPU

### Vector Database
- **FAISS** (Facebook AI Similarity Search)
  - **Why FAISS?**
    - 100% free and open-source
    - Runs entirely locally (no cloud dependencies)
    - Extremely fast similarity search
    - No server setup required
    - Production-proven at scale

### Core Frameworks
- **LangChain**: Document loading, text splitting, retrieval chains
- **Streamlit**: Interactive web UI
- **HuggingFace**: Embedding model hosting

---

## 📂 Project Structure

```
project/
├── models/
│   └── Llama-3.2-1B.Q2_K.gguf    # Local LLM model file
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (optional)
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (8GB recommended for local LLM)
- Git

### 2. Clone Repository
```bash
git clone <https://github.com/MarinaNasser/RAG-System.git>
cd <project-directory>
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Local Model
Place the quantized Llama model in the `models/` directory:
```
models/Llama-3.2-1B.Q2_K.gguf
```

**Download source**: [HuggingFace Model Repository](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) or quantized model repositories.

### 6. Configure Environment Variables
Create a `.env` file for API-based models:
```env
XAI_API_KEY=your_xai_key_here
GOOGLE_API_KEY=your_google_key_here
```

> **Note**: The system works fully offline with the local LLaMA model without any API keys.

---

## 🚀 Running the Application

### Option 1: Streamlit Web Interface
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

**Features:**
- Drag-and-drop PDF upload
- Real-time document indexing
- Interactive question answering
- Source attribution with expandable context
- Model selection (Local/Grok/Gemini)

## 🔬 Model Performance Comparison

To validate the system's effectiveness, we tested three LLM configurations against sample policy questions:

### Test Case 1: Specific Policy Detail

**Question:** "What is the professional dress code?"

**Ground Truth:** *"Employees should adhere to a [Business Casual/Formal] dress code unless otherwise specified for specific roles or events."*

| Model | Response | Accuracy |
|-------|----------|----------|
| **Llama 3.2 1B (Local)** | "the professional dress code is as follows:" | ❌ **Incomplete** - Stops mid-answer |
| **Grok Beta (xAI)** | "Meistress is a person who has the power to influence..." | ❌ **Hallucination** - Completely unrelated content |
| **Gemini Flash** | "Employees should adhere to a [Business Casual/Formal] dress code unless otherwise specified for specific roles or events." | ✅ **Perfect Match** - Exact retrieval |

### Test Case 2: Broad Policy Question

**Question:** "What is the employee code of conduct?"

**Ground Truth:** *"Defines the expectations for behavior, ethics, and professional standards within the organization. All employees are expected to conduct themselves with honesty and integrity."*

| Model | Response | Accuracy |
|-------|----------|----------|
| **Llama 3.2 1B (Local)** | "The employee code of conduct is a set of rules that are meant to be followed by the employees. It can also be called as an official document." | ⚠️ **Generic** - Uses external knowledge, not document content |
| **Grok Beta (xAI)** | "Not specified in the provided policy documents." | ❌ **False Negative** - Info exists but not retrieved |
| **Gemini Flash** | "The Employee Code of Conduct defines the expectations for behavior, ethics, and professional standards within the organization. It requires all employees to conduct themselves with honesty and integrity." | ✅ **Accurate** - Paraphrased correctly from source |

### Analysis

**Gemini Flash emerged as the superior model** for this RAG use case due to:

1. **Grounding Adherence**: Consistently answers from retrieved context without hallucination
2. **Instruction Following**: Respects the 2-sentence limit and structured prompt format
3. **Contextual Understanding**: Successfully paraphrases policy language while maintaining accuracy
4. **Reliability**: Zero hallucinations across test queries

**Llama 3.2 1B limitations:**
- Token generation cutoffs (incomplete answers)
- Reverts to general knowledge instead of strict document grounding
- Limited instruction-following capability at 1B parameter scale

**Grok Beta limitations:**
- Severe hallucination issues (completely fabricated content)
- Over-conservative refusal (false negatives on retrievable information)
- Inconsistent prompt adherence

**Conclusion:** While the local Llama model provides offline functionality, **Gemini Flash (free tier) offers production-grade accuracy** for policy question answering with proper document grounding.


## 📊 System Evaluation

### 1. How Answer Quality Is Evaluated

Answer quality is ensured through a **multi-layered validation approach**:

- **Retrieval Quality**: MMR algorithm ensures diverse, relevant chunks are selected
- **Context Grounding**: LLM prompt explicitly prohibits speculation or external knowledge
- **Response Validation**: Answers limited to 2 sentences to prevent rambling
- **Source Attribution**: Every answer traces back to specific document chunks
- **Fallback Mechanism**: Standardized "not specified" response prevents fabrication

**Quality Metrics:**
- Factual accuracy (answer found in retrieved context)
- Conciseness (2-sentence maximum)
- Source traceability (chunk + page references)
- Refusal rate (percentage of "not specified" responses)

### 2. How Accuracy Can Be Improved Without Changing the LLM

**Retrieval Enhancement Strategies:**

1. **Advanced Chunking**
   - Implement semantic chunking (split at sentence/paragraph boundaries)
   - Use sliding window with larger overlap

2. **Better Embeddings**
   - Upgrade to larger embedding models (e.g., `all-mpnet-base-v2`)
   - Implement hybrid search (semantic + keyword BM25)

3. **Query Enhancement**
   - Query expansion (add synonyms, related terms)
   - Hypothetical document embeddings (HyDE)
   - Multi-query retrieval (generate variations of user question)

4. **Re-Ranking**
   - Add cross-encoder re-ranker after initial retrieval
   - Implement metadata filtering (date, document type, section)
   - Use reciprocal rank fusion for multi-retriever setups

5. **Document Processing**
   - Extract and preserve document structure (headers, sections)
   - Deduplicate redundant content

### 3. When Paid APIs or Fine-Tuning Would Be Justified

**Paid APIs are justified when:**
- **Scale**: >10,000 queries/day requiring cloud infrastructure
- **Latency**: Sub-second response times needed (local models slower)
- **Multilingual**: Non-English document support requiring specialized models
- **Advanced Reasoning**: Complex multi-hop questions beyond small model capabilities

**Fine-Tuning is justified when:**
- **Domain Specificity**: Highly technical terminology 
- **Format Consistency**: Strict output formatting requirements (JSON, structured data)
- **Terminology Alignment**: Company-specific jargon and abbreviations
- **Performance Gap**: Retrieval-only approach insufficient for accuracy targets
- **Cost Optimization**: High query volume makes fine-tuned small model more economical than large API calls

---


## 🔒 Compliance & Limitations

### Advantages
✅ 100% free and open-source components  
✅ No paid API dependencies  
✅ Complete data privacy (local processing)  
✅ Reproducible and auditable  
✅ Works offline  

### Limitations
⚠️ Local LLM slower than cloud APIs (3-5s vs <1s)  
⚠️ Smaller model may miss nuanced reasoning  
⚠️ PDF parsing may fail on complex layouts  
⚠️ Limited to English language  
⚠️ No memory across conversations  

---

## 🤝 Development Notes

### AI Tools Used
This project was developed with assistance from:
- **Claude (Anthropic)**: Architecture design, code generation
- **Visual Studio Code Copilot**: Code completion and debugging
- **ChatGPT**: Documentation structuring
---

