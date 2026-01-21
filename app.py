import streamlit as st
import os
import tempfile
from pathlib import Path
import re
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =========================================================
# Configuration
# =========================================================
MODEL_PATH = os.path.join("models", "Llama-3.2-1B.Q2_K.gguf")

XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# Model Registries
# =========================================================
GROK_MODELS = {
    "Grok-4-0709": "grok-4-0709",
}

GEMINI_MODELS = {
    "Gemini 3 Flash": "gemini-3-flash-preview",
}

# =========================================================
# Imports
# =========================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google import genai

# =========================================================
# Prompts
# =========================================================
LLAMA_PROMPT = """You are a policy assistant.

Answer the QUESTION using ONLY the POLICY TEXT.

If the answer is not explicitly stated, respond EXACTLY with:
Not specified in the provided policy documents.

RULES:
- Maximum 2 sentences
- Do NOT ask questions
- Do NOT explain
- Do NOT repeat the question

POLICY TEXT:
{context}

QUESTION:
{question}

ANSWER:"""

GROK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an enterprise policy assistant. "
     "Answer strictly from the provided policy text. "
     "If the answer is not explicitly stated, reply exactly with: "
     "'Not specified in the provided policy documents.' "
     "Maximum 2 sentences. Do not explain."),
    ("human",
     "POLICY TEXT:\n{context}\n\nQUESTION:\n{question}\n\nFINAL ANSWER:")
])

GEMINI_PROMPT = """
You are an enterprise policy assistant.

Answer the QUESTION using ONLY the POLICY TEXT.

If the answer is not explicitly stated, respond EXACTLY with:
Not specified in the provided policy documents.

RULES:
- Maximum 2 sentences
- Do NOT ask questions
- Do NOT explain
- Do NOT repeat the question

POLICY TEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Digital Hub Policy Assistant",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# Utility Functions
# =========================================================
def normalize_policy_text(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compress_context(docs, max_chars: int = 2500) -> str:
    combined = ""
    for doc in docs:
        if len(combined) + len(doc.page_content) > max_chars:
            break
        combined += doc.page_content + "\n\n"
    return combined.strip()

# =========================================================
# Title
# =========================================================
st.title("📚 Digital Hub Policy Assistant")
st.markdown(
    "Ask questions about **Employee Code of Conduct**, "
    "**Remote Work Policy**, and **Performance Management** documents."
)

# =========================================================
# Session State
# =========================================================
for key in [
    "vector_store",
    "llm",
    "loaded_files",
    "model_choice",
    "gemini_client",
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "loaded_files" else None

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.subheader("⚙️ Settings")

    st.session_state.model_choice = st.radio(
        "Choose model",
        ["Local LLaMA"]
        + list(GROK_MODELS.keys())
        + list(GEMINI_MODELS.keys())
    )

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=150,
        max_value=400,
        value=250,
        step=50
    )

    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=100,
        value=50,
        step=25
    )


    st.markdown("---")
    st.subheader("📄 Uploaded Files")

    if st.session_state.loaded_files:
        for f in st.session_state.loaded_files:
            st.write(f"✅ {f}")
    else:
        st.info("No documents uploaded")

# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(["📤 Upload & Index", "❓ Ask a Question"])

# =========================================================
# Upload & Index
# =========================================================
with tab1:
    uploaded_files = st.file_uploader(
        "Upload policy PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Indexing documents..."):
            documents = []
            temp_dir = tempfile.mkdtemp()

            for file in uploaded_files:
                path = Path(temp_dir) / file.name
                path.write_bytes(file.getbuffer())

                loader = PyPDFLoader(str(path))
                docs = loader.load()

                for d in docs:
                    d.page_content = normalize_policy_text(d.page_content)

                documents.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)

            # Initialize selected model
            choice = st.session_state.model_choice

            if choice == "Local LLaMA":
                st.session_state.llm = LlamaCpp(
                    model_path=MODEL_PATH,
                    temperature=0.0,
                    max_tokens=80,
                    n_ctx=4096,
                    repeat_penalty=1.2,
                    verbose=False
                )

            elif choice in GROK_MODELS:
                st.session_state.llm = ChatOpenAI(
                    model=GROK_MODELS[choice],
                    api_key=XAI_API_KEY,
                    base_url="https://api.x.ai/v1",
                    temperature=0.7,
                    max_tokens=500
                )

            elif choice in GEMINI_MODELS:
                os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
                st.session_state.gemini_client = genai.Client()

            st.session_state.loaded_files = [f.name for f in uploaded_files]
            st.success("✅ Documents indexed successfully")

# =========================================================
# Ask Question (RAG)
# =========================================================
with tab2:
    if not st.session_state.vector_store:
        st.warning("Please upload and index documents first.")
        st.stop()

    question = st.text_area(
        "Ask a question",
        placeholder="What is the professional dress code?"
    )

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 16}
            )

            docs = retriever.invoke(question)
            context = compress_context(docs)

            choice = st.session_state.model_choice
            answer = None

            # -------- Local LLaMA --------
            if choice == "Local LLaMA":
                try:
                    prompt = PromptTemplate(
                        template=LLAMA_PROMPT,
                        input_variables=["context", "question"]
                    )
                    chain = prompt | st.session_state.llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": question})
                    answer = answer.split("\n")[0].strip()
                except Exception as e:
                    st.error(f"❌ Error with Llama: {str(e)}")

            elif choice in GROK_MODELS:
                if not XAI_API_KEY:
                    st.error("❌ XAI_API_KEY environment variable not set. Please configure it.")
                else:
                    try:
                        # Use a simpler prompt format
                        simple_prompt = f"""You are an enterprise policy assistant.

Answer the following question using ONLY the provided policy text.
If the answer is not explicitly stated, reply: "Not specified in the provided policy documents."

POLICY TEXT:
{context}

QUESTION: {question}

ANSWER:"""
                        
                        response = st.session_state.llm.invoke(simple_prompt)
                        
                        if hasattr(response, 'content'):
                            answer = response.content
                        else:
                            answer = str(response)
                        
                        answer = answer.strip()
                        if "\n" in answer:
                            answer = answer.split("\n")[0].strip()
                    except Exception as e:
                        st.error(f"❌ Error with Grok: {str(e)}")

            # -------- Gemini --------
            elif choice in GEMINI_MODELS:
                prompt_text = GEMINI_PROMPT.format(
                    context=context,
                    question=question
                )
                answer = None
                if not GEMINI_API_KEY:
                    st.error("❌ GEMINI_API_KEY environment variable not set. Please configure it.")
                else:
                    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
                    client = genai.Client()

                    response = client.models.generate_content(
                        model=GEMINI_MODELS[choice],
                        contents=prompt_text
                    )

                    answer = response.text.strip()
                    answer = answer.split("\n")[0].strip()

            # -------- Output --------
            st.markdown("### 📝 Answer")
            if answer:
                st.write(answer)

            st.markdown("### 📚 Sources Used")
            for i, doc in enumerate(docs, 1):
                st.markdown(
                    f"**Source {i}** — Page {doc.metadata.get('page', '?')}"
                )
                with st.expander("View source text"):
                    st.write(doc.page_content[:800])


