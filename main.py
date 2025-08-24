import json
import os
import re
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd
import torch
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# PDF file paths
PDF_PATHS = [
    "./content/Capgemini_-_2025-02-25_-_2024_Consolidated_Financial_Statements.pdf",
    "./content/Capgemini_-_2024-02-20_-_2023_Consolidated_Financial_Statements.pdf",
]

# Output paths
CSV_PATH = "./content/capgemini_financial_QA_pairs.csv"
JSONL_PATH = "./content/capgemini_financial_QA_pairs.jsonl"
MEMORY_FILE = "./content/qa_memory_bank.json"

# ----------------- PDF Extraction -----------------
def extract_text_pdfminer(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path) or ""
        if len(text.strip()) < 500:
            st.warning(f"[PDFMiner] Extracted very little text from {path}")
        return text
    except Exception as e:
        st.error(f"[PDFMiner] Failed for {path}: {e}")
        return ""

def extract_text_pymupdf(path: str) -> str:
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        st.error(f"[PyMuPDF] Failed for {path}: {e}")
        return ""

def extract_text_ocr(path: str) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=200)
        text = "\n".join([pytesseract.image_to_string(img) for img in pages])
        return text
    except Exception as e:
        st.error(f"[OCR] Failed for {path}: {e}")
        return ""

def extract_text_safely(path: str) -> str:
    for method, extractor in [
        ("PDFMiner", extract_text_pdfminer),
        ("PyMuPDF", extract_text_pymupdf),
        ("OCR", extract_text_ocr),
    ]:
        text = extractor(path)
        if len(text.strip()) >= 500:
            st.info(f"[Extractor] Using {method} for {path}")
            return text
    st.error(f"Extraction failed for {path}")
    return ""

HEADER_PATTERNS = [r"^CAPGEMINI.*$", r"^\s*Page\s+\d+\s*$", r"^\s*\d+\s*$"]

def clean_text(raw: str) -> str:
    raw = raw.replace("\r", "")
    raw = re.sub(r"-\n", "", raw)
    lines = raw.split("\n")
    cleaned = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines 
               if ln.strip() and not any(re.search(pat, ln, re.I) for pat in HEADER_PATTERNS)]
    return "\n".join(cleaned)

SECTION_TITLES = [
    "Consolidated Income Statement", 
    "Consolidated Statement of Financial Position", 
    "Consolidated Statement of Cash Flows"
]

def locate_sections(text: str) -> Dict[str, Tuple[int, int]]:
    positions = [(m.start(), title) for title in SECTION_TITLES for m in re.finditer(re.escape(title), text, re.I)]
    positions.sort()
    return {
        title: (start, positions[i + 1][0] if i + 1 < len(positions) else len(text))
        for i, (start, title) in enumerate(positions)
    }

NUM_RE = re.compile(r"\(?-?\d[\d,]*\)?")

def strip_parens(n: str) -> float:
    neg = n.startswith("(") and n.endswith(")")
    n = n.strip("()").replace(",", "")
    try:
        val = float(n)
    except:
        val = float("nan")
    return -val if neg else val

def parse_years_from_context(section_text: str) -> List[int]:
    yrs = list(dict.fromkeys(int(y) for y in re.findall(r"(20\d{2})", section_text)))
    return yrs[-3:]

def parse_metric_table(section_text: str, years: Optional[List[int]]) -> Dict[str, Dict[int, float]]:
    lines = section_text.split("\n")[1:]
    metrics = {}
    for ln in lines:
        nums = NUM_RE.findall(ln)
        if len(nums) >= 2:
            metric = re.split(NUM_RE, ln, maxsplit=1)[0].strip(" .:-")
            if not metric: continue
            if not years: years = [0, 1]
            vals = [strip_parens(x) for x in nums[-len(years):]]
            year_map = {y: v for y, v in zip(reversed(years), reversed(vals))}
            metrics[metric] = {**metrics.get(metric, {}), **year_map}
    return metrics

# ----------------- Q/A Generation -----------------
def euro_fmt(x: float) -> str:
    try:
        s = f"{abs(x):,.0f}"
    except:
        return str(x)
    return f"({s})" if x < 0 else f"€{s}"

def make_qa_from_metrics(metrics: Dict[str, Dict[int, float]], section_name: str) -> List[Tuple[str, str]]:
    qa = []
    for metric, year_vals in metrics.items():
        for year, val in sorted(year_vals.items()):
            if year in (0, 1): continue
            if "income" in section_name.lower():
                q = f"What was {metric.lower()} in {year}?"
                a = f"{metric} in {year} was {euro_fmt(val)} million."
            elif "financial" in section_name.lower():
                q = f"What was {metric.lower()} at December 31, {year}?"
                a = f"{metric} at December 31, {year} was {euro_fmt(val)} million."
            elif "cash" in section_name.lower():
                q = f"What was {metric.lower()} in {year}?"
                a = f"{metric} in {year} was {euro_fmt(val)} million."
            qa.append((q, a))
    return qa

def dedupe_keep_order(pairs: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    seen, out = set(), []
    for q,a in pairs:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append((q,a))
    return out

# ----------------- Retrieval -----------------
def preprocess_query(query: str) -> str:
    return re.sub(r'\W+', ' ', query.lower()).strip()

def dense_retrieve(query: str, embedding_model, index, all_chunks_with_metadata, top_k: int = 5) -> List[Dict]:
    q_emb = embedding_model.encode(preprocess_query(query)).reshape(1,-1).astype('float32')
    _, idx = index.search(q_emb, top_k)
    return [all_chunks_with_metadata[i] for i in idx[0]]

def sparse_retrieve(query: str, bm25, all_chunks_with_metadata, top_k: int = 5) -> List[Dict]:
    return bm25.get_top_n(preprocess_query(query).split(), all_chunks_with_metadata, n=top_k)

def hybrid_retrieve(query: str, bm25, embedding_model, index, all_chunks_with_metadata, top_k_dense=5, top_k_sparse=5) -> List[Dict]:
    seen, results = set(), []
    for r in dense_retrieve(query, embedding_model, index, all_chunks_with_metadata, top_k_dense) + sparse_retrieve(query, bm25, all_chunks_with_metadata, top_k_sparse):
        if r['id'] not in seen:
            seen.add(r['id'])
            results.append(r)
    return results

# ----------------- Memory -----------------
def load_memory_bank() -> dict:
    return json.load(open(MEMORY_FILE,"r",encoding="utf-8")) if os.path.exists(MEMORY_FILE) else {}

def save_memory_bank(bank: dict):
    json.dump(bank, open(MEMORY_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def retrieve_from_memory(query: str, memory_bank, top_k=3):
    tokens = set(query.lower().split())
    scored = [(len(tokens & set(q.split()))+data['importance'], q, data['answer']) for q,data in memory_bank.items() if tokens & set(q.split())]
    return sorted(scored,key=lambda x:x[0],reverse=True)[:top_k]

# ----------------- Generation (Fixed with anti-repetition) -----------------
def generate_final_answer(query: str, chunks: list, memory_hits: list, tokenizer_gen, model_gen, top_k=3) -> str:
    context = "\n\n".join([c['text'] for c in chunks[:top_k]]+[f"Q: {q}\nA: {a}" for _,q,a in memory_hits[:top_k]])
    prompt = f"Answer the question based on the following:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer_gen(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model_gen.device)
    
    # Generate with repetition penalty to prevent loops
    output_ids = model_gen.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.7,
        repetition_penalty=1.2,  # Prevent repetition
        no_repeat_ngram_size=3   # No 3-gram repetitions
    )
    
    gen_text = tokenizer_gen.decode(output_ids[0], skip_special_tokens=True)
    answer = gen_text.split("Answer:")[-1].strip()
    
    # Remove any duplicate sentences
    sentences = answer.split('.')
    unique_sentences = []
    for sent in sentences:
        if sent.strip() and sent.strip() not in unique_sentences:
            unique_sentences.append(sent.strip())
    
    return '. '.join(unique_sentences[:3]) + '.' if unique_sentences else answer

def validate_query(query: str) -> bool:
    if re.search(r"kill|suicide|attack|password|credit", query, re.I): return False
    if not any(k in query.lower() for k in ["revenue","income","profit","loss","cash","assets","liabilities","financial"]): return False
    return True

def chunk_text(text, chunk_overlap, size):
    return [text[i:i+size] for i in range(0, len(text), size - chunk_overlap)]

# ----------------- Pipeline -----------------
@st.cache_resource
def load_rag_resources():
    """Cache expensive resources for RAG pipeline"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    GEN_MODEL_NAME = "distilgpt2"
    tokenizer_gen = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    if tokenizer_gen.pad_token is None:
        tokenizer_gen.pad_token = tokenizer_gen.eos_token
    model_gen = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")
    return embedding_model, tokenizer_gen, model_gen

def model_pipeline(query: str):
    all_docs = {}
    for p in PDF_PATHS:
        if os.path.exists(p):
            raw = extract_text_safely(p)
            cleaned = clean_text(raw)
            sections = locate_sections(cleaned)
            all_docs[p] = {"text": cleaned, "sections": sections}
        else:
            st.warning(f"[Missing File] {p}")

    section_key_map = {
        "Consolidated Income Statement": "Income Statement",
        "Consolidated Statement of Financial Position": "Financial Position",
        "Consolidated Statement of Cash Flows": "Cash Flows"
    }

    qa_pairs = []
    for path, info in all_docs.items():
        text = info["text"]
        for k, label in section_key_map.items():
            if k in info["sections"]:
                s, e = info["sections"][k]
                sec_text = text[s:e]
                years = parse_years_from_context(sec_text)
                metrics = parse_metric_table(sec_text, years)
                qa_pairs.extend(make_qa_from_metrics(metrics, label))

    qa_pairs = dedupe_keep_order(qa_pairs)
    pd.DataFrame(qa_pairs, columns=["Q","A"]).to_csv(CSV_PATH, index=False, encoding="utf-8")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for q,a in qa_pairs:
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False)+"\n")

    # Chunk documents
    chunk_size_small, chunk_size_large, chunk_overlap = 100, 400, 20
    document_chunks = {}
    for path, info in all_docs.items():
        text = info["text"]
        document_chunks[path] = {
            "small_chunks": chunk_text(text, chunk_overlap, chunk_size_small),
            "large_chunks": chunk_text(text, chunk_overlap, chunk_size_large)
        }

    all_chunks_with_metadata, cid = [], 0
    for path, chunk_sets in document_chunks.items():
        for size_label, chunks in chunk_sets.items():
            size = chunk_size_small if size_label == "small_chunks" else chunk_size_large
            for chunk in chunks:
                all_chunks_with_metadata.append({
                    "id": f"chunk_{cid}",
                    "source_file": path,
                    "chunk_size": size,
                    "text": chunk
                })
                cid += 1

    embedding_model, tokenizer_gen, model_gen = load_rag_resources()
    
    chunk_texts = [c["text"] for c in all_chunks_with_metadata]
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
    embeddings_array = np.array(chunk_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    tokenized_corpus = [c["text"].split() for c in all_chunks_with_metadata]
    bm25 = BM25Okapi(tokenized_corpus)

    memory_bank = load_memory_bank()
    for q,a in qa_pairs:
        memory_bank[q.lower()] = {"answer": a, "importance": 1}
    save_memory_bank(memory_bank)

    if validate_query(query):
        chunks = hybrid_retrieve(query, bm25, embedding_model, index, all_chunks_with_metadata)
        memory_hits = retrieve_from_memory(query, memory_bank)
        raw_answer = generate_final_answer(query, chunks, memory_hits, tokenizer_gen, model_gen)
        return raw_answer, chunks, memory_hits
    else:
        return "Query rejected by validation.", [], []
    
def input_guardrail(query: str) -> bool:
    """Block irrelevant or harmful queries."""
    blocked_keywords = ["joke", "politics", "violence"]
    return not any(word in query.lower() for word in blocked_keywords)

@st.cache_resource
def load_finetuned_resources():
    """Cache fine-tuned model resources"""
    model_path = "./finetuned_model"
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return tokenizer, model, True
    else:
        return None, None, False

def finetune_pipline(query: str):
    tokenizer, model, ready = load_finetuned_resources()
    if ready == False:
        return
    
    model.eval()

    def generate_answer(question, max_length=100):
        # Tokenize input
        inputs = tokenizer(question, return_tensors="pt")
        prompt = f"Question: {question} Answer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        # Generate output
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove prompt from output (just show the answer)
        answer = output_text[len(prompt):].strip()
        return answer.strip()
    
    return generate_answer(query)


# ----------------- Streamlit UI (Updated with RAG and Fine-tune) -----------------
def main():
    st.set_page_config(page_title="Capgemini Financial Assistant", page_icon="📊", layout="wide")
    
    st.title("📊 Capgemini Financial Assistant")
    st.markdown("Ask questions about Capgemini's financial statements using RAG or Fine-tuned models")
    
    # Sidebar for system info
    with st.sidebar:
        st.header("📊 System Status")
        st.markdown("---")
        
        # Check files
        st.subheader("📁 Files Status")
        files_to_check = [
            ("Q&A CSV", CSV_PATH),
            ("Memory Bank", MEMORY_FILE),
            ("2024 PDF", PDF_PATHS[0]),
            ("2023 PDF", PDF_PATHS[1]),
            ("Fine-tuned Model", "./finetuned_model")
        ]
        
        for name, path in files_to_check:
            if os.path.exists(path):
                if os.path.isfile(path):
                    size = os.path.getsize(path) / 1024
                    st.success(f"✅ {name}: {size:.1f} KB")
                else:
                    st.success(f"✅ {name}: Directory exists")
            else:
                st.warning(f"⚠️ {name}: Not found")
        
        st.markdown("---")
        st.subheader("ℹ️ Information")
        st.markdown("""
        **Available Modes:**
        - **RAG**: Retrieval-Augmented Generation using document chunks and memory
        - **Fine-tuned**: Model specifically trained on Q&A pairs
        
        **Sample Questions:**
        - What was revenue in 2023?
        - What is total equity for 2024?
        - What was net income in 2023?
        - What were total assets in 2024?
        """)
    
    # Main interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input("Enter your financial question:", 
                            placeholder="e.g., What was Capgemini's revenue in 2023?")
    
    with col2:
        mode = st.radio("Choose Model", ["RAG", "Fine-tuned"])
    
    # Example questions
    st.markdown("### 💡 Quick Questions")
    example_cols = st.columns(4)
    examples = [
        "What was revenue in 2023?",
        "What is total equity for 2024?",
        "What was net income in 2023?",
        "What were total assets in 2024?"
    ]
    
    for col, example in zip(example_cols, examples):
        with col:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                query = example
    
    # Run button
    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("⚠️ Please enter a query")
            return
        
        with st.spinner("Generating answer..."):
            if mode == "RAG":
                answer, chunks, memory_hits = model_pipeline(query)
                
                # Display answer
                st.success("✅ **RAG Pipeline Answer**")
                st.markdown(f"### {answer}")
                
                # Display sources in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📄 Retrieved Chunks", expanded=True):
                        for i, c in enumerate(chunks[:3], 1):
                            st.markdown(f"**Chunk {i}**")
                            st.markdown(f"*Source: {os.path.basename(c['source_file'])}*")
                            st.markdown(f"*Size: {c['chunk_size']} chars*")
                            st.text(c['text'][:300] + "...")
                            st.markdown("---")
                
                with col2:
                    with st.expander("🧠 Memory Hits", expanded=True):
                        if memory_hits:
                            for i, (_, q, a) in enumerate(memory_hits[:3], 1):
                                st.markdown(f"**Memory {i}**")
                                st.markdown(f"Q: {q}")
                                st.markdown(f"A: {a}")
                                st.markdown("---")
                        else:
                            st.info("No memory hits found")
            
            else:  # Fine-tuned
                answer = finetune_pipline(query)
                st.success("✅ **Fine-tuned Model Answer**")
                st.markdown(f"### {answer}")
    
    # Additional options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Memory Bank"):
                memory = load_memory_bank()
                st.info(f"Memory bank loaded with {len(memory)} entries")
        
        with col2:
            if st.button("📊 Show Q&A Pairs"):
                if os.path.exists(CSV_PATH):
                    df = pd.read_csv(CSV_PATH)
                    st.dataframe(df.head(10))
                    st.info(f"Total Q&A pairs: {len(df)}")
                else:
                    st.warning("Q&A CSV not found. Run RAG pipeline first.")

if __name__ == "__main__":
    main()