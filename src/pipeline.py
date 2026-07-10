import os
import time
import json
import hashlib
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ─── CACHE ───────────────────────────────────────────
os.makedirs("cache", exist_ok=True)

def get_cache_key(question, context, embedding_name, chunk_size, use_reranking, generator_model):
    """
    Cache key mein woh SAB kuch hona chahiye jo answer ko affect kar sakta hai --
    sirf context nahi, warna alag configs same cached answer chura lenge.
    """
    raw = f"{question}|{context}|{embedding_name}|{chunk_size}|{use_reranking}|{generator_model}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return f"cache/{h}.json"

def load_cache(key):
    if os.path.exists(key):
        with open(key) as f:
            return json.load(f)
    return None

def save_cache(key, data):
    with open(key, "w") as f:
        json.dump(data, f)

# ─── EMBEDDINGS ──────────────────────────────────────
def get_embeddings(model_name):
    if model_name == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    elif model_name == "baai":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(f"Unknown embedding: {model_name}")

# ─── RERANKER ────────────────────────────────────────
def get_reranker():
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(
            "BAAI/bge-reranker-base",
            use_fp16=False
        )
        return reranker
    except Exception as e:
        print(f"Reranker load error: {e}")
        print("WARNING: reranking will silently no-op (fall back to top-3 "
              "un-reranked) for every question until this is fixed.")
        return None

def rerank_docs(reranker, query, docs, top_k=3):
    if reranker is None or not docs:
        return docs[:top_k]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs)

    if isinstance(scores, float):
        scores = [scores]

    scored = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    return [doc for _, doc in scored[:top_k]]

# ─── SPLITTER ────────────────────────────────────────
def get_splitter(chunk_size, overlap_pct=0.15):
    overlap = int(chunk_size * overlap_pct)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

# ─── RETRIEVAL METRICS ───────────────────────────────
def compute_retrieval_metrics(retrieved_docs, context, gold_answer, k=5):
    """
    Hit@k: kya retrieved chunks mein actual GOLD ANSWER text hai?

    (Pehle ye context ke sirf shuru ke 100 chars check karta tha --
    galat signal deta tha, kyunki answer passage mein kahin bhi ho
    sakta hai, sirf shuru mein nahi. Ab jo chunk asal mein answer
    contain karta hai, use retrieve karna hi "hit" maana jayega --
    jo ki retrieval quality ka sahi signal hai.)
    """
    retrieved_text = " ".join(
        [doc.page_content for doc in retrieved_docs]
    ).lower()

    gold_answer_clean = (gold_answer or "").lower().strip()
    hit = 1 if gold_answer_clean and gold_answer_clean in retrieved_text else 0

    return {
        "hit_at_k": hit,
        "num_retrieved": len(retrieved_docs)
    }

# ─── MAIN PIPELINE ───────────────────────────────────
def run_single_experiment(
    row,
    chunk_size,
    embedding_name,
    use_reranking,
    llm,
    generator_model_name,
    reranker=None,
    top_k=5,
    top_k_rerank=3,
    use_cache=True
):
    start_time = time.time()

    # ── Cache check ──
    # Poori (question, context, config) combination ke liye cache --
    # isse crash ke baad re-run pe already-done rows dobara call nahi
    # hongi, aur Groq rate limits pe bhi pressure kam hoga.
    cache_key = get_cache_key(
        row['question'],
        row['context'],
        embedding_name,
        chunk_size,
        use_reranking,
        generator_model_name
    )

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            cached["latency_ms"] = 0.0   # cache hit -- effectively free/instant
            cached["was_cached"] = True
            return cached

    # Split
    splitter = get_splitter(chunk_size)
    chunks = splitter.split_text(row['context'])

    # Embed + Index
    embeddings = get_embeddings(embedding_name)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Retrieve
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    retrieved_docs = retriever.invoke(row['question'])

    # Rerank (optional)
    if use_reranking and reranker:
        final_docs = rerank_docs(
            reranker,
            row['question'],
            retrieved_docs,
            top_k=top_k_rerank
        )
    else:
        final_docs = retrieved_docs[:top_k_rerank]

    # Retrieval metrics
    ret_metrics = compute_retrieval_metrics(
        final_docs,
        row['context'],
        row['gold_answer']
    )

    # Generate answer
    context_text = "\n\n".join(
        [doc.page_content for doc in final_docs]
    )

    prompt = PromptTemplate.from_template(
        """Answer the question using only the 
context below. Be concise.

Context: {context}

Question: {question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_text,
        "question": row['question']
    })

    elapsed = time.time() - start_time

    result = {
        "question": row['question'],
        "gold_answer": row['gold_answer'],
        "generated_answer": answer,
        "chunk_size": chunk_size,
        "embedding": embedding_name,
        "reranking": use_reranking,
        "latency_ms": round(elapsed * 1000, 2),
        "hit_at_k": ret_metrics["hit_at_k"],
        "num_docs_retrieved": ret_metrics["num_retrieved"],
        "was_cached": False
    }

    if use_cache:
        save_cache(cache_key, result)

    return result