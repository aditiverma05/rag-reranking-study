import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import pandas as pd
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline import (
    get_embeddings, get_splitter, get_reranker,
    rerank_docs, compute_retrieval_metrics
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_one_config(
    df, chunk_size, emb_name,
    use_rerank, llm, reranker,
    seed, dataset_name
):
    results = []
    
    # Seed ke liye sample
    sampled = df.sample(
        n=min(50, len(df)),
        random_state=seed
    )
    
    splitter = get_splitter(chunk_size)
    embeddings = get_embeddings(emb_name)
    
    prompt = PromptTemplate.from_template(
        """Answer using only the context below. Be concise.

Context: {context}
Question: {question}
Answer:"""
    )
    
    for _, row in tqdm(
        sampled.iterrows(),
        total=len(sampled),
        desc=f"chunk={chunk_size}|emb={emb_name}|rerank={use_rerank}|seed={seed}"
    ):
        try:
            start = time.time()
            
            # Chunk + Index
            chunks = splitter.split_text(row['context'])
            if not chunks:
                continue
                
            vectorstore = FAISS.from_texts(
                chunks, embeddings
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            
            # Retrieve
            retrieved = retriever.invoke(row['question'])
            
            # Rerank
            if use_rerank and reranker:
                final_docs = rerank_docs(
                    reranker,
                    row['question'],
                    retrieved,
                    top_k=3
                )
            else:
                final_docs = retrieved[:3]
            
            # Retrieval metric
            ret_metric = compute_retrieval_metrics(
                final_docs, row['context']
            )
            
            # Generate
            context_text = "\n\n".join(
                [d.page_content for d in final_docs]
            )
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context_text,
                "question": row['question']
            })
            
            elapsed = round((time.time() - start) * 1000, 2)
            
            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "chunk_size": chunk_size,
                "embedding": emb_name,
                "reranking": use_rerank,
                "question": row['question'],
                "gold_answer": row['gold_answer'],
                "generated_answer": answer,
                "hit_at_k": ret_metric["hit_at_k"],
                "latency_ms": elapsed
            })
            
            # Rate limit — 4 sec between calls
            time.sleep(4)
            
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    return results

def main():
    config = load_config()
    df = pd.read_csv("data/squad_sample.csv")
    
    print("=" * 50)
    print("DAY 3 — SQuAD Experiments")
    print("=" * 50)
    print(f"Dataset size: {len(df)}")
    print(f"Configs: 3 chunk × 2 emb × 2 rerank = 12")
    print(f"Seeds: {config['experiment']['seeds']}")
    print(f"Questions per config per seed: 50")
    print(f"Total runs: 12 × 3 seeds × 50 = 1,800")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0
    )
    reranker = get_reranker()
    
    all_results = []
    os.makedirs("results", exist_ok=True)
    
    chunk_sizes = config['experiment']['chunk_sizes']
    emb_names = [e['name'] for e in config['embeddings']]
    rerank_opts = config['reranking']['enabled']
    seeds = config['experiment']['seeds']
    
    total_configs = (
        len(chunk_sizes) * len(emb_names) *
        len(rerank_opts) * len(seeds)
    )
    done = 0
    
    for seed in seeds:
        for chunk_size in chunk_sizes:
            for emb_name in emb_names:
                for use_rerank in rerank_opts:
                    done += 1
                    print(f"\n[{done}/{total_configs}] "
                          f"seed={seed} chunk={chunk_size} "
                          f"emb={emb_name} rerank={use_rerank}")
                    
                    batch = run_one_config(
                        df=df,
                        chunk_size=chunk_size,
                        emb_name=emb_name,
                        use_rerank=use_rerank,
                        llm=llm,
                        reranker=reranker,
                        seed=seed,
                        dataset_name="squad"
                    )
                    all_results.extend(batch)
                    
                    # Save checkpoint
                    pd.DataFrame(all_results).to_csv(
                        "results/squad_results.csv",
                        index=False
                    )
                    print(f"Checkpoint: {len(all_results)} results saved")
    
    print("\n" + "=" * 50)
    print(f"DONE! Total results: {len(all_results)}")
    print("Saved to results/squad_results.csv")
    
    # Quick summary
    results_df = pd.DataFrame(all_results)
    print("\nQuick Summary:")
    summary = results_df.groupby(
        ['chunk_size', 'embedding', 'reranking']
    )['hit_at_k'].mean().reset_index()
    print(summary.to_string())

if __name__ == "__main__":
    main()