import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
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
    seed, dataset_name, n_questions=50
):
    results = []

    sampled = df.sample(
        n=min(n_questions, len(df)),
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
                final_docs, row['context'], row['gold_answer']
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

            time.sleep(1)

        except Exception as e:
            print(f"\nError: {e}")
            continue

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick sanity check: 1 seed x 1 chunk size x 5 questions "
             "(both rerank settings), ~30-60 sec, before the full sweep."
    )
    args = parser.parse_args()

    config = load_config()
    df = pd.read_csv("data/squad_sample.csv")

    chunk_sizes = config['experiment']['chunk_sizes']
    emb_names = [e['name'] for e in config['embeddings']]
    rerank_opts = config['reranking']['enabled']
    seeds = config['experiment']['seeds']

    n_questions = 50
    if args.smoke_test:
        chunk_sizes = chunk_sizes[:1]
        seeds = seeds[:1]
        n_questions = 5
        print("*** SMOKE TEST MODE: 1 seed, 1 chunk size, "
              f"{n_questions} questions per config ***\n")

    num_configs = len(chunk_sizes) * len(emb_names) * len(rerank_opts)
    total_configs = num_configs * len(seeds)
    total_runs = total_configs * n_questions

    print("=" * 50)
    print("DAY 3 — SQuAD Experiments")
    print("=" * 50)
    print(f"Dataset size: {len(df)}")
    print(f"Configs: {len(chunk_sizes)} chunk × {len(emb_names)} emb × "
          f"{len(rerank_opts)} rerank = {num_configs}")
    print(f"Seeds: {seeds}")
    print(f"Questions per config per seed: {n_questions}")
    print(f"Total runs: {num_configs} × {len(seeds)} seeds × {n_questions} = {total_runs}")

    llm = ChatGroq(
        model=config['generator']['model'],
        temperature=0.0
    )
    reranker = get_reranker()

    all_results = []
    os.makedirs("results", exist_ok=True)

    out_path = "results/squad_smoketest.csv" if args.smoke_test else "results/squad_results.csv"

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
                        dataset_name="squad",
                        n_questions=n_questions
                    )
                    all_results.extend(batch)

                    pd.DataFrame(all_results).to_csv(out_path, index=False)
                    print(f"Checkpoint: {len(all_results)} results saved to {out_path}")

    print("\n" + "=" * 50)
    print(f"DONE! Total results: {len(all_results)}")
    print(f"Saved to {out_path}")

    results_df = pd.DataFrame(all_results)
    print("\nQuick Summary:")
    summary = results_df.groupby(
        ['chunk_size', 'embedding', 'reranking']
    )['hit_at_k'].mean().reset_index()
    print(summary.to_string())

    if args.smoke_test:
        print("\n*** SMOKE TEST COMPLETE ***")
        print("If you see real hit_at_k numbers and generated_answer text above")
        print("(not all errors), embeddings + reranker + Groq are all working.")
        print(f"Now run the full sweep: python src/squad_experiments.py")

if __name__ == "__main__":
    main()