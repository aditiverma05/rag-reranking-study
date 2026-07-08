import yaml
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline import get_reranker, run_single_experiment

load_dotenv()

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_experiments(dataset_name, df, config):
    """Run all 12 configs on a dataset"""
    
    llm = ChatGoogleGenerativeAI(
        model=config['generator']['model'],
        temperature=config['generator']['temperature']
    )
    
    reranker = get_reranker()
    
    chunk_sizes = config['experiment']['chunk_sizes']
    embeddings = [e['name'] for e in config['embeddings']]
    reranking_options = config['reranking']['enabled']
    
    all_results = []
    
    # Total = 3 chunk × 2 embed × 2 rerank = 12 configs
    total = len(chunk_sizes) * len(embeddings) * len(reranking_options)
    print(f"\nRunning {total} configs on {dataset_name}...")
    print(f"Dataset size: {len(df)} questions")
    
    config_num = 0
    for chunk_size in chunk_sizes:
        for emb_name in embeddings:
            for use_rerank in reranking_options:
                config_num += 1
                config_label = (
                    f"chunk={chunk_size} | "
                    f"emb={emb_name} | "
                    f"rerank={use_rerank}"
                )
                print(f"\n[{config_num}/{total}] {config_label}")
                
                config_results = []
                for _, row in tqdm(
                    df.iterrows(),
                    total=len(df),
                    desc=config_label[:30]
                ):
                    try:
                        result = run_single_experiment(
                            row=row,
                            chunk_size=chunk_size,
                            embedding_name=emb_name,
                            use_reranking=use_rerank,
                            llm=llm,
                            reranker=reranker if use_rerank else None,
                            top_k=config['retrieval']['top_k'],
                            top_k_rerank=config['retrieval']['top_k_after_rerank']
                        )
                        result['dataset'] = dataset_name
                        config_results.append(result)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                
                all_results.extend(config_results)
                
                # Save after each config (safe checkpoint)
                os.makedirs('results', exist_ok=True)
                pd.DataFrame(all_results).to_csv(
                    f'results/{dataset_name}_results.csv',
                    index=False
                )
                print(f"Checkpoint saved: {len(all_results)} results")
    
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    config = load_config()
    
    # Load datasets
    squad_df = pd.read_csv('data/squad_sample.csv')
    hotpot_df = pd.read_csv('data/hotpot_sample.csv')
    
    print("=" * 50)
    print("EXPERIMENT RUNNER")
    print("=" * 50)
    print(f"Configs per dataset: 12")
    print(f"Questions per dataset: {len(squad_df)}")
    print(f"Total runs: 12 × 2 datasets = 24 configs")
    print(f"Seeds: {config['experiment']['seeds']}")
    
    # Test with 3 questions first (dry run)
    print("\nDRY RUN with 3 questions...")
    small_squad = squad_df.head(3)
    results = run_experiments("squad_dryrun", small_squad, config)
    print(f"\nDry run complete! {len(results)} results")
    print(results[['chunk_size','embedding','reranking',
                   'hit_at_k','latency_ms']].to_string())