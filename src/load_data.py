from datasets import load_dataset
import pandas as pd
import random
import os

def load_squad(n_samples=150, seed=42):
    random.seed(seed)
    print("Loading SQuAD v2...")
    dataset = load_dataset("squad_v2", split="validation")
    
    # Only answerable questions
    answerable = [x for x in dataset 
                  if x['answers']['text']]
    sampled = random.sample(answerable, n_samples)
    
    records = []
    for item in sampled:
        records.append({
            'id': item['id'],
            'question': item['question'],
            'context': item['context'],
            'gold_answer': item['answers']['text'][0],
            'dataset': 'squad'
        })
    
    df = pd.DataFrame(records)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/squad_sample.csv', index=False)
    print(f"✅ SQuAD: {len(df)} samples saved")
    return df

def load_hotpot(n_samples=150, seed=42):
    random.seed(seed)
    print("Loading HotpotQA...")
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True
    )
    sampled = random.sample(list(dataset), n_samples)
    
    records = []
    for item in sampled:
        # Combine all context passages
        context_parts = []
        for title, sentences in zip(
            item['context']['title'],
            item['context']['sentences']
        ):
            context_parts.append(
                f"{title}: {' '.join(sentences)}"
            )
        
        records.append({
            'id': item['id'],
            'question': item['question'],
            'context': ' '.join(context_parts),
            'gold_answer': item['answer'],
            'dataset': 'hotpot'
        })
    
    df = pd.DataFrame(records)
    df.to_csv('data/hotpot_sample.csv', index=False)
    print(f"✅ HotpotQA: {len(df)} samples saved")
    return df

if __name__ == "__main__":
    squad_df = load_squad(n_samples=150, seed=42)
    hotpot_df = load_hotpot(n_samples=150, seed=42)
    
    print("\n📊 Dataset Summary:")
    print(f"SQuAD samples: {len(squad_df)}")
    print(f"HotpotQA samples: {len(hotpot_df)}")
    print("\nSample SQuAD question:")
    print(squad_df['question'].iloc[0])
    print("\nSample HotpotQA question:")
    print(hotpot_df['question'].iloc[0])