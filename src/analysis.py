import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import re
import string
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/figures', exist_ok=True)

# ─── F1 SCORING (SQuAD-style token overlap) ──────────
# Computed locally from generated_answer vs gold_answer already in the
# CSVs -- no API calls needed, no extra Groq quota burned.
def _normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def token_f1(prediction, gold):
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ─── LOAD DATA ────────────────────────────────────────
def load_results():
    squad = pd.read_csv('results/squad_results.csv')
    hotpot = pd.read_csv('results/hotpot_results.csv')

    # Drop exact duplicate rows if any slipped in from a resumed run
    squad = squad.drop_duplicates(
        subset=['seed', 'chunk_size', 'embedding', 'reranking', 'question']
    )
    hotpot = hotpot.drop_duplicates(
        subset=['seed', 'chunk_size', 'embedding', 'reranking', 'question']
    )

    squad['dataset'] = 'SQuAD v2 (Single-hop)'
    hotpot['dataset'] = 'HotpotQA (Multi-hop)'

    df = pd.concat([squad, hotpot], ignore_index=True)

    # Compute F1 locally
    df['answer_f1'] = df.apply(
        lambda r: token_f1(r['generated_answer'], r['gold_answer']), axis=1
    )

    print(f"Total results loaded: {len(df)}")
    print(f"SQuAD: {len(squad)} | HotpotQA: {len(hotpot)}")
    return df

# ─── TABLE 1: MAIN EFFECTS ───────────────────────────
def make_main_effects_table(df):
    summary = df.groupby(
        ['dataset', 'chunk_size', 'embedding', 'reranking']
    ).agg(
        hit_at_k_mean=('hit_at_k', 'mean'),
        hit_at_k_std=('hit_at_k', 'std'),
        f1_mean=('answer_f1', 'mean'),
        f1_std=('answer_f1', 'std'),
        latency_mean=('latency_ms', 'mean'),
        latency_std=('latency_ms', 'std'),
        n=('hit_at_k', 'count')
    ).reset_index().round(3)

    summary.to_csv('results/main_effects_table.csv', index=False)
    print("\nMain Effects Table:")
    print(summary.to_string())
    return summary

# ─── SIGNIFICANCE TEST ───────────────────────────────
def reranking_significance_test(df):
    print("\n" + "=" * 50)
    print("SIGNIFICANCE TEST: Reranking Effect")
    print("=" * 50)

    results = {}
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]

        # Aggregate to (seed, chunk_size, embedding) level so the paired
        # test compares like-for-like configs, not raw unpaired rows.
        pivot = sub.groupby(
            ['seed', 'chunk_size', 'embedding', 'reranking']
        )['hit_at_k'].mean().reset_index()
        pivot_wide = pivot.pivot_table(
            index=['seed', 'chunk_size', 'embedding'],
            columns='reranking', values='hit_at_k'
        ).dropna()

        if True not in pivot_wide.columns or False not in pivot_wide.columns:
            print(f"\n{dataset}: not enough paired data for a test, skipping.")
            continue

        rerank_true = pivot_wide[True]
        rerank_false = pivot_wide[False]

        try:
            stat, p_value = stats.wilcoxon(rerank_true, rerank_false)
        except ValueError:
            # all differences are zero -- wilcoxon can't run
            stat, p_value = np.nan, 1.0

        mean_diff = rerank_true.mean() - rerank_false.mean()

        print(f"\n{dataset}:")
        print(f"  Rerank=True  mean hit@k: {rerank_true.mean():.3f}")
        print(f"  Rerank=False mean hit@k: {rerank_false.mean():.3f}")
        print(f"  Difference: {mean_diff:+.3f}")
        print(f"  Wilcoxon p-value: {p_value:.4f}")
        print(f"  Significant (p<0.05): {p_value < 0.05}")

        results[dataset] = {
            'mean_with_rerank': rerank_true.mean(),
            'mean_without_rerank': rerank_false.mean(),
            'difference': mean_diff,
            'p_value': p_value,
            'significant': bool(p_value < 0.05)
        }

    return results

# ─── FIGURE 1: CHUNK SIZE EFFECT ─────────────────────
def plot_chunk_size_effect(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    datasets = df['dataset'].unique()
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, dataset in zip(axes, datasets):
        sub = df[df['dataset'] == dataset]
        chunk_data = sub.groupby('chunk_size').agg(
            mean=('hit_at_k', 'mean'), std=('hit_at_k', 'std')
        ).reset_index()

        ax.bar(
            chunk_data['chunk_size'].astype(str), chunk_data['mean'],
            yerr=chunk_data['std'], color=colors, alpha=0.8,
            capsize=5, edgecolor='black', linewidth=0.5
        )
        ax.set_xlabel('Chunk Size (tokens)', fontsize=12)
        ax.set_ylabel('Hit@k (mean ± std)', fontsize=12)
        ax.set_title(f'Chunk Size Effect\n{dataset}', fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/fig1_chunk_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: fig1_chunk_size.png")

# ─── FIGURE 2: RERANKING EFFECT (Hit@k + F1) ─────────
def plot_reranking_effect(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    datasets = df['dataset'].unique()

    for col, dataset in enumerate(datasets):
        sub = df[df['dataset'] == dataset]

        for row, metric in enumerate(['hit_at_k', 'answer_f1']):
            ax = axes[row][col]
            rerank_data = sub.groupby('reranking').agg(
                mean=(metric, 'mean'), std=(metric, 'std')
            ).reset_index()

            labels = ['No Reranking', 'With Reranking']
            means = rerank_data.set_index('reranking').reindex([False, True])['mean'].values
            stds = rerank_data.set_index('reranking').reindex([False, True])['std'].values

            bars = ax.bar(
                labels, means, yerr=stds,
                color=['#EF5350', '#66BB6A'], alpha=0.85,
                capsize=6, edgecolor='black', linewidth=0.5, width=0.5
            )
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold'
                )
            metric_label = 'Hit@k' if metric == 'hit_at_k' else 'Answer F1'
            ax.set_ylabel(f'{metric_label} (mean ± std)', fontsize=12)
            ax.set_title(f'{metric_label} — Reranking Effect\n{dataset}', fontsize=12)
            ax.set_ylim(0, 1.2)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/fig2_reranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: fig2_reranking.png")

# ─── FIGURE 3: PARETO FRONTIER (F1 vs Latency) ───────
def plot_pareto_frontier(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    datasets = df['dataset'].unique()

    for ax, dataset in zip(axes, datasets):
        sub = df[df['dataset'] == dataset]
        pareto = sub.groupby(
            ['chunk_size', 'embedding', 'reranking']
        ).agg(
            quality=('answer_f1', 'mean'),
            latency=('latency_ms', 'mean')
        ).reset_index()

        colors = pareto['reranking'].map({True: '#E53935', False: '#1E88E5'})
        sizes = pareto['chunk_size'].map({256: 80, 512: 150, 1024: 250})

        ax.scatter(
            pareto['latency'], pareto['quality'], c=colors, s=sizes,
            alpha=0.8, edgecolors='black', linewidth=0.5
        )
        for _, row in pareto.iterrows():
            ax.annotate(
                f"c={row['chunk_size']}", (row['latency'], row['quality']),
                textcoords="offset points", xytext=(5, 5), fontsize=7
            )

        patch1 = mpatches.Patch(color='#E53935', label='With Reranking')
        patch2 = mpatches.Patch(color='#1E88E5', label='No Reranking')
        ax.legend(handles=[patch1, patch2], fontsize=9)
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Answer Quality (F1)', fontsize=12)
        ax.set_title(f'Quality-Latency Pareto\n{dataset}', fontsize=13)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/fig3_pareto.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: fig3_pareto.png")

# ─── DECISION RULE ────────────────────────────────────
def generate_decision_rule(df, sig_results):
    print("\n" + "=" * 50)
    print("PRACTITIONER DECISION RULE")
    print("=" * 50)

    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        best = sub.groupby(
            ['chunk_size', 'embedding', 'reranking']
        )['answer_f1'].mean().idxmax()
        fastest = sub.groupby(
            ['chunk_size', 'embedding', 'reranking']
        )['latency_ms'].mean().idxmin()
        print(f"\n{dataset}:")
        print(f"  Best F1 config: {best}")
        print(f"  Fastest config: {fastest}")

    hotpot_sig = sig_results.get('HotpotQA (Multi-hop)', {}).get('significant', False)
    squad_sig = sig_results.get('SQuAD v2 (Single-hop)', {}).get('significant', False)

    rule = f"""
DECISION FLOWCHART:
==================
Is this a multi-hop question (requires reasoning across multiple facts)?
├── YES → Reranking significant on HotpotQA: {hotpot_sig}
│         {"Use reranking -- statistically justified." if hotpot_sig else "Reranking effect not statistically significant at this sample size."}
└── NO (single-hop/factoid) →
    Reranking significant on SQuAD: {squad_sig}
    {"Use reranking if latency budget allows." if squad_sig else "Skip reranking -- no significant quality gain, saves latency."}

(Exact numbers -- mean F1/Hit@k with and without reranking, per dataset --
are in results/main_effects_table.csv. Fill this rule in with the real
numbers before pasting into the paper; this is a template, not a
substitute for reading the table.)
"""
    print(rule)
    with open('results/decision_rule.txt', 'w', encoding='utf-8') as f:
        f.write(rule)
    print("Decision rule saved: results/decision_rule.txt")

# ─── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("DAY 5 — Analysis & Visualization")
    print("=" * 50)

    df = load_results()

    print("\n[1/5] Main effects table...")
    summary = make_main_effects_table(df)

    print("\n[2/5] Significance tests...")
    sig_results = reranking_significance_test(df)

    print("\n[3/5] Figure 1 — Chunk size effect...")
    plot_chunk_size_effect(df)

    print("\n[4/5] Figure 2 — Reranking effect (Hit@k + F1)...")
    plot_reranking_effect(df)

    print("\n[5/5] Figure 3 — Pareto frontier...")
    plot_pareto_frontier(df)

    generate_decision_rule(df, sig_results)

    print("\n" + "=" * 50)
    print("DAY 5 COMPLETE!")
    print("Outputs saved in results/figures/")
    print("=" * 50)