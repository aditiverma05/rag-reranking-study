1. RAGAS: Automated Evaluation of Retrieval-Augmented Generation (2023)

Paper: RAGAS: Automated Evaluation of Retrieval Augmented Generation

Problem
Evaluating RAG systems is difficult because there are multiple components (retrieval and generation), and traditional evaluation often requires expensive human annotations or reference answers.

Main Idea
The paper proposes RAGAS, a reference-free evaluation framework that measures different aspects of a RAG pipeline without relying on human-labeled ground truth. Instead of treating RAG as a single black box, it evaluates retrieval quality and generation quality separately.

Key Metrics
• Faithfulness
• Answer Relevancy
• Context Precision
• Context Recall
• Context Relevancy

These metrics collectively assess whether:
• the retrieved context is useful,
• the answer is supported by that context,
• and the answer addresses the user's question.

Results
The authors show that these automated metrics correlate well with human judgments, enabling faster and cheaper evaluation of RAG systems.

Relevance to Our Project
This is the evaluation framework we will use.

We specifically plan to use:
• Faithfulness
• Answer Relevancy

For retrieval, we'll compute gold-passage metrics (Hit@k, MRR, Context Precision/Recall) directly from the datasets rather than relying solely on LLM judges.

2. Lost in the Middle: How Language Models Use Long Contexts (2023)

Problem
Large-context LLMs can accept many retrieved documents, but do they actually use information equally from all positions?

Main Idea
The authors investigate how document position affects an LLM's ability to retrieve and use relevant information.

They place the correct evidence:
• at the beginning,
• in the middle,
• at the end
of long contexts and compare performance.

Key Finding
LLMs perform best when the relevant passage is:
• near the beginning, or
• near the end.

Performance drops significantly when the relevant information is buried in the middle of a long context—a phenomenon the paper calls "Lost in the Middle."

Results
The paper shows that simply increasing context length does not guarantee better performance. If retrieval returns many chunks, important evidence can become harder for the model to use.

Relevance to Our Project
This directly motivates studying:
• chunk size (256 vs 512 vs 1024),
• reranking.

Our hypothesis:
• Small chunks → better retrieval precision.
• Large chunks → more context but more irrelevant tokens.
• Reranking should move the most relevant chunks toward the top.

3. Retrieval-Augmented Generation for Large Language Models: A Survey (2023)

Problem
LLMs suffer from hallucinations, outdated knowledge, limited explainability, and expensive retraining.

Main Idea
The survey reviews the complete RAG ecosystem and categorizes RAG systems into:
• Naive RAG
• Advanced RAG
• Modular RAG

Important Components
• Chunking
• Embedding models
• Retrievers
• Rerankers
• Generators
• Evaluation frameworks
• Benchmarks
• Future research directions

Key Findings
Retrieval quality is often the main bottleneck in RAG systems. Improvements in chunking, embeddings, and reranking frequently have a larger impact than changing the generator model alone.

Relevance to Our Project
This survey justifies our experimental design:
• vary chunk size
• compare embedding models
• compare reranking on/off
• keep the generator fixed
• measure quality, latency, and cost

Overall Takeaways

RAGAS:
Provides reference-free evaluation metrics for RAG quality.

Lost in the Middle:
Shows document position affects LLM performance and motivates chunk size and reranking experiments.

RAG Survey:
Shows retrieval quality is central to RAG performance and provides the conceptual foundation for our study.
