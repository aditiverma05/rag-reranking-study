import os
from dotenv import load_dotenv
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_baseline():
    print("Loading 5 SQuAD samples...")
    df = pd.read_csv('data/squad_sample.csv').head(5)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=76
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0
    )
    
    prompt = PromptTemplate.from_template(
        """Answer the question using only the context below.
Context: {context}
Question: {question}
Answer:"""
    )
    
    print("\nRunning baseline on 5 questions...\n")
    results = []
    
    for i, row in df.iterrows():
        chunks = splitter.split_text(row['context'])
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(row['question'])
        
        print(f"Q{i+1}: {row['question']}")
        print(f"Gold:  {row['gold_answer']}")
        print(f"Model: {answer[:150]}")
        print("-" * 50)
        
        results.append({
            'question': row['question'],
            'gold_answer': row['gold_answer'],
            'model_answer': answer
        })
    
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv(
        'results/baseline_results.csv',
        index=False
    )
    print("\nBaseline complete!")
    print("Saved to results/baseline_results.csv")

if __name__ == "__main__":
    run_baseline()