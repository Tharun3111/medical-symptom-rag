import os
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.rag_pipeline import RAGPipeline
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


def create_test_cases():
    """Create test cases for evaluation."""
    test_cases = [
        {
            "question": "I have a fever, headache, and body aches for 3 days",
            "ground_truth": "These symptoms commonly indicate influenza (flu) or a viral infection."
        },
        {
            "question": "I have chest pain and shortness of breath",
            "ground_truth": "Chest pain and shortness of breath can indicate serious conditions like heart attack or pulmonary issues and require immediate medical attention."
        },
        {
            "question": "I have a persistent cough and sore throat for a week",
            "ground_truth": "A persistent cough and sore throat lasting a week may indicate bronchitis, upper respiratory infection, or allergies."
        },
        {
            "question": "I have severe headache with sensitivity to light",
            "ground_truth": "Severe headache with light sensitivity is a common symptom of migraine."
        },
        {
            "question": "I have stomach pain, nausea, and diarrhea",
            "ground_truth": "These symptoms typically indicate gastroenteritis or food poisoning."
        }
    ]
    return test_cases


def run_evaluation():
    """Run RAGAS evaluation on the RAG system."""
    print("="*60)
    print("RAGAS Evaluation for Medical Symptom Checker")
    print("="*60)
    
    # Initialize pipeline
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    pipeline = RAGPipeline(store_dir)
    
    # Get test cases
    test_cases = create_test_cases()
    
    # Prepare data for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("\n🔄 Running RAG pipeline on test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        
        print(f"Test Case {i}/{len(test_cases)}: {question[:50]}...")
        
        # Get retrieval results
        results = pipeline.retriever.retrieve(question, top_k=3)
        context = [result['text'] for result in results]
        
        # Generate answer
        result = pipeline.generate_diagnosis(question)
        answer = result['diagnosis']
        
        # Store for RAGAS
        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)
        
        print(f"  ✓ Retrieved {len(context)} contexts")
        print(f"  ✓ Generated answer ({len(answer)} chars)\n")
    
    # Create RAGAS dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("="*60)
    print("📊 Running RAGAS Evaluation...")
    print("="*60)
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    
    # Display results
    print("\n" + "="*60)
    print("✅ EVALUATION RESULTS")
    print("="*60)
    
    df = result.to_pandas()
    
    # Summary statistics
    print("\n📈 Summary Metrics:")
    print("-" * 60)
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if metric in df.columns:
            mean_score = df[metric].mean()
            print(f"  {metric.replace('_', ' ').title()}: {mean_score:.3f}")
    
    print("\n" + "="*60)
    print("📋 Detailed Results by Test Case:")
    print("="*60)
    
    # Show per-question results
    for i, row in df.iterrows():
        print(f"\nTest Case {i+1}:")
        print(f"  Question: {questions[i][:60]}...")
        if 'faithfulness' in df.columns:
            print(f"  Faithfulness: {row['faithfulness']:.3f}")
        if 'answer_relevancy' in df.columns:
            print(f"  Answer Relevancy: {row['answer_relevancy']:.3f}")
        if 'context_precision' in df.columns:
            print(f"  Context Precision: {row['context_precision']:.3f}")
        if 'context_recall' in df.columns:
            print(f"  Context Recall: {row['context_recall']:.3f}")
    
    # Save results
    results_path = project_root / 'eval' / 'evaluation_results.csv'
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("📊 Interpretation:")
    print("="*60)
    print("""
  - Faithfulness (0-1): How well the answer is grounded in retrieved context
    → Higher is better (>0.7 is good)
    
  - Answer Relevancy (0-1): How relevant the answer is to the question
    → Higher is better (>0.8 is good)
    
  - Context Precision (0-1): How relevant the retrieved contexts are
    → Higher is better (>0.6 is good)
    
  - Context Recall (0-1): How much of the ground truth is in the context
    → Higher is better (>0.7 is good)
    """)
    
    return result


if __name__ == "__main__":
    run_evaluation()