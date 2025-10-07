import os
from pathlib import Path
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI

from rag.retriever import MedlineRetriever
from rag.prompts import create_diagnosis_prompt

# Load environment variables
load_dotenv()


class RAGPipeline:
    """Manages the complete RAG workflow for medical symptom checking."""
    
    def __init__(self, store_dir: Path):
        """
        Initialize RAG pipeline.
        
        Args:
            store_dir: Directory containing FAISS index and metadata
        """
        print("🚀 Initializing RAG Pipeline...")
        
        # Initialize retriever
        self.retriever = MedlineRetriever(store_dir)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        print("✅ RAG Pipeline ready!")
    
    def generate_diagnosis(
        self, 
        user_symptoms: str
    ) -> Dict[str, str]:
        """
        Generate diagnosis directly from symptoms (no follow-ups).
        
        Args:
            user_symptoms: User's symptom description
        
        Returns:
            Dict with diagnosis and retrieved_sources
        """
        print(f"🔍 Retrieving relevant medical information...")
        
        # Retrieve relevant medical info
        results = self.retriever.retrieve(user_symptoms, top_k=3)
        context = self.retriever.format_context(results)
        
        print(f"✓ Retrieved {len(results)} relevant sources")
        
        # Create simple conversation history
        conversation_history = [
            {"role": "user", "content": user_symptoms}
        ]
        
        # Generate diagnosis
        prompt = create_diagnosis_prompt(conversation_history, context)
        
        print(f"🤖 Generating diagnosis with GPT-3.5...")
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        diagnosis_text = response.choices[0].message.content.strip()
        
        # Extract sources for citations
        sources = [
            {
                'title': result['title'],
                'url': result['url'],
                'relevance_score': result['score']
            }
            for result in results
        ]
        
        return {
            'diagnosis': diagnosis_text,
            'sources': sources
        }


class ConversationManager:
    """Manages conversation state for a symptom checking session."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.conversation_history: List[Dict[str, str]] = []
        self.stage = "initial"  # initial, complete
    
    def add_user_message(self, message: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
    
    def add_assistant_message(self, message: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            'role': 'assistant',
            'content': message
        })
    
    def process_message(self, user_input: str) -> Dict:
        """
        Process user message and return diagnosis directly.
        
        Returns:
            Dict with 'type' (diagnosis) and 'content'
        """
        # Add user input to history
        self.add_user_message(user_input)
        
        # Generate diagnosis directly
        if self.stage == "initial":
            self.stage = "diagnosis"
            result = self.pipeline.generate_diagnosis(user_input)
            self.add_assistant_message(result['diagnosis'])
            self.stage = "complete"
            return {
                'type': 'diagnosis',
                'content': result['diagnosis'],
                'sources': result['sources']
            }
        
        # After diagnosis
        elif self.stage == "complete":
            return {
                'type': 'complete',
                'content': "Thank you for using the symptom checker. If you have new symptoms, please start a new session."
            }


if __name__ == "__main__":
    # Test the pipeline
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    
    print("="*60)
    print("Testing RAG Pipeline (No Follow-ups)")
    print("="*60)
    
    # Initialize
    pipeline = RAGPipeline(store_dir)
    manager = ConversationManager(pipeline)
    
    # Test with direct diagnosis
    print("\n👤 User: I have a fever, headache, and body aches for 3 days")
    response = manager.process_message("I have a fever, headache, and body aches for 3 days")
    
    if response['type'] == 'diagnosis':
        print(f"\n{'='*60}")
        print("🤖 DIAGNOSIS")
        print(f"{'='*60}")
        print(f"{response['content']}\n")
        print(f"{'='*60}")
        print(f"📚 Sources Used:")
        for source in response['sources']:
            print(f"   - {source['title']} (relevance: {source['relevance_score']:.3f})")
        print(f"{'='*60}")