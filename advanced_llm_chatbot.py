"""
Advanced LLM ChatBot with RAG and Semantic Search
Uses local embeddings for intelligent context retrieval
"""

import numpy as np
from typing import List, Tuple, Dict
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_chatbot import LocalLLMChatBot


class RAGChatBot(LocalLLMChatBot):
    """
    ChatBot with Retrieval-Augmented Generation.
    Retrieves relevant context before generating responses.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_history: int = 10
    ):
        """
        Initialize RAG-enabled chatbot.
        
        Args:
            model_name: HuggingFace LLM model
            embedding_model: SentenceTransformer model for embeddings
            max_history: Conversation history size
        """
        super().__init__(model_name, max_history)
        
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        self._update_embeddings()
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        
    def add_document_file(self, filepath: str) -> None:
        """Load documents from a text file (one per line)."""
        try:
            with open(filepath, 'r') as f:
                documents = [line.strip() for line in f if line.strip()]
            self.add_documents(documents)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            
    def _update_embeddings(self) -> None:
        """Update embeddings for all documents."""
        if self.documents:
            self.embeddings = self.embedder.encode(self.documents)
            
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k most relevant documents."""
        if not self.documents:
            return []
            
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
        
    def generate_response(
        self,
        user_input: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate response with retrieved context."""
        
        # Check knowledge base
        kb_response = self._check_knowledge_base(user_input)
        if kb_response:
            return kb_response
        
        # Retrieve relevant context
        context_docs = self.retrieve_context(user_input)
        context_text = " ".join(context_docs) if context_docs else ""
        
        # Build prompt with context
        conv_context = self._build_context()
        prompt = f"{conv_context}Context: {context_text}\nUser: {user_input}\nAssistant:"
        
        # Generate response
        with torch.no_grad():
            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = outputs[0]['generated_text']
        response = full_response[len(prompt):].strip()
        
        if '.' in response:
            response = response.split('.')[0] + '.'
        
        return response
        
    def save_documents(self, filepath: str) -> None:
        """Save documents to JSON."""
        with open(filepath, 'w') as f:
            json.dump({'documents': self.documents}, f)
            
    def load_documents(self, filepath: str) -> None:
        """Load documents from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.add_documents(data['documents'])


# Example usage
if __name__ == "__main__":
    import torch
    
    bot = RAGChatBot()
    
    # Add some documents
    documents = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text data.",
        "Deep learning uses neural networks with multiple layers."
    ]
    bot.add_documents(documents)
    
    # Add knowledge
    bot.add_knowledge("what is python", "Python is a versatile programming language!")
    
    print("\n=== RAG ChatBot ===\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
            
        response = bot.chat(user_input)
        print(f"Bot: {response}\n")