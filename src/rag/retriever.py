"""Retrieval component for the RAG system."""

import faiss
import numpy as np
from typing import List, Dict, Tuple
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch

class FinancialDataRetriever:
    def __init__(self, config: Dict):
        """Initialize the retriever with configuration."""
        self.retriever_type = config['rag']['retriever']
        self.max_documents = config['rag']['max_documents']
        self.similarity_threshold = config['rag']['similarity_threshold']
        
        # Initialize FAISS index
        self.dimension = 768  # BERT embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize transformer model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Initialize Elasticsearch if needed
        if self.retriever_type == "elasticsearch":
            self.es = Elasticsearch()
            
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text using BERT."""
        tokens = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
        
    def index_documents(self, documents: List[Dict]):
        """Index documents for retrieval."""
        if self.retriever_type == "faiss":
            texts = [doc['text'] for doc in documents]
            embeddings = self.encode_text(texts)
            self.index.add(embeddings)
            self.documents = documents
        else:
            for doc in documents:
                self.es.index(index="financial_data", document=doc)
                
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve relevant documents."""
        k = k or self.max_documents
        query_embedding = self.encode_text([query])
        
        if self.retriever_type == "faiss":
            distances, indices = self.index.search(query_embedding, k)
            results = [
                {
                    'document': self.documents[idx],
                    'score': float(1 / (1 + dist))
                }
                for dist, idx in zip(distances[0], indices[0])
                if 1 / (1 + dist) >= self.similarity_threshold
            ]
        else:
            response = self.es.search(
                index="financial_data",
                query={
                    "match": {
                        "text": query
                    }
                },
                size=k
            )
            results = [
                {
                    'document': hit['_source'],
                    'score': hit['_score']
                }
                for hit in response['hits']['hits']
                if hit['_score'] >= self.similarity_threshold
            ]
            
        return results

