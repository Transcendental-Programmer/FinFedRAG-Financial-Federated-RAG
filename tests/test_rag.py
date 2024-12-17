"""test_rag.py module."""

import pytest
from src.rag.retriever import FinancialDataRetriever
from src.rag.generator import RAGGenerator
import yaml

@pytest.fixture
def rag_config():
    with open('config/server_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['rag'] = {
            'retriever': 'faiss',
            'max_documents': 5,
            'similarity_threshold': 0.7
        }
        return config

@pytest.fixture
def retriever(rag_config):
    return FinancialDataRetriever(rag_config)

@pytest.fixture
def generator(rag_config):
    return RAGGenerator(rag_config)

def test_retriever_initialization(retriever, rag_config):
    assert retriever.retriever_type == rag_config['rag']['retriever']
    assert retriever.max_documents == rag_config['rag']['max_documents']

def test_document_indexing(retriever):
    test_documents = [
        {'text': 'Financial report 2023', 'id': 1},
        {'text': 'Market analysis Q4', 'id': 2},
        {'text': 'Investment strategy', 'id': 3}
    ]
    
    retriever.index_documents(test_documents)
    assert retriever.index.ntotal == len(test_documents)

def test_document_retrieval(retriever):
    # Index test documents
    test_documents = [
        {'text': 'Financial report 2023', 'id': 1},
        {'text': 'Market analysis Q4', 'id': 2},
        {'text': 'Investment strategy', 'id': 3}
    ]
    retriever.index_documents(test_documents)
    
    # Test retrieval
    query = "financial report"
    results = retriever.retrieve(query)
    assert len(results) > 0
    assert all('document' in result for result in results)
    assert all('score' in result for result in results)

def test_generator_initialization(generator):
    assert hasattr(generator, 'model')
    assert hasattr(generator, 'tokenizer')

def test_text_generation(generator):
    retrieved_docs = [
        {
            'document': {'text': 'Financial market analysis shows positive trends'},
            'score': 0.9
        }
    ]
    
    generated_text = generator.generate(
        query="Summarize market trends",
        retrieved_docs=retrieved_docs
    )
    
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0

def test_context_preparation(generator):
    retrieved_docs = [
        {
            'document': {'text': 'Doc 1 content'},
            'score': 0.9
        },
        {
            'document': {'text': 'Doc 2 content'},
            'score': 0.8
        }
    ]
    
    context = generator.prepare_context(retrieved_docs)
    assert isinstance(context, str)
    assert 'Doc 1 content' in context
    assert 'Doc 2 content' in context

