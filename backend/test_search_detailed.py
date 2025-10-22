#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from services.vector_store import vector_store
from services.gemini_service import gemini_service
from services.rag_service import rag_service

def test_search_detailed():
    print("=== DETAILED SEARCH TEST ===")
    
    query = "what is zippy?"
    print(f"Query: {query}")
    
    # Test intent classification
    print("\n1. Intent Classification:")
    intent = gemini_service.classify_intent(query)
    print(f"   Intent: {intent}")
    
    # Test embedding
    print("\n2. Generate Embedding:")
    embedding = gemini_service.generate_query_embedding(query)
    print(f"   Embedding length: {len(embedding)}")
    
    # Test vector store search directly
    print("\n3. Vector Store Search (WATTMONK):")
    raw_results = vector_store.search_similar(embedding, source='WATTMONK', top_k=3)
    print(f"   Raw results count: {len(raw_results)}")
    
    for i, result in enumerate(raw_results):
        print(f"   Raw Result {i+1}:")
        print(f"     Keys: {list(result.keys())}")
        print(f"     Content: {result.get('content', 'NO CONTENT')[:100]}...")
        print(f"     Metadata: {result.get('metadata', 'NO METADATA')}")
        print(f"     Score: {result.get('score', 'NO SCORE')}")
        print()
    
    # Test RAG service formatting
    print("\n4. RAG Service Formatting:")
    formatted_sources = rag_service._format_sources(raw_results)
    print(f"   Formatted sources count: {len(formatted_sources)}")
    
    for i, source in enumerate(formatted_sources):
        print(f"   Formatted Source {i+1}:")
        print(f"     Source: {source.get('source', 'MISSING')}")
        print(f"     Filename: {source.get('filename', 'MISSING')}")
        print(f"     Score: {source.get('relevance_score', 'MISSING')}")
        print(f"     Preview: {source.get('preview', 'MISSING')[:100]}...")
        print()
    
    # Test full RAG query
    print("\n5. Full RAG Query:")
    try:
        full_result = rag_service.query(query, top_k=3)
        print(f"   Answer: {full_result['answer'][:200]}...")
        print(f"   Intent: {full_result['intent_classification']}")
        print(f"   Sources count: {len(full_result['sources'])}")
        print(f"   Context used: {full_result['context_used']}")
        print(f"   Confidence: {full_result['confidence_score']}")
        
        for i, source in enumerate(full_result['sources']):
            print(f"   Source {i+1}: {source.get('source', 'MISSING')} - {source.get('filename', 'MISSING')}")
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_search_detailed()
