#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from services.vector_store import vector_store
from services.gemini_service import gemini_service

def test_zippy_query():
    print("=== DEBUGGING ZIPPY QUERY ===")
    
    query = "what is zippy?"
    print(f"Query: {query}")
    
    # Test intent classification
    print("\n1. Testing Intent Classification:")
    intent = gemini_service.classify_intent(query)
    print(f"   Classified intent: {intent}")
    
    # Test embedding generation
    print("\n2. Testing Embedding Generation:")
    embedding = gemini_service.generate_query_embedding(query)
    print(f"   Embedding generated, length: {len(embedding)}")
    
    # Test search in WATTMONK collection specifically
    print("\n3. Testing WATTMONK Collection Search:")
    wattmonk_results = vector_store.search_similar(embedding, source='WATTMONK', top_k=5)
    print(f"   WATTMONK results count: {len(wattmonk_results)}")
    
    for i, result in enumerate(wattmonk_results):
        print(f"   Result {i+1}:")
        print(f"     Source: {result.get('source', 'Unknown')}")
        print(f"     Filename: {result.get('filename', 'Unknown')}")
        print(f"     Score: {result.get('relevance_score', 'Unknown')}")
        preview = result.get('preview', 'No preview')
        print(f"     Preview: {preview[:150]}...")
        print()
    
    # Test search in ALL collections
    print("\n4. Testing ALL Collections Search:")
    all_results = vector_store.search_similar(embedding, source=None, top_k=5)
    print(f"   All collections results count: {len(all_results)}")
    
    for i, result in enumerate(all_results):
        print(f"   Result {i+1}:")
        print(f"     Source: {result.get('source', 'Unknown')}")
        print(f"     Filename: {result.get('filename', 'Unknown')}")
        print(f"     Score: {result.get('relevance_score', 'Unknown')}")
        preview = result.get('preview', 'No preview')
        print(f"     Preview: {preview[:100]}...")
        print()
    
    # Check collection stats
    print("\n5. Collection Statistics:")
    stats = vector_store.get_collection_stats()
    for source, stat in stats.items():
        print(f"   {source}: {stat['document_count']} documents")

if __name__ == "__main__":
    test_zippy_query()
