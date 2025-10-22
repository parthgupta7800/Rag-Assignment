#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from services.vector_store import vector_store

def check_wattmonk_content():
    print("=== CHECKING WATTMONK CONTENT ===")
    
    # Get the WATTMONK collection directly
    collection = vector_store.collections['WATTMONK']
    
    # Get all documents
    results = collection.get(include=["documents", "metadatas"])
    
    print(f"Total WATTMONK documents: {len(results['documents'])}")
    
    # Search for "zippy" in the content
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        if 'zippy' in doc.lower():
            print(f"\nFound 'zippy' in document {i+1}:")
            print(f"Metadata: {metadata}")
            print(f"Content: {doc}")
            print("-" * 50)
    
    # Also check all content for context
    print(f"\nAll WATTMONK content:")
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        print(f"\nDocument {i+1}:")
        print(f"Metadata: {metadata}")
        print(f"Content: {doc[:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    check_wattmonk_content()
