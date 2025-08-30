#!/usr/bin/env python3
"""
Quick Model Test - Test just 2 models to verify download speed
"""

import time
from sentence_transformers import SentenceTransformer

def test_model_download(model_name: str):
    """Test downloading and loading a single model"""
    print(f"🔄 Testing model: {model_name}")
    start_time = time.time()
    
    try:
        model = SentenceTransformer(model_name)
        download_time = time.time() - start_time
        
        # Test encoding
        test_query = "নামজারি কিভাবে করব?"
        embedding = model.encode([test_query])
        
        total_time = time.time() - start_time
        
        print(f"✅ {model_name}")
        print(f"   Download: {download_time:.1f}s")
        print(f"   Total: {total_time:.1f}s")
        print(f"   Embedding shape: {embedding.shape}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False

def main():
    print("🚀 QUICK MODEL DOWNLOAD TEST")
    print("=" * 40)
    
    # Test just 2 models first
    models = [
        'paraphrase-multilingual-mpnet-base-v2',  # Current model (smaller)
        'intfloat/multilingual-e5-large'         # New model (larger)
    ]
    
    for model_name in models:
        success = test_model_download(model_name)
        if not success:
            print("Stopping due to failure...")
            break

if __name__ == "__main__":
    main()
