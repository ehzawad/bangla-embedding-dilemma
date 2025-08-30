#!/usr/bin/env python3
"""
Production Semantic System - Refactored and Organized
Bengali Q&A Classification with Enhanced Pattern Management and Accuracy
"""

import pandas as pd
import numpy as np
import faiss
import re
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from patterns import compile_all_patterns, match_patterns, check_anti_patterns


@dataclass
class ClassificationResult:
    tag: str
    score: float
    confidence: float
    method: str
    reasoning: str


class PatternMatch(NamedTuple):
    pattern: str
    tag: str
    priority: int
    description: str



# Pattern matching functions are now imported from patterns.py module

# ========================================
# ACCURACY IMPROVEMENT FUNCTIONS  
# ========================================

def calculate_enhanced_confidence(scores: List[float], best_score: float) -> float:
    """Enhanced confidence calculation with better scoring"""
    if len(scores) <= 1:
        return min(best_score / 1.2, 0.95)
    
    # Sort scores in descending order
    scores_sorted = sorted(scores, reverse=True)
    best = scores_sorted[0]
    second = scores_sorted[1] if len(scores_sorted) > 1 else 0
    
    # Calculate margin-based confidence
    margin = best - second
    base_confidence = best / (best + second * 0.6)
    
    # Boost confidence for clear winners
    if margin > 0.3:
        base_confidence *= 1.15
    elif margin > 0.2:
        base_confidence *= 1.1
    
    return min(base_confidence, 0.95)


def apply_semantic_boosting(query: str, semantic_scores: Dict[str, float]) -> Dict[str, float]:
    """Apply enhanced semantic boosting for better accuracy"""
    boosted_scores = semantic_scores.copy()
    query_lower = query.lower()
    
    # Boost procedure queries more aggressively
    if any(word in query_lower for word in ['কিভাবে', 'নিয়ম', 'পদ্ধতি', 'প্রক্রিয়া', 'আবেদন']):
        if 'namjari_application_procedure' in boosted_scores:
            boosted_scores['namjari_application_procedure'] *= 1.25
    
    # Boost inheritance queries, but only when context is strong
    inheritance_words = ['ওয়ারিশ', 'উত্তরাধিকার', 'হাল ওয়াশিাননামা', 'মৃত্যু সনদ']
    if any(word in query_lower for word in inheritance_words):
        if 'namjari_inheritance_documents' in boosted_scores:
            boosted_scores['namjari_inheritance_documents'] *= 1.2
    
    # Boost khatian correction for correction-related queries
    if any(word in query_lower for word in ['ভুল', 'সংশোধন', 'বানান', 'দাগ নম্বর']):
        if 'namjari_khatian_correction' in boosted_scores:
            boosted_scores['namjari_khatian_correction'] *= 1.3
    
    # Boost representative queries only when really about representatives
    if any(phrase in query_lower for phrase in ['তার হয়ে', 'প্রতিনিধি', 'পাওয়ার অফ']):
        if 'namjari_by_representative' in boosted_scores:
            boosted_scores['namjari_by_representative'] *= 1.2
    
    # Boost hearing-related queries
    if any(word in query_lower for word in ['শুনানি', 'আমিন']):
        if 'namjari_hearing_documents' in boosted_scores:
            boosted_scores['namjari_hearing_documents'] *= 1.15
        if 'namjari_hearing_notification' in boosted_scores:
            boosted_scores['namjari_hearing_notification'] *= 1.15
    
    # Boost status queries
    if any(word in query_lower for word in ['স্ট্যাটাস', 'অপেক্ষা', 'প্রক্রিয়াধীন']):
        if 'namjari_status_check' in boosted_scores:
            boosted_scores['namjari_status_check'] *= 1.2
    
    # De-boost inheritance for khatian-related queries to prevent confusion
    if any(word in query_lower for word in ['খতিয়ান', 'জরিপ', '৪ ভাই']) and 'namjari_inheritance_documents' in boosted_scores:
        boosted_scores['namjari_inheritance_documents'] *= 0.7
    
    return boosted_scores


# ========================================
# MAIN CLASSIFICATION SYSTEM
# ========================================

class ProductionSemanticSystem:
    """Production-ready direct embedding semantic classification system - Refactored"""
    
    def __init__(self):
        self.semantic_model = None
        self.faiss_index = None
        self.training_data = None
        self.embeddings = None
        self.keyword_vectorizer = None
        self.keyword_embeddings = None
        
        # Compile organized patterns
        self.patterns, self.anti_patterns = compile_all_patterns()
        print(f"🎯 Loaded {len(self.patterns)} organized patterns")
    
    def train(self) -> bool:
        """Train the production system with direct embeddings"""
        print("🚀 TRAINING PRODUCTION DIRECT EMBEDDING SYSTEM (REFACTORED)")
        print("=" * 60)
        
        # Load enhanced training data
        print("📊 Loading ultra-augmented training data...")
        self.training_data = pd.read_csv('ultra_augmented_training_data.csv')
        print(f"   Training examples: {len(self.training_data)}")
        
        # Load Bengali-specific model
        print("🧠 Loading Bengali sentence similarity model (L3Cube)...")
        self.semantic_model = SentenceTransformer('l3cube-pune/bengali-sentence-similarity-sbert')
        
        # Generate DIRECT embeddings (full query text)
        print("🔄 Generating direct semantic embeddings...")
        questions = self.training_data['question'].tolist()
        
        # Process in batches to show progress
        batch_size = 500
        embeddings = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            batch_embeddings = self.semantic_model.encode(batch)
            embeddings.extend(batch_embeddings)
            print(f"   Processed {min(i+batch_size, len(questions))}/{len(questions)} questions")
        
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Build optimized FAISS index
        print("🔍 Building FAISS HNSW index...")
        dimension = self.embeddings.shape[1]
        
        # Use HNSW for better performance
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        self.faiss_index.hnsw.efConstruction = 200
        self.faiss_index.hnsw.efSearch = 100
        self.faiss_index.add(self.embeddings)
        
        # Build keyword index
        print("📝 Building TF-IDF keyword index...")
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )
        self.keyword_embeddings = self.keyword_vectorizer.fit_transform(questions)
        
        print("✅ Production direct embedding system trained!")
        return True
    
    def classify_query(self, query: str, k: int = 10) -> Optional[ClassificationResult]:
        """Classify a query using organized pattern matching and semantic search"""
        if not self.semantic_model or not self.faiss_index:
            return None
        
        # Check organized patterns first (highest priority)
        pattern_match = match_patterns(query, self.patterns)
        if pattern_match:
            return ClassificationResult(
                tag=pattern_match.tag,
                score=0.95,
                confidence=0.90,
                method="pattern_match",
                reasoning=f"Pattern: {pattern_match.description}"
            )
        
        # DIRECT semantic search using full query embedding
        query_embedding = self.semantic_model.encode([query])[0]
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        semantic_scores, semantic_indices = self.faiss_index.search(query_vector, k)
        
        # Keyword-based search
        query_keywords = self.keyword_vectorizer.transform([query])
        keyword_similarities = cosine_similarity(query_keywords, self.keyword_embeddings).flatten()
        keyword_top_indices = np.argsort(keyword_similarities)[-k:][::-1]
        
        # Combine semantic and keyword results with direct embedding weights
        combined_scores = {}
        
        # Process semantic results (primary weight - higher for direct embeddings)
        for i, (score, idx) in enumerate(zip(semantic_scores[0], semantic_indices[0])):
            if idx < len(self.training_data):
                tag = self.training_data.iloc[idx]['tag']
                combined_scores[tag] = combined_scores.get(tag, 0) + score * 0.75
        
        # Process keyword results (secondary weight)
        for i, idx in enumerate(keyword_top_indices):
            if idx < len(self.training_data):
                tag = self.training_data.iloc[idx]['tag']
                keyword_score = keyword_similarities[idx]
                combined_scores[tag] = combined_scores.get(tag, 0) + keyword_score * 0.25
        
        if not combined_scores:
            return None
        
        # Apply semantic boosting
        combined_scores = apply_semantic_boosting(query, combined_scores)
        
        # Get best result
        best_tag = max(combined_scores.keys(), key=lambda x: combined_scores[x])
        best_score = combined_scores[best_tag]
        
        # Check anti-patterns to prevent misclassification
        if check_anti_patterns(query, best_tag, self.anti_patterns):
            # Try second best
            remaining_scores = {k: v for k, v in combined_scores.items() if k != best_tag}
            if remaining_scores:
                best_tag = max(remaining_scores.keys(), key=lambda x: remaining_scores[x])
                best_score = remaining_scores[best_tag]
            else:
                return None
        
        # Enhanced confidence calculation
        scores = list(combined_scores.values())
        confidence = calculate_enhanced_confidence(scores, best_score)
        
        return ClassificationResult(
            tag=best_tag,
            score=best_score,
            confidence=confidence,
            method="semantic_hybrid",
            reasoning=f"Direct embedding semantic + keyword hybrid with boosting"
        )
    
    def evaluate(self) -> Tuple[float, List[Dict]]:
        """Evaluate the production system"""
        print("🔍 Evaluating production direct embedding system (REFACTORED)")
        
        eval_df = pd.read_csv('evaluation_dataset_conversational_final_corrected.csv')
        
        correct = 0
        total = len(eval_df)
        failures = []
        confidence_scores = []
        method_counts = {}
        
        for i, row in eval_df.iterrows():
            query = row['question']
            expected = row['expected_tag']
            
            result = self.classify_query(query)
            
            if result:
                predicted = result.tag
                confidence_scores.append(result.confidence)
                method_counts[result.method] = method_counts.get(result.method, 0) + 1
                
                if predicted == expected:
                    correct += 1
                    print(f"✅ {i+1:2d}: {expected} (conf: {result.confidence:.3f}, {result.method})")
                else:
                    failures.append({
                        'index': i+1,
                        'query': query[:80] + '...' if len(query) > 80 else query,
                        'expected': expected,
                        'predicted': predicted,
                        'confidence': result.confidence,
                        'method': result.method
                    })
                    print(f"❌ {i+1:2d}: Expected {expected}, got {predicted} (conf: {result.confidence:.3f})")
            else:
                failures.append({
                    'index': i+1,
                    'query': query[:80] + '...' if len(query) > 80 else query,
                    'expected': expected,
                    'predicted': 'None',
                    'confidence': 0.0,
                    'method': 'none'
                })
                print(f"❌ {i+1:2d}: Expected {expected}, got None")
        
        accuracy = correct / total
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        print(f"\n🏆 PRODUCTION DIRECT EMBEDDING RESULTS (REFACTORED)")
        print("=" * 50)
        print(f"📊 Accuracy: {accuracy:.3f} ({correct}/{total}) = {accuracy*100:.1f}%")
        print(f"🔮 Average Confidence: {avg_confidence:.3f}")
        print(f"📚 Training Data: {len(self.training_data)} examples")
        print(f"🎯 Approach: L3Cube Bengali SBERT + Organized patterns + Semantic boosting")
        
        print(f"\n📈 Method Distribution:")
        for method, count in method_counts.items():
            percentage = (count / total) * 100
            print(f"   {method}: {count} ({percentage:.1f}%)")
        
        if failures:
            print(f"\n❌ Failures ({len(failures)}):")
            for failure in failures[:10]:
                print(f"   {failure['index']:2d}: {failure['expected']} → {failure['predicted']}")
                print(f"      '{failure['query']}'")
        
        return accuracy, failures


def main():
    """Test the refactored production direct embedding system"""
    print("🚀 TESTING REFACTORED PRODUCTION DIRECT EMBEDDING SYSTEM")
    print("=" * 60)
    print("🎯 Organized patterns + Enhanced semantic classification")
    print("🔧 Modular design with accuracy improvements")
    
    system = ProductionSemanticSystem()
    
    if not system.train():
        print("❌ Training failed")
        return
    
    accuracy, failures = system.evaluate()
    
    print(f"\n🎉 REFACTORED PRODUCTION SYSTEM TEST COMPLETE!")
    print(f"📊 Final Accuracy: {accuracy:.1%}")
    print(f"📚 Training Size: {len(system.training_data)} examples")
    print(f"🏆 L3Cube Bengali SBERT model test with organized patterns")
    
    return accuracy


if __name__ == "__main__":
    main()