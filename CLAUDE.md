# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Run the main system
```bash
python3 production_semantic_system.py
```

This command automatically:
1. Loads training data from `training_data.csv` (2,520 examples)
2. Trains the L3Cube Bengali sentence similarity model
3. Builds FAISS HNSW index for semantic search
4. Compiles 54 organized pattern definitions from `patterns.py`
5. Evaluates against `test_data.csv` (69 examples)
6. Reports accuracy and failure analysis

### Key Dependencies

Install required packages:
```bash
pip install pandas numpy faiss-cpu sentence-transformers scikit-learn
```

For GPU acceleration, use `faiss-gpu` instead of `faiss-cpu`.

## Architecture Overview

This is a Bengali Q&A classification system for land/property queries (namjari-related) using semantic embeddings and pattern matching.

### Core Components

- **`production_semantic_system.py`**: Main classification system with hybrid approach
  - `ProductionSemanticSystem` class: Main classifier with train/evaluate/classify methods
  - `ClassificationResult` dataclass: Structured classification outputs
  - Direct embedding approach using `l3cube-pune/bengali-sentence-similarity-sbert`
  - FAISS HNSW index for fast semantic similarity search
  - Hybrid classification combining pattern matching + semantic similarity + keyword matching

- **`patterns.py`**: Modular pattern definitions
  - 54 organized Bengali regex patterns across different priority levels
  - Functions: `compile_all_patterns()`, `match_patterns()`, `check_anti_patterns()`
  - `PatternMatch` named tuple for structured pattern definitions
  - Critical failure patterns for fixing known misclassifications

### Data Files

- **`training_data.csv`**: Training dataset (2,520 examples) with questions, tags, and answers
- **`test_data.csv`**: Test dataset (69 examples) with Bengali queries and expected classifications

### Classification Process

1. **Pattern Matching**: High-priority Bengali regex patterns (72.5% of classifications)
2. **Semantic Search**: L3Cube Bengali SBERT embeddings with FAISS index
3. **Keyword Matching**: TF-IDF vectorization for keyword-based fallbacks
4. **Anti-patterns**: Prevents known misclassifications
5. **Confidence Scoring**: Enhanced confidence calculation with semantic boosting

### Classification Categories

The system handles 15 categories of Bengali land registration queries:
- `namjari_application_procedure`, `namjari_inheritance_documents`, `namjari_by_representative`
- `namjari_status_check`, `namjari_hearing_documents`, `namjari_hearing_notification`
- `namjari_rejected_appeal`, `namjari_khatian_copy`, `namjari_khatian_correction`
- `namjari_fee`, `namjari_required_documents`, `namjari_eligibility`
- Plus conversation handling: `repeat_again`, `agent_calling`, `goodbye`, `irrelevant`, `greetings`

### Key Methods

- `system.train()`: Train the model and build FAISS index
- `system.evaluate()`: Run evaluation on test dataset  
- `system.classify_query(query)`: Classify a single Bengali query
- Pattern functions from `patterns.py`: Handle organized pattern matching

### Performance Metrics

Current system achieves:
- **88.4% accuracy** (61/69 correct) on evaluation dataset
- **87.7% average confidence**
- **Method distribution**: 72.5% pattern matches, 27.5% semantic classification