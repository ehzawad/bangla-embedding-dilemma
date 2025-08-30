# Bengali Namjari Q&A Classification System

This is a production-ready Bengali Q&A classification system for land/property queries (namjari-related) using FAISS indexes and semantic embeddings with L3Cube Bengali SBERT model.

## Project Overview

The system classifies Bengali conversational queries about land registration procedures into specific categories using a direct embedding approach with FAISS vector search, enhanced with organized pattern matching for improved accuracy.

## Main Components

- **production_semantic_system.py**: Main classification system with training, evaluation, and inference
- **patterns.py**: Modular pattern definitions with 54 organized Bengali regex patterns
- **evaluation_dataset_conversational_final_corrected.csv**: Test dataset (69 examples) with Bengali queries and expected classifications  
- **ultra_augmented_training_data.csv**: Training dataset (2,520 examples) with questions, tags, and answers

## Key Architecture

The system uses a **direct embedding approach with modular patterns**:
1. **Training**: Full query text → single vector embedding using `l3cube-pune/bengali-sentence-similarity-sbert`
2. **Indexing**: FAISS HNSW index for fast semantic similarity search
3. **Classification**: Hybrid approach combining pattern matching + semantic similarity + keyword matching
4. **Modular Design**: Organized pattern definitions in separate module for maintainability

Core classes:
- `ProductionSemanticSystem`: Main classifier with train/evaluate/classify methods
- `ClassificationResult`: Data class for classification outputs
- `PatternMatch`: Named tuple for pattern definitions

## Common Commands

### Run the system
```bash
python3 production_semantic_system.py
```

### Train and evaluate
The main script automatically:
1. Loads training data from `ultra_augmented_training_data.csv` (2,520 examples)
2. Trains the L3Cube Bengali sentence similarity model
3. Builds FAISS HNSW index for semantic search
4. Compiles 54 organized pattern definitions from `patterns.py`
5. Evaluates against `evaluation_dataset_conversational_final_corrected.csv` (69 examples)
6. Reports accuracy and failure analysis

### Key Methods
- `system.train()`: Train the model and build indexes
- `system.evaluate()`: Run evaluation on test dataset
- `system.classify_query(query)`: Classify a single Bengali query
- Pattern functions imported from `patterns.py`: `compile_all_patterns()`, `match_patterns()`, `check_anti_patterns()`

## Dependencies

Required Python packages (inferred from imports):
- pandas
- numpy  
- faiss-cpu or faiss-gpu
- sentence-transformers
- scikit-learn

## Classification Categories

The system classifies queries into these Bengali land registration categories:
- `namjari_application_procedure`: How to apply for land registration
- `namjari_inheritance_documents`: Documents needed for inheritance cases
- `namjari_by_representative`: Using a representative for applications
- `namjari_status_check`: Checking application status
- `namjari_hearing_documents`: Documents for hearings
- `namjari_hearing_notification`: Hearing notifications/scheduling
- `namjari_rejected_appeal`: Appealing rejected applications
- `namjari_khatian_copy`: Getting land record copies
- `namjari_khatian_correction`: Correcting land records
- `namjari_fee`: Fee-related queries
- `namjari_required_documents`: General document requirements
- `namjari_eligibility`: Eligibility questions
- Plus conversation handling: `repeat_again`, `agent_calling`, `goodbye`, `irrelevant`, `greetings`

## Performance

**Current Results (L3Cube Bengali SBERT + Modular Patterns):**
- **Accuracy**: 88.4% (61/69 correct) on expanded evaluation dataset
- **Average Confidence**: 87.7%
- **Training Data**: 2,520 examples
- **Test Data**: 69 examples (expanded from original 49)
- **Method Distribution**: 72.5% pattern matches, 27.5% semantic classification
- **Model**: `l3cube-pune/bengali-sentence-similarity-sbert`

The system demonstrates robust performance on challenging Bengali conversational queries with edge cases including regional dialects, complex inheritance scenarios, and multi-step procedures.