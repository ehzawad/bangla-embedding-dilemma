# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bengali Q&A classification system for land/property queries (namjari-related) using FAISS indexes and semantic embeddings. The system classifies Bengali conversational queries about land registration procedures into specific categories using direct embedding approach with FAISS vector search.

## Main Components

- **production_semantic_system.py**: Main classification system with training, evaluation, and inference
- **evaluation_dataset_conversational_final_corrected.csv**: Test dataset (49 examples) with Bengali queries and expected classifications
- **ultra_augmented_training_data.csv**: Training dataset (2,475 examples) with questions, tags, and answers

## Key Architecture

The system uses a **direct embedding approach**:
1. **Training**: Full query text → single vector embedding using `intfloat/multilingual-e5-large-instruct`
2. **Indexing**: FAISS HNSW index for fast semantic similarity search
3. **Classification**: Hybrid approach combining semantic + keyword matching
4. **Failure Fixes**: Regex patterns for edge cases that semantic search misses

Core classes:
- `ProductionSemanticSystem`: Main classifier with train/evaluate/classify methods
- `ClassificationResult`: Data class for classification outputs

## Common Commands

### Run the system
```bash
python3 production_semantic_system.py
```

### Train and evaluate
The main script automatically:
1. Loads training data from `ultra_augmented_training_data.csv`
2. Trains the multilingual sentence transformer model
3. Builds FAISS index for semantic search
4. Evaluates against `evaluation_dataset_conversational_final_corrected.csv`
5. Reports accuracy and failure analysis

### Key Methods
- `system.train()`: Train the model and build indexes
- `system.evaluate()`: Run evaluation on test dataset
- `system.classify_query(query)`: Classify a single Bengali query

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

Current system achieves ~93.9% accuracy on the evaluation dataset using the direct embedding approach with failure fix patterns for edge cases.