# UniversalRAG

Implementation of UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities.

## Overview

UniversalRAG is a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. It features:

- Multi-modal support (text, images, videos)
- Modality-aware routing mechanism
- Multiple granularity levels for each modality
- Dynamic corpus selection based on query type

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `models/` - Core model implementations
  - `processors/` - Data processors for different modalities
  - `retrievers/` - Retrieval components
  - `utils/` - Utility functions
- `examples/` - Example usage and demos
- `tests/` - Unit tests

## Usage

Coming soon...

## Citation

```bibtex
@article{universalrag2024,
  title={UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities},
  author={...},
  journal={arXiv preprint arXiv:2504.20734},
  year={2024}
}
```
