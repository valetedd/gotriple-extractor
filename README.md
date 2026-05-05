# GoTriple Extractor

**Multilingual NLP pipeline for extracting entities, relations, and linking them to the GoTriple knowledge base.**

---

## Overview

`gotriple_extractor` is a research‑grade toolkit that processes raw multilingual text, extracts named entities and semantic relations, and aligns the results with the **GoTriple** dataset – a large, cross‑disciplinary collection of scholarly texts.  The project showcases:

* **Chunk‑level processing** – texts are recursively chunked (≈512 tokens) with overlap refinement to preserve context.
* **Entity extraction** – multiple back‑ends are supported:
  * **GLiNER** (large‑scale token classification)
  * **spaCy** multilingual models
  * **LLM‑driven extraction** (prompt‑based, configurable temperature & n‑best) via the Graphia LLM endpoint.
* **Relation extraction** – GLiREL and prompt‑based LLM extraction with schema validation.
* **Linking** – heuristic matching of extracted entities to GoTriple entries.
* **Benchmarking** – reproducible evaluation on CONER, NERD, and SciERC datasets with precision/recall visualisations.

All components are orchestrated through a small set of Python scripts that can be chained together or invoked individually.

---

## Repository Structure

```
├── benchmark/                 # Benchmark scripts & results
│   ├── *.parquet               # Datasets (CONER, NERD, SciERC)
│   └── benchmark_results/      # Figures & CSV analyses
├── linking/                   # Entity linking utilities
├── text_dataset/              # Raw GoTriple‑derived JSON files
├── utils.py                   # Helper functions (cleaning, rechunking, etc.)
├── data_repr.py               # Pydantic schemas & dataclasses
├── extraction.py              # Core extractor abstractions
├── entity_extraction.py       # High‑level entity extraction driver
├── relation_extraction.py     # Relation extraction and candidate sampling
├── prompt.md                  # Prompt used for LLM‑based extraction
├── pyproject.toml             # Poetry/PEP‑517 build config
└── README.md                  # ★ This file ★
```

---

## Getting Started

### Prerequisites

* **Python 3.12+** (tested with 3.12.3)
* **uv**  for dependency management
* **GPU** (optional but recommended for GLiNER/GLiREL and LLM inference)
* Access to the Graphia LLM endpoint – set `API_KEY` in a ``.env`` file.

### Installation

```bash
# Clone the repository
git clone https://github.com/your‑org/gotriple_extractor.git
cd gotriple_extractor

# Install dependencies (uv preferred)
uv sync
# or, using pip
pip install -r requirements.txt

# Load environment variables
cp .env   # Edit .env to add your API_KEY and AVAILABLE_MODELS
```

### Quick Test

```bash
# Run a tiny end‑to‑end entity extraction on a single discipline
python entity_extraction.py
```

You should see JSON output under ``predictions-llm/`` containing entities extracted by the default LLM extractor. ``predicitons-gliner_spacy`` includes enitty extraction data using the more lightweight GLiNER and spaCy models.

---

## How It Works

### 1. Data Preparation

Raw GoTriple JSON files live in ``text_dataset/``.  The ``utils.clean_and_rechunk`` function:

1. Reads each line‑delimited JSON document.
2. Re‑assembles the original page text via ``clean_page_text``.
3. Recursively chunks the text (default 512 tokens, 20 char minimum) and adds a 50‑token overlap.
4. Writes the processed chunks to ``dataset/extracted/`` as ``.ndjson``.

If you want to replicate the text dataset creation, you can download the [GoTriple metadata dump](https://zenodo.org/records/15784401) and place the JSON files in the ``gotriple_dump`` directory.

### 2. Entity Extraction

The ``EntityExtractor`` class abstracts three concrete implementations:

| Backend | How it works | Typical use‑case |
|--------|--------------|-----------------|
| **gliner** | Loads a HuggingFace checkpoint (cached locally) and performs token‑level classification. | Fast, deterministic extraction when a suitable tag set is available. |
| **spacy** | Loads a multilingual spaCy pipeline (e.g., ``xx_ent_wiki_sm``) and pipes documents. | Low‑resource environments, quick prototyping. |
| **base-llm** | Sends the chunk to a remote LLM with a structured prompt (see ``prompt.md``). Supports ``n``‑best sampling. | Highest recall, flexible label set, language‑agnostic extraction. |

Extraction returns an ``AnnotatedChunk`` which stores the original span, a list of ``Entity`` objects, and the model type.

### 3. Relation Extraction

Implemented in ``relation_extraction.py``.  It can:

* Load GLiREL models or use the same LLM infrastructure with a relation‑specific prompt.
* Enforce **relation constraints** (allowed head/tail entity types) via the ``validate_relation_constraints`` helper.
* Sample a balanced set of candidate chunks for manual annotation (useful for active‑learning loops).

### 4. Linking

The ``linking/add_chunk_id.py`` script demonstrates a lightweight heuristic:

* For each extracted entity, it searches the original chunk list for the first occurrence.
* Stores the matching ``chunk_id`` alongside the linked GoTriple record.
* Unmatched entities are logged to ``linking/log.json`` for later inspection.

### 5. Benchmarking

The ``benchmark/run_benchmark.py`` script evaluates entity extraction quality on standard datasets:

* Loads a dataset parquet file, maps dataset‑specific entity labels to the GoTriple schema, filters empty examples, and processes them in 512‑sample chunks.
* Uses the GLiNER extractor (configurable threshold) to generate predictions.
* Writes per‑sample JSON lines to ``results/benchmark_results/combined_results.jsonl`` and aggregates timing information.
* Visualisations (precision‑recall curves, macro‑vs‑micro F1, label support) are pre‑generated under ``benchmark/benchmark_results/figures/``.

---

## Running the Full Pipeline

```bash

# 1️⃣ Entity extraction (example with LLM backend)
python entity_extraction.py

# 2️⃣ (Optional) Relation extraction – see the CLI in relation_extraction.py
python relation_extraction.py --predictions ./annotated_test-final --source ./text_dataset/extracted_2 --languages en it fr de --samples 500

# 3️⃣ Linking to GoTriple entries
python linking/add_chunk_id.py

# 4️⃣ Benchmark against external corpora
python benchmark/run_benchmark.py
```

Each step writes its artefacts to clearly named directories; you can rerun any step independently by adjusting the input paths.

---

## Configuration

* **Model selection** – pass ``model_type``/``model_name`` to the ``EntityExtractor`` or ``RelationExtractor`` constructors.
* **Prompt customization** – edit ``prompt.md``; the placeholder ``{ENTITY_TAGS}`` is automatically substituted with the tag list you provide.
* **Environment variables** – ``API_KEY`` (Graphia LLM), ``AVAILABLE_MODELS`` (comma‑separated list of permitted remote models) are read from ``.env``.
* **Chunking parameters** – tweak ``chunk_size`` and ``context_size`` in ``utils.clean_and_rechunk``.

---

## Extending the Toolkit

* **Add a new entity backend** – implement a class with ``extract`` and ``extract_doc`` methods and register it in ``EntityExtractor._load_model``.
* **Custom relation schema** – modify ``DEFAULT_RELATION_CONSTRAINTS`` in ``relation_extraction.py`` and re‑run the sampling utilities.
* **Fine‑tune GLiNER/GLiREL** – replace the HuggingFace repo ID with your fine‑tuned checkpoint; the cache will be reused.

---

## License & Citation

This project is released under the **MIT License**.  If you use the toolkit in academic work, please cite the original GoTriple paper and this repository:


## Acknowledgements

We thank the developers of **GLiNER**, **GLiREL**, **spaCy**, and the Graphia team for providing the LLM endpoint.  Special thanks to the contributors of the CONER, NERD, and SciERC benchmark datasets.
