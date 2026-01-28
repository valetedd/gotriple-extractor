import os

import pandas as pd
from datasets import load_dataset

from data_repr import *
from extraction import EntityExtractor

DATASET_PATHS = [
    "./benchmarking_data/coner.parquet",
    "./benchmarking_data/nerd.parquet",
    "./benchmarking_data/scier.parquet",
]

### ---------- Mappings to make labels more usable in prompting setting ----------- ###
multiconer_mapping = {
    # ───────────── Location (LOC) ─────────────
    "Facility": "Location",
    "OtherLOC": "Location",
    "HumanSettlement": "Location",
    "Station": "Location",
    # ───────────── Cultural Object ─────────────
    "VisualWork": "Cultural Object",
    "MusicalWork": "Cultural Object",
    "WrittenWork": "Cultural Object",
    "ArtWork": "Creative Object",
    "Software": "Software",  # Only fine-grained entity tag
    # ───────────── Group (GRP) ─────────────
    "MusicalGRP": "Organization",
    "PublicCORP": "Organization",
    "PrivateCORP": "Organization",
    "AerospaceManufacturer": "Organization",
    "SportsGRP": "Organization",
    "CarManufacturer": "Organization",
    "ORG": "Organization",
    # ───────────── Person (PER) ─────────────
    "Scientist": "Person",
    "Artist": "Person",
    "Athlete": "Person",
    "Politician": "Person",
    "Cleric": "Person",
    "SportsManager": "Person",
    "OtherPER": "Person",
    # ───────────── Product (PROD) ─────────────
    "Clothing": "Product",
    "Vehicle": "Product",
    "Food": "Product",
    "Drink": "Product",
    "OtherPROD": "Product",
    # ───────────── Medical (MED) ─────────────
    "Medication/Vaccine": "Medical term",
    "MedicalProcedure": "Medical term",
    "AnatomicalStructure": "Medical term",
    "Symptom": "Medical term",
    "Disease": "Medical term",
}

multinerd_mapping = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "ANIM": "Animal",
    "BIO": "Biological Entity",
    "CEL": "Celestial Object",
    "DIS": "Disease",
    "EVE": "Event",
    "FOOD": "Food",
    "INST": "Instrument",
    "MEDIA": "Media",
    "MYTH": "Mythological Entity",
    "PLANT": "Plant",
    "TIME": "Time",
    "VEHI": "Vehicle",
}

# Entities in benchmark overlapping with GoTriple data
gotriple_ents = [
    "Person",
    "Organization",
    "Location",
    "Cultural Object",
    "Event",
    "Software",
    "Dataset",
    "Time",
]


def run_benchmark(ds_path, model: EntityExtractor):

    dataset = pd.read_parquet(ds_path)
    inputs = dataset["text"].to_list()
    preds = model.extract_doc(inputs, max_concurrent=50)


def main():
    os.makedirs("./benchmarking_data/benchmark_results/", exist_ok=True)

    extr_llm = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        model_config={"temperature": 0.2},
        entity_tags=gotriple_ents,
        prompt="./prompt.md",
    )

    extr_gliner = EntityExtractor(
        model_type="gliner",
        model_name="knowledgator/gliner-x-large",
        entity_tags=gotriple_ents,
        model_config={"threshold": 0.6, "batch_size": 64},
    )

    for dataset in DATASET_PATHS:
        run_benchmark(dataset, extr_gliner)
        run_benchmark(dataset, extr_llm)


if __name__ == "__main__":
    for ds in DATASET_PATHS:
        df = pd.read_parquet(ds)
        avg_len = df["text"].str.len().mean()
        print(ds.split("/")[-1], ": ", avg_len)

    main()
