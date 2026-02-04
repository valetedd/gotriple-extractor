import gc
import json
import os
import time

import pandas as pd
import torch
from tqdm.auto import tqdm

from data_repr import *
from extraction import EntityExtractor, Extractor

DATASET_PATHS = [
    "./benchmarking_data/coner.parquet",
    "./benchmarking_data/nerd.parquet",
    "./benchmarking_data/scier.parquet",
]

OUTPUT_PATH = "./benchmarking_data/benchmark_results/"
TIMINGS_PATH = os.path.join(OUTPUT_PATH, "timings.json")
os.makedirs(OUTPUT_PATH, exist_ok=True)

OUTPUT_JSONL = os.path.join(OUTPUT_PATH, "combined_results.jsonl")

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
    "ArtWork": "Cultural Object",
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
    "EVE": "Event",
    "TIME": "Time",
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


def filter_entities(entities, mapping):
    """Filter and remap entity labels based on the mappings specified above."""
    filtered = []

    for ent_text, ent_label in entities:
        if mapping:
            mapped = mapping.get(ent_label)
            if mapped in gotriple_ents:
                filtered.append((ent_text, mapped))
        else:
            # for scier data
            if ent_label == "Dataset":
                filtered.append((ent_text, "Dataset"))

    return filtered


def stream_filtered_parquet(path, mapping, chunksize=512):
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} samples from {path}")

    df["entities"] = df["entities"].apply(lambda x: filter_entities(x, mapping))

    df = df[df["entities"].map(len) > 0]
    print(f"Kept {len(df)} samples after filtering")

    for i in range(0, len(df), chunksize):
        yield df.iloc[i : i + chunksize]

    del df


def run_benchmark(
    extr: Extractor,
    ds_name: str,
    ds_path: str,
    timings: dict,
    mapping=None,
):
    start_time = time.perf_counter()
    num_samples = 0
    with tqdm(desc=ds_name, unit="chunks") as pbar:
        for chunk in stream_filtered_parquet(ds_path, mapping, chunksize=512):

            num_samples += len(chunk)

            doc_annotations = extr.extract_doc(chunk["text"].tolist())

            pbar.update(1)

            preds = [ann.annotations for ann in doc_annotations.chunks]
            with open(OUTPUT_JSONL, "a") as f:
                for text, gold, pred in zip(chunk["text"], chunk["entities"], preds):
                    f.write(
                        json.dumps(
                            {
                                "text": text,
                                "gold_entities": gold,
                                f"preds_{extr.type}": pred,
                                "source": ds_name,
                            }
                        )
                        + "\n"
                    )

            del chunk, preds
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - start_time

        timings[ds_name] = {
            "num_samples": num_samples,
            "seconds": round(elapsed, 2),
        }

        with open(TIMINGS_PATH, "w") as f:
            json.dump(timings, f, indent=2)

        print(f"Finished {ds_name}: {num_samples} samples " f"in {elapsed:.2f}s")


def main():

    # extr_gliner = EntityExtractor(
    #     model_type="gliner",
    #     model_name="knowledgator/gliner-x-large",
    #     entity_tags=gotriple_ents,
    #     model_config={"threshold": 0.6, "batch_size": 16},
    # )

    extr_llm = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=gotriple_ents,
        model_config={"temperature": 0.2},
        prompt="./prompt.md",
    )

    open(OUTPUT_JSONL, "w").close()
    timings = {}

    for ds_path in DATASET_PATHS:
        ds_name = os.path.basename(ds_path).replace(".parquet", "")

        if ds_name == "coner":
            mapping = multiconer_mapping
        elif ds_name == "nerd":
            mapping = multinerd_mapping
        else:
            mapping = None

        print(f"\nProcessing {ds_name}...")
        # gliner_results = run_benchmark(
        #     extr=extr_gliner,
        #     ds_path=ds_path,
        #     ds_name=ds_name,
        #     mapping=mapping,
        #     timings=timings,
        # )
        llm_results = run_benchmark(
            extr=extr_llm,
            ds_name=ds_name,
            mapping=mapping,
            ds_path=ds_path,
            timings=timings,
        )

    print(f"\n Results written to {OUTPUT_JSONL}")
    print(f"Timings written to {TIMINGS_PATH}")


if __name__ == "__main__":
    for ds in DATASET_PATHS:
        df = pd.read_parquet(ds)
        avg_len = df["text"].str.len().mean()
        print(ds.split("/")[-1], ": ", avg_len)

    main()
