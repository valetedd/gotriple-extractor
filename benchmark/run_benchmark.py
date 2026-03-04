import gc
import json
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

# Paths
DATASET_PATHS = [
    "./benchmark/coner.parquet",
    "./benchmark/nerd.parquet",
    "./benchmark/scier.parquet",
]

OUTPUT_PATH = "./results/benchmark_results/"
TIMINGS_PATH = os.path.join(OUTPUT_PATH, "timings.json")
os.makedirs(OUTPUT_PATH, exist_ok=True)

OUTPUT_JSONL = os.path.join(OUTPUT_PATH, "combined_results.jsonl")

# Entity mappings
multiconer_mapping = {
    "Facility": "Location",
    "OtherLOC": "Location",
    "HumanSettlement": "Location",
    "Station": "Location",
    "VisualWork": "Creative Work",
    "MusicalWork": "Creative Work",
    "WrittenWork": "Creative Work",
    "ArtWork": "Creative Object",
    "Software": "Software",
    "MusicalGRP": "Organization",
    "PublicCORP": "Organization",
    "PrivateCORP": "Organization",
    "AerospaceManufacturer": "Organization",
    "SportsGRP": "Organization",
    "CarManufacturer": "Organization",
    "ORG": "Organization",
    "Scientist": "Person",
    "Artist": "Person",
    "Athlete": "Person",
    "Politician": "Person",
    "Cleric": "Person",
    "SportsManager": "Person",
    "OtherPER": "Person",
    "Clothing": "Product",
    "Vehicle": "Product",
    "Food": "Product",
    "Drink": "Product",
    "OtherPROD": "Product",
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

gotriple_ents = [
    "Person",
    "Organization",
    "Location",
    "Creative Work",
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


def main():

    extr_gliner = EntityExtractor(
        model_type="gliner",
        model_name="knowledgator/gliner-x-large",
        entity_tags=gotriple_ents,
        model_config={"threshold": 0.8, "batch_size": 128},
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
        start_time = time.perf_counter()
        num_samples = 0
        with tqdm(desc=ds_name, unit="chunks") as pbar:
            for chunk in stream_filtered_parquet(ds_path, mapping, chunksize=512):

                num_samples += len(chunk)

                doc_annotations = extr_gliner.extract_doc(chunk["text"].tolist())

                pbar.update(1)

                preds = [ann.annotations for ann in doc_annotations.chunks]
                with open(OUTPUT_JSONL, "a") as f:
                    for text, gold, pred in zip(
                        chunk["text"], chunk["entities"], preds
                    ):
                        f.write(
                            json.dumps(
                                {
                                    "text": text,
                                    "gold_entities": gold,
                                    "preds_gliner": pred,
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

    print(f"\n Results written to {OUTPUT_JSONL}")
    print(f"Timings written to {TIMINGS_PATH}")


main()
