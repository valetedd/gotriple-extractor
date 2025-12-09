import json
import os
import re
from pathlib import Path
from typing import List

import polars as pl
from chonkie.chunker import RecursiveChunker
from chonkie.refinery import OverlapRefinery

from dataset.build_dataset import clean_page_text


def fix_broken(
    root: Path = Path("./dataset/extracted_data"),
    broken_files: List[str] = ["anthro-bio.json", "hisphilso.json", "hist.json"],
):
    for fname in broken_files:
        file = root / fname
        text = file.read_text()

        # Split pretty-printed objects: each new object begins with "{" at the start of a line
        parts = re.split(r"(?m)^\{", text)
        objects = []

        for p in parts:
            p = p.strip()
            if not p:
                continue
            obj_text = "{" + p  # restore the leading bracket for each object
            try:
                obj = json.loads(obj_text)
                objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid chunk in {fname}: {e}")

        out_path = file.with_suffix(".ndjson")
        with out_path.open("w") as out:
            for obj in objects:
                out.write(json.dumps(obj) + "\n")

        print(f"✔ Recovered {len(objects)} JSON objects → {out_path.name}")


def merge_outputs(original_path, other_path, ignore=None, output_path=None):
    if not output_path:
        output_path = original_path

    # Load JSON files
    with open(original_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    with open(other_path, "r", encoding="utf-8") as f:
        other = json.load(f)

    # Convert other JSON to a dict by doc_id for fast lookup
    other_by_id = {doc["doc_id"]: doc for doc in other}

    merged = []

    for doc in orig:  # looping over doc data in original JSON
        if ignore:
            if doc["model"] in ignore:
                merged.append(doc)
                continue
        doc_id = doc["doc_id"]

        if doc_id not in other_by_id:
            print(
                f"Warning: doc_id {doc_id} not found in labeled data. Keeping original."
            )
            merged.append(doc)
            continue

        correct_doc = other_by_id[doc_id]
        merged.append(correct_doc)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged output saved to: {output_path}")


def add_true_id(path: Path, discipline: str, out_path: str | Path | None = None):
    if not out_path:
        out_path = path / discipline
    elif isinstance(out_path, Path):
        out_path = out_path / discipline

    else:
        out_path = out_path + discipline

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        obj["doc_id"] = str(obj["doc_id"])
        obj["doc_id"] += f"_{discipline[:-5]}"

    with open(out_path, "w", encoding="utf-8") as o:
        json.dump(data, o, indent=2, ensure_ascii=False)
    print(f"Saved new data to {out_path}")


def clean_and_rechunk(chunker, overlap_ref, path: Path, out_dir):
    if isinstance(out_dir, str):
        out_path = Path(out_dir) / path.name
    elif isinstance(out_dir, Path):
        out_path = out_dir / path.name
    else:
        raise ValueError("output_path can only be a Path object or a string")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for obj in data:
        reconstructed = " ".join(clean_page_text(b) for b in obj["text"])
        chunks = chunker(reconstructed)
        obj["text"] = [cleaned.text for cleaned in overlap_ref(chunks)]
    pl.LazyFrame(data).sink_ndjson(out_path)
    print(f"Saved new data to {out_path}")


if __name__ == "__main__":
    ORI = "./annotated_test-rechunked/"
    OTHER = "./annotated_test-rechunked/llm/"
    OUT = "./merged/"
    os.makedirs(OUT, exist_ok=True)
    for file in Path(ORI).iterdir():
        all = []

        if "time" in file.name or not file.is_file():
            continue
        print("Processing now", file.name)
        other_f = f"{OTHER}{file.name}"
        print(f"Merging {file} with {other_f}")

        with open(file, "r", encoding="utf-8") as f:
            orig = json.load(f)

        with open(other_f, "r", encoding="utf-8") as f:
            other = json.load(f)

        all.extend(orig)
        all.extend(other)

        with open(OUT + file.name, "w", encoding="utf-8") as f:
            json.dump(all, f, indent=2, ensure_ascii=False)
