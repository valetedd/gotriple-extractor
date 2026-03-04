import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

WORKING_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WORKING_DIR))

from extraction import EntityExtractor

LINKING_THRESHOLD = 0.80
OUTPUT_DIR = WORKING_DIR / Path(__file__).parent


def _judge_payload(text: str, annotations: Any, linked: List[str]) -> str:
    """
    Build a single-string payload for the judge prompt.
    Keeping it JSON makes it easy for the judge to parse reliably.
    """

    return f"""INPUT:\n {text}\n\n
               ANNOTATIONS:\n{annotations}\n\n
               LINKED ENTITIES:\n{linked}"""


def _normalize_doc_text(doc: List[str] | str) -> List[str]:
    if isinstance(doc, str):
        return [doc]
    return doc


def annotate_judge(
    *,
    fpath: Path,
    linked_data: Path,
    annotator: EntityExtractor,
    judge: EntityExtractor,
    out_dir: Path,
    ann_max_concurrent: int = 20,
    judge_max_concurrent: int = 20,
    chunk_batch_size: int = 500,
    attempts: int = 3,
) -> None:
    """
    Chunk-first pipeline for one discipline file.
    Output is written chunk-wise (no grouping/sorting by document).
    """
    discipline = fpath.name.split(".", maxsplit=1)[0]

    # Stop/resume support
    out_json = out_dir / f"{discipline}.json"
    out_time = out_dir / f"{discipline}_time.json"
    if out_json.exists():
        return
    if chunk_batch_size < 1:
        raise ValueError("chunk_batch_size must be >= 1")

    data = pd.read_json(fpath)[["id", "in_language", "text"]]

    with open(linked_data, mode="r", encoding="utf-8") as f:
        linked_raw = json.load(f)

    linked_by_doc_chunk: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for ent in linked_raw:
        doc_key = ent.get("doc_id")
        linked_entity = ent.get("linked_entity")
        if doc_key is None or not linked_entity:
            continue
        if ent.get("confidence", 0) <= LINKING_THRESHOLD:
            continue
        try:
            chunk_idx = int(ent.get("chunk_idx", -1))
        except (TypeError, ValueError):
            continue
        if chunk_idx < 0:
            continue
        linked_by_doc_chunk[(str(doc_key), chunk_idx)].append(linked_entity)

    # Flatten docs into chunk tasks
    flat_chunks: List[str] = []
    flat_meta: List[Tuple[str, int, str | None]] = []
    for row in data.itertuples(index=False):
        doc_id = str(row.id)
        doc_chunks = _normalize_doc_text(row.text)
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            flat_chunks.append(chunk_text)
            flat_meta.append((doc_id, chunk_idx, row.in_language))

    total_chunks = len(flat_chunks)
    if total_chunks == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_time, "w", encoding="utf-8") as ot:
            json.dump({}, ot)
        with open(out_json, "w", encoding="utf-8") as o:
            json.dump([], o, ensure_ascii=False)
        return

    results: List[Dict[str, Any]] = []
    elapsed_total: Dict[str, float] = defaultdict(float)

    started_at = datetime.now().astimezone()
    print(f"[{discipline}] started at {started_at.isoformat(timespec='seconds')}")

    total_steps = total_chunks * 2  # annotator + judge
    pbar = (
        tqdm(
            total=total_steps,
            desc=f"{discipline}",
            unit="step",
            leave=True,
            miniters=1,
            smoothing=0,
        )
        if tqdm is not None
        else None
    )

    try:
        for start_idx in range(0, total_chunks, chunk_batch_size):
            end_idx = min(start_idx + chunk_batch_size, total_chunks)
            chunk_batch = flat_chunks[start_idx:end_idx]
            meta_batch = flat_meta[start_idx:end_idx]

            t0 = time()
            ann_doc = annotator.extract_doc(
                chunk_batch,
                doc_id=start_idx,
                max_concurrent=ann_max_concurrent,
                attempts=attempts,
            )
            elapsed_total["annotator_sec"] += time() - t0
            if pbar is not None:
                pbar.update(len(chunk_batch))

            if len(ann_doc.chunks) != len(chunk_batch):
                raise RuntimeError(
                    f"[{discipline}] annotator returned {len(ann_doc.chunks)} chunks, expected {len(chunk_batch)}"
                )

            judge_inputs: List[str] = []
            for local_idx, ann_chunk in enumerate(ann_doc.chunks):
                doc_id, chunk_idx, _ = meta_batch[local_idx]
                linked = linked_by_doc_chunk.get((doc_id, chunk_idx), [])
                judge_inputs.append(
                    _judge_payload(ann_chunk.span, ann_chunk.annotations, linked)
                )

            t1 = time()
            judged_doc = judge.extract_doc(
                judge_inputs,
                doc_id=start_idx,
                max_concurrent=judge_max_concurrent,
                attempts=attempts,
            )
            elapsed_total["judge_sec"] += time() - t1
            if pbar is not None:
                pbar.update(len(chunk_batch))

            if len(judged_doc.chunks) != len(chunk_batch):
                raise RuntimeError(
                    f"[{discipline}] judge returned {len(judged_doc.chunks)} chunks, expected {len(chunk_batch)}"
                )

            for local_idx, (ann_chunk, judged_chunk) in enumerate(
                zip(ann_doc.chunks, judged_doc.chunks)
            ):
                doc_id, chunk_idx, lang = meta_batch[local_idx]
                results.append(
                    {
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "lang": lang,
                        "annotator_model": annotator.name,
                        "judge_model": judge.name,
                        "annotator_ents": ann_chunk.annotations,
                        "judged_ents": judged_chunk.annotations,
                    }
                )

    finally:
        if pbar is not None:
            pbar.close()

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    with open(out_time, "w", encoding="utf-8") as ot:
        json.dump(elapsed_total, ot)

    with open(out_json, "w", encoding="utf-8") as o:
        json.dump(results, o, ensure_ascii=False)


def main():
    entity_types = [
        "Person",
        "Organization",
        "Event",
        "Place",
        "Point of Interest",
        "Time",
        "Creative Work",
        "Archival Metadata",
        "Citation",
        "Publication title",
        "Project",
        "Dataset",
        "Semantic Web vocabulary",
        "Software",
    ]

    annotator = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=entity_types,
        model_config={"temperature": 0.2},
        prompt=WORKING_DIR / "silver_standard/ann_prompt.md",
    )

    judge = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=entity_types,
        model_config={"temperature": 0.2},
        prompt=WORKING_DIR / "silver_standard/judge_prompt.md",
    )

    data_path = WORKING_DIR / "text_dataset/extracted_2/"
    out_dir = OUTPUT_DIR / "annotated_judged"
    os.makedirs(out_dir, exist_ok=True)

    CHUNK_BATCH_SIZE = 50
    ANN_MAX_CONCURRENT = 50
    JUDGE_MAX_CONCURRENT = 50
    ATTEMPTS = 3

    for f in data_path.iterdir():

        print("Processing now:", f.name)
        annotate_judge(
            fpath=f,
            linked_data=Path("./linking/mgenre_gliner.json"),
            annotator=annotator,
            judge=judge,
            out_dir=out_dir,
            chunk_batch_size=CHUNK_BATCH_SIZE,
            ann_max_concurrent=ANN_MAX_CONCURRENT,
            judge_max_concurrent=JUDGE_MAX_CONCURRENT,
            attempts=ATTEMPTS,
        )


if __name__ == "__main__":
    main()
