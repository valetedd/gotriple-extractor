import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple

import pandas as pd

WORKING_DIR = Path(__file__).parent.parent
print(WORKING_DIR)
sys.path.insert(0, str(WORKING_DIR))

from extraction import EntityExtractor

OUTPUT_DIR = WORKING_DIR / Path(__file__).parent


def _judge_payload(text: str, annotations: Any) -> str:
    """
    Build a single-string payload for the judge prompt.
    Keeping it JSON makes it easy for the judge to parse reliably.
    """
    return json.dumps({"text": text, "annotations": annotations}, ensure_ascii=False)


def _process_doc_sync(
    *,
    discipline: str,
    doc_id: int,
    lang: str | None,
    doc: List[str] | str,
    annotator: EntityExtractor,
    judge: EntityExtractor,
    ann_max_concurrent: int,
    judge_max_concurrent: int,
    attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Runs annotator -> judge for one document.
    This is deliberately synchronous so it can be executed in a thread via asyncio.to_thread.

    Returns:
      (record, timing_dict)
    """
    timing: Dict[str, float] = defaultdict(float)

    # Normalize doc format
    if isinstance(doc, str):
        doc = [doc]

    # 1) Annotate (doc-level; chunk fan-out happens inside extractor)
    t0 = time()
    ann_doc = annotator.extract_doc(
        doc, doc_id, max_concurrent=ann_max_concurrent, attempts=attempts
    )
    timing["annotator_sec"] += time() - t0

    # Build judge inputs per chunk
    judge_inputs: List[str] = [
        _judge_payload(chunk.span, chunk.annotations) for chunk in ann_doc.chunks
    ]

    # 2) Judge (doc-level; chunk fan-out happens inside extractor)
    t1 = time()
    judged_doc = judge.extract_doc(
        judge_inputs, doc_id, max_concurrent=judge_max_concurrent, attempts=attempts
    )
    timing["judge_sec"] += time() - t1

    # Pack results (keep the same overall style as main.py)
    record = {
        "doc_id": f"{discipline}_{ann_doc.doc_id}",
        "lang": lang,
        "annotator_model": annotator.name,
        "judge_model": judge.name,
        "annotator_ents": [(c.chunk_id, c.annotations) for c in ann_doc.chunks],
        # judged_doc.chunks are aligned with judge_inputs order (chunk_id set by extract_doc enumerator)
        "judged_ents": [(c.chunk_id, c.annotations) for c in judged_doc.chunks],
    }
    return record, timing


async def annotate_and_judge_domain(
    *,
    fpath: Path,
    annotator: EntityExtractor,
    judge: EntityExtractor,
    out_dir: Path,
    doc_concurrency: int = 8,
    ann_max_concurrent: int = 20,
    judge_max_concurrent: int = 20,
    attempts: int = 3,
) -> None:
    """
    Async doc-parallel runner for one discipline file.
    """
    discipline = fpath.name.split(".", maxsplit=1)[0]

    # Stop/resume support
    out_json = out_dir / f"{discipline}.json"
    out_time = out_dir / f"{discipline}_time.json"
    if out_json.exists():
        return

    print(fpath)
    data = pd.read_json(fpath)
    required_cols = {"id", "in_language", "text"}
    missing = required_cols - set(data.columns)
    if "id" in missing:
        data = data.reset_index().rename(columns={"index": "id"})
        missing.remove("id")
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[["id", "in_language", "text"]]
    print(data)

    results: List[Dict[str, Any]] = []
    elapsed_total: Dict[str, float] = defaultdict(float)
    queue: asyncio.Queue = asyncio.Queue(maxsize=doc_concurrency * 2)

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            row_id, lang, doc = item
            record, timing = await asyncio.to_thread(
                _process_doc_sync,
                discipline=discipline,
                doc_id=row_id,
                lang=lang,
                doc=doc,
                annotator=annotator,
                judge=judge,
                ann_max_concurrent=ann_max_concurrent,
                judge_max_concurrent=judge_max_concurrent,
                attempts=attempts,
            )
            results.append(record)
            for k, v in timing.items():
                elapsed_total[k] += v
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(doc_concurrency)]

    for row in data.itertuples(index=False):
        await queue.put((row.id, row.in_language, row.text))

    for _ in workers:
        await queue.put(None)

    await queue.join()
    await asyncio.gather(*workers)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    with open(out_time, "w", encoding="utf-8") as ot:
        json.dump(elapsed_total, ot)

    with open(out_json, "w", encoding="utf-8") as o:
        json.dump(results, o, ensure_ascii=False)


async def main_async():
    entity_types = [
        "Person",
        "Organization",
        "Event",
        "Place",
        "Point of Interest",
        "Time",
        "Cultural_Object",
        "Archival Metadata",
        "Citation",
        "Publication title",
        "Project",
        "Dataset",
        "Semantic Artefact (ontology or vocabulary terms)",
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
        model_name="Qwen3-Coder-30B-A3B-Instruct-Q8_0",
        entity_tags=entity_types,
        model_config={"temperature": 0.2},
        prompt=WORKING_DIR / "silver_standard/judge_prompt.md",
    )

    data_path = WORKING_DIR / "text_dataset/extracted_2/"
    out_dir = OUTPUT_DIR / "annotated_judged"
    os.makedirs(out_dir, exist_ok=True)

    # Tune these for your hardware / rate limits:
    DOC_CONCURRENCY = 8  # how many docs in-flight
    ANN_MAX_CONCURRENT = 20  # annotator chunk fan-out per doc
    JUDGE_MAX_CONCURRENT = 30  # judge chunk fan-out per doc (often can be higher)
    ATTEMPTS = 3

    for f in data_path.iterdir():

        print("Processing now:", f.name)
        await annotate_and_judge_domain(
            fpath=f,
            annotator=annotator,
            judge=judge,
            out_dir=out_dir,
            doc_concurrency=DOC_CONCURRENCY,
            ann_max_concurrent=ANN_MAX_CONCURRENT,
            judge_max_concurrent=JUDGE_MAX_CONCURRENT,
            attempts=ATTEMPTS,
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
