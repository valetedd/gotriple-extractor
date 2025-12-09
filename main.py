import json
import os
from collections import defaultdict
from pathlib import Path
from time import time
from typing import List

import polars as pl

from extraction import EntityExtractor


def extract_entities(
    f: Path | str,
    extractors: List[EntityExtractor],
    discipline: str,
    write_to: Path | str,
) -> None:
    data = (
        pl.scan_ndjson(f).select(pl.col("in_language"), pl.col("text")).with_row_index()
    )
    result = []
    elapsed_total = defaultdict(float)

    for id, lang, doc in data.collect(engine="streaming").iter_rows():
        for extr in extractors:
            start = time()
            ann_doc = extr.extract_doc(doc, id)
            elapsed_total[extr.type] += time() - start
            result.append(
                {
                    "doc_id": f"{discipline}_{ann_doc.doc_id}",
                    "model": extr.type,
                    "lang": lang,
                    "ents": [
                        (chunk.chunk_id, chunk.annotations) for chunk in ann_doc.chunks
                    ],
                }
            )

    with open(f"{write_to}/{discipline}_time.json", mode="w", encoding="utf-8") as ot:
        json.dump(elapsed_total, ot)

    with open(f"{write_to}/{discipline}.json", mode="w", encoding="utf-8") as o:
        json.dump(result, o)


def main(path, output_dir, extractors: List[EntityExtractor]):
    # NOTE: Assuming that acquisition as been run
    os.makedirs(output_dir, exist_ok=True)
    for file in Path(path).iterdir():  # Iterating over discipline-specific JSONs
        discipline = file.name.split(".", maxsplit=1)[0]

        print("Processing now: ", file.name)
        extract_entities(file, extractors, discipline=discipline, write_to=output_dir)


if __name__ == "__main__":

    entity_types = ["person name", "organization", "location name", "time references"]

    extr_llm = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=entity_types,
        model_config={"temperature": 0.2},
        prompt="./prompt.md",
    )

    # extr_spacy = EntityExtractor(model_type="spacy", model_name="xx_ent_wiki_sm")
    # extr_gliner = EntityExtractor(
    #     model_type="gliner",
    #     model_name="knowledgator/gliner-x-large",
    #     entity_tags=entity_types,
    # )

    main(
        path="./dataset/extracted_2/",
        output_dir="./annotated_test-rechunked/llm/",
        extractors=[extr_llm],
    )
