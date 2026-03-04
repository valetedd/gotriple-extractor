import json
import os
from collections import defaultdict
from pathlib import Path
from time import sleep, time
from typing import List

import polars as pl
from tqdm import tqdm

from data_repr import *
from extraction import EntityExtractor


def extract_entities(
    f: Path | str,
    extractors: List[EntityExtractor],
    discipline: str,
    write_to: Path | str,
) -> None:
    data = (
        pl.read_json(f).select(pl.col("in_language"), pl.col("text")).with_row_index()
    )
    if discipline == "lang":
        print("restarting from doc 16 for lang")
        data = data.with_columns(
            pl.col("id").str.split("_").list.last().cast(pl.Int32).alias("doc_n")
        ).filter(pl.col("doc_n") >= 16)
        discipline = "lang2"
    result = []
    elapsed_total = defaultdict(float)

    total_docs = data.height
    with tqdm(total=total_docs, desc=discipline, unit="doc", leave=True) as pbar:
        for id, lang, doc in data.iter_rows():
            sleep(10.0)
            for extr in extractors:
                start = time()
                ann_doc = extr.extract_doc(doc, id, max_concurrent=50)
                elapsed_total[extr.type] += time() - start

                # Handling cases where multiple choices are output at once
                # Necessary only for LLM preds with n > 1
                if isinstance(ann_doc, list):
                    doc_id = ann_doc[0].doc_id
                    repeated_chunks: List[List[AnnotatedChunk]] = [
                        choice.chunks for choice in ann_doc
                    ]
                    annotations = map(
                        lambda lst: [x.annotations for x in lst], repeated_chunks
                    )
                    ents = {
                        f"chunk_{n}": choice for n, choice in enumerate(annotations)
                    }
                else:
                    doc_id = ann_doc.doc_id
                    ents = [chunk.annotations for chunk in ann_doc.chunks]
                result.append(
                    {
                        "doc_id": f"{discipline}_{doc_id}",
                        "model": extr.type,
                        "lang": lang,
                        "ents": ents,
                    }
                )
            pbar.update(1)

    with open(f"{write_to}/{discipline}_time.json", mode="w", encoding="utf-8") as ot:
        json.dump(elapsed_total, ot)

    with open(f"{write_to}/{discipline}.json", mode="w", encoding="utf-8") as o:
        json.dump(result, o)


def main(path, output_dir, extractors: List[EntityExtractor]):
    # NOTE: Assuming that acquisition as been run
    os.makedirs(output_dir, exist_ok=True)
    files = list(Path(path).iterdir())  # Iterating over discipline-specific JSONs
    for file in files:
        if file.name in [
            f.name for f in Path(output_dir).iterdir()
        ]:  # useful for stopping and resuming
            continue
        discipline = file.name.split(".", maxsplit=1)[0]

        print("Processing now: ", file.name)
        extract_entities(file, extractors, discipline=discipline, write_to=output_dir)


if __name__ == "__main__":

    #####################  AS PER DELIVERABLE ####################
    # "Person",
    # "Organization",
    # "Event",
    # "Place",
    # "Point of Interest",
    # "Time Period",
    # "Cultural Object",
    # "Archival Metadata",
    # "Citation",
    # "Publication",
    # "Project",
    # "Dataset",
    # "Semantic Artefact",
    # "Software",

    ###################################

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

    extr_llm = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=entity_types,
        model_config={"temperature": 0.2, "n": 4},
        prompt="./prompt.md",
    )

    # extr_spacy = EntityExtractor(model_type="spacy", model_name="xx_ent_wiki_sm")
    # extr_gliner = EntityExtractor(
    #     model_type="gliner",
    #     model_name="knowledgator/gliner-x-large",
    #     entity_tags=entity_types,
    #     model_config={"threshold": 0.6},
    # )

    main(
        path="./text_dataset/extracted_2/",
        output_dir="./annotated_test-final/",
        extractors=[extr_llm],
    )
