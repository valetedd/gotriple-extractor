from typing import List

import polars as pl

from chunking import dump_to_text
from extraction import *


def main():
    # NOTE:Assuming that chunking as been run

    data: List[List[str]] = (
        pl.scan_ndjson("./extracted_data/")
        .select(pl.col("chunks"))
        .collect()
        .to_series()
        .to_list()
    )

    entity_types = [
        "Person",
        "Organization",
        "Event",
        "Place",  # schema:Place, dcterms:Location, cidoc:E53_Place
        "Point of Interest",  # schema:Place, gn:Feature (geonames), cidoc:E27_Site
        "Time Period",  # dcterms:PeriodOfTime, cidoc:E52_Time-Span
        "Cultural Object",  # schema:CreativeWork, cidoc:E22_Man-Made_Object (or cidoc:E24_Physical_Man-Made_Thing)·
        "Archival Metadata",  # dcterms:BibliographicResource? rico:Record or rico:RecordSet (RiC-O)?
        "Citation",  # cito:Citation (Open Citations)
        "Publication",  # triple:Document, fabio:Expression, foaf:Document
        "Project",  # triple:Project, schema:ResearchProject, foaf:Project
        "Dataset",  # triple:Dataset, schema:Dataset, dcat:Dataset
        "Semantic Artefact",  # triple:SemanticArtefact, adms:SemanticAsset
        "Software",  # triple:Software, swo:SWO_0000001 (Software ontology), schema:SoftwareSourceCode, schema:SoftwareApplication
    ]
    extr = EntityExtractor(
        "gliner", model_name="knowledgator/gliner-x-large", entity_tags=entity_types
    )
    for doc in data:
        page_entities = []
        for page in doc:

            for chunk in page:
                if len(chunk) < 20:
                    continue
                chunk_entities = extr.extract(chunk)
                print("==================")
                print(f"Chunk: \n{chunk}")
                print(f"\nEntities: {chunk_entities.annotations}")
                print("==================")
                print("\n\n")
                if chunk_entities:
                    page_entities.append(chunk_entities)


if __name__ == "__main__":
    main()
