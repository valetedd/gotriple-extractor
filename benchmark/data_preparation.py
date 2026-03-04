from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset


def extract_from_iob2(
    tokens: List[str], ner_tags: List[str], mapping: Dict[str, str] | None = None
) -> List[Dict[str, str]]:
    """
    Extract entities from IOB2 format.

    Args:
        tokens: List of tokens
        ner_tags: List of NER tags in IOB2 format (e.g., 'B-PER', 'I-PER', 'O')
        mapping: Dict of integers to a certain B or I label

    Returns:
        List of dictionaries with 'entity' (surface form) and 'label' (entity type)
    """
    entities = []
    current_entity = []
    current_label = None

    for token, tag in zip(tokens, ner_tags):
        if (isinstance(tag, int) or type(tag) == np.int64) and not mapping:
            raise ValueError(
                "A mapping has to be specified for datasets with numeric labels"
            )
        if tag == 0:
            continue
        if mapping:
            tag = mapping[tag]
        if tag.startswith("B-"):
            # Save previous entity if exists
            if current_entity:
                entities.append([" ".join(current_entity), current_label])
            # Start new entity
            current_entity = [token]
            current_label = tag[2:]  # Remove 'B-' prefix

        elif tag.startswith("I-"):
            if current_entity and current_label == tag[2:]:
                current_entity.append(token)
            else:
                # Handle malformed annotation - treat as new entity
                if current_entity:
                    entities.append([" ".join(current_entity), current_label])
                current_entity = [token]
                current_label = tag[2:]

        else:  # 'O' or other non-entity tag
            if current_entity:
                entities.append([" ".join(current_entity), current_label])
                current_entity = []
                current_label = None

    if current_entity:
        entities.append([" ".join(current_entity), current_label])
    return entities


def convert_dataset(
    df: pd.DataFrame,
    tokens_col: str = "tokens",
    ner_col: str = "ner_tags",
    mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Convert IOB2 dataset to simplified entity format.

    Args:
        df: DataFrame with tokens and NER tags
        tokens_col: Name of the tokens column
        ner_col: Name of the NER tags column

    Returns:
        DataFrame with 'text' and 'entities' columns
    """
    results = []

    for idx, row in df.iterrows():
        tokens = row[tokens_col]
        ner_tags = row[ner_col]

        # Convert to strings if they're not already
        if isinstance(ner_tags[0], int):
            # If tags are encoded as integers, cast to string
            ner_tags = [str(tag) for tag in ner_tags]

        # Create full text
        text = " ".join(tokens)

        # Extract entities
        entities = extract_from_iob2(tokens, ner_tags, mapping)

        results.append({"text": text, "entities": entities})

    return pd.DataFrame(results)


def main():
    multinerd = pd.read_parquet("./benchmark/original_datasets/multinerd_val.parquet")
    multiconer: pd.DataFrame = load_dataset(
        "multiconer/multiconer_v2", "Multilingual (MULTI)", trust_remote_code=True
    )["train"].to_pandas()
    scier_ds = pd.read_json("./benchmark/original_datasets/scier.jsonl", lines=True)[
        ["doc_id", "sentence", "ner"]
    ]
    scier_ds.columns = ["doc_id", "text", "entities"]

    # needed cause multinerd uses only integer labels
    multinerd_mapping = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    # # invert index for fast access
    multinerd_mapping = {v: k for k, v in multinerd_mapping.items()}

    converted_nerd = convert_dataset(multinerd, mapping=multinerd_mapping)
    print(converted_nerd.columns, "\n", converted_nerd.head())
    converted_nerd.to_parquet("./benchmark/nerd.parquet")

    converted_multiconer = convert_dataset(multiconer)
    print(converted_multiconer.columns, "\n", converted_multiconer.head())
    converted_multiconer.to_parquet("./benchmark/coner.parquet")

    print(scier_ds.columns, "\n", scier_ds.head())
    scier_ds.to_parquet("./benchmark/scier.parquet")


if __name__ == "__main__":
    main()
