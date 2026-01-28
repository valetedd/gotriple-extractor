from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline

DEVICE = 0 if torch.cuda.is_available() else -1

NEL_MODEL_NAME = "impresso-project/nel-mgenre-multilingual"


class mGENREEntityData(Dataset):
    def __init__(
        self,
        extr_df: pd.DataFrame,
        full_text_data: pd.DataFrame,
        from_model: str,
        batch_size=16,
    ):
        extr = extr_df[extr_df["model"] == from_model]
        self.from_model = from_model
        joined = extr.merge(full_text_data, left_on="doc_id", right_on="id")
        del extr  # Free memory
        joined.rename(columns={"text": "chunks"}, inplace=True)
        # First explosion: one row per chunk annotation
        exploded = joined.explode("ents")
        del joined  # Free memory
        # Creating separate columns for chunk idx and annotation list of dicts
        exploded["chunk_idx"] = exploded["ents"].apply(lambda r: r[0])
        exploded["ents"] = exploded["ents"].apply(lambda r: r[1])
        # Second explosion: one row per entity annotation (in each chunk)
        df = exploded.explode(column="ents").reset_index(drop=True)
        del exploded  # Free memory
        print("After explosion\n", df)

        # Extract entity fields directly without json_normalize
        df["start"] = df["ents"].apply(
            lambda x: x.get("start") if isinstance(x, dict) else None
        )
        df["end"] = df["ents"].apply(
            lambda x: x.get("end") if isinstance(x, dict) else None
        )
        df["text"] = df["ents"].apply(
            lambda x: x.get("text") if isinstance(x, dict) else None
        )
        df["label"] = df["ents"].apply(
            lambda x: x.get("label") if isinstance(x, dict) else None
        )
        df.drop(columns=["ents", "id"], inplace=True)
        df = df.dropna(subset="text")
        df = df.drop_duplicates(subset=["doc_id", "text"])
        print(f"Unique doc_ids: {df['doc_id'].nunique()}")
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        print("After dropping\n", df)
        # Extract only the relevant chunk and annotate it (no need to copy entire list)
        df["annotated_chunk"] = df.apply(self.insert_annotations, axis=1)
        df.drop(columns=["chunks"], inplace=True)
        self.data = df
        self.data.to_json(
            f"test_{self.from_model}.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )

    def insert_annotations(self, s):
        chunk_idx = s["chunk_idx"]
        ent = s["text"]
        chunk = s["chunks"][chunk_idx]
        return chunk.replace(ent, f"[START] {ent} [END]", 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "doc_id": row["doc_id"],
            "text": row["text"],
            "label": row["label"],
            "annotated_chunk": row["annotated_chunk"],
        }


def main():
    # Load the NEL pipeline
    nel_tokenizer = AutoTokenizer.from_pretrained(NEL_MODEL_NAME)
    nel_pipeline = pipeline(
        task="generic-nel",
        model=NEL_MODEL_NAME,
        tokenizer=nel_tokenizer,
        trust_remote_code=True,
        device=DEVICE,
    )

    # Load data
    extr_df = pd.concat(
        pd.read_json(f, orient="records") for f in Path("../annotated_test/").iterdir()
    )
    full_text_df = pd.concat(
        pd.read_json(f, orient="records")
        for f in Path("../dataset/extracted_2/").iterdir()
    )
    dataset = mGENREEntityData(
        extr_df=extr_df, from_model="gliner", full_text_data=full_text_df
    )

    # Run linking in batches
    batch_size = 256
    results = []

    for i in range(0, len(dataset), batch_size):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch = [dataset[j] for j in batch_indices]

        # Extract annotated chunks for the pipeline
        texts = [item["annotated_chunk"] for item in batch]

        # Run mGENRE pipeline on batch
        predictions = nel_pipeline(texts, num_beams=5, num_return_sequences=1)

        # Combine predictions with original data
        for item, pred in zip(batch, predictions):
            results.append(
                {
                    "doc_id": item["doc_id"],
                    "entity_text": item["text"],
                    "label": item["label"],
                    "linked_entity": pred,
                }
            )

        print(f"Processed {min(i + batch_size, len(dataset))}/{len(dataset)}")

    # Save results
    with open("linking_results.json", "w", encoding="utf-8") as f:
        import json

        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Linked {len(results)} entities.")


if __name__ == "__main__":
    main()
