import json
from pathlib import Path

import pandas as pd

DATA_DIR = "./text_dataset/extracted_2/"
LINKED_DATA = "./linking/mgenre_linking_results_base-llm.json"
linked = pd.read_json(LINKED_DATA)
log = "./linking/log.json"
open(log, mode="w").close()

results = []
for f in Path(DATA_DIR).iterdir():
    domain = f.name.removesuffix(".json")
    print(domain)
    df = pd.read_json(f, orient="records")[["id", "text"]]
    relevant_linked = linked[linked["doc_id"].str.contains(domain)]
    merged = df.merge(relevant_linked, left_on="id", right_on="doc_id")
    chunk_ids = []
    for _, row in merged.iterrows():
        idx = None
        chunks = row["text"]
        ent = row["entity_text"]
        filtered_chunks = [(i, c) for i, c in enumerate(chunks) if ent in c]
        if len(filtered_chunks) > 0:
            idx = filtered_chunks[0][0]
        else:
            with open(log, mode="a") as f:
                f.write(json.dumps({"ent": ent, "chunks": chunks}))

        chunk_ids.append(idx)

    merged["chunk_id"] = chunk_ids
    print(len(merged))
    final = merged.dropna(subset="chunk_id").drop(columns=["id", "text"])
    print(len(final))
    final["chunk_id"] = final["chunk_id"].astype(int)
    results.append(final)

results_df = pd.concat(results)
print(results_df)
results_df.to_json("./linking/mgenre_base-llm.json", orient="records", indent=2)
