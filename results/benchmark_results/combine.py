"""
Basic script to combine the benchmarking results if needed
"""

import json
from typing import Dict, Set


def clean_ents(obj: Dict, allowed_labels: Set):
    obj["preds_llm"] = [
        annotation
        for annotation in obj["preds_llm"]
        if annotation["label"] in allowed_labels
    ]
    return obj


def main():
    gotriple_ents = {
        "Person",
        "Organization",
        "Location",
        "Cultural Object",
        "Event",
        "Software",
        "Dataset",
        "Time",
    }

    with open("./gliner_results.jsonl", mode="r") as f:
        gliner_res = [json.loads(obj) for obj in f]

    with open("./llm_results.jsonl", mode="r") as ff:
        llm_res = [json.loads(obj) for obj in ff]

    open("./processed_combined.jsonl", mode="w").close()
    open("./combined_results.jsonl", mode="w").close()
    for line_gli, line_llm in zip(gliner_res, llm_res):
        with open("./combined_results.jsonl", mode="a") as final:
            final.write(json.dumps(line_gli) + "\n")
            final.write(json.dumps(line_llm) + "\n")
        processed_line_llm = clean_ents(line_llm, gotriple_ents)
        with open("./processed_combined.jsonl", mode="a") as final_processed:

            final_processed.write(json.dumps(line_gli) + "\n")
            final_processed.write(json.dumps(processed_line_llm) + "\n")


if __name__ == "__main__":
    main()
