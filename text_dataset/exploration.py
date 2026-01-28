import gzip
import os
from io import BytesIO
from pathlib import Path
from typing import List

import polars as pl
import pymupdf
import requests

PATH = "./gotriple_dump/"

results = []
for file in Path(PATH).iterdir():  # Iterating over discipline-specific JSONs
    if (not file.is_file()) or (file.suffix != ".gz"):
        continue
    print(file.name)

    with gzip.open(file, mode="rb") as f:
        data = (
            pl.scan_ndjson(f)
            .select(["id", "in_language"])
            .with_columns(pl.col("in_language").list.first())
            .group_by("in_language")
            .agg(pl.col("id").count().alias("count"))
            .collect(engine="streaming")
            .sort("count")
        )
        print(data)
        results.append(data)

first_df = results.pop(0)
for df in results:
    first_df.join(df, pl.col("in_language"))

print(first_df)
