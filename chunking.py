import gzip
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import pymupdf
import requests
import spacy
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_repr import *

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


DOMAIN_REDIRECTS = {
    "edusj.mosuljournals.com": "edusj.uomosul.edu.iq",
    "radab.mosuljournals.com": "radab.uomosul.edu.iq",
}


def get_urls(df: pl.LazyFrame, sample_size: int | float):
    urls = df.select(pl.col("url")).collect()
    if isinstance(sample_size, int):
        urls = urls.sample(n=sample_size, seed=42).to_series().to_list()
    elif isinstance(sample_size, float):
        urls = urls.sample(fraction=sample_size, seed=42).to_series().to_list()
    print(f"Returning {len(urls)} URLs")
    return urls


def fetch_pdf(url: str, min_chunk_len: int = 20):
    doc = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko)"
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com",
    }

    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        doc = pymupdf.open(stream=BytesIO(response.content), filetype="pdf")
        doc_blocks = []
        full_text = []
        for i, page in enumerate(doc):

            text = page.get_text("text").strip()
            blocks = [
                b[4].strip().replace("\n", " ")
                for b in page.get_text("blocks")
                if b[6] == 0 or len(b[4] < min_chunk_len)
            ]  # type 0 = text block; cutting out coordinates etc.
            if blocks:
                print(f"\nFirst block:\n\n {blocks[0][:100]}")
                doc_blocks.append(blocks)
            if text:
                full_text.append(text.strip())

        # if len(full_text) <= 2 or not any(doc_blocks[:5]):
        #     with open(file="./unavailable_pdfs.txt", mode="a", encoding="utf-8") as f:
        #         f.write(url + "\n")
        #     raise ValueError("Found an invalid PDF")

        print("\n\nSuccessfully retrieved a PDF!")
        print(f"Pages: {len(full_text)}")
        return Document(chunks=doc_blocks, text=" ".join(full_text))

    except Exception as e:
        print(f"Oops...{e}")
        return None

    finally:
        if doc:
            doc.close()


def get_abstracts(df: pl.LazyFrame):
    try:
        abstracts = (
            df.with_columns(
                pl.col("abstract")
                .list.eval(
                    pl.when(pl.element().struct.field("language") == "en").then(
                        pl.element().struct.field("text")
                    )
                )
                .list.first()
            )
            .select(pl.col("abstract"))
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        result = []
        for abs in abstracts:

            blocks = re.split(r"\n\s*\n+", abs)  # splitting by paragraph

            doc_obj = Document(chunks=[blocks], text=abs)
            result.append(doc_obj)
        return result
    except Exception as e:
        print("No abstract either")
        print(e)


def dump_to_text(
    path: str,
    ouput_dir: str,
    frac_or_num: int | float = 1.0,
    exclude: Optional[Tuple[str]] = None,
    write_data: bool = True,
):

    for file in Path(path).iterdir():  # Iterating over discipline-specific JSONs
        domain_denom = file.name.split("_", maxsplit=1)[0]
        if (
            not file.is_file()
            or file.suffix != ".gz"
            or (exclude and domain_denom in exclude)
        ):
            continue

        print("Processing now: ", file.name)

        with gzip.open(file, mode="rb") as f:
            data = (
                pl.scan_ndjson(f)
                .filter(pl.col("url").list.len() > 0)
                .with_columns(
                    pl.col("url").list.first(), pl.col("abstract").list.first()
                )
                .filter(pl.col("url").str.ends_with(".pdf"))
                .filter(pl.col("in_language").list.first() == "en")
            )
            print(
                "Number of data points for current discipline:",
                data.select(pl.len()).collect().item(),
            )

        collected_data = []

        urls = get_urls(df=data, sample_size=frac_or_num)

        for url in urls:
            if any((key for key in DOMAIN_REDIRECTS if key in url)):
                split_url = url.split("/")
                new_domain = DOMAIN_REDIRECTS[split_url[2]]
                new_url = "/".join(split_url[3:])
                url = f"http://{new_domain}/{new_url}"
            document_repr = fetch_pdf(url=url)
            if not document_repr:
                try:
                    abstract_raw = (
                        data.filter(pl.col("url").str.contains(url))
                        .select(pl.col("abstract").struct.field("text"))
                        .collect(engine="streaming")
                        .row(0)
                    )
                    abstract = abstract_raw[0].strip()
                    if len(abstract) < 50:
                        raise ValueError("Abstract discarded due to length")
                    document_repr = Document(
                        chunks=[re.split(r"\n\s*\n+", abstract)], text=abstract
                    )  # chunks have to be matrices, representing pages. Abstracts only occupy one page, representing pages. Abstracts only occupy one page

                    if not abstract or len(abstract) < 20:
                        continue
                    print(abstract)
                except Exception as e:
                    print(f"Error occurred while extracting abstract: {e}")
                    continue

            doc_data = {
                "url": url,
                "chunks": document_repr.chunks,
                "text": document_repr.text,
            }
            collected_data.append(doc_data)

        result = pl.DataFrame(collected_data)
        print(result)

        if write_data:
            file_name = f"./{file.name.split("_")[0]}_dataset.json"
            path = os.path.join(ouput_dir, file_name)
            result.lazy().sink_ndjson(path)

        return result


def dump_to_abstract(
    path: str,
    ouput_dir: str,
    frac_or_num: int | float = 1.0,
    exclude: Optional[Tuple[str]] = None,
    write_data: bool = True,
):
    for file in Path(path).iterdir():  # Iterating over discipline-specific JSONs
        domain_denom = file.name.split("_", maxsplit=1)[0]
        if (
            not file.is_file()
            or file.suffix != ".gz"
            or (exclude and domain_denom in exclude)
        ):
            continue

        print("Processing now: ", file.name)

        with gzip.open(file, mode="rb") as f:
            data = (
                pl.scan_ndjson(f)
                .with_columns(
                    pl.col("url").list.first(), pl.col("abstract").list.first()
                )
                .filter(pl.col("in_language").list.first() == "en")
            )
            print(
                "Number of data points for current discipline:",
                data.select(pl.len()).collect().item(),
            )

        collected_data = []
        abstracts = get_abstracts(data)
        if not abstracts:
            raise ValueError("Error extracting abstracts")
        for document_repr in abstracts:
            doc_data = {
                "chunks": document_repr.chunks,
                "text": document_repr.text,
            }
            collected_data.append(doc_data)

        result = pl.DataFrame(collected_data)
        print(result)

        if write_data:
            file_name = f"./{file.name.split("_")[0]}_dataset.json"
            path = os.path.join(ouput_dir, file_name)
            result.lazy().sink_ndjson(path)

        return result


if __name__ == "__main__":
    DUMP_PATH = "./gotriple_dump/"
    OUTPUT_DIR = "./extracted_data/"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    dump_to_text(path=DUMP_PATH, ouput_dir=OUTPUT_DIR, frac_or_num=10)
