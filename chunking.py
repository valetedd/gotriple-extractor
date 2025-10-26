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


def get_urls(df: pl.LazyFrame, n_sample: int, frac: float = 1.0):
    urls = (
        df.select(pl.col("url")).collect().sample(n=100, seed=42).to_series().to_list()
    )
    print(f"Returning {len(urls)} URLs")
    return urls


def fetch_pdf(
    url: str,
):
    doc = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
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

            blocks = page.get_text("blocks")
            text = page.get_text("text")  # Try different extraction method
            print(f"Page {i}: {len(blocks)} blocks, {len(text)} chars")
            if blocks:
                print(f"  First block: {blocks[0]}")
                doc_blocks.append(blocks)
            if text:
                full_text.append(text)

        if len(full_text) <= 2 or not any(doc_blocks[:5]):
            with open(file="./unavailable_pdfs.txt", mode="a", encoding="utf-8") as f:
                f.write(url + "\n")
            raise ValueError("Found an invalid PDF")

        print("\n\nRetrieved a PDF:")
        print(f"Pages: {len(full_text)}")
        print(f"\nSample:\n{full_text[0][:100]}")
        return PDFDocument(blocks=doc_blocks, text=" ".join(full_text))

    except Exception as e:
        print(f"Oops...{e}")
        return []

    finally:
        if doc:
            doc.close()


def get_abstracts(df: pl.LazyFrame):
    try:
        abstracts = (
            df.select(pl.col("abstract").struct.field("text"))
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        for abs in abstracts:

            blocks = re.split(r"\n\s*\n+")  # splitting by paragraph

            pdf_doc = PDFDocument(blocks=blocks, text=abs)
        return abstracts
    except Exception as e:
        print("No abstract either")
        print(e)


def abstract_fallback(df: pl.LazyFrame, url: str):
    try:
        abstract = (
            df.filter(pl.col("url").str.contains(url))
            .select(pl.col("abstract").struct.field("text"))
            .collect(engine="streaming")
            .row(0)
        )
        print(abstract)
        return abstract
    except Exception as e:
        print("No abstract either")
        print(e)


def process_dump(
    path: str,
    exclude: Optional[Tuple[str]] = None,
    frac: float = 1.0,
    abstract_only: bool = False,
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
            )
            print(
                "Number of data points for current discipline:",
                data.select(pl.len()).collect().item(),
            )

        if abstract_only:
            abstracts = get_abstracts(data)

        urls = get_urls(df=data, n_sample=100, frac=frac)

        for url in urls:
            if any((key for key in DOMAIN_REDIRECTS if key in url)):
                split_url = url.split("/")
                new_domain = DOMAIN_REDIRECTS[split_url[2]]
                new_url = "/".join(split_url[3:])
                url = f"http://{new_domain}/{new_url}"
            document_repr = fetch_pdf(url=url)
            if not document_repr:
                abstract = abstract_fallback(data, url)
                if not abstract or len(abstract) < 20:
                    continue
                print(abstract)


if __name__ == "__main__":

    if not os.path.exists("./available_pdfs.txt"):
        os.makedirs("./available_pdfs.txt")

    DUMP_PATH = "./gotriple_dump/"
    process_dump(path=DUMP_PATH)
