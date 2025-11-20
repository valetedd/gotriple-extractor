import gc
import gzip
import json
import os
import string
import threading
import unicodedata
from collections import Counter, defaultdict

import regex

stop_event = threading.Event()
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import polars as pl
import pymupdf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount(
    "https://", HTTPAdapter(max_retries=retries, pool_maxsize=50, pool_connections=50)
)


DOMAIN_REDIRECTS = {
    "edusj.mosuljournals.com": "edusj.uomosul.edu.iq",
    "radab.mosuljournals.com": "radab.uomosul.edu.iq",
    "alaw.mosuljournals.com": "alaw.uomosul.edu.iq",
}

domain_fails = defaultdict(int)

CONCURRENT = False
MIN_BLOCK_LEN = 50  # Filters single PDF pages based on the length of its blocks
MIN_PAGES = 1  # Allows filtering out short PDFs
MIN_WORDS = 20 # Filter to exclude PDFs with little words
MIN_SAMPLE_SIZE = 1  # Ensuring a lower bound for discipline-language pairs representation in the final dataset
FAILURES_PATIENCE = 3  # How many times a domain can fail before getting blacklisted from the dataset building process
REQUIRES_FULL_TEXT = True
VALID_SCRIPTS = {
    "Latin",
    "Cyrillic",
    "Arabic",
    "Hebrew",
    "Han",
    "Hiragana",
    "Katakana",
    "Hangul",
    "Devanagari",
    "Thai",
    "Greek",
    "Bengali",
    "Ethiopic",
    "Myanmar",
    "Georgian",
    "Khmer",
    "Lao",
    "Tamil",
    "Telugu",
    "Gujarati",
    "Gurmukhi",
    "Malayalam",
    "Sinhala",
}

# Precompile script regexes for speed
SCRIPT_REGEXES = {
    script: regex.compile(rf"\p{{Script={script}}}") for script in VALID_SCRIPTS
}


def get_ratios(stats_path: str):
    with open(stats_path, mode="r") as f:
        data = json.load(f)

    ratios = {}
    total_docs_n = sum((l["doc_count"] for l in data["buckets"]))

    for lang_data in data["buckets"]:
        lang = lang_data["key"]
        count_per_lang = lang_data["doc_count"]
        lang_percentage = count_per_lang / total_docs_n
        ratios[lang] = {
            "ratio": round(lang_percentage, ndigits=4),
            "disciplines%": {},
        }
        for disc_data in lang_data["by_discipline"]["buckets"]:
            disc_code = disc_data["key"]
            ratio = disc_data["doc_count"] / count_per_lang
            ratios[lang]["disciplines%"][disc_code] = round(ratio, ndigits=4)

    return ratios


def is_good_pdf(chunks):

    def detect_script(char):
        if char.isdigit() or char.isspace():
            return None
        if char in string.punctuation:
            return None

        for script, rx in SCRIPT_REGEXES.items():
            if rx.fullmatch(char):
                return script
        return "Other"

    txt = " ".join(chunks).strip()
    if not txt:
        return False

    len_txt = len(txt)

    # Checking for alphanumeric symbols
    letters = [c for c in txt if c.isalpha()]
    if len(letters) < 30:
        return False
    if len(letters) / len_txt < 0.05:
        return False

    # Detecting number of scripts
    scripts = [detect_script(c) for c in letters if detect_script(c) is not None]
    if not scripts:
        return False

    scount = Counter(scripts)
    total = sum(scount.values())

    if scount.get("Other", 0) / total > 0.85:
        return False

    real_scripts = {s for s in scount if s != "Other"}
    if len(real_scripts) > 4:
        return False

    # Using unicodedata to detect symbols and box-drawing chars
    def is_symbol_like(char):
        # category starting with 'S' indicates a Symbol (Math, Currency, Modifier etc.)
        cat = unicodedata.category(char)
        if cat and cat.startswith("S"):
            return True
        # detect box-drawing / block-drawing characters by name heuristic
        name = unicodedata.name(char, "")
        if "BOX DRAW" in name or "BOX-DRAW" in name or "BOX_DRAW" in name:
            return True
        return False

    symbol_like = sum(1 for c in txt if is_symbol_like(c))
    if symbol_like / len_txt > 0.25:
        return False

    # Minimum number of words per PDF
    words = regex.findall(r"\p{Letter}+", txt)
    if len(words) < 10:
        return False


    return True


def get_text(resp: BytesIO):
    doc_blocks = []
    with pymupdf.open(stream=resp, filetype="pdf") as doc:
        if doc.page_count <= MIN_PAGES:
            raise ValueError("PDF resource discarded due to limited length")
        for page in doc:
            page_blocks = page.get_text("blocks", flags=pymupdf.TEXT_INHIBIT_SPACES)
            # 6th position in block tuples represents the content type; 0 == text
            blocks = [
                txt_content
                for b in page_blocks
                if len(txt_content := b[4].strip()) >= MIN_BLOCK_LEN
                and all(txt_content)
                and b[6] == 0
            ]
            if not blocks:
                break
            doc_blocks.extend(blocks)
    if not is_good_pdf(chunks=doc_blocks):
        print(f"The following was not deemed a good PDF:\n{doc_blocks}")
        raise ValueError("PDF does not meet the criteria")
    return doc_blocks


def write_to_buf(resp: requests.Response):
    buf = BytesIO()
    for chunk in resp.iter_content(chunk_size=50000):
        if chunk:
            buf.write(chunk)
    buf.seek(0)
    return buf


def is_pdf_file(buf: BytesIO) -> bool:
    pos = buf.tell()
    buf.seek(0)
    header = buf.read(1024)  # read first 1KB
    buf.seek(pos)
    return b"%PDF-" in header[:1024]


def fetch_pdf(url: str):

    if stop_event.is_set():
        return [""], False

    attempted = False
    domain = url.split("/")[2]

    while True:

        try:
            if not attempted:
                with requests.get(
                    url, stream=True, allow_redirects=True, timeout=(10, 20)
                ) as response:
                    response.raise_for_status()
                    buf = write_to_buf(response)

            else:

                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko)"
                        "Chrome/118.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive",
                    "Referer": "https://www.google.com",
                }

                with session.get(
                    url,
                    headers=headers,
                    timeout=(10, 20),
                    stream=True,
                    allow_redirects=True,
                ) as response:

                    response.raise_for_status()
                    # read into BytesIO in chunks to avoid creating huge intermediate bytes objects
                    buf = write_to_buf(response)

            if not is_pdf_file(buf):
                raise TypeError(
                    f"Skipping resource because it doesn't seem to be a PDF file"
                )

            doc_blocks = get_text(buf)

            # free the buffer explicitly after use (optional)
            buf.close()
            domain_fails[domain] -= 1
            return doc_blocks, True

        except (ValueError, TypeError) as e:
            print(f"{e}: Moving on with other URL")
            break
        except Exception as e:
            print(f"Error: {e}")
            if attempted:
                print("Failed to retrieve PDF via current URL. Moving on with other URL")

                domain_fails[domain] += 1
                print(f"{domain} has failed {domain_fails[domain]} times!")
                break
            else:
                attempted = True
                print("Retrying...")
                continue

    return [""], False


def process_until_target(urls: Iterable, target: int) -> pl.LazyFrame | None:
    results = []
    success_count = 0
    try:
        for url in urls:
            split_url = url.split("/")
            if (domain := split_url[2]) in DOMAIN_REDIRECTS:
                split_url[2] = DOMAIN_REDIRECTS[
                    split_url[2]
                ]  # The 3d element of the split is the domain: http://example.foo/
                url = "/".join(split_url)

            if domain_fails[domain] >= FAILURES_PATIENCE:
                continue
            try:
                text_blocks, success = fetch_pdf(url)
                if success:
                    success_count += 1
                    print("appending text...")
                    results.append({"url": url, "text": text_blocks})
                if success_count == target:
                    break
            except Exception as e:
                print(f"Failed {url}: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        raise

    if not results:
        return None

    return pl.LazyFrame(results).cast({"text": pl.List(pl.String)})


def process_until_target_conc(urls: Iterable, target: int):
    results = []
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_pdf, url): url for url in urls}
        try:

            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                url = futures[future]
                try:
                    text, success = future.result()
                    if success:
                        success_count += 1
                        results.append({"url": url, "text": text})
                        if success_count >= target:
                            print("Target reached!")
                            stop_event.set()
                            break
                except Exception as e:
                    print(f"Failed {url}: {e}")

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down...")
            stop_event.set()
            raise

        finally:
            executor.shutdown(cancel_futures=True)
    return pl.LazyFrame(results)


def urls_generator(df: pl.LazyFrame):
    for batch in df.select("url").collect(engine="streaming").iter_rows():
        yield batch[0]


def process_discipline(
    df: pl.LazyFrame,
    stats_path: str,
    disc: str,
    languages: Literal["all"] | List[str] = "all",
    ds_size: int = 1000,
):

    result = []
    if languages == "all":
        languages = [
            "en",
            "fr",
            "de",
            "es",
            "pt",
            "pl",
            "ru",
            "it",
            "hr",
            "tr",
            "uk",
            "sv",
            "el",
            "ar",
            "nl",
            "sl",
            "fi",
            "ca",
            "sr",
            "no",
            "hu",
            "da",
            "he",
            "sq",
            "undefined",
            "other",
        ]
    ratios = get_ratios(stats_path=stats_path)

    for lang in languages:
        # TODO:Correct sampling by implementing proportionate reallocation of the remainder
        lang_sample_n = round(ds_size * ratios[lang]["ratio"])
        if len(result) == lang_sample_n:
            break
        disc_ratio = ratios[lang]["disciplines%"][disc]
        disc_sample_n = max(round(lang_sample_n * disc_ratio), MIN_SAMPLE_SIZE)
        print(
            f"Fetching {disc_sample_n} PDFs for discipline {disc} in language with code '{lang}'"
        )
        df_by_lang = df.filter(pl.col("in_language") == lang)
        urls = urls_generator(df_by_lang)
        if CONCURRENT:
            blocks_df = process_until_target_conc(
                urls,
                target=disc_sample_n,
            )
        else:
            blocks_df = process_until_target(
                urls,
                target=disc_sample_n,
            )
        if blocks_df is None:
            continue
        lang_df = df_by_lang.join(blocks_df, how="inner", on="url")
        del df_by_lang
        result.append(lang_df)
        del lang_df
        gc.collect()

    if not result:
        return None

    return pl.concat(result)


def main(
    path: str,
    output_dir: str,
    stats_json: str,
    disc_blacklist: Optional[tuple[str]] = None,
    dataset_size: int = 1000,
):
    os.makedirs(output_dir, exist_ok=True)

    for file in Path(path).iterdir():  # Iterating over discipline-specific JSONs
        discipline = file.name.split("_", maxsplit=1)[0]
        if (
            not file.is_file()
            or file.suffix != ".gz"
            or (disc_blacklist and discipline in disc_blacklist)
        ):
            continue

        print("Processing now: ", file.name)

        with gzip.open(file, mode="rb") as f:

            # Preprocessing using the native polars API + a user defined lambda
            df = (
                pl.scan_ndjson(f)
                .filter(
                    (
                        pl.col("abstract").list.len() > 0
                    )  # ensuring each datapoint has an abstract
                    & (pl.col("headline").list.len() > 0)  # same for titles
                    & (
                        pl.col("in_language").list.len() == 1
                    )  # we only want data with unambiguous language attribution
                )
                # Flatten in_language list into code string
                .with_columns(in_language=pl.col("in_language").list.first())
                # Process abstracts and headlines using map_elements (row-level)
                # This ensures that the attributed language matches the abstract and the headline
                .with_columns(
                    abstract_text=pl.struct(["abstract", "in_language"]).map_elements(
                        lambda s: next(
                            (
                                x["text"]
                                for x in s["abstract"]
                                if x["lang"] == s["in_language"]
                            ),
                            None,
                        ),
                        return_dtype=pl.Utf8,
                    ),
                    headline_text=pl.struct(["headline", "in_language"]).map_elements(
                        lambda s: next(
                            (
                                x["text"]
                                for x in s["headline"]
                                if x["lang"] == s["in_language"]
                            ),
                            None,
                        ),
                        return_dtype=pl.Utf8,
                    ),
                )
                .select(
                    pl.col("url").list.first(),
                    "in_language",
                    "abstract_text",
                    "headline_text",
                )
            )

        processed = process_discipline(
            df,
            stats_path=stats_json,
            disc=discipline,
            languages="all",
            ds_size=dataset_size,
        )
        del df
        if processed is not None:
            print(f"WRITING JSON FOR {discipline}")
            processed.sink_ndjson(f"{output_dir}/{discipline}.json")
        del processed
        gc.collect()


if __name__ == "__main__":
    # ratios = get_ratios("./langauagesAndDisciplines.json")
    # print(ratios.keys())
    # processed_files = [f.name.split(".")[0] for f in Path( "./extracted_data/" ).iterdir()]
    #
    # main(
    #     path="./gotriple_dump/",
    #     output_dir="./extracted_data/",
    #     stats_json="./langauagesAndDisciplines.json",
    #     dataset_size=1000,
    #     disc_blacklist=tuple(processed_files)
    # )
    df = pl.scan_ndjson("./extracted_data/")
    print( df.select(pl.len()).collect().item() )
    print("Dropping nulls:", df.drop_nulls().select(pl.len()).collect().item())
    print(df.head(5).collect())
