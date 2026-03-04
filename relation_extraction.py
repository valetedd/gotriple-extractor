#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import spacy
import torch
from glirel import GLiREL

DEFAULT_LANGUAGES = ("en", "it", "fr", "de")
DEFAULT_RELATION_CONSTRAINTS = {
    "authored by": {
        "allowed_head": ["Creative Work", "Software", "Publication", "Dataset"],
        "allowed_tail": ["Person", "Project", "Organization"],
    },
    "related to": {"allowed_head": ["Person"], "allowed_tail": ["Person"]},
    "participant in": {
        "allowed_head": ["Person", "Organization"],
        "allowed_tail": ["Event"],
    },
    "location": {
        "allowed_head": ["Organization", "Event", "Place", "Point of Interest"],
        "allowed_tail": ["Place"],
    },
    "occurred during": {
        "allowed_head": ["Event"],
        "allowed_tail": ["Time"],
    },
    "published by": {
        "allowed_head": ["Publication"],
        "allowed_tail": ["Organization", "Person", "Project"],
    },
}

ENTITY_LABEL_ALIASES = {
    "publication": "Publication",
    "publication title": "Publication",
    "publication_title": "Publication",
    "point of interest": "Point of Interest",
    "point_of_interest": "Point of Interest",
}


@dataclass
class ChunkCandidate:
    source_file: str
    discipline: str
    doc_id: str
    lang: str
    chunk_idx: int
    text: str
    entities_char: list[dict[str, Any]]


def _choice_key_index(key: str) -> int:
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else -1


def _select_choice_entities(value: Any, choice_index: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    if not value:
        return []

    # Already a plain list of entities
    if all(isinstance(x, dict) for x in value):
        return value

    # List of choices -> each choice is a list[entity]
    if 0 <= choice_index < len(value) and isinstance(value[choice_index], list):
        return [x for x in value[choice_index] if isinstance(x, dict)]

    for item in value:
        if isinstance(item, list):
            return [x for x in item if isinstance(x, dict)]
    return []


def extract_chunk_entities(
    ents: Any, chunk_idx: int, choice_index: int
) -> list[dict[str, Any]]:
    if ents is None:
        return []

    if isinstance(ents, list):
        # Format: [[chunk_idx, [ent, ...]], ...] (annotated_test-full)
        if 0 <= chunk_idx < len(ents):
            chunk_item = ents[chunk_idx]
            if (
                isinstance(chunk_item, list)
                and len(chunk_item) == 2
                and isinstance(chunk_item[0], int)
                and isinstance(chunk_item[1], list)
            ):
                return [x for x in chunk_item[1] if isinstance(x, dict)]
            if isinstance(chunk_item, list) and all(
                isinstance(x, dict) for x in chunk_item
            ):
                return [x for x in chunk_item if isinstance(x, dict)]

        # Fallback search by explicit chunk id
        for item in ents:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], int)
                and item[0] == chunk_idx
                and isinstance(item[1], list)
            ):
                return [x for x in item[1] if isinstance(x, dict)]

        if 0 <= chunk_idx < len(ents) and isinstance(ents[chunk_idx], list):
            return [x for x in ents[chunk_idx] if isinstance(x, dict)]
        return []

    if not isinstance(ents, dict):
        return []

    chunk_key = f"chunk_{chunk_idx}"
    if chunk_key in ents:
        return _select_choice_entities(ents.get(chunk_key), choice_index)

    preferred_choice_key = f"choice_{choice_index}"
    if preferred_choice_key in ents and isinstance(ents[preferred_choice_key], list):
        chunks = ents[preferred_choice_key]
        if 0 <= chunk_idx < len(chunks) and isinstance(chunks[chunk_idx], list):
            return [x for x in chunks[chunk_idx] if isinstance(x, dict)]

    choice_keys = sorted(
        (k for k in ents.keys() if k.startswith("choice_")),
        key=_choice_key_index,
    )
    for key in choice_keys:
        chunks = ents.get(key)
        if isinstance(chunks, list) and 0 <= chunk_idx < len(chunks):
            if isinstance(chunks[chunk_idx], list):
                return [x for x in chunks[chunk_idx] if isinstance(x, dict)]

    return []


def normalize_char_entities(
    raw_entities: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    text_len = len(text)
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        try:
            start = int(ent.get("start", ent.get("start_char")))
            end = int(ent.get("end", ent.get("end_char")))
        except Exception:
            continue
        if start < 0 or end < 0:
            continue
        if start >= end:
            continue

        start = min(start, text_len)
        end = min(end, text_len)
        if start >= end:
            continue

        label = str(ent.get("label", "")).strip()
        if not label:
            continue
        ent_text = str(ent.get("text", "")).strip()
        confidence = ent.get("confidence", ent.get("score"))

        out.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": ent_text,
                "confidence": confidence,
            }
        )
    return out


def canonical_entity_label(label: str) -> str:
    text = str(label).strip().replace("_", " ")
    low = re.sub(r"\s+", " ", text).strip().lower()
    if low in ENTITY_LABEL_ALIASES:
        return ENTITY_LABEL_ALIASES[low]
    return text


def validate_relation_constraints(schema: Any) -> dict[str, dict[str, list[str]]]:
    if not isinstance(schema, dict) or not schema:
        raise ValueError("Relation constraint schema must be a non-empty dict.")

    normalized: dict[str, dict[str, list[str]]] = {}
    for rel_label, rules in schema.items():
        if not isinstance(rel_label, str) or not rel_label.strip():
            raise ValueError(f"Invalid relation label key: {rel_label!r}")
        if not isinstance(rules, dict):
            raise ValueError(f"Rules for relation `{rel_label}` must be a dict.")

        for field in ("allowed_head", "allowed_tail"):
            if field not in rules:
                raise ValueError(f"Missing `{field}` for relation `{rel_label}`.")
            values = rules[field]
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"`{field}` for relation `{rel_label}` must be a non-empty list of strings."
                )
            if not all(isinstance(x, str) and x.strip() for x in values):
                raise ValueError(
                    f"`{field}` for relation `{rel_label}` must contain only non-empty strings."
                )

        normalized[rel_label.strip()] = {
            "allowed_head": sorted(
                {canonical_entity_label(x) for x in rules["allowed_head"]}
            ),
            "allowed_tail": sorted(
                {canonical_entity_label(x) for x in rules["allowed_tail"]}
            ),
        }

    return normalized


def collect_candidates(
    predictions_dir: Path,
    source_dir: Path,
    languages: set[str],
    choice_index: int,
    min_entities: int,
    model_filter: str | None,
) -> list[ChunkCandidate]:
    candidates: list[ChunkCandidate] = []

    pred_files = sorted(
        p for p in predictions_dir.glob("*.json") if "time" not in p.name.lower()
    )
    for pred_path in pred_files:
        source_path = source_dir / pred_path.name
        if not source_path.exists():
            print(f"[WARN] Missing source file for {pred_path.name}: {source_path}")
            continue

        try:
            pred_docs = json.loads(pred_path.read_text(encoding="utf-8"))
            src_docs = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Skipping {pred_path.name}: failed to parse JSON ({exc})")
            continue

        if not isinstance(pred_docs, list) or not isinstance(src_docs, list):
            print(f"[WARN] Skipping {pred_path.name}: expected list JSON.")
            continue

        src_by_id: dict[str, dict[str, Any]] = {}
        for d in src_docs:
            if isinstance(d, dict) and isinstance(d.get("id"), str):
                src_by_id[d["id"]] = d

        discipline = pred_path.stem
        for doc in pred_docs:
            if not isinstance(doc, dict):
                continue
            if model_filter and str(doc.get("model", "")).strip() != model_filter:
                continue
            lang = str(doc.get("lang", "")).strip()
            if lang not in languages:
                continue

            doc_id = doc.get("doc_id")
            if not isinstance(doc_id, str):
                continue
            src_doc = src_by_id.get(doc_id)
            if not src_doc:
                continue

            text_chunks = src_doc.get("text")
            if not isinstance(text_chunks, list):
                continue

            ents = doc.get("ents")
            for chunk_idx, chunk_text in enumerate(text_chunks):
                if not isinstance(chunk_text, str):
                    continue
                raw_chunk_entities = extract_chunk_entities(
                    ents, chunk_idx, choice_index
                )
                char_entities = normalize_char_entities(raw_chunk_entities, chunk_text)
                if len(char_entities) < min_entities:
                    continue
                candidates.append(
                    ChunkCandidate(
                        source_file=pred_path.name,
                        discipline=discipline,
                        doc_id=doc_id,
                        lang=lang,
                        chunk_idx=chunk_idx,
                        text=chunk_text,
                        entities_char=char_entities,
                    )
                )
    return candidates


def sample_balanced(
    candidates: list[ChunkCandidate],
    total_samples: int,
    languages: list[str],
    rng: random.Random,
) -> tuple[list[ChunkCandidate], dict[str, int]]:
    by_lang: dict[str, list[ChunkCandidate]] = {lang: [] for lang in languages}
    for cand in candidates:
        if cand.lang in by_lang:
            by_lang[cand.lang].append(cand)

    for lang in languages:
        rng.shuffle(by_lang[lang])

    total_available = sum(len(v) for v in by_lang.values())
    if total_available < total_samples:
        raise ValueError(
            f"Requested {total_samples} samples but only {total_available} available across {languages}."
        )

    quotas = {
        lang: min(total_samples // len(languages), len(by_lang[lang]))
        for lang in languages
    }
    remaining = total_samples - sum(quotas.values())

    while remaining > 0:
        progressed = False
        langs_by_capacity = sorted(
            languages,
            key=lambda lang: (len(by_lang[lang]) - quotas[lang]),
            reverse=True,
        )
        for lang in langs_by_capacity:
            if remaining == 0:
                break
            if quotas[lang] < len(by_lang[lang]):
                quotas[lang] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    if remaining > 0:
        raise ValueError("Could not allocate balanced sample quotas.")

    sampled: list[ChunkCandidate] = []
    for lang in languages:
        sampled.extend(by_lang[lang][: quotas[lang]])
    rng.shuffle(sampled)
    return sampled, quotas


def map_char_span_to_token_span(
    start_char: int,
    end_char: int,
    offsets: list[tuple[int, int]],
) -> tuple[int, int] | None:
    start_token: int | None = None
    end_token: int | None = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        if tok_start < end_char and tok_end > start_char:
            if start_token is None:
                start_token = idx
            end_token = idx
    if start_token is None or end_token is None:
        return None
    return start_token, end_token


def tokenize_glirel_regex(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\w+(?:[-_]\w+)*|\S", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def tokenize_whitespace(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def convert_entities_to_token_spans(
    text: str,
    char_entities: list[dict[str, Any]],
    tokenization: str,
    nlp_tokenizer: Any | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[list[Any]]]:
    if tokenization == "spacy":
        if nlp_tokenizer is None:
            raise ValueError("spaCy tokenizer is required when tokenization='spacy'.")
        doc = nlp_tokenizer(text)
        tokens: list[str] = [tok.text for tok in doc]
        offsets: list[tuple[int, int]] = [
            (tok.idx, tok.idx + len(tok.text)) for tok in doc
        ]
    elif tokenization == "glirel_regex":
        tokens, offsets = tokenize_glirel_regex(text)
    elif tokenization == "whitespace":
        tokens, offsets = tokenize_whitespace(text)
    else:
        raise ValueError(f"Unsupported tokenization mode: {tokenization}")

    token_entities: list[dict[str, Any]] = []
    model_ner: list[list[Any]] = []
    seen = set()

    for ent in char_entities:
        ent_text = str(ent.get("text", "")).strip()
        mapped = map_char_span_to_token_span(ent["start"], ent["end"], offsets)

        start_token: int | None = None
        end_token: int | None = None
        mapped_text = ""
        sim = 0.0
        alignment_method = "char_offset"

        hint_token_start: int | None = None
        if mapped is not None:
            hint_token_start = mapped[0]
            start_token, end_token = mapped
            mapped_text = _tokens_to_surface(tokens[start_token : end_token + 1])
            sim = _similarity(mapped_text, ent_text)

        needs_realign = (mapped is None) or (sim < 0.75)
        if needs_realign and ent_text:
            best = _best_token_ngram_match(
                tokens=tokens,
                target_text=ent_text,
                hint_token_start=hint_token_start,
            )
            if best is not None and best[2] >= 0.70:
                start_token, end_token, sim = best
                mapped_text = _tokens_to_surface(tokens[start_token : end_token + 1])
                alignment_method = "text_fuzzy"

        if start_token is None or end_token is None:
            continue

        key = (start_token, end_token, ent["label"])
        if key in seen:
            continue
        seen.add(key)

        token_slice = tokens[start_token : end_token + 1]
        decoded = _tokens_to_surface(token_slice)
        if not decoded:
            decoded = ent_text

        if 0 <= start_token < len(offsets) and 0 <= end_token < len(offsets):
            char_start = offsets[start_token][0]
            char_end = offsets[end_token][1]
        else:
            char_start = ent["start"]
            char_end = ent["end"]

        token_entity = {
            "start_token": start_token,
            "end_token": end_token,
            "label": ent["label"],
            "text": ent_text,
            "text_tokenized": decoded,
            "char_start": char_start,
            "char_end": char_end,
            "confidence": ent.get("confidence"),
            "alignment_method": alignment_method,
            "alignment_score": round(float(sim), 4),
        }
        token_entities.append(token_entity)
        model_ner.append([start_token, end_token, ent["label"], decoded])

    return tokens, token_entities, model_ner


def _normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^\w]+", "", str(text).lower(), flags=re.UNICODE)


def _similarity(a: str, b: str) -> float:
    na = _normalize_for_match(a)
    nb = _normalize_for_match(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _tokens_to_surface(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    return text.strip()


def _best_token_ngram_match(
    tokens: list[str],
    target_text: str,
    hint_token_start: int | None = None,
) -> tuple[int, int, float] | None:
    if not tokens:
        return None
    target_norm = _normalize_for_match(target_text)
    if not target_norm:
        return None

    target_words = re.findall(r"\w+", target_text, flags=re.UNICODE)
    if target_words:
        min_len = max(1, len(target_words) - 2)
        max_len = len(target_words) + 4
    else:
        min_len = 1
        max_len = 6

    n_tokens = len(tokens)
    max_len = min(max_len, n_tokens)
    best: tuple[int, int, float] | None = None
    best_score = -1.0

    for span_len in range(min_len, max_len + 1):
        for i in range(0, n_tokens - span_len + 1):
            j = i + span_len - 1
            cand = _tokens_to_surface(tokens[i : j + 1])
            cand_norm = _normalize_for_match(cand)
            if not cand_norm:
                continue
            base = SequenceMatcher(None, target_norm, cand_norm).ratio()

            proximity_bonus = 0.0
            if hint_token_start is not None:
                dist = abs(i - hint_token_start)
                proximity_bonus = 0.05 * (1.0 - min(dist / max(n_tokens, 1), 1.0))

            score = base + proximity_bonus
            if score > best_score:
                best_score = score
                best = (i, j, base)

    return best


def build_entity_label_indexes(
    token_entities: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int], set[str]], dict[str, set[str]]]:
    span_index: dict[tuple[int, int], set[str]] = defaultdict(set)
    text_index: dict[str, set[str]] = defaultdict(set)

    for ent in token_entities:
        try:
            start = int(ent["start_token"])
            end = int(ent["end_token"])
        except Exception:
            continue
        label = canonical_entity_label(str(ent.get("label", "")).strip())
        if not label:
            continue
        span_index[(start, end)].add(label)

        tokenized_surface = _normalize_surface(ent.get("text_tokenized", ""))
        if tokenized_surface:
            text_index[tokenized_surface].add(label)
        original_surface = _normalize_surface(ent.get("text", ""))
        if original_surface:
            text_index[original_surface].add(label)

    return span_index, text_index


def relation_allowed_by_constraints(
    rel: dict[str, Any],
    span_labels: dict[tuple[int, int], set[str]],
    text_labels: dict[str, set[str]],
    constraints: dict[str, dict[str, list[str]]],
) -> bool:
    rel_label = str(rel.get("label", "")).strip()
    rules = constraints.get(rel_label)
    if not rules:
        return True

    head_pos = rel.get("head_pos")
    tail_pos = rel.get("tail_pos")
    if not (isinstance(head_pos, list) and len(head_pos) == 2):
        return False
    if not (isinstance(tail_pos, list) and len(tail_pos) == 2):
        return False

    try:
        head_span = (
            int(head_pos[0]),
            int(head_pos[1]) - 1,
        )  # GLiREL returns end as exclusive
        tail_span = (int(tail_pos[0]), int(tail_pos[1]) - 1)
    except Exception:
        return False

    head_types = span_labels.get(head_span, set())
    tail_types = span_labels.get(tail_span, set())
    if not head_types:
        head_types = text_labels.get(
            _normalize_surface(rel.get("head_text", "")), set()
        )
    if not tail_types:
        tail_types = text_labels.get(
            _normalize_surface(rel.get("tail_text", "")), set()
        )

    allowed_head = set(rules["allowed_head"])
    allowed_tail = set(rules["allowed_tail"])
    return bool(head_types & allowed_head) and bool(tail_types & allowed_tail)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample balanced multilingual chunks, convert char-span entities to token-level "
            "spans with spaCy tokenization, and run batched GLiREL relation extraction."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("annotated_test-full"),
        help="Directory containing annotated prediction JSON files.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("text_dataset/extracted_2"),
        help="Directory containing source JSON files with chunk text.",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="gliner",
        help=(
            "Optional model filter for input annotation docs (e.g. `gliner`, `base-llm`). "
            "Set empty string to disable filtering."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/glirel_relation_sample.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jackboyla/glirel-large-v0",
        help="GLiREL model name or local path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of chunks to sample.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Languages to include in the balanced sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--choice-index",
        type=int,
        default=0,
        help="Which choice index to use from `choice_*` / per-chunk choices.",
    )
    parser.add_argument(
        "--min-entities",
        type=int,
        default=1,
        help="Minimum number of char-level entities required per sampled chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GLiREL relation prediction.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Relation confidence threshold passed to GLiREL.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--relation-label",
        action="append",
        dest="relation_labels",
        default=None,
        help="Relation label to predict. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help=(
            "Use only relation labels for GLiREL prediction and disable "
            "head/tail entity-type restriction filtering."
        ),
    )
    parser.add_argument(
        "--spacy-lang",
        type=str,
        default="xx",
        help=(
            "spaCy language code for tokenizer construction. "
            "Use 'xx' for multilingual tokenization."
        ),
    )
    parser.add_argument(
        "--tokenization",
        choices=["spacy", "glirel_regex", "whitespace"],
        default="glirel_regex",
        help=(
            "Tokenization used for char->token mapping and GLiREL input. "
            "`glirel_regex` mirrors GLiREL internal regex splitting."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    requested_languages = [str(x).strip() for x in args.languages if str(x).strip()]
    language_set = set(requested_languages)

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if not requested_languages:
        raise ValueError("No languages requested.")

    relation_labels = (
        [lbl.strip() for lbl in args.relation_labels if lbl and lbl.strip()]
        if args.relation_labels
        else list(DEFAULT_RELATION_CONSTRAINTS.keys())
    )
    relation_labels = list(dict.fromkeys(relation_labels))
    relation_constraints = validate_relation_constraints(DEFAULT_RELATION_CONSTRAINTS)
    print(
        "[INFO] Relation schema validated: "
        f"{len(relation_constraints)} relation types with allowed head/tail constraints."
    )
    unknown_relations = [r for r in relation_labels if r not in relation_constraints]
    if unknown_relations:
        print(
            "[WARN] No default constraints found for relation labels: "
            + ", ".join(unknown_relations)
        )
    use_constraints = not args.labels_only
    if use_constraints:
        print("[INFO] Constraint filtering is enabled.")
    else:
        print("[INFO] Labels-only mode enabled: skipping head/tail type restrictions.")

    print("[INFO] Collecting candidate chunks...")
    model_filter = args.input_model.strip() if args.input_model else None
    if model_filter:
        print(f"[INFO] Input model filter: {model_filter}")
    candidates = collect_candidates(
        predictions_dir=args.predictions_dir,
        source_dir=args.source_dir,
        languages=language_set,
        choice_index=args.choice_index,
        min_entities=args.min_entities,
        model_filter=model_filter,
    )

    if not candidates:
        raise ValueError("No candidates found for the requested filters.")

    print(f"[INFO] Candidate chunks: {len(candidates)}")
    sampled, quotas = sample_balanced(
        candidates=candidates,
        total_samples=args.samples,
        languages=requested_languages,
        rng=rng,
    )
    print(f"[INFO] Sampled chunks: {len(sampled)}")
    print(f"[INFO] Language quotas: {quotas}")

    device = resolve_device(args.device)
    print(f"[INFO] Loading GLiREL model `{args.model_name}` on `{device}`...")
    model = GLiREL.from_pretrained(args.model_name, map_location=device).to(device)
    spacy_tokenizer = None
    if args.tokenization == "spacy":
        print(f"[INFO] Building spaCy tokenizer (`{args.spacy_lang}`)...")
        spacy_tokenizer = spacy.blank(args.spacy_lang).tokenizer
    else:
        print(f"[INFO] Using tokenization mode: {args.tokenization}")

    prepared: list[dict[str, Any]] = []
    alignment_methods = defaultdict(int)
    alignment_scores: list[float] = []
    for cand in sampled:
        tokens, token_entities, model_ner = convert_entities_to_token_spans(
            text=cand.text,
            char_entities=cand.entities_char,
            tokenization=args.tokenization,
            nlp_tokenizer=spacy_tokenizer,
        )
        for ent in token_entities:
            method = str(ent.get("alignment_method", "unknown"))
            alignment_methods[method] += 1
            try:
                alignment_scores.append(float(ent.get("alignment_score", 0.0)))
            except Exception:
                pass
        prepared.append(
            {
                "source_file": cand.source_file,
                "discipline": cand.discipline,
                "doc_id": cand.doc_id,
                "lang": cand.lang,
                "chunk_idx": cand.chunk_idx,
                "text": cand.text,
                "tokenized_text": tokens,
                "entities_char": cand.entities_char,
                "entities_token": token_entities,
                "_ner_for_model": model_ner,
            }
        )

    print(
        f"[INFO] Running relation predictions in batches of {args.batch_size} "
        f"(threshold={args.threshold})..."
    )
    to_predict = [x for x in prepared if x["_ner_for_model"]]
    skipped = 0
    for item in prepared:
        if not item["_ner_for_model"]:
            item["relations"] = []
            item.pop("_ner_for_model", None)
            skipped += 1

    raw_pred_count = 0
    kept_pred_count = 0
    for start in range(0, len(to_predict), args.batch_size):
        batch = to_predict[start : start + args.batch_size]
        batch_tokens = [x["tokenized_text"] for x in batch]
        batch_ner = [x["_ner_for_model"] for x in batch]
        batch_predictions = model.batch_predict_relations(
            texts=batch_tokens,
            labels=relation_labels,
            flat_ner=True,
            threshold=args.threshold,
            ner=batch_ner,
        )

        for item, rels in zip(batch, batch_predictions):
            span_label_index, text_label_index = build_entity_label_indexes(
                item["entities_token"]
            )
            normalized_rels: list[dict[str, Any]] = []
            for rel in rels:
                raw_pred_count += 1
                head_raw = rel.get("head_text")
                tail_raw = rel.get("tail_text")

                if isinstance(head_raw, list):
                    head_decoded = _tokens_to_surface(head_raw)
                    head_tokens = head_raw
                else:
                    head_decoded = str(head_raw)
                    head_tokens = []

                if isinstance(tail_raw, list):
                    tail_decoded = _tokens_to_surface(tail_raw)
                    tail_tokens = tail_raw
                else:
                    tail_decoded = str(tail_raw)
                    tail_tokens = []

                normalized_rel = {
                    "head_pos": rel.get("head_pos"),
                    "tail_pos": rel.get("tail_pos"),
                    "head_tokens": head_tokens,
                    "tail_tokens": tail_tokens,
                    "head_text": head_decoded,
                    "tail_text": tail_decoded,
                    "label": rel.get("label"),
                    "score": rel.get("score"),
                }
                if (not use_constraints) or relation_allowed_by_constraints(
                    normalized_rel,
                    span_label_index,
                    text_label_index,
                    relation_constraints,
                ):
                    normalized_rels.append(normalized_rel)
                    kept_pred_count += 1

            item["relations"] = normalized_rels
            item.pop("_ner_for_model", None)

    if skipped:
        print(
            f"[INFO] Skipped GLiREL inference for {skipped} chunks with zero token-level entities."
        )
    print(
        "[INFO] Relation predictions: "
        f"raw={raw_pred_count}, kept_after_constraints={kept_pred_count}"
    )

    by_lang = defaultdict(int)
    for item in prepared:
        by_lang[item["lang"]] += 1

    output = {
        "config": {
            "predictions_dir": str(args.predictions_dir),
            "source_dir": str(args.source_dir),
            "model_name": args.model_name,
            "device": device,
            "samples": args.samples,
            "languages": requested_languages,
            "seed": args.seed,
            "choice_index": args.choice_index,
            "min_entities": args.min_entities,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "labels_only": args.labels_only,
            "spacy_lang": args.spacy_lang,
            "tokenization": args.tokenization,
            "relation_labels": relation_labels,
            "relation_constraints": {
                k: relation_constraints[k]
                for k in relation_labels
                if k in relation_constraints
            },
        },
        "sample_stats": {
            "total_chunks": len(prepared),
            "by_language": dict(by_lang),
            "entity_alignment_methods": dict(alignment_methods),
            "entity_alignment_score_avg": (
                round(sum(alignment_scores) / len(alignment_scores), 4)
                if alignment_scores
                else 0.0
            ),
            "relations_raw": raw_pred_count,
            "relations_kept_after_constraints": kept_pred_count,
        },
        "items": prepared,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Wrote output to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
