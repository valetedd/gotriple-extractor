from dataclasses import dataclass
from typing import Iterator, List, Literal


@dataclass
class Document:
    chunks: List[str] | Iterator[List[str]]
    text: str


@dataclass
class AnnotatedSpan:
    span: str
    annotations: tuple[str] | tuple[tuple[str]]
    annotation_type: Literal["ent", "rel", "triple"]
