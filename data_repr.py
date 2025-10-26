from dataclasses import dataclass
from typing import List, Literal


@dataclass
class PDFDocument:
    blocks: List[List[List]]
    text: str


@dataclass
class AnnotatedSpan:
    span: str
    annotations: tuple[str]
    annotation_type: Literal["ent", "rel", "triple"]
