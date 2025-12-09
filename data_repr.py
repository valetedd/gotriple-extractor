from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, TypeAlias, Union

from pydantic import BaseModel, Field


@dataclass
class AnnotatedChunk:
    chunk_id: int
    span: str
    annotations: List[Dict[str, Any]]
    annotation_t: Literal["ent", "rel", "triple"]


@dataclass
class Document:
    doc_id: int
    chunks: Sequence[AnnotatedChunk]


class Entity(BaseModel):
    start: int = Field(description="Character start position")
    end: int = Field(description="Character end position")
    label: str = Field(description="Entity type/category")
    text: str = Field(description="The actual entity text")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ExtractedEntities(BaseModel):
    """Container for extracted entities"""

    entities: List[Entity]
