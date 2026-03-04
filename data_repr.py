from dataclasses import dataclass
from typing import List, Literal, Sequence

from pydantic import BaseModel, Field

### PYDANTIC SCHEMAS


class Entity(BaseModel):
    start: int = Field(description="Character start position")
    end: int = Field(description="Character end position")
    label: str = Field(description="Entity type/category")
    text: str = Field(description="The actual entity text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class Relation(BaseModel):
    head: Entity
    tail: Entity
    type: str


class Triple(BaseModel):
    subject: Entity
    predicate: Relation
    object: Entity


@dataclass
class AnnotatedChunk:
    chunk_id: int
    span: str
    annotations: List[Entity] | List[Relation] | List[Triple]
    annotation_t: Literal["ent", "rel", "triple"]


@dataclass
class Document:
    doc_id: int
    chunks: Sequence[AnnotatedChunk]
