You are an expert NER judge performing a final refinement pass.

TASK:
Produce the best possible NER annotation for the INPUT TEXT, by refining the provided ANNOTATIONS. 
Use the LINKED ENTITIES to corroborate your refinement task.

CONSTRAINTS:
- Output a valid JSON array with this schema:
    - "text": string — exact substring copied from the INPUT TEXT (IMPORTANT: do not normalize this)
    - "label": string — must be one of: {ENTITY_TAGS}
    - "start": integer — starting character offset of the entity span (0-based)
    - "end": integer — ending character offset (exclusive)
    - "confidence": number — float between 0.0 and 1.0 (inclusive) expressing your confidence.

- Read the INPUT TEXT carefully and understand its meaning.
- Understand the relation between the annotation and their context (the INPUT TEXT)
- Consult the LINKED ENTITIES strictly to ensure the quality of the ANNOTATIONS.
- DO NOT overgenerate entities.
- ALWAYS output a JSON array and nothing else.
