You are an expert NER judge performing a final refinement pass.

TASK:
Produce the best possible NER annotation for the INPUT TEXT, by refining the provided ANNOTATIONS_TO_REVIEW.

CONSTRAINTS:
- Output a valid JSON array with this schema:
    - "text": string — exact substring copied from the INPUT TEXT (do not normalize)
    - "label": string — must be one of: {ENTITY_TAGS}
    - "start": integer — starting character offset of the entity span (0-based)
    - "end": integer — ending character offset (exclusive)
    - "score": number — float between 0.0 and 1.0 (inclusive)

- Use WIKIPEDIA-LINKED ENTITIES as authoritative disambiguation. 
- ALWAYS output a JSON array and nothing else
