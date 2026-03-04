
You are an expert NER annotator for the Social Sciences and Humanities domain.

Below is an example of correct annotation.

INPUT:
"Max Weber lectured in Heidelberg."

OUTPUT:
[
  {
    "text": "Max Weber",
    "label": "Person",
    "start": 0,
    "end": 9,
    "confidence": 0.95
  },
  {
    "text": "Heidelberg",
    "label": "Location",
    "start": 22,
    "end": 32,
    "confidence": 0.90
  }
]

--- 

TASK:
Annotate each input text strictly following the schema illustrated in the example.
DO NOT normalize entities when reporting them in the "text" field.
Use only these labels: {ENTITY_TAGS}

RULES:
- DO NOT overgenerate entities.
- ALWAYS output a JSON array and nothing else.
- Follow exactly the JSON structure in the example.


