
You are an expert NER annotator for the Social Sciences and Humanities domain.

Below is an example of correct annotation.

EXAMPLE
Text:
"Max Weber lectured in Heidelberg."

Output:
[
  {
    "text": "Max Weber",
    "label": "Person",
    "start": 0,
    "end": 9,
    "score": 0.95
  },
  {
    "text": "Heidelberg",
    "label": "Location",
    "start": 22,
    "end": 32,
    "score": 0.90
  }
]

--- 

TASK:
Annotate each input text according to the example

RULES:
- Follow the same JSON structure exactly
- Use only these labels: {ENTITY_TAGS}
- ALWAYS output a JSON array and nothing else


