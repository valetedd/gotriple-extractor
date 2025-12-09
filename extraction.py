import ast
import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol

import spacy
import torch
from dotenv import load_dotenv

load_dotenv()
import os

from gliner import GLiNER
from litellm import acompletion, completion
from openai import AsyncOpenAI

from data_repr import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class Extractor(Protocol):
    """
    Abstract class defining a shared method for extraction
    """

    @abstractmethod
    def extract(self, data: str, chunk_id: int = 0) -> AnnotatedChunk: ...

    @abstractmethod
    def extract_doc(self, data: List[str], doc_id: int = 0) -> Document: ...


class EntityExtractor:
    """
    Class for entity extraction methods.
    """

    def __init__(
        self,
        model_type: Literal["gliner", "spacy", "base-llm"],
        model_name: str,
        entity_tags: List[str] | None = None,
        model_config: Optional[Dict] | None = None,
        prompt: Optional[str] = None,
        prompt_vars: Optional[Dict[str, Any]] = None,
    ):
        self.type = model_type
        self.name = model_name
        self.tags = entity_tags
        self.prompt = prompt
        self._model = self._load_model(
            model_config=model_config, prompt_vars=prompt_vars
        )

    def extract(self, data: str, chunk_id: int = 0) -> AnnotatedChunk:
        if self._model is None:
            raise ValueError("No model was initialized for this instance")

        return self._model.extract(data, chunk_id)

    def extract_doc(self, data: List[str], doc_id: int = 0, **kwargs) -> Document:
        return self._model.extract_doc(data, doc_id, **kwargs)

    def _load_model(
        self,
        model_config: Optional[Dict[str, Any]],
        prompt_vars: Optional[Dict[str, Any]],
    ):

        match (self.type):
            case "gliner":
                return GlinerEntityExtractor(
                    gliner_model=self.name,
                    entity_tags=self.tags,
                    gliner_config=(model_config or {}),
                )
            case "spacy":
                return SpacyEntityExtractor(
                    spacy_model=self.name,
                    entity_tags=self.tags,
                )
            case "base-llm":
                if not self.prompt:
                    raise ValueError("A prompt has to specified for LLM extraction")
                return LLMExtractor(
                    prompt=self.prompt,
                    model_name=self.name,
                    entity_tags=self.tags,
                    config=model_config,
                    task="ent",
                    prompt_variables=prompt_vars,
                )

            case _:
                raise ValueError(f"Unknown model: {self.name}")


class GlinerEntityExtractor:
    def __init__(
        self,
        gliner_model: str,
        entity_tags: List[str] | None,
        gliner_config: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not entity_tags:
            raise ValueError(
                "A set of entity tags has to be specified for GLiNER to work correctly"
            )
        # config = GLiNERConfig.from_pretrained(gliner_model)
        # if gliner_config:
        #     config.__dict__.update(gliner_config)
        # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # self.model = GLiNER(config=config, tokenizer=tokenizer).to(DEVICE)
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=gliner_model,
            local_files_only=True,  # Don't download, just get the cache path
        )
        self.config = gliner_config
        self.tags = entity_tags
        self.model: GLiNER = GLiNER.from_pretrained(
            pretrained_model_name_or_path=local_path,
            map_location=DEVICE,
        ).to(DEVICE)

    def extract(self, data: str, chunk_id: int = 0):
        entities = self.model.predict_entities(data, self.tags, **(self.config or {}))
        annotations = [ent for ent in entities]
        return AnnotatedChunk(
            chunk_id=chunk_id, span=data, annotations=annotations, annotation_t="ent"
        )

    def extract_doc(self, data: List[str], doc_id: int = 0) -> Document:
        result = []
        preds = self.model.run(data, self.tags, **(self.config or {}))
        for i, (chunk_ents, b) in enumerate(zip(preds, data)):
            ann_chunk = AnnotatedChunk(
                chunk_id=i, span=b, annotations=chunk_ents, annotation_t="ent"
            )
            result.append(ann_chunk)

        if not result:
            raise ValueError("No data")

        return Document(doc_id=doc_id, chunks=result)


class SpacyEntityExtractor:
    def __init__(
        self,
        spacy_model: str,
        entity_tags: List[str] | None,
    ) -> None:
        self.model = spacy.load(spacy_model)
        self.tags = self.validate_tags(entity_tags)

    def extract(self, data, chunk_id: int) -> AnnotatedChunk:
        nlp = self.model(data)
        entities = []
        for ent in nlp.ents:
            if self.tags:
                if not ent.label_ in self.tags:
                    continue
            annotation = {
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "label": ent.label_,
                "score": -1,
            }
            entities.append(annotation)

        return AnnotatedChunk(
            chunk_id=chunk_id, span=data, annotations=entities, annotation_t="ent"
        )

    def extract_doc(self, data: List[str], doc_id: int = 0) -> Document:
        docs = list(self.model.pipe(data))

        results = []
        for i, (block, doc) in enumerate(zip(data, docs)):
            ents = [
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                    "score": -1,
                }
                for ent in doc.ents
            ]
            ann_chunk = AnnotatedChunk(
                chunk_id=i, span=block, annotations=ents, annotation_t="ent"
            )
            results.append(ann_chunk)

        return Document(doc_id=doc_id, chunks=results)

    @staticmethod
    def validate_tags(tags: List[str] | None):
        # TODO:impelement tag validation
        return tags


class RelationExtractor:
    """
    Class for relation extraction methods.
    """

    def __init__(
        self,
        model_type: Literal["base-llm", "rebel"],
        model_name: str,
        relation_tags: List[str] | None = None,
        model_config: Optional[Dict] = None,
        prompt: Optional[str] = None,
        prompt_vars: Optional[dict[str, str]] = None,
    ):
        self.type = model_type
        self.name = model_name
        self._model = self._load_model(model_config)
        self.tags = relation_tags
        self.prompt = prompt
        self.prompt_vars = prompt_vars

    def extract(self, data, chunk_id: int) -> AnnotatedChunk:
        return AnnotatedChunk(
            chunk_id=chunk_id, span="", annotations=[{}], annotation_t="ent"
        )

    def _load_model(self, model_config: Optional[Dict] = None):

        match (self.type):
            case "base-llm":
                if not self.prompt:
                    raise ValueError("A prompt has to specified for LLM extraction")
                return LLMExtractor(
                    prompt=self.prompt,
                    model_name=self.name,
                    entity_tags=self.tags,
                    config=(model_config or {}),
                    task="rel",
                    prompt_variables=self.prompt_vars,
                )

            case "rebel":
                return

            case "glirel":
                return

            case _:
                raise ValueError(f"Unknown model: {self.type}")


class LLMExtractor:
    def __init__(
        self,
        prompt: str,
        model_name: str,
        entity_tags: List[str],
        config: Optional[Dict[str, Any]],
        task: Literal["ent", "rel", "triple"],
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not task in ["ent", "rel", "triple"]:
            raise ValueError(
                f"An unsupported task was specified for this extractor: {task}. It can either be 'ent', 'rel' or 'triple'"
            )

        if not prompt_variables:
            prompt_variables = {"ENTITY_TAGS": entity_tags}
        else:
            prompt_variables["ENTITY_TAGS"] = entity_tags
        self.prompt = self._get_formatted_prompt(prompt, prompt_variables)
        self.tags = entity_tags
        self.config = config
        self.model_name = model_name
        if task not in ["ent", "rel", "triple"]:
            raise TypeError(
                "'task' parameter has to have one of these values: ['ent', 'rel', 'triple']"
            )
        self.task = task
        self.client = AsyncOpenAI(
            base_url="https://llm.graphia-ssh.eu/v1/",
            api_key=os.environ.get("API_KEY"),
        )
        available_models = os.environ.get("AVAILABLE_MODELS")
        if available_models and not any(
            [
                model
                for model in available_models.split("; ")
                if model in self.model_name
            ]
        ):
            raise ValueError(
                "Unsupported model was specified. Try selecting another one"
            )

    def extract(self, data: str, chunk_id: int = 0) -> AnnotatedChunk:

        try:
            response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": data},
                    ],
                    **(self.config or {}),
                )
            response_content = response.choices[0].message.content
            return AnnotatedChunk(
                chunk_id=chunk_id,
                span=data,
                annotations=ast.literal_eval(response_content),
                annotation_t=self.task,
            )
        except Exception as e:
            print(f"{e}: failed to extract text!")
            return AnnotatedChunk(
                chunk_id=chunk_id, span=data, annotations=[], annotation_t=self.task
            )

    async def extract_async(
        self, data: str, chunk_id: int = 0, semaphore: asyncio.Semaphore | None = None
    ) -> AnnotatedChunk:
        """Async extraction for a single chunk with concurrency control."""
        if semaphore is None:
            raise TypeError(
                "For async version of extract you need to specify a semaphore object. Use 'extract' instead for a synchronous implementation"
            )

        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": data},
                    ],
                    **(self.config or {}),
                )
                response_content = response.choices[0].message.content
                return AnnotatedChunk(
                    chunk_id=chunk_id,
                    span=data,
                    annotations=ast.literal_eval(response_content),
                    annotation_t=self.task,
                )
            except Exception as e:
                print(f"{e}: failed to extract text for chunk {chunk_id}!")
                return AnnotatedChunk(
                    chunk_id=chunk_id, span=data, annotations=[], annotation_t=self.task
                )

    def extract_doc(
        self, data: List[str], doc_id: int = 0, max_concurrent: int = 10
    ) -> Document:
        """Extract entities from all chunks in a document using async."""

        async def _gather_chunks(tasks: List) -> List[AnnotatedChunk]:
            """Helper to gather all async results."""
            return await asyncio.gather(*tasks)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all chunks
        tasks = [
            self.extract_async(chunk, chunk_id, semaphore)
            for chunk_id, chunk in enumerate(data)
        ]

        # Run all tasks concurrently
        chunks = asyncio.run(_gather_chunks(tasks))

        return Document(doc_id=doc_id, chunks=chunks)

    @staticmethod
    def _get_formatted_prompt(prompt, prompt_variables):
        if not isinstance(prompt, str):
            raise TypeError("'prompt' must be a string")
        if os.path.isfile(prompt):
            with open(prompt, mode="r", encoding="utf-8") as f:
                prompt = f.read()
        if prompt_variables:
            throw = False
            not_included_ks = []
            for k in prompt_variables.keys():
                if k not in prompt:
                    if not throw:
                        throw = True
                        not_included_ks.append(k)
            if throw:
                raise ValueError(
                    f"The following variables were not found in the prompt:\n {not_included_ks}"
                )
            return prompt.format(**prompt_variables)


class JointExtractor:
    """
    Class for joint triplet extraction implementations, with fused entity and
    relation extraction modules.
    """

    def __init__(
        self,
        model_name: str,
        entity_tags: List[str] | None,
    ):
        self.model_name = model_name
        self.tags = entity_tags
        self._model = self._load_model()

    def extract(self, data) -> AnnotatedChunk:
        if self._model is None:
            raise AttributeError("Invalid model specified as self._model attribute")
        return self._model.extract(data)

    def _load_model(self, *args, **kwargs):

        match (self.model_name):
            case "rebel":
                pass
            case "base-llm":
                return


@dataclass
class ModularExtractor:
    """
    Dataclass for two-component pipeline implementations, with distinct entity
    and relation extractors.

    Fields
    - entity_extractor
    - relation_extractor
    """

    entity_extractor: EntityExtractor
    relation_extractor: RelationExtractor


class TripleExtractor:
    """
    General class for instantiating the extraction pipeline.
    """

    def __init__(self, data, extractor: JointExtractor | ModularExtractor):
        pass


if __name__ == "__main__":
    sentence = "Epicuro fu un filosofo dell'Antica Grecia. Tra il 1981 e il 1991 vi furono varie battaglie finanziate dal Nuovo Ordine dei Templari, fondati da Paperino nel VII secolo"
    sentence2 = "John was born in Massachussets, in the city of New Haven, in 1998. His mother was Clarissa"

    entity_types = ["person name", "organization", "location name", "time references"]

    # Aailable models: DeepSeek-V3.1-vLLM; Qwen3-Coder-30B-A3B-Instruct-Q8_0
    extr = EntityExtractor(
        model_type="base-llm",
        model_name="DeepSeek-V3.1-vLLM",
        entity_tags=entity_types,
        model_config={"temperature": 0.2},
        prompt="./prompt.md",
    )
    print("============ LLM ============\n")
    ents = extr.extract(sentence).annotations
    print("\t", ents)
    ents2 = extr.extract(sentence2).annotations
    print("\t", ents2)

    ents3 = extr.extract_doc([sentence, sentence2])
    print(ents3)

    # print("============ spaCy ============\n")
    # extr_spacy = EntityExtractor(model_type="spacy", model_name="xx_ent_wiki_sm")
    # ents = extr_spacy.extract(sentence)
    # print("\t", ents.annotations)
    # ents2 = extr_spacy.extract(sentence2)
    # print("\t", ents2.annotations)
    #
    # print("============ GLiNER ============\n")
    # extr_gliner = EntityExtractor(
    #     model_type="gliner",
    #     model_name="knowledgator/gliner-x-large",
    #     entity_tags=[
    #         "person name",
    #         "organization",
    #         "location name",
    #         "time references",
    #     ],
    #     model_config={
    #         "threshold": 0.6,
    #         "batch_size": 6
    #     }
    # )
    # ents = extr_gliner.extract(sentence).annotations
    # pprint(ents)
    #
    # ents2 = extr_gliner.extract(sentence2).annotations
    # pprint(ents2)
