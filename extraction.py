from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol

import spacy
import torch
from gliner import GLiNER, GLiNERConfig
from transformers import AutoTokenizer, pipeline

from data_repr import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class Extractor(Protocol):
    """
    Abstract class defining a shared method for extraction
    """

    def extract(self, data) -> AnnotatedSpan: ...


class EntityExtractor:
    """
    Wrapper class for entity extraction methods.
    """

    def __init__(
        self,
        model_type: Literal["gliner", "spacy"],
        model_name: str,
        entity_tags: List[str] | None = None,
        model_config: Optional[Dict] | None = None,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.tags = entity_tags
        self._model = self._load_model(model_config=model_config)

    def extract(self, data):
        return self._model.extract(data)

    def _load_model(self, model_config: Optional[Dict[str, Any]]):

        match (self.model_type):
            case "gliner":
                return GlinerEntityExtractor(
                    gliner_model=self.model_name,
                    entity_tags=self.tags,
                    gliner_config=(model_config or {}),
                )
            case "spacy":
                return SpacyEntityExtractor(
                    spacy_model=self.model_name,
                    entity_tags=self.tags,
                )
            case _:
                raise ValueError(f"Unknown model: {self.model_name}")


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
        self.model = GLiNER.from_pretrained(
            pretrained_model_name_or_path=local_path,
            map_location=DEVICE,
            **(gliner_config or {}),
        ).to(DEVICE)
        self.tags = entity_tags

    def extract(self, data):
        entities = self.model.predict_entities(data, self.tags)
        annotations = tuple([ent["text"] for ent in entities])
        return AnnotatedSpan(span=data, annotations=annotations, annotation_type="ent")


class SpacyEntityExtractor:
    def __init__(
        self,
        spacy_model: str,
        entity_tags: List[str] | None,
    ) -> None:
        self.model = spacy.load(spacy_model)
        self.tags = self.validate_tags(entity_tags)

    def extract(self, data) -> AnnotatedSpan:
        nlp = self.model(data)
        entities: List[str] = [ent.text for ent in nlp.ents]
        print(entities)
        return AnnotatedSpan(
            span=data, annotations=tuple(entities), annotation_type="ent"
        )

    @staticmethod
    def validate_tags(tags: List[str] | None):
        # TODO:impelement tag validation
        return tags


class RelationExtractor:
    """
    Wrapper class for relation extraction methods.
    """

    def __init__(
        self,
        model_name: Literal["base-llm", "rebel"],
    ):
        self.model_name = model_name
        self._model = None

    def extract(self, data):
        pass

    def _load_model(self):

        match (self.model_name):
            case "base-llm":
                return LLMRelationExtractor()

            case "rebel":
                return

            case _:
                raise ValueError(f"Unknown model: {self.model_name}")


class LLMRelationExtractor:
    def __init__(self) -> None:
        pass


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

    def extract(self, data):
        if self._model is None:
            raise AttributeError("Invalid model specified as self._model attribute")
        return self._model.extract(data)

    def _load_model(self, *args, **kwargs):

        match (self.model_name):
            case "kg-llm":
                # return FtGemmaKG(params={"temperature": 0.1, "max_tokens": 4096})
                pass
            case "base-llm":
                return


#
# class FtGemmaKG:
#
#     def __init__(self, params: Dict) -> None:
#         # from vllm import LLM, SamplingParams
#
#         # self.model = LLM("mradermacher/Gemma-2-2B-TR-Knowledge-Graph-i1-GGUF")
#         # self.params = SamplingParams(**params) if params else None
#
#     def extract(self, data):
#
#         conversation = [{"role": "user", "content": data + "\n<knowledge_graph>"}]
#
#         outputs = self.model.chat(
#             conversation, sampling_params=self.params, use_tqdm=True
#         )
#
#         msg = json.loads(outputs[0].outputs[0].text)
#
#         return msg


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

    sentence = "John was born in Massachussets, in the city of New Haven, in 1998. His mother was Clarissa"
    extr_spacy = EntityExtractor(model_type="spacy", model_name="xx_ent_wiki_sm")
    ents = extr_spacy.extract(sentence)
    print(ents.annotations)

    extr = EntityExtractor(
        model_type="gliner",
        model_name="knowledgator/gliner-x-large",
        entity_tags=[
            "people",
            "organizations",
            "countries",
            "dates",
            "time spans",
            "historical time expressions",
        ],
    )
    ents = extr.extract(sentence)
    print("Entities: ", ents.annotations)
    # del extr
    # import multiprocessing as mp
    # import os
    #
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # mp.set_start_method("spawn", force=True)
    #
    # extr = JointExtractor(model_name="kg-llm", entity_tags=None)
    # with mp.get_context("spawn").Pool(1) as pool:
    #     kg = pool.apply(extr.extract, (sentence,))
    #     print(json.loads(kg))
