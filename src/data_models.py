import uuid
from typing import Any

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    # Add just casue moulinette required that field name
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


# Using pydantic dataclass to add a layer of validation
@dataclass
class Chunk:
    id: int
    content: str
    start_index: int
    end_index: int
    file_path: str


@dataclass
class CachedResources:
    model: Any
    tokenizer: Any
