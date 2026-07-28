# TODO: CHECK THE UV SYNC COMPLAINING ABOUT BM25 EXTRA CORE
# TODO: IMPLEMENT DOCSTRINGS LATER
import json
import sys
from pathlib import Path
from typing import Any, cast

import bm25s
import fire
import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from src.answer_generator import get_chat_template, load_model
from src.data_models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.docs_chunking import Chunk, Chunker
from src.docs_indexing import indexing
from src.docs_loading import Document, load_files
from src.search import (
    load_retriever,
    save_to_json_file,
    search_batch,
    search_one,
)


class Boss:
    """"""

    def __init__(self) -> None:
        """"""
        pass

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_path: str = "data/raw/vllm-0.10.1",
        processed_path: str = "data/processed/",
        method: str = "bm25",
    ) -> None:
        """A quick method which fired by the CMD line argument,
        Its manage the indexing stage
        """

        try:
            # Load the files into program
            docs: list[Document] = load_files(input_path=raw_path)

            # Build the chunker and start processing the files
            chunker: Chunker = Chunker(files=docs, max_size=max_chunk_size)
            chunks: list[Chunk] = chunker.process_files()
            # Run the main index processing
            indexing(
                chunks=chunks, processed_path=processed_path, method=method
            )
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            sys.exit()

    def search(
        self,
        query: str,
        k: int = 1,
        processed_path: str = "data/processed/",
        question_id: str = "0",
    ) -> MinimalSearchResults | list[str]:
        """"""

        # Check if the index exist to be loaded ALSO pickled chunks
        path: Path = Path(processed_path)
        if not path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            sys.exit()

        # Load the retriever AND Chunks that was pickled
        retriever: bm25s.BM25
        chunks: list[Chunk]
        retriever, chunks = load_retriever(processed_path)
        # Retrieve results for one question
        retrieve_results: list[Chunk] = search_one(
            query=query, k=k, retriever=retriever, chunks=chunks
        )

        # Build Minimal search result to be returned and used later in answers
        min_search_result: MinimalSearchResults = MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=[],  # filled below
        )

        sources_path_results: list[str] = []
        # Validate each chunk by a MinimalSource model
        # and print the formate required
        for chunk in retrieve_results:
            # Validating chunk
            min_source: MinimalSource = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index,
            )
            sources_path_results.append(
                f"{min_source.file_path} [{min_source.first_character_index}:{
                    min_source.last_character_index
                }]"
            )
            min_search_result.retrieved_sources.append(min_source)

        # return min_search_result
        return sources_path_results

    def search_dataset(
        self,
        dataset_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        k: int = 1,
        save_directory: str = "data/output/search_results/AnsweredQuestions",
        processed_path: str = "data/processed/",
        save_file: str = "StudentSearchResults.json",
    ) -> None:
        # TODO: IMPROVE LATER
        """
        Reach a batch of questions from the provided dataset path and operate search
        over all of them after validating the loaded questions,

        Args:

        Returns:


        """

        # Check if dataset exist and can be opened
        try:
            # Read the json dataset
            with open(dataset_path, "r") as f:
                dataset_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: dataset file not found: {dataset_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Check if the index exist to be loaded
        p_path: Path = Path(processed_path)
        if not p_path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            sys.exit(1)

        # Check the scope of the questions to determine the output path
        d_path: Path = Path(dataset_path)
        questions_scope: str
        if "AnsweredQuestions" in d_path.parts:
            questions_scope = "AnsweredQuestions"
        elif "UnansweredQuestions" in d_path.parts:
            questions_scope = "UnansweredQuestions"
        else:
            print(f"Cannot determine dataset scope from path: {dataset_path}")
            sys.exit(1)

        # Get validator model based on the scope
        ValidatorModel: type[AnsweredQuestion | UnansweredQuestion]
        if questions_scope == "AnsweredQuestions":
            ValidatorModel = AnsweredQuestion
        elif questions_scope == "UnansweredQuestions":
            ValidatorModel = UnansweredQuestion

        # Validate each question AND build a list of questions
        # to be used in retrieving
        batch_questions: list[str] = []
        validated_batch: list[AnsweredQuestion | UnansweredQuestion] = []
        for q in dataset_data["rag_questions"]:
            validated_batch.append(ValidatorModel.model_validate(q))
            batch_questions.append(q["question"])

        # Now Validate the full questions batch (Rag dataset)
        _: RagDataset = RagDataset(rag_questions=validated_batch)

        # Load the indexed files
        retriever, chunks = load_retriever(processed_path)
        batch_results: list[list[Chunk]] = search_batch(
            queries=batch_questions,
            k=k,
            retriever=retriever,
            chunks=chunks,
        )

        # THE TARGET IS TO BUILD StudentSearchResults obj, to do that:
        # Validate the search result chunks by [MinimalSource] AND
        # Validate and build [MinimalSearchResults] which represent
        # to search result for a single question and by collect all
        # the required ones we can build StudentSearchResults,
        boss_search_result: StudentSearchResults = StudentSearchResults(
            search_results=[], k=k
        )

        # Iterate over each question
        for i, question_result in tqdm.tqdm(enumerate(batch_results)):
            mini_search_result: MinimalSearchResults = MinimalSearchResults(
                question_id=validated_batch[i].question_id,
                question=validated_batch[i].question,
                retrieved_sources=[],
            )
            # Collect the validated chunks [MinimalSource]
            # Validate the chunks retrieved as result
            for chunk in tqdm.tqdm(question_result):
                mini_search_result.retrieved_sources.append(
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.start_index,
                        last_character_index=chunk.end_index,
                    )
                )
            boss_search_result.search_results.append(mini_search_result)

        # Create paths if not exist and save the json result
        dir_path: Path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        full_path: Path = dir_path / save_file
        save_to_json_file(file_path=full_path, obj=boss_search_result)

    def answer(
        self,
        query: str,
        k: int = 1,
        generator_model_name: str = "Qwen/Qwen3-0.6B",
        embeddings_model_name: str | None = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        processed_path: str = "data/processed/",
        question_id: str | int = "0",
        save_path: str | None = None,
        # This is just helpful for answer_dataset
        cached_pack: dict[str, Any] | None = None,
        winning_chunks: list[Chunk] | None = None,
    ) -> MinimalAnswer:
        """Answer a single query using the retrieved context."""

        # check if its called normally
        if cached_pack is None:
            try:
                # Load retriever and chunks
                retriever, chunks = load_retriever(
                    processed_path=processed_path
                )

                # Retrieve the winning chunks to build the response
                winning_chunks: list[Chunk] = search_one(
                    query=query,
                    k=k,
                    retriever=retriever,
                    chunks=chunks,
                )

                # Load model
                # TODO: MAYBE LATER MAKE IT HANDLE ANOTHER MODELS
                model, tokenizer = load_model(
                    model_name=generator_model_name, cache_dir=cache_dir
                )
            except Exception as e:
                print(e)
                sys.exit()

        # If the 'answer' method called from 'answer_dataset'
        # we skip loading the retriever since 'answer_dataset'
        # already have dataset
        else:
            # retriever = cached_pack["retriever"]
            # chunks = cached_pack["chunks"]
            # Retrieve the winning chunks to build the response
            # winning_chunks: list[Chunk] = search_one(
            #     query=query,
            #     k=k,
            #     retriever=retriever,
            #     chunks=chunks,
            # )
            model = cached_pack["model"]
            tokenizer = cached_pack["tokenizer"]

        messages: list[dict[str, str]] = get_chat_template(
            chunks=winning_chunks,
            query=query,
        )

        # tokenized_result currently consist of 'input_ids' and 'attention_mask'
        # the mask is helpful later when processing a batch of input since
        # the tokenized will apply a padding to match the length of each prompt
        # int the batch so attention mask let the model know which token is real vs pad

        # cast to shut mypy from complaining about the multi types return
        tokenized_result: BatchEncoding = cast(
            typ=BatchEncoding,
            val=tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ),
        )

        # Generate the model response
        generated_ids = model.generate(**tokenized_result, max_new_tokens=1024)

        # Strip out the initial prompt from the generated result
        prompt_len = tokenized_result["input_ids"].shape[-1]
        output_ids = generated_ids[0][prompt_len:].tolist()

        # Qwen3's </think> token id is 151668, find it from the end
        # in case the answer content itself contains that literal id somehow
        try:
            think_end_idx: int = len(output_ids) - output_ids[::-1].index(
                151668
            )
        except ValueError:
            think_end_idx = (
                0  # no thinking block found — whole output is the answer
            )

        # thinking_content = tokenizer.decode(
        #     output_ids[:think_end_idx],
        #     skip_special_tokens=True,
        # ).strip()

        answer_text = tokenizer.decode(
            output_ids[think_end_idx:],
            skip_special_tokens=True,
        ).strip()

        # Build the models for later usage... maybe
        assert winning_chunks is not None  # to shut the checker
        sources: list[MinimalSource] = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index,
            )
            for chunk in winning_chunks
        ]

        min_answer: MinimalAnswer = MinimalAnswer(
            question_id=str(question_id),
            question=query,
            retrieved_sources=sources,
            answer=answer_text,
        )

        if save_path is not None:
            save_to_json_file(file_path=save_path, obj=min_answer)

        return min_answer

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        generator_model_name: str = "Qwen/Qwen3-0.6B",
        cache_dir: str | None = None,
        processed_path: str = "data/processed/",
        # question_id: str = "0",
        save_path: str | None = None,
    ) -> StudentSearchResultsAndAnswer:
        """"""

        try:
            # Load search results from dataset search stage
            with open(student_search_results_path, "r") as f:
                data: dict = json.load(f)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Load the model
        model, tokenizer = load_model(
            model_name=generator_model_name, cache_dir=cache_dir
        )
        retriever, chunks = load_retriever(processed_path=processed_path)

        # Build the full obj
        final_boss = StudentSearchResultsAndAnswer(
            search_results=[],
            k=data["k"],
        )

        # Build a pack to be passed so the answer not try
        # to load retrieval and model each time
        cached_pack = {
            "model": model,
            "tokenizer": tokenizer,
        }

        for result in tqdm.tqdm(data["search_results"]):
            winning_chunks = search_one(
                query=result["question"],
                k=data["k"],
                retriever=retriever,
                chunks=chunks,
            )

            answer_obj: MinimalAnswer = self.answer(
                query=result["question"],
                k=data["k"],
                question_id=result["question_id"],
                cached_pack=cached_pack,
                winning_chunks=winning_chunks,
            )

            print(answer_obj)
            final_boss.search_results.append(answer_obj)

        print("all done")
        return final_boss


if __name__ == "__main__":
    fire.Fire(component=Boss)
