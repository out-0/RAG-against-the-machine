# TODO: IMPLEMENT DOCSTRINGS LATER
import fire
import json
from pathlib import Path
import bm25s

from src.data_models import RagDataset
from src.docs_chunking import Chunk, Chunker
from src.documents_loading import Document, load_files
from src.indexer import indexing
from src.search import load_retriever, search_one, search_batch
from src.data_models import (
    MinimalSource,
    AnsweredQuestion,
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
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
            indexing(chunks=chunks, processed_path=processed_path, method=method)
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            exit()

    def search(
        self,
        query: str,
        k: int = 1,
        processed_path: str = "data/processed/",
    ) -> None:
        """"""

        # Check if the index exist to be loaded ALSO pickled chunks
        path: Path = Path(processed_path)
        if not path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            exit()

        # Load the retrever AND Chunks that was pickled
        retriever: bm25s.BM25
        chunks: list[Chunk]
        retriever, chunks = load_retriever(processed_path)
        # Retrieve results for one question
        results: list[Chunk] = search_one(
            query=query, k=k, retriever=retriever, chunks=chunks
        )

        # Validate each chunk by a MinimalSource model and print the formate required
        for chunk in results:
            # Validating chunk
            validated_chunk: MinimalSource = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index,
            )
            print(
                f"{validated_chunk.file_path} [{validated_chunk.first_character_index}:{validated_chunk.last_character_index}]"
            )
            # print(f"{winner.content[winner.start_index : winner.end_index]}")

    def search_dataset(
        self,
        dataset_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        k: int = 1,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        processed_path: str = "data/processed/",
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
            exit()
        except Exception as e:
            print(f"Error: {e}")
            exit()

        # Check if the index exist to be loaded
        p_path: Path = Path(processed_path)
        if not p_path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            exit()

        # Check the scope of the questions to determine the output path
        d_path: Path = Path(dataset_path)
        questions_scope: str
        if "AnsweredQuestions" in d_path.parts:
            questions_scope = "AnsweredQuestions"
        elif "UnansweredQuestions" in d_path.parts:
            questions_scope = "UnansweredQuestions"
        else:
            print(f"Cannot determine dataset scope from path: {dataset_path}")
            exit()

        # Get validator model based on the scope
        ValidatorModel: type[AnsweredQuestion | UnansweredQuestion]
        if questions_scope == "AnsweredQuestions":
            ValidatorModel = AnsweredQuestion
        elif questions_scope == "UnansweredQuestions":
            ValidatorModel = UnansweredQuestion
        else:
            # Unreachable case, save to remove
            print("BOYAAA: SOMETHING HAPPEN FROM NOWHERE, GOOD LUCK FIGURE IT OUT")
            exit()

        # Validate each question AND build a list of questions to be used in retrieving
        batch_questions: list[str] = []
        validated_batch: list[ValidatorModel] = []
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
        for i, question_result in enumerate(batch_results):
            mini_search_result: MinimalSearchResults = MinimalSearchResults(
                question_id=validated_batch[i].question_id,
                question=validated_batch[i].question,
                retrieved_sources=[],
            )
            # Collect the validated chunks [MinimalSource]
            # Validate the chunks retrieved as result
            for chunk in question_result:
                mini_search_result.retrieved_sources.append(
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.start_index,
                        last_character_index=chunk.end_index,
                    )
                )
            boss_search_result.search_results.append(mini_search_result)

        print(boss_search_result)
        print(type(boss_search_result))

        # TODO: WRITE THE RESULT OR EMBED IT

        # TODO: CHECK IF THE PATH IS ALREADY EXIST , IF NO, CREATE IT, KEEP IT DYNAMIC WITH FALLBACK


if __name__ == "__main__":
    fire.Fire(component=Boss)
