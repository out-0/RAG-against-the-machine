import fire
import tqdm
import json
from pathlib import Path
import bm25s

from src.docs_chunking import Chunk, Chunker
from src.documents_loading import Document, load_files
from src.indexer import indexing
from src.search import load_retriever, search_one, search_batch
from src.data_models import MinimalSource, AnsweredQuestion, UnansweredQuestion


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
            print(e)
            exit()

    def search(
        self, query: str, k: int = 1, processed_path: str = "data/processed/"
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

        # Print the formated results
        for chunk in results:
            # Validating
            validated_chunk = MinimalSource(
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
        pass

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
            # Unrechable case
            print("BOOOYAKACHAAH something happen from nowhere")
            exit()

        # Validate the questions AND build a list of questions to be used in retrieving
        batch_questions: list[str] = []
        for q in dataset_data["rag_questions"]:
            ValidatorModel.model_validate(q)
            batch_questions.append(q["question"])

        # Load the indexed files
        retriever, chunks = load_retriever(processed_path)
        batch_results: list[list[Chunk]] = search_batch(
            queries=batch_questions, k=k, retriever=retriever, chunks=chunks
        )

        print(questions_scope)

        # TODO: WRITE THE RESULT OR EMBED IT

        # TODO: CHECK IF THE PATH IS ALREADY EXIST , IF NO, CREATE IT, KEEP IT DYNAMIC WITH FALLBACK


if __name__ == "__main__":
    fire.Fire(component=Boss)
