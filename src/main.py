import json
import sys
from pathlib import Path

import fire
import tqdm

from src.answer_generator import get_chat_template, load_model
from src.custom_print import print_green, print_red, print_yellow
from src.data_models import (
    AnsweredQuestion,
    CachedResources,
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.docs_chunking import Chunker
from src.docs_indexing import indexing
from src.docs_loading import Document, load_files
from src.incremental_indexing import handle_incremental_indexing
from src.recall import rag_recall_at_k
from src.search import (
    load_retriever,
    save_to_json_file,
    search_batch,
    search_one,
)


class Boss:
    """
    This is the main class of the program
    """

    def usage(self) -> None:
        """A quick usage guide"""

        print_green("""
        ██╗   ██╗███████╗ █████╗  ██████╗ ███████╗
        ██║   ██║██╔════╝██╔══██╗██╔════╝ ██╔════╝
        ██║   ██║███████╗███████║██║  ███╗█████╗
        ██║   ██║╚════██║██╔══██║██║   ██║██╔══╝
        ╚██████╔╝███████║██║  ██║╚██████╔╝███████╗
         ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
        """)

        print("""
        uv run python -m src.main search
          --query "How to configure OpenAi server?"
          --k 5
          --processed_path data/processed/
          --use_hybrid (optional)
          --use_semantic (optional)

        ###### Arguments:

        - `--query`: The query to search for.
        - `--k`: The number of top results to return.
        - `--processed_path`: The path to read index from (index).
        - `--use_hybrid`: Use hybrid retrieval (BM25 + semantic).
        - `--use_semantic`: Use semantic retrieval (FAISS+sentence embeddings)

        retun a list of chunks that are relevant to the query

        #### Running a batch search

        uv run python -m src.main search_dataset
          --dataset_path dataset_path.json
          --save_directory data/output/search_results
          --processed_path data/processed/
          --k 10

        ###### Arguments:

        - `--dataset_path`: The path to the dataset file.
        - `--save_directory`: The directory to save the search results.
        - `--processed_path`: The path to read index from (index).
        - `--k`: The number of top results to return for each question.
        - `--use_hybrid`: Use hybrid retrieval (BM25 + semantic).
        - `--use_semantic`: Use semantic retrieval
                            (FAISS + sentence embeddings)

        retun a list of lists of chunks that are relevant to the query

        #### Running answer and answer_dataset

        uv run python -m src.main answer
          --query "How to configure OpenAi server?"
          --k 5
          --generator_model Qwen3-0.6B
          --cache_dir ... # optional to override default huggingface cache dir
          --processed_path data/processed/
          --

        uv run python -m src.main answer_dataset
          --student_search_results_path {path to search results}
          --save_directory {path to save answers}

        ###### Arguments:

        - `student_search_results_path`
        - `save_directory`
        - `generator_model_name = "Qwen/Qwen3-0.6B"`
        - `cache_dir`
        - `processed_path`
        """)

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_path: str = "data/raw/vllm-0.10.1",
        processed_path: str = "data/processed/",
        embedding_model_name: str | None = "all-MiniLM-L6-v2",
        use_embedding: bool = False,
        incremental: bool = False,
    ) -> None:
        """Indexing operation that create a special map that link each
        word to the relevant docs for it

        Args:
            - max_chunk_size (int): The maximum size of a chunk
            - raw_path (str): The path to the raw files
            - processed_path (str): The path to save the indexed lookup
            - embedding_model_name (str): The name of the embeddings model
            - use_embedding (bool): Whether to use embeddings
            - incremental (bool): Whether to use incremental indexing

        Returns:
            - None
        """
        print_green("""
    ██╗███╗   ██╗██████╗ ███████╗██╗  ██╗
    ██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝
    ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝
    ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗
    ██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗
    ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
        """)

        try:
            # if not raw_path:
            #     raise ValueError(
            #         "Please provide a valid path to the raw files"
            #     )

            # Load the files into program
            docs: list[Document] = load_files(input_path=raw_path)

            if not docs:
                raise ValueError(
                    "Warning: No files found\n"
                    "Make sure the raw files path is correct"
                )

            # Handle incremental indexing if enabled
            if incremental:
                handle_incremental_indexing(
                    docs=docs,
                    processed_path=processed_path,
                    max_chunk_size=max_chunk_size,
                    use_embedding=use_embedding,
                    embeddings_model_name=embedding_model_name,
                )
                return

            # Build the chunker and start processing the files
            chunker: Chunker = Chunker(files=docs, max_size=max_chunk_size)
            chunks: list[Chunk] = chunker.process_files()

            # Run the main index processing
            indexing(
                chunks=chunks,
                processed_path=processed_path,
                use_embedding=use_embedding,
                embeddings_model_name=embedding_model_name,
            )
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            sys.exit()

    def search(
        self,
        query: str,
        k: int = 1,
        processed_path: str = "data/processed/",
        question_id: str = "0",  # just optional
        use_hybrid: bool = False,
        use_embedding: bool = False,
    ) -> MinimalSearchResults | list[str]:
        """Search operation that return a list of chunks
        that are relevant to the query

        Args:
            - query (str): The query to search for
            - k (int): The number of top results to return
            - processed_path (str): The path to the processed files
            - question_id (str): The id of the question
            - use_hybrid (bool): Use hybrid search or not
            - use_embedding (bool): Use embedding search or not

        Returns:
            - A list of chunks that are relevant to the query
        """

        print_green("""
    ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
    ███████╗█████╗  ███████║██████╔╝██║     ███████║
    ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
    ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        """)

        # Check if the index exist to be loaded ALSO pickled chunks
        path: Path = Path(processed_path)
        if not path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            sys.exit()

        # << Load the retriever AND Chunks that was pickled >>
        retriever, chunks = load_retriever(processed_path)
        # Retrieve results for one question
        try:
            retrieve_results: list[Chunk] = search_one(
                query=query,
                k=k,
                retriever=retriever,
                chunks=chunks,
                use_hybrid=use_hybrid,
                processed_path=processed_path,
                use_embedding=use_embedding,
            )
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            sys.exit()

        # Build Minimal search result to be returned and used later in answers
        min_search_result: MinimalSearchResults = MinimalSearchResults(
            question_id=question_id,
            question=query,
            question_str=query,
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

        # Return the path of the sources to match the demo format
        return sources_path_results

    def search_dataset(
        self,
        dataset_path: str | None = None,
        k: int = 1,
        save_directory: str | None = None,
        processed_path: str = "data/processed/",
        save_file: str | None = None,
        use_hybrid: bool = False,
        use_embedding: bool = False,
    ) -> None:
        """
        Search a dataset of questions and save the results

        Args:
            - dataset_path: path to the dataset
            - k: number of chunks to be retrieved for each question
            - save_directory: directory to save the results
            - processed_path: path to the processed files
            - save_file: name of the file to save the results
            - use_hybrid: use hybrid search or not
            - use_embedding: use embedding search or not

        Returns:
            - None
        """

        print_green("""
    ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
    ███████╗█████╗  ███████║██████╔╝██║     ███████║
    ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
    ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        """)

        # Some checks for values, even if fire interface can do the job
        try:
            if dataset_path is None:
                raise TypeError(
                    "Error: Dataset path is required with flag --dataset_path"
                )
            if save_directory is None:
                raise TypeError(
                    "Error: "
                    "save directory is required with flag --save_directory"
                )
            # Read the json dataset
            with open(dataset_path, "r") as f:
                dataset_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: dataset file not found: {dataset_path}")
            sys.exit(1)
        except TypeError as e:
            print(e)
            sys.exit(1)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Check if the index exist to be loaded
        p_path: Path = Path(processed_path)
        if not p_path.is_dir():
            print(
                "Error: no indexed files exist"
                "Make sure to run indexer first or check the processed path"
            )
            sys.exit(1)

        # Validate each question AND build a list of questions
        # to be used in retrieving
        batch_questions: list[str] = []
        validated_batch: list[AnsweredQuestion | UnansweredQuestion] = []
        for q in dataset_data["rag_questions"]:
            validated_batch.append(UnansweredQuestion.model_validate(q))
            batch_questions.append(q["question"])

        try:
            # Now Validate the full questions batch (Rag dataset)
            _: RagDataset = RagDataset(rag_questions=validated_batch)

            # Load the indexed files
            retriever, chunks = load_retriever(processed_path)
            # Retrieve the results for the batch of questions,
            # this will return a list of list of chunks
            batch_results: list[list[Chunk]] = search_batch(
                queries=batch_questions,
                k=k,
                retriever=retriever,
                chunks=chunks,
                processed_path=processed_path,
                use_hybrid=use_hybrid,
                use_embedding=use_embedding,
            )
        except Exception as e:
            print_red(e)
            sys.exit(1)

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
                # Add just cause moulinette required that field name
                # But provided model use first one
                question_str=validated_batch[i].question,
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

        # Create paths if not exist and save the json result
        dataset_path_p: Path = Path(dataset_path)
        dir_path: Path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        save_file_name = save_file if save_file else dataset_path_p.parts[-1]
        full_path: Path = dir_path / save_file_name
        save_to_json_file(file_path=full_path, obj=boss_search_result)
        print_green(f"Saved student_search_results to {full_path}")

    def answer(
        self,
        query: str,
        k: int = 1,
        generator_model_name: str = "Qwen/Qwen3-0.6B",
        cache_dir: str | None = None,
        processed_path: str = "data/processed/",
        question_id: str | int = "0",
        save_path: str | None = None,
        # This is just helpful for answer_dataset
        cached_resources: CachedResources | None = None,
        winning_chunks: list[Chunk] | None = None,
    ) -> MinimalAnswer:
        """Answer a single query using the retrieved context.

        Args:
            - query (str): The query to answer.
            - k (int): The number of chunks to retrieve.
            - generator_model_name (str): The name of the generator model.
            - cache_dir (str | None): The cache directory.
            - processed_path (str): The path to the processed files.
            - question_id (str | int): The question id.
            - save_path (str | None): The path to save the answer.
            - cached_resources (CachedResources | None): The cached resources.
            - winning_chunks (list[Chunk] | None): The winning chunks.

        Returns:
            - MinimalAnswer: The answer.
        """

        print_green("""
     █████╗ ███╗   ██╗███████╗██╗    ██╗███████╗██████╗
    ██╔══██╗████╗  ██║██╔════╝██║    ██║██╔════╝██╔══██╗
    ███████║██╔██╗ ██║███████╗██║ █╗ ██║█████╗  ██████╔╝
    ██╔══██║██║╚██╗██║╚════██║██║███╗██║██╔══╝  ██╔══██╗
    ██║  ██║██║ ╚████║███████║╚███╔███╔╝███████╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
        """)

        # check if its called normally
        if cached_resources is None:
            try:
                # Load retriever and chunks
                retriever, chunks = load_retriever(
                    processed_path=processed_path
                )

                # Retrieve the winning chunks to build the response
                winning_chunks = search_one(
                    query=query,
                    k=k,
                    retriever=retriever,
                    chunks=chunks,
                    processed_path=processed_path,
                )

                # Load model
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
            model = cached_resources.model
            tokenizer = cached_resources.tokenizer

        try:
            messages: list[dict[str, str]] = get_chat_template(
                chunks=winning_chunks,
                query=query,
            )

            # tokenized_result currently consist of 'input_ids' and
            # 'attention_mask' the mask is helpful later when processing a
            # batch of input since the tokenized will apply a padding to
            # match the length of each prompt int the batch so attention
            # mask let the model know which token is real vs pad
            tokenized_result = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        except Exception as e:
            print(e)
            sys.exit(1)

        # Generate the model response
        generated_ids = model.generate(
            **tokenized_result,
            max_new_tokens=1024,
        )

        # Strip out the initial prompt from the generated result
        # caues result hold initial prompt also which not needed.
        prompt_len = tokenized_result["input_ids"].shape[
            -1
        ]  # -1 hold sequence length
        output_ids = generated_ids[0][prompt_len:].tolist()

        # Qwen3's </think> token id is 151668, find it from the end
        # in case the answer content itself contains that literal id somehow
        try:
            think_end_idx: int = len(output_ids) - output_ids[::-1].index(
                151668
            )
        except ValueError:
            # no thinking block found — whole output is the answer
            think_end_idx = 0

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
            question_str=query,
            retrieved_sources=sources,
            answer=answer_text,
        )

        if save_path is not None:
            save_to_json_file(file_path=save_path, obj=min_answer)

        print_green(min_answer.answer)
        return min_answer

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        generator_model_name: str = "Qwen/Qwen3-0.6B",
        cache_dir: str | None = None,
        processed_path: str = "data/processed/",
    ) -> StudentSearchResultsAndAnswer:
        """

        Args:
            student_search_results_path: The path to student's search results.
            save_directory (str): The directory to save the answers.
            generator_model_name: The name of the generator model.
            cache_dir: The cache directory. Defaults to None.
            processed_path: The path to the processed data.

        Returns:
            StudentSearchResultsAndAnswer
        """

        print_green("""
     █████╗ ███╗   ██╗███████╗██╗    ██╗███████╗██████╗
    ██╔══██╗████╗  ██║██╔════╝██║    ██║██╔════╝██╔══██╗
    ███████║██╔██╗ ██║███████╗██║ █╗ ██║█████╗  ██████╔╝
    ██╔══██║██║╚██╗██║╚════██║██║███╗██║██╔══╝  ██╔══██╗
    ██║  ██║██║ ╚████║███████║╚███╔███╔╝███████╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
        """)

        try:
            # Load search results from dataset search stage
            with open(student_search_results_path, "r") as f:
                data: dict = json.load(f)

            print(f"Loaded {len(data['search_results'])} questions")
        except Exception as e:
            print_red(e)
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
        cached_resources: CachedResources = CachedResources(
            model=model, tokenizer=tokenizer
        )

        for result in tqdm.tqdm(data["search_results"]):
            # Get the winning chunks for the current question
            # To be used in answer below
            # note: its better to note reseach here and just relay
            # on search operation but the result from Minimal source
            # in search_dataset is just holding the file_path,
            # and index's, so its either reconstruct the chunk based
            # on its metadata or just research which the one i went
            # with, there is other option like shove the chunk also
            # as metadata but its will require some quick changes
            # which probably will introduce some new issue, also am lazy.
            winning_chunks = search_one(
                query=result["question"],
                k=data["k"],
                retriever=retriever,
                chunks=chunks,
                processed_path=processed_path,
            )

            # Generate the answer for the current question using
            # the winning chunks
            answer_obj: MinimalAnswer = self.answer(
                query=result["question"],
                k=data["k"],
                question_id=result["question_id"],
                cached_resources=cached_resources,
                winning_chunks=winning_chunks,
            )

            final_boss.search_results.append(answer_obj)

        # Save the result
        search_result_path = Path(student_search_results_path)
        file_name = search_result_path.name
        save_path = Path(save_directory) / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_json_file(file_path=save_path, obj=final_boss)
        print_green(
            f"Saved student_search_results_and_answer "
            f"to {save_directory}{file_name}"
        )

        return final_boss

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int,
    ) -> dict[str, str]:
        """
        This is query level formula:
        Recall@k = (Number of queries with target in top-k) / (Total queries)
        """

        print_green(r"""
    ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗
    ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║
    ██████╔╝█████╗  ██║     ███████║██║     ██║
    ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║
    ██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗
    ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝
    """)

        # Some checks
        try:
            k = int(k)
            if k <= 0:
                raise ValueError
            print("Loading student results...")
            with open(student_search_results_path, "r") as f:
                student_results = json.load(f)
            print("Student data is Valid: True")
            print("Loading Dataset...")
            with open(dataset_path, "r") as df:
                dataset_data = json.load(df)
            print("Dataset loaded")

        except json.decoder.JSONDecodeError as e:
            print("Error: Malformed Json file:")
            print(e)
            sys.exit(1)
        except ValueError as e:
            print("Error: Stop bullshiting and use correct non negative value")
            print(e)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(e)
            sys.exit(1)

        try:
            # Extract the list of student results
            questions_results: list = student_results["search_results"]
            # Extract the truth sources for each question
            questions_truth: list = dataset_data["rag_questions"]

            if len(questions_results) != len(questions_truth):
                print(
                    "Warning: Expecting a match between questions "
                    "count withing two data provided"
                )
                return {"Error": "Internal Data Mismatch"}
        except KeyError as e:
            print(e)
            sys.exit(1)

        all_recall_scores: list[float] = []

        for i in tqdm.tqdm(range(len(questions_results))):
            # Extract the correct sources that we evaluate again
            question_truth_sources: list[dict[str, str | int]] = (
                questions_truth[i]["sources"]
            )

            # Now extract the student retrieved sources for current question
            student_magic_result: list[dict[str, str | int]] = (
                questions_results[i]["retrieved_sources"]
            )

            # Get recall score for current question
            single_recall_result: float = rag_recall_at_k(
                retrieved_results=student_magic_result,
                ground_truths=question_truth_sources,
                k=k,
                iou_threshold=0.05,
            )

            all_recall_scores.append(single_recall_result)

        print(f"Total number of questions: {len(questions_truth)}")
        print(
            "Total number of questions "
            f"with student sources: {len(questions_results)}"
        )

        print("Evaluation Results...")
        print(f"questions evaluated: {len(questions_results)}")

        final_result: float = sum(all_recall_scores) / len(all_recall_scores)
        print(f"Recall@{k}: {final_result:.3f} ({final_result * 100:.2f}%)")

        print_yellow("""
    ⠀⠀⢀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢀⠀⣿⡂⢹⡇⠀⠀⣰⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⡇⢸⣇⢸⣇⠀⢀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢾⠀⠀⣯⡀⡆⠀⠀
    ⢸⣷⢸⣇⣸⣇⠀⣾⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣠⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⣂⠀⣿⡄⢸⡀⣤
    ⢠⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣊⡝⠛⠙⠂⠄⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣦⣼⣷⣼⣁⠼
    ⢸⣿⣿⣿⣿⣿⣿⣀⢀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⡻⣥⢋⡔⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣂⣜⣿⡟⢿⣿⣿⣄
    ⠈⣿⣿⣿⣿⣿⣿⣿⠿⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣷⢯⣿⣾⡔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢪⣷⣿⢿⣿⣿
    ⠀⣿⣿⣟⢿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡟⠛⠉⡉⢸⡉⠁⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢢⣽⣗⣿⠇
    ⠀⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠺⣿⡇⣤⡤⢔⡿⣇⠀⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣯⠀
    ⠘⡟⣛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡇⣿⣿⠗⡲⠏⠟⠿⠀⠈⠓⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠍⠁⠁⠀
    ⠃⡜⡠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣼⣿⡟⢡⡿⠿⠷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⠒⠂⠂
    ⠐⢐⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠸⣡⢶⣿⣟⡃⠀⠘⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡇⠀⡀⠀
    ⢠⡏⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡰⢨⠣⠉⠉⠋⠉⠀⠀⠀⠀⢈⠀⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡿⠀⠀⠀⠀
    ⢺⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣽⡿⢛⢭⠏⣢⠍⠈⠖⠀⠀⠒⣶⢦⡁⠂⠀⠀⠀⠀⠀⠯⠤⣤⣴⢶⣍⠝⣯⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⢌⣿⠱⠀⠀⠀⠀⠀
    ⣯⣯⠸⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠄⠀⠈⠀⠁⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠏⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠧⠍⠶⠤⠈⣆⠀⠀⠀⠀⠀⠀⠀⣷⡻⠀⣼⠀⠀⠀
    ⣯⣨⡀⢀⡠⠤⣐⠤⣀⣰⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠑⠐⠐⠢⠺⠥⡾⠉⡠⠀⠀⠀
    ⠋⠙⠈⠉⠉⠁⠈⠈⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠓⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠇⣣⡁⢶⣠⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢶⠀⡶⣲⠀⣆⡒⣰⠒⢦⢰⠀⢰⡆⣴⠐⣶⠒⣐⣒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣺⣿⣿⣿⠛
    ⠀⠀⠑⢌⠻⣗⣔⠉⡅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠞⠚⠃⠻⠴⠃⠦⠝⠘⠤⠎⠸⠤⠘⠧⠞⠀⠛⠀⠰⠤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡟⣾⣿⣿⣿⠃⠀
    ⠀⠀⠀⠀⠉⠢⠁⠀⠀⠀⠀⢀⣤⣤⣤⣄⠀⠀⢠⣤⠀⠀⣤⣄⠀⠀⠀⣤⣤⠀⢠⣤⣤⣤⣤⣤⡄⢠⣤⣄⠀⠀⠀⠀⣤⣤⡄⠀⠀⠀⢠⣤⡄⠀⠀⠀⢘⡮⡝⣿⣿⡿⢆⠁⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⠏⠉⠉⢿⣷⠀⢸⣿⠀⠠⣿⣿⣧⡀⠀⣿⣿⠀⢸⣿⡏⠉⠉⠉⠁⢼⣿⣿⡄⠀⠀⢸⡿⣿⡇⠀⠀⢀⣿⢻⣷⠀⠀⠀⠞⡜⣹⣿⣿⡙⢆⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⠀⢸⣿⠀⠐⣿⡯⢻⣷⡀⣿⣿⠀⢸⣿⣷⣶⣶⡆⠀⢺⣿⠹⣿⡀⢠⣿⠃⣿⡇⠀⠀⣾⡟⠀⢿⣧⠀⠀⠀⠠⢽⣿⣯⡙⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⡀⠀⠀⣠⣤⠀⢸⣿⠀⢈⣿⡧⠀⠹⣿⣿⣿⠀⢸⣿⡇⠀⠀⠀⠀⢸⣿⡄⢻⣧⣾⡏⢠⣿⡇⠀⣼⣿⣷⣶⣾⣿⣇⠀⠀⠀⠘⣿⢣⠜⠁⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣶⣾⣿⠏⠀⢸⣿⠀⠀⣿⡷⠀⠀⠹⣿⣿⠀⢸⣿⣿⣿⣿⣿⡆⢸⣿⡆⠀⢿⡿⠀⢰⣿⡇⢀⣿⡏⠀⠀⠀⢹⣿⡀⠀⠀⠀⠀⠈⡆⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀⠀⠀⠈⠉⠀⠀⠉⠁⠀⠀⠀⠉⠉⠀⠈⠉⠉⠈⠉⠉⠁⠈⠉⠀⠀⠈⠁⠀⠀⠉⠁⠈⠉⠀⠀⠀⠀⠈⠉⠁⠐⡀⠀⠀⠀⠀⠀⠀⠀
""")
        return {
            f"Recall@{k}": f"{final_result:.3f} ({final_result * 100:.2f}%)"
        }


def main() -> None:
    """
    The main entry point for the application.
    """
    fire.Fire(component=Boss)
