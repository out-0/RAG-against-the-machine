from fastapi import FastAPI

from src.data_models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.main import Boss

app = FastAPI()
boss = Boss()


@app.get("/")
def SayHello() -> dict:
    # TODO: COMPLETING THIS MESSAGE
    message: dict = {
        "message": "Hello, you can use the next entrypoints",
        "entries": {
            "/index": {"method": "POST", "parameters": ["max_chunk_size", "raw_path", "processed_path", "embedding_model_name", "use_embedding"]},
            "/search": {"method": "GET", "parameters": ["query", "k", "processed_path", "question_id", "use_hybrid", "use_embedding"]},
            "/search_dataset": {"method": "POST", "parameters": ["dataset_path", "k", "save_directory", "processed_path", "save_file"]},
            "/answer": {"method": "POST", "parameters": ["query", "k", "generator_model_name", "cache_dir", "processed_path", "question_id", "save_path"]},
            "/answer_dataset": {"method": "POST", "parameters": ["student_search_results_path", "save_directory", "generator_model_name", "cache_dir", "processed_path", "save_path"]},
            "/evaluate": {"method": "POST", "parameters": ["student_search_results_path", "dataset_path", "k"]},
        },
    }
    return message


@app.post("/index")
def index_endpoint(
    max_chunk_size: int = 2000,
    raw_path: str = "data/raw/vllm-0.10.1",
    processed_path: str = "data/processed/",
    embedding_model_name: str | None = "all-MiniLM-L6-v2",
    use_embedding: bool = False,
) -> None:
    boss.index(
        max_chunk_size=max_chunk_size,
        raw_path=raw_path,
        processed_path=processed_path,
        embedding_model_name=embedding_model_name,
        use_embedding=use_embedding,
    )


@app.get("/search")
def search_endpoint(
    query: str,
    k: int = 1,
    processed_path: str = "data/processed/",
    question_id: str = "0",
    use_hybrid: bool = False,
    use_embedding: bool = False,
) -> MinimalSearchResults | list[str]:
    """"""
    return boss.search(
        query=query,
        k=k,
        processed_path=processed_path,
        question_id=question_id,
        use_hybrid=use_hybrid,
        use_embedding=use_embedding,
    )


@app.post("/search_dataset")
def search_dataset_endpoint(
    dataset_path: str,
    k: int = 1,
    save_directory: str = "data/output/search_results/",
    processed_path: str = "data/processed/",
    save_file: str | None = None,
) -> None:
    boss.search_dataset(
        dataset_path=dataset_path,
        k=k,
        save_directory=save_directory,
        processed_path=processed_path,
        save_file=save_file,
    )


@app.post("/answer")
def answer_endpoint(
    query: str,
    k: int = 1,
    generator_model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: str | None = None,
    processed_path: str = "data/processed/",
    question_id: str | int = "0",
    save_path: str | None = None,
) -> MinimalAnswer:
    return boss.answer(
        query=query,
        k=k,
        generator_model_name=generator_model_name,
        cache_dir=cache_dir,
        processed_path=processed_path,
        question_id=question_id,
        save_path=save_path,
    )


@app.post("/answer_dataset")
def answer_dataset_endpoint(
    student_search_results_path: str,
    save_directory: str,
    generator_model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: str | None = None,
    processed_path: str = "data/processed/",
    save_path: str | None = None,
) -> StudentSearchResultsAndAnswer:
    return boss.answer_dataset(
        student_search_results_path=student_search_results_path,
        save_directory=save_directory,
        generator_model_name=generator_model_name,
        cache_dir=cache_dir,
        processed_path=processed_path,
        save_path=save_path,
    )


@app.post("/evaluate")
def evaluate_endpoint(
    student_search_results_path: str,
    dataset_path: str,
    k: int,
) -> None:
    boss.evaluate(
        student_search_results_path=student_search_results_path,
        dataset_path=dataset_path,
        k=k,
    )

