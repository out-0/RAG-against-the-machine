import os
import re
import subprocess
import tempfile
import uuid
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse

from src.answer_generator import load_model
from src.custom_print import print_green, print_yellow
from src.data_models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.main import Boss
from src.repo_cache import (
    hash_indexable_files,
    load_repo_cache,
    save_repo_cache_entry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES_PATH = PROJECT_ROOT / "data" / "repos"
INDEXES_PATH = PROJECT_ROOT / "data" / "indexes"
REPO_CACHE_PATH = PROJECT_ROOT / "data" / "repo_cache.json"
PROMPT_PAGE_PATH = PROJECT_ROOT / "prompt.html"
ANSWER_PAGE_PATH = PROJECT_ROOT / "answer.html"
INDEX_PAGE = PROJECT_ROOT / "index.html"

app = FastAPI()
boss = Boss()
model, tokenizer = load_model()

# TODO: LATER AFTER ANDSWERING BY LEXICAL, ASK FOR USER SATIFACTION AND PROVIDE
# OPTION FOR USING EMBEDDING MODEL FOR SEMANTIC INDEXING


def _cache_key(owner: str, repository: str) -> str:
    return f"https://github.com/{owner.casefold()}/{repository.casefold()}"


def _cached_index(
    cache: dict[str, object],
    key: str,
) -> tuple[dict[str, str], Path, str] | None:
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None

    files = entry.get("files")
    index_reference = entry.get("index")
    if not isinstance(files, dict) or not isinstance(index_reference, str):
        return None
    if not all(
        isinstance(file_path, str) and isinstance(file_hash, str)
        for file_path, file_hash in files.items()
    ):
        return None

    relative_index_path = Path(index_reference)
    if (
        relative_index_path.is_absolute()
        or ".." in relative_index_path.parts
        or not relative_index_path.parts
    ):
        return None

    index_path = (INDEXES_PATH / relative_index_path).resolve()
    if (
        not index_path.is_relative_to(INDEXES_PATH.resolve())
        or not (index_path / "chunks.pkl").is_file()
    ):
        return None
    return files, index_path, index_reference


def _activate_repository(
    downloaded_path: Path,
    repository_path: Path,
    temporary_root: Path,
) -> Path | None:
    repository_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = temporary_root / "previous-repository"
    had_previous = repository_path.exists() or repository_path.is_symlink()
    if had_previous:
        repository_path.rename(backup_path)

    try:
        downloaded_path.rename(repository_path)
    except OSError as exc:
        if had_previous and backup_path.exists():
            backup_path.rename(repository_path)
        raise HTTPException(
            status_code=500,
            detail="Could not install the downloaded repository locally.",
        ) from exc

    return backup_path if had_previous else None


def _restore_repository(
    repository_path: Path,
    backup_path: Path | None,
    temporary_root: Path,
) -> None:
    if backup_path is None or not backup_path.exists():
        return
    failed_path = temporary_root / "failed-repository"
    if repository_path.exists() or repository_path.is_symlink():
        repository_path.rename(failed_path)
    backup_path.rename(repository_path)


def _build_repository_index(
    repository_path: Path,
    index_path: Path,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.parent.resolve().is_relative_to(INDEXES_PATH.resolve()):
        raise HTTPException(
            status_code=400,
            detail="The index path resolves outside the local index area.",
        )

    with tempfile.TemporaryDirectory(
        prefix=".index-",
        dir=index_path.parent,
    ) as temporary_directory:
        staged_index = Path(temporary_directory) / "index"
        try:
            index_endpoint(
                raw_path=str(repository_path),
                processed_path=str(staged_index),
            )
        except SystemExit as exc:
            raise HTTPException(
                status_code=500,
                detail="The repository was downloaded, but indexing failed.",
            ) from exc
        if not (staged_index / "chunks.pkl").is_file():
            raise HTTPException(
                status_code=500,
                detail="Indexing did not produce the expected chunk data.",
            )
        staged_index.rename(index_path)


@app.get("/", response_class=HTMLResponse)
def home_ui():
    return (INDEX_PAGE).read_text(encoding="utf-8")


@app.post("/process-repo")
def process_repo(repo_link: str = Form(...)) -> HTMLResponse:
    repo_url, owner, repository = _parse_github_repo(repo_link)
    repo_path = REPOSITORIES_PATH / owner.casefold() / repository.casefold()
    REPOSITORIES_PATH.mkdir(parents=True, exist_ok=True)
    INDEXES_PATH.mkdir(parents=True, exist_ok=True)
    if not repo_path.resolve().is_relative_to(REPOSITORIES_PATH.resolve()):
        raise HTTPException(
            status_code=400,
            detail=(
                "The repository path resolves outside the local clone area."
            ),
        )

    try:
        cache = load_repo_cache(REPO_CACHE_PATH)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the repository index cache.",
        ) from exc

    cache_key = _cache_key(owner, repository)
    cache_entry = _cached_index(cache, cache_key)
    with tempfile.TemporaryDirectory(
        prefix=".download-",
        dir=REPOSITORIES_PATH,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        downloaded_path = temporary_root / "repository"
        _clone_repository(repo_url, downloaded_path)
        try:
            file_hashes = hash_indexable_files(downloaded_path)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not hash the downloaded repository files.",
            ) from exc
        if not file_hashes:
            raise HTTPException(
                status_code=422,
                detail="The repository contains no Python or Markdown files.",
            )

        if (
            cache_entry is not None
            and len(cache_entry[0]) == len(file_hashes)
            and cache_entry[0] == file_hashes
        ):
            _activate_repository(
                downloaded_path,
                repo_path,
                temporary_root,
            )
        else:
            backup_path = _activate_repository(
                downloaded_path,
                repo_path,
                temporary_root,
            )
            relative_index_path = (
                Path(owner.casefold())
                / repository.casefold()
                / uuid.uuid4().hex
            )
            index_path = INDEXES_PATH / relative_index_path
            index_ready = False
            try:
                _build_repository_index(repo_path, index_path)
                save_repo_cache_entry(
                    cache_path=REPO_CACHE_PATH,
                    url=cache_key,
                    file_hashes=file_hashes,
                    index_path=relative_index_path.as_posix(),
                )
                index_ready = True
            except (OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Could not save the repository index cache.",
                ) from exc
            finally:
                if not index_ready:
                    _restore_repository(
                        repo_path,
                        backup_path,
                        temporary_root,
                    )

    prompt_page = Template(
        PROMPT_PAGE_PATH.read_text(encoding="utf-8")
    ).substitute(
        owner=escape(owner),
        repository=escape(repository),
    )
    return HTMLResponse(prompt_page)


@app.post("/ask", response_class=HTMLResponse)
def ask_repository(
    query: str = Form(...),
    owner: str = Form(...),
    repository: str = Form(...),
) -> HTMLResponse:
    query = query.strip()
    print_yellow(query)  # Log the query to the server
    if not query:
        raise HTTPException(status_code=422, detail="Enter a question.")
    valid_owner = re.fullmatch(r"[A-Za-z0-9_-]+", owner) is not None
    valid_repository = re.fullmatch(r"[A-Za-z0-9_.-]+", repository) is not None
    if not valid_owner or not valid_repository or repository in {".", ".."}:
        raise HTTPException(
            status_code=422,
            detail="The repository identifier is invalid.",
        )

    try:
        cache = load_repo_cache(REPO_CACHE_PATH)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the repository index cache.",
        ) from exc
    cache_entry = _cached_index(cache, _cache_key(owner, repository))
    if cache_entry is None:
        raise HTTPException(
            status_code=404,
            detail="The repository index could not be found.",
        )
    processed_path = cache_entry[1]
    if not processed_path.resolve().is_relative_to(INDEXES_PATH.resolve()):
        raise HTTPException(
            status_code=400,
            detail="The index path resolves outside the local index area.",
        )

    try:
        result = answer_endpoint(
            query=query,
            processed_path=str(processed_path),
        )
    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not generate an answer for this question.",
        ) from exc

    sources = "".join(
        (
            "<li><code>"
            f"{escape(source.file_path)}"
            "</code><span>"
            f"Characters {source.first_character_index}–"
            f"{source.last_character_index}"
            "</span></li>"
        )
        for source in result.retrieved_sources
    )
    if not sources:
        sources = '<li class="empty">No source files were returned.</li>'
    answer_page = Template(ANSWER_PAGE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        answer_page.substitute(
            owner=escape(owner),
            repository=escape(repository),
            query=escape(result.question),
            answer=escape(result.answer),
            sources=sources,
        )
    )


def _parse_github_repo(repo_link: str) -> tuple[str, str, str]:
    """Return the canonical URL, owner, and repository for a GitHub URL."""
    try:
        parsed = urlparse(repo_link.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Enter a valid HTTPS GitHub repository URL.",
        ) from exc

    match = re.fullmatch(
        r"/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?",
        parsed.path,
    )
    if (
        parsed.scheme != "https"
        or hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise HTTPException(
            status_code=422,
            detail="Enter an HTTPS GitHub repository URL, such as "
            "https://github.com/owner/repository.",
        )

    owner, repository = match.groups()
    if repository in {".", ".."}:
        raise HTTPException(
            status_code=422,
            detail="The GitHub repository URL is invalid.",
        )
    repository = repository.removesuffix(".git")
    if not repository:
        raise HTTPException(
            status_code=422,
            detail="The GitHub repository URL is invalid.",
        )
    canonical_url = f"https://github.com/{owner}/{repository}.git"
    return canonical_url, owner, repository


def _clone_repository(repo_url: str, destination: Path) -> None:
    """Clone a public GitHub repository without invoking a shell."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--", repo_url, str(destination)],
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Cloning the repository timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start Git to clone the repository.",
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip()[-1000:]
        raise HTTPException(
            status_code=502,
            detail=f"Could not clone the repository: {message}",
        )
    if not (destination / ".git").is_dir():
        raise HTTPException(
            status_code=502,
            detail="Git finished without creating a valid repository.",
        )


@app.get("/usage")
def home() -> dict:
    """Endpoint action for home operation"""
    message: dict = {
        "message": "Hello, you can use the next entrypoints",
        "entries": {
            "/index": {
                "method": "POST",
                "parameters": [
                    "max_chunk_size",
                    "raw_path",
                    "processed_path",
                    "embedding_model_name",
                    "use_embedding",
                ],
            },
            "/search": {
                "method": "GET",
                "parameters": [
                    "query",
                    "k",
                    "processed_path",
                    "question_id",
                    "use_hybrid",
                    "use_embedding",
                ],
            },
            "/search_dataset": {
                "method": "POST",
                "parameters": [
                    "dataset_path",
                    "k",
                    "save_directory",
                    "processed_path",
                    "save_file",
                ],
            },
            "/answer": {
                "method": "POST",
                "parameters": [
                    "query",
                    "k",
                    "generator_model_name",
                    "cache_dir",
                    "processed_path",
                    "question_id",
                    "save_path",
                ],
            },
            "/answer_dataset": {
                "method": "POST",
                "parameters": [
                    "student_search_results_path",
                    "save_directory",
                    "generator_model_name",
                    "cache_dir",
                    "processed_path",
                    "save_path",
                ],
            },
            "/evaluate": {
                "method": "POST",
                "parameters": [
                    "student_search_results_path",
                    "dataset_path",
                    "k",
                ],
            },
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
) -> dict:
    """Endpoint action for index operation"""
    boss.index(
        max_chunk_size=max_chunk_size,
        raw_path=raw_path,
        processed_path=processed_path,
        embedding_model_name=embedding_model_name,
        use_embedding=use_embedding,
    )
    return {"Status": "Index Success"}


@app.get("/search")
def search_endpoint(
    query: str,
    k: int = 1,
    processed_path: str = "data/processed/",
    question_id: str = "0",
    use_hybrid: bool = False,
    use_embedding: bool = False,
) -> MinimalSearchResults | list[str]:
    """Endpoint action for search operation"""
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
    save_directory: str = "data/output/search_results/UnansweredQuestions",
    processed_path: str = "data/processed/",
    save_file: str | None = None,
) -> dict:
    """Endpoint action for search operation"""
    boss.search_dataset(
        dataset_path=dataset_path,
        k=k,
        save_directory=save_directory,
        processed_path=processed_path,
        save_file=save_file,
    )

    return {"Status": f"Search Success, Saved Under {save_directory}"}


@app.post("/answer")
def answer_endpoint(
    query: str,
    k: int = 10,
    generator_model_name: str = "Qwen/Qwen3-0.6B",
    cache_dir: str | None = None,
    processed_path: str = "data/processed/",
    question_id: str | int = "0",
    save_path: str | None = None,
) -> MinimalAnswer:
    """Endpoint action for answer operation"""
    return boss.answer(
        model=model,
        tokenizer=tokenizer,
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
) -> StudentSearchResultsAndAnswer:
    """Endpoint action for answer operation"""
    return boss.answer_dataset(
        student_search_results_path=student_search_results_path,
        save_directory=save_directory,
        generator_model_name=generator_model_name,
        cache_dir=cache_dir,
        processed_path=processed_path,
    )


@app.post("/evaluate")
def evaluate_endpoint(
    student_search_results_path: str,
    dataset_path: str,
    k: int,
) -> dict:
    """Endpoint action for evaluate operation"""
    result: dict = boss.evaluate(
        student_search_results_path=student_search_results_path,
        dataset_path=dataset_path,
        k=k,
    )
    return result
