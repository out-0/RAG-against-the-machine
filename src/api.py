import os
import re
import subprocess
import tempfile
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse

from src.answer_generator import load_model
from src.data_models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.main import Boss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES_PATH = PROJECT_ROOT / "data" / "indexes"
PROMPT_PAGE_PATH = PROJECT_ROOT / "prompt.html"
ANSWER_PAGE_PATH = PROJECT_ROOT / "answer.html"
INDEX_PAGE = PROJECT_ROOT / "index.html"

app = FastAPI()
boss = Boss()
model, tokenizer = load_model()

# TODO: LATER AFTER ANDSWERING BY LEXICAL, ASK FOR USER SATIFACTION AND PROVIDE
# OPTION FOR USING EMBEDDING MODEL FOR SEMANTIC INDEXING


@app.get("/", response_class=HTMLResponse)
def home_ui():
    return (INDEX_PAGE).read_text(encoding="utf-8")


@app.post("/process-repo")
def process_repo(repo_link: str = Form(...)):
    repo_url, owner, repository = _parse_github_repo(repo_link)
    repo_path = REPOSITORIES_PATH / owner / repository
    resolved_repo_root = REPOSITORIES_PATH.resolve()
    if not repo_path.resolve().is_relative_to(resolved_repo_root):
        raise HTTPException(
            status_code=400,
            detail=(
                "The repository path resolves outside the local clone area."
            ),
        )

    if repo_path.exists():
        if not (repo_path / ".git").is_dir():
            raise HTTPException(
                status_code=409,
                detail="A non-repository already exists at the clone path.",
            )
        try:
            remote = subprocess.run(
                ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not verify the existing local repository.",
            ) from exc
        if remote.returncode != 0 or remote.stdout.strip().removesuffix(
            ".git"
        ) != repo_url.removesuffix(".git"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "The existing local repository does not match this URL."
                ),
            )
    else:
        with tempfile.TemporaryDirectory(
            prefix=".clone-", dir=PROJECT_ROOT
        ) as temporary_directory:
            temporary_repo_path = Path(temporary_directory) / "repository"
            _clone_repository(repo_url, temporary_repo_path)
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temporary_repo_path.rename(repo_path)
            except FileExistsError:
                if not (repo_path / ".git").is_dir():
                    raise HTTPException(
                        status_code=409,
                        detail="A repository appeared at the clone path.",
                    )

    processed_path = PROJECT_ROOT / "data" / "indexes" / owner / repository
    try:
        index_endpoint(
            raw_path=str(repo_path),
            processed_path=str(processed_path),
        )
    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail="The repository was cloned, but indexing failed.",
        ) from exc

    if not (processed_path / "chunks.pkl").is_file():
        raise HTTPException(
            status_code=500,
            detail="Indexing did not produce the expected chunk data.",
        )

    prompt_page = PROMPT_PAGE_PATH.read_text(encoding="utf-8")
    prompt_page = Template(prompt_page).substitute(
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
    if not query:
        raise HTTPException(status_code=422, detail="Enter a question.")
    valid_owner = re.fullmatch(r"[A-Za-z0-9_-]+", owner) is not None
    valid_repository = re.fullmatch(
        r"[A-Za-z0-9_.-]+", repository
    ) is not None
    if (
        not valid_owner
        or not valid_repository
        or repository in {".", ".."}
    ):
        raise HTTPException(
            status_code=422,
            detail="The repository identifier is invalid.",
        )

    processed_path = REPOSITORIES_PATH / owner / repository
    if not processed_path.resolve().is_relative_to(
        REPOSITORIES_PATH.resolve()
    ):
        raise HTTPException(
            status_code=400,
            detail="The index path resolves outside the local index area.",
        )
    if not (processed_path / "chunks.pkl").is_file():
        raise HTTPException(
            status_code=404,
            detail="The repository index could not be found.",
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
        sources = (
            '<li class="empty">No source files were returned.</li>'
        )
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
    k: int = 1,
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
