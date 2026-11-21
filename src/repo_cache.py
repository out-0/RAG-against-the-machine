import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_CACHE_LOCK = threading.Lock()
_INDEXABLE_EXTENSIONS = {".py", ".md"}


def hash_indexable_files(repository_path: Path) -> dict[str, str]:
    """Hash the Python and Markdown files used by the indexer."""
    if not repository_path.is_dir():
        raise ValueError(f"Repository directory not found: {repository_path}")

    hashes: dict[str, str] = {}
    for path in sorted(repository_path.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in _INDEXABLE_EXTENSIONS
        ):
            continue
        relative_path = path.relative_to(repository_path).as_posix()
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_repo_cache(cache_path: Path) -> dict[str, Any]:
    """Load the URL-keyed cache, or return an empty cache on first use."""
    if not cache_path.exists():
        return {}

    with cache_path.open(encoding="utf-8") as cache_file:
        cache: Any = json.load(cache_file)
    if not isinstance(cache, dict) or not all(
        isinstance(url, str) for url in cache
    ):
        raise ValueError(f"Invalid repository cache format: {cache_path}")
    return cache


def save_repo_cache_entry(
    cache_path: Path,
    url: str,
    file_hashes: dict[str, str],
    index_path: str,
) -> None:
    """Atomically update one URL entry without dropping other entries."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        cache = load_repo_cache(cache_path)
        cache[url] = {"files": file_hashes, "index": index_path}

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=".repo-cache-",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(cache, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
