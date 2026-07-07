from pathlib import Path
from typing import List


class Document:
    """"""
    def __init__(self, path: str, content: str, extension: str) -> None:
        """"""
        self.path: str = path
        self.content: str = content
        self.extension: str = extension

    def __repr__(self) -> str:
        """"""
        return (
            f"Document:\n"
            f"        path={self.path}\n"
            f"        content_length={len(self.content)}\n"
            f"        extension={self.extension}"
        )


def load_files() -> List[Document]:
    """"""

    # Knowledge base path
    vllm_repo_path: str = "vllm-0.10.1"

    # Targeted files
    extensions: List[str] = [".py", ".md"]

    docs_list: List[Document] = [] 

    for path in Path(vllm_repo_path).rglob("*"):
        if path.suffix in extensions:
            file_text: str = path.read_text(encoding="utf-8", errors="ignore")
            docs_list.append(
                Document(
                    # the above 'path' is pathlib.PosixPath so i got it to str
                    path=str(path),
                    content=file_text,
                    extension=path.suffix
                )
            )
    return docs_list

load_files()
