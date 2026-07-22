from pathlib import Path


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
            f"\nDocument:\n"
            f"        path={self.path}\n"
            f"        content_length={len(self.content)}\n"
            f"        extension={self.extension}\n"
        )


def load_files(input_path: str) -> list[Document]:
    """Load the files from the provided path and constructe the targeted
    files (py | md) as a Document instances

    Args:
        - input_path = path to the files processed

    Returns:
        - list of Document instances
    """

    path = Path(input_path)

    # Targeted files
    extensions: list[str] = [".py", ".md"]
    docs_list: list[Document] = []
    file_content: str = ""

    if path.is_file():
        if path.suffix in extensions:
            file_content = path.read_text(encoding="utf-8", errors="ignore")
            docs_list.append(
                Document(
                    path=str(path),
                    content=file_content,
                    extension=path.suffix,
                )
            )

    elif path.is_dir():
        try:
            for path in path.rglob("*"):
                if path.suffix in extensions:
                    file_content = path.read_text(encoding="utf-8", errors="ignore")
                    docs_list.append(
                        Document(
                            # the above 'path' is pathlib.PosixPath so i got it to str
                            path=str(path),
                            content=file_content,
                            extension=path.suffix,
                        )
                    )
        except Exception:
            raise ValueError("Error: processing input files")
    return docs_list
