from pathlib import Path


class Document:
    """
    A class that represent a document
    """

    def __init__(self, path: str, content: str, extension: str) -> None:
        """
        Constructor
        Args:
            - path (str): The path to the document
            - content (str): The content of the document
            - extension (str): The extension of the document
        """
        self.path: str = path
        self.content: str = content
        self.extension: str = extension

    def __repr__(self) -> str:
        """
        Return a string representation of the document
        """
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
            for p in path.rglob("*"):
                if p.suffix in extensions:
                    file_content = p.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    docs_list.append(
                        Document(
                            path=str(p),
                            content=file_content,
                            extension=p.suffix,
                        )
                    )
        except Exception:
            raise ValueError("Error: processing input files")
    return docs_list
