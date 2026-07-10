# TODO: CREATE ONE SPLITER BY THE PYTHON CODE TEXT SPLITER FROM LANGCHAIN
# TODO: CREATE ANOTHER ONE USING AST (ABSTRACT SYNTAX TREE)
import ast
from documents_loading import Document

class Chunker:
    """"""

    def __init__(self, files: list[Document]) -> None:
        """"""
        self.files: list[Document] = files
        # That just expreiment its not a strict rule
        self.max_chunk_size: int = 500

    def chunk_python_file(self, file: Document) -> None:
        """"""
        print(file.content)
        tree: ast.Module = ast.parse(source=file.content)

        chunks: list = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extracting the real soucrce code of a node
                code: str | None = ast.get_source_segment(file.content, node)

                print(code)
        exit()

    def chunk_markdown_file(self, file: Document) -> None:
        """"""
        pass

    def process_files(self) -> None:
        """"""

        for file in self.files:
            match file.extension:
                case ".py":
                    self.chunk_python_file(file)
                case ".md":
                    pass
                    self.chunk_markdown_file(file)
                case _:
                    # This default should not reached for the current situation
                    pass

    def set_metadata(self) -> None:
        """"""
        pass

    @staticmethod
    def get_len(text_chunk: str) -> int:
        """This method intended to be used for the chunks to calculate length
        of a chunk excluding whitespaces characters, cause even a 2 files with
        the same lines count can vary on intensive of code each file holding.
        """

        return len(text_chunk) - sum(1 for c in text_chunk if c.isspace())


