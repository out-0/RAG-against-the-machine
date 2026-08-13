from itertools import count
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.custom_print import print_red
from src.data_models import Chunk
from src.docs_loading import Document


class Chunker:
    """Handles syntax-aware document chunking using LangChain text splitters

    while preserving file paths, IDs, and character start/end offsets.
    """

    def __init__(
        self,
        files: list[Document],
        max_size: int = 2000,
        overlap: int = 200,
    ) -> None:
        """Initialize the chunker with documents and size constraints.

        Args:
            files: List of Document objects to be chunked.
            max_size: Maximum character length per chunk.
            overlap: Character overlap between consecutive chunks to preserve
              context.
        """
        self.files: list[Document] = files
        self.max_size: int = max_size
        self.overlap: int = overlap
        self._id_generator = count(1)

    def _get_splitter_for_file(
        self, file_path: str
    ) -> RecursiveCharacterTextSplitter | None:
        """Selects a language-appropriate splitter based on file extension.

        Args:
            file_path: Path string to determine file type.

        Returns:
            Configured RecursiveCharacterTextSplitter instance.
        """
        ext: str = Path(file_path).suffix.lower()

        # Custom separators tailored for Python code structure
        if ext == ".py":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_size,
                chunk_overlap=self.overlap,
                add_start_index=True,
            )

        # Custom separators tailored for Markdown headers and lists
        if ext == ".md":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_size,
                chunk_overlap=self.overlap,
                add_start_index=True,
            )

        print_red("Warning: File type is not supported.")
        return None

    def process_files(self) -> list[Chunk]:
        """Processes all documents and outputs structured Chunk instances.

        Returns:
            List of populated Chunk dataclass/pydantic objects.
        """
        all_chunks: list[Chunk] = []

        for doc in self.files:
            file_path = doc.path
            content = doc.content

            # Just protextion against empty files
            if not content.strip():
                continue

            # Retrieve appropriate splitter & process document
            splitter = self._get_splitter_for_file(file_path)
            if not splitter:
                continue
            # Split the file
            lc_docs = splitter.create_documents(
                texts=[content],
                # metadatas=[{"file_path": file_path}],
            )

            # Build Chunk instances from splited parts
            for lc_doc in lc_docs:
                start_idx: int = lc_doc.metadata.get("start_index", 0)
                end_idx: int = start_idx + len(lc_doc.page_content)

                chunk = Chunk(
                    id=next(self._id_generator),
                    content=lc_doc.page_content,
                    start_index=start_idx,
                    end_index=end_idx,
                    file_path=file_path,
                )
                all_chunks.append(chunk)

        return all_chunks
