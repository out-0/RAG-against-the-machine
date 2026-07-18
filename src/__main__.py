from src.documents_loading import Document, load_files
from src.docs_chunking import Chunker, Chunk
import fire
import tqdm
from src.indexer import Indexer


class Boss:
    def index(
        self,
        max_chunk_size: int = 2000,
        raw_path: str = "data/raw/vllm-0.10.1",
        processed_path: str = "data/processed/",
        method: str = "bm25",
    ) -> None:
        print(max_chunk_size)
        print(method)

        # Load the files into program
        docs: list[Document] = load_files(input_path=raw_path)

        # Build the chunker and start processing the files
        chunker = Chunker(files=docs, max_size=max_chunk_size)
        chunks: list[Chunk] = chunker.process_files()
        print(chunks[0])

        # Build the indexer wrapper
        indexer: Indexer = Indexer(
            chunks=chunks,
            method=method,
            processed_path=processed_path,
        )

        print(len(chunks))


if __name__ == "__main__":
    fire.Fire(component=Boss)
