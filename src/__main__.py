from src.documents_loading import Document, load_files
from src.docs_chunking import Chunker, Chunk
import fire
import tqdm


class Orchestrator:
    def index(
        self,
        max_chunk_size: int = 2000,
        raw_path: str = "data/raw/vllm-0.10.1",
        processed_dir: str = "data/processed/",
    ) -> None:

        # Load the files
        docs: list[Document] = load_files(input_path=raw_path)

        chunker = Chunker(files=docs, max_size=max_chunk_size)
        chunks: list[Chunk] = chunker.process_files()

        print(len(chunks))


if __name__ == "__main__":
    fire.Fire(Orchestrator)
