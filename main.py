from documents_loading import Document, load_files
from docs_chunking import Chunker

def main() -> None:
    """"""

    # Knowledge base path
    vllm_repo_path: str = "vllm-0.10.1"

    docs: list[Document] = load_files(input_path=vllm_repo_path)

    chunkme = Chunker(docs)
    chunkme.process_files()


main()
