from documents_loading import Document, load_files
from docs_chunking import Chunker

def main() -> None:
    """"""

    # Knowledge base path
    #vllm_repo_path: str = "vllm-0.10.1"
    #vllm_repo_path: str = "tokens_check.py"
    mk_test = "meetups.md"

    #docs: list[Document] = load_files(input_path=vllm_repo_path)
    docs: list[Document] = load_files(input_path=mk_test)
    chunkme = Chunker(docs, 2000)
    s = chunkme.process_files()
    for d in s:
        print(d.id)
        print(len(d.content))
        print(d.content)

main()
