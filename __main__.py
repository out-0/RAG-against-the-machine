from documents_loading import Document, load_files
from docs_chunking import Chunker
import fire


def main() -> None:
    """"""

    # Knowledge base path
    vllm_repo_path: str = "vllm-0.10.1"
    # vllm_repo_path: str = "tokens_check.py"
    # mk_test = "meetups.md"

    docs: list[Document] = load_files(input_path=vllm_repo_path)
    # docs: list[Document] = load_files(input_path=mk_test)
    chunkme = Chunker(docs, 2000)

    fire.Fire(Chunker)

    s = chunkme.process_files()
    for d in s:
        if len(d.content) > 2000:
            print("oversized detected")
            raise ValueError()
            exit()
        print(d.id)
        print(len(d.content))
        print(d.content)


class Boss:
    def index(max_chunk_size=2000, path="data/raw/vllm-0.10.1") -> None:
        """"""
        pass


if __name__ == "__main__":
    fire.Fire(Boss)
