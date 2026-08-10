import pickle
import hashlib
import pikle
from pathlib import Path


# Check if there is new files add'd (optional feature i think)
# Check if hash of a file is changed
# Collect the affected files
# re-chunk and index them
# detect the old chunks from the embedding index
# remove them
# add the new chunks

def handle_incremental_indexing(docs: list, processed_path: str = "data/processed/") -> None:
    hash_path = Path(processed_path) / "files_hash.pkl"

    if not hash_path.exists():
        # First run — index everything, save hashes
        full_index(docs)
        hash_documents(docs, processed_path)
        return

    with open(hash_path, "rb") as f:
        old_metadata: dict = pickle.load(f)

    changed_files = []
    for doc in docs:
        new_hash = generate_hash(doc.content)
        if old_metadata.get(doc.file_path) != new_hash:
            changed_files.append(doc)

    if not changed_files:
        print("No changes detected.")
        return

    reindex_changed_files(changed_files)
    hash_documents(docs, processed_path)  # save fresh hashes for next time





    else:
        # Hash documents and save them and process the normal idexing
        hash_documents(processed_path=processed_path)



# To hash documents content
def generate_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def hash_documents(docs: list, processed_path) -> None:
    """
    loop on passed documents and generate hash for each file content
    """

    metadata: dict = {}
    # Loop on documents

    for doc in docs:
        hashh = generate_hash(doc.content)
        metadata[doc.file_path] = hashh

    with open(Path(processed_path) / 'files_hash.pkl', 'wb') as f:
        pickle.dump(metadata, f)


