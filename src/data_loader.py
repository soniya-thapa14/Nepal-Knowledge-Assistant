#data_loader.py

from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Load all .txt files from a directory and return them as a list of Document objects.

    Args:
        data_dir (str): Path to the raw text files directory.

    Returns:
        List[Document]: A list of Document objects with text and metadata.
    """

    print("=" * 70)
    print("LOADING NEPAL DOCUMENTS")
    print("=" * 70)

    documents: List[Document] = []

    for file_path in Path(data_dir).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={"source": file_path.name}  # keep track of file name
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents from {data_dir}")
    return documents


