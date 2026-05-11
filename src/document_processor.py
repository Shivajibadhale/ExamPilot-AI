from dataclasses import dataclass
from typing import BinaryIO, Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class ProcessedDocument:
    full_text: str
    chunks: list[Document]
    vectorstore: FAISS


class DocumentProcessor:
    """Extracts PDF text and stores embeddings in FAISS.

    FAISS can run locally for a lightweight MVP. For larger cohorts, this layer
    can be moved to AMD Developer Cloud infrastructure, pairing ROCm-enabled AMD
    GPUs with scalable embedding and retrieval workflows.
    """

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, uploaded_files: Iterable[BinaryIO]) -> ProcessedDocument:
        documents = []
        for file in uploaded_files:
            documents.extend(self._pdf_to_documents(file))

        if not documents:
            raise ValueError("No readable text was found in the uploaded PDFs.")

        chunks = self.splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        full_text = "\n\n".join(doc.page_content for doc in documents)
        return ProcessedDocument(full_text=full_text, chunks=chunks, vectorstore=vectorstore)

    @staticmethod
    def _pdf_to_documents(file: BinaryIO) -> list[Document]:
        reader = PdfReader(file)
        docs = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            cleaned = " ".join(text.split())
            if cleaned:
                docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={"source": getattr(file, "name", "uploaded PDF"), "page": page_number},
                    )
                )
        return docs
