import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
from neuralnode.rag import (
    CSVLoader,
    InMemoryVectorStore,
    JSONLoader,
    DocumentLoader,
    RecursiveCharacterTextSplitter,
    Document,
    TextLoader,
    split_documents,
)


class TestRAG(unittest.TestCase):
    def test_splitter(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = split_documents(
            [Document(content="\n\n".join(["x" * 80] * 3), metadata={})],
            chunk_size=100,
            chunk_overlap=10,
        )
        self.assertGreater(len(chunks), 1)

    def test_inmemory_store(self):
        store = InMemoryVectorStore()
        store.add([ObjectStoreHelper.chunk()])
        self.assertGreaterEqual(len(store.chunks), 1)

    def test_text_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            p.write_text("hello rag", encoding="utf-8")
            docs = TextLoader().load(p)
        self.assertTrue(docs)
        self.assertIn("hello rag", docs[0].content)

    def test_document_loader_preserves_csv_and_json_loaders(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "records.csv"
            csv_path.write_text("name,age\nAda,36\n", encoding="utf-8")
            json_path = Path(d) / "records.json"
            json_path.write_text('[{"name": "Ada"}, {"name": "Grace"}]', encoding="utf-8")

            csv_docs = DocumentLoader().load(csv_path)
            json_docs = DocumentLoader().load(json_path)

        self.assertEqual(len(csv_docs), 1)
        self.assertEqual(csv_docs[0].metadata["type"], "csv")
        self.assertEqual(len(json_docs), 2)
        self.assertEqual(json_docs[0].metadata["type"], "json")

    def test_csv_and_json_loader_classes_are_not_text_loader_aliases(self):
        self.assertIsNot(CSVLoader, TextLoader)
        self.assertIsNot(JSONLoader, TextLoader)


def chunk():
    from neuralnode.rag import Chunk

    return Chunk(content="doc1", doc_id="d1")


class ObjectStoreHelper:
    chunk = staticmethod(chunk)


if __name__ == "__main__":
    unittest.main()
