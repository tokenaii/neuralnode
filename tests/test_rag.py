import unittest
import sys
sys.path.insert(0, "src")
from neuralnode.rag import (
    InMemoryVectorStore,
    RecursiveCharacterTextSplitter,
    Document,
    TextLoader,
    split_documents,
)

class TestRAG(unittest.TestCase):
    def test_splitter(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = split_documents([Document(content="\n\n".join(["x"*80]*3), metadata={})], chunk_size=100, chunk_overlap=10)
        self.assertGreater(len(chunks), 1)

    def test_inmemory_store(self):
        store = InMemoryVectorStore()
        store.add([ObjectStoreHelper.chunk()])
        self.assertGreaterEqual(len(store.chunks), 1)

    def test_text_loader(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        p = os.path.join(d, "t.txt")
        open(p, "w", encoding="utf-8").write("hello rag")
        docs = TextLoader().load(p)
        self.assertTrue(docs)
        self.assertIn("hello rag", docs[0].content)

def chunk():
    from neuralnode.rag import Chunk
    return Chunk(content="doc1", doc_id="d1")

class ObjectStoreHelper:
    chunk = staticmethod(chunk)

if __name__ == "__main__":
    unittest.main()
