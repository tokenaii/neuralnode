import tempfile, os, unittest
import sys
sys.path.insert(0, "src")
from neuralnode.memory import ConversationMemory, FileMemory, SlidingWindowMemory
from neuralnode.core import Message

class TestMemory(unittest.TestCase):
    def test_conversation_count_and_search(self):
        m = ConversationMemory(max_messages=10)
        m.add(Message.user("hello world"))
        m.add(Message.assistant("hi"))
        self.assertEqual(m.count(), 2)
        found = m.search("hello")
        self.assertEqual(len(found), 1)

    def test_conversation_trim(self):
        m = ConversationMemory(max_messages=3)
        for i in range(10):
            m.add(Message.user(str(i)))
        self.assertLessEqual(m.count(), 3)

    def test_file_memory_persistence(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "mem.json")
        fm = FileMemory(path)
        fm.add(Message.user("persist me"))
        fm2 = FileMemory(path)
        self.assertEqual(fm2.count(), 1)
        self.assertEqual(fm2.get()[0].content, "persist me")

    def test_sliding_window(self):
        sw = SlidingWindowMemory(max_pairs=2)
        for i in range(5):
            sw.add(Message.user("q"+str(i)))
            sw.add(Message.assistant("a"+str(i)))
        self.assertEqual(sw.count(), 4)

if __name__ == "__main__":
    unittest.main()
