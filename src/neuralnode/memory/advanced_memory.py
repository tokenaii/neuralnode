"""Advanced memory system for NeuralNode."""

import json
import pickle
import sqlite3
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib

from ..core.types import Message, MessageRole


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: datetime
    importance: float = 1.0  # 0.0 to 1.0
    category: str = "general"
    source: str = "user"
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata,
        }


class ShortTermMemory:
    """Short-term working memory (conversation buffer)."""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    def add(self, message: Message) -> None:
        """Add message to short-term memory."""
        self.messages.append(message)
        
        # Trim if exceeds max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent(self, n: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-n:]
    
    def clear(self) -> None:
        """Clear short-term memory."""
        self.messages.clear()
    
    def to_context_string(self, n: Optional[int] = None) -> str:
        """Convert to context string for prompts."""
        messages = self.messages if n is None else self.messages[-n:]
        return "\n".join([
            f"{m.role.value}: {m.content}"
            for m in messages
        ])


class LongTermMemory:
    """Long-term persistent memory with SQLite."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".neuralnode" / "memory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        cursor = self._conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                category TEXT DEFAULT 'general',
                source TEXT DEFAULT 'user',
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        
        self._conn.commit()
    
    def add(self, entry: MemoryEntry) -> int:
        """Add memory entry. Returns entry ID."""
        cursor = self._conn.cursor()
        
        embedding_blob = pickle.dumps(entry.embedding) if entry.embedding else None
        
        cursor.execute("""
            INSERT INTO memories (content, timestamp, importance, category, source, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.content,
            entry.timestamp.isoformat(),
            entry.importance,
            entry.category,
            entry.source,
            json.dumps(entry.metadata),
            embedding_blob,
        ))
        
        self._conn.commit()
        return cursor.lastrowid
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Search memories by content similarity (simplified)."""
        cursor = self._conn.cursor()
        
        # Build query
        sql = "SELECT * FROM memories WHERE (content LIKE ? OR metadata LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        sql += " AND importance >= ? ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.extend([min_importance, limit])
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        entries = []
        for row in rows:
            entries.append(MemoryEntry(
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                importance=row[3],
                category=row[4],
                source=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
                embedding=pickle.loads(row[7]) if row[7] else None,
            ))
        
        return entries
    
    def get_by_category(self, category: str, limit: int = 100) -> List[MemoryEntry]:
        """Get memories by category."""
        cursor = self._conn.cursor()
        
        cursor.execute("""
            SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?
        """, (category, limit))
        
        rows = cursor.fetchall()
        return [
            MemoryEntry(
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                importance=row[3],
                category=row[4],
                source=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]
    
    def delete(self, memory_id: int) -> bool:
        """Delete memory by ID."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0
    
    def clear_category(self, category: str) -> int:
        """Clear all memories in category."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE category = ?", (category,))
        self._conn.commit()
        return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cursor = self._conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
        by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "total_entries": total,
            "by_category": by_category,
        }


class SemanticMemory:
    """Memory with semantic search using embeddings."""
    
    def __init__(
        self,
        long_term: Optional[LongTermMemory] = None,
        embedding_function: Optional[Any] = None,
    ):
        self.long_term = long_term or LongTermMemory()
        self.embedding_function = embedding_function
    
    def add(
        self,
        content: str,
        category: str = "general",
        importance: float = 1.0,
    ) -> int:
        """Add memory with embedding."""
        # Generate embedding if function available
        embedding = None
        if self.embedding_function:
            try:
                embedding = self.embedding_function(content)
            except Exception:
                pass
        
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            category=category,
            embedding=embedding,
        )
        
        return self.long_term.add(entry)
    
    def semantic_search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Search by semantic similarity."""
        if not self.embedding_function:
            # Fallback to text search
            return self.long_term.search(query, limit=limit)
        
        # Get all memories with embeddings
        cursor = self.long_term._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        if not rows:
            return self.long_term.search(query, limit=limit)
        
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Calculate similarities
        import math
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)
        
        scored = []
        for row in rows:
            emb = pickle.loads(row[7])
            score = cosine_similarity(query_embedding, emb)
            entry = MemoryEntry(
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                importance=row[3],
                category=row[4],
                source=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
            )
            scored.append((entry, score))
        
        # Sort by score and return top
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored[:limit]]


class WorkingMemory:
    """Active working memory for current task."""
    
    def __init__(self):
        self.facts: Dict[str, Any] = {}
        self.current_goal: Optional[str] = None
        self.subtasks: List[str] = []
        self.context: Dict[str, Any] = {}
    
    def set_fact(self, key: str, value: Any) -> None:
        """Set a fact in working memory."""
        self.facts[key] = value
    
    def get_fact(self, key: str) -> Any:
        """Get a fact from working memory."""
        return self.facts.get(key)
    
    def set_goal(self, goal: str) -> None:
        """Set current goal."""
        self.current_goal = goal
    
    def add_subtask(self, subtask: str) -> None:
        """Add subtask."""
        self.subtasks.append(subtask)
    
    def complete_subtask(self, subtask: str) -> None:
        """Mark subtask complete."""
        if subtask in self.subtasks:
            self.subtasks.remove(subtask)
    
    def clear(self) -> None:
        """Clear working memory."""
        self.facts.clear()
        self.current_goal = None
        self.subtasks.clear()
        self.context.clear()


class CompleteMemorySystem:
    """Complete memory system combining all memory types."""
    
    def __init__(
        self,
        short_term_max: int = 50,
        db_path: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        self.short_term = ShortTermMemory(max_messages=short_term_max)
        self.long_term = LongTermMemory(db_path=db_path)
        self.semantic = SemanticMemory(self.long_term, embedding_function)
        self.working = WorkingMemory()
    
    def add_message(self, message: Message, importance: float = 1.0) -> None:
        """Add message to short-term and optionally long-term."""
        self.short_term.add(message)
        
        # Store important messages in long-term
        if importance > 0.7:
            self.semantic.add(
                content=f"{message.role.value}: {message.content}",
                category="conversation",
                importance=importance,
            )
    
    def remember(
        self,
        content: str,
        category: str = "general",
        importance: float = 1.0,
    ) -> int:
        """Explicitly remember something."""
        return self.semantic.add(content, category, importance)
    
    def recall(
        self,
        query: str,
        use_semantic: bool = True,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Recall relevant memories."""
        if use_semantic:
            return self.semantic.semantic_search(query, limit)
        else:
            return self.long_term.search(query, limit=limit)
    
    def get_conversation_context(self, n: int = 10) -> List[Message]:
        """Get recent conversation context."""
        return self.short_term.get_recent(n)
    
    def get_working_memory(self) -> WorkingMemory:
        """Get working memory for current task."""
        return self.working
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
    
    def clear_long_term(self, category: Optional[str] = None) -> None:
        """Clear long-term memory."""
        if category:
            self.long_term.clear_category(category)
        else:
            # Clear all - use with caution
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term": {
                "messages": len(self.short_term.messages),
                "max": self.short_term.max_messages,
            },
            "long_term": self.long_term.get_stats(),
        }


# Integration with ConversationMemory
class AdvancedConversationMemory:
    """Drop-in replacement for ConversationMemory with full memory system."""
    
    def __init__(
        self,
        max_messages: int = 50,
        enable_long_term: bool = True,
        db_path: Optional[str] = None,
    ):
        self.short_term = ShortTermMemory(max_messages=max_messages)
        self.long_term_enabled = enable_long_term
        
        if enable_long_term:
            self.long_term = LongTermMemory(db_path=db_path)
            self.semantic = SemanticMemory(self.long_term)
    
    def add_message(self, message: Message) -> None:
        """Add message to memory."""
        self.short_term.add(message)
        
        # Store in long-term if significant
        if self.long_term_enabled:
            content = f"{message.role.value}: {message.content}"
            self.semantic.add(content, category="conversation")
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages."""
        return self.short_term.get_recent(limit or self.short_term.max_messages)
    
    def clear(self) -> None:
        """Clear memory."""
        self.short_term.clear()
    
    def to_list(self) -> List[Dict[str, str]]:
        """Convert to list format."""
        return [m.to_dict() for m in self.short_term.messages]
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history."""
        if not self.long_term_enabled:
            return []
        
        results = self.semantic.semantic_search(query)
        return [r.to_dict() for r in results]
