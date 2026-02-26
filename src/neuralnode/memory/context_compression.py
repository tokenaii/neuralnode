"""
Context Compression for efficient memory management.
Compresses old messages while preserving important information.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class CompressedMessage:
    """Compressed message with summary and key facts."""
    original_role: str
    original_content: str
    summary: str
    key_facts: List[str]
    importance_score: float  # 0-1, higher = more important
    timestamp: datetime
    compression_ratio: float  # How much it was compressed


class MessageCompressor:
    """Compress messages to save tokens while preserving meaning."""
    
    def __init__(self, llm=None, target_ratio: float = 0.3):
        """
        Initialize compressor.
        
        Args:
            llm: Language model for summarization
            target_ratio: Target compression ratio (0.3 = 70% reduction)
        """
        self.llm = llm
        self.target_ratio = target_ratio
    
    def compress(
        self,
        messages: List[Dict[str, str]],
        preserve_last_n: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Compress messages while preserving recent ones.
        
        Args:
            messages: List of messages (role, content)
            preserve_last_n: Number of recent messages to keep uncompressed
        
        Returns:
            Compressed message list
        """
        if len(messages) <= preserve_last_n:
            return messages
        
        # Keep recent messages as-is
        recent = messages[-preserve_last_n:]
        to_compress = messages[:-preserve_last_n]
        
        # Compress older messages
        compressed = self._compress_batch(to_compress)
        
        # Combine
        return compressed + recent
    
    def _compress_batch(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Compress a batch of messages."""
        if not messages:
            return []
        
        # For single message, use simple compression
        if len(messages) == 1:
            compressed = self._compress_single(messages[0])
            return [compressed] if compressed else messages
        
        # For multiple messages, create summary
        if self.llm:
            summary = self._create_summary(messages)
            return [{
                "role": "system",
                "content": f"[Previous conversation summary: {summary}]"
            }]
        
        # Without LLM, use simple truncation
        return self._simple_compress(messages)
    
    def _compress_single(self, message: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Compress a single message."""
        content = message.get("content", "")
        
        # Skip system messages
        if message.get("role") == "system":
            return message
        
        # Calculate importance
        importance = self._calculate_importance(content)
        
        # If very important, keep mostly intact
        if importance > 0.8:
            return message
        
        # If low importance, heavily compress
        if importance < 0.3:
            return {
                "role": message["role"],
                "content": self._extractive_summary(content, max_length=50),
            }
        
        # Medium importance - moderate compression
        return {
            "role": message["role"],
            "content": self._extractive_summary(content, max_length=100),
        }
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score of content."""
        score = 0.5
        
        # Increase for action items
        action_indicators = [
            "must", "need to", "should", "will", "plan to",
            "todo", "task", "action", "deadline", "schedule",
            "important", "critical", "urgent", "key point",
            "remember", "don't forget", "note that",
        ]
        content_lower = content.lower()
        for indicator in action_indicators:
            if indicator in content_lower:
                score += 0.1
        
        # Increase for code/technical content
        if any(c in content for c in ['```', 'def ', 'class ', 'import ', 'function']):
            score += 0.15
        
        # Increase for URLs and references
        if "http" in content or "www." in content:
            score += 0.05
        
        # Decrease for simple acknowledgments
        simple_words = ["ok", "okay", "yes", "no", "thanks", "thank you", "got it"]
        if content.lower().strip() in simple_words:
            score -= 0.3
        
        # Decrease for very short messages
        if len(content) < 20:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extractive_summary(self, content: str, max_length: int = 100) -> str:
        """Create extractive summary (no LLM needed)."""
        if len(content) <= max_length:
            return content
        
        # Split into sentences
        sentences = content.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Select most informative sentences
        selected = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            
            # Score sentence
            score = self._sentence_score(sentence)
            if score > 0.3:  # Threshold
                selected.append(sentence)
                current_length += len(sentence) + 1
        
        if not selected:
            # Fallback: truncate with ellipsis
            return content[:max_length-3] + "..."
        
        return ". ".join(selected) + "."
    
    def _sentence_score(self, sentence: str) -> float:
        """Score sentence importance."""
        score = 0.5
        
        # Longer sentences often have more info
        words = len(sentence.split())
        if words > 10:
            score += 0.1
        if words < 5:
            score -= 0.2
        
        # Specific terms
        important_terms = [
            "decided", "agreed", "conclusion", "result", "outcome",
            "because", "therefore", "so", "thus", "consequently",
            "however", "but", "although", "though",
            "specifically", "namely", "that is",
        ]
        for term in important_terms:
            if term in sentence.lower():
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _create_summary(self, messages: List[Dict[str, str]]) -> str:
        """Create LLM-based summary of messages."""
        if not self.llm:
            return self._simple_summary(messages)
        
        # Prepare content
        content = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        ])
        
        prompt = f"""Summarize this conversation concisely, preserving key information:

{content[:2000]}

Summary (keep under 200 words):
"""
        
        try:
            response = self.llm.chat(prompt)
            summary = response.text if hasattr(response, 'text') else str(response)
            return summary[:500]  # Limit summary length
        except:
            return self._simple_summary(messages)
    
    def _simple_summary(self, messages: List[Dict[str, str]]) -> str:
        """Create simple summary without LLM."""
        # Extract key sentences
        all_content = " ".join([m.get("content", "") for m in messages])
        return self._extractive_summary(all_content, max_length=200)
    
    def _simple_compress(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Simple compression without LLM."""
        compressed = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            if len(content) > 200:
                compressed.append({
                    "role": msg["role"],
                    "content": content[:197] + "...",
                })
            else:
                compressed.append(msg)
        
        return compressed


class SlidingWindowMemory:
    """
    Memory with sliding window and compression.
    Keeps recent messages in full, compresses older ones.
    """
    
    def __init__(
        self,
        llm=None,
        max_messages: int = 20,
        preserve_recent: int = 5,
        compression_threshold: int = 10,
    ):
        self.llm = llm
        self.max_messages = max_messages
        self.preserve_recent = preserve_recent
        self.compression_threshold = compression_threshold
        self.compressor = MessageCompressor(llm)
        self.messages: List[Dict[str, str]] = []
    
    def add(self, role: str, content: str) -> None:
        """Add message to memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Compress if over threshold
        if len(self.messages) > self.compression_threshold:
            self._compress()
    
    def _compress(self) -> None:
        """Compress messages to fit in window."""
        self.messages = self.compressor.compress(
            self.messages,
            preserve_last_n=self.preserve_recent,
        )
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages, optionally limited."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        
        return {
            "message_count": len(self.messages),
            "total_characters": total_chars,
            "avg_message_length": total_chars / len(self.messages) if self.messages else 0,
        }


class HierarchicalMemory:
    """
    Multi-level memory system:
    - Level 1: Recent messages (full detail)
    - Level 2: Compressed summaries
    - Level 3: Long-term facts
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.recent = SlidingWindowMemory(llm, max_messages=10, preserve_recent=5)
        self.summaries: List[str] = []
        self.key_facts: Dict[str, str] = {}
    
    def add(self, role: str, content: str) -> None:
        """Add message."""
        self.recent.add(role, content)
        
        # Extract key facts periodically
        if len(self.recent.messages) % 5 == 0:
            self._extract_facts()
    
    def _extract_facts(self) -> None:
        """Extract key facts from recent conversation."""
        if not self.llm:
            return
        
        recent_content = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in self.recent.messages[-5:]
        ])
        
        prompt = f"""Extract key facts from this conversation that should be remembered:

{recent_content}

List 3-5 key facts (one per line):
"""
        
        try:
            response = self.llm.chat(prompt)
            facts = response.text if hasattr(response, 'text') else str(response)
            
            # Store facts
            for i, fact in enumerate(facts.split('\n')[:5]):
                if fact.strip():
                    self.key_facts[f"fact_{len(self.key_facts)}"] = fact.strip()
        
        except:
            pass
    
    def get_context(self) -> str:
        """Get full context for prompting."""
        parts = []
        
        # Add key facts
        if self.key_facts:
            parts.append("Key facts:")
            for fact in self.key_facts.values():
                parts.append(f"- {fact}")
            parts.append("")
        
        # Add recent messages
        parts.append("Recent conversation:")
        for msg in self.recent.get_messages():
            parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(parts)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in format for LLM."""
        # Create system message with context
        system_content = ""
        
        if self.key_facts:
            system_content += "Key facts from previous conversation:\n"
            for fact in self.key_facts.values():
                system_content += f"- {fact}\n"
        
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        messages.extend(self.recent.get_messages())
        return messages


# Convenience functions
def compress_messages(
    messages: List[Dict[str, str]],
    llm=None,
    preserve_last: int = 3,
) -> List[Dict[str, str]]:
    """Quick message compression."""
    compressor = MessageCompressor(llm)
    return compressor.compress(messages, preserve_last_n=preserve_last)


def create_sliding_memory(llm=None, max_messages: int = 20) -> SlidingWindowMemory:
    """Quick sliding window memory creation."""
    return SlidingWindowMemory(llm, max_messages=max_messages)
