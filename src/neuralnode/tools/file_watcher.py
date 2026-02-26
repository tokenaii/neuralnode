"""File watching tool for monitoring file system changes."""

import os
import time
import threading
from typing import Callable, Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
from datetime import datetime


class FileEventType(Enum):
    """Types of file events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """A file system event."""
    event_type: FileEventType
    path: str
    is_directory: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    old_path: Optional[str] = None  # For move events
    size: Optional[int] = None
    hash: Optional[str] = None


class FileWatcher:
    """Watch files and directories for changes."""
    
    def __init__(
        self,
        recursive: bool = True,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.recursive = recursive
        self.ignore_patterns = ignore_patterns or [
            "*.tmp", "*.temp", "*.log",
            "__pycache__", ".git", ".env",
            "node_modules", ".pytest_cache",
        ]
        self._watching: Set[str] = set()
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._callbacks: List[Callable[[FileEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_interval = 1.0  # seconds
    
    def watch(self, path: str) -> None:
        """Add a path to watch."""
        abs_path = os.path.abspath(path)
        self._watching.add(abs_path)
        self._snapshots[abs_path] = self._snapshot_path(abs_path)
    
    def unwatch(self, path: str) -> None:
        """Remove a path from watching."""
        abs_path = os.path.abspath(path)
        self._watching.discard(abs_path)
        self._snapshots.pop(abs_path, None)
    
    def on_change(self, callback: Callable[[FileEvent], None]) -> None:
        """Register a callback for file events."""
        self._callbacks.append(callback)
    
    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            self._check_changes()
            time.sleep(self._poll_interval)
    
    def _check_changes(self) -> None:
        """Check for changes in watched paths."""
        for path in list(self._watching):
            old_snapshot = self._snapshots.get(path, {})
            new_snapshot = self._snapshot_path(path)
            
            # Detect changes
            events = self._detect_changes(old_snapshot, new_snapshot, path)
            
            for event in events:
                self._emit_event(event)
            
            self._snapshots[path] = new_snapshot
    
    def _snapshot_path(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Create a snapshot of a path."""
        snapshot = {}
        
        if os.path.isfile(path):
            snapshot[path] = self._file_info(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(d)]
                
                for file in files:
                    if self._should_ignore(file):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        snapshot[file_path] = self._file_info(file_path)
                    except (OSError, IOError):
                        pass
                
                if not self.recursive:
                    break
        
        return snapshot
    
    def _file_info(self, path: str) -> Dict[str, Any]:
        """Get file information."""
        stat = os.stat(path)
        
        # Calculate hash for small files
        file_hash = None
        if stat.st_size < 10 * 1024 * 1024:  # Only for files < 10MB
            try:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            except (OSError, IOError):
                pass
        
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "is_dir": os.path.isdir(path),
            "hash": file_hash,
        }
    
    def _detect_changes(
        self,
        old_snapshot: Dict[str, Dict[str, Any]],
        new_snapshot: Dict[str, Dict[str, Any]],
        base_path: str,
    ) -> List[FileEvent]:
        """Detect changes between snapshots."""
        events = []
        
        old_files = set(old_snapshot.keys())
        new_files = set(new_snapshot.keys())
        
        # Created files
        for path in new_files - old_files:
            info = new_snapshot[path]
            events.append(FileEvent(
                event_type=FileEventType.CREATED,
                path=path,
                is_directory=info.get("is_dir", False),
                size=info.get("size"),
                hash=info.get("hash"),
            ))
        
        # Deleted files
        for path in old_files - new_files:
            info = old_snapshot[path]
            events.append(FileEvent(
                event_type=FileEventType.DELETED,
                path=path,
                is_directory=info.get("is_dir", False),
            ))
        
        # Modified files
        for path in old_files & new_files:
            old_info = old_snapshot[path]
            new_info = new_snapshot[path]
            
            # Check if modified (mtime or size changed)
            if (old_info["mtime"] != new_info["mtime"] or
                old_info["size"] != new_info["size"] or
                (old_info["hash"] and new_info["hash"] and
                 old_info["hash"] != new_info["hash"])):
                
                events.append(FileEvent(
                    event_type=FileEventType.MODIFIED,
                    path=path,
                    is_directory=new_info.get("is_dir", False),
                    size=new_info.get("size"),
                    hash=new_info.get("hash"),
                ))
        
        return events
    
    def _should_ignore(self, name: str) -> bool:
        """Check if a file should be ignored."""
        for pattern in self.ignore_patterns:
            if pattern in name or name.endswith(pattern.replace("*", "")):
                return True
        return False
    
    def _emit_event(self, event: FileEvent) -> None:
        """Emit event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in file watcher callback: {e}")
    
    def get_watched_paths(self) -> List[str]:
        """Get list of watched paths."""
        return list(self._watching)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class SmartFileWatcher(FileWatcher):
    """Smart file watcher with AI-powered actions."""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self._setup_smart_callbacks()
    
    def _setup_smart_callbacks(self) -> None:
        """Setup intelligent file handling."""
        self.on_change(self._smart_handler)
    
    def _smart_handler(self, event: FileEvent) -> None:
        """Intelligent event handler."""
        if not self.llm:
            return
        
        # Analyze file changes with AI
        if event.event_type == FileEventType.MODIFIED:
            if event.path.endswith(('.py', '.js', '.ts')):
                # Code file changed - could analyze for issues
                pass
            elif event.path.endswith(('.md', '.txt')):
                # Document changed - could summarize
                pass


# Convenience function
def watch_directory(
    path: str,
    on_change: Callable[[FileEvent], None],
    recursive: bool = True,
) -> FileWatcher:
    """Quick directory watching setup."""
    watcher = FileWatcher(recursive=recursive)
    watcher.watch(path)
    watcher.on_change(on_change)
    watcher.start()
    return watcher
