import os
import json
import logging
import requests
import subprocess
import shlex
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

from ..core import BaseTool, ToolResult, ParameterSpec

logger = logging.getLogger(__name__)


# ==================== Web & Search Tools ====================


class GoogleSearchTool(BaseTool):
    """Search Google using SerpAPI or similar."""

    name = "google_search"
    description = "Search Google for information"
    parameters = {
        "query": ParameterSpec(type="string", description="The search query", required=True),
        "num_results": ParameterSpec(
            type="integer", description="Number of results to return", required=False, default=5
        ),
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")

    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """Execute Google search."""
        if not self.api_key:
            return ToolResult.error("SERPAPI_API_KEY not set")

        try:
            url = "https://serpapi.com/search"
            params = {"q": query, "api_key": self.api_key, "engine": "google", "num": num_results}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract organic results
            results = data.get("organic_results", [])
            formatted = []
            for r in results[:num_results]:
                formatted.append(
                    f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}\nLink: {r.get('link', '')}"
                )

            return ToolResult.ok(content="\n\n".join(formatted), raw_results=results)

        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")


class WikipediaTool(BaseTool):
    """Search Wikipedia."""

    name = "wikipedia"
    description = "Search Wikipedia for information on a topic"

    parameters = {
        "query": ParameterSpec(type="string", description="The topic to search for", required=True),
        "sentences": ParameterSpec(
            type="integer", description="Number of sentences to return", required=False, default=5
        ),
    }

    def execute(self, query: str, sentences: int = 5, **kwargs) -> ToolResult:
        """Execute Wikipedia search."""
        try:
            # Wikipedia API
            search_url = "https://en.wikipedia.org/w/api.php"

            # Search for page
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            }

            response = requests.get(search_url, params=search_params, timeout=30)
            data = response.json()

            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return ToolResult.error(f"No Wikipedia article found for: {query}")

            page_title = search_results[0]["title"]

            # Get page content
            content_params = {
                "action": "query",
                "prop": "extracts",
                "exsentences": sentences,
                "exlimit": 1,
                "titles": page_title,
                "explaintext": 1,
                "format": "json",
            }

            response = requests.get(search_url, params=content_params, timeout=30)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page = list(pages.values())[0]
            extract = page.get("extract", "No content found")

            return ToolResult.ok(
                content=f"Wikipedia article: {page_title}\n\n{extract}",
                title=page_title,
                url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
            )

        except Exception as e:
            return ToolResult.error(f"Wikipedia search failed: {e}")


class WebScraperTool(BaseTool):
    """Scrape web pages."""

    name = "web_scraper"
    description = "Scrape and extract text from a web page"

    parameters = {
        "url": ParameterSpec(type="string", description="The URL to scrape", required=True)
    }

    def execute(self, url: str, **kwargs) -> ToolResult:
        """Scrape web page."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            html = response.text

            # Simple text extraction (could use BeautifulSoup for better results)
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)

            except ImportError:
                # Fallback - just return raw HTML
                text = html[:5000]

            return ToolResult.ok(
                content=text[:5000], url=url, title=self._extract_title(html)  # Limit length
            )

        except Exception as e:
            return ToolResult.error(f"Scraping failed: {e}")

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        import re

        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1) if match else "Unknown"


# ==================== Code Execution Tools ====================


class PythonREPLTool(BaseTool):
    """Execute Python code with proper timeout and isolation."""

    name = "python_repl"
    description = "Execute Python code safely with timeout protection"

    def __init__(self, timeout: int = 30, allowed_modules: Optional[List[str]] = None):
        self.timeout = timeout
        self.allowed_modules = allowed_modules

    parameters = {
        "code": ParameterSpec(
            type="string", description="The Python code to execute", required=True
        )
    }

    def execute(self, code: str, **kwargs) -> ToolResult:
        """Execute Python code safely with multiprocessing for timeout."""
        import multiprocessing
        from io import StringIO
        import sys

        def run_code_in_process(code, result_queue):
            """Run code in a separate process."""
            try:
                # Capture output
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                stdout_buffer = StringIO()
                stderr_buffer = StringIO()

                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer

                # Create restricted globals
                restricted_globals = {
                    "__builtins__": {
                        "True": True,
                        "False": False,
                        "None": None,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                        "len": len,
                        "range": range,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                        "sum": sum,
                        "min": min,
                        "max": max,
                        "abs": abs,
                        "round": round,
                        "print": print,
                    }
                }

                exec(code, restricted_globals)

                output = stdout_buffer.getvalue()
                error = stderr_buffer.getvalue()

                sys.stdout = old_stdout
                sys.stderr = old_stderr

                if error:
                    result_queue.put(ToolResult.error(f"Execution error: {error}"))
                else:
                    result_queue.put(
                        ToolResult.ok(
                            content=output or "Code executed successfully (no output)",
                            variables={
                                k: str(v)
                                for k, v in restricted_globals.items()
                                if not k.startswith("__")
                            },
                        )
                    )

            except Exception as e:
                result_queue.put(ToolResult.error(f"Execution error: {e}"))

        # Create a queue for the result
        result_queue = multiprocessing.Queue()

        # Create and start the process
        process = multiprocessing.Process(target=run_code_in_process, args=(code, result_queue))
        process.start()

        # Wait for the process to complete with timeout
        process.join(timeout=self.timeout)

        if process.is_alive():
            # Process is still running - terminate it
            process.terminate()
            process.join()
            return ToolResult.error(f"Code execution timed out after {self.timeout} seconds")

        # Get the result
        if not result_queue.empty():
            return result_queue.get()
        else:
            return ToolResult.error("No result from code execution")


class BashTool(BaseTool):
    """Execute bash commands."""

    name = "bash"
    description = "Execute bash commands"

    SAFE_COMMANDS = {
        "cat",
        "cd",
        "cls",
        "dir",
        "echo",
        "find",
        "git",
        "grep",
        "head",
        "ls",
        "more",
        "node",
        "npm",
        "pip",
        "pwd",
        "python",
        "python3",
        "pytest",
        "tail",
        "type",
        "where",
        "which",
        "whoami",
    }

    def __init__(self, allowed_commands: Optional[List[str]] = None, timeout: int = 30):
        self.allowed_commands = set(allowed_commands or self.SAFE_COMMANDS)
        self.timeout = timeout

    parameters = {
        "command": ParameterSpec(
            type="string", description="The bash command to execute", required=True
        )
    }

    def execute(self, command: str, **kwargs) -> ToolResult:
        """Execute bash command."""
        try:
            cmd_parts = shlex.split(command)
        except ValueError as exc:
            return ToolResult.error(f"Invalid command syntax: {exc}")
        if not cmd_parts:
            return ToolResult.error("Empty command")

        cmd_base = Path(cmd_parts[0]).name.lower()
        if cmd_base not in self.allowed_commands:
            return ToolResult.error(f"Command '{cmd_base}' not allowed")

        # Block dangerous commands
        dangerous = ["rm -rf /", "> /dev/sda", "mkfs", "dd if=/dev/zero", ":(){:|:&};:"]
        for d in dangerous:
            if d in command:
                return ToolResult.error(f"Dangerous command blocked: {d}")

        try:
            result = subprocess.run(
                cmd_parts, shell=False, capture_output=True, text=True, timeout=self.timeout
            )

            output = result.stdout
            error = result.stderr

            if result.returncode != 0:
                return ToolResult.error(f"Exit code {result.returncode}: {error}")

            return ToolResult.ok(
                content=output or "Command executed successfully (no output)",
                returncode=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return ToolResult.error(f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            return ToolResult.error(f"Execution error: {e}")


# ==================== Database Tools ====================


class SQLDatabaseTool(BaseTool):
    """Execute SQL queries."""

    name = "sql_database"
    description = "Query a SQL database"

    def __init__(self, connection_string: Optional[str] = None, tables: Optional[List[str]] = None):
        self.connection_string = connection_string
        self.tables = tables

    parameters = {
        "query": ParameterSpec(type="string", description="The SQL query to execute", required=True)
    }

    def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute SQL query."""
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(self.connection_string)

            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(query))

                # Fetch results
                rows = result.fetchall()
                columns = result.keys()

                # Format as string
                lines = [" | ".join(columns)]
                lines.append("-" * len(lines[0]))
                for row in rows[:50]:  # Limit rows
                    lines.append(" | ".join(str(c) for c in row))

                return ToolResult.ok(
                    content="\n".join(lines), columns=list(columns), row_count=len(rows)
                )

        except ImportError:
            return ToolResult.error("sqlalchemy required. Install: pip install sqlalchemy")
        except Exception as e:
            return ToolResult.error(f"SQL error: {e}")


class MongoDBTool(BaseTool):
    """Query MongoDB."""

    name = "mongodb"
    description = "Query a MongoDB database"

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        self.database = database
        self.collection = collection

    parameters = {
        "operation": ParameterSpec(
            type="string",
            description="The operation: find, insert, update, delete",
            required=True,
            enum=["find", "insert", "update", "delete"],
        ),
        "filter": ParameterSpec(
            type="object", description="The query filter (for find, update, delete)", required=False
        ),
        "document": ParameterSpec(
            type="object", description="The document (for insert, update)", required=False
        ),
    }

    def execute(
        self,
        operation: str,
        filter: Optional[Dict] = None,
        document: Optional[Dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute MongoDB operation."""
        try:
            from pymongo import MongoClient

            client = MongoClient(self.connection_string)
            db = client[self.database]
            coll = db[self.collection]

            filter = filter or {}

            if operation == "find":
                results = list(coll.find(filter).limit(10))
                # Convert ObjectId to string
                for r in results:
                    if "_id" in r:
                        r["_id"] = str(r["_id"])

                return ToolResult.ok(content=json.dumps(results, indent=2), results=results)

            elif operation == "insert":
                result = coll.insert_one(document or {})
                return ToolResult.ok(
                    content=f"Inserted document with ID: {result.inserted_id}",
                    inserted_id=str(result.inserted_id),
                )

            elif operation == "update":
                result = coll.update_one(filter, {"$set": document or {}})
                return ToolResult.ok(
                    content=f"Updated {result.modified_count} documents",
                    matched_count=result.matched_count,
                    modified_count=result.modified_count,
                )

            elif operation == "delete":
                result = coll.delete_one(filter)
                return ToolResult.ok(
                    content=f"Deleted {result.deleted_count} documents",
                    deleted_count=result.deleted_count,
                )

            else:
                return ToolResult.error(f"Unknown operation: {operation}")

        except ImportError:
            return ToolResult.error("pymongo required. Install: pip install pymongo")
        except Exception as e:
            return ToolResult.error(f"MongoDB error: {e}")


# ==================== File Tools ====================


class FileReadTool(BaseTool):
    """Read file contents."""

    name = "file_read"
    description = "Read contents of a file"

    def __init__(self, allowed_paths: Optional[List[str]] = None):
        self.allowed_paths = allowed_paths

    parameters = {
        "file_path": ParameterSpec(type="string", description="The path to the file", required=True)
    }

    def execute(self, file_path: str, **kwargs) -> ToolResult:
        """Read file."""
        path = Path(file_path)

        # Security check
        if self.allowed_paths:
            resolved = path.resolve()
            allowed = any(
                str(resolved).startswith(str(Path(p).resolve())) for p in self.allowed_paths
            )
            if not allowed:
                return ToolResult.error(f"Access to {file_path} not allowed")

        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult.ok(
                content=content[:10000], file_path=str(path), size=len(content)  # Limit size
            )
        except Exception as e:
            return ToolResult.error(f"Failed to read file: {e}")


class FileWriteTool(BaseTool):
    """Write to files."""

    name = "file_write"
    description = "Write content to a file"

    def __init__(self, allowed_paths: Optional[List[str]] = None):
        self.allowed_paths = allowed_paths

    parameters = {
        "file_path": ParameterSpec(
            type="string", description="The path to the file", required=True
        ),
        "content": ParameterSpec(type="string", description="The content to write", required=True),
        "append": ParameterSpec(
            type="boolean",
            description="Append to file instead of overwrite",
            required=False,
            default=False,
        ),
    }

    def execute(self, file_path: str, content: str, append: bool = False, **kwargs) -> ToolResult:
        """Write file."""
        path = Path(file_path)

        # Security check
        if self.allowed_paths:
            resolved = path.resolve()
            allowed = any(
                str(resolved).startswith(str(Path(p).resolve())) for p in self.allowed_paths
            )
            if not allowed:
                return ToolResult.error(f"Access to {file_path} not allowed")

        try:
            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

            return ToolResult.ok(
                content=f"File written successfully: {path}",
                file_path=str(path),
                bytes_written=len(content),
            )
        except Exception as e:
            return ToolResult.error(f"Failed to write file: {e}")


class ListDirectoryTool(BaseTool):
    """List directory contents."""

    name = "list_directory"
    description = "List files and directories"

    def __init__(self, allowed_paths: Optional[List[str]] = None):
        self.allowed_paths = allowed_paths

    parameters = {
        "path": ParameterSpec(type="string", description="The directory path", required=True),
        "recursive": ParameterSpec(
            type="boolean", description="List recursively", required=False, default=False
        ),
    }

    def execute(self, path: str, recursive: bool = False, **kwargs) -> ToolResult:
        """List directory."""
        dir_path = Path(path)

        # Security check
        if self.allowed_paths:
            resolved = dir_path.resolve()
            allowed = any(
                str(resolved).startswith(str(Path(p).resolve())) for p in self.allowed_paths
            )
            if not allowed:
                return ToolResult.error(f"Access to {path} not allowed")

        try:
            if recursive:
                items = list(dir_path.rglob("*"))
            else:
                items = list(dir_path.iterdir())

            # Format listing
            lines = []
            for item in items[:100]:  # Limit items
                item_type = "📁" if item.is_dir() else "📄"
                lines.append(f"{item_type} {item.relative_to(dir_path)}")

            return ToolResult.ok(
                content="\n".join(lines) or "Empty directory",
                path=str(dir_path),
                item_count=len(items),
            )
        except Exception as e:
            return ToolResult.error(f"Failed to list directory: {e}")


# ==================== API Tools ====================


class APITool(BaseTool):
    """Generic API tool."""

    name = "api_request"
    description = "Make HTTP API requests"

    parameters = {
        "url": ParameterSpec(type="string", description="The API URL", required=True),
        "method": ParameterSpec(
            type="string",
            description="HTTP method",
            required=False,
            default="GET",
            enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
        ),
        "headers": ParameterSpec(type="object", description="Request headers", required=False),
        "body": ParameterSpec(
            type="object", description="Request body (for POST/PUT/PATCH)", required=False
        ),
    }

    def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        body: Optional[Dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Make API request."""
        try:
            headers = headers or {}

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=body, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=body, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                return ToolResult.error(f"Unsupported method: {method}")

            response.raise_for_status()

            # Try to parse as JSON
            try:
                data = response.json()
                content = json.dumps(data, indent=2)
            except:
                content = response.text[:5000]

            return ToolResult.ok(
                content=content, status_code=response.status_code, headers=dict(response.headers)
            )

        except Exception as e:
            return ToolResult.error(f"API request failed: {e}")


# Convenience functions
def create_tool(tool_type: str, **kwargs) -> BaseTool:
    """Create a tool by type."""
    tools = {
        "google_search": GoogleSearchTool,
        "wikipedia": WikipediaTool,
        "web_scraper": WebScraperTool,
        "python_repl": PythonREPLTool,
        "bash": BashTool,
        "sql": SQLDatabaseTool,
        "mongodb": MongoDBTool,
        "file_read": FileReadTool,
        "file_write": FileWriteTool,
        "list_directory": ListDirectoryTool,
        "api": APITool,
    }

    tool_cls = tools.get(tool_type.lower())
    if not tool_cls:
        raise ValueError(f"Unknown tool type: {tool_type}")

    return tool_cls(**kwargs)


__all__ = [
    # Base
    "ToolResult",
    "BaseTool",
    # Web & Search
    "GoogleSearchTool",
    "WikipediaTool",
    "WebScraperTool",
    # Code
    "PythonREPLTool",
    "BashTool",
    # Databases
    "SQLDatabaseTool",
    "MongoDBTool",
    # Files
    "FileReadTool",
    "FileWriteTool",
    "ListDirectoryTool",
    # API
    "APITool",
    # Convenience
    "create_tool",
]
