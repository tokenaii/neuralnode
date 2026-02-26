"""Tools module for agent integration."""

from typing import Any, Dict, List, Optional
import json
import requests


class WebSearch:
    """Web search tool using various search APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "duckduckgo",
    ):
        self.api_key = api_key
        self.search_engine = search_engine

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        if self.search_engine == "duckduckgo":
            return self._search_duckduckgo(query, num_results)
        elif self.search_engine == "serpapi":
            return self._search_serpapi(query, num_results)
        else:
            raise ValueError(f"Unknown search engine: {self.search_engine}")

    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except ImportError:
            raise ImportError(
                "duckduckgo-search is required. "
                "Install with: pip install duckduckgo-search"
            )

    def _search_serpapi(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search using SerpAPI."""
        if not self.api_key:
            raise ValueError("SerpAPI requires an API key")
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results,
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        
        return results


class BrowserTools:
    """Browser automation tools."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._driver = None

    def _get_driver(self):
        """Get or create Selenium WebDriver."""
        if self._driver is None:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                self._driver = webdriver.Chrome(options=options)
            except ImportError:
                raise ImportError(
                    "selenium is required for browser automation. "
                    "Install with: pip install selenium"
                )
        return self._driver

    def navigate(self, url: str) -> str:
        """Navigate to a URL."""
        driver = self._get_driver()
        driver.get(url)
        return f"Navigated to {url}"

    def get_page_text(self) -> str:
        """Get text content of current page."""
        driver = self._get_driver()
        return driver.find_element("tag name", "body").text

    def search_and_click(self, query: str, link_text: Optional[str] = None) -> str:
        """Search and click on a result."""
        # Navigate to Google
        self.navigate(f"https://www.google.com/search?q={query}")
        
        # If link_text provided, try to click it
        if link_text:
            driver = self._get_driver()
            try:
                link = driver.find_element("partial link text", link_text)
                link.click()
                return f"Clicked on '{link_text}'"
            except Exception as e:
                return f"Could not click '{link_text}': {e}"
        
        return "Search results loaded"

    def fill_form(self, form_data: Dict[str, str]) -> str:
        """Fill form fields."""
        driver = self._get_driver()
        
        for field_name, value in form_data.items():
            try:
                # Try finding by name, id, or placeholder
                for strategy in ["name", "id"]:
                    try:
                        element = driver.find_element(strategy, field_name)
                        element.clear()
                        element.send_keys(value)
                        break
                    except:
                        continue
            except Exception as e:
                return f"Could not fill field '{field_name}': {e}"
        
        return "Form filled"

    def close(self) -> None:
        """Close browser."""
        if self._driver:
            self._driver.quit()
            self._driver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Calculator:
    """Calculator tool for mathematical operations."""

    def calculate(self, expression: str) -> str:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Math expression like "2 + 2 * 3"
            
        Returns:
            Result as string
        """
        try:
            # Use eval with restricted globals/locals
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "sum": sum,
            }
            
            # Clean the expression
            expression = expression.replace("^", "**")
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class CodeRunner:
    """Code execution tool (use with caution)."""

    def __init__(self, allowed_languages: Optional[List[str]] = None):
        self.allowed_languages = allowed_languages or ["python"]

    def run(self, code: str, language: str = "python") -> str:
        """
        Run code (Python only, in restricted environment).
        
        Args:
            code: Code to run
            language: Programming language
            
        Returns:
            Output as string
        """
        if language not in self.allowed_languages:
            return f"Language '{language}' not allowed"
        
        if language == "python":
            return self._run_python(code)
        
        return f"Language '{language}' not supported"

    def _run_python(self, code: str) -> str:
        """Run Python code safely."""
        import io
        import sys
        
        # Create restricted environment
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        # Restricted globals
        restricted_globals = {
            "__builtins__": {
                "True": True,
                "False": False,
                "None": None,
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
                "pow": pow,
                "round": round,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "print": lambda *args, **kwargs: stdout.write(" ".join(map(str, args)) + "\n"),
            }
        }
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            exec(code, restricted_globals)
            output = stdout.getvalue()
            if not output:
                output = "Code executed successfully (no output)"
            return output
        except Exception as e:
            return f"Error: {e}\n{stderr.getvalue()}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_default_tools() -> Dict[str, Any]:
    """Get dictionary of default tools."""
    return {
        "calculator": Calculator(),
        "web_search": WebSearch(),
    }
