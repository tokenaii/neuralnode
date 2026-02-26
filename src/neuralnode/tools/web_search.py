"""Web search tools for agents."""

import json
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.error import HTTPError
import ssl

# Import Tool base class
from ..tools import Tool


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str = ""


class DuckDuckGoSearch:
    """DuckDuckGo search (free, no API key needed)."""
    
    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate"):
        self.region = region
        self.safesearch = safesearch
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo."""
        try:
            # Use DuckDuckGo HTML interface
            url = "https://html.duckduckgo.com/html/"
            data = urllib.parse.urlencode({"q": query, "kl": self.region})
            data = data.encode("utf-8")
            
            # Create request with headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = urllib.request.Request(url, data=data, headers=headers)
            
            # Make request with SSL context
            context = ssl.create_default_context()
            response = urllib.request.urlopen(req, context=context, timeout=30)
            html = response.read().decode("utf-8")
            
            return self._parse_results(html, num_results)
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _parse_results(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse HTML results."""
        results = []
        
        # Simple regex parsing for DuckDuckGo HTML
        # Find result blocks
        pattern = r'<a class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.DOTALL)
        
        for i, (url, title, snippet) in enumerate(matches[:num_results]):
            # Clean HTML tags
            title = re.sub(r'<[^>]+>', '', title)
            snippet = re.sub(r'<[^>]+>', '', snippet)
            
            # Decode HTML entities
            import html as html_module
            title = html_module.unescape(title)
            snippet = html_module.unescape(snippet)
            
            results.append(SearchResult(
                title=title.strip(),
                url=urllib.parse.unquote(url),
                snippet=snippet.strip(),
                source="DuckDuckGo"
            ))
        
        return results


class BingSearch:
    """Bing search (requires API key)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search Bing."""
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = urllib.parse.urlencode({"q": query, "count": num_results})
            url = f"{self.endpoint}?{params}"
            
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read().decode("utf-8"))
            
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="Bing"
                ))
            
            return results
        except Exception as e:
            print(f"Bing search error: {e}")
            return []


class SerpAPISearch:
    """SerpAPI search (requires API key, supports Google, Bing, etc.)."""
    
    def __init__(self, api_key: str, engine: str = "google"):
        self.api_key = api_key
        self.engine = engine
        self.endpoint = "https://serpapi.com/search"
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search via SerpAPI."""
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "engine": self.engine,
                "num": num_results,
                "api_key": self.api_key,
                "output": "json"
            })
            url = f"{self.endpoint}?{params}"
            
            response = urllib.request.urlopen(url, timeout=30)
            data = json.loads(response.read().decode("utf-8"))
            
            results = []
            
            # Parse based on engine
            if self.engine == "google":
                for item in data.get("organic_results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="Google (SerpAPI)"
                    ))
            elif self.engine == "bing":
                for item in data.get("organic_results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="Bing (SerpAPI)"
                    ))
            
            return results
        except Exception as e:
            print(f"SerpAPI search error: {e}")
            return []


class WebSearch(Tool):
    """Unified web search with fallback options."""
    
    def __init__(
        self,
        duckduckgo: bool = True,
        bing_api_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
    ):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results"}
            }
        )
        self.engines: List[Any] = []
        
        if duckduckgo:
            self.engines.append(DuckDuckGoSearch())
        
        if bing_api_key:
            self.engines.append(BingSearch(bing_api_key))
        
        if serpapi_key:
            self.engines.append(SerpAPISearch(serpapi_key))
    
    def execute(self, query: str, num_results: int = 10) -> str:
        """Execute web search."""
        results = self.search(query, num_results)
        return json.dumps([{
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet
        } for r in results], ensure_ascii=False, indent=2)
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using available engines."""
        all_results = []
        
        for engine in self.engines:
            try:
                results = engine.search(query, num_results)
                all_results.extend(results)
                
                # If we got results, return them
                if results:
                    return results[:num_results]
            except Exception as e:
                continue
        
        # Return all collected results if no engine succeeded individually
        return all_results[:num_results]
    
    def search_with_summarization(
        self,
        query: str,
        llm=None,
        num_results: int = 5,
    ) -> Dict[str, Any]:
        """Search and summarize results using LLM."""
        results = self.search(query, num_results)
        
        if not results:
            return {"results": [], "summary": "No results found."}
        
        if llm:
            # Create context from results
            context = "\n\n".join([
                f"Source {i+1}: {r.title}\n{r.snippet}\nURL: {r.url}"
                for i, r in enumerate(results)
            ])
            
            prompt = f"""Based on the following search results, provide a comprehensive answer to: {query}

Search Results:
{context}

Provide a well-structured answer with citations to the sources."""
            
            try:
                response = llm.chat(prompt)
                summary = response.text if hasattr(response, 'text') else str(response)
            except Exception as e:
                summary = f"Error generating summary: {e}"
        else:
            summary = "LLM not available for summarization."
        
        return {
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                }
                for r in results
            ],
            "summary": summary,
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent."""
        return {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    def as_tool(self, llm=None):
        """Convert to a tool that can be used with agents."""
        from ..agents.agent import create_tool
        
        @create_tool(
            name="web_search",
            description="Search the web for current information. Returns a summary and list of sources.",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results"}
            }
        )
        def search_tool(query: str, num_results: int = 5) -> str:
            result = self.search_with_summarization(query, llm, num_results)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        return search_tool


class NewsSearch(WebSearch):
    """Specialized news search."""
    
    def search_news(self, topic: str, days: int = 7, num_results: int = 10) -> List[SearchResult]:
        """Search for recent news."""
        query = f"{topic} news last {days} days"
        return self.search(query, num_results)


class ImageSearch(WebSearch):
    """Image search."""
    
    def search_images(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Search for images."""
        # Use DuckDuckGo image search
        try:
            url = "https://duckduckgo.com/"
            params = urllib.parse.urlencode({"q": query, "iax": "images", "ia": "images"})
            # This is a simplified version - full implementation would parse image results
            return [{"query": query, "note": "Image search requires full browser automation"}]
        except Exception as e:
            return [{"error": str(e)}]


# Convenience functions
def search(query: str, num_results: int = 5) -> List[SearchResult]:
    """Quick search using DuckDuckGo."""
    searcher = DuckDuckGoSearch()
    return searcher.search(query, num_results)


def search_with_answer(query: str, llm=None, num_results: int = 5) -> str:
    """Search and get an answer."""
    searcher = WebSearch()
    result = searcher.search_with_summarization(query, llm, num_results)
    return result.get("summary", "No answer available.")
