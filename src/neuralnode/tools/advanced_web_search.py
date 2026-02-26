"""Professional Web Search using proper APIs."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class SearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    score: float = 0.0


class TavilySearch:
    """
    Tavily AI Search API - Professional search for AI agents.
    Website: https://tavily.com
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key required. Get one at https://tavily.com")
        self.base_url = "https://api.tavily.com"
    
    def search(
        self,
        query: str,
        search_depth: str = "basic",  # "basic" or "advanced"
        include_answer: bool = True,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            search_depth: "basic" or "advanced" (advanced costs more)
            include_answer: Include AI-generated answer
            max_results: Maximum results to return
            include_domains: Whitelist domains
            exclude_domains: Blacklist domains
        """
        import requests
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "max_results": max_results,
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            return {
                "error": str(e),
                "results": [],
                "answer": None,
            }
    
    def extract_content(self, urls: List[str]) -> Dict[str, Any]:
        """Extract clean content from URLs."""
        import requests
        
        payload = {
            "api_key": self.api_key,
            "urls": urls,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/extract",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            return {"error": str(e), "results": []}


class DuckDuckGoSearch:
    """
    DuckDuckGo Search using official duckduckgo-search library.
    No API key required!
    """
    
    def __init__(self):
        self._ddgs = None
        self._init_ddgs()
    
    def _init_ddgs(self):
        """Initialize DuckDuckGo search."""
        try:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        except ImportError:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",  # Worldwide
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,  # "d", "w", "m", "y"
    ) -> List[SearchResult]:
        """
        Search DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum results
            region: Region code (e.g., "us-en", "uk-en")
            safesearch: "on", "moderate", or "off"
            timelimit: Time limit ("d"=day, "w"=week, "m"=month, "y"=year)
        """
        try:
            results = self._ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results,
            )
            
            search_results = []
            for r in results:
                search_results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source="DuckDuckGo",
                    score=r.get("score", 0.0),
                ))
            
            return search_results
        
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def news(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        timelimit: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search news articles."""
        try:
            results = self._ddgs.news(
                keywords=query,
                region=region,
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            )
            
            search_results = []
            for r in results:
                search_results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("body", ""),
                    source=r.get("source", "News"),
                    published_date=r.get("date"),
                ))
            
            return search_results
        
        except Exception as e:
            print(f"News search error: {e}")
            return []
    
    def images(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search images."""
        try:
            results = self._ddgs.images(
                keywords=query,
                max_results=max_results,
            )
            return list(results)
        
        except Exception as e:
            print(f"Image search error: {e}")
            return []


class SerpAPISearch:
    """
    SerpAPI - Professional search API supporting Google, Bing, etc.
    Website: https://serpapi.com
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SerpAPI key required. Get one at https://serpapi.com")
        self.base_url = "https://serpapi.com/search"
    
    def search(
        self,
        query: str,
        engine: str = "google",
        num_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search using SerpAPI.
        
        Args:
            query: Search query
            engine: Search engine ("google", "bing", "yahoo", etc.)
            num_results: Number of results
            **kwargs: Additional parameters
        """
        import requests
        
        params = {
            "q": query,
            "engine": engine,
            "num": num_results,
            "api_key": self.api_key,
            "output": "json",
        }
        params.update(kwargs)
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def parse_results(self, response: Dict) -> List[SearchResult]:
        """Parse SerpAPI response into SearchResult objects."""
        results = []
        
        organic_results = response.get("organic_results", [])
        for r in organic_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                source="SerpAPI",
                published_date=r.get("date"),
            ))
        
        return results


class BingSearch:
    """
    Microsoft Bing Search API.
    Website: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BING_API_KEY")
        if not self.api_key:
            raise ValueError("Bing API key required")
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    def search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        mkt: str = "en-US",
        safesearch: str = "Moderate",
    ) -> List[SearchResult]:
        """Search using Bing API."""
        import requests
        
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": count,
            "offset": offset,
            "mkt": mkt,
            "safesearch": safesearch,
        }
        
        try:
            response = requests.get(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            web_pages = data.get("webPages", {}).get("value", [])
            for page in web_pages:
                results.append(SearchResult(
                    title=page.get("name", ""),
                    url=page.get("url", ""),
                    snippet=page.get("snippet", ""),
                    source="Bing",
                    published_date=page.get("dateLastCrawled"),
                ))
            
            return results
        
        except requests.RequestException as e:
            print(f"Bing search error: {e}")
            return []


class WebSearch:
    """
    Unified Web Search with automatic provider selection.
    Uses best available search provider.
    """
    
    PROVIDER_PRIORITY = ["tavily", "bing", "serpapi", "duckduckgo"]
    
    def __init__(self, provider: str = "auto", **kwargs):
        """
        Initialize web search.
        
        Args:
            provider: "auto", "tavily", "duckduckgo", "serpapi", or "bing"
            **kwargs: Provider-specific arguments
        """
        self.provider = provider
        self._search_fn = None
        self._init_provider(**kwargs)
    
    def _init_provider(self, **kwargs):
        """Initialize search provider."""
        if self.provider == "auto":
            # Try providers in priority order
            for provider_name in self.PROVIDER_PRIORITY:
                try:
                    self._try_init_provider(provider_name, **kwargs)
                    self.provider = provider_name
                    print(f"✅ Using search provider: {provider_name}")
                    return
                except (ImportError, ValueError) as e:
                    continue
            
            raise RuntimeError("No search provider available")
        
        else:
            self._try_init_provider(self.provider, **kwargs)
    
    def _try_init_provider(self, name: str, **kwargs):
        """Try to initialize a specific provider."""
        if name == "tavily":
            self._tavily = TavilySearch(kwargs.get("api_key"))
            self._search_fn = self._search_tavily
        
        elif name == "duckduckgo":
            self._ddg = DuckDuckGoSearch()
            self._search_fn = self._search_duckduckgo
        
        elif name == "serpapi":
            self._serpapi = SerpAPISearch(kwargs.get("api_key"))
            self._search_fn = self._search_serpapi
        
        elif name == "bing":
            self._bing = BingSearch(kwargs.get("api_key"))
            self._search_fn = self._search_bing
        
        else:
            raise ValueError(f"Unknown provider: {name}")
    
    def _search_tavily(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using Tavily."""
        results = self._tavily.search(
            query,
            max_results=kwargs.get("max_results", 5),
            search_depth=kwargs.get("search_depth", "basic"),
        )
        
        tavily_results = results.get("results", [])
        search_results = []
        
        for r in tavily_results:
            search_results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source="Tavily",
                score=r.get("score", 0.0),
            ))
        
        return search_results
    
    def _search_duckduckgo(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        return self._ddg.search(
            query,
            max_results=kwargs.get("max_results", 5),
        )
    
    def _search_serpapi(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using SerpAPI."""
        response = self._serpapi.search(
            query,
            num_results=kwargs.get("max_results", 5),
        )
        return self._serpapi.parse_results(response)
    
    def _search_bing(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using Bing."""
        return self._bing.search(
            query,
            count=kwargs.get("max_results", 5),
        )
    
    def search(self, query: str, max_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional provider-specific parameters
        
        Returns:
            List of SearchResult objects
        """
        return self._search_fn(query, max_results=max_results, **kwargs)
    
    def search_with_answer(self, query: str) -> Dict[str, Any]:
        """
        Search and get AI-generated answer (Tavily only).
        
        Returns:
            Dict with 'answer' and 'results'
        """
        if hasattr(self, '_tavily'):
            result = self._tavily.search(query, include_answer=True)
            return {
                "answer": result.get("answer"),
                "results": [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        source="Tavily",
                    )
                    for r in result.get("results", [])
                ],
            }
        
        # Fallback to regular search
        results = self.search(query)
        return {
            "answer": None,
            "results": results,
        }


# Convenience functions
def quick_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Quick search using best available provider."""
    searcher = WebSearch(provider="auto")
    return searcher.search(query, max_results=max_results)


def search_news(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search news articles."""
    ddg = DuckDuckGoSearch()
    return ddg.news(query, max_results=max_results)
