"""Web search tools for NeuralNode."""

import re
import json
import urllib.request
import urllib.parse
import ssl
from typing import List, Dict, Any, Optional
from html import unescape

from ...core import BaseTool, ToolResult, ParameterSpec
# Import SearchResult from multisearch to avoid duplication
from ..multisearch import SearchResult


class WebSearchTool(BaseTool):
    """Web search using DuckDuckGo (no API key required)."""
    
    name = "web_search"
    description = "Search the web for current information on any topic. Returns search results with titles, URLs, and snippets."
    parameters = {
        "query": ParameterSpec(
            type="string",
            description="The search query to look up",
            required=True
        ),
        "num_results": ParameterSpec(
            type="integer",
            description="Number of results to return (1-10)",
            required=False,
            default=5
        )
    }
    
    def __init__(self):
        super().__init__()
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """Execute web search."""
        try:
            results = self._search_duckduckgo(query, min(num_results, 10))
            
            if not results:
                return ToolResult.error("No search results found. Try a different query.")
            
            formatted = self._format_results(results)
            return ToolResult.ok(
                content=formatted,
                results=[r.__dict__ for r in results]
            )
        except Exception as e:
            return ToolResult.error(f"Search failed: {str(e)}")
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface."""
        # DuckDuckGo HTML endpoint
        url = "https://html.duckduckgo.com/html/"
        
        # Encode the query
        data = urllib.parse.urlencode({"q": query, "kl": "wt-wt"})
        data = data.encode("utf-8")
        
        # Create request with headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        # Make request
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        return self._parse_results(html, num_results)
    
    def _parse_results(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse search results from HTML."""
        results = []
        
        # Find result blocks - DuckDuckGo uses 'result' class
        # Pattern: result blocks with title link and snippet
        result_blocks = re.findall(
            r'<div class="result[^"]*"[^>]*>.*?<\/div>\s*<\/div>\s*<\/div>',
            html,
            re.DOTALL
        )
        
        for block in result_blocks[:num_results]:
            # Extract title and URL
            title_match = re.search(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>',
                block,
                re.DOTALL
            )
            
            # Extract snippet
            snippet_match = re.search(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)<\/a>',
                block,
                re.DOTALL
            )
            
            if title_match:
                url = unescape(title_match.group(1))
                title = self._clean_html(title_match.group(2))
                snippet = self._clean_html(snippet_match.group(1)) if snippet_match else ""
                
                # Skip ad results
                if "duckduckgo.com/y.js" not in url:
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="DuckDuckGo"
                    ))
        
        return results
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        # Remove tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = unescape(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Format results for display."""
        lines = ["Web Search Results:"]
        
        for i, r in enumerate(results, 1):
            lines.extend([
                f"\n{i}. {r.title}",
                f"   URL: {r.url}",
                f"   {r.snippet[:200]}{'...' if len(r.snippet) > 200 else ''}"
            ])
        
        return "\n".join(lines)


class NewsSearchTool(WebSearchTool):
    """News search using web search with news filter."""
    
    name = "news_search"
    description = "Search for recent news articles on a topic. Returns news results with titles, URLs, and snippets."
    parameters = {
        "query": ParameterSpec(
            type="string",
            description="The news topic to search for",
            required=True
        ),
        "num_results": ParameterSpec(
            type="integer",
            description="Number of results to return (1-10)",
            required=False,
            default=5
        ),
        "days": ParameterSpec(
            type="integer",
            description="How recent the news should be (days)",
            required=False,
            default=7
        )
    }
    
    def execute(self, query: str, num_results: int = 5, days: int = 7) -> ToolResult:
        """Execute news search."""
        # Add news-specific terms to query
        news_query = f"{query} news"
        
        try:
            results = self._search_duckduckgo(news_query, min(num_results, 10))
            
            if not results:
                return ToolResult.error("No news results found. Try a different query.")
            
            formatted = self._format_results(results)
            return ToolResult.ok(
                content=formatted,
                results=[r.__dict__ for r in results]
            )
        except Exception as e:
            return ToolResult.error(f"News search failed: {str(e)}")


class URLFetcherTool(BaseTool):
    """Fetch and extract content from a URL."""
    
    name = "fetch_url"
    description = "Fetch the content of a webpage given its URL. Returns the page text content."
    parameters = {
        "url": ParameterSpec(
            type="string",
            description="The URL to fetch",
            required=True
        ),
        "max_length": ParameterSpec(
            type="integer",
            description="Maximum characters to return",
            required=False,
            default=5000
        )
    }
    
    def __init__(self):
        super().__init__()
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def execute(self, url: str, max_length: int = 5000) -> ToolResult:
        """Fetch URL content."""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
                content_type = response.headers.get('Content-Type', '')
                
                # Handle binary content
                if 'application/pdf' in content_type:
                    return ToolResult.ok(
                        content=f"PDF document at {url} - cannot extract text directly",
                        url=url,
                        type="pdf"
                    )
                
                # Read HTML
                html = response.read().decode('utf-8', errors='ignore')
            
            # Extract text
            text = self._extract_text(html)
            
            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"
            
            return ToolResult.ok(
                content=text,
                url=url,
                length=len(text)
            )
            
        except urllib.error.HTTPError as e:
            return ToolResult.error(f"HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            return ToolResult.error(f"URL Error: {e.reason}")
        except Exception as e:
            return ToolResult.error(f"Failed to fetch URL: {str(e)}")
    
    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?<\/script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?<\/style>', '', html, flags=re.DOTALL)
        
        # Remove common navigation/footer elements
        html = re.sub(r'<nav[^>]*>.*?<\/nav>', '', html, flags=re.DOTALL)
        html = re.sub(r'<footer[^>]*>.*?<\/footer>', '', html, flags=re.DOTALL)
        html = re.sub(r'<header[^>]*>.*?<\/header>', '', html, flags=re.DOTALL)
        
        # Convert common block elements to newlines
        html = re.sub(r'<(p|div|h[1-6]|li)[^>]*>', '\n', html)
        html = re.sub(r'<br\s*\/?>', '\n', html)
        
        # Remove remaining tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode entities
        html = unescape(html)
        
        # Clean up whitespace
        html = re.sub(r'\n\s*\n+', '\n\n', html)
        html = re.sub(r'[ \t]+', ' ', html)
        
        return html.strip()
