"""Test tools system."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import neuralnode as nn
from neuralnode.core import ToolResult


class TestWebSearchTool:
    """Test WebSearchTool."""
    
    def test_init(self):
        """Test tool initialization."""
        tool = nn.WebSearchTool()
        assert tool.name == "web_search"
        assert "DuckDuckGo" in tool.description
    
    def test_get_schema(self):
        """Test schema generation."""
        tool = nn.WebSearchTool()
        schema = tool.get_schema()
        assert schema["type"] == "function"
        assert "query" in schema["function"]["parameters"]["properties"]
    
    @patch('urllib.request.urlopen')
    def test_execute_success(self, mock_urlopen):
        """Test successful search."""
        # Mock HTML response
        html = '''
        <html>
        <div class="result">
            <a class="result__a" href="http://example.com">Example Title</a>
            <a class="result__snippet">Example snippet text</a>
        </div>
        </html>
        '''
        mock_response = MagicMock()
        mock_response.read.return_value = html.encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        tool = nn.WebSearchTool()
        result = tool.execute("test query", num_results=1)
        
        assert result.success is True
        assert "Example Title" in result.content
    
    def test_execute_empty_query(self):
        """Test with empty query."""
        tool = nn.WebSearchTool()
        result = tool.execute("")
        # Should either return results or error gracefully
        assert isinstance(result, ToolResult)


class TestFileTools:
    """Test file system tools."""
    
    def test_file_read_tool(self, tmp_path):
        """Test FileReadTool."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        tool = nn.FileReadTool(allowed_paths=[str(tmp_path)])
        result = tool.execute(path=str(test_file))
        
        assert result.success is True
        assert result.content == "Hello World"
    
    def test_file_read_not_allowed(self, tmp_path):
        """Test reading outside allowed paths."""
        tool = nn.FileReadTool(allowed_paths=["/other/path"])
        result = tool.execute(path=str(tmp_path / "test.txt"))
        
        assert result.success is False
        assert "not allowed" in result.content.lower()
    
    def test_file_write_tool(self, tmp_path):
        """Test FileWriteTool."""
        test_file = tmp_path / "output.txt"
        
        tool = nn.FileWriteTool(allowed_paths=[str(tmp_path)])
        result = tool.execute(
            path=str(test_file),
            content="Test content"
        )
        
        assert result.success is True
        assert test_file.read_text() == "Test content"
    
    def test_list_directory_tool(self, tmp_path):
        """Test ListDirectoryTool."""
        # Create some files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        
        tool = nn.ListDirectoryTool(allowed_paths=[str(tmp_path)])
        result = tool.execute(path=str(tmp_path))
        
        assert result.success is True
        assert "file1.txt" in result.content
        assert "file2.txt" in result.content


class TestCommandRunner:
    """Test CommandRunner tool."""
    
    def test_allowed_command(self):
        """Test running allowed command."""
        tool = nn.CommandRunner()
        result = tool.execute(command="echo hello")
        
        assert result.success is True
        assert "hello" in result.output["stdout"]
    
    def test_blocked_command(self):
        """Test running blocked command."""
        tool = nn.CommandRunner()
        result = tool.execute(command="rm -rf /")
        
        assert result.success is False
        assert "not allowed" in result.content.lower()
    
    def test_dangerous_chars(self):
        """Test command with dangerous characters."""
        tool = nn.CommandRunner()
        result = tool.execute(command="ls; rm -rf /")
        
        assert result.success is False


class TestSystemInfo:
    """Test SystemInfo tool."""
    
    def test_get_info(self):
        """Test getting system info."""
        tool = nn.SystemInfo()
        result = tool.execute()
        
        assert result.success is True
        assert "platform" in result.data
        assert "python_version" in result.data


class TestURLFetcher:
    """Test URLFetcher tool."""
    
    @patch('urllib.request.urlopen')
    def test_fetch_success(self, mock_urlopen):
        """Test successful URL fetch."""
        html = "<html><body>Test content</body></html>"
        mock_response = MagicMock()
        mock_response.read.return_value = html.encode()
        mock_response.headers.get.return_value = "text/html"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        tool = nn.URLFetcher()
        result = tool.execute(url="http://example.com")
        
        assert result.success is True
        assert "Test content" in result.content
    
    def test_fetch_pdf(self):
        """Test handling PDF URLs."""
        tool = nn.URLFetcher()
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "application/pdf"
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = tool.execute(url="http://example.com/doc.pdf")
            
            assert result.success is True
            assert "PDF" in result.content


class TestUniversalSearch:
    """Test UniversalSearch tool."""
    
    def test_init(self):
        """Test initialization."""
        tool = nn.UniversalSearch()
        assert tool.name == "multi_source_search"
    
    def test_list_sources(self):
        """Test listing available sources."""
        tool = nn.UniversalSearch()
        sources = tool.list_available_sources()
        
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "duckduckgo" in sources


class TestToolResult:
    """Test ToolResult helper methods."""
    
    def test_ok_factory(self):
        """Test ok factory method."""
        result = ToolResult.ok("Success", data={"key": "value"})
        
        assert result.success is True
        assert result.content == "Success"
        assert result.data["key"] == "value"
    
    def test_error_factory(self):
        """Test error factory method."""
        result = ToolResult.error("Failed", error_code=500)
        
        assert result.success is False
        assert "Failed" in result.content
        assert result.error_code == 500
    
    def test_to_dict(self):
        """Test serialization."""
        result = ToolResult.ok("Test")
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["content"] == "Test"
