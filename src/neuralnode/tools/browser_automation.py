"""Browser automation with Playwright for agent control."""

import asyncio
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import base64
from io import BytesIO

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not installed. Browser automation disabled.")
    print("Install with: pip install playwright && playwright install")


@dataclass
class BrowserAction:
    """A browser action result."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class BrowserController:
    """Full browser control for AI agents."""
    
    def __init__(self, headless: bool = True, viewport: Optional[Dict[str, int]] = None):
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._history: List[Dict[str, Any]] = []
    
    async def start(self, browser_type: str = "chromium") -> BrowserAction:
        """Start browser session."""
        if not PLAYWRIGHT_AVAILABLE:
            return BrowserAction(False, "Playwright not installed")
        
        try:
            self._playwright = await async_playwright().start()
            
            if browser_type == "chromium":
                self._browser = await self._playwright.chromium.launch(headless=self.headless)
            elif browser_type == "firefox":
                self._browser = await self._playwright.firefox.launch(headless=self.headless)
            elif browser_type == "webkit":
                self._browser = await self._playwright.webkit.launch(headless=self.headless)
            else:
                return BrowserAction(False, f"Unknown browser: {browser_type}")
            
            self._context = await self._browser.new_context(viewport=self.viewport)
            self._page = await self._context.new_page()
            
            return BrowserAction(True, f"Browser started: {browser_type}")
        except Exception as e:
            return BrowserAction(False, f"Failed to start browser: {str(e)}")
    
    async def stop(self) -> BrowserAction:
        """Stop browser session."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            
            return BrowserAction(True, "Browser stopped")
        except Exception as e:
            return BrowserAction(False, f"Error stopping browser: {str(e)}")
    
    async def navigate(self, url: str) -> BrowserAction:
        """Navigate to URL."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            response = await self._page.goto(url, wait_until="networkidle", timeout=30000)
            self._history.append({"action": "navigate", "url": url})
            
            status = response.status if response else "unknown"
            return BrowserAction(True, f"Navigated to {url} (status: {status})")
        except Exception as e:
            return BrowserAction(False, f"Navigation failed: {str(e)}")
    
    async def click(self, selector: str) -> BrowserAction:
        """Click element by selector."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.click(selector, timeout=10000)
            self._history.append({"action": "click", "selector": selector})
            return BrowserAction(True, f"Clicked: {selector}")
        except Exception as e:
            return BrowserAction(False, f"Click failed: {str(e)}")
    
    async def fill(self, selector: str, text: str) -> BrowserAction:
        """Fill input field."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.fill(selector, text, timeout=10000)
            self._history.append({"action": "fill", "selector": selector, "text": text})
            return BrowserAction(True, f"Filled {selector} with: {text[:50]}...")
        except Exception as e:
            return BrowserAction(False, f"Fill failed: {str(e)}")
    
    async def type(self, selector: str, text: str, delay: int = 50) -> BrowserAction:
        """Type text with delay (human-like)."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.type(selector, text, delay=delay, timeout=10000)
            return BrowserAction(True, f"Typed: {text[:50]}...")
        except Exception as e:
            return BrowserAction(False, f"Type failed: {str(e)}")
    
    async def press(self, key: str) -> BrowserAction:
        """Press keyboard key."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.keyboard.press(key)
            return BrowserAction(True, f"Pressed: {key}")
        except Exception as e:
            return BrowserAction(False, f"Key press failed: {str(e)}")
    
    async def get_text(self, selector: Optional[str] = None) -> BrowserAction:
        """Get text from element or page."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    return BrowserAction(True, f"Text from {selector}", {"text": text})
                return BrowserAction(False, f"Element not found: {selector}")
            else:
                # Get all page text
                text = await self._page.evaluate("() => document.body.innerText")
                return BrowserAction(True, "Page text extracted", {"text": text})
        except Exception as e:
            return BrowserAction(False, f"Get text failed: {str(e)}")
    
    async def get_html(self) -> BrowserAction:
        """Get page HTML."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            html = await self._page.content()
            return BrowserAction(True, "HTML extracted", {"html": html})
        except Exception as e:
            return BrowserAction(False, f"Get HTML failed: {str(e)}")
    
    async def screenshot(self, path: Optional[str] = None, full_page: bool = False) -> BrowserAction:
        """Take screenshot."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            if path:
                await self._page.screenshot(path=path, full_page=full_page)
                return BrowserAction(True, f"Screenshot saved: {path}")
            else:
                screenshot_bytes = await self._page.screenshot(full_page=full_page)
                # Convert to base64 for LLM
                base64_str = base64.b64encode(screenshot_bytes).decode()
                return BrowserAction(True, "Screenshot captured", {"base64": base64_str})
        except Exception as e:
            return BrowserAction(False, f"Screenshot failed: {str(e)}")
    
    async def scroll(self, direction: str = "down", amount: int = 500) -> BrowserAction:
        """Scroll page."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            if direction == "down":
                await self._page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self._page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "bottom":
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "top":
                await self._page.evaluate("window.scrollTo(0, 0)")
            
            return BrowserAction(True, f"Scrolled {direction}")
        except Exception as e:
            return BrowserAction(False, f"Scroll failed: {str(e)}")
    
    async def wait_for(self, selector: str, timeout: int = 10000) -> BrowserAction:
        """Wait for element."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return BrowserAction(True, f"Element found: {selector}")
        except Exception as e:
            return BrowserAction(False, f"Wait failed: {str(e)}")
    
    async def get_links(self) -> BrowserAction:
        """Get all links on page."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            links = await self._page.evaluate("""
                () => Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                }))
            """)
            return BrowserAction(True, f"Found {len(links)} links", {"links": links})
        except Exception as e:
            return BrowserAction(False, f"Get links failed: {str(e)}")
    
    async def get_form_fields(self) -> BrowserAction:
        """Get all form fields."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            fields = await self._page.evaluate("""
                () => Array.from(document.querySelectorAll('input, textarea, select')).map(el => ({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || 'text',
                    name: el.name,
                    id: el.id,
                    placeholder: el.placeholder,
                    required: el.required
                }))
            """)
            return BrowserAction(True, f"Found {len(fields)} form fields", {"fields": fields})
        except Exception as e:
            return BrowserAction(False, f"Get form fields failed: {str(e)}")
    
    async def fill_form(self, data: Dict[str, str], submit: bool = False) -> BrowserAction:
        """Fill form with data."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            for selector, value in data.items():
                # Try different selector strategies
                selectors = [
                    f'[name="{selector}"]',
                    f'#{selector}',
                    selector,
                ]
                
                for sel in selectors:
                    try:
                        await self._page.fill(sel, value, timeout=2000)
                        break
                    except:
                        continue
            
            if submit:
                await self._page.press('body', 'Enter')
            
            return BrowserAction(True, f"Form filled with {len(data)} fields")
        except Exception as e:
            return BrowserAction(False, f"Fill form failed: {str(e)}")
    
    async def solve_captcha(self, selector: str) -> BrowserAction:
        """Notify about captcha (manual solve required)."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self.screenshot("captcha.png")
            return BrowserAction(
                False,
                "CAPTCHA detected. Screenshot saved to captcha.png. Please solve manually.",
                {"captcha_selector": selector}
            )
        except Exception as e:
            return BrowserAction(False, f"CAPTCHA handling failed: {str(e)}")
    
    async def download_file(self, url: str, save_path: str) -> BrowserAction:
        """Download file."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            import urllib.request
            import urllib.parse
            
            # Handle relative URLs
            if url.startswith('/'):
                current_url = self._page.url
                parsed = urllib.parse.urlparse(current_url)
                url = f"{parsed.scheme}://{parsed.netloc}{url}"
            
            urllib.request.urlretrieve(url, save_path)
            return BrowserAction(True, f"Downloaded to: {save_path}")
        except Exception as e:
            return BrowserAction(False, f"Download failed: {str(e)}")
    
    async def get_current_url(self) -> str:
        """Get current page URL."""
        if self._page:
            return self._page.url
        return ""
    
    async def get_page_title(self) -> str:
        """Get page title."""
        if self._page:
            return await self._page.title()
        return ""
    
    async def go_back(self) -> BrowserAction:
        """Go back in history."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.go_back()
            return BrowserAction(True, "Navigated back")
        except Exception as e:
            return BrowserAction(False, f"Go back failed: {str(e)}")
    
    async def go_forward(self) -> BrowserAction:
        """Go forward in history."""
        if not self._page:
            return BrowserAction(False, "Browser not started")
        
        try:
            await self._page.go_forward()
            return BrowserAction(True, "Navigated forward")
        except Exception as e:
            return BrowserAction(False, f"Go forward failed: {str(e)}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get browser action history."""
        return self._history.copy()
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class BrowserTools:
    """Tools for browser automation that can be used with agents."""
    
    def __init__(self):
        self.controller: Optional[BrowserController] = None
    
    def get_tools(self, controller: BrowserController) -> List[Dict[str, Any]]:
        """Get tool definitions for agent."""
        self.controller = controller
        
        return [
            {
                "name": "navigate",
                "description": "Navigate to a URL",
                "parameters": {
                    "url": {"type": "string", "description": "URL to navigate to"}
                },
                "handler": self._navigate,
            },
            {
                "name": "click",
                "description": "Click on an element",
                "parameters": {
                    "selector": {"type": "string", "description": "CSS selector of element to click"}
                },
                "handler": self._click,
            },
            {
                "name": "fill",
                "description": "Fill a form field",
                "parameters": {
                    "selector": {"type": "string", "description": "CSS selector of input"},
                    "text": {"type": "string", "description": "Text to enter"}
                },
                "handler": self._fill,
            },
            {
                "name": "get_page_text",
                "description": "Get text content of the page",
                "parameters": {},
                "handler": self._get_text,
            },
            {
                "name": "screenshot",
                "description": "Take a screenshot of the page",
                "parameters": {},
                "handler": self._screenshot,
            },
            {
                "name": "scroll",
                "description": "Scroll the page",
                "parameters": {
                    "direction": {"type": "string", "enum": ["down", "up", "bottom", "top"]},
                },
                "handler": self._scroll,
            },
        ]
    
    async def _navigate(self, url: str) -> str:
        result = await self.controller.navigate(url)
        return result.message
    
    async def _click(self, selector: str) -> str:
        result = await self.controller.click(selector)
        return result.message
    
    async def _fill(self, selector: str, text: str) -> str:
        result = await self.controller.fill(selector, text)
        return result.message
    
    async def _get_text(self) -> str:
        result = await self.controller.get_text()
        return result.data.get("text", "") if result.success else result.message
    
    async def _screenshot(self) -> str:
        result = await self.controller.screenshot()
        return "Screenshot taken" if result.success else result.message
    
    async def _scroll(self, direction: str = "down") -> str:
        result = await self.controller.scroll(direction)
        return result.message
