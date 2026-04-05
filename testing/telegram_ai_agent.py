import os
import sys
import time
import base64
import subprocess
import requests
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import edge_tts

from dotenv import load_dotenv

# Add the src structure to the sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import neuralnode as nn
from neuralnode.core import BaseTool, ToolResult, ParameterSpec
from neuralnode.integrations.telegram import TelegramBot, TelegramBotConfig

# ==============================================================================
# 🛠️ FRAMEWORKS & LIBRARIES USED:
# 1. UI Automation: 'pyautogui' (Mouse, Keyboard, Screenshots)
# 2. OS Control: 'subprocess', 'os' (Launch apps, Shell commands, Shutdown)
# 3. Web: 'webbrowser' (Open URLs)
# 4. Voice: 'edge-tts' (Text-to-speech voice notes)
# 5. Local LLM: 'Ollama' (Reasoning and Vision analysis)
# 6. Messaging: 'python-telegram-bot' (Telegram integration)
# ==============================================================================
AGENT_SYSTEM_PROMPT = """You are Vision, a powerful AI agent with FULL CONTROL over Assem's Windows PC.
You have access to many tools. If Assem asks you to do something (open app, screenshot, click, type), USE THE TOOLS.
Never say you cannot do something if a tool exists for it.
Format your reasoning:
Thought: reasoning...
Action: tool_name Input: {...}
If no tool needed, respond directly with 'Final Answer: [your response]'
Call the user Assem.
"""

FAST_SYSTEM_PROMPT = """You are Vision, an AI assistant. Assem is your user.
Respond directly and concisely. No scratchpad. No 'Final Answer:'. Just speak to Assem.
"""

# ==============================================================================
# ⚙️ Global Settings
# ==============================================================================
SHOW_THINKING = False  # Set to True for ReAct (Reasoning), False for SimpleAgent
VOICE_RESPONSE_ENABLED = True # 🎤 Send voice notes instead of text by default
OLLAMA_PATH = r"C:\Users\Assem\AppData\Local\Programs\Ollama\ollama.exe"

# 🚀 RESOURCE SETTINGS: Adjust these to speed up/slow down the model
OLLAMA_OPTIONS = {
    "num_thread": 16,      # 🧵 Increased threads for faster processing
    "num_gpu": 200,        # 🎮 Force all layers to GPU if possible
    "num_ctx": 4096,       # 🧠 Context size
    "temperature": 0.4,    # 🌡️ Lower temperature for faster, more direct answers
    "top_k": 20,           # 🎯 More focused output
    "top_p": 0.8,
}

# ==============================================================================
# Configure logging
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_LLM = None # Global for tools to share
AUTONOMOUS_STATE = {"enabled": False, "last_analysis": "Initial state."}

async def autonomous_monitor_loop(agent, bot):
    """Background task to analyze screen every 5 seconds."""
    logger.info("Autonomous Monitor loop started (Inactive by default).")
    while True:
        try:
            if AUTONOMOUS_STATE["enabled"]:
                # 1. Capture screen silently
                path = "automonitor.png"
                import pyautogui
                pyautogui.screenshot().save(path)
                
                # 2. Analyze with LLM
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                # ✍️ Rigid System Prompt for autonomous monitor
                monitor_prompt = [
                    nn.Message.system("Analyze this PC screen briefly. What is the current active task or window? Keep it under 20 words."),
                    nn.Message.user("Analyze image.", metadata={"images": [img_b64]})
                ]
                
                logger.info("Autonomous Loop: Analyzing screen...")
                resp = await agent.llm.achat(monitor_prompt, ollama_options=OLLAMA_OPTIONS)
                AUTONOMOUS_STATE["last_analysis"] = resp.content
                logger.info(f"Autonomous Loop Result: {resp.content}")
                
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Autonomous Loop Error: {e}")
            await asyncio.sleep(10) # Wait longer on error

def ensure_ollama_running():
    """Starts Ollama server if not already running."""
    import socket
    
    # Check if Ollama is already listening on default port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 11434)) == 0:
            logger.info("Ollama server is already running.")
            return True

    # Try to start it
    if not os.path.exists(OLLAMA_PATH):
        logger.warning(f"Ollama executable not found at {OLLAMA_PATH}. Please check the path.")
        return False

    logger.info("Starting Ollama server...")
    try:
        # 0x00000010 is CREATE_NEW_CONSOLE on Windows
        creationflags = 0x00000010 if os.name == 'nt' else 0
        subprocess.Popen([OLLAMA_PATH, "serve"], 
                        creationflags=creationflags,
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for it to wake up
        for _ in range(15):
            time.sleep(1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', 11434)) == 0:
                    logger.info("Ollama server started successfully.")
                    return True
        logger.error("Ollama server failed to start within 15 seconds.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch Ollama: {e}")
        return False


# ==============================================================================
# 📬 Media Queue — shared between tools and bot
# ==============================================================================
class MediaType(Enum):
    PHOTO = "photo"
    VOICE = "voice"

@dataclass
class MediaItem:
    media_type: MediaType
    file_path: str
    caption: str = ""

class MediaQueue:
    def __init__(self):
        self._items: List[MediaItem] = []

    def push(self, item: MediaItem):
        self._items.append(item)

    def pop_all(self) -> List[MediaItem]:
        items = self._items.copy()
        self._items.clear()
        return items

MEDIA_QUEUE = MediaQueue()


# ==============================================================================
# 🛠️ AUTOMATION TOOLS
# ==============================================================================

class AppLauncherTool(BaseTool):
    """Opens any application on the Windows PC."""
    name = "launch_app"
    description = (
        "Open any application installed on Assem's Windows PC. "
        "Provide the app name: 'spotify', 'vscode', 'notepad', 'edge', 'chrome', "
        "'discord', 'telegram', 'vlc', 'steam', 'obs', 'word', 'excel', 'explorer', "
        "'calculator', 'paint', 'terminal', 'powershell', 'pycharm', 'postman' etc."
    )
    parameters = {
        "app": ParameterSpec(
            type="string",
            description="Name of the application to open (e.g., 'spotify', 'vscode', 'edge')",
            required=True
        ),
        "wait_seconds": ParameterSpec(
            type="string",
            description="Seconds to wait after launching (default: '2'). Use '4' or '5' for heavy apps like Spotify.",
            required=False
        )
    }

    APP_MAP: Dict[str, str] = {
        "spotify": "start spotify:",
        "vscode": "code",
        "vs code": "code",
        "code": "code",
        "notepad": "notepad",
        "notepad++": "notepad++",
        "edge": "start msedge",
        "microsoft edge": "start msedge",
        "chrome": "start chrome",
        "google chrome": "start chrome",
        "firefox": "start firefox",
        "discord": "start discord:",
        "telegram": "start tdesktop:",
        "vlc": "start vlc",
        "steam": "start steam:",
        "obs": "start obs64",
        "word": "start winword",
        "excel": "start excel",
        "powerpoint": "start powerpnt",
        "explorer": "explorer",
        "file explorer": "explorer",
        "calculator": "calc",
        "paint": "mspaint",
        "cmd": "start cmd",
        "terminal": "wt",
        "windows terminal": "wt",
        "powershell": "start powershell",
        "task manager": "taskmgr",
        "control panel": "control",
        "settings": "start ms-settings:",
        "pycharm": "start pycharm",
        "postman": "start postman",
    }

    def execute(self, app: str = "", wait_seconds: str = "2", **kwargs) -> ToolResult:
        try:
            app = app or kwargs.get("app") or kwargs.get("app_name") or kwargs.get("query", "")
            if not app:
                return ToolResult.error("Application name is required.")
            app_lower = app.lower().strip()
            wait = float(wait_seconds) if wait_seconds else 2.0
            cmd = self.APP_MAP.get(app_lower, f"start {app_lower}")
            logger.info(f"Launching: {app} -> {cmd}")
            subprocess.Popen(cmd, shell=True)
            time.sleep(wait)
            return ToolResult.ok(content=f"Launched '{app}'. Waited {wait}s.")
        except Exception as e:
            return ToolResult.error(f"Failed to launch '{app}': {e}")


class SpotifyControlTool(BaseTool):
    """Real Spotify control using pyautogui."""
    name = "spotify_control"
    description = (
        "Control Spotify: open it, search for a song by name, and play it. "
        "use action='play' with song name, action='pause', action='next', action='previous'."
    )
    parameters = {
        "action": ParameterSpec(
            type="string",
            description="Action: 'play' (search+play), 'pause', 'next', 'previous', 'open'",
            required=True
        ),
        "song": ParameterSpec(
            type="string",
            description="Song name to search. Required for action='play'.",
            required=False
        ),
        "song_name": ParameterSpec(
            type="string",
            description="Alias for song.",
            required=False
        )
    }

    def execute(self, action: str = "", song: str = "", **kwargs) -> ToolResult:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.3
            
            # Robust parameter handling
            action = action or kwargs.get("action", "open")
            action = action.lower().strip()
            
            # Agent might send 'song_name' or 'query' instead of 'song'
            song = song or kwargs.get("song") or kwargs.get("song_name") or kwargs.get("query", "")

            if action == "open":
                subprocess.Popen("start spotify:", shell=True)
                time.sleep(4)
                return ToolResult.ok(content="Spotify opened.")

            if action == "play":
                if not song:
                    # Just press space if no song provided
                    subprocess.Popen("start spotify:", shell=True)
                    time.sleep(1.5)
                    pyautogui.press("space")
                    return ToolResult.ok(content="Spotify Play/Pause toggled.")
                
                subprocess.Popen("start spotify:", shell=True)
                time.sleep(3.5)
                # Search
                pyautogui.hotkey("ctrl", "l")
                time.sleep(0.5)
                pyautogui.hotkey("ctrl", "a")
                pyautogui.press("backspace")
                pyautogui.write(song, interval=0.05)
                time.sleep(1)
                pyautogui.press("enter")
                time.sleep(1.5)
                pyautogui.press("enter") # Start playing
                return ToolResult.ok(content=f"Searched and playing '{song}' on Spotify.")

            elif action == "pause":
                subprocess.Popen("start spotify:", shell=True)
                time.sleep(1.5)
                pyautogui.press("space")
                return ToolResult.ok(content="Toggled play/pause.")

            elif action == "next":
                pyautogui.hotkey("ctrl", "right")
                return ToolResult.ok(content="Skipped to next track.")

            elif action == "previous":
                pyautogui.hotkey("ctrl", "left")
                return ToolResult.ok(content="Went to previous track.")

            return ToolResult.error(f"Unknown Spotify action: {action}")
        except Exception as e:
            return ToolResult.error(f"Spotify control failed: {e}")


class EdgeBrowserTool(BaseTool):
    """Controls Microsoft Edge browser."""
    name = "edge_browser"
    description = "Open Microsoft Edge and navigate to a URL or perform a web search."
    parameters = {
        "action": ParameterSpec(
            type="string",
            description="'open_url' to navigate to a URL, 'search' to search the web",
            required=True
        ),
        "query": ParameterSpec(
            type="string",
            description="URL (e.g., 'youtube.com') or search query",
            required=True
        )
    }

    def execute(self, action: str, query: str) -> ToolResult:
        try:
            if action == "open_url":
                if not query.startswith("http"):
                    query = "https://" + query
                subprocess.Popen(f'start msedge "{query}"', shell=True)
                return ToolResult.ok(content=f"Opened Edge at: {query}")
            elif action == "search":
                from urllib.parse import quote
                search_url = f"https://www.bing.com/search?q={quote(query)}"
                subprocess.Popen(f'start msedge "{search_url}"', shell=True)
                return ToolResult.ok(content=f"Opened Edge search for: '{query}'")
            return ToolResult.error(f"Unknown action '{action}'. Use: open_url, search")
        except Exception as e:
            return ToolResult.error(f"Edge browser failed: {e}")


class KeyboardShortcutTool(BaseTool):
    """Press keyboard shortcuts on the PC."""
    name = "keyboard_shortcut"
    description = (
        "Press any keyboard shortcut or key combination. "
        "Examples: 'win+d' (show desktop), 'alt+f4' (close), 'ctrl+c' (copy), "
        "'win+e' (file explorer), 'ctrl+l' (address bar), 'ctrl+shift+esc' (task manager)."
    )
    parameters = {
        "keys": ParameterSpec(
            type="string",
            description="Keys joined with '+'. Examples: 'win+d', 'ctrl+alt+del', 'alt+f4', 'enter', 'escape'",
            required=True
        ),
        "wait_after": ParameterSpec(
            type="string",
            description="Seconds to wait after pressing (default: '0.5')",
            required=False
        )
    }

    def execute(self, keys: str = "", wait_after: str = "0.5", **kwargs) -> ToolResult:
        try:
            if not keys:
                keys = kwargs.get("query", "")
            if not keys:
                return ToolResult.error("Keys are required.")
            import pyautogui
            pyautogui.FAILSAFE = False
            key_list = [k.strip().lower() for k in keys.split("+")]
            if len(key_list) == 1:
                pyautogui.press(key_list[0])
            else:
                pyautogui.hotkey(*key_list)
            time.sleep(float(wait_after) if wait_after else 0.5)
            return ToolResult.ok(content=f"Pressed: {keys}")
        except ImportError:
            return ToolResult.error("pyautogui not installed. Run: pip install pyautogui pillow")
        except Exception as e:
            return ToolResult.error(f"Keyboard shortcut failed: {e}")


class TypeTextTool(BaseTool):
    """Types text into the currently active window."""
    name = "type_text"
    description = "Type text into the currently focused/active window on the PC."
    parameters = {
        "text": ParameterSpec(
            type="string",
            description="The text to type",
            required=True
        ),
        "press_enter": ParameterSpec(
            type="string",
            description="Press Enter after typing? 'true' or 'false' (default: 'false')",
            required=False
        )
    }

    def execute(self, text: str = "", press_enter: str = "false", **kwargs) -> ToolResult:
        try:
            if not text:
                text = kwargs.get("query", "")
            if not text:
                return ToolResult.error("Text is required.")
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.write(text, interval=0.04)
            if press_enter.lower() == "true":
                time.sleep(0.2)
                pyautogui.press("enter")
            return ToolResult.ok(content=f"Typed: '{text}'" + (" + Enter" if press_enter.lower() == "true" else ""))
        except ImportError:
            return ToolResult.error("pyautogui not installed.")
        except Exception as e:
            return ToolResult.error(f"Typing failed: {e}")


class WindowManagerTool(BaseTool):
    """Manages open windows on the PC."""
    name = "window_manager"
    description = "Manage open windows: list, focus, minimize, maximize, or close them."
    parameters = {
        "action": ParameterSpec(
            type="string",
            description="'list', 'focus', 'minimize', 'maximize', 'close'",
            required=True
        ),
        "window_name": ParameterSpec(
            type="string",
            description="Part of the window title (e.g., 'Spotify', 'Edge'). Not needed for 'list'.",
            required=False
        )
    }

    def execute(self, action: str, window_name: str = "") -> ToolResult:
        try:
            action = action.lower().strip()
            try:
                import pygetwindow as gw
                HAS_GW = True
            except ImportError:
                HAS_GW = False

            if action == "list":
                if HAS_GW:
                    windows = [w for w in gw.getAllTitles() if w.strip()]
                    return ToolResult.ok(content="Open windows:\n" + "\n".join(f"• {w}" for w in windows[:30]))
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object -ExpandProperty MainWindowTitle"],
                    capture_output=True, text=True, timeout=10
                )
                return ToolResult.ok(content="Open windows:\n" + result.stdout[:2000])

            if not window_name:
                return ToolResult.error("window_name required for this action.")

            if HAS_GW:
                matches = gw.getWindowsWithTitle(window_name)
                if not matches:
                    all_w = gw.getAllWindows()
                    matches = [w for w in all_w if window_name.lower() in w.title.lower() and w.title.strip()]
                if not matches:
                    return ToolResult.error(f"No window found matching '{window_name}'.")
                win = matches[0]
                if action == "focus":
                    if win.isMinimized:
                        win.restore()
                    win.activate()
                    return ToolResult.ok(content=f"Focused: '{win.title}'")
                elif action == "minimize":
                    win.minimize()
                    return ToolResult.ok(content=f"Minimized: '{win.title}'")
                elif action == "maximize":
                    win.maximize()
                    return ToolResult.ok(content=f"Maximized: '{win.title}'")
                elif action == "close":
                    win.close()
                    return ToolResult.ok(content=f"Closed: '{win.title}'")
            return ToolResult.error("pygetwindow not installed. Run: pip install pygetwindow")
        except Exception as e:
            return ToolResult.error(f"Window management failed: {e}")


class CheckScreenTool(BaseTool):
    """Silently captures and analyzes the screen — does NOT send to Telegram."""
    name = "check_screen"
    description = (
        "Capture a screenshot and get an AI description of what's currently visible on Assem's screen. "
        "Use this AFTER every action to verify it worked. "
        "This does NOT send anything to Telegram — it only returns a text description for your reasoning."
    )
    parameters = {
        "wait_seconds": ParameterSpec(
            type="string",
            description="Seconds to wait before capturing (default: '1'). Use '2'-'3' after launching apps.",
            required=False
        ),
        "question": ParameterSpec(
            type="string",
            description="What specifically to look for (e.g., 'Is Spotify open and playing?'). Default: general overview.",
            required=False
        )
    }

    def execute(self, wait_seconds: str = "1", question: str = "") -> ToolResult:
        try:
            wait = float(wait_seconds) if wait_seconds else 1.0
            time.sleep(wait)

            path = str(Path(__file__).parent / "_check_screen_tmp.png")
            try:
                import pyautogui
                pyautogui.FAILSAFE = False
                pyautogui.screenshot().save(path)
            except ImportError:
                return ToolResult.error("pyautogui not installed. Run: pip install pyautogui pillow")

            if not os.path.exists(path):
                return ToolResult.error("Screenshot capture failed.")

            if not LOCAL_LLM:
                return ToolResult.ok(content="Screenshot taken but local vision analysis unavailable (LLM not connected).")

            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            try:
                os.remove(path)
            except Exception:
                pass

            prompt = question or (
                "Describe what is currently visible on this Windows PC screen. "
                "List: 1) Which applications are open/focused, 2) Any important content "
                "(URLs, file names, error messages, playing music, etc.), 3) Overall state."
            )
            
            try:
                msg = nn.Message.user(prompt)
                msg.metadata["images"] = [img_b64]
                logger.info(f"Checking screen with local Ollama ({LOCAL_LLM.model})...")
                resp = LOCAL_LLM.chat([msg])
                return ToolResult.ok(content="[Screen State]\n" + resp.content)
            except Exception as ex:
                logger.warning(f"Local screen check vision failed: {ex}")
                return ToolResult.ok(content="[Screen State] Screenshot captured, but local vision analysis failed or is not supported by this model.")
        except Exception as e:
            logger.error(f"check_screen error: {e}")
            return ToolResult.error(f"Screen check failed: {e}")


class ScreenshotTool(BaseTool):
    """Takes a screenshot, analyzes it with AI, and SENDS it to Assem on Telegram."""
    name = "take_screenshot"
    description = (
        "Capture a screenshot of Assem's current screen. "
        "The screenshot will be SENT to Assem on Telegram as a photo with an AI description. "
        "Use this ONLY when Assem explicitly asks to see the screen."
    )
    parameters = {
        "wait_seconds": ParameterSpec(
            type="string",
            description="Seconds to wait before capturing (default: '1.5').",
            required=False
        ),
        "analyze_prompt": ParameterSpec(
            type="string",
            description="What to ask AI about the screenshot.",
            required=False
        ),
        "query": ParameterSpec(
            type="string",
            description="Alias for analyze_prompt.",
            required=False
        )
    }

    def execute(self, wait_seconds: str = "1.5", analyze_prompt: str = "", **kwargs) -> ToolResult:
        try:
            analyze_prompt = analyze_prompt or kwargs.get("analyze_prompt") or kwargs.get("query", "")
            wait = float(wait_seconds) if wait_seconds else 1.5
            time.sleep(wait)

            path = str(Path(__file__).parent / "_screenshot_send.png")
            try:
                import pyautogui
                pyautogui.FAILSAFE = False
                pyautogui.screenshot().save(path)
                logger.info("Screenshot captured with pyautogui.")
            except ImportError:
                ps = (
                    "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
                    "$bmp=New-Object System.Drawing.Bitmap("
                    "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width,"
                    "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height);"
                    "$g=[System.Drawing.Graphics]::FromImage($bmp);"
                    f"$g.CopyFromScreen(0,0,0,0,$bmp.Size);$bmp.Save('{path.replace(chr(92), '/')}');"
                    "$g.Dispose();$bmp.Dispose()"
                )
                subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=15)
                if not os.path.exists(path):
                    return ToolResult.error("Screenshot failed. Install pyautogui: pip install pyautogui pillow")

            analysis = "Screenshot captured."
            if LOCAL_LLM and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    
                    prompt = analyze_prompt or "Describe in detail what is visible on this screen."
                    msg = nn.Message.user(prompt)
                    msg.metadata["images"] = [img_b64]
                    
                    logger.info(f"Analyzing screenshot with local Ollama ({LOCAL_LLM.model})...")
                    resp = LOCAL_LLM.chat([msg])
                    analysis = resp.content
                except Exception as ex:
                    logger.warning(f"Local vision analysis failed or not supported: {ex}")
                    analysis = "Screenshot captured (Local vision skipped)."

            # ✅ Queue photo to send on Telegram
            MEDIA_QUEUE.push(MediaItem(
                media_type=MediaType.PHOTO,
                file_path=path,
                caption=f"📸 Screenshot\n\n{analysis[:900]}"
            ))
            logger.info("Screenshot queued for Telegram delivery.")
            return ToolResult.ok(content=f"Screenshot taken and will be sent to Telegram.\nAnalysis: {analysis[:300]}")
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return ToolResult.error(f"Screenshot failed: {e}")


class VoiceNoteTool(BaseTool):
    """Converts text to speech using edge_tts and SENDS it as a Telegram voice note."""
    name = "send_voice_note"
    description = (
        "Convert text to speech and SEND it as a voice note on Telegram. "
        "Use ONLY when Assem explicitly asks for a voice note or to 'read this aloud'."
    )
    parameters = {
        "text": ParameterSpec(
            type="string",
            description="The text to convert to speech and send as a voice note.",
            required=True
        )
    }

    def execute(self, text: str = "", **kwargs) -> ToolResult:
        try:
            text = text or kwargs.get("text") or kwargs.get("query", "")
            if not text:
                return ToolResult.error("Text for voice note is required.")
                
            audio_path = str(Path(__file__).parent / "_voice_note.mp3")
            
            logger.info(f"Generating Voice Note (edge-tts): {text[:60]}...")
            
            async def generate_voice():
                communicate = edge_tts.Communicate(text, "en-US-EmmaNeural")
                await communicate.save(audio_path)
            
            try:
                asyncio.run(generate_voice())
            except RuntimeError:
                # Fallback for complex loop environments
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(generate_voice())
                except:
                    import subprocess
                    # Ultra fallback using command line
                    subprocess.run(["edge-tts", "--text", text, "--voice", "en-US-EmmaNeural", "--write-media", audio_path], check=True)

            if os.path.exists(audio_path):
                MEDIA_QUEUE.push(MediaItem(media_type=MediaType.VOICE, file_path=audio_path))
                logger.info("Voice note queued for Telegram delivery.")
                return ToolResult.ok(content="Voice note generated and will be sent to Telegram.")
            else:
                return ToolResult.error("Failed to generate voice file.")
        except Exception as e:
            logger.error(f"Voice note error: {e}")
            return ToolResult.error(f"Voice note failed: {e}")


class MouseClickTool(BaseTool):
    """Perform mouse clicks at specific coordinates."""
    name = "mouse_click"
    description = "Click at specific (x, y) coordinates on the screen. Use 'double' or 'right' for click_type."
    parameters = {
        "x": ParameterSpec(type="string", description="X coordinate (0 to screen width)", required=True),
        "y": ParameterSpec(type="string", description="Y coordinate (0 to screen height)", required=True),
        "click_type": ParameterSpec(type="string", description="'left', 'right', 'double'", required=False),
        "wait_after": ParameterSpec(type="string", description="Wait after click (default: 0.5)", required=False)
    }

    def execute(self, x: str = "", y: str = "", click_type: str = "left", wait_after: str = "0.5", **kwargs) -> ToolResult:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            x = x or kwargs.get("x") or kwargs.get("query")
            y = y or kwargs.get("y")
            if not x or not y:
                return ToolResult.error("X and Y coordinates are required.")
            
            px, py = int(x), int(y)
            ct = click_type.lower() if click_type else "left"
            if ct == "double":
                pyautogui.doubleClick(px, py)
            elif ct == "right":
                pyautogui.rightClick(px, py)
            else:
                pyautogui.click(px, py)
            
            time.sleep(float(wait_after) if wait_after else 0.5)
            return ToolResult.ok(content=f"Clicked {ct} at ({px}, {py})")
        except Exception as e:
            return ToolResult.error(f"Mouse click failed: {e}")


class ExecuteCommandTool(BaseTool):
    """Run any PowerShell/CMD command."""
    name = "execute_command"
    description = "Execute a PowerShell or CMD command on Assem's Windows PC."
    parameters = {
        "command": ParameterSpec(
            type="string",
            description="The command to run",
            required=True
        ),
        "query": ParameterSpec(
            type="string",
            description="Alias for command.",
            required=False
        )
    }

    def execute(self, command: str = "", **kwargs) -> ToolResult:
        try:
            command = command or kwargs.get("command") or kwargs.get("query", "")
            if not command:
                return ToolResult.error("Command is required.")
                
            logger.info(f"Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            output = result.stdout + (f"\n[stderr]: {result.stderr}" if result.stderr else "")
            return ToolResult.ok(content=(output.strip() or "[Command ran with no output]")[:3000])
        except Exception as e:
            return ToolResult.error(f"Command failed: {e}")


class SystemControlTool(BaseTool):
    """Controls PC state: lock, shutdown, restart, unlock."""
    name = "system_control"
    description = (
        "Control PC power and session state. "
        "Actions: 'lock', 'unlock', 'shutdown', 'restart'."
    )
    parameters = {
        "action": ParameterSpec(
            type="string",
            description="Action to perform: 'lock', 'unlock', 'shutdown', 'restart'",
            required=True
        ),
        "query": ParameterSpec(
            type="string",
            description="Alias for action.",
            required=False
        )
    }

    def execute(self, action: str = "", **kwargs) -> ToolResult:
        try:
            import pyautogui
            action = action or kwargs.get("action") or kwargs.get("query", "")
            action = action.lower().strip()

            if action == "lock":
                pyautogui.hotkey('win', 'l')
                return ToolResult.ok(content="PC Locked using Win+L.")

            if action == "unlock":
                # Unlock sequence: Press space, wait 8s, type password, enter
                logger.info("Starting unlock sequence...")
                pyautogui.press('space')
                time.sleep(8)
                pyautogui.write("198525", interval=0.1)
                pyautogui.press('enter')
                return ToolResult.ok(content="Unlock sequence (Space -> 8s -> Password -> Enter) executed.")

            if action == "shutdown":
                logger.info("Shutting down PC...")
                subprocess.Popen("shutdown /s /t 5", shell=True)
                return ToolResult.ok(content="PC is shutting down in 5 seconds.")

            if action == "restart":
                logger.info("Restarting PC...")
                subprocess.Popen("shutdown /r /t 5", shell=True)
                return ToolResult.ok(content="PC is restarting in 5 seconds.")

            return ToolResult.error(f"Unknown system action: {action}")
        except Exception as e:
            return ToolResult.error(f"System control failed: {e}")


class QuickTelegramScreenshotTool(BaseTool):
    """Takes a screenshot via system shortcut and pastes it directly into Telegram."""
    @property
    def name(self) -> str:
        return "quick_tg_screenshot"
    
    @property
    def description(self) -> str:
        return "Captures screen using Win+PrtSc and pastes it directly into the Telegram window using Ctrl+V. Use this when Assem wants a fast system screenshot sent to him."
    
    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {}

    def execute(self) -> ToolResult:
        try:
            import pyautogui
            import pygetwindow as gw
            import time
            
            logger.info("Macro: Taking system screenshot (Win+PrtSc)...")
            pyautogui.hotkey('win', 'prtsc')
            time.sleep(0.5)
            # Ensure it's in clipboard (Standard PrtSc)
            pyautogui.press('prtsc')
            time.sleep(1.0)
            
            logger.info("Macro: Finding Telegram window...")
            wins = gw.getWindowsWithTitle('Telegram')
            if not wins:
                return ToolResult.error("Telegram window not found/active.")
            
            tg = wins[0]
            tg.activate()
            time.sleep(0.5)
            
            logger.info("Macro: Pasting and Sending...")
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(1.0)
            pyautogui.press('enter')
            time.sleep(0.5)
            # Re-minimize or return focus if needed? Nah, just stay there.
            
            return ToolResult.ok(content="Screenshot captured and pasted into Telegram successfully.")
        except Exception as e:
            return ToolResult.error(f"Macro failed: {e}")


# ==============================================================================
# 🤖 Vision Telegram Bot — delivers queued media (photos + voice notes)
# ==============================================================================

def clean_agent_response(text: str) -> str:
    """Removes Thought/Action/Final Answer markers for a cleaner Telegram experience."""
    if not text:
        return text
        
    import re
    original = text
    
    # Remove "Thought: ...", "Action: ...", "PARAMETERS: ...", "Input: ..."
    text = re.sub(r'(?i)Thought:.*?(?=Action:|Final Answer:|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)Action:.*?\n', '', text)
    text = re.sub(r'(?i)Input:.*?\n', '', text)
    text = re.sub(r'(?i)PARAMETERS:.*?\n', '', text)
    
    # Remove "Final Answer:" label itself
    text = re.sub(r'(?i)Final Answer:\s*', '', text)
    
    # If cleaning removed EVERYTHING, return the original text (it's better than nothing)
    cleaned = text.strip()
    if "Maximum iterations reached" in cleaned:
        return "I tried to process that, but the loop took too long. Please try again or be more specific!"
    return cleaned if cleaned else original.strip()


async def generate_voice_file(text: str, output_path: str):
    """Generates an MP3 file using edge_tts."""
    try:
        communicate = edge_tts.Communicate(text, "en-US-EmmaNeural")
        await communicate.save(output_path)
        return True
    except Exception as e:
        logger.error(f"TTS Generation failed: {e}")
        return False


class VisionTelegramBot(TelegramBot):
    """
    Extends TelegramBot to deliver media queued by tools (screenshots, voice notes).
    Does NOT auto-screenshot on every message — the agent calls check_screen/take_screenshot
    explicitly as part of its reasoning.
    """

    async def _handle_text_message(self, update, context):
        """Robust message handling with Fast Mode bypass and async execution."""
        user = update.effective_user
        session = self._get_or_create_session(user)
        
        if not session.authorized:
            await update.message.reply_text("🚫 Unauthorized.")
            return

        user_message = update.message.text
        chat_id = update.effective_chat.id

        # ⚡ COMMAND: /voice
        if user_message.strip().lower().startswith("/voice"):
            global VOICE_RESPONSE_ENABLED
            cmd = user_message.strip().lower().split()
            if len(cmd) > 1 and cmd[1] == "on":
                VOICE_RESPONSE_ENABLED = True
                await update.message.reply_text("🎤 Voice Responses: **ENABLED**")
            else:
                VOICE_RESPONSE_ENABLED = False
                await update.message.reply_text("🎤 Voice Responses: **DISABLED** (Text mode)")
            return

        # ⚡ COMMAND: /monitor

        # 1. Send Typing action
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        try:
            # 🚀 ACTION DETECTION: If message implies action, always use Agent even if short
            action_keywords = ["screenshot", "open", "launch", "click", "type", "press", "window", "close", "shutdown", "restart", "spotify", "edge", "lock", "unlock"]
            is_action = any(k in user_message.lower() for k in action_keywords)

            # 💨 FAST MODE: Pure chat only
            if not SHOW_THINKING and not is_action and len(user_message.split()) < 20:
                # 🛠️ Use Ultra-Fast Direct Prompt
                messages = [
                    nn.Message.system(FAST_SYSTEM_PROMPT),
                    nn.Message.user(user_message)
                ]
                logger.info("Fast Mode (Direct): Requesting LLM...")
                resp = await self.agent.llm.achat(messages, ollama_options=OLLAMA_OPTIONS)
                response_text = clean_agent_response(resp.content)
            else:
                # 🧠 AGENT MODE: Use ReAct reasoning (Real control)
                logger.info(f"Agent Mode ({type(self.agent).__name__}): Active. is_action={is_action}")
                raw_response = await self.agent.arun(user_message)
                response_text = clean_agent_response(raw_response)

            # Fallback if empty
            if not response_text or len(response_text.strip()) < 1:
                response_text = "I'm here, Assem. How can I help you today?"

            # 2. Save history
            from datetime import datetime
            session.chat_history.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})
            session.chat_history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
            self._save_sessions()

            # 3. Send Response (Voice or Text)
            if VOICE_RESPONSE_ENABLED and len(response_text) > 0:
                voice_path = str(Path(__file__).parent / f"response_{int(time.time())}.mp3")
                logger.info(f"Generating voice response into {voice_path}...")
                
                success = await generate_voice_file(response_text, voice_path)
                if success and os.path.exists(voice_path):
                    with open(voice_path, 'rb') as voice:
                        await context.bot.send_voice(chat_id=chat_id, voice=voice)
                    # Cleanup after sending
                    try: os.remove(voice_path) 
                    except: pass
                else:
                    # Fallback to text if TTS fails
                    await self._send_long_message(update, f"⚠️ (Voice failed) {response_text}")
            else:
                await self._send_long_message(update, response_text)

        except Exception as e:
            await self._handle_error(update, context, e)
        finally:
            # 📸 Flush any queued media
            await self._flush_media_queue(update, context)

    async def _handle_error(self, update, context, error):
        """Send human-readable error messages to Assem."""
        err_str = str(error)
        logger.error(f"Agent Error: {err_str}", exc_info=True)
        
        friendly_msg = "❌ **Sorry Assem, I encountered an issue while processing your request.**\n\n"
        
        if "429" in err_str:
            friendly_msg += "The local model is currently busy or receiving too many requests. Please wait a moment."
        elif "quota" in err_str.lower():
            friendly_msg += "There is an issue with model availability at the moment."
        elif "pyautogui" in err_str.lower() or "import" in err_str.lower():
            friendly_msg += "There is an issue with library dependencies on your PC (e.g., pyautogui not installed)."
        elif "unexpected keyword argument" in err_str:
            friendly_msg += "An error occurred passing data to the tool. I've tried to fix it automatically, please try again."
        else:
            friendly_msg += f"Technical reason: `{err_str[:200]}`"

        try:
            await update.message.reply_text(friendly_msg, parse_mode='Markdown')
        except Exception:
            pass

    async def _flush_media_queue(self, update, context):
        """Send screenshots and voice notes queued by tools to Telegram."""
        items = MEDIA_QUEUE.pop_all()
        if not items:
            return

        logger.info(f"Flushing media queue: {len(items)} items found.")

        chat_id = update.effective_chat.id
        reply_id = update.message.message_id

        for item in items:
            try:
                if not os.path.exists(item.file_path):
                    logger.warning(f"Media file missing, skipping: {item.file_path}")
                    continue

                if item.media_type == MediaType.PHOTO:
                    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
                    with open(item.file_path, "rb") as photo_f:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=photo_f,
                            caption=item.caption[:1024] if item.caption else None,
                            reply_to_message_id=reply_id
                        )
                    logger.info("✅ Screenshot sent to Telegram successfully")

                elif item.media_type == MediaType.VOICE:
                    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
                    with open(item.file_path, "rb") as voice_f:
                        await context.bot.send_voice(
                            chat_id=chat_id,
                            voice=voice_f,
                            reply_to_message_id=reply_id
                        )
                    logger.info("✅ Voice note sent to Telegram successfully")

            except Exception as e:
                logger.error(f"Failed to send media '{item.file_path}': {e}", exc_info=True)
            finally:
                try:
                    if os.path.exists(item.file_path):
                        os.remove(item.file_path)
                except Exception:
                    pass


# ==============================================================================
# 🚀 Main
# ==============================================================================

def main():
    global LOCAL_LLM

    print("=" * 60)
    print("🤖  Vision Agent — Full PC Control via Telegram")
    print(f"    Thinking Mode: {'ON (ReAct)' if SHOW_THINKING else 'OFF (Simple)'}")
    print("=" * 60)

    load_dotenv()

    # 1. Automate Ollama Server Start
    if not ensure_ollama_running():
        print("❌ Could not ensure Ollama is running. Please start it manually.")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
        return

    print(f"\n🔍 Connecting to Local Ollama ({'qwen3.5:4b'})...")
    try:
        # Pass OLLAMA_OPTIONS for resource control
        LOCAL_LLM = nn.create_provider("ollama", model="qwen3.5:4b", base_url="http://localhost:11434", ollama_options=OLLAMA_OPTIONS)
        print(f"✅ LLM ready: {type(LOCAL_LLM).__name__} ({LOCAL_LLM.model})")
        llm = LOCAL_LLM
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return

    print("\n🛠️  Loading Tools...")
    tools = [
        SystemControlTool(),     # 🔒 Lock/Shutdown/Restart/Unlock
        CheckScreenTool(),       # 👁️ Verify screen state after each action
        AppLauncherTool(),       # 🚀 Open any app
        SpotifyControlTool(),    # 🎵 Control Spotify
        EdgeBrowserTool(),       # 🌐 Microsoft Edge
        KeyboardShortcutTool(),  # ⌨️ Keyboard shortcuts
        TypeTextTool(),          # 📝 Type text
        WindowManagerTool(),     # 🪟 Manage windows
        ScreenshotTool(),        # 📸 Screenshot → Telegram
        VoiceNoteTool(),         # 🎤 Voice note → Telegram
        ExecuteCommandTool(),    # 💻 PowerShell/CMD
        MouseClickTool(),        # 🖱️ Mouse control
        nn.PythonREPLTool(),     # 🐍 Python code
        QuickTelegramScreenshotTool(), # ⚡ New system macro
    ]
    print(f"✅ {len(tools)} tools loaded successfully.")

    if SHOW_THINKING:
        print("\n🧠 Creating ReAct Agent (Thinking Mode ON)...")
        agent = nn.ReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            max_iterations=15
        )
    else:
        print("\n⚡ Creating Fast Agent (Thinking Mode OFF)...")
        agent = nn.ReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT.strip() + "\nCRITICAL: Use tools immediately. Do not hesitate.",
            max_iterations=5  # Increased to allow tool execution + observation
        )

    print("\n📱 Starting Telegram Bot & Autonomous Monitor...")
    bot = VisionTelegramBot(
        token=token,
        agent=agent,
        config=TelegramBotConfig(
            token=token,
            enable_voice=False,
            persistence_path="vision_sessions.json"
        )
    )

    async def run_all():
        # Start the background screen monitor
        asyncio.create_task(autonomous_monitor_loop(agent, bot))
        
        # Start the bot (blocking/polling done via start_async wrapper)
        bot_task = bot.start_async()
        if bot_task:
            print("\n✅ Vision Agent is LIVE on Telegram!")
            print("🚀 Autonomous Monitor Loop: ACTIVE (Waiting for /monitor on)")
            
            # 🛑 KEEP ALIVE: Wait for bot task and stay running
            try:
                await bot_task
                while True:
                    await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass
        else:
            print("❌ Failed to start bot async.")

    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\n🛑 Stopping Vision Agent...")
        bot.stop()


if __name__ == "__main__":
    main()
