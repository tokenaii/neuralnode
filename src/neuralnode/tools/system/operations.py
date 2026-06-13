"""Date-time and system operation tools for NeuralNode agents."""

import os
import sys
import subprocess
import platform
import logging
import getpass
from datetime import datetime, timezone
from typing import Optional

from ...core import BaseTool, ToolResult, ParameterSpec

logger = logging.getLogger(__name__)

_OS = platform.system().lower()  # 'windows', 'linux', 'darwin'


class DateTimeTool(BaseTool):
    """Give the agent real-time access to the current date, time, and timezone.

    The agent can call this tool with no parameters to get the full
    timestamp, or pass a ``format`` string (Python strftime).
    """

    name = "get_datetime"
    description = (
        "Get the current date and time. "
        "Returns date, time, timezone, day of week, and unix timestamp."
    )
    parameters = {
        "format": ParameterSpec(
            type="string",
            description="Optional strftime format string, e.g. '%Y-%m-%d %H:%M'",
            required=False,
            default=None,
        ),
        "timezone_offset": ParameterSpec(
            type="integer",
            description="UTC offset in hours, e.g. 2 for UTC+2. None = local time.",
            required=False,
            default=None,
        ),
    }

    def execute(
        self,
        format: Optional[str] = None,
        timezone_offset: Optional[int] = None,
    ) -> ToolResult:
        try:
            if timezone_offset is not None:
                from datetime import timedelta
                tz = timezone(timedelta(hours=timezone_offset))
                now = datetime.now(tz)
            else:
                now = datetime.now()

            if format:
                formatted = now.strftime(format)
                return ToolResult.ok(content=formatted)

            info = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "timezone": str(now.tzinfo) if now.tzinfo else "local",
                "iso": now.isoformat(),
                "unix_timestamp": int(now.timestamp()),
            }

            text = (
                f"Current Date & Time:\n"
                f"  Date:      {info['date']}\n"
                f"  Time:      {info['time']}\n"
                f"  Day:       {info['day_of_week']}\n"
                f"  Timezone:  {info['timezone']}\n"
                f"  ISO:       {info['iso']}\n"
                f"  Unix:      {info['unix_timestamp']}"
            )
            return ToolResult.ok(content=text, **info)
        except Exception as e:
            return ToolResult.error(f"Failed to get datetime: {e}")


class SystemOperationsTool(BaseTool):
    """Perform system-level operations on the user's machine.

    Supported operations (cross-platform where possible):
        - ``shutdown``   – shut down the machine
        - ``restart``    – reboot the machine
        - ``lock``       – lock the screen / workstation
        - ``unlock``     – unlock (Windows only, requires password)
        - ``sleep``      – put into sleep mode
        - ``hibernate``  – hibernate (Windows / Linux)
        - ``logout``     – log out the current user
        - ``screenshot`` – capture a screenshot
        - ``open_app``   – launch an application
        - ``clipboard_get``  – read clipboard text
        - ``clipboard_set``  – write text to clipboard
        - ``volume_set``     – set system volume (0-100)
        - ``volume_mute``    – toggle mute
        - ``brightness_set`` – set screen brightness (0-100)
        - ``notify``     – show a desktop notification

    .. warning::
        Operations like shutdown/restart are **destructive**. Use with care.
    """

    name = "system_operations"
    description = (
        "Perform system operations: shutdown, restart, lock, unlock, sleep, "
        "hibernate, logout, screenshot, open_app, clipboard_get, clipboard_set, "
        "volume_set, volume_mute, brightness_set, notify."
    )
    parameters = {
        "operation": ParameterSpec(
            type="string",
            description=(
                "The operation to perform. One of: shutdown, restart, lock, unlock, "
                "sleep, hibernate, logout, screenshot, open_app, clipboard_get, "
                "clipboard_set, volume_set, volume_mute, brightness_set, notify"
            ),
            required=True,
        ),
        "value": ParameterSpec(
            type="string",
            description=(
                "Value for the operation (e.g. app name for open_app, "
                "text for clipboard_set, 0-100 for volume_set/brightness_set, "
                "password for unlock, message for notify)."
            ),
            required=False,
            default=None,
        ),
        "delay": ParameterSpec(
            type="integer",
            description="Delay in seconds before executing (for shutdown/restart). Default 0.",
            required=False,
            default=0,
        ),
    }

    OPERATIONS = {
        "shutdown", "restart", "lock", "unlock", "sleep", "hibernate",
        "logout", "screenshot", "open_app", "clipboard_get", "clipboard_set",
        "volume_set", "volume_mute", "brightness_set", "notify",
    }

    @staticmethod
    def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(args, shell=False, check=False, **kwargs)

    @staticmethod
    def _clamp_percent(value) -> int:
        level = int(value)
        return max(0, min(100, level))

    def execute(
        self,
        operation: str,
        value: Optional[str] = None,
        delay: int = 0,
    ) -> ToolResult:
        op = operation.lower().strip()
        if op not in self.OPERATIONS:
            return ToolResult.error(
                f"Unknown operation '{op}'. Supported: {', '.join(sorted(self.OPERATIONS))}"
            )

        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            return ToolResult.error(f"Operation '{op}' is not implemented yet.")

        try:
            return handler(value=value, delay=delay)
        except Exception as e:
            return ToolResult.error(f"Operation '{op}' failed: {e}")

    # ── Power operations ──────────────────────

    def _op_shutdown(self, value=None, delay=0) -> ToolResult:
        minutes = str(max(int(delay) // 60, 0))
        if _OS == "windows":
            self._run(["shutdown", "/s", "/t", str(max(int(delay), 0))])
        elif _OS == "linux":
            self._run(["shutdown", "-h", f"+{minutes}"])
        elif _OS == "darwin":
            self._run(["sudo", "shutdown", "-h", f"+{minutes}"])
        return ToolResult.ok(f"Shutdown initiated (delay={delay}s).")

    def _op_restart(self, value=None, delay=0) -> ToolResult:
        minutes = str(max(int(delay) // 60, 0))
        if _OS == "windows":
            self._run(["shutdown", "/r", "/t", str(max(int(delay), 0))])
        elif _OS == "linux":
            self._run(["shutdown", "-r", f"+{minutes}"])
        elif _OS == "darwin":
            self._run(["sudo", "shutdown", "-r", f"+{minutes}"])
        return ToolResult.ok(f"Restart initiated (delay={delay}s).")

    def _op_lock(self, value=None, delay=0) -> ToolResult:
        if _OS == "windows":
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        elif _OS == "linux":
            for args in (
                ["loginctl", "lock-session"],
                ["xdg-screensaver", "lock"],
                ["gnome-screensaver-command", "-l"],
            ):
                try:
                    if self._run(args).returncode == 0:
                        break
                except FileNotFoundError:
                    continue
        elif _OS == "darwin":
            self._run(["pmset", "displaysleepnow"])
        return ToolResult.ok("Screen locked.")

    def _op_unlock(self, value=None, delay=0) -> ToolResult:
        if _OS != "windows":
            return ToolResult.error("Unlock is only supported on Windows.")
        if not value:
            return ToolResult.error("Password is required to unlock. Pass it via the 'value' parameter.")
        # Note: programmatic unlock in Windows is very restricted.
        # This uses a basic approach – in practice the user must type their password.
        return ToolResult.ok(
            "Unlock requested. Due to OS security, the user must enter "
            "their password at the lock screen."
        )

    def _op_sleep(self, value=None, delay=0) -> ToolResult:
        if _OS == "windows":
            self._run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        elif _OS == "linux":
            self._run(["systemctl", "suspend"])
        elif _OS == "darwin":
            self._run(["pmset", "sleepnow"])
        return ToolResult.ok("System entering sleep mode.")

    def _op_hibernate(self, value=None, delay=0) -> ToolResult:
        if _OS == "windows":
            self._run(["shutdown", "/h"])
        elif _OS == "linux":
            self._run(["systemctl", "hibernate"])
        elif _OS == "darwin":
            return ToolResult.error("macOS does not support hibernate directly.")
        return ToolResult.ok("System entering hibernate mode.")

    def _op_logout(self, value=None, delay=0) -> ToolResult:
        if _OS == "windows":
            self._run(["shutdown", "/l"])
        elif _OS == "linux":
            user = getpass.getuser()
            result = self._run(["loginctl", "terminate-user", user])
            if result.returncode != 0:
                self._run(["pkill", "-KILL", "-u", user])
        elif _OS == "darwin":
            self._run(["osascript", "-e", 'tell application "System Events" to log out'])
        return ToolResult.ok("Logout initiated.")

    # ── Utility operations ────────────────────

    def _op_screenshot(self, value=None, delay=0) -> ToolResult:
        output = value or "screenshot.png"
        try:
            import pyautogui
            img = pyautogui.screenshot()
            img.save(output)
            return ToolResult.ok(f"Screenshot saved to {os.path.abspath(output)}")
        except ImportError:
            # Fallback for Windows
            if _OS == "windows":
                try:
                    from PIL import ImageGrab
                    img = ImageGrab.grab()
                    img.save(output)
                    return ToolResult.ok(f"Screenshot saved to {os.path.abspath(output)}")
                except ImportError:
                    return ToolResult.error("Install pyautogui or Pillow for screenshots: pip install pyautogui")
            elif _OS == "linux":
                try:
                    subprocess.run(["scrot", output], check=True)
                except Exception:
                    subprocess.run(["gnome-screenshot", "-f", output], check=True)
                return ToolResult.ok(f"Screenshot saved to {output}")
            elif _OS == "darwin":
                subprocess.run(["screencapture", output], check=True)
                return ToolResult.ok(f"Screenshot saved to {output}")
            return ToolResult.error("pip install pyautogui for screenshots")

    def _op_open_app(self, value=None, delay=0) -> ToolResult:
        if not value:
            return ToolResult.error("Specify the app name/path in the 'value' parameter.")
        if _OS == "windows":
            os.startfile(value)  # type: ignore[attr-defined]
        elif _OS == "linux":
            subprocess.Popen([value], start_new_session=True)
        elif _OS == "darwin":
            subprocess.Popen(["open", "-a", value], start_new_session=True)
        return ToolResult.ok(f"Opened application: {value}")

    def _op_clipboard_get(self, value=None, delay=0) -> ToolResult:
        try:
            import pyperclip
            text = pyperclip.paste()
            return ToolResult.ok(content=text)
        except ImportError:
            if _OS == "windows":
                import ctypes
                CF_TEXT = 1
                ctypes.windll.user32.OpenClipboard(0)
                try:
                    handle = ctypes.windll.user32.GetClipboardData(CF_TEXT)
                    if handle:
                        data = ctypes.c_char_p(handle)
                        text = data.value.decode('utf-8', errors='replace')
                        return ToolResult.ok(content=text)
                    return ToolResult.ok(content="")
                finally:
                    ctypes.windll.user32.CloseClipboard()
            return ToolResult.error("pip install pyperclip for clipboard access")

    def _op_clipboard_set(self, value=None, delay=0) -> ToolResult:
        if value is None:
            return ToolResult.error("Specify the text in the 'value' parameter.")
        try:
            import pyperclip
            pyperclip.copy(value)
            return ToolResult.ok("Text copied to clipboard.")
        except ImportError:
            if _OS == "windows":
                subprocess.run(["clip"], input=value.encode(), check=True)
                return ToolResult.ok("Text copied to clipboard.")
            return ToolResult.error("pip install pyperclip for clipboard access")

    def _op_volume_set(self, value=None, delay=0) -> ToolResult:
        if value is None:
            return ToolResult.error("Specify volume level (0-100) in the 'value' parameter.")
        level = self._clamp_percent(value)
        if _OS == "windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level / 100.0, None)
                return ToolResult.ok(f"Volume set to {level}%.")
            except ImportError:
                # Fallback using nircmd if available
                self._run(["nircmd.exe", "setsysvolume", str(int(level / 100 * 65535))])
                return ToolResult.ok(f"Volume set to {level}% (via nircmd).")
        elif _OS == "linux":
            self._run(["amixer", "set", "Master", f"{level}%"])
            return ToolResult.ok(f"Volume set to {level}%.")
        elif _OS == "darwin":
            self._run(["osascript", "-e", f"set volume output volume {level}"])
            return ToolResult.ok(f"Volume set to {level}%.")
        return ToolResult.error("Volume control not supported on this platform.")

    def _op_volume_mute(self, value=None, delay=0) -> ToolResult:
        if _OS == "windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMute()
                volume.SetMute(not current, None)
                state = "muted" if not current else "unmuted"
                return ToolResult.ok(f"Volume {state}.")
            except ImportError:
                self._run(["nircmd.exe", "mutesysvolume", "2"])
                return ToolResult.ok("Volume mute toggled.")
        elif _OS == "linux":
            self._run(["amixer", "set", "Master", "toggle"])
            return ToolResult.ok("Volume mute toggled.")
        elif _OS == "darwin":
            self._run(["osascript", "-e", "set volume with output muted"])
            return ToolResult.ok("Volume mute toggled.")
        return ToolResult.error("Mute not supported on this platform.")

    def _op_brightness_set(self, value=None, delay=0) -> ToolResult:
        if value is None:
            return ToolResult.error("Specify brightness (0-100) in the 'value' parameter.")
        level = self._clamp_percent(value)
        if _OS == "windows":
            try:
                import wmi
                c = wmi.WMI(namespace="wmi")
                methods = c.WmiMonitorBrightnessMethods()[0]
                methods.WmiSetBrightness(level, 0)
                return ToolResult.ok(f"Brightness set to {level}%.")
            except (ImportError, Exception):
                # Fallback using PowerShell
                ps = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
                self._run(["powershell", "-Command", ps])
                return ToolResult.ok(f"Brightness set to {level}%.")
        elif _OS == "linux":
            try:
                self._run(["brightnessctl", "set", f"{level}%"])
            except FileNotFoundError:
                return ToolResult.error("Install brightnessctl for Linux brightness control.")
            return ToolResult.ok(f"Brightness set to {level}%.")
        elif _OS == "darwin":
            self._run(["brightness", str(level / 100.0)])
            return ToolResult.ok(f"Brightness set to {level}%.")
        return ToolResult.error("Brightness control not supported on this platform.")

    def _op_notify(self, value=None, delay=0) -> ToolResult:
        message = value or "NeuralNode notification"
        if _OS == "windows":
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast("NeuralNode", message, duration=5, threaded=True)
                return ToolResult.ok(f"Notification sent: {message}")
            except ImportError:
                # PowerShell fallback
                ps = (
                    f"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
                    f"ContentType = WindowsRuntime] > $null; "
                    f"$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); "
                    f"$text = $template.GetElementsByTagName('text'); "
                    f"$text.Item(0).AppendChild($template.CreateTextNode('{message}')) > $null; "
                    f"$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                    f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('NeuralNode').Show($toast)"
                )
                self._run(["powershell", "-Command", ps])
                return ToolResult.ok(f"Notification sent: {message}")
        elif _OS == "linux":
            self._run(["notify-send", "NeuralNode", message])
            return ToolResult.ok(f"Notification sent: {message}")
        elif _OS == "darwin":
            self._run(["osascript", "-e", f'display notification "{message}" with title "NeuralNode"'])
            return ToolResult.ok(f"Notification sent: {message}")
        return ToolResult.error("Notifications not supported on this platform.")
