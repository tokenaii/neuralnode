"""Time context utilities for real-time awareness."""

from datetime import datetime
from typing import Optional


def get_current_time_info() -> dict:
    """Get current time and date information."""
    now = datetime.now()
    return {
        "current_time": now.strftime("%H:%M:%S"),
        "current_date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
        "timezone": "local",
        "iso_timestamp": now.isoformat(),
    }


def inject_time_context(custom_prompt: Optional[str] = None) -> str:
    """
    Generate a system prompt with current time context.
    
    Args:
        custom_prompt: Optional custom system prompt to prepend
        
    Returns:
        System prompt string with time context
    """
    time_info = get_current_time_info()
    
    time_context = f"""You have access to real-time information:
- Current Date: {time_info['current_date']}
- Current Time: {time_info['current_time']}
- Day: {time_info['day_of_week']}
- Timezone: {time_info['timezone']}

Use this information when relevant to the user's query."""
    
    if custom_prompt:
        return f"{custom_prompt}\n\n{time_context}"
    
    return time_context


def format_time_for_prompt(time_format: str = "full") -> str:
    """
    Format current time in various formats for prompts.
    
    Args:
        time_format: One of "full", "short", "date_only", "time_only"
        
    Returns:
        Formatted time string
    """
    now = datetime.now()
    
    formats = {
        "full": now.strftime("%Y-%m-%d %H:%M:%S (%A)"),
        "short": now.strftime("%Y-%m-%d %H:%M"),
        "date_only": now.strftime("%Y-%m-%d"),
        "time_only": now.strftime("%H:%M:%S"),
    }
    
    return formats.get(time_format, formats["full"])
