"""Calendar integration for Google and Outlook calendars."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CalendarEvent:
    """Calendar event structure."""
    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    event_id: Optional[str] = None
    calendar_id: str = "primary"
    is_all_day: bool = False
    recurrence: Optional[str] = None


class BaseCalendarClient:
    """Base class for calendar clients."""
    
    def list_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[CalendarEvent]:
        """List calendar events."""
        raise NotImplementedError
    
    def create_event(self, event: CalendarEvent) -> Optional[str]:
        """Create a new calendar event."""
        raise NotImplementedError
    
    def update_event(self, event_id: str, event: CalendarEvent) -> bool:
        """Update an existing event."""
        raise NotImplementedError
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event."""
        raise NotImplementedError
    
    def find_free_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 60,
    ) -> List[datetime]:
        """Find available time slots."""
        raise NotImplementedError


class GoogleCalendarClient(BaseCalendarClient):
    """Google Calendar client."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        self.credentials = credentials
        self._service = None
        
        if credentials:
            self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate with Google Calendar API."""
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            
            if self.credentials:
                creds = Credentials.from_authorized_user_info(self.credentials)
                self._service = build('calendar', 'v3', credentials=creds)
        except ImportError:
            print("Google API client not installed. Install: pip install google-api-python-client")
    
    def list_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[CalendarEvent]:
        """List Google Calendar events."""
        if not self._service:
            print("Not authenticated with Google Calendar")
            return []
        
        # Default to today if not specified
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=7)
        
        try:
            # Call the Calendar API
            events_result = self._service.events().list(
                calendarId='primary',
                timeMin=start_date.isoformat(),
                timeMax=end_date.isoformat(),
                maxResults=limit,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            calendar_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse datetime
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    is_all_day = False
                else:
                    start_dt = datetime.strptime(start, '%Y-%m-%d')
                    end_dt = datetime.strptime(end, '%Y-%m-%d')
                    is_all_day = True
                
                calendar_events.append(CalendarEvent(
                    title=event.get('summary', 'No title'),
                    start_time=start_dt,
                    end_time=end_dt,
                    description=event.get('description'),
                    location=event.get('location'),
                    attendees=[a.get('email') for a in event.get('attendees', [])],
                    event_id=event.get('id'),
                    is_all_day=is_all_day,
                ))
            
            return calendar_events
        except Exception as e:
            print(f"Error listing Google Calendar events: {e}")
            return []
    
    def create_event(self, event: CalendarEvent) -> Optional[str]:
        """Create a Google Calendar event."""
        if not self._service:
            print("Not authenticated with Google Calendar")
            return None
        
        try:
            event_body = {
                'summary': event.title,
                'description': event.description,
                'location': event.location,
            }
            
            if event.is_all_day:
                event_body['start'] = {
                    'date': event.start_time.strftime('%Y-%m-%d'),
                }
                event_body['end'] = {
                    'date': event.end_time.strftime('%Y-%m-%d'),
                }
            else:
                event_body['start'] = {
                    'dateTime': event.start_time.isoformat(),
                }
                event_body['end'] = {
                    'dateTime': event.end_time.isoformat(),
                }
            
            if event.attendees:
                event_body['attendees'] = [{'email': e} for e in event.attendees]
            
            if event.recurrence:
                event_body['recurrence'] = [event.recurrence]
            
            created_event = self._service.events().insert(
                calendarId=event.calendar_id,
                body=event_body
            ).execute()
            
            return created_event.get('id')
        except Exception as e:
            print(f"Error creating Google Calendar event: {e}")
            return None
    
    def delete_event(self, event_id: str) -> bool:
        """Delete a Google Calendar event."""
        if not self._service:
            print("Not authenticated with Google Calendar")
            return False
        
        try:
            self._service.events().delete(calendarId='primary', eventId=event_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting Google Calendar event: {e}")
            return False


class OutlookCalendarClient(BaseCalendarClient):
    """Microsoft Outlook Calendar client."""
    
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to Microsoft Graph API."""
        import requests
        
        if not self.access_token:
            print("Not authenticated with Outlook")
            return None
        
        url = f"{self.graph_endpoint}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                return None
            
            if response.status_code in [200, 201]:
                return response.json() if response.content else {}
            else:
                print(f"Outlook API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error making Outlook request: {e}")
            return None
    
    def list_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[CalendarEvent]:
        """List Outlook Calendar events."""
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=7)
        
        # Format dates for OData query
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        endpoint = f"/me/calendarview?startDateTime={start_str}&endDateTime={end_str}&$top={limit}"
        
        data = self._make_request('GET', endpoint)
        
        if not data:
            return []
        
        events = []
        for item in data.get('value', []):
            start = item.get('start', {}).get('dateTime')
            end = item.get('end', {}).get('dateTime')
            
            if start and end:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                
                events.append(CalendarEvent(
                    title=item.get('subject', 'No title'),
                    start_time=start_dt,
                    end_time=end_dt,
                    description=item.get('bodyPreview'),
                    location=item.get('location', {}).get('displayName'),
                    event_id=item.get('id'),
                ))
        
        return events
    
    def create_event(self, event: CalendarEvent) -> Optional[str]:
        """Create an Outlook Calendar event."""
        event_data = {
            'subject': event.title,
            'body': {
                'contentType': 'text',
                'content': event.description or ''
            },
            'start': {
                'dateTime': event.start_time.isoformat(),
                'timeZone': 'UTC'
            },
            'end': {
                'dateTime': event.end_time.isoformat(),
                'timeZone': 'UTC'
            }
        }
        
        if event.location:
            event_data['location'] = {
                'displayName': event.location
            }
        
        if event.attendees:
            event_data['attendees'] = [
                {
                    'emailAddress': {
                        'address': email
                    },
                    'type': 'required'
                }
                for email in event.attendees
            ]
        
        result = self._make_request('POST', '/me/events', event_data)
        
        return result.get('id') if result else None
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an Outlook Calendar event."""
        result = self._make_request('DELETE', f'/me/events/{event_id}')
        return result is not None


class CalendarTool:
    """Calendar tool for AI agents."""
    
    def __init__(self, client: Optional[BaseCalendarClient] = None):
        self.client = client
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent."""
        return {
            "name": "calendar",
            "description": "Manage calendar events - list, create, or delete events",
            "parameters": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "delete"],
                    "description": "Action to perform"
                },
                "title": {
                    "type": "string",
                    "description": "Event title (for create)"
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format (for create)"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format (for create)"
                },
                "event_id": {
                    "type": "string",
                    "description": "Event ID (for delete)"
                },
            },
            "required": ["action"]
        }
    
    def execute(self, action: str, **kwargs) -> str:
        """Execute calendar action."""
        if not self.client:
            return "Error: Calendar client not configured"
        
        try:
            if action == "list":
                events = self.client.list_events()
                if not events:
                    return "No upcoming events found."
                
                result = "Upcoming events:\n\n"
                for event in events[:10]:
                    result += f"📅 {event.title}\n"
                    result += f"   Time: {event.start_time.strftime('%Y-%m-%d %H:%M')}\n"
                    if event.location:
                        result += f"   Location: {event.location}\n"
                    result += f"   ID: {event.event_id}\n\n"
                return result
            
            elif action == "create":
                event = CalendarEvent(
                    title=kwargs.get("title", "New Event"),
                    start_time=datetime.fromisoformat(kwargs.get("start_time", "")),
                    end_time=datetime.fromisoformat(kwargs.get("end_time", "")),
                    description=kwargs.get("description"),
                    location=kwargs.get("location"),
                )
                
                event_id = self.client.create_event(event)
                if event_id:
                    return f"Event created successfully with ID: {event_id}"
                else:
                    return "Failed to create event"
            
            elif action == "delete":
                event_id = kwargs.get("event_id")
                if not event_id:
                    return "Error: event_id required for delete"
                
                success = self.client.delete_event(event_id)
                if success:
                    return f"Event {event_id} deleted successfully"
                else:
                    return f"Failed to delete event {event_id}"
            
            else:
                return f"Unknown action: {action}"
        
        except Exception as e:
            return f"Error: {str(e)}"


# Convenience functions
def list_google_calendar_events(
    credentials: Dict[str, Any],
    start_date: Optional[datetime] = None,
    days: int = 7,
) -> List[CalendarEvent]:
    """Quick function to list Google Calendar events."""
    client = GoogleCalendarClient(credentials=credentials)
    
    if not start_date:
        start_date = datetime.now()
    
    end_date = start_date + timedelta(days=days)
    
    return client.list_events(start_date, end_date)


def create_google_calendar_event(
    credentials: Dict[str, Any],
    title: str,
    start_time: datetime,
    end_time: datetime,
    description: Optional[str] = None,
) -> Optional[str]:
    """Quick function to create Google Calendar event."""
    client = GoogleCalendarClient(credentials=credentials)
    
    event = CalendarEvent(
        title=title,
        start_time=start_time,
        end_time=end_time,
        description=description,
    )
    
    return client.create_event(event)
