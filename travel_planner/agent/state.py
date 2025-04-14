from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentState:
    preferences: Dict[str, Any] = field(default_factory=dict)
    destinations: List[Dict[str, Any]] = field(default_factory=list)
    itinerary: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    is_followup: bool = False
    error: Optional[str] = None
    followup_responses: List[Dict[str, str]] = field(default_factory=list)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state with a default if not found."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to attributes."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of attributes."""
        setattr(self, key, value)

    def update_preferences(self, new_preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        self.preferences.update(new_preferences)
        self._add_to_history("preferences_updated", str(new_preferences))
    
    def add_destinations(self, destinations: List[Dict[str, Any]]) -> None:
        """Add recommended destinations"""
        self.destinations.extend(destinations)
        self._add_to_history("destinations_added", str(destinations))
    
    def set_itinerary(self, itinerary: Dict[str, Any]) -> None:
        """Set the travel itinerary"""
        self.itinerary = itinerary
        self._add_to_history("itinerary_created", str(itinerary))
    
    def _add_to_history(self, action: str, details: str) -> None:
        """Add an action to conversation history"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }) 