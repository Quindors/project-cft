#!/usr/bin/env python3
"""
sliding_window_simple.py

Start simple: Just ONE sliding window (30 seconds of recent activity).
We'll add more scales later once this works well.
"""

from typing import List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Event:
    """Single activity event."""
    timestamp: datetime
    key: str  # e.g., "WIN:Chrome - YouTube"
    
    def __str__(self):
        return self.key


class SimpleWindowAnalyzer:
    """
    Manages a single 30-second sliding window.
    Much simpler than multi-scale - just tracks recent activity.
    """
    
    WINDOW_SEC = 30  # Look at last 30 seconds
    
    def __init__(self):
        self.events: List[Event] = []
    
    def add_event(self, timestamp: datetime, key: str):
        """Add a new event to the stream."""
        event = Event(timestamp=timestamp, key=key)
        self.events.append(event)
        
        # Trim old events (keep 5 minutes of history just in case)
        cutoff = timestamp - timedelta(seconds=300)
        self.events = [e for e in self.events if e.timestamp > cutoff]
    
    def get_recent_events(self, at_time: datetime = None) -> List[Event]:
        """
        Get events from the last 30 seconds.
        """
        if not self.events:
            return []
        
        if at_time is None:
            at_time = self.events[-1].timestamp
        
        cutoff = at_time - timedelta(seconds=self.WINDOW_SEC)
        return [e for e in self.events if e.timestamp > cutoff and e.timestamp <= at_time]
    
    def get_summary(self, at_time: datetime = None) -> Dict:
        """
        Get summary of the last 30 seconds.
        """
        recent_events = self.get_recent_events(at_time)
        
        if at_time is None:
            at_time = self.events[-1].timestamp if self.events else datetime.now()
        
        # Find current window from ALL events (not just last 30 seconds)
        current_window = None
        for event in reversed(self.events):
            if event.timestamp <= at_time:
                current_window = event.key
                break
        
        if current_window is None:
            return {
                "count": 0,
                "unique_windows": 0,
                "switches": 0,
                "dwell_time": 0,
                "current_window": None,
                "events": []
            }
        
        # Count unique windows in last 30 seconds
        unique = len(set(e.key for e in recent_events)) if recent_events else 1
        
        # Count switches in last 30 seconds
        switches = sum(1 for i in range(1, len(recent_events)) 
                      if recent_events[i].key != recent_events[i-1].key) if recent_events else 0
        
        # Calculate dwell time (how long on current window)
        # IMPORTANT: Dwell time can exceed 30 seconds if user has been on same window longer
        dwell_start = at_time
        for event in reversed(self.events):  # Use ALL events, not just recent
            if event.timestamp > at_time:
                continue  # Skip future events
            if event.key == current_window:
                dwell_start = event.timestamp
            else:
                break  # Stop when we find a different window
        
        dwell_time = (at_time - dwell_start).total_seconds()
        
        return {
            "count": len(recent_events),
            "unique_windows": unique,
            "switches": switches,
            "dwell_time": dwell_time,
            "current_window": current_window,
            "events": [str(e) for e in recent_events]
        }
    
    def format_for_display(self, at_time: datetime = None) -> str:
        """Format recent activity for display."""
        summary = self.get_summary(at_time)
        
        if summary["count"] == 0:
            return "No recent activity (last 30 seconds)"
        
        output = []
        output.append("=== LAST 30 SECONDS ===")
        output.append(f"Total events: {summary['count']}")
        output.append(f"Unique windows: {summary['unique_windows']}")
        output.append(f"Window switches: {summary['switches']}")
        output.append(f"Current window: {summary['current_window']}")
        output.append(f"Time on current window: {summary['dwell_time']:.1f} seconds")
        
        output.append("\nRecent events (oldest → newest):")
        for event in summary['events'][-10:]:  # Show last 10
            output.append(f"  - {event}")
        
        return "\n".join(output)
    
    def check_distracting_keywords(self, at_time: datetime = None) -> Dict[str, int]:
        """Check for distracting keywords in recent activity."""
        events = self.get_recent_events(at_time)
        
        distracting_keywords = [
            "youtube", "reddit", "twitter", "discord", 
            "instagram", "tiktok", "netflix", "twitch",
            "facebook", "game", "gaming"
        ]
        
        keywords_found = {}
        
        for event in events:
            key_lower = event.key.lower()
            for keyword in distracting_keywords:
                if keyword in key_lower:
                    keywords_found[keyword] = keywords_found.get(keyword, 0) + 1
        
        return keywords_found
    
    def assess_focus(self, at_time: datetime = None) -> Dict:
        """
        Simple focus assessment based on 30-second window.
        Returns a dict with assessment and reasoning.
        """
        summary = self.get_summary(at_time)
        keywords = self.check_distracting_keywords(at_time)
        
        if summary["current_window"] is None:
            return {
                "assessment": "NO_DATA",
                "confidence": 0.0,
                "reason": "No recent activity to assess"
            }
        
        # Simple heuristics
        score = 0.0
        reasons = []
        
        # Check for distracting keywords
        if keywords:
            score -= 0.5
            reasons.append(f"Distracting apps detected: {list(keywords.keys())}")
        else:
            # Check if current window itself is distracting
            current_lower = summary["current_window"].lower()
            distracting = ["youtube", "reddit", "twitter", "discord", 
                          "instagram", "tiktok", "netflix", "twitch",
                          "facebook", "game", "gaming"]
            if any(word in current_lower for word in distracting):
                score -= 0.5
                reasons.append(f"Currently on distracting app")
            else:
                score += 0.3
                reasons.append("No obviously distracting apps")
        
        # Check dwell time
        dwell = summary["dwell_time"]
        if dwell < 5:
            score -= 0.2
            reasons.append(f"Very short dwell time ({dwell:.1f}s) - may be scattered")
        elif dwell < 20:
            score += 0.2
            reasons.append(f"Healthy dwell time ({dwell:.1f}s)")
        elif dwell < 300:  # Less than 5 minutes
            score += 0.1
            reasons.append(f"Extended focus on current window ({dwell:.1f}s)")
        else:  # More than 5 minutes on same window
            # Long dwell could be good (deep focus) or bad (passive viewing)
            # Check if it's a distracting app
            current_lower = summary["current_window"].lower()
            if any(word in current_lower for word in ["youtube", "reddit", "netflix", "twitch"]):
                score -= 0.4
                reasons.append(f"Very long dwell on distracting app ({dwell/60:.1f} min) - likely passive viewing")
            else:
                score += 0.2
                reasons.append(f"Deep focus session ({dwell/60:.1f} min)")
        
        # Check switches (only if we have events in the window)
        if summary["count"] > 0:
            if summary["switches"] > 5:
                score -= 0.3
                reasons.append(f"High switching rate ({summary['switches']} switches in 30s)")
            elif summary["switches"] <= 1:
                score += 0.2
                reasons.append("Stable focus (minimal switching)")
        elif summary["count"] == 0 and dwell > 30:
            # No events in window but long dwell = stable focus
            score += 0.3
            reasons.append(f"Very stable (no switches in last 30s, {dwell:.0f}s on current window)")
        
        # Make assessment
        if score > 0.3:
            assessment = "ON-TASK"
            confidence = min(1.0, score)
        elif score < -0.3:
            assessment = "OFF-TASK"
            confidence = min(1.0, -score)
        else:
            assessment = "UNCERTAIN"
            confidence = 0.5
        
        return {
            "assessment": assessment,
            "confidence": confidence,
            "score": score,
            "reason": "; ".join(reasons)
        }


def demo():
    """Demo the simple sliding window."""
    analyzer = SimpleWindowAnalyzer()
    now = datetime.now()
    
    print("="*70)
    print("SIMPLE SLIDING WINDOW DEMO (30 seconds)")
    print("="*70)
    
    # Scenario 1: Focused work
    print("\n" + "-"*70)
    print("SCENARIO 1: Focused coding")
    print("-"*70)
    
    events = [
        (now - timedelta(seconds=30), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=25), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=20), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=15), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=10), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=5), "WIN:VS Code - main.py"),
        (now, "WIN:VS Code - main.py"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display())
    assessment = analyzer.assess_focus()
    print(f"\nASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")
    
    # Scenario 2: Distracted
    print("\n" + "-"*70)
    print("SCENARIO 2: Watching YouTube")
    print("-"*70)
    
    analyzer = SimpleWindowAnalyzer()  # Reset
    events = [
        (now - timedelta(seconds=30), "WIN:Chrome - YouTube - Funny Videos"),
        (now - timedelta(seconds=25), "WIN:Chrome - YouTube - Funny Videos"),
        (now - timedelta(seconds=20), "WIN:Chrome - YouTube - Funny Videos"),
        (now - timedelta(seconds=15), "WIN:Chrome - YouTube - Funny Videos"),
        (now - timedelta(seconds=10), "WIN:Chrome - YouTube - Funny Videos"),
        (now - timedelta(seconds=5), "WIN:Chrome - YouTube - Funny Videos"),
        (now, "WIN:Chrome - YouTube - Funny Videos"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display())
    assessment = analyzer.assess_focus()
    print(f"\nASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")
    
    # Scenario 3: Task switching
    print("\n" + "-"*70)
    print("SCENARIO 3: Rapid task switching")
    print("-"*70)
    
    analyzer = SimpleWindowAnalyzer()  # Reset
    events = [
        (now - timedelta(seconds=30), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=25), "WIN:Chrome - Stack Overflow"),
        (now - timedelta(seconds=20), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=15), "WIN:Discord - #general"),
        (now - timedelta(seconds=10), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=5), "WIN:Chrome - Twitter"),
        (now, "WIN:VS Code - main.py"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display())
    assessment = analyzer.assess_focus()
    print(f"\nASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demo()