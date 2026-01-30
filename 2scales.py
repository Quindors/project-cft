#!/usr/bin/env python3
"""
sliding_window_two_scales.py

Step 2: Add MESO window (3 minutes) alongside MICRO window (30 seconds).

MICRO (30s): What's happening RIGHT NOW?
MESO (3min): What TASK are you working on?
"""

from typing import List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Event:
    """Single activity event."""
    timestamp: datetime
    key: str
    
    def __str__(self):
        return self.key


class TwoScaleAnalyzer:
    """
    Analyzes activity at two time scales:
    - MICRO: Last 30 seconds (immediate activity)
    - MESO: Last 3 minutes (current task)
    """
    
    MICRO_SEC = 30    # 30 seconds
    MESO_SEC = 180    # 3 minutes
    
    def __init__(self):
        self.events: List[Event] = []
    
    def add_event(self, timestamp: datetime, key: str):
        """Add a new event to the stream."""
        event = Event(timestamp=timestamp, key=key)
        self.events.append(event)
        
        # Trim old events (keep 5 minutes of history)
        cutoff = timestamp - timedelta(seconds=300)
        self.events = [e for e in self.events if e.timestamp > cutoff]
    
    def get_events_in_window(self, window_sec: int, at_time: datetime = None) -> List[Event]:
        """Get events within a specific time window."""
        if not self.events:
            return []
        
        if at_time is None:
            at_time = self.events[-1].timestamp
        
        cutoff = at_time - timedelta(seconds=window_sec)
        return [e for e in self.events if e.timestamp > cutoff and e.timestamp <= at_time]
    
    def get_current_window(self, at_time: datetime = None) -> str:
        """Find current window from all events."""
        if not self.events:
            return None
        
        if at_time is None:
            at_time = self.events[-1].timestamp
        
        for event in reversed(self.events):
            if event.timestamp <= at_time:
                return event.key
        
        return None
    
    def get_dwell_time(self, at_time: datetime = None) -> float:
        """Calculate how long user has been on current window."""
        current_window = self.get_current_window(at_time)
        if current_window is None:
            return 0.0
        
        if at_time is None:
            at_time = self.events[-1].timestamp if self.events else datetime.now()
        
        # Walk backwards to find when current window started
        dwell_start = at_time
        for event in reversed(self.events):
            if event.timestamp > at_time:
                continue
            if event.key == current_window:
                dwell_start = event.timestamp
            else:
                break
        
        return (at_time - dwell_start).total_seconds()
    
    def get_summary(self, window_sec: int, at_time: datetime = None) -> Dict:
        """Get summary for a specific window size."""
        events = self.get_events_in_window(window_sec, at_time)
        
        if not events:
            return {
                "count": 0,
                "unique_windows": 0,
                "switches": 0,
                "events": []
            }
        
        unique = len(set(e.key for e in events))
        switches = sum(1 for i in range(1, len(events)) if events[i].key != events[i-1].key)
        
        return {
            "count": len(events),
            "unique_windows": unique,
            "switches": switches,
            "events": [str(e) for e in events]
        }
    
    def get_two_scale_context(self, at_time: datetime = None) -> Dict:
        """
        Get context at both scales.
        This is the key method that shows the power of multi-scale analysis.
        """
        if at_time is None:
            at_time = datetime.now()
        
        micro = self.get_summary(self.MICRO_SEC, at_time)
        meso = self.get_summary(self.MESO_SEC, at_time)
        current = self.get_current_window(at_time)
        dwell = self.get_dwell_time(at_time)
        
        return {
            "timestamp": at_time,
            "current_window": current,
            "dwell_time": dwell,
            "micro": micro,
            "meso": meso
        }
    
    def check_distracting_keywords(self, text: str) -> bool:
        """Check if text contains distracting keywords."""
        if not text:
            return False
        
        text_lower = text.lower()
        distracting = [
            "youtube", "reddit", "twitter", "discord",
            "instagram", "tiktok", "netflix", "twitch",
            "facebook", "game", "gaming"
        ]
        
        return any(word in text_lower for word in distracting)
    
    def compare_scales(self, context: Dict) -> Dict:
        """
        Compare micro vs meso scales to understand the pattern.
        This is where two-scale analysis shines!
        """
        micro = context["micro"]
        meso = context["meso"]
        current = context["current_window"]
        dwell = context["dwell_time"]
        
        insights = {
            "pattern": None,
            "confidence": 0.5,
            "explanation": []
        }
        
        # Pattern 1: Stable focus (same activity across both scales)
        if micro["count"] > 0 and meso["count"] > 0:
            if micro["unique_windows"] == 1 and meso["unique_windows"] <= 2:
                insights["pattern"] = "STABLE_FOCUS"
                insights["confidence"] = 0.8
                insights["explanation"].append("Consistent activity across both scales")
        
        # Pattern 2: Recent switch (different micro vs meso)
        if micro["count"] > 0 and meso["count"] > 0:
            # Get most common window in meso
            meso_windows = {}
            for event_str in meso["events"]:
                meso_windows[event_str] = meso_windows.get(event_str, 0) + 1
            
            if meso_windows:
                prev_window = max(meso_windows, key=meso_windows.get)
                if current and current != prev_window:
                    insights["pattern"] = "RECENT_SWITCH"
                    insights["confidence"] = 0.7
                    insights["explanation"].append(f"Switched from {prev_window[:40]} to current window")
        
        # Pattern 3: High churn (lots of switching)
        if meso["count"] > 0:
            switch_rate = meso["switches"] / (self.MESO_SEC / 60)  # switches per minute
            if switch_rate > 3:
                insights["pattern"] = "HIGH_CHURN"
                insights["confidence"] = 0.8
                insights["explanation"].append(f"High switching rate: {switch_rate:.1f} switches/min")
        
        # Pattern 4: Extended dwell
        if dwell > 120:  # More than 2 minutes
            insights["pattern"] = "EXTENDED_DWELL"
            insights["confidence"] = 0.8
            insights["explanation"].append(f"Extended time on current window: {dwell/60:.1f} min")
        
        # Check if current or meso contains distracting content
        distraction_in_current = self.check_distracting_keywords(current or "")
        distraction_in_meso = any(self.check_distracting_keywords(e) for e in meso["events"])
        
        if distraction_in_current:
            insights["explanation"].append("Currently on distracting app")
        elif distraction_in_meso and not distraction_in_current:
            insights["explanation"].append("Was recently on distracting apps, now switched away")
        
        return insights
    
    def assess_focus_two_scale(self, at_time: datetime = None) -> Dict:
        """
        Assess focus using both scales.
        The key improvement over single-scale!
        """
        context = self.get_two_scale_context(at_time)
        insights = self.compare_scales(context)
        
        current = context["current_window"]
        dwell = context["dwell_time"]
        micro = context["micro"]
        meso = context["meso"]
        
        if current is None:
            return {
                "assessment": "NO_DATA",
                "confidence": 0.0,
                "reason": "No activity to assess",
                "context": context
            }
        
        score = 0.0
        reasons = []
        
        # Check current window
        if self.check_distracting_keywords(current):
            score -= 0.4
            reasons.append("Currently on distracting app")
        else:
            score += 0.3
            reasons.append("On productive app")
        
        # Use insights from scale comparison
        if insights["pattern"] == "STABLE_FOCUS":
            score += 0.3
            reasons.append("Stable focus across time scales")
        elif insights["pattern"] == "HIGH_CHURN":
            score -= 0.4
            reasons.append("High switching rate indicates distraction")
        elif insights["pattern"] == "RECENT_SWITCH":
            # Neutral - switching is normal
            reasons.append("Recently switched tasks")
        
        # Dwell time assessment (context-aware)
        if dwell < 5:
            score -= 0.2
            reasons.append(f"Very short dwell ({dwell:.0f}s)")
        elif dwell < 30:
            score += 0.2
            reasons.append(f"Active engagement ({dwell:.0f}s)")
        elif dwell < 180:  # Less than 3 minutes
            if self.check_distracting_keywords(current):
                score -= 0.1
                reasons.append(f"Extended dwell on distraction ({dwell/60:.1f}min)")
            else:
                score += 0.3
                reasons.append(f"Good focus session ({dwell/60:.1f}min)")
        else:  # More than 3 minutes
            if self.check_distracting_keywords(current):
                score -= 0.5
                reasons.append(f"Very long dwell on distraction ({dwell/60:.1f}min) - likely passive")
            else:
                score += 0.4
                reasons.append(f"Deep focus session ({dwell/60:.1f}min)")
        
        # Check meso for context
        if meso["count"] > 0:
            distraction_count = sum(1 for e in meso["events"] if self.check_distracting_keywords(e))
            if distraction_count > meso["count"] * 0.7:
                score -= 0.3
                reasons.append(f"Mostly distracting apps in last 3min ({distraction_count}/{meso['count']})")
            elif distraction_count > 0 and not self.check_distracting_keywords(current):
                score += 0.2
                reasons.append("Recovered from distraction")
        
        # Make assessment
        if score > 0.4:
            assessment = "ON-TASK"
            confidence = min(1.0, score)
        elif score < -0.4:
            assessment = "OFF-TASK"
            confidence = min(1.0, -score)
        else:
            assessment = "UNCERTAIN"
            confidence = 0.5
        
        return {
            "assessment": assessment,
            "confidence": confidence,
            "score": score,
            "reason": "; ".join(reasons),
            "context": context,
            "insights": insights
        }
    
    def format_for_display(self, at_time: datetime = None) -> str:
        """Format two-scale analysis for display."""
        context = self.get_two_scale_context(at_time)
        insights = self.compare_scales(context)
        
        output = []
        output.append("="*70)
        output.append("TWO-SCALE ANALYSIS")
        output.append("="*70)
        
        output.append(f"\nCurrent window: {context['current_window']}")
        output.append(f"Dwell time: {context['dwell_time']:.1f}s ({context['dwell_time']/60:.1f}min)")
        
        output.append("\n--- MICRO (last 30 seconds) ---")
        micro = context["micro"]
        output.append(f"Events: {micro['count']}, Unique: {micro['unique_windows']}, Switches: {micro['switches']}")
        if micro["events"]:
            output.append("Recent:")
            for event in micro["events"][-5:]:
                output.append(f"  • {event}")
        
        output.append("\n--- MESO (last 3 minutes) ---")
        meso = context["meso"]
        output.append(f"Events: {meso['count']}, Unique: {meso['unique_windows']}, Switches: {meso['switches']}")
        
        output.append("\n--- PATTERN ANALYSIS ---")
        output.append(f"Pattern: {insights['pattern'] or 'NORMAL'}")
        output.append(f"Confidence: {insights['confidence']:.0%}")
        for exp in insights["explanation"]:
            output.append(f"  • {exp}")
        
        return "\n".join(output)


def demo():
    """Demo two-scale analysis with realistic scenarios."""
    
    print("\n" + "="*70)
    print("TWO-SCALE SLIDING WINDOW DEMO")
    print("="*70)
    print("\nMICRO (30s): Immediate activity")
    print("MESO (3min): Current task context")
    print()
    
    # Scenario 1: Brief YouTube check while working
    print("\n" + "-"*70)
    print("SCENARIO 1: Brief YouTube Check (Work-Related)")
    print("-"*70)
    
    analyzer = TwoScaleAnalyzer()
    now = datetime.now()
    
    events = [
        # Been coding for 2.5 minutes
        (now - timedelta(seconds=170), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=150), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=130), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=110), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=90), "WIN:VS Code - main.py"),
        # Quick YouTube check 25 seconds ago
        (now - timedelta(seconds=25), "WIN:Chrome - YouTube - Python Tutorial"),
        # Back to coding
        (now - timedelta(seconds=10), "WIN:VS Code - main.py"),
        (now, "WIN:VS Code - main.py"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display(now))
    assessment = analyzer.assess_focus_two_scale(now)
    print(f"\n{'='*70}")
    print(f"ASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.0%})")
    print(f"Reason: {assessment['reason']}")
    print(f"{'='*70}")
    
    # Scenario 2: Extended YouTube viewing
    print("\n" + "-"*70)
    print("SCENARIO 2: Extended YouTube Viewing (Distraction)")
    print("-"*70)
    
    analyzer = TwoScaleAnalyzer()
    
    events = [
        # Was coding 3 minutes ago
        (now - timedelta(seconds=180), "WIN:VS Code - homework.py"),
        # Switched to YouTube 2.5 minutes ago and stayed
        (now - timedelta(seconds=150), "WIN:Chrome - YouTube - Funny Cats"),
        (now, "WIN:Chrome - YouTube - Funny Cats"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display(now))
    assessment = analyzer.assess_focus_two_scale(now)
    print(f"\n{'='*70}")
    print(f"ASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.0%})")
    print(f"Reason: {assessment['reason']}")
    print(f"{'='*70}")
    
    # Scenario 3: Recovery from distraction
    print("\n" + "-"*70)
    print("SCENARIO 3: Recovery from Distraction")
    print("-"*70)
    
    analyzer = TwoScaleAnalyzer()
    
    events = [
        # Distracted for first 2 minutes
        (now - timedelta(seconds=180), "WIN:Discord - #general"),
        (now - timedelta(seconds=150), "WIN:Chrome - Reddit"),
        (now - timedelta(seconds=120), "WIN:Discord - #general"),
        # Recovered: back to work 1 minute ago
        (now - timedelta(seconds=60), "WIN:VS Code - project.py"),
        (now, "WIN:VS Code - project.py"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    print(analyzer.format_for_display(now))
    assessment = analyzer.assess_focus_two_scale(now)
    print(f"\n{'='*70}")
    print(f"ASSESSMENT: {assessment['assessment']} (confidence: {assessment['confidence']:.0%})")
    print(f"Reason: {assessment['reason']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()