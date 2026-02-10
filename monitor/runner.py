"""
runner.py - Main monitoring loop for cft productivity tracker

Category-based classification system with critic verification.
"""

import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import sys

from monitor.config import Settings, load_settings
from monitor.windows_monitor import get_active_window_title
from monitor.keystroke_monitor import KeystrokeMonitor
from monitor.sheets_sink import SheetsSink
from monitor.decider import decide_with_critic
from monitor.popup import show_popup


class ProductivityRunner:
    """Main runner for productivity monitoring."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sheets_sink = None
        self.keystroke_monitor = None
        self.categories = []
        self.user_goals = ""
        
        # Event buffer
        self.events_buffer: List[Tuple[str, str]] = []  # (timestamp, window_title)
        
        # State tracking
        self.running = True
        self.last_decision_time = None
        
    def initialize(self):
        """Initialize all components."""
        print("[runner] Initializing productivity monitor...")
        
        # Initialize Google Sheets
        try:
            self.sheets_sink = SheetsSink(self.settings)
            print("[runner] ✓ Google Sheets initialized")
        except Exception as e:
            print(f"[runner] ✗ Failed to initialize Google Sheets: {e}")
            sys.exit(1)
        
        # Load categories from sheets
        try:
            self.categories = self.sheets_sink.get_categories()
            print(f"[runner] ✓ Loaded {len(self.categories)} categories")
            for cat in self.categories:
                status = "✓" if cat['is_productive'] else "✗"
                print(f"[runner]   {status} {cat['name']}: {cat['description']}")
        except Exception as e:
            print(f"[runner] ✗ Failed to load categories: {e}")
            sys.exit(1)
        
        # Load user goals
        try:
            self.user_goals = self.settings.load_user_goals()
            print(f"[runner] ✓ Loaded user goals ({len(self.user_goals)} chars)")
        except Exception as e:
            print(f"[runner] ! Warning: Could not load user goals: {e}")
            print(f"[runner]   Using default goals")
        
        # Initialize keystroke monitor
        try:
            self.keystroke_monitor = KeystrokeMonitor()
            self.keystroke_monitor.start()
            print("[runner] ✓ Keystroke monitor started")
        except Exception as e:
            print(f"[runner] ! Warning: Could not start keystroke monitor: {e}")
    
    def collect_window_event(self):
        """Collect current window title and add to buffer."""
        try:
            window_title = get_active_window_title()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to buffer
            self.events_buffer.append((timestamp, window_title))
            
            # Keep only recent events (last N seconds)
            cutoff_size = self.settings.context_window_size
            if len(self.events_buffer) > cutoff_size:
                self.events_buffer = self.events_buffer[-cutoff_size:]
            
            return window_title
        except Exception as e:
            print(f"[runner] Error collecting window: {e}")
            return None
    
    def should_make_decision(self) -> bool:
        """Check if it's time to make a decision."""
        if self.last_decision_time is None:
            return True
        
        elapsed = time.time() - self.last_decision_time
        return elapsed >= self.settings.decision_interval
    
    def make_decision(self) -> Optional[Dict[str, Any]]:
        """Run the decision pipeline and return results."""
        if len(self.events_buffer) < self.settings.min_events:
            print(f"[runner] Not enough events yet ({len(self.events_buffer)}/{self.settings.min_events})")
            return None
        
        try:
            # Get keystroke summary
            keystroke_summary = ""
            if self.keystroke_monitor:
                keystroke_summary = self.keystroke_monitor.get_summary()
            
            # Make decision
            print(f"\n[runner] Making decision on {len(self.events_buffer)} events...")
            decision = decide_with_critic(
                events_context=self.events_buffer,
                keystroke_summary=keystroke_summary,
                settings=self.settings,
                window_dwell_info=None,  # Not used anymore
                categories=self.categories,
                user_goals=self.user_goals,
            )
            
            # Log decision
            self._log_decision(decision)
            
            # Update state
            self.last_decision_time = time.time()
            
            return decision
            
        except Exception as e:
            print(f"[runner] Error making decision: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_decision(self, decision: Dict[str, Any]):
        """Log decision to console and sheets."""
        # Extract results
        category_name = decision.get("final_category_name", "Unknown")
        is_productive = decision.get("final_is_productive", None)
        conf = decision.get("final_conf", 0.0)
        reason = decision.get("final_reason", "")
        critic_ran = decision.get("critic_ran", False)
        
        # Console output
        productive_emoji = "✓" if is_productive else "✗"
        critic_emoji = "🔍" if critic_ran else "  "
        
        print(f"[runner] {critic_emoji} Decision: {productive_emoji} {category_name} (conf={conf:.2f})")
        print(f"[runner]   Reason: {reason}")
        
        # Log to Google Sheets
        try:
            self._write_to_sheets(decision)
            print(f"[runner] ✓ Logged to Google Sheets")
        except Exception as e:
            print(f"[runner] ✗ Failed to log to sheets: {e}")
    
    def _write_to_sheets(self, decision: Dict[str, Any]):
        """Write decision to Google Sheets with new schema."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract keystroke events
        keystroke_events = []
        if self.keystroke_monitor:
            keystroke_events = self.keystroke_monitor.get_recent_events()
        
        # Build the row data
        row_data = {
            "timestamp": timestamp,
            "category_id": decision.get("final_category_id", ""),
            "category_name": decision.get("final_category_name", ""),
            "is_productive": decision.get("final_is_productive"),
            
            # Primary model
            "primary_label": decision.get("primary", {}).get("category_name", ""),
            "primary_confidence": decision.get("primary", {}).get("conf", 0.0),
            "primary_reason": decision.get("primary", {}).get("reason", ""),
            
            # Critic model
            "critic_label": decision.get("critic", {}).get("category_name", "") if decision.get("critic") else "",
            "critic_confidence": decision.get("critic", {}).get("conf", 0.0) if decision.get("critic") else 0.0,
            "critic_reason": decision.get("critic", {}).get("reason", "") if decision.get("critic") else "",
            
            # Vision model
            "vision_label": decision.get("vision", {}).get("category_name", "") if decision.get("vision") else "",
            "vision_confidence": decision.get("vision", {}).get("conf", 0.0) if decision.get("vision") else 0.0,
            "vision_reason": "",  # Vision doesn't provide reason
            
            # Human label (empty for now)
            "human_label": "",
            "human_reason": "",
            
            # Keystroke data
            "typed_text": "",  # We don't log actual text for privacy
            "keystroke_events": keystroke_events,
        }
        
        self.sheets_sink.append_row(row_data)
    
    def show_alert(self, decision: Dict[str, Any]):
        """Show popup alert if user is off-task."""
        is_productive = decision.get("final_is_productive")
        
        if is_productive is False and self.settings.popup.enabled:
            category_name = decision.get("final_category_name", "Unknown")
            reason = decision.get("final_reason", "")
            
            message = f"Off-task: {category_name}"
            if reason:
                message += f"\n{reason}"
            
            show_popup(
                message,
                duration=self.settings.popup.duration,
                sound_enabled=self.settings.popup.sound_enabled,
            )
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[runner] Shutting down...")
        
        if self.keystroke_monitor:
            self.keystroke_monitor.stop()
            print("[runner] ✓ Keystroke monitor stopped")
        
        print("[runner] ✓ Shutdown complete")
    
    def run(self):
        """Main monitoring loop."""
        print("\n[runner] Starting monitoring loop...")
        print(f"[runner] Poll interval: {self.settings.poll_interval}s")
        print(f"[runner] Decision interval: {self.settings.decision_interval}s")
        print(f"[runner] Context window: {self.settings.context_window_size} events")
        print("[runner] Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Collect window event
                window_title = self.collect_window_event()
                
                # Check if we should make a decision
                if self.should_make_decision():
                    decision = self.make_decision()
                    
                    if decision:
                        # Show alert if off-task
                        self.show_alert(decision)
                        
                        # Clear keystroke buffer after decision
                        if self.keystroke_monitor:
                            self.keystroke_monitor.clear()
                
                # Sleep until next poll
                time.sleep(self.settings.poll_interval)
                
        except KeyboardInterrupt:
            print("\n[runner] Interrupted by user")
        except Exception as e:
            print(f"\n[runner] Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    """Entry point."""
    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        print(f"[runner] Failed to load settings: {e}")
        sys.exit(1)
    
    # Create and run
    runner = ProductivityRunner(settings)
    runner.initialize()
    runner.run()


if __name__ == "__main__":
    main()