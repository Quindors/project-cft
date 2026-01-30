#!/usr/bin/env python3
"""
test_with_real_data.py

Test the simple 30-second window with your actual win-log.py data.
Shows the last 5 minutes of activity analyzed with 30-second windows.
"""

import os
import json
from datetime import datetime, timedelta
from sliding_window_simple import SimpleWindowAnalyzer


def load_events_from_log(date_str: str = None):
    """Load events from win-log.py output."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    log_path = os.path.join("logs", f"windows_{date_str}.jsonl")
    
    if not os.path.exists(log_path):
        print(f"❌ No log file found: {log_path}")
        print("\nMake sure win-log.py is running!")
        print("Start it with: python win-log.py")
        return []
    
    events = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            timestamp = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
            key = record["key"]
            events.append((timestamp, key))
    
    return events


def analyze_recent_activity():
    """
    Analyze the most recent activity with 30-second windows.
    Shows how assessments change over time.
    """
    print("="*70)
    print("TESTING WITH YOUR REAL DATA")
    print("="*70)
    
    # Load events
    print("\nLoading events from logs/...")
    events = load_events_from_log()
    
    if not events:
        return
    
    print(f"✓ Loaded {len(events)} events")
    
    # Use actual current time as "now"
    now = datetime.now()
    
    # Show time range
    if events:
        first = events[0][0]
        last = events[-1][0]
        duration = (last - first).total_seconds() / 60
        now = datetime.now()
        time_since_last = (now - last).total_seconds()
        print(f"  Time range: {first.strftime('%H:%M:%S')} to {last.strftime('%H:%M:%S')} ({duration:.1f} minutes)")
        print(f"  Time since last event: {time_since_last:.1f} seconds ago")
        if time_since_last > 60:
            print(f"  ⚠️  Note: Been {time_since_last/60:.1f} minutes since last event - you may still be on same window")
    
    # Create analyzer and load all events
    analyzer = SimpleWindowAnalyzer()
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    # Analyze the last 5 minutes with 30-second snapshots
    print("\n" + "="*70)
    print("ANALYSIS: Last 5 Minutes (30-second windows)")
    print("="*70)
    
    if not events:
        print("No events to analyze")
        return
    
    # Take snapshots every minute for the last 5 minutes
    snapshots = []
    for minutes_ago in range(5, -1, -1):
        snapshot_time = now - timedelta(minutes=minutes_ago)
        
        # Only analyze if we have events around this time
        recent = analyzer.get_recent_events(snapshot_time)
        if recent:
            snapshots.append(snapshot_time)
    
    if not snapshots:
        print("Not enough recent activity to analyze")
        return
    
    # Analyze each snapshot
    for i, snapshot_time in enumerate(snapshots):
        time_str = snapshot_time.strftime("%H:%M:%S")
        minutes_ago = int((now - snapshot_time).total_seconds() / 60)
        
        summary = analyzer.get_summary(snapshot_time)
        assessment = analyzer.assess_focus(snapshot_time)
        keywords = analyzer.check_distracting_keywords(snapshot_time)
        
        print(f"\n[{time_str}] ({minutes_ago} min ago)")
        print(f"  Current window: {summary['current_window']}")
        print(f"  Events: {summary['count']}, Switches: {summary['switches']}, "
              f"Dwell: {summary['dwell_time']:.1f}s")
        
        if keywords:
            print(f"  ⚠️  Distracting: {list(keywords.keys())}")
        
        # Show assessment with emoji
        if assessment['assessment'] == 'ON-TASK':
            emoji = "✅"
        elif assessment['assessment'] == 'OFF-TASK':
            emoji = "❌"
        else:
            emoji = "❓"
        
        print(f"  {emoji} {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
        print(f"     Reason: {assessment['reason']}")
    
    # Current state
    print("\n" + "="*70)
    print("CURRENT STATE (right now)")
    print("="*70)
    
    # Show actual current time vs last event time
    last_event_time = events[-1][0]
    time_on_current = (now - last_event_time).total_seconds()
    
    print(f"\nCurrent time: {now.strftime('%H:%M:%S')}")
    print(f"Last event logged: {last_event_time.strftime('%H:%M:%S')} ({time_on_current:.1f}s ago)")
    print()
    
    print(analyzer.format_for_display())
    assessment = analyzer.assess_focus()
    
    print(f"\n{'='*70}")
    if assessment['assessment'] == 'ON-TASK':
        print("✅ CURRENTLY ON-TASK")
    elif assessment['assessment'] == 'OFF-TASK':
        print("❌ CURRENTLY OFF-TASK")
    else:
        print("❓ CURRENTLY UNCERTAIN")
    
    print(f"Confidence: {assessment['confidence']:.0%}")
    print(f"Reason: {assessment['reason']}")
    print(f"{'='*70}")
    
    # Summary stats
    print("\n" + "-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70)
    
    # Count assessments over last 5 minutes
    on_task_count = 0
    off_task_count = 0
    uncertain_count = 0
    
    for snapshot_time in snapshots:
        assessment = analyzer.assess_focus(snapshot_time)
        if assessment['assessment'] == 'ON-TASK':
            on_task_count += 1
        elif assessment['assessment'] == 'OFF-TASK':
            off_task_count += 1
        else:
            uncertain_count += 1
    
    total = len(snapshots)
    print(f"Last {total} snapshots:")
    print(f"  ✅ ON-TASK:   {on_task_count:2d} ({on_task_count/total*100:5.1f}%)")
    print(f"  ❌ OFF-TASK:  {off_task_count:2d} ({off_task_count/total*100:5.1f}%)")
    print(f"  ❓ UNCERTAIN: {uncertain_count:2d} ({uncertain_count/total*100:5.1f}%)")


def show_configuration():
    """Show current configuration."""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"Window size: {SimpleWindowAnalyzer.WINDOW_SEC} seconds")
    print("\nTo adjust, edit sliding_window_simple.py:")
    print("  - Change WINDOW_SEC (line 18) for different window size")
    print("  - Adjust dwell time thresholds in assess_focus() method")
    print("  - Modify distracting keywords list")


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("SIMPLE SLIDING WINDOW - REAL DATA TEST")
    print("="*70)
    print("\nThis will analyze your actual win-log.py data using")
    print("30-second sliding windows to show focus patterns.")
    print()
    
    analyze_recent_activity()
    show_configuration()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Does the 30-second window feel right?")
    print("  2. Are the assessments accurate?")
    print("  3. Ready to integrate into your monitor?")
    print("\nSee STEP1_SINGLE_WINDOW.md for integration instructions.")


if __name__ == "__main__":
    main()