#!/usr/bin/env python3
"""
test_realistic_dwell.py

Test dwell time calculation with realistic data (events only logged on window changes).
This simulates how win-log.py actually works.
"""

from datetime import datetime, timedelta
from sliding_window_simple import SimpleWindowAnalyzer


def test_long_dwell():
    """
    Scenario: User has been on VS Code for 5 minutes straight.
    win-log.py logs ONE event when they switched to VS Code, then nothing until they switch away.
    """
    print("="*70)
    print("TEST: Long Dwell Time (5 minutes on VS Code)")
    print("="*70)
    
    analyzer = SimpleWindowAnalyzer()
    now = datetime.now()
    
    # User was on Discord, then switched to VS Code 5 minutes ago
    # win-log.py only logs when window CHANGES, so we only have 2 events total:
    events = [
        (now - timedelta(minutes=6), "WIN:Discord - #general"),  # Was on Discord
        (now - timedelta(minutes=5), "WIN:VS Code - main.py"),    # Switched to VS Code 5 min ago
        # Nothing else logged because user stayed on VS Code!
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    # Check summary NOW (5 minutes after switching to VS Code)
    summary = analyzer.get_summary(now)
    
    print(f"\nEvents in last 30 seconds: {summary['count']}")
    print(f"Current window: {summary['current_window']}")
    print(f"Dwell time: {summary['dwell_time']:.1f} seconds ({summary['dwell_time']/60:.1f} minutes)")
    
    expected_dwell = 5 * 60  # 5 minutes
    actual_dwell = summary['dwell_time']
    
    if abs(actual_dwell - expected_dwell) < 5:  # Within 5 seconds is good enough
        print(f"\n✅ CORRECT: Dwell time is ~{actual_dwell/60:.1f} minutes")
        print(f"   (Expected ~{expected_dwell/60:.1f} minutes)")
    else:
        print(f"\n❌ WRONG: Dwell time is {actual_dwell/60:.1f} minutes")
        print(f"   (Expected ~{expected_dwell/60:.1f} minutes)")
    
    assessment = analyzer.assess_focus(now)
    print(f"\nAssessment: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")


def test_recent_switch():
    """
    Scenario: User switched from YouTube to VS Code 10 seconds ago.
    """
    print("\n" + "="*70)
    print("TEST: Recent Switch (10 seconds on VS Code)")
    print("="*70)
    
    analyzer = SimpleWindowAnalyzer()
    now = datetime.now()
    
    events = [
        (now - timedelta(minutes=5), "WIN:Chrome - YouTube"),  # Was watching YouTube
        (now - timedelta(seconds=10), "WIN:VS Code - main.py"), # Switched 10s ago
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    summary = analyzer.get_summary(now)
    
    print(f"\nEvents in last 30 seconds: {summary['count']}")
    print(f"Current window: {summary['current_window']}")
    print(f"Dwell time: {summary['dwell_time']:.1f} seconds")
    
    if 9 <= summary['dwell_time'] <= 11:  # Should be ~10 seconds
        print(f"\n✅ CORRECT: Dwell time is ~10 seconds")
    else:
        print(f"\n❌ WRONG: Dwell time is {summary['dwell_time']:.1f} seconds (expected ~10)")
    
    assessment = analyzer.assess_focus(now)
    print(f"\nAssessment: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")


def test_multiple_switches():
    """
    Scenario: User has been switching between apps frequently.
    """
    print("\n" + "="*70)
    print("TEST: Multiple Switches in Last 2 Minutes")
    print("="*70)
    
    analyzer = SimpleWindowAnalyzer()
    now = datetime.now()
    
    # User switching between apps
    events = [
        (now - timedelta(minutes=2), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=90), "WIN:Chrome - YouTube"),
        (now - timedelta(seconds=60), "WIN:Discord - #general"),
        (now - timedelta(seconds=40), "WIN:VS Code - main.py"),
        (now - timedelta(seconds=20), "WIN:Chrome - Twitter"),
        (now - timedelta(seconds=5), "WIN:VS Code - main.py"),
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    summary = analyzer.get_summary(now)
    
    print(f"\nEvents in last 30 seconds: {summary['count']}")
    print(f"Window switches in last 30 seconds: {summary['switches']}")
    print(f"Current window: {summary['current_window']}")
    print(f"Dwell time on current: {summary['dwell_time']:.1f} seconds")
    
    if 4 <= summary['dwell_time'] <= 6:  # Should be ~5 seconds
        print(f"\n✅ CORRECT: Dwell time is ~5 seconds")
    else:
        print(f"\n❌ WRONG: Dwell time is {summary['dwell_time']:.1f} seconds (expected ~5)")
    
    assessment = analyzer.assess_focus(now)
    print(f"\nAssessment: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")


def test_passive_viewing():
    """
    Scenario: User has been on YouTube for 10 minutes (passive viewing).
    This should be detected as OFF-TASK due to long dwell time on distracting app.
    """
    print("\n" + "="*70)
    print("TEST: Passive Viewing (10 minutes on YouTube)")
    print("="*70)
    
    analyzer = SimpleWindowAnalyzer()
    now = datetime.now()
    
    events = [
        (now - timedelta(minutes=11), "WIN:VS Code - homework.py"),
        (now - timedelta(minutes=10), "WIN:Chrome - YouTube - Funny Cats"),
        # Still on YouTube for 10 minutes straight
    ]
    
    for timestamp, key in events:
        analyzer.add_event(timestamp, key)
    
    summary = analyzer.get_summary(now)
    
    print(f"\nCurrent window: {summary['current_window']}")
    print(f"Dwell time: {summary['dwell_time']:.1f} seconds ({summary['dwell_time']/60:.1f} minutes)")
    print(f"Keywords detected: {analyzer.check_distracting_keywords(now)}")
    
    assessment = analyzer.assess_focus(now)
    print(f"\nAssessment: {assessment['assessment']} (confidence: {assessment['confidence']:.2f})")
    print(f"Reason: {assessment['reason']}")
    
    if assessment['assessment'] == 'OFF-TASK' or assessment['assessment'] == 'UNCERTAIN':
        print(f"\n✅ CORRECT: Detected as {assessment['assessment']}")
        print(f"   (Long dwell on YouTube suggests passive viewing)")
    else:
        print(f"\n⚠️  Assessment is ON-TASK - should probably be OFF-TASK or UNCERTAIN")


def main():
    print("\n" + "="*70)
    print("REALISTIC DWELL TIME TESTS")
    print("="*70)
    print("\nThese tests simulate how win-log.py actually works:")
    print("- Events are only logged when the window CHANGES")
    print("- If you stay on one app, there's only one event logged")
    print("- Dwell time should extend beyond the 30-second window")
    print()
    
    test_long_dwell()
    test_recent_switch()
    test_multiple_switches()
    test_passive_viewing()
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
    print("\nIf all tests show ✅ CORRECT, then dwell time is working properly!")
    print("Now try with your real data: python test_with_real_data.py")


if __name__ == "__main__":
    main()