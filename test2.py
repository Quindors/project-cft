#!/usr/bin/env python3
"""
Quick test for Phase 2, Step 2 - verify config can load user goals
"""

from monitor.config import DEFAULT_SETTINGS

goals = DEFAULT_SETTINGS.load_user_goals()

print("="*70)
print("USER GOALS LOADED:")
print("="*70)
print(goals)
print("="*70)
print("\n✅ If you see your goals above, Step 2 is working!")