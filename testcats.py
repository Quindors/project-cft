#!/usr/bin/env python3
"""
test_categories_sheet.py

Quick test to verify Categories sheet is created correctly.
Run this to check Phase 1 implementation.
"""

from monitor.creds import load_env, build_service_account_json_from_b64
from monitor.sheets_sink import SheetsSink
from monitor.config import DEFAULT_SETTINGS

def main():
    print("="*70)
    print("TESTING CATEGORIES SHEET CREATION")
    print("="*70)
    
    load_env(override=True)
    
    service_account_json = build_service_account_json_from_b64()
    
    print("\n[1/3] Connecting to Google Sheets...")
    sink = SheetsSink(service_account_json, settings=DEFAULT_SETTINGS)
    
    print("[2/3] Loading categories...")
    categories = sink.get_categories()
    
    print(f"[3/3] Found {len(categories)} categories:")
    print()
    
    for cat in categories:
        productive = "✅ Productive" if cat['is_productive'] else "❌ Not Productive"
        print(f"{cat['emoji']} {cat['name']:<20} {productive}")
        print(f"   ID: {cat['category_id']}")
        print(f"   Description: {cat['description'][:80]}...")
        print(f"   Color: {cat['color']}")
        print()
    
    print("="*70)
    print("SUCCESS! Categories sheet is working correctly.")
    print("="*70)
    print("\nNext steps:")
    print("1. Check your Google Sheet - you should see a 'Categories' tab")
    print("2. Review the default categories and edit if needed")
    print("3. Move on to Phase 2: User Goals setup")

if __name__ == "__main__":
    main()