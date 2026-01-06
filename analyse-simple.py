#!/usr/bin/env python3
"""
analyze_simple.py

Analyze factor performance and recommend optimal threshold.

Usage:
    python analyze_simple.py tests/results/test_results_TIMESTAMP.jsonl
"""

import json
import sys


def load_results(filepath):
    """Load test results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_overrides(results):
    """Analyze when factors overrode LLM."""
    
    stats = {
        "total_decisions": 0,
        "factor_overrides": 0,
        "llm_decisions": 0,
        "override_details": [],
        "aggregate_scores": [],
    }
    
    for result in results:
        stats["total_decisions"] += 1
        
        reason = result.get("reason", "")
        factors = result.get("factors", {})
        
        # Compute aggregate for all decisions
        if factors:
            weights = {
                "window_relevance": 0.30,
                "dwell_time": 0.20,
                "keystroke_activity": 0.20,
                "trajectory": 0.20,
                "risky_keywords": 0.10,
            }
            agg_score = sum(factors.get(k, 0) * w for k, w in weights.items())
            stats["aggregate_scores"].append(agg_score)
        else:
            agg_score = 0.0
        
        # Check if factors overrode
        if "[Factors decisive]" in reason:
            stats["factor_overrides"] += 1
            
            stats["override_details"].append({
                "timestamp": result.get("timestamp"),
                "label": result.get("label"),
                "aggregate_score": agg_score,
                "reason": reason,
                "events": result.get("events", [])[:3],
            })
        else:
            stats["llm_decisions"] += 1
    
    return stats


def print_report(stats, results):
    """Print analysis report."""
    
    print("\n" + "="*70)
    print("FACTOR OVERRIDE ANALYSIS")
    print("="*70)
    
    total = stats["total_decisions"]
    overrides = stats["factor_overrides"]
    llm = stats["llm_decisions"]
    
    print(f"\nTotal decisions: {total}")
    print(f"  LLM decisions:    {llm:3d} ({llm/total*100:.1f}%)")
    print(f"  Factor overrides: {overrides:3d} ({overrides/total*100:.1f}%)")
    
    # Show aggregate score distribution
    if stats["aggregate_scores"]:
        scores = stats["aggregate_scores"]
        avg = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        print(f"\nAggregate Score Statistics:")
        print(f"  Average:  {avg:+.2f}")
        print(f"  Maximum:  {max_score:+.2f}")
        print(f"  Minimum:  {min_score:+.2f}")
        print(f"  Range:    {min_score:+.2f} to {max_score:+.2f}")
        
        # Count how close we got to threshold
        close_to_override = sum(1 for s in scores if abs(s) > 0.6)
        print(f"\nDecisions with |score| > 0.6: {close_to_override}/{len(scores)}")
        print(f"Decisions with |score| > 0.7: {overrides}/{len(scores)} (current threshold)")
    
    if stats["override_details"]:
        print("\n" + "-"*70)
        print("OVERRIDE DETAILS")
        print("-"*70)
        
        for i, detail in enumerate(stats["override_details"][:10], 1):
            print(f"\n{i}. {detail['timestamp']} → {detail['label']}")
            print(f"   Aggregate score: {detail['aggregate_score']:+.2f}")
            print(f"   Reason: {detail['reason'][:100]}...")
            events_str = [e[:40] for e in detail['events']]
            print(f"   Recent windows: {events_str}")
        
        if len(stats["override_details"]) > 10:
            print(f"\n   ... and {len(stats['override_details']) - 10} more")
    else:
        print("\n" + "-"*70)
        print("NO OVERRIDES DETECTED")
        print("-"*70)
        print("\nFactors never reached the decisive threshold (|score| > 0.7)")
        print("All decisions were made by the LLM.")
        
        # Show how close we got
        print("\nDecisions by aggregate score strength:")
        print("-" * 40)
        for result in results[:10]:
            factors = result.get("factors", {})
            if factors:
                weights = {
                    "window_relevance": 0.30,
                    "dwell_time": 0.20,
                    "keystroke_activity": 0.20,
                    "trajectory": 0.20,
                    "risky_keywords": 0.10,
                }
                agg_score = sum(factors.get(k, 0) * w for k, w in weights.items())
                ts = result.get("timestamp", "")
                events = result.get("events", [])
                last_event = events[-1][:50] if events else "N/A"
                print(f"{ts}  score={agg_score:+.2f}  {last_event}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more")
    
    print("\n" + "="*70)
    
    # Recommendations
    override_rate = overrides / total if total > 0 else 0
    
    print("\nRECOMMENDATIONS:")
    
    if override_rate == 0:
        if stats["aggregate_scores"]:
            max_abs = max(abs(s) for s in stats["aggregate_scores"])
            print(f"  • No overrides occurred. Maximum |score| was {max_abs:.2f}")
            if max_abs > 0.5:
                print(f"  • Factors got close! Try lowering DECISIVE_THRESHOLD to 0.6")
                print(f"    (in decide_hybrid_simple() function)")
            else:
                print(f"  • Factors were weak. Consider:")
                print(f"    - Running on more diverse trace data")
                print(f"    - Adjusting factor weights")
    elif override_rate < 0.05:
        print("  • Very few overrides (<5%). Factors rarely override LLM.")
        print("  • If you want more factor influence, decrease DECISIVE_THRESHOLD to 0.6")
    elif override_rate < 0.15:
        print("  • Moderate override rate (5-15%). Good balance.")
        print("  • Review the override details above to verify they make sense.")
    elif override_rate < 0.30:
        print("  • High override rate (15-30%). Factors frequently override LLM.")
        print("  • If too aggressive, increase DECISIVE_THRESHOLD to 0.8")
    else:
        print("  • Very high override rate (>30%). Factors dominate decisions.")
        print("  • Consider increasing DECISIVE_THRESHOLD or checking factor computation.")
    
    print()


def analyze_threshold_tuning(results):
    """Analyze how changing threshold would affect override rate."""
    
    if not results:
        return
    
    print("\n" + "="*70)
    print("THRESHOLD TUNING SIMULATION")
    print("="*70)
    
    # Calculate aggregates for all results
    aggregates = []
    for result in results:
        factors = result.get("factors", {})
        if factors:
            weights = {
                "window_relevance": 0.30,
                "dwell_time": 0.20,
                "keystroke_activity": 0.20,
                "trajectory": 0.20,
                "risky_keywords": 0.10,
            }
            agg_score = sum(factors.get(k, 0) * w for k, w in weights.items())
            aggregates.append(abs(agg_score))
    
    if not aggregates:
        return
    
    print(f"\nSimulating different DECISIVE_THRESHOLD values:")
    print(f"{'Threshold':<12} {'Would Override':<15} {'Override %':<12}")
    print("-" * 40)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        would_override = sum(1 for a in aggregates if a > threshold)
        pct = would_override / len(aggregates) * 100 if aggregates else 0
        marker = " ← current" if threshold == 0.7 else ""
        print(f"{threshold:<12.2f} {would_override}/{len(aggregates):<13} {pct:>6.1f}%{marker}")
    
    print(f"\nRecommended threshold based on your data:")
    
    max_agg = max(aggregates)
    if max_agg > 0.7:
        print(f"  ✓ Current threshold (0.7) is working - factors are overriding")
    elif max_agg > 0.65:
        print(f"  → Lower to 0.6 - you're very close (max score: {max_agg:.2f})")
        print(f"     This would enable {sum(1 for a in aggregates if a > 0.6)} overrides")
    elif max_agg > 0.55:
        print(f"  → Lower to 0.55 - moderate signals (max score: {max_agg:.2f})")
    else:
        print(f"  ⚠ Factors are weak (max: {max_agg:.2f})")
        print(f"     Consider using pure LLM mode or check factor computation")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_simple.py <results_file.jsonl>")
        print("\nExample:")
        print("  python analyze_simple.py tests/results/test_results_20260105_185550.jsonl")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"Loading results from {filepath}...")
    results = load_results(filepath)
    print(f"Loaded {len(results)} decisions")
    
    stats = analyze_overrides(results)
    print_report(stats, results)
    analyze_threshold_tuning(results)


if __name__ == "__main__":
    main()