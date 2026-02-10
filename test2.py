#!/usr/bin/env python3
"""
Test for Phase 3, Step 2 - show what the critic prompt looks like when filled in
"""

# Example categories
example_categories = [
    {"name": "Deep Work", "description": "VS Code, coding, focused tasks", "is_productive": True},
    {"name": "Homework", "description": "OneNote, calc problems, textbook PDFs", "is_productive": True},
    {"name": "Research & Learning", "description": "Khan Academy, Wolfram, educational content", "is_productive": True},
    {"name": "Social Media", "description": "Discord chatting, Twitter, Reddit", "is_productive": False},
    {"name": "Entertainment", "description": "YouTube entertainment, Netflix, funny videos", "is_productive": False},
]

# Example user goals
example_goals = """I'm a high school student working on:
- Calculus homework (using OneNote, Wolfram Alpha, Khan Academy)
- Personal coding project "cft" 
- Learning ML and AI on the side"""

# Example activity - looks like research but might be a rabbit hole
example_activity = "WIN:L'Hôpital's rule - Wikipedia, WIN:Riemann zeta function - Wikipedia, WIN:Millennium Prize Problems - Wikipedia"

# Example initial decision
initial_decision = "CATEGORY=Research & Learning | OFF_SCORE=0.3 | CONF=0.7 | REASON=Educational Wikipedia articles about math topics"

# Build the prompt
prompt_template = open("prompt-critic-new.txt", "r").read()

# Fill in the placeholders
category_list = "\n".join([
    f"- \"{cat['name']}\": {cat['description']} ({'PRODUCTIVE' if cat['is_productive'] else 'NOT PRODUCTIVE'})"
    for cat in example_categories
])

filled_prompt = prompt_template.replace("[insert goals here]", example_goals)
filled_prompt = filled_prompt.replace("[insert categories here]", category_list)
filled_prompt = filled_prompt.replace("[insert here]", initial_decision)
filled_prompt = filled_prompt.replace("[insert activity here]", example_activity)

print("="*70)
print("EXAMPLE CRITIC PROMPT:")
print("="*70)
print(filled_prompt)
print("="*70)
print("\n✅ This is what the critic will see!")
print("\nExpected critic response for this example:")
print("CATEGORY=Entertainment")
print("OFF_SCORE=0.6")
print("CONF=0.8")
print("REASON=Started with relevant calc topic (L'Hôpital's rule) but drifted to")
print("       advanced math topics (Riemann zeta, Millennium Prize) not needed for")
print("       high school homework - this is a rabbit hole")