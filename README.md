# cft — AI-Powered Productivity Monitor

A desktop productivity tracker that uses LLM-based classification to categorize your activities in real time. It monitors active window titles and keystrokes, classifies your activity into user-defined categories, and logs everything to Google Sheets.

## How It Works

1. **win-log.py** captures your active window title whenever it changes
2. **monitor-sheets.py** reads those logs, sends context to an LLM, and gets a category classification
3. Results are written to Google Sheets with a daily worksheet
4. A popup alert fires when you drift into non-productive categories

The system uses a two-pass verification approach: a primary model classifies the activity, and a critic model double-checks ambiguous or low-confidence results. If they disagree, an optional vision tiebreaker (screenshot analysis) resolves the conflict.

## Quick Start

1. Set up your `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   GCP_SERVICE_ACCOUNT_B64=<base64-encoded service account JSON>
   ```

2. Edit `user_goals.txt` to describe your current projects and goals

3. Start the window logger and monitor:
   ```
   python win-log.py
   python monitor-sheets.py
   ```
   Or use `startup_launch.vbs` to run both silently on startup.

## Category System

cft classifies your activity into categories rather than a simple on-task/off-task label. Each category has a `is_productive` flag that determines whether it counts as focused work time.

### Default Categories

| Category | Productive | Description |
|----------|------------|-------------|
| Deep Work | Yes | VS Code, coding, writing, focused tasks |
| Homework | Yes | OneNote, textbook PDFs, Desmos, assignment work |
| Research & Learning | Yes | Khan Academy, Wolfram Alpha, educational YouTube, docs |
| Communication | Yes | Teams for schoolwork, email, project collaboration |
| Social Media | No | Twitter/X, Instagram, TikTok, Reddit, Discord chat |
| Entertainment | No | YouTube entertainment, Netflix, Twitch, streams |
| Gaming | No | Steam, Minecraft, Roblox, game launchers |
| Shopping | No | Amazon, eBay, online shopping |
| Other Off-Task | No | Any other off-task activity not covered above |
| Ambiguous | — | Activity that can't be clearly categorized from available context |

### Managing Categories

Categories live in the **"Categories" sheet** in your Google Spreadsheet. To manage them, open the spreadsheet and go to the Categories tab.

**Columns:**

| Column | Required | Description |
|--------|----------|-------------|
| `category_id` | Yes | Short unique ID, e.g. `deep_work`, `gaming`. Use lowercase with underscores. |
| `name` | Yes | Display name shown in logs and popups, e.g. "Deep Work" |
| `description` | Yes | Tells the LLM what belongs in this category. Be specific — include app names, site names, and examples. |
| `color` | No | Hex color for future dashboard use, e.g. `#2ecc71` |
| `is_productive` | Yes | `TRUE`, `FALSE`, or leave blank for ambiguous categories. Determines if the category counts as focused work. |
| `created_at` | No | Auto-filled on creation |

**Adding a new category:**

1. Open the Categories sheet
2. Add a new row at the bottom
3. Fill in `category_id`, `name`, `description`, and `is_productive` at minimum
4. Write a clear description — this is what the AI reads to decide if an activity fits

Example — adding a "Music" category for background music apps:
```
category_id:    music
name:           Music
description:    Spotify, Apple Music, SoundCloud, background music while working
color:          #1DB954
is_productive:  TRUE
```

**Editing a category:**

Just edit the cell directly in Google Sheets. Changes take effect the next time the monitor starts (categories are loaded on startup).

**Tips for good descriptions:**
- List specific app names and websites that belong in the category
- Mention what the user is *doing*, not just the app name
- The LLM uses this text to match window titles, so be concrete
- Example: instead of "educational content", write "Khan Academy, Wolfram Alpha, Wikipedia for homework topics, educational YouTube tutorials"

**Removing a category:**

Delete the entire row from the Categories sheet. Any future logs will no longer use that category. Existing logs that reference it are unaffected.

## Correcting Misclassifications

The AI won't always get it right. You can correct mistakes directly in Google Sheets, and those corrections help you track accuracy over time.

### How to Recategorize

1. Open the daily **"Focus Logs - YYYY-MM-DD"** sheet
2. Find the row with the wrong classification
3. Edit these columns:
   - **`human_label`** — write the correct category name (must match a name from the Categories sheet exactly)
   - **`human_reason`** — optionally explain why, e.g. "was actually watching a coding tutorial, not entertainment"
4. Leave the original AI columns (`category_id`, `category_name`, etc.) unchanged — they serve as the AI's original prediction for accuracy tracking

### What to Look For

Common misclassifications to watch for:

- **YouTube**: The AI may confuse educational tutorials with entertainment. Check the window title.
- **Wikipedia**: Research for homework vs. rabbit-holing into unrelated topics.
- **Discord**: Homework help channels vs. general chatting.
- **Browser new tabs**: Brief transitions that get classified as off-task.

### Bulk Recategorization

To fix many rows at once in Google Sheets:
1. Sort or filter the sheet by `category_name` to find all instances of a wrong category
2. Use Sheets' Find & Replace (Ctrl+H) on the `human_label` column
3. Or use a formula to fill `human_label` based on a pattern in the event columns

## User Goals

Edit `user_goals.txt` in the project root to tell the AI what you're working on. This helps it distinguish productive from unproductive use of the same apps.

```
I'm a high school student working on:
- Calculus homework (using OneNote, Wolfram Alpha, Khan Academy, Desmos)
- Personal coding project "cft"
- Learning about ML and AI

When I use:
- VS Code: it's for the cft project
- YouTube: could be educational OR entertainment
- Discord: usually chatting, sometimes homework help
```

Update this file whenever your projects change. The AI reads it fresh each time the monitor starts.

## Project Structure

```
project cft/
├── monitor/
│   ├── __init__.py
│   ├── __main__.py        # Entry point: python -m monitor
│   ├── app.py             # Initialization (env, creds, sources, sink)
│   ├── runner.py           # Main monitoring loop
│   ├── decider.py          # LLM calls, parsing, critic/vision logic
│   ├── config.py           # All settings and defaults
│   ├── creds.py            # Environment and credential loading
│   ├── sheets_sink.py      # Google Sheets read/write
│   ├── sources.py          # Window event and keystroke sources
│   └── alerts.py           # Popup alerts
├── prompt-main.txt         # Primary LLM prompt template
├── prompt-critic.txt       # Critic LLM prompt template
├── user_goals.txt          # Your current projects and goals
├── monitor-sheets.py       # Entry point: python monitor-sheets.py
├── win-log.py              # Active window logger
├── key-log.py              # Keystroke logger
├── mouse-log.py            # Mouse activity logger
├── analysis-sheets-live.py # Live dashboard (Dash + Plotly)
├── run-monitor.bat         # Windows batch launcher
├── run-winlog.bat          # Windows batch launcher
├── startup_launch.vbs      # Silent startup launcher
├── developer notes.md      # Migration checklist and todo
└── logs/                   # Daily JSONL log files (gitignored)
```

## Configuration

All settings are in `monitor/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o-mini` | Primary classification model |
| `poll_interval` | `2.0` | Seconds between window event collection |
| `decision_interval` | `60.0` | Seconds between classification decisions |
| `context_window_size` | `30` | Number of recent events sent to LLM |
| `min_events` | `5` | Minimum events before first decision |
| `off_threshold` | `0.60` | OFF_SCORE threshold |
| `critic.enabled` | `True` | Run critic on ambiguous classifications |
| `critic.vision_enabled` | `True` | Use screenshot tiebreaker on disagreements |
| `popup.enabled` | `True` | Show popup alerts for non-productive categories |