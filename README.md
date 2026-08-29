one day we will do a rigorous comparison of the models trust


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
| System | — | System windows (Snipping Tool, Task Switching, lock screen, etc.) |

### Managing Categories

Categories live in the **"Categories" sheet** in your Google Spreadsheet. To manage them, open the spreadsheet and go to the Categories tab.

**Columns:**

| Column | Required | Description |
|--------|----------|-------------|
| `name` | Yes | Display name shown in logs and popups, e.g. "Deep Work" |
| `description` | Yes | Tells the LLM what belongs in this category. Be specific — include app names, site names, and examples. |
| `is_productive` | Yes | `TRUE`, `FALSE`, or leave blank for ambiguous categories. Determines if the category counts as focused work. |

**Adding a new category:**

1. Open the Categories sheet
2. Add a new row at the bottom
3. Fill in `name`, `description`, and `is_productive`
4. Write a clear description — this is what the AI reads to decide if an activity fits

Example — adding a "Music" category for background music apps:
```
name:           Music
description:    Spotify, Apple Music, SoundCloud, background music while working
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
3. Edit the **`human_label`** column — select the correct category (auto-populates upon startup)
4. Leave the original AI columns (`category_name`, `is_productive`, etc.) unchanged — they serve as the AI's original prediction for accuracy tracking

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

All settings are in `monitor/config.py`.

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o-mini` | Primary classification model |
| `poll_interval` | `2.0` | Seconds between window event collection |
| `decision_interval` | `60.0` | Seconds between classification decisions |
| `context_window_size` | `10` | Number of recent events sent to LLM |
| `min_events` | `1` | Minimum events before first decision |
| `off_threshold` | `0.60` | OFF_SCORE threshold |
| `critic.enabled` | `True` | Run critic on ambiguous classifications |
| `critic.vision_enabled` | `True` | Use screenshot tiebreaker on disagreements |
| `popup.enabled` | `True` | Show popup alerts for non-productive categories |

## Idle detection

The ledger assumes the window in front of you is what you are doing; an empty
chair breaks that. After **5 minutes without keyboard or mouse input** the
monitor stands down — nothing is classified, nothing is logged, the widget and
tray go grey ("Away") — and the first input starts it again from a clean
slate. Without this, a productive window parked overnight logs hour after
hour of "work", which any goal fed by those hours would happily pay out.

The first idle minutes still log: below the threshold, stillness is
indistinguishable from reading. The trade-off runs the other way for long
videos — watching a lecture hands-off past the threshold reads as away, so
raise `CFT_IDLE_SEC` if that is a real part of your day (the occasional
scroll or seek already counts as input).

Alerts are gated separately and more tightly: a popup only fires if there was
input in the last 60 seconds, so the monitor never nags a chair its owner
left three minutes ago.

| Variable | Default | Description |
|----------|---------|-------------|
| `CFT_IDLE_SEC` | `300` | Seconds without input before the monitor stands down. `0` disables idle detection |
| `CFT_IDLE_WSL_PROBE` | off | WSL dev mode only: set `1` to probe the Windows host via PowerShell (cached, but heavy) |

Idle time is read from `GetLastInputInfo` on Windows and `xprintidle` on
Linux/X11. Where no probe exists the feature fails open — no signal means
"present", never a monitor that silently stops logging forever.

## Beeminder (optional)

Push your daily focus hours to a [Beeminder](https://www.beeminder.com) goal,
which charges you real money when your cumulative total ends a day on the wrong
side of the line you committed to.

This is per-user: Beeminder's account name is part of the API endpoint and the
token belongs to that account, so **every person running the app needs their own
Beeminder account, their own goals and their own token.** Nothing is shared, and
the integration stays entirely off until the `.env` names all three.

### Setup

1. Make a Beeminder account, then log in and open
   `https://www.beeminder.com/api/v1/auth_token.json` to get your personal
   token.
2. Fill in the **Connect Beeminder** card on the dashboard's Overview tab.
   "Create a goal for me" makes the Do More goal (units: hours) on your
   account with the daily commitment and starting leeway you choose — leeway
   is safety buffer, days you could do nothing before a derail. Or pick
   "I already have a goal" and give its slug (the last part of its URL:
   `beeminder.com/alice/focus` → `focus`).
3. Either way the card verifies everything against Beeminder, saves it to the
   `.env` next to the exe and starts syncing immediately — no restart. The
   card only shows while Beeminder is unconfigured (and "Not now" hides it on
   that browser).

Prefer a text editor? Add the same values to the `.env` by hand and restart:

```
BEEMINDER_USER=your-beeminder-username
BEEMINDER_TOKEN=your-personal-auth-token
BEEMINDER_FOCUS_GOAL=focus
```

Either way, the monitor pushes today's hours every five minutes, on shutdown,
and once more for yesterday after midnight rolls over.

### Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BEEMINDER_USER` | — | Your Beeminder username. Required. |
| `BEEMINDER_TOKEN` | — | Personal auth token. Required. |
| `BEEMINDER_FOCUS_GOAL` | — | Do More goal fed daily productive hours |
| `BEEMINDER_SESSIONS_GOAL` | — | Optional goal fed one point per finished focus session (needs ≥10 logged minutes) |
| `BEEMINDER_SYNC_SEC` | `300` | Seconds between pushes |
| `BEEMINDER_SECONDS_PER_ROW` | `decision_interval` | Seconds of activity each logged row stands for |
| `BEEMINDER_STRICT_INTENT` | on | Off-intent work during a focus session counts as off-task; set `0` to count any productive work |
| `BEEMINDER_DRY_RUN` | off | Print what would be sent, send nothing |
| `BEEMINDER_TIMEOUT_SEC` | `15` | Per-request timeout |

### How the numbers are worked out

Hours are inferred, not measured: each row in `focus_logs` stands for one
`decision_interval` of activity, and a row counts as productive when its
category is flagged productive. A row you have relabelled in the review tab
counts as its *human* label, so correcting the classifier changes the total —
and because every push reuses one request id per day, re-pushing simply updates
that day's datapoint in place rather than adding a second one.

During a focus session the goal measures the work you declared, not work in
general: a productive-category row logged off-intent (ALIGN below
`intention.off_intent_max`, the same line the drift alerts use) counts as
off-task. Partially-aligned work still counts, and work done with no session
active is judged by its category alone. Set `BEEMINDER_STRICT_INTENT=0` if you
want any productive-category work to count regardless of the active session.

The sessions goal has its own floor: a finished session only posts a point if
at least ten minutes of activity were actually logged under it. How a session
ended matters less than whether anyone was there for it — one clicked "Done"
moments after starting, or left to expire in front of an empty chair (idle
detection stops the logging), posts nothing.

### Before you put money on it

- **Beemind focus hours, not distraction minutes.** A Do Less goal fed by a
  monitor you can close is gameable in the wrong direction — quitting the exe
  would report zero distraction. On a Do More goal, quitting it just stops the
  credit.
- **Uptime gaps cost more than misclassification.** A wrong category costs
  minutes; the app not running costs the whole stretch. Disabling the autostart
  task for a rebuild, a sleeping machine, or a crash all silently undercount.
- **An empty chair earns nothing.** Idle detection (above) stops the hour
  count five minutes after your last input, so parking a productive window
  overnight doesn't feed the goal — and a session that expires unattended
  fails the sessions goal's ten-logged-minutes floor rather than counting
  as finished.
- **Start at a $0 pledge** and run `BEEMINDER_DRY_RUN=1` (or watch the goal
  flat) for a week or two, comparing what gets pushed against the dashboard,
  before letting a derail cost anything. Keep a week of safety buffer.
