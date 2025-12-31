# Todo
## fine tuning
- ~~add abstaining which triggers the critic~~ ✓
(next session)
- Sliding Window Sizes:
    - Micro-context (last 30 sec): Immediate activity
    - Meso-context (last 3 min): Current task
    - Macro-context (last 15 min): Session trajectory
- Instead of binary ON/OFF, use multi-factor scoring:
    - Window relevance:      0.7  (math-related)
    - Dwell time:            -0.3 (too long without returning)
    - Keystroke activity:    -0.4 (minimal typing)
    - Trajectory:            -0.5 (drifting away from homework)
    - Historical pattern:    -0.2 (user has 70% accuracy on similar)
    - AGGREGATE SCORE: -0.7 → OFF-TASK (confidence: 0.85)
- prompt mutation strategy:
    - Don't just append rules (prompt bloat)
    - Identify conflicting patterns and rewrite sections
    - Example: If system thinks "low typing = focused reading" but mistakes show "low typing = passive consumption", update that heuristic

## devops
- start hosting it
- run script automatically

## analytics
- check dashboard accuracy
- weekly report

## bugs/improvements
- have the popup actually pop up on the front of the screen
- have the popup auto-close after 3 seconds
- increase startup speed of monitor-sheets.py

## 12/15
Abstain as a first-class outcome: reliability increases massively when the system is allowed to say “UNKNOWN / NEEDS REVIEW” instead of guessing.
3) Two-pass verification (LLM judge + LLM auditor)
If you don’t care about cost, this is one of the most reliable patterns:
Pass A: classify + provide the evidence it used.
Pass B: independently critique: “Is the label justified by the evidence? What would flip it?”
If disagreement or weak evidence → UNKNOWN / human review.
- check if we are calling LLM every window sample, every 3 sec, or every change
- better model maybe
- remove google sheets duplicate params
3) Add “time in current context” as a first-class signal
LLMs get much more consistent if you include:
“current window has been active for X seconds"
“last change was Y seconds ago”
“in last 30 seconds: app switches = N”
Real off-task usually has dwell time. That one feature dramatically improves reliability.

## 12/1
- we will be hosting the dashboard

## 11/24
startup manager 

## 11/17
for next time:
- start hosting it
- expand inputs
    - consider screenshots?
- run script automatically
- have the popup actually pop up on the front of the screen
- have the popup auto-close after 3 seconds
- increase startup speed of monitor-sheets.py
- add ability of self-correction (fine tuning loop)

## 11/10
- should we add a "reason" next to the "human label"
- recall precision and f1

## 10/27
- logging confidence level
to do:
- separate tab on Google Sheets for GPT's errors & corrections for incorrectly labelled logs
- the tab will rank by confidence level, GPT takes in by confidence (priority: higher confidence)
- modify monitor.py to write directly to sheets instead of jsonl
- improve smoothness of popup (make sure it pops up instead of just alert sound)





# Process

## Part 1: Too Fast
- started by writing `master.py`
- saved keystrokes to `keystrokes.txt`, windows to `window_log.csv` and screenshots to `screenshots.json`
- ended up not working well for analysis

## Part 2: Just Keystrokes
- wrote an analysis program (deleted) for just `keystrokes.txt`
- had ChatGPT generate dummy data, analysed with `analysis_dummy.py`
- wrote full `keystroke_logger.py` to save keystrokes in daily `.json` files
- wrote `monitor.py` to analyse keystrokes

## Part 3: Early Automation
- tried using Windows Task Scheduler
- made `run_logger.cmd` and `run_monitor.cmd` for that purpose
- ultimately too inconsistent with triggers
- settled on manual initiation with a looping `monitor.py`


# Keystroke Analysis Project Roadmap ✅

## Phase 1 — Keystroke capture
- [ ] Print keystrokes to stdout
- [ ] Write to `logs/keystrokes_RAW.log`
- [ ] Rotate daily → `data/keystrokes_YYYY-MM-DD.jsonl`
- [ ] Buffer + flush every 1s
- [ ] Heartbeat → `logs/collector_health.log`

## Phase 2 — Data windows
- [ ] Slice keystrokes into time windows
- [ ] Save each window with timestamp + ID
- [ ] Attach dummy on/off-task labels
- [ ] Store in JSONL format

## Phase 3 — Model analysis
- [ ] Build prompt with raw keystrokes only
- [ ] Send windows to GPT-5-nano
- [ ] Collect model outputs
- [ ] Compare predictions to gold labels
- [ ] Print quick accuracy summary

## Phase 4 — Background service
- [ ] Run logger as background process on Windows
- [ ] One JSONL file per day
- [ ] Auto-start with system boot (optional)

## Phase 5 — Evaluation
- [ ] Run analyzer on daily JSONL
- [ ] Summarize accuracy across windows
- [ ] Inspect sample outputs manually
- [ ] Adjust window size / buffer settings

## Phase 6 — Extensions (optional)
- [ ] Add real labels from user activity
- [ ] Add simple dashboard or viewer
- [ ] Explore fine-tuning or RAG pipeline
- [ ] Optimize for speed/cost




Perfect—focusing on \*\*keystrokes only\*\*. Here’s a tight plan to get useful on/off-task signals without window titles or app context.



\# 1) What “moment” means



\* Use rolling windows: e.g., \*\*10s\*\*, \*\*30s\*\*, \*\*60s\*\* (pick one as primary; keep the others as context).

\* Score each window independently, then \*\*smooth\*\* with short hysteresis (e.g., require two consecutive “off” windows to flip the state).



\# 2) Feature set (from keystrokes only)



\*\*Volume \& cadence\*\*



\* Keystrokes per minute (KPM), per window.

\* Burstiness: coefficient of variation of inter-key intervals; number/length of bursts.

\* Pauses: fraction of time idle; longest pause; count of micro-pauses (e.g., >1s, >3s).



\*\*Key taxonomy mix\*\*



\* Character vs. control vs. navigation ratios.

\* Edit intensity: backspace/delete rate; backspace-to-character ratio.

\* Structure keys: enter/tab frequency (lines/fields advanced).

\* Modifier usage: proportion of keys involving Ctrl/Alt/Meta; sequences like Ctrl+C/V, Ctrl+S.



\*\*Text-likeness\*\*



\* Printable character ratio.

\* Alphanumeric vs. symbol share; digit density (numbers often indicate data entry/calc).

\* Shannon entropy of token stream in the window (very low entropy may be spam/auto-repeat).



\*\*Sequence signatures\*\*



\* N-grams of special keys (e.g., BACKSPACE×n, ENTER ENTER, TAB NAV).

\* Regex-like motifs: “type–backspace–retype”, “type–enter”, “paste-like spike” (very high KPM + few character tokens).



\*\*Stability\*\*



\* Drift features comparing current to previous window (delta KPM, delta backspace rate).



\# 3) Label space \& outputs



\* \*\*Labels:\*\* `on\_task`, `off\_task`, `uncertain`.

\* \*\*Confidence:\*\* 0–1; map to certainty bands (e.g., ≥0.8 high, 0.6–0.8 medium).

\* \*\*Rationale:\*\* short, human-readable reason (e.g., “sustained typing with low corrections”).



\# 4) Lightweight rules (fast, obvious cases)



\* \*\*Likely on-task:\*\* KPM above threshold (e.g., >90 in 30s window), normal backspace ratio (≤0.25), consistent bursts, some ENTER/TAB.

\* \*\*Likely off-task:\*\* near-idle (KPM \\~0–5), high NAV keys without characters, long pause (>80% idle).

\* \*\*Escalate to LLM\*\* only for ambiguous windows (e.g., moderate KPM with high backspace; symbol-heavy typing; sporadic bursts).



\# 5) GPT-5 Nano prompt design (features-in, judgment-out)



Provide a compact, \*\*model-friendly record\*\* for one window plus the previous window. Keep it deterministic and privacy-preserving (no raw text). Example structure you’ll send (conceptually, not code):



\*\*System (policy):\*\*

“You classify user focus based solely on keystroke statistics. Do not assume applications. Output JSON with fields: label ∈ {on\\\_task, off\\\_task, uncertain}, confidence∈\\\[0,1], reason.”



\*\*User (window summary):\*\*



\* window\\\_secs: 30

\* kpm: 112

\* idle\\\_fraction: 0.12

\* burstiness\\\_cv: 0.65

\* char\\\_ratio: 0.78

\* backspace\\\_rate: 0.18

\* enter\\\_rate: 0.06

\* tab\\\_rate: 0.02

\* nav\\\_rate: 0.03

\* modifier\\\_rate: 0.04

\* digit\\\_ratio: 0.10

\* entropy\\\_bits\\\_per\\\_key: 3.2

\* special\\\_ngrams: {“BACKSPACE×3”:2, “ENTER→ENTER”:0, “CTRL+C”:0, “CTRL+V”:0}

\* deltas\\\_vs\\\_prev: {kpm:+35, idle:-0.20, backspace:+0.05}



\*\*Assistant (expected output):\*\*

`{"label":"on\_task","confidence":0.86,"reason":"Sustained typing, moderate edits, structured keys present, low idle."}`



Tips:



\* Use \*\*consistent units\*\* and names; avoid raw lists of keys (privacy + token budget).

\* Clamp outliers and round to 2–3 decimals to keep prompts tiny.



\# 6) Training \& calibration (without window titles)



\*\*Labeling strategy\*\*



\* Create \*\*task-conditioned sessions\*\* (typing text, coding exercises, form-filling, note-taking) and \*\*non-task sessions\*\* (idle, random navigation without typing, key-spam).

\* Ask participants to mark “on-task” periods; or script tasks with timers to auto-label.

\* Include \*\*reading tasks\*\* with low typing but occasional ENTER/TAB (to avoid bias toward “typing = on-task”).



\*\*Calibration\*\*



\* Start with rules only; measure precision/recall.

\* Enable LLM on “gray zone” windows; track \*\*disagreement rate\*\* vs. rules.

\* Build a small \*\*human-rated adjudication set\*\* to tune thresholds and LLM temperature (ideally 0).



\# 7) Evaluation



\* \*\*Window-level metrics:\*\* accuracy, F1 for on-task, AUROC, PR-AUC.

\* \*\*Temporal stability:\*\* flip rate (label changes per minute), mean time to detect switches.

\* \*\*Coverage:\*\* fraction of windows labeled `uncertain` (target a small, useful abstain band).



\# 8) Privacy \& safety



\* Never transmit raw characters; only \*\*aggregates\*\* and \*\*rates\*\*.

\* Redact or bin digits/symbol counts; no reconstruction possible.

\* Consider \*\*client-side feature extraction\*\*, LLM sees features only.



\# 9) Edge cases \& mitigations



\* \*\*Copy–paste only:\*\* spikes in KPM with low character ratio → flag as uncertain unless pattern repeats with subsequent typing.

\* \*\*Heavy editing:\*\* high backspace/delete can be on-task (drafting). Use presence of sustained bursts to avoid false “off”.

\* \*\*Keyboard shortcuts workflows:\*\* high modifier/nav usage (e.g., power users). Learn app-agnostic “productive shortcut signatures” (Ctrl+S, Ctrl+F sequences) as positive signals.

\* \*\*Micro-breaks:\*\* short idle gaps shouldn’t flip the label—use hysteresis.



\# 10) Minimal viable pipeline



1\. Aggregate keystrokes into 30s windows with the feature set above.

2\. Apply rules; if ambiguous, send the \*\*compact feature block\*\* to GPT-5 Nano.

3\. Get `label`, `confidence`, `reason`; smooth over time; log outputs for review.



If you want, we can next tighten the \*\*exact feature dictionary\*\* and the \*\*rules thresholds\*\* you’ll use to gate the LLM—still no code.



