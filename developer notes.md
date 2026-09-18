## goals list

* ~~explore user intention input~~ done — session intentions with ALIGN scoring, drift alerts, breaks
* increase the doubtfulness of the LLM (it doesn't say ambiguous enough)
* create mobile companion app; package current desktop app further?? increase security

## done recently (aug 2026)

* idle detection: monitor stands down after 5 min without input (CFT\_IDLE\_SEC)
* beeminder integration: daily focus hours + finished sessions pushed as datapoints

  * setup card on the dashboard (create a goal with chosen commitment/leeway, or connect an existing one)
  * off-intent work counts as off-task by default (BEEMINDER\_STRICT\_INTENT=0 to disable)
  * neutral categories are productive only while declared: a row under a session with ALIGN ≥ 0.5 counts as productive (Ambiguous/System never do); the dashboard applies the same rule and its pie shows that time as "\<category> (in session)"
  * .env knobs: BEEMINDER\_USER/TOKEN/FOCUS\_GOAL/SESSIONS\_GOAL, SYNC\_SEC, DRY\_RUN
* weekly email report: the last seven days mailed to user-defined recipients (yourself, a parent, a coach) every REPORT\_DAY/REPORT\_HOUR

  * setup card on the dashboard (recipients, SMTP with Gmail/Outlook/iCloud presets, day + hour, preview, send now); the mail server login is checked before anything is saved, so a bad app password is a readable error rather than a silent Monday
  * totals through beeminder.label\_rows/bucket, so the inbox figure is the datapoint figure — corrections, strict intent and declared-neutral all included
  * catch-up scheduling (a machine asleep through the slot sends on wake, once, for the latest week), with a first-run baseline so switching it on doesn't immediately post last week
  * .env knobs: REPORT\_EMAIL\_TO/SMTP\_HOST/SMTP\_PORT/SMTP\_USER/SMTP\_PASSWORD/EMAIL\_FROM/SMTP\_SECURITY/DAY/HOUR
* error messages rewritten app-wide as **what happened + what to do about it** (monitor/problems.py, src/components/ErrorNote.jsx)

  * `Problem(message, fix)` is a str subclass, so old callers that printed or matched on the sentence keep working; the API answers `{error, fix}` and the dashboard sets the fix in its own quieter line
  * `UserError` carries one up from intentions/breaks/intent\_corrections through the API
  * covered: every dashboard API rejection, SMTP failures, Beeminder token/goal/network failures, the intent gate, the first-run API-key dialog
  * NOT covered on purpose: console logs (boot timings, `[tray]` diagnostics) — those are for debugging, and prose would make cft.log worse
  * guarded by `test_every_rejection_says_what_to_do_about_it` (no 4xx without a fix) and `test_no_rejection_leaks_python_at_the_reader` (no field names, exception classes or tracebacks) — add new endpoints to that list

## beeminder follow-ups

* ~~known soft spot: a session left to expire unattended still counts as "finished" for the sessions goal~~ closed — a session only counts with ≥10 logged minutes behind it (empty chairs log ~nothing past the idle threshold, and quick-clicked "Done"s can't farm the goal either)
* ~~dashboard stats count raw category\_name; beeminder counts human\_label + strict intent, so the two disagree after corrections~~ mostly closed — the dashboard now counts human\_label and declared neutral work the way beeminder does; strict intent (an off-intent productive row → off-task) is still beeminder-only, so the two can still differ by that much during a session

note: implement an AB testing system
use claude code with chrome

