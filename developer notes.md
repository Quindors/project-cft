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

## beeminder follow-ups

* ~~known soft spot: a session left to expire unattended still counts as "finished" for the sessions goal~~ closed — a session only counts with ≥10 logged minutes behind it (empty chairs log ~nothing past the idle threshold, and quick-clicked "Done"s can't farm the goal either)
* ~~dashboard stats count raw category\_name; beeminder counts human\_label + strict intent, so the two disagree after corrections~~ mostly closed — the dashboard now counts human\_label and declared neutral work the way beeminder does; strict intent (an off-intent productive row → off-task) is still beeminder-only, so the two can still differ by that much during a session

note: implement an AB testing system
use claude code with chrome

