## goals list
- ~~explore user intention input~~ done — session intentions with ALIGN scoring, drift alerts, breaks
- increase the doubtfulness of the LLM (it doesn't say ambiguous enough)

## done recently (aug 2026)
- idle detection: monitor stands down after 5 min without input (CFT_IDLE_SEC)
- beeminder integration: daily focus hours + finished sessions pushed as datapoints
  - setup card on the dashboard (create a goal with chosen commitment/leeway, or connect an existing one)
  - off-intent work counts as off-task by default (BEEMINDER_STRICT_INTENT=0 to disable)
  - .env knobs: BEEMINDER_USER/TOKEN/FOCUS_GOAL/SESSIONS_GOAL, SYNC_SEC, DRY_RUN

## beeminder follow-ups
- known soft spot: a session left to expire unattended still counts as "finished" for the sessions goal
- dashboard stats count raw category_name; beeminder counts human_label + strict intent, so the two disagree after corrections — consider teaching the dashboard the same math

note: implement an AB testing system
use claude code with chrome
