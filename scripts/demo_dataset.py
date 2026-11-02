#!/usr/bin/env python3
"""Swap a demonstration dataset into the live focus.db — and back out again.

The demo keeps everything that is *yours* (categories, projects, intent
corrections, app state) and replaces only the activity history: focus_logs
and the session intentions. It is generated from your real category list, so
every category you have gets screen time in the mix the dashboard, Beeminder
and the weekly report all bucket differently:

  productive   productive-category rows outside a session, and in-session
               rows scored on-intent (ALIGN >= 0.5)
  drift        productive-category rows logged in a session but off its
               intent (ALIGN < 0.5) — real work, not the declared work
  off-intent   unproductive rows inside a session (YouTube mid-essay)
  off-task     Entertainment / Social Media / Gaming outside any session
  neutral      Communication, Composition, Music, Project CFT, Ambiguous,
               System — counted only when declared as a session's work
  rulings      Review-tab verdicts (human_label, human_align) on a few rows

    python scripts/demo_dataset.py install            # back up, then swap in
    python scripts/demo_dataset.py status             # which dataset is live
    python scripts/demo_dataset.py restore            # put the real data back

Works on the running monitor: the swap is one SQLite transaction, the exe's
next query sees the new rows, and anything it logs *during* the demo is
carried back into the real history on restore. Needs only the standard
library, so the Windows python can run it against %LOCALAPPDATA%\\cft (the
exe's data dir); CFT_DATA_DIR or --data-dir point it elsewhere.

Two things the demo cannot hide from, because the monitor computes them
from focus_logs: a running exe pushes TODAY's demo hours to Beeminder (the
real value is re-sent after restore — same day, same requestid), and the
weekly email (REPORT_DAY/REPORT_HOUR in the exe's .env) would mail demo
numbers if it fires while the demo is installed. Restore before then.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FMT = "%Y-%m-%d %H:%M:%S"
ROW_SEC = 10                 # the monitor's decision interval: one row per 10s
STATE_KEY = "demo_dataset"   # app_state flag that says a demo is installed
DEFAULT_DAYS = 14
DEFAULT_SEED = 1101

# --------------------------------------------------------------------------
# The window titles the demo is made of, keyed by the category the classifier
# files them under. Only categories that exist in YOUR categories table are
# used; anything missing here is skipped, anything missing there is dropped.
# --------------------------------------------------------------------------
TITLES = {
    "College Application": [
        "supplements prep - Google Docs - Google Chrome",
        "Supplements thread 2 (purdue) - Claude - Google Chrome",
        "Supplements thread 3 (mit leq) - Claude - Google Chrome",
        "My Colleges | Common App - Google Chrome",
        "MIT Admissions - SlideRoom - Submission - Google Chrome",
        "Personal insight questions | UC Application - Google Chrome",
        "WordCounter - Count Words & Correct Writing - Google Chrome",
        "What if my application is disrupted by a major event beyond my control? | MIT Admissions - Google Chrome",
        "personal statement - Google Docs - Google Chrome",
        "Activities and distinctions - Google Chrome",
    ],
    "Econ": [
        "AP Classroom - Unit 3 Progress Check: MCQ - Google Chrome",
        "Elasticity and its applications - Khan Academy - Google Chrome",
        "Macro Unit 3 notes (Read-Only) - OneNote",
        "Price ceilings and floors - Khan Academy - Google Chrome",
    ],
    "Chem": [
        "AP Chemistry - Equilibrium FRQ practice - Google Chrome",
        "Chem lab report 4 - Google Docs - Google Chrome",
        "Le Chatelier's principle - Khan Academy - Google Chrome",
        "Unit 7 ANSWERS MC Progress Check (Read-Only) - OneNote",
    ],
    "ASE": [
        "ASE seminar response 6 - Google Docs - Google Chrome",
        "Beloved annotations - OneNote",
        "Toni Morrison - Nobel lecture - Google Chrome",
    ],
    "Lin Alg": [
        "Lin Alg HW 7 - Google Docs - Google Chrome",
        "MATH 208 - Canvas - Google Chrome",
        "matrix inverse 3x3 - Wolfram|Alpha - Google Chrome",
    ],
    "Rocketry": [
        "drone new | Assembly 1 - Google Chrome",
        "fin can v3 | Part Studio 1 - Google Chrome",
        "TARC 2027 Rules and Requirements - Google Chrome",
        "OpenRocket - tarc_v3.ork",
        "Bambu Studio - fin_can_v3.3mf",
        "Omelette Attendance/Launch Data - Google Sheets - Google Chrome",
    ],
    "Clarinet": [
        "Weber Concertino, Op. 26 - clarinet - YouTube - Google Chrome",
        "Albert Clarinet Audition 20260929 - Dropbox - Google Chrome",
        "Baermann Scale Studies op. 63 - PDF - Google Chrome",
        "All-State clarinet excerpts 2027 - Google Chrome",
    ],
    "Research & Learning": [
        "Eigenvalues and eigenvectors - Wikipedia - Google Chrome",
        "Essence of linear algebra, chapter 14 - YouTube - Google Chrome",
        "sqlite3 — DB-API 2.0 interface for SQLite databases - Google Chrome",
        "How do I fill out the self-reported coursework? | MIT Admissions - Google Chrome",
    ],
    "Personal Projects": [
        "cycloidal reducer v2 | Part Studio 1 - Google Chrome",
        "gait_planner.py - robot-dog - Visual Studio Code",
        "Cycloidal drive ratio calculator - Google Chrome",
    ],
    "Project CFT": [
        "focus-dashboard - Google Chrome",
        "dashboard_server.py - project-cft - Visual Studio Code",
        "Quindors/focus-dashboard: focus dashboard for project cft - Google Chrome",
        "cft",
    ],
    "Composition": [
        "bagatelle",
        "waltz",
        "hollentanz - MuseScore Studio",
        "Adagio and Tarantella - Wind Repertory Project - Google Chrome",
    ],
    "Music": [
        "Sibelius: Symphony No. 2 - YouTube Music - Google Chrome",
        "Spotify Premium",
        "Discography of Sibelius symphony cycles - Wikipedia - Google Chrome",
    ],
    "Communication": [
        "Chat | Running Start | Microsoft Teams - Google Chrome",
        "Chat | rocketry team omelette | Microsoft Teams - Google Chrome",
        "Chat | clarinets & j(ed) | Microsoft Teams - Google Chrome",
        "Inbox - Gmail - Google Chrome",
        "Marching Band Banquet - Saturday, Nov. 7 - Gmail - Google Chrome",
    ],
    "System": [
        "Windows Default Lock Screen",
        "Task Switching",
        "Settings",
        "File Explorer",
        "Snipping Tool Overlay",
    ],
    "Ambiguous": [
        "New Tab - Google Chrome",
        "Untitled - Google Chrome",
        "Google Search - Google Chrome",
        "Save As",
        "Calculator",
        "2026-10-23 - Football Home Game 4 - Google Drive - Google Chrome",
    ],
    "Entertainment": [
        "Dance of the Knights but all the musicians have ADHD. - YouTube - Google Chrome",
        "Top 10 Marching Band Fails - YouTube - Google Chrome",
        "Netflix - Google Chrome",
        "Twitch - Google Chrome",
        "I built a cycloidal drive from scratch - YouTube - Google Chrome",
    ],
    "Social Media": [
        "(2) Home / X - Google Chrome",
        "Instagram - Google Chrome",
        "r/clarinet - Reddit - Google Chrome",
        "TikTok - Google Chrome",
    ],
    "Gaming": [
        "Minecraft 1.21.4 - Singleplayer",
        "Steam",
        "Hollow Knight: Silksong",
    ],
}

# Reasons the title classifier gives, per category; {t} is the title's
# leading phrase (before the first " - ").
REASONS = {
    "College Application": "title contains '{t}', indicating work related to college applications.",
    "Econ": "title contains '{t}', which relates to AP Economics coursework.",
    "Chem": "title contains '{t}', which relates to AP Chemistry.",
    "ASE": "title contains '{t}', consistent with Advanced Studies in English reading and responses.",
    "Lin Alg": "title contains '{t}', which relates to the linear algebra course.",
    "Rocketry": "title contains '{t}', indicating rocketry CAD or team logistics.",
    "Clarinet": "title contains '{t}', which is clarinet repertoire or audition material.",
    "Research & Learning": "title contains '{t}', an educational reference page.",
    "Personal Projects": "title contains '{t}', which matches the cycloidal / robot dog personal projects.",
    "Project CFT": "title contains '{t}', which is this app or its dashboard.",
    "Composition": "title contains '{t}', a score open in a notation program.",
    "Music": "title contains '{t}', indicating music playback rather than interaction.",
    "Communication": "The window title '{t}' indicates chat or email, consistent with previous classifications.",
    "System": "'{t}' is a system program.",
    "Ambiguous": "The window title '{t}' does not provide specific evidence of a productive category and is too generic to classify definitively.",
    "Entertainment": "The title '{t}' indicates a video or stream watched for entertainment.",
    "Social Media": "'{t}' is a social media feed.",
    "Gaming": "'{t}' is a game or game launcher.",
}

TYPED = [
    "[SHIFT]The bi[BACKSPACE][BACKSPACE] [SHIFT]ANd[BACKSPACE][BACKSPACE][BACKSPACE] ",
    "[LEFT][BACKSPACE][BACKSPACE][BACKSPACE]my",
    "[CTRL_L]\x03", "[ALT_L][TAB]", " of them earlier[SHIFT]? ",
    "[SHIFT]Winter 2027, [SHIFT]Math[BACKSPACE][BACKSPACE][BACKSPACE][SHIFT]ATH",
    "[BACKSPACE][BACKSPACE][BACKSPACE]ic mechanism", "ok im cooked[ENTER]",
]

OFF = ("Entertainment", "Social Media", "Gaming")
NEUTRAL_DECLARABLE = ("Composition", "Communication", "Music", "Project CFT")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
class Demo:
    """Builds focus_logs rows and intentions from a day script, using only
    the categories and projects that exist in the target database."""

    def __init__(self, cats: dict, projects: list, now: datetime, days: int, rng: random.Random):
        self.cats = cats                       # name -> is_productive (1/0/None)
        self.projects = {p["name"]: p for p in projects}
        self.now = now.replace(microsecond=0)
        self.days = days
        self.rng = rng
        self.rows: list = []                   # focus_logs rows (dict)
        self.sessions: list = []               # intentions (dict with "rows")
        self.titles = {c: t for c, t in TITLES.items() if c in cats}

    # -- helpers ------------------------------------------------------------
    def pick(self, cat: str) -> str:
        return self.rng.choice(self.titles[cat])

    def have(self, *names) -> list:
        return [n for n in names if n in self.titles]

    def project_id(self, name: str):
        p = self.projects.get(name)
        return p["id"] if p else None

    def reason(self, cat: str, title: str) -> str:
        return REASONS.get(cat, "title matches the category.").format(t=title.split(" - ")[0])

    def events_for(self, pool: list) -> str:
        k = self.rng.randint(0, min(5, len(pool)))
        return json.dumps(self.rng.sample(pool, k)) if k else None

    def emit(self, t: datetime, cat: str, title: str, *, session=None, align=None,
             vision_from=None, human_label=None, human_align=None, pool=None):
        """One focus_logs row. vision_from=X means the vision pass overrode a
        title-only call of category X (lower confidence, [vision] reason)."""
        if vision_from:
            conf = 0.65
            reason = (f"[vision] The screenshot shows {title.split(' - ')[0]!r} content that "
                      f"belongs to {cat}. (overrode {vision_from}: "
                      f"{self.reason(vision_from, title)})")
        else:
            conf = 0.85
            reason = self.reason(cat, title)
        typed = None
        if cat in ("College Application", "ASE", "Chem", "Communication") and self.rng.random() < 0.12:
            typed = self.rng.choice(TYPED)
        row = {
            "timestamp": t.strftime(FMT), "current_window": title,
            "category_name": cat, "confidence": conf, "reason": reason,
            "human_label": human_label, "typed_text": typed,
            "events": self.events_for(pool or []), "reviewed_at": None,
            "intention_id": session["id"] if session else None,
            "align_score": None if align is None else round(align, 2),
            "human_align": human_align,
        }
        if human_label:
            row["reviewed_at"] = (t + timedelta(hours=2)).strftime(FMT)
        if human_align is not None:
            row["reviewed_at"] = (t + timedelta(hours=3)).strftime(FMT)
        self.rows.append(row)
        if session:
            session["rows"].append(row)
        return row

    def stretch(self, start: datetime, minutes: float, states: list, *, session=None,
                pool=None) -> datetime:
        """Walk `minutes` in ROW_SEC steps through weighted (weight, spec)
        states, each held for a run of 1-6 minutes. spec: dict with cat, and
        optionally align (lo, hi), vision_from, human_label, human_align."""
        end = start + timedelta(minutes=minutes)
        t = start
        pool = pool or []
        weights = [w for w, _ in states]
        while t < end and t < self.now:
            spec = self.rng.choices([s for _, s in states], weights=weights)[0]
            cat = spec["cat"]
            if cat not in self.titles:
                t += timedelta(minutes=1)
                continue
            title = self.pick(cat)
            run_end = min(end, t + timedelta(minutes=self.rng.uniform(1, 6)))
            while t < run_end and t < self.now:
                align = None
                if session and "align" in spec:
                    lo, hi = spec["align"]
                    align = self.rng.uniform(lo, hi)
                self.emit(t, cat, title, session=session, align=align,
                          vision_from=spec.get("vision_from"),
                          human_label=spec.get("human_label"),
                          human_align=spec.get("human_align"), pool=pool)
                t += timedelta(seconds=ROW_SEC)
        return end

    def session(self, text: str, start: datetime, planned_min: int, cat: str, *,
                project: str = None, untimed: bool = False, status: str = "completed",
                worked_min: int = None, paused_sec: int = 0) -> dict:
        """Register an intention. The caller then runs stretch(...) with it;
        ended_at is fixed up from the planned/worked minutes."""
        worked = worked_min if worked_min is not None else planned_min
        # A Marathon has no planned end; the monitor caps it at nine hours.
        ends = start + timedelta(minutes=(planned_min if not untimed else 9 * 60), seconds=paused_sec)
        ended = start + timedelta(minutes=worked, seconds=paused_sec)
        if ended > self.now:
            ended = self.now
        s = {
            "id": None, "text": text, "started_at": start.strftime(FMT),
            "ends_at": ends.strftime(FMT), "ended_at": ended.strftime(FMT),
            "status": status, "llm_category": cat,
            "project_id": self.project_id(project) if project else None,
            "untimed": 1 if untimed else 0, "paused_sec": paused_sec,
            "rows": [], "_worked": worked,
        }
        s["id"] = -(len(self.sessions) + 1)   # placeholder, remapped on insert
        self.sessions.append(s)
        return s

    # -- the states a stretch is made of ------------------------------------
    def on(self, cat, lo=0.8, hi=0.97):
        return {"cat": cat, "align": (lo, hi)}

    def drift(self, cat):            # another productive category, off-intent
        return {"cat": cat, "align": (0.25, 0.45)}

    def fuzzy(self, cat):            # neutral / ambiguous, partly on intent
        return {"cat": cat, "align": (0.45, 0.65)}

    def off(self, cat):              # off-intent
        return {"cat": cat, "align": (0.03, 0.15)}

    def loose(self, cat):            # outside any session: no ALIGN
        return {"cat": cat}

    # -- the day scripts ------------------------------------------------------
    def build(self) -> None:
        first = (self.now - timedelta(days=self.days - 1)).replace(hour=0, minute=0, second=0)
        for d in range(self.days):
            day = first + timedelta(days=d)
            if day.weekday() >= 5:
                self.weekend(day, d)
            else:
                self.weekday(day, d)
        self.history()
        # Today's script runs past the clock: drop sessions that have not
        # started yet (their stretches emitted nothing) and any row after now.
        self.sessions = [s for s in self.sessions if s["started_at"] <= self.now.strftime(FMT)]
        self.rows = [r for r in self.rows if r["timestamp"] <= self.now.strftime(FMT)]

    def jitter(self, day: datetime, hour: int, minute: int, spread: int = 20) -> datetime:
        return day.replace(hour=hour, minute=minute) + timedelta(minutes=self.rng.randint(-spread, spread))

    def weekday(self, day: datetime, d: int) -> None:
        rng = self.rng
        pool = sum(self.titles.values(), [])
        # Morning: laptop open for messages and music before school.
        t = self.jitter(day, 7, 5, 8)
        self.stretch(t, rng.uniform(10, 22), [
            (4, self.loose("Communication")), (2, self.loose("Music")),
            (2, self.loose("Ambiguous")), (1, self.loose("System")),
            (1, self.loose("Social Media")),
        ], pool=pool)
        # Lunch on some days: a drift into feeds, no session.
        if rng.random() < 0.6:
            self.stretch(self.jitter(day, 11, 55, 6), rng.uniform(8, 16), [
                (3, self.loose("Communication")), (2, self.loose("Social Media")),
                (2, self.loose("Entertainment")), (1, self.loose("Ambiguous")),
            ], pool=pool)

        # Afternoon: a Schoolwork session on the course of the day.
        courses = self.have("Econ", "Chem", "ASE", "Lin Alg")
        course = courses[d % len(courses)] if courses else "Research & Learning"
        t = self.jitter(day, 15, 45, 15)
        planned = rng.choice([45, 45, 60, 20])
        status = rng.choices(["completed", "expired", "abandoned"], weights=[5, 3, 1])[0]
        worked = planned if status == "expired" else (
            rng.randint(4, 9) if status == "abandoned" else rng.randint(planned - 12, planned - 2))
        s = self.session(f"{course.lower()} homework" if course != "ASE" else "ase seminar response",
                         t, planned, course, project="Schoolwork", status=status, worked_min=worked,
                         paused_sec=rng.choice([0, 0, 60, 120]))
        drift_cat = rng.choice(self.have("Rocketry", "Personal Projects", "Research & Learning") or [course])
        self.stretch(t, worked, [
            (12, self.on(course)), (2, self.drift(drift_cat)), (2, self.fuzzy("Ambiguous")),
            (1, self.fuzzy("Communication")), (1, self.off("Social Media")), (1, self.off("Entertainment")),
        ], session=s, pool=pool)
        t = t + timedelta(minutes=worked + rng.randint(1, 4))
        # More homework with no session declared: productive, no ALIGN.
        t = self.stretch(t, rng.uniform(25, 45), [
            (6, self.loose(course)), (2, self.loose("Research & Learning")),
            (1, self.loose("Ambiguous")), (1, self.loose("System")),
        ], pool=pool)
        # The post-homework slump.
        self.stretch(t + timedelta(minutes=2), rng.uniform(6, 15), [
            (4, self.loose("Entertainment")), (2, self.loose("Social Media")),
            (1, self.loose("Music")),
        ], pool=pool)

        # Evening: college applications, most nights as a Marathon.
        t = self.jitter(day, 19, 10, 20)
        apps = rng.choice(["Supplement essays", "Supplement essays", "Other college applications"])
        text = {"Supplement essays": rng.choice(["finish purdue 1 & 2", "polish mit saqs", "uc piq #3 draft"]),
                "Other college applications": rng.choice(["mit maker portfolio forms", "self-reported coursework", "activities list"])}[apps]
        untimed = rng.random() < 0.5
        planned = 0 if untimed else rng.choice([45, 60, 60])
        worked = rng.randint(70, 130) if untimed else rng.randint(planned - 10, planned)
        status = "completed" if untimed or rng.random() < 0.6 else "expired"
        s = self.session(text, t, planned, "College Application", project=apps, untimed=untimed,
                         status=status, worked_min=worked, paused_sec=rng.choice([0, 60, 300, 600]))
        states = [
            (14, self.on("College Application")), (2, self.fuzzy("Ambiguous")),
            (1, self.fuzzy("Communication")), (1, self.drift("Composition")),
            (1, self.off("Entertainment")), (1, self.off("Social Media")),
        ]
        if d % 4 == 1:
            # Vision overrode the title: the essay doc was open but the screen
            # was a cycloidal reducer — classified Personal Projects (drift).
            states.append((2, {"cat": "Personal Projects", "align": (0.3, 0.4),
                               "vision_from": "College Application"}))
        self.stretch(t, worked, states, session=s, pool=pool)
        t = t + timedelta(minutes=worked + rng.randint(2, 8))

        # Then composition. Every third night it is DECLARED (a Composition
        # session — neutral work that counts because it was the plan);
        # otherwise it is loose neutral time that counts as neither.
        if "Composition" in self.titles:
            if d % 3 == 0:
                s = self.session("orchestrate the bagatelle", t, 45, "Composition",
                                 status="completed", worked_min=rng.randint(30, 44))
                self.stretch(t, s["_worked"], [
                    (10, self.on("Composition", 0.75, 0.95)), (2, self.fuzzy("Music")),
                    (1, self.off("Entertainment")),
                ], session=s, pool=pool)
                t = t + timedelta(minutes=s["_worked"] + 3)
            else:
                t = self.stretch(t, rng.uniform(10, 25), [
                    (6, self.loose("Composition")), (2, self.loose("Music")), (1, self.loose("Ambiguous")),
                ], pool=pool)
        # Late: off-task, then messages.
        t = self.stretch(t + timedelta(minutes=3), rng.uniform(10, 25), [
            (4, self.loose("Gaming")), (3, self.loose("Entertainment")), (2, self.loose("Social Media")),
            (1, self.loose("Music")),
        ], pool=pool)
        self.stretch(t, rng.uniform(4, 10), [
            (4, self.loose("Communication")), (1, self.loose("System")), (1, self.loose("Ambiguous")),
        ], pool=pool)

    def weekend(self, day: datetime, d: int) -> None:
        rng = self.rng
        pool = sum(self.titles.values(), [])
        # Morning: rocketry CAD or clarinet, declared, no project.
        t = self.jitter(day, 9, 40, 25)
        if d % 2 == 0 and "Rocketry" in self.titles:
            s = self.session("fin can v3 cad + print prep", t, 0, "Rocketry", untimed=True,
                             status="completed", worked_min=rng.randint(70, 120), paused_sec=600)
            self.stretch(t, s["_worked"], [
                (12, self.on("Rocketry")), (2, self.fuzzy("Communication")),
                (2, self.drift("Personal Projects")), (1, self.fuzzy("Ambiguous")),
                (1, self.off("Entertainment")),
            ], session=s, pool=pool)
        else:
            s = self.session("run weber concertino, slow", t, 45, "Clarinet",
                             status="expired", worked_min=45)
            self.stretch(t, 45, [
                (10, self.on("Clarinet")), (3, self.fuzzy("Music")), (1, self.off("Social Media")),
            ], session=s, pool=pool)
        t = t + timedelta(minutes=s["_worked"] + 10)
        self.stretch(t, rng.uniform(20, 40), [
            (4, self.loose("Research & Learning")), (2, self.loose("Ambiguous")),
            (2, self.loose("Communication")), (1, self.loose("Project CFT")),
        ], pool=pool)

        # Afternoon: off-task block, then a long college-app Marathon.
        self.stretch(self.jitter(day, 13, 20, 20), rng.uniform(25, 50), [
            (5, self.loose("Gaming")), (4, self.loose("Entertainment")),
            (2, self.loose("Social Media")), (1, self.loose("Music")),
        ], pool=pool)
        t = self.jitter(day, 15, 40, 20)
        s = self.session("supplements marathon", t, 0, "College Application",
                         project="Supplement essays", untimed=True, status="completed",
                         worked_min=rng.randint(120, 180), paused_sec=rng.choice([600, 900, 1500]))
        self.stretch(t, s["_worked"], [
            (14, self.on("College Application")), (2, self.fuzzy("Ambiguous")),
            (2, self.fuzzy("Communication")), (1, self.drift("Rocketry")),
            (1, self.off("Entertainment")), (1, self.off("Social Media")), (1, self.off("Gaming")),
        ], session=s, pool=pool)
        t = t + timedelta(minutes=s["_worked"] + 15)

        # Evening: personal projects loose (with a correction) and composition.
        t = self.stretch(t, rng.uniform(20, 45), [
            (5, {"cat": "Ambiguous", "human_label": "Personal Projects"}),   # Review-tab correction
            (4, self.loose("Personal Projects")), (1, self.loose("Project CFT")),
        ], pool=pool)
        t = self.stretch(t, rng.uniform(20, 40), [
            (6, self.loose("Composition")), (2, {"cat": "Music", "human_label": "Entertainment"}),
            (1, self.loose("Ambiguous")),
        ], pool=pool)
        self.stretch(t + timedelta(minutes=5), rng.uniform(12, 30), [
            (4, self.loose("Entertainment")), (3, self.loose("Social Media")),
            (2, self.loose("Communication")), (1, self.loose("System")),
        ], pool=pool)

    def history(self) -> None:
        """Sessions older than the dense window, for the forest's life story:
        a Personal Statement left to fade, and the archived Linear Algebra
        project's last waterings. Plus the misjudged-essay session the
        Review tab ruled on."""
        rng = self.rng
        pool = sum(self.titles.values(), [])
        # Personal Statement: last watered 11 days ago (fading on the forest),
        # and its essay rows the model scored off-intent because the topic was
        # music — ruled ON intent by hand (human_align = 1).
        t = (self.now - timedelta(days=11)).replace(hour=20, minute=15, second=0)
        s = self.session("rework the analysis paragraph", t, 45, "College Application",
                         project="Personal Statement", status="completed", worked_min=41)
        self.stretch(t, 20, [(1, self.on("College Application"))], session=s, pool=pool)
        self.stretch(t + timedelta(minutes=20), 14, [
            (1, {"cat": "Composition", "align": (0.05, 0.2), "human_label": "College Application",
                 "human_align": 1}),
        ], session=s, pool=pool)
        self.stretch(t + timedelta(minutes=34), 4, [(1, self.on("College Application"))], session=s, pool=pool)
        # ...and a Wikipedia detour the model let pass (0.6-0.75), ruled OFF.
        self.stretch(t + timedelta(minutes=38), 3, [
            (1, {"cat": "Research & Learning", "align": (0.6, 0.75), "human_align": 0}),
        ], session=s, pool=pool)
        for days_ago in (13, 17, 24):
            t = (self.now - timedelta(days=days_ago)).replace(hour=19, minute=30, second=0)
            s = self.session("personal statement draft", t, 60, "College Application",
                             project="Personal Statement", status="expired", worked_min=60)
            self.stretch(t, 12, [(8, self.on("College Application")), (1, self.fuzzy("Ambiguous"))],
                         session=s, pool=pool)
        # Linear Algebra (archived): three waterings 26-40 days back.
        for days_ago in (26, 33, 40):
            t = (self.now - timedelta(days=days_ago)).replace(hour=16, minute=0, second=0)
            s = self.session("lin alg pset", t, 45, "Lin Alg", project="Linear Algebra",
                             status="completed", worked_min=40)
            self.stretch(t, 10, [(6, self.on("Lin Alg")), (1, self.drift("Rocketry"))],
                         session=s, pool=pool)
        # One abandoned Marathon and one superseded start, like real life.
        t = (self.now - timedelta(days=5)).replace(hour=21, minute=50, second=0)
        s = self.session("one more supplement", t, 0, "College Application",
                         project="Supplement essays", untimed=True, status="abandoned", worked_min=3)
        self.stretch(t, 3, [(1, self.on("College Application"))], session=s, pool=pool)

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict:
        kinds = {"productive": 0, "drift": 0, "off_intent": 0, "off_task": 0,
                 "neutral": 0, "neutral_declared": 0}
        for r in self.rows:
            label = r["human_label"] or r["category_name"]
            prod = self.cats.get(label)
            eff = 1.0 if r["human_align"] == 1 else 0.0 if r["human_align"] == 0 else r["align_score"]
            in_session = r["intention_id"] is not None and eff is not None
            if prod == 1:
                kinds["drift" if in_session and eff < 0.5 else "productive"] += 1
            elif prod == 0:
                kinds["off_intent" if in_session and eff < 0.5 else "off_task"] += 1
            elif in_session and eff >= 0.5 and label.lower() not in ("ambiguous", "system"):
                kinds["neutral_declared"] += 1
            else:
                kinds["neutral"] += 1
        hours = {k: round(v * ROW_SEC / 3600, 1) for k, v in kinds.items()}
        return {"rows": len(self.rows), "sessions": len(self.sessions), "hours": hours,
                "rulings": sum(1 for r in self.rows if r["human_align"] is not None),
                "corrections": sum(1 for r in self.rows if r["human_label"])}


# --------------------------------------------------------------------------
# Install / restore
# --------------------------------------------------------------------------
def data_dir(arg) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("CFT_DATA_DIR"):
        return Path(os.environ["CFT_DATA_DIR"])
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if not base:
        sys.exit("no %LOCALAPPDATA%: pass --data-dir (the folder holding focus.db)")
    return Path(base) / "cft"


def connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_state(conn: sqlite3.Connection):
    try:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (STATE_KEY,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return json.loads(row[0]) if row else None


def install(args) -> int:
    d = data_dir(args.data_dir)
    db = d / "focus.db"
    if not db.exists():
        sys.exit(f"{db} does not exist — run the monitor once first, or pass --data-dir")
    conn = connect(db)
    for table in ("focus_logs", "categories", "intentions", "projects", "app_state"):
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
            sys.exit(f"{db} has no {table} table; start the monitor once so it creates the schema")
    state = get_state(conn)
    if state:
        sys.exit(f"a demo dataset is already installed (since {state['installed_at']}); "
                 f"run `restore` first")

    cats = {r["name"]: r["is_productive"] for r in conn.execute("SELECT name, is_productive FROM categories")}
    projects = [dict(r) for r in conn.execute("SELECT id, name, status FROM projects")]
    if not cats:
        sys.exit("no categories in the database — the demo is built from them")

    now = datetime.now()
    demo = Demo(cats, projects, now, args.days, random.Random(args.seed))
    demo.build()
    demo.rows.sort(key=lambda r: r["timestamp"])

    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup = d / f"focus.db.pre-demo-{stamp}"
    conn.execute("VACUUM INTO ?", (str(backup),))
    real_logs = conn.execute("SELECT COUNT(*) FROM focus_logs").fetchone()[0]
    real_sessions = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
    with connect(backup) as b:
        if (b.execute("SELECT COUNT(*) FROM focus_logs").fetchone()[0] != real_logs
                or b.execute("SELECT COUNT(*) FROM intentions").fetchone()[0] != real_sessions):
            sys.exit(f"backup {backup} does not match the live database; nothing changed")

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM focus_logs")
        conn.execute("DELETE FROM intentions")
        # Sessions first, so rows can point at their real ids.
        for s in demo.sessions:
            cur = conn.execute(
                "INSERT INTO intentions (text, started_at, ends_at, ended_at, status, llm_category, "
                "project_id, untimed, paused_sec) VALUES (?,?,?,?,?,?,?,?,?)",
                (s["text"], s["started_at"], s["ends_at"], s["ended_at"], s["status"],
                 s["llm_category"], s["project_id"], s["untimed"], s["paused_sec"]))
            for r in s["rows"]:
                r["intention_id"] = cur.lastrowid
        conn.executemany(
            "INSERT INTO focus_logs (timestamp, current_window, category_name, confidence, reason, "
            "human_label, typed_text, events, reviewed_at, intention_id, align_score, human_align) "
            "VALUES (:timestamp, :current_window, :category_name, :confidence, :reason, :human_label, "
            ":typed_text, :events, :reviewed_at, :intention_id, :align_score, :human_align)",
            demo.rows)
        logs_hi = conn.execute("SELECT MAX(id) FROM focus_logs").fetchone()[0]
        sess_lo = conn.execute("SELECT MIN(id) FROM intentions").fetchone()[0]
        sess_hi = conn.execute("SELECT MAX(id) FROM intentions").fetchone()[0]
        state = {
            "installed_at": now.strftime(FMT), "backup": str(backup),
            "logs_watermark": logs_hi, "sessions_lo": sess_lo, "sessions_watermark": sess_hi,
            "real_logs": real_logs, "real_sessions": real_sessions,
            "days": args.days, "seed": args.seed,
        }
        conn.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)",
                     (STATE_KEY, json.dumps(state)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    conn.close()

    s = demo.summary()
    print(f"backed up {real_logs} rows / {real_sessions} sessions to {backup}")
    print(f"installed demo: {s['rows']} rows over {args.days} days, {s['sessions']} sessions "
          f"(ids {sess_lo}-{sess_hi}), {s['corrections']} corrections, {s['rulings']} intent rulings")
    print("hours by bucket: " + ", ".join(f"{k} {v}" for k, v in s["hours"].items()))
    print("restore with: python scripts/demo_dataset.py restore")
    return 0


def restore(args) -> int:
    d = data_dir(args.data_dir)
    db = d / "focus.db"
    conn = connect(db)
    state = get_state(conn)
    if not state:
        sys.exit("no demo dataset is installed here (nothing to restore)")
    backup = Path(state["backup"])
    if not backup.exists():
        sys.exit(f"the backup {backup} is gone; the real history cannot be restored from here")

    logs_wm, sess_lo, sess_wm = state["logs_watermark"], state["sessions_lo"], state["sessions_watermark"]
    conn.execute("ATTACH DATABASE ? AS real", (str(backup),))
    conn.execute("BEGIN IMMEDIATE")
    try:
        live_logs = conn.execute("SELECT COUNT(*) FROM focus_logs WHERE id > ?", (logs_wm,)).fetchone()[0]
        live_sessions = conn.execute("SELECT COUNT(*) FROM intentions WHERE id > ?", (sess_wm,)).fetchone()[0]
        conn.execute("DELETE FROM focus_logs WHERE id <= ?", (logs_wm,))
        conn.execute("DELETE FROM intentions WHERE id <= ?", (sess_wm,))
        # Rows the monitor logged during the demo under a demo session: keep
        # the rows, drop the pointer to a session that no longer exists.
        conn.execute("UPDATE focus_logs SET intention_id = NULL, align_score = NULL, human_align = NULL "
                     "WHERE intention_id BETWEEN ? AND ?", (sess_lo, sess_wm))
        cols = [r[1] for r in conn.execute("PRAGMA real.table_info(focus_logs)")]
        mine = {r[1] for r in conn.execute("PRAGMA main.table_info(focus_logs)")}
        cols = [c for c in cols if c in mine]
        cl = ", ".join(cols)
        conn.execute(f"INSERT INTO main.focus_logs ({cl}) SELECT {cl} FROM real.focus_logs")
        icols = [r[1] for r in conn.execute("PRAGMA real.table_info(intentions)")]
        imine = {r[1] for r in conn.execute("PRAGMA main.table_info(intentions)")}
        il = ", ".join(c for c in icols if c in imine)
        conn.execute(f"INSERT INTO main.intentions ({il}) SELECT {il} FROM real.intentions")
        conn.execute("DELETE FROM app_state WHERE key = ?", (STATE_KEY,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    n_logs = conn.execute("SELECT COUNT(*) FROM focus_logs").fetchone()[0]
    n_sess = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
    conn.execute("DETACH DATABASE real")
    conn.close()
    print(f"restored {state['real_logs']} rows / {state['real_sessions']} sessions from {backup.name}, "
          f"kept {live_logs} rows / {live_sessions} sessions logged during the demo "
          f"(now {n_logs} rows / {n_sess} sessions)")
    if args.keep_backup:
        print(f"backup kept at {backup}")
    else:
        backup.unlink()
        print("backup removed")
    cloud_cleanup(args.env, sess_lo, sess_wm)
    return 0


def cloud_cleanup(env_path, lo: int, hi: int) -> None:
    """The exe's startup backfill mirrors every intention to Supabase, demo
    ones included (monitor/supa_mirror.mirror_intentions). Delete those ids
    from the cloud copy, so the hosted dashboard's offline fallback does not
    keep showing demo sessions. Best effort: says what to do by hand if it
    cannot."""
    creds = {}
    for p in [env_path] if env_path else []:
        try:
            for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    creds[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
    url = creds.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = creds.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    hint = (f"delete intentions with id between {lo} and {hi} from the Supabase copy if the "
            f"monitor ran while the demo was installed (its startup backfill mirrors them)")
    if not (url and key):
        print(f"cloud copy not touched (no SUPABASE_URL/KEY; pass --env <the exe's .env>): {hint}")
        return
    req = Request(f"{url.rstrip('/')}/rest/v1/intentions?id=gte.{lo}&id=lte.{hi}", method="DELETE",
                  headers={"apikey": key, "Authorization": f"Bearer {key}", "Prefer": "return=minimal"})
    try:
        with urlopen(req, timeout=15):
            pass
        print(f"cloud copy: removed demo sessions {lo}-{hi} (if any were mirrored)")
    except (HTTPError, URLError, OSError) as e:
        print(f"cloud copy not cleaned ({e}): {hint}")


def status(args) -> int:
    d = data_dir(args.data_dir)
    db = d / "focus.db"
    if not db.exists():
        print(f"no database at {db}")
        return 1
    conn = connect(db)
    state = get_state(conn)
    n_logs = conn.execute("SELECT COUNT(*) FROM focus_logs").fetchone()[0]
    n_sess = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
    span = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM focus_logs").fetchone()
    if state:
        live = conn.execute("SELECT COUNT(*) FROM focus_logs WHERE id > ?", (state["logs_watermark"],)).fetchone()[0]
        print(f"DEMO dataset installed since {state['installed_at']} ({n_logs} rows / {n_sess} sessions, "
              f"{live} logged live since); backup: {state['backup']}")
    else:
        print(f"real dataset ({n_logs} rows / {n_sess} sessions, {span[0]} .. {span[1]})")
    conn.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", help="folder holding focus.db (default: CFT_DATA_DIR or %%LOCALAPPDATA%%\\cft)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install", help="back up the real history and swap the demo in")
    i.add_argument("--days", type=int, default=DEFAULT_DAYS, help="days of dense history (default 14)")
    i.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (same seed = same demo)")
    r = sub.add_parser("restore", help="put the real history back, keeping anything logged meanwhile")
    r.add_argument("--keep-backup", action="store_true", help="leave the pre-demo backup file in place")
    r.add_argument("--env", help="the exe's .env, for the Supabase cleanup of mirrored demo sessions")
    sub.add_parser("status", help="say which dataset is live")
    args = ap.parse_args(argv)
    return {"install": install, "restore": restore, "status": status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
