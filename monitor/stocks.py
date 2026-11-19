# monitor/stocks.py
"""A stock per project, bought with focused hours — on paper.

Give a project a ticker and every productive hour worked inside it buys a
fixed dollar amount of that stock in an Alpaca PAPER account: pretend money,
real prices. The holding only ever grows by working; what it is worth floats
with the market. It is a score to watch, not money — nothing here can reach a
live brokerage account (see _is_paper), and nothing is ever sold.

What counts as a productive hour is not decided here. It is the hours goal's
own rule (monitor/beeminder.py: bucket), narrowed to the rows logged during
the project's sessions — so drifting off-intent, an unproductive category, or
work done outside a session buys nothing, and every hole closed for Beeminder
is closed for this too.

The ledger is a pair of local tables in focus.db:

  project_tickers   one row per stretch of time a project had a ticker. Hours
                    count from the moment the ticker was set, never before:
                    a project with a year of history does not buy a year of
                    stock the day it gets one.
  stock_buys        one row per order sent to the broker.

There is no "already paid for" watermark. What a project is owed is simply

    rate x productive hours while it had a ticker  -  dollars already sent

recomputed from the rows every time, so a session refiled into the project,
or rows relabelled on the Review tab, change the figure the way they change
the Beeminder datapoint, and the next buy settles the difference.

Buys are sent whenever enough is owed, market open or not: Alpaca queues an
after-hours order for the next open, and the row reads "queued" until it
fills. Each row is written BEFORE its order is sent and carries the order's
client id, so a timeout or a crash mid-send is resolved by asking the broker
about that id — never by buying again.

Prices come from Alpaca's delayed SIP feed: every US exchange, fifteen
minutes late, which is what the free plan allows and plenty for a holding
that is looked at, not traded.

Like Beeminder, entirely opt-in (nothing happens until the .env carries the
paper account's keys), best-effort, and off the critical path.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from monitor.paths import data_dir
from monitor.problems import Problem, UserError

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
# The only hosts an order may be sent to: Alpaca's paper endpoint, and this
# machine (the E2E test's stand-in broker).
_PAPER_HOSTS = ("paper-api.alpaca.markets", "127.0.0.1", "localhost")
FEED = "delayed_sip"

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_FOREVER = "9999-12-31 00:00:00"
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

# Alpaca will not take a dollar-amount order under one dollar.
BROKER_MIN_USD = 1.0

# A buy's life. PENDING: the row exists, the broker has not confirmed it yet.
# QUEUED: accepted, waiting to fill (after hours, usually). FILLED: shares.
# REFUSED: the broker said no. DROPPED: accepted, then cancelled or expired
# without filling. The last two free the dollars to be owed again.
PENDING, QUEUED, FILLED, REFUSED, DROPPED = "pending", "queued", "filled", "refused", "dropped"
_SPENT = (PENDING, QUEUED, FILLED)


@dataclass(frozen=True)
class Config:
    key: str
    secret: str
    base_url: str
    data_url: str
    usd_per_hour: float
    min_buy_usd: float
    sync_sec: float
    timeout_sec: float
    price_ttl_sec: float
    retry_sec: float


class BrokerError(Exception):
    """Alpaca answered, and the answer was no."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"{status} {message}")


class Unreachable(Exception):
    """Alpaca did not answer at all."""


# ------------------- configuration -------------------

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[stocks] {name}={raw!r} is not a number; using {default}")
        return default


def _base(url: str) -> str:
    """An endpoint as typed in a .env, down to scheme://host — people paste
    the dashboard's own ".../v2" form."""
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v2") else url


def config() -> Config:
    """Read on every call, like beeminder.config(), so a .env edit followed
    by a restart is all a change ever needs."""
    return Config(
        key=_env("APCA_API_KEY_ID"),
        secret=_env("APCA_API_SECRET_KEY"),
        base_url=_base(_env("APCA_BASE_URL", PAPER_URL)),
        data_url=_base(_env("APCA_DATA_URL", DATA_URL)),
        usd_per_hour=max(0.0, _env_float("STOCKS_USD_PER_HOUR", 15.0)),
        min_buy_usd=max(BROKER_MIN_USD, _env_float("STOCKS_MIN_BUY_USD", 5.0)),
        sync_sec=max(1.0, _env_float("STOCKS_SYNC_SEC", 300.0)),
        timeout_sec=_env_float("STOCKS_TIMEOUT_SEC", 8.0),
        price_ttl_sec=_env_float("STOCKS_PRICE_TTL_SEC", 60.0),
        retry_sec=_env_float("STOCKS_RETRY_SEC", 3600.0),
    )


def _is_paper(cfg: Config) -> bool:
    """Whether orders would go to pretend money. The one check standing
    between this module and a real brokerage account, so it is a whitelist:
    anything that is not Alpaca's paper endpoint is treated as live."""
    return (urlparse(cfg.base_url).hostname or "").lower() in _PAPER_HOSTS


def enabled(cfg: Optional[Config] = None) -> bool:
    cfg = cfg or config()
    return bool(cfg.key and cfg.secret and _is_paper(cfg))


def _setup_problem(cfg: Config) -> Optional[Problem]:
    if not (cfg.key and cfg.secret):
        return Problem(
            "No paper trading account is connected, so projects can't hold stock yet.",
            "Make a free account on alpaca.markets, generate keys on its Paper "
            "dashboard, and put them in the .env next to the app as "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY. Then restart the monitor.")
    if not _is_paper(cfg):
        return Problem(
            "This only ever buys with Alpaca's paper money, and the address in "
            "the .env points at a live account.",
            "Set APCA_BASE_URL=https://paper-api.alpaca.markets in the .env next "
            "to the app, use the keys from the Paper dashboard, and restart "
            "the monitor.")
    return None


# ------------------- the broker -------------------

def _call(cfg: Config, method: str, url: str, body: Optional[Dict] = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method, headers={
        "APCA-API-KEY-ID": cfg.key,
        "APCA-API-SECRET-KEY": cfg.secret,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=cfg.timeout_sec) as resp:
            raw = resp.read()
    except HTTPError as e:
        try:
            said = json.loads(e.read() or b"{}").get("message") or e.reason
        except Exception:
            said = e.reason
        raise BrokerError(e.code, " ".join(str(said).split())) from None
    except (URLError, OSError, TimeoutError) as e:
        raise Unreachable(str(getattr(e, "reason", e))) from None
    try:
        return json.loads(raw or b"null")
    except ValueError:
        raise Unreachable("an answer that wasn't JSON") from None


def _clean_symbol(symbol) -> str:
    symbol = "".join(str(symbol or "").split()).upper()
    if not _SYMBOL_RE.match(symbol):
        raise UserError(
            "That doesn't look like a ticker.",
            "A ticker is the short code a stock trades under, like SPY or "
            "AAPL: a few letters, no spaces.")
    return symbol


def lookup_asset(symbol: str, cfg: Optional[Config] = None) -> Dict:
    """Alpaca's record for a ticker, or a UserError saying why a project
    can't hold it."""
    cfg = cfg or config()
    try:
        asset = _call(cfg, "GET", f"{cfg.base_url}/v2/assets/{quote(symbol, safe='')}")
    except BrokerError as e:
        if e.status == 404:
            raise UserError(
                f"Alpaca doesn't know a stock called {symbol}.",
                "Check the ticker on the company's own page or a quote site: "
                "it has to be a US-listed stock or fund, by its trading code.")
        if e.status in (401, 403):
            raise UserError(
                "Alpaca wouldn't accept the paper account's keys.",
                "Generate new keys on the Paper dashboard at app.alpaca.markets, "
                "replace APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env "
                "next to the app, and restart the monitor.")
        raise UserError(
            f"Alpaca couldn't look up {symbol} just now.",
            "Try again in a minute; if it keeps happening, check that "
            "app.alpaca.markets itself is up.")
    except Unreachable:
        raise UserError(
            f"Couldn't reach Alpaca to check {symbol}.",
            "Check this PC's internet connection and try again: the ticker is "
            "only saved once Alpaca confirms it exists.")
    asset = asset or {}
    if not asset.get("tradable") or asset.get("status", "active") != "active":
        raise UserError(
            f"{symbol} isn't trading on Alpaca right now.",
            "Pick a different ticker for this project: one that is listed "
            "and active today.")
    if not asset.get("fractionable"):
        raise UserError(
            f"{symbol} can't be bought in fractions of a share.",
            "An hour of work buys a few dollars' worth, which only works for "
            "stocks Alpaca sells in fractions. Most large stocks and funds "
            "are; pick one of those instead.")
    return asset


# ------------------- the ledger -------------------

def _now_str() -> str:
    return datetime.now().strftime(_TS_FMT)


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """CREATE TABLE IF NOT EXISTS project_tickers (
               id         INTEGER PRIMARY KEY AUTOINCREMENT,
               project_id INTEGER NOT NULL,
               symbol     TEXT NOT NULL,
               asset_name TEXT NOT NULL DEFAULT '',
               started_at TEXT NOT NULL,
               ended_at   TEXT
           );
           CREATE TABLE IF NOT EXISTS stock_buys (
               id              INTEGER PRIMARY KEY AUTOINCREMENT,
               project_id      INTEGER NOT NULL,
               symbol          TEXT NOT NULL,
               usd             REAL NOT NULL,
               client_order_id TEXT NOT NULL UNIQUE,
               order_id        TEXT,
               status          TEXT NOT NULL,
               qty             REAL NOT NULL DEFAULT 0,
               price           REAL,
               error           TEXT,
               created_at      TEXT NOT NULL,
               updated_at      TEXT NOT NULL,
               filled_at       TEXT
           );
           CREATE INDEX IF NOT EXISTS idx_stock_buys_project ON stock_buys(project_id);"""
    )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(data_dir() / "focus.db"), timeout=5)
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)
    return conn


def _current(conn: sqlite3.Connection, project_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM project_tickers WHERE project_id = ? AND ended_at IS NULL "
        "ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()


def set_ticker(project_id, symbol) -> Dict:
    """Give a project a ticker, change it, or (symbol None/"") take it off.

    Hours start counting from now. Changing the ticker keeps every share
    already bought; new hours buy the new one. Returns the project's entry
    as status() lists it.
    """
    from monitor import projects
    cfg = config()
    problem = _setup_problem(cfg)
    if problem is not None:
        raise UserError(problem.message, problem.fix)
    project = projects.get(project_id)
    if project is None:
        raise UserError(
            "That project isn't there any more.",
            "Refresh the Projects tab: it may have been deleted since the "
            "page loaded.")
    pid = project["id"]

    wanted = None
    if str(symbol or "").strip():
        wanted = _clean_symbol(symbol)

    with _connect() as conn:
        current = _current(conn, pid)
        have = current["symbol"] if current is not None else None
    if wanted != have:
        # Asked of the broker before anything is written, so a typo never
        # leaves the project without the ticker it had.
        asset = lookup_asset(wanted, cfg) if wanted else None
        now = _now_str()
        with _connect() as conn:
            conn.execute(
                "UPDATE project_tickers SET ended_at = ? "
                "WHERE project_id = ? AND ended_at IS NULL", (now, pid))
            if wanted:
                conn.execute(
                    "INSERT INTO project_tickers (project_id, symbol, asset_name, started_at) "
                    "VALUES (?,?,?,?)", (pid, wanted, str(asset.get("name") or ""), now))
    return next(p for p in status(cfg)["projects"] if p["project_id"] == pid)


def _earned(conn: sqlite3.Connection, project_id: int, cfg: Config) -> Tuple[float, float]:
    """(productive seconds, dollars earned) over every stretch the project
    had a ticker."""
    from monitor import beeminder
    bcfg = beeminder.config()
    seconds = 0.0
    for span in conn.execute(
            "SELECT started_at, ended_at FROM project_tickers WHERE project_id = ?",
            (project_id,)).fetchall():
        seconds += beeminder.project_productive_sec(
            project_id, span["started_at"], span["ended_at"] or _FOREVER,
            seconds_per_row=bcfg.seconds_per_row, strict_intent=bcfg.strict_intent)
    return seconds, cfg.usd_per_hour * seconds / 3600.0


def _spent(conn: sqlite3.Connection, project_id: int) -> float:
    marks = ",".join("?" * len(_SPENT))
    row = conn.execute(
        f"SELECT COALESCE(SUM(usd), 0) AS usd FROM stock_buys "
        f"WHERE project_id = ? AND status IN ({marks})", (project_id, *_SPENT)).fetchone()
    return float(row["usd"])


# ------------------- buying -------------------

def _apply(conn: sqlite3.Connection, buy_id: int, order: Dict) -> None:
    """Bring a buy's row in line with what the broker says about its order."""
    state = str(order.get("status") or "")
    qty = float(order.get("filled_qty") or 0)
    price = order.get("filled_avg_price")
    price = float(price) if price not in (None, "") else None
    now = _now_str()
    if state == "filled" or (state in ("canceled", "expired", "rejected") and qty > 0 and price):
        # A partial fill that was then cancelled still bought something: keep
        # the shares, and charge the project only for what they cost.
        conn.execute(
            "UPDATE stock_buys SET status = ?, order_id = ?, qty = ?, price = ?, "
            "usd = CASE WHEN ? THEN usd ELSE ? END, error = NULL, updated_at = ?, "
            "filled_at = ? WHERE id = ?",
            (FILLED, order.get("id"), qty, price, state == "filled",
             round(qty * (price or 0), 2), now, now, buy_id))
    elif state in ("canceled", "expired", "rejected"):
        conn.execute(
            "UPDATE stock_buys SET status = ?, order_id = ?, error = ?, updated_at = ? WHERE id = ?",
            (DROPPED, order.get("id"), f"the order was {state} before it filled", now, buy_id))
    else:
        conn.execute(
            "UPDATE stock_buys SET status = ?, order_id = ?, error = NULL, updated_at = ? WHERE id = ?",
            (QUEUED, order.get("id"), now, buy_id))


def _send(conn: sqlite3.Connection, cfg: Config, buy: sqlite3.Row) -> None:
    try:
        order = _call(cfg, "POST", f"{cfg.base_url}/v2/orders", {
            "symbol": buy["symbol"],
            "notional": f"{buy['usd']:.2f}",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": buy["client_order_id"],
        })
    except BrokerError as e:
        if e.status == 429 or e.status >= 500:
            raise Unreachable(e.message)       # their trouble, not a refusal
        conn.execute(
            "UPDATE stock_buys SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (REFUSED, e.message, _now_str(), buy["id"]))
        print(f"[stocks] {buy['symbol']} ${buy['usd']:.2f} refused: {e.message}")
        return
    _apply(conn, buy["id"], order or {})
    print(f"[stocks] sent ${buy['usd']:.2f} of {buy['symbol']} "
          f"for project {buy['project_id']} ({(order or {}).get('status')})")


def _reconcile(conn: sqlite3.Connection, cfg: Config) -> None:
    """Ask the broker about every buy that had not settled when last seen."""
    for buy in conn.execute(
            "SELECT * FROM stock_buys WHERE status IN (?, ?) ORDER BY id",
            (PENDING, QUEUED)).fetchall():
        try:
            if buy["order_id"]:
                order = _call(cfg, "GET", f"{cfg.base_url}/v2/orders/{quote(buy['order_id'], safe='')}")
            else:
                order = _call(cfg, "GET", f"{cfg.base_url}/v2/orders:by_client_order_id?"
                              + urlencode({"client_order_id": buy["client_order_id"]}))
        except BrokerError as e:
            if e.status == 404 and buy["status"] == PENDING:
                # The send never arrived. Same client id, so a send that DID
                # arrive and is merely slow to show up cannot double it.
                _send(conn, cfg, buy)
                conn.commit()
                continue
            raise Unreachable(e.message)
        _apply(conn, buy["id"], order or {})
        conn.commit()


def _recently_refused(conn: sqlite3.Connection, project_id: int, symbol: str,
                      cfg: Config) -> Optional[sqlite3.Row]:
    """The project's last buy of this ticker, if the broker refused it less
    than retry_sec ago — a refusal is rarely cured by asking again at once."""
    last = conn.execute(
        "SELECT * FROM stock_buys WHERE project_id = ? AND symbol = ? "
        "ORDER BY id DESC LIMIT 1", (project_id, symbol)).fetchone()
    if last is None or last["status"] != REFUSED:
        return None
    age = (datetime.now() - datetime.strptime(last["updated_at"], _TS_FMT)).total_seconds()
    return last if age < cfg.retry_sec else None


_settle_lock = threading.Lock()
_health_lock = threading.Lock()
_health: Dict[str, Optional[str]] = {"unreachable": None}


def _note_reach(error: Optional[str]) -> None:
    with _health_lock:
        _health["unreachable"] = error


def settle(cfg: Optional[Config] = None) -> Dict:
    """Catch the ledger up with the broker, then buy for every project that
    is owed enough. Safe to call from anywhere, any number of times: one
    settle runs at a time, and what is owed is recomputed from the rows.
    """
    cfg = cfg or config()
    placed = 0
    if not enabled(cfg):
        return {"placed": 0}
    with _settle_lock:
        try:
            with _connect() as conn:
                _reconcile(conn, cfg)
                for span in conn.execute(
                        "SELECT * FROM project_tickers WHERE ended_at IS NULL ORDER BY id").fetchall():
                    pid, symbol = span["project_id"], span["symbol"]
                    _, earned = _earned(conn, pid, cfg)
                    owed = earned - _spent(conn, pid)
                    usd = math.floor(owed * 100 + 1e-6) / 100
                    if usd < cfg.min_buy_usd or _recently_refused(conn, pid, symbol, cfg):
                        continue
                    now = _now_str()
                    cur = conn.execute(
                        "INSERT INTO stock_buys (project_id, symbol, usd, client_order_id, "
                        "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                        (pid, symbol, usd, f"cft-{pid}-{uuid.uuid4().hex[:16]}", PENDING, now, now))
                    conn.commit()     # on disk before the broker hears of it
                    buy = conn.execute("SELECT * FROM stock_buys WHERE id = ?",
                                       (cur.lastrowid,)).fetchone()
                    _send(conn, cfg, buy)
                    conn.commit()
                    placed += 1
            _note_reach(None)
        except Unreachable as e:
            _note_reach(str(e) or "no answer")
            print(f"[stocks] couldn't reach Alpaca ({e}); will retry next tick")
    return {"placed": placed}


# ------------------- reading it back -------------------

_cache_lock = threading.Lock()
_price_cache: Dict[str, Tuple[float, str, float]] = {}   # symbol -> (price, traded_at, fetched)
_clock_cache: Dict[str, object] = {"at": 0.0, "clock": None}


def _prices(symbols: List[str], cfg: Config) -> Dict[str, Tuple[float, str]]:
    """Latest trade per symbol from the delayed SIP feed, cached briefly. A
    symbol the feed can't price right now is simply absent."""
    now = time.time()
    with _cache_lock:
        stale = [s for s in symbols
                 if s not in _price_cache or now - _price_cache[s][2] >= cfg.price_ttl_sec]
    if stale:
        try:
            got = _call(cfg, "GET", f"{cfg.data_url}/v2/stocks/trades/latest?"
                        + urlencode({"symbols": ",".join(sorted(stale)), "feed": FEED}))
            with _cache_lock:
                for sym, trade in ((got or {}).get("trades") or {}).items():
                    if trade and trade.get("p"):
                        _price_cache[sym] = (float(trade["p"]), str(trade.get("t") or ""), now)
            _note_reach(None)
        except (BrokerError, Unreachable) as e:
            _note_reach(str(e) or "no answer")
    with _cache_lock:
        return {s: _price_cache[s][:2] for s in symbols if s in _price_cache}


def _clock(cfg: Config) -> Optional[Dict]:
    now = time.time()
    with _cache_lock:
        if _clock_cache["clock"] is not None and now - float(_clock_cache["at"]) < cfg.price_ttl_sec:
            return _clock_cache["clock"]        # type: ignore[return-value]
    try:
        got = _call(cfg, "GET", f"{cfg.base_url}/v2/clock") or {}
    except (BrokerError, Unreachable) as e:
        _note_reach(str(e) or "no answer")
        return None
    clock = {"is_open": bool(got.get("is_open")),
             "next_open": got.get("next_open"), "next_close": got.get("next_close")}
    with _cache_lock:
        _clock_cache.update(at=now, clock=clock)
    return clock


def _refusal_problem(buy: sqlite3.Row) -> Dict[str, str]:
    return Problem(
        f"Alpaca refused the last buy of {buy['symbol']}: {buy['error'] or 'no reason given'}.",
        f"The ${buy['usd']:.2f} is still owed and is tried again within the hour. "
        "If it keeps being refused, give this project a different ticker, or "
        "check the Paper account on app.alpaca.markets.").as_json()


def status(cfg: Optional[Config] = None) -> Dict:
    """Everything the Projects tab shows about stocks. Never raises for a
    broker that is down: prices and the clock just read as unknown."""
    from monitor import projects
    cfg = cfg or config()
    setup = _setup_problem(cfg)
    out: Dict = {
        "configured": setup is None,
        # Keys are there, whether or not they are usable: lets the page tell
        # "never set up" (say nothing) from "set up wrong" (say what).
        "keys": bool(cfg.key and cfg.secret),
        "paper": True,
        "usd_per_hour": cfg.usd_per_hour,
        "min_buy_usd": cfg.min_buy_usd,
        "feed": FEED,
        "market": None,
        "sync_running": _sync_thread is not None and _sync_thread.is_alive(),
        "problem": setup.as_json() if setup is not None else None,
        "projects": [],
        "totals": {"earned_usd": 0.0, "invested_usd": 0.0, "queued_usd": 0.0,
                   "owed_usd": 0.0, "value_usd": 0.0, "gain_usd": 0.0},
    }

    rows = {p["id"]: p for p in projects.list_all(include_archived=True)}
    with _connect() as conn:
        buys_by: Dict[int, List[sqlite3.Row]] = {}
        for b in conn.execute("SELECT * FROM stock_buys ORDER BY id DESC").fetchall():
            buys_by.setdefault(b["project_id"], []).append(b)
        spans = {pid: _current(conn, pid) for pid in rows}
        # A deleted project's shares were still earned; they stay in the totals.
        ids = list(rows) + [pid for pid in buys_by if pid not in rows]
        earned_by = {pid: _earned(conn, pid, cfg) for pid in ids}

    held = sorted({b["symbol"] for bs in buys_by.values() for b in bs if b["status"] == FILLED}
                  | {s["symbol"] for s in spans.values() if s is not None})
    prices = _prices(held, cfg) if (held and setup is None) else {}
    if setup is None:
        out["market"] = _clock(cfg)

    priced_everything = True
    for pid in ids:
        project = rows.get(pid)
        span = spans.get(pid)
        buys = buys_by.get(pid, [])
        seconds, earned = earned_by[pid]
        invested = sum(b["qty"] * (b["price"] or 0) for b in buys if b["status"] == FILLED)
        queued = sum(b["usd"] for b in buys if b["status"] in (PENDING, QUEUED))
        spent = sum(b["usd"] for b in buys if b["status"] in _SPENT)

        holdings = []
        value: Optional[float] = 0.0
        for sym in sorted({b["symbol"] for b in buys if b["status"] == FILLED}):
            shares = sum(b["qty"] for b in buys if b["symbol"] == sym and b["status"] == FILLED)
            cost = sum(b["qty"] * (b["price"] or 0) for b in buys
                       if b["symbol"] == sym and b["status"] == FILLED)
            price = prices.get(sym, (None, None))[0]
            worth = None if price is None else shares * price
            value = None if (value is None or worth is None) else value + worth
            holdings.append({"symbol": sym, "shares": round(shares, 9),
                             "cost_usd": round(cost, 2), "price": price,
                             "value_usd": None if worth is None else round(worth, 2)})

        symbol = span["symbol"] if span is not None else None
        shown = symbol or (buys[0]["symbol"] if buys else None)
        price, price_at = prices.get(shown, (None, None)) if shown else (None, None)
        latest = next((b for b in buys if b["symbol"] == symbol), None) if symbol else None
        gain = None if value is None else value - invested

        out["projects"].append({
            "project_id": pid,
            "name": project["name"] if project else None,
            "project_status": project["status"] if project else "deleted",
            "symbol": symbol,
            "asset_name": span["asset_name"] if span is not None else None,
            "since": span["started_at"] if span is not None else None,
            "productive_hours": round(seconds / 3600.0, 2),
            "earned_usd": round(earned, 2),
            "invested_usd": round(invested, 2),
            "queued_usd": round(queued, 2),
            "owed_usd": round(max(0.0, earned - spent), 2),
            "held_symbol": shown,
            "shares": round(sum(h["shares"] for h in holdings if h["symbol"] == shown), 9),
            "price": price,
            "price_at": price_at,
            "value_usd": None if value is None else round(value, 2),
            "gain_usd": None if gain is None else round(gain, 2),
            "gain_pct": None if (gain is None or invested <= 0) else round(100.0 * gain / invested, 2),
            "holdings": holdings,
            "buys": [{"id": b["id"], "symbol": b["symbol"], "usd": round(b["usd"], 2),
                      "status": b["status"], "shares": round(b["qty"], 9), "price": b["price"],
                      "created_at": b["created_at"], "filled_at": b["filled_at"]}
                     for b in buys[:5]],
            "problem": _refusal_problem(latest)
            if (latest is not None and latest["status"] == REFUSED) else None,
        })
        t = out["totals"]
        t["earned_usd"] += earned
        t["invested_usd"] += invested
        t["queued_usd"] += queued
        t["owed_usd"] += max(0.0, earned - spent)
        if value is None:
            priced_everything = False
        else:
            t["value_usd"] += value

    t = out["totals"]
    if not priced_everything:
        t["value_usd"] = t["gain_usd"] = None
    else:
        t["gain_usd"] = t["value_usd"] - t["invested_usd"]
    out["totals"] = {k: (None if v is None else round(v, 2)) for k, v in t.items()}

    with _health_lock:
        unreachable = _health["unreachable"]
    if out["problem"] is None and unreachable:
        out["problem"] = Problem(
            "Couldn't reach Alpaca just now, so prices may be out of date and "
            "nothing new has been bought.",
            "Nothing is lost: what your hours earned is kept here and bought "
            "as soon as Alpaca answers again. Check this PC's internet "
            "connection if it lasts.").as_json()
    return out


# ------------------- background loop -------------------

_sync_lock = threading.Lock()
_sync_thread: Optional[threading.Thread] = None
_never = threading.Event()


def start_background_sync(stop_event: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """Settle every sync_sec until the stop event fires (or the process
    ends). Idempotent, and a no-op until the paper account is configured."""
    global _sync_thread
    with _sync_lock:
        if _sync_thread is not None and _sync_thread.is_alive():
            return _sync_thread
        cfg = config()
        if not enabled(cfg):
            return None
        stop = stop_event or _never

        def loop() -> None:
            while True:
                try:
                    settle()
                except Exception as e:
                    print(f"[stocks] settle failed ({e!r}); will retry next tick")
                if stop.wait(config().sync_sec):
                    break

        _sync_thread = threading.Thread(target=loop, daemon=True, name="stocks-sync")
        _sync_thread.start()
        print(f"[stocks] ${cfg.usd_per_hour:g} of paper stock per productive project hour, "
              f"settling every {cfg.sync_sec:.0f}s")
        return _sync_thread
