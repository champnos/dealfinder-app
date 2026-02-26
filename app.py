# DealFinder – eBay UK Scanner | Streamlit 1.53.1 compatible
# Full app: Scan + Products + Profiles + Auto-scan + Telegram (manual) + Delete
# FIX: must-include words now require ALL words (prevents Slim picking non-Slim)

from pathlib import Path
import os
import shutil
import re
import json
import time
import random
import copy
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DealFinder",
    layout="wide",
    initial_sidebar_state="expanded",
)


import requests
from dotenv import load_dotenv
from streamlit.components.v1 import html  # meta refresh (Streamlit-safe)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "dealfinder_config.json")

LOGO_PATH = os.path.join(SCRIPT_DIR, "assets", "dealfinder_logo.png")


# -----------------------------
# API call counter + throttling telemetry (shared by app + scanner)
# -----------------------------
API_COUNTER_PATH = os.path.join(SCRIPT_DIR, "dealfinder_api_calls.json")
SCAN_LOCK_PATH = os.path.join(SCRIPT_DIR, "dealfinder_scan.lock")
SCAN_LOCK_MAX_AGE_MIN = 30  # stale lock auto-clear
EBAY_DAILY_LIMIT = 5000  # typical default for Browse API; adjust if your account differs
FETCH_LIMIT = 200  # eBay Browse API max per request

def _today_ymd() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _next_midnight_local_iso() -> str:
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).date()
    nxt = datetime.combine(tomorrow, datetime.min.time())
    return nxt.isoformat(timespec="seconds")

def load_api_counter() -> Dict[str, Any]:
    base = {
        "date": _today_ymd(),
        "count": 0,
        "last_429_at": None,
        "last_429_retry_after": None,
        "last_429_url": None,
        "last_429_status": None,
        "resets_at_local": _next_midnight_local_iso(),
    }
    try:
        if os.path.exists(API_COUNTER_PATH):
            with open(API_COUNTER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        else:
            data = {}
    except Exception:
        return base

    if not isinstance(data, dict) or data.get("date") != _today_ymd():
        return base

    for k, v in base.items():
        data.setdefault(k, v)
    data["resets_at_local"] = _next_midnight_local_iso()
    try:
        data["count"] = int(data.get("count", 0) or 0)
    except Exception:
        data["count"] = 0
    return data

def _save_api_counter(data: Dict[str, Any]) -> None:
    try:
        tmp = API_COUNTER_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, API_COUNTER_PATH)
    except Exception:
        # fail open
        return

def bump_api_counter(n: int = 1) -> int:
    n = int(n or 0)
    data = load_api_counter()
    if n > 0:
        data["count"] = int(data.get("count", 0)) + n
        _save_api_counter(data)
    return int(data.get("count", 0))

def record_429(url: str, retry_after: Any, status: Any = 429) -> None:
    data = load_api_counter()
    data["last_429_at"] = _now_iso()
    data["last_429_retry_after"] = str(retry_after) if retry_after is not None else None
    data["last_429_url"] = str(url)[:300] if url else None
    data["last_429_status"] = status
    _save_api_counter(data)




SEEN_MAP_PATH = os.path.join(SCRIPT_DIR, "dealfinder_first_seen.json")

def load_first_seen_map() -> Dict[str, str]:
    try:
        if not os.path.exists(SEEN_MAP_PATH):
            return {}
        with open(SEEN_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_first_seen_map(m: Dict[str, str]) -> None:
    try:
        tmp_path = SEEN_MAP_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        os.replace(tmp_path, SEEN_MAP_PATH)
    except Exception:
        pass


UI_SETTINGS_PATH = os.path.join(SCRIPT_DIR, "dealfinder_ui_settings.json")

def load_ui_settings() -> Dict[str, Any]:
    try:
        if not os.path.exists(UI_SETTINGS_PATH):
            return {}
        with open(UI_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_ui_settings(data: Dict[str, Any]) -> None:
    tmp_path = UI_SETTINGS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, UI_SETTINGS_PATH)

def persist_ui_key(key: str) -> None:
    ui = load_ui_settings()
    ui[key] = st.session_state.get(key)
    save_ui_settings(ui)

# -----------------------------
# Shared scan lock (prevents manual scan overlapping with background scanner)
# -----------------------------
def scan_lock_exists() -> bool:
    if not os.path.exists(SCAN_LOCK_PATH):
        return False
    try:
        age_min = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(SCAN_LOCK_PATH))).total_seconds() / 60.0
        if age_min > SCAN_LOCK_MAX_AGE_MIN:
            os.remove(SCAN_LOCK_PATH)
            return False
    except Exception:
        return True
    return True

def acquire_scan_lock() -> bool:
    if scan_lock_exists():
        return False
    try:
        with open(SCAN_LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(datetime.now().isoformat(timespec="seconds"))
        return True
    except Exception:
        return False

def release_scan_lock() -> None:
    try:
        if os.path.exists(SCAN_LOCK_PATH):
            os.remove(SCAN_LOCK_PATH)
    except Exception:
        pass


EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


def ebay_get_json(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int = 30, max_retries: int = 6) -> Dict[str, Any]:
    """GET helper with basic backoff for eBay Browse API rate limiting (HTTP 429) and transient 5xx."""
    for attempt in range(max_retries):
        bump_api_counter(1)
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            if r.status_code == 429:
                try:
                    record_429(str(r.url), r.headers.get('Retry-After'), r.status_code)
                except Exception:
                    pass
                try:
                    if 'shown_429' not in st.session_state:
                        st.session_state['shown_429'] = True
                        st.warning(f"eBay rate limit hit (429). Retrying… (Retry-After: {r.headers.get('Retry-After')})")
                except Exception:
                    pass
            # Respect Retry-After if present; otherwise exponential backoff with small jitter.
            ra = r.headers.get("Retry-After")
            try:
                wait_s = int(ra) if ra is not None else None
            except Exception:
                wait_s = None
            if wait_s is None:
                wait_s = min(60, (2 ** attempt)) + random.random()
            time.sleep(wait_s)
            continue
        r.raise_for_status()
        return r.json()
    # Final attempt (raise the last response if available)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Config I/O
# -----------------------------
def save_config(cfg: Dict[str, Any]) -> None:
    """Save dealfinder_config.json safely.

    Guarantees:
    - Never drops sections like rare_items by accident (merges with on-disk config).
    - Creates timestamped backups before every overwrite.
    - Atomic write via .tmp + replace.
    """
    existing: Dict[str, Any] = {}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        if not isinstance(existing, dict):
            existing = {}
    except Exception:
        existing = {}

    merged = dict(existing)
    merged.update(cfg or {})

    merged.setdefault("consoles", {})
    merged.setdefault("profiles", {})
    merged.setdefault("rare_items", {})
    merged.setdefault("workshop_jobs", {})
    merged.setdefault("rare_finds", {})

    # Backup current config before overwrite
    try:
        if os.path.exists(CONFIG_PATH):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(SCRIPT_DIR, f"dealfinder_config.backup_{ts}.json")
            shutil.copy2(CONFIG_PATH, backup_path)
    except Exception:
        pass

    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp_path, CONFIG_PATH)


def save_all_config(consoles: Dict[str, Any], profiles: Dict[str, Any], rare_items: Dict[str, Any], workshop_jobs: Dict[str, Any] = None, rare_finds: Dict[str, Any] = None) -> None:
    """Save full config using current in-memory sections, via safe save_config."""
    cfg = load_config()
    cfg["consoles"] = consoles
    cfg["profiles"] = profiles
    cfg["rare_items"] = rare_items
    if workshop_jobs is not None:
        cfg["workshop_jobs"] = workshop_jobs
    if rare_finds is not None:
        cfg["rare_finds"] = rare_finds
    save_config(cfg)


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        cfg = {"consoles": {}, "profiles": {}, "rare_items": {}}
        save_config(cfg)
        return cfg

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if not isinstance(cfg, dict) or "consoles" not in cfg or "profiles" not in cfg:
            raise ValueError("Config missing required keys")

        cfg.setdefault("rare_items", {})

        # Backfill/normalize expected fields (consoles)
        for _, c in cfg["consoles"].items():
            c.setdefault("name", "Product")
            c.setdefault("search_base", "")
            c.setdefault("search_bases", [])
            c.setdefault("default_sell_price", 0.0)
            c.setdefault("fee_rate", 0.13)
            c.setdefault("ship_out", 0.0)
            c.setdefault("packaging", 0.0)
            c.setdefault("must_include_any", [])
            c.setdefault("exclude_words", [])
            c.setdefault("min_buy_total", 0.0)
            if not isinstance(c.get("must_include_any"), list):
                c["must_include_any"] = []
            if not isinstance(c.get("exclude_words"), list):
                c["exclude_words"] = []

            # Phase 2: multiple search bases per console (backwards compatible)
            if not isinstance(c.get("search_bases"), list):
                c["search_bases"] = []
            c["search_bases"] = [str(x).strip() for x in c.get("search_bases", []) if str(x).strip()]
            if not c["search_bases"] and str(c.get("search_base", "")).strip():
                c["search_bases"] = [str(c.get("search_base", "")).strip()]
            # Keep legacy search_base in sync with first base
            c["search_base"] = c["search_bases"][0] if c["search_bases"] else ""

        # Backfill/normalize expected fields (profiles)
        for _, p in cfg["profiles"].items():
            p.setdefault("console_id", "")
            p.setdefault("fault_query", "")
            p.setdefault("exclude_words", [])
            p.setdefault("parts", 0.0)
            p.setdefault("extra_costs", 0.0)
            p.setdefault("target_profit", 0.0)
            p.setdefault("sell_price_override", None)
            if not isinstance(p.get("exclude_words"), list):
                p["exclude_words"] = []

        # Backfill/normalize expected fields (rare items)
        for _, r in cfg["rare_items"].items():
            r.setdefault("name", "Item")
            r.setdefault("search_query", "")
            r.setdefault("sell_price", 0.0)
            r.setdefault("fee_rate", 0.13)
            r.setdefault("ship_out", 0.0)
            r.setdefault("packaging", 0.0)
            r.setdefault("extra_costs", 0.0)
            r.setdefault("target_profit", 0.0)
            r.setdefault("must_include_any", [])
            r.setdefault("exclude_words", [])
            r.setdefault("min_buy_total", 0.0)
            r.setdefault("category_id", "")
            if not isinstance(r.get("must_include_any"), list):
                r["must_include_any"] = []
            if not isinstance(r.get("exclude_words"), list):
                r["exclude_words"] = []

        # Backfill/normalize expected fields (workshop jobs)
        cfg.setdefault("workshop_jobs", {})
        for _, j in cfg["workshop_jobs"].items():
            j.setdefault("device_name", "Device")
            j.setdefault("console_id", "")
            j.setdefault("ebay_url", "")
            j.setdefault("buy_price", 0.0)
            j.setdefault("parts_cost", 0.0)
            j.setdefault("extra_costs", 0.0)
            j.setdefault("notes", "")
            j.setdefault("date_purchased", "")
            j.setdefault("status", "in_progress")
            j.setdefault("sell_price", 0.0)
            j.setdefault("date_sold", "")
            j.setdefault("fee_rate", 0.13)
            j.setdefault("ship_out", 0.0)
            j.setdefault("packaging", 0.0)
            j.setdefault("expected_sell_price", 0.0)

        # Backfill expected_sell_price, ship_out and packaging from linked console when not yet set
        for _, j in cfg["workshop_jobs"].items():
            if j.get("console_id"):
                linked = cfg["consoles"].get(j["console_id"], {})
                if not j.get("expected_sell_price"):
                    j["expected_sell_price"] = float(linked.get("default_sell_price") or 0.0)
                if not j.get("ship_out"):
                    j["ship_out"] = float(linked.get("ship_out") or 0.0)
                if not j.get("packaging"):
                    j["packaging"] = float(linked.get("packaging") or 0.0)

        # Backfill/normalize expected fields (rare finds)
        cfg.setdefault("rare_finds", {})
        for _, j in cfg["rare_finds"].items():
            j.setdefault("item_name", "Item")
            j.setdefault("ebay_buy_url", "")
            j.setdefault("buy_price", 0.0)
            j.setdefault("date_purchased", "")
            j.setdefault("condition", "")
            j.setdefault("expected_sell_price", 0.0)
            j.setdefault("fee_rate", 0.13)
            j.setdefault("ship_out", 0.0)
            j.setdefault("packaging", 0.0)
            j.setdefault("notes", "")
            j.setdefault("status", "in_stock")
            j.setdefault("sell_price", 0.0)
            j.setdefault("date_sold", "")

        return cfg
    except Exception:
        cfg = {"consoles": {}, "profiles": {}, "rare_items": {}}
        save_config(cfg)
        return cfg

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def next_console_id(consoles: Dict[str, Any], base: str) -> str:
    base = slugify(base)
    cand = base
    i = 2
    while cand in consoles:
        cand = f"{base}_{i}"
        i += 1
    return cand



def ebay_item_id(it: Dict[str, Any]) -> str:
    """Stable key for an eBay item. Prefer API itemId; fallback to /itm/<id> in URL; else normalized URL."""
    if it.get("itemId"):
        return str(it["itemId"])
    url = str(it.get("itemWebUrl") or "")
    m = re.search(r"/itm/(?:[^/]+/)?(\d{9,15})", url)
    if m:
        return m.group(1)
    return url.split("?", 1)[0].lower()

# -----------------------------
# Phase 2: Search bases editor
# -----------------------------

def render_search_bases_editor(state_key: str, bases: List[str]) -> List[str]:
    """
    Option 1 UI: stacked-but-clean.
    Each base is a full-width input, with a small right-aligned 🗑️ action beneath it.
    This avoids Streamlit column wrapping and looks consistent at any zoom.
    """
    if not isinstance(bases, list):
        bases = [str(bases)] if bases else [""]

    bases = [str(b) for b in bases]
    if not bases:
        bases = [""]

    if state_key not in st.session_state:
        st.session_state[state_key] = bases

    bases_state = st.session_state[state_key]

    for i in range(len(bases_state)):
        bases_state[i] = st.text_input(
            "",
            value=bases_state[i],
            key=f"{state_key}::input::{i}",
            placeholder="e.g. ps5, playstation 5",
            label_visibility="collapsed",
        )

        # Right-aligned remove action (separate row, intentional)
        action_cols = st.columns([20, 2], gap="small")
        with action_cols[1]:
            clicked = st.button(
                "🗑️",
                key=f"{state_key}::rm::{i}",
                help="Remove this base",
                use_container_width=True,
            )

        if clicked:
            if len(bases_state) > 1:
                bases_state.pop(i)
            else:
                bases_state[0] = ""
            st.session_state[state_key] = bases_state
            st.rerun()

        # Small separator for readability (tight)
        st.markdown('<div style="height: 4px"></div>', unsafe_allow_html=True)

    if st.button("+ Add search base", key=f"{state_key}::add"):
        bases_state.append("")
        st.session_state[state_key] = bases_state
        st.rerun()

    return [b.strip() for b in bases_state if b.strip()]


# -----------------------------
# Telegram (manual scan only)
# -----------------------------
def telegram_send(bot_token: str, chat_id: str, text: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        url = f"https://api.telegram.org/bot{(bot_token or '').strip()}/sendMessage"
        payload = {"chat_id": (chat_id or "").strip(), "text": text, "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=20)
        try:
            j = r.json()
        except Exception:
            j = {"raw": r.text}
        ok = (r.status_code == 200) and bool(j.get("ok", False))
        return ok, j
    except Exception as e:
        return False, {"exception": str(e)}


# -----------------------------
# eBay API helpers
# -----------------------------
def get_app_token(client_id: str, client_secret: str) -> str:
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(EBAY_OAUTH_URL, auth=auth, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def search_live_bin(token: str, marketplace_id: str, q: str, limit: int, category_id: str = "", max_price_gbp: float = 0.0) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace_id}
    params = {"q": q, "limit": min(limit, 200), "filter": "buyingOptions:{FIXED_PRICE}"}
    if category_id:
        params["filter"] += f",categoryIds:{{{category_id}}}"
    if max_price_gbp > 0:
        params["filter"] += f",price:[0..{max_price_gbp}],priceCurrency:GBP"
    params["sort"] = "price"
    data = ebay_get_json(BROWSE_SEARCH_URL, headers=headers, params=params, timeout=30)
    return data.get("itemSummaries", [])


def search_live_auctions_ending(token: str, marketplace_id: str, q: str, limit: int, ending_within_hours: int, category_id: str = "", max_price_gbp: float = 0.0) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace_id}
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=int(ending_within_hours))
    params = {
        "q": q,
        "limit": min(limit, 200),
        "filter": f"buyingOptions:{{AUCTION}},itemEndDate:[{iso_utc(now)}..{iso_utc(end)}]",
    }
    if category_id:
        params["filter"] += f",categoryIds:{{{category_id}}}"
    if max_price_gbp > 0:
        params["filter"] += f",price:[0..{max_price_gbp}],priceCurrency:GBP"
    params["sort"] = "price"
    data = ebay_get_json(BROWSE_SEARCH_URL, headers=headers, params=params, timeout=30)
    return data.get("itemSummaries", [])


def search_live_both(token: str, marketplace_id: str, q: str, limit: int, ending_within_hours: int, category_id: str = "", max_price_gbp: float = 0.0) -> List[Dict[str, Any]]:
    items = []
    items += search_live_bin(token, marketplace_id, q, limit, category_id, max_price_gbp)
    items += search_live_auctions_ending(token, marketplace_id, q, limit, ending_within_hours, category_id, max_price_gbp)

    seen = set()
    merged = []
    for it in items:
        url = it.get("itemWebUrl") or ""
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        merged.append(it)
    return merged


def _detect_make_offer(item: Dict[str, Any]) -> bool:
    opts = item.get("buyingOptions") or []
    if isinstance(opts, str):
        opts = [opts]
    opts_norm = {str(o).strip().upper() for o in opts}
    return ("BEST_OFFER" in opts_norm) or ("MAKE_OFFER" in opts_norm)


def _detect_item_mode_label(item: Dict[str, Any]) -> str:
    opts = item.get("buyingOptions") or []
    if isinstance(opts, str):
        opts = [opts]
    opts_norm = {str(o).strip().upper() for o in opts}
    if "AUCTION" in opts_norm:
        return "Auction"
    if "FIXED_PRICE" in opts_norm:
        return "Buy It Now"
    return "Unknown"




def _parse_end_date(item: Dict[str, Any]) -> Any:
    """Parse eBay itemEndDate to timezone-aware datetime (UTC). Returns None if missing/invalid."""
    try:
        s = item.get("itemEndDate") or item.get("endDate") or ""
        s = str(s).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _enforce_auction_window(df: pd.DataFrame, ending_within_hours: int) -> pd.DataFrame:
    """Keep non-auctions. Keep auctions only if ending within window. Fail-closed on missing end date."""
    if df is None or df.empty:
        return df
    if "mode" not in df.columns or "end_date" not in df.columns:
        return df
    try:
        now = datetime.now(timezone.utc)
        end_limit = now + timedelta(hours=int(ending_within_hours))
    except Exception:
        return df

    end_dt = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    is_auction = df["mode"].astype(str).str.lower().eq("auction")
    keep = (~is_auction) | (end_dt.notna() & (end_dt >= now) & (end_dt <= end_limit))
    return df.loc[keep].copy()

def live_to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        mode = _detect_item_mode_label(it)

        # Price handling:
        # - If mode is Auction (including Auction+BIN listings), show current bid where available.
        # - Otherwise show the fixed price.
        price_val = None
        try:
            if mode == "Auction":
                # Browse API may include currentBidPrice for auctions.
                price_val = (it.get("currentBidPrice") or {}).get("value")
                if price_val is None:
                    # Fallbacks seen in some payloads
                    price_val = (it.get("bidPrice") or {}).get("value")
            if price_val is None:
                price_val = (it.get("price") or {}).get("value")
        except Exception:
            price_val = (it.get("price") or {}).get("value")

        if price_val is None:
            continue

        ship_in = 0.0
        try:
            ship_in = float((it.get("shippingOptions") or [{}])[0].get("shippingCost", {}).get("value", 0.0))
        except Exception:
            ship_in = 0.0

        price = float(price_val)
        rows.append({
            "title": it.get("title") or "",
            "price": price,
            "shipping_in": ship_in,
            "buy_total": price + ship_in,
            "condition": it.get("condition") or "",
            "url": it.get("itemWebUrl") or "",
            "make_offer": _detect_make_offer(it),
            "seller_feedback": (it.get("seller") or {}).get("feedbackScore"),
            "seller_positive": (it.get("seller") or {}).get("feedbackPercentage"),
            "mode": mode,
            "end_date": it.get("itemEndDate") or "",
        })
    return pd.DataFrame(rows)


# -----------------------------
# Filters & maths
# -----------------------------
def net_after_fees(sell_price: float, fee_rate: float, ship_out: float, packaging: float) -> float:
    return sell_price * (1.0 - fee_rate) - ship_out - packaging


def max_buy_price(
    sell_price: float,
    fee_rate: float,
    ship_out: float,
    packaging: float,
    parts: float,
    extra_costs: float,
    target_profit: float
) -> float:
    return net_after_fees(sell_price, fee_rate, ship_out, packaging) - parts - extra_costs - target_profit


def title_must_include_filter(df: pd.DataFrame, must_all: List[str]) -> pd.DataFrame:
    """
    FIX: Include listing if ANY word in must_all appears in the title.
    This prevents PS5 Slim matching non-slim PS5 titles.
    """
    if df.empty or not must_all:
        return df
    must_all = [m.lower().strip() for m in must_all if str(m).strip()]
    t = df["title"].fillna("").str.lower()
    return df[t.apply(lambda s: any(m in s for m in must_all))]


def title_exclude_filter(df: pd.DataFrame, exclude_words: List[str]) -> pd.DataFrame:
    if df.empty or not exclude_words:
        return df
    exclude_words = [w.lower().strip() for w in exclude_words if str(w).strip()]
    t = df["title"].fillna("").str.lower()
    return df[t.apply(lambda s: not any(w in s for w in exclude_words))]


def min_buy_total_filter(df: pd.DataFrame, min_buy_total: float) -> pd.DataFrame:
    if df.empty or float(min_buy_total) <= 0:
        return df
    return df[df["buy_total"] >= float(min_buy_total)]


def below_max_buy_filter(df: pd.DataFrame, max_buy: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["buy_total"] <= float(max_buy)]


def condition_filter(df: pd.DataFrame, allowed_conditions: List[str]) -> pd.DataFrame:
    if df.empty or not allowed_conditions:
        return df
    allowed = {c.strip().lower() for c in allowed_conditions if str(c).strip()}
    cond = df["condition"].fillna("").str.lower()
    return df[cond.isin(allowed)]


# -----------------------------
# Active hours helper
# -----------------------------
def is_within_active_hours(now_local: datetime, start_hour: int, end_hour: int) -> bool:
    h = now_local.hour
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= h < end_hour
    return (h >= start_hour) or (h < end_hour)


# -----------------------------
# Scan engine
# -----------------------------
def run_scan(
    consoles: Dict[str, Any],
    profiles: Dict[str, Any],
    selected_profiles: List[str],
    offline_mode: bool,
    marketplace: str,
    scan_mode_label: str,
    scan_mode: str,
    ending_hours: int,
    only_below_max_buy: bool,
    apply_condition_filter: bool,
    selected_conditions: List[str],
) -> pd.DataFrame:
    if not selected_profiles:
        return pd.DataFrame()

    token = None
    if not offline_mode:
        client_id = os.getenv("EBAY_CLIENT_ID")
        client_secret = os.getenv("EBAY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("Missing EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in .env.")
        token = get_app_token(client_id, client_secret)

    # Step 1: collect unique console_ids from selected_profiles (preserve order)
    seen_cids: set = set()
    unique_console_ids = []
    for prof_name in selected_profiles:
        p = profiles.get(prof_name, {})
        cid = p.get("console_id")
        if cid in consoles and cid not in seen_cids:
            unique_console_ids.append(cid)
            seen_cids.add(cid)

    # Step 2: fetch and cache per (console_id, base)
    items_by_console: Dict[str, List[Dict[str, Any]]] = {}
    for console_id in unique_console_ids:
        c = consoles[console_id]
        if offline_mode:
            items_by_console[console_id] = []
            continue
        cap = float(c.get("default_sell_price", 0.0))
        bases = c.get("search_bases") or []
        if not isinstance(bases, list):
            bases = [str(bases)]
        bases = [str(b).strip() for b in bases if str(b).strip()]
        if not bases:
            bases = [str(c.get("search_base", "")).strip()]
        bases = [b for b in bases if b]
        items: List[Dict[str, Any]] = []
        for b in bases:
            if scan_mode == "bin":
                items += search_live_bin(token, marketplace, b, FETCH_LIMIT, max_price_gbp=cap)
            elif scan_mode == "auctions_ending":
                items += search_live_auctions_ending(token, marketplace, b, FETCH_LIMIT, ending_hours, max_price_gbp=cap)
            else:
                items += search_live_both(token, marketplace, b, FETCH_LIMIT, ending_hours, max_price_gbp=cap)
        items_by_console[console_id] = items

    # Step 3: per-profile loop, reuse cached items
    all_rows = []
    filter_log: List[Dict[str, Any]] = []
    raw_dfs: List[pd.DataFrame] = []

    for prof_name in selected_profiles:
        p = profiles.get(prof_name, {})
        console_id = p.get("console_id")
        if console_id not in consoles:
            continue
        c = consoles[console_id]

        fault_q = str(p.get("fault_query", "")).strip()

        sell_price = float(p.get("sell_price_override") or c.get("default_sell_price", 0.0))
        fee_rate = float(c.get("fee_rate", 0.13))
        ship_out = float(c.get("ship_out", 0.0))
        packaging = float(c.get("packaging", 0.0))

        parts = float(p.get("parts", 0.0))
        extra_costs = float(p.get("extra_costs", 0.0))
        target_profit = float(p.get("target_profit", 0.0))

        mx_buy = max_buy_price(sell_price, fee_rate, ship_out, packaging, parts, extra_costs, target_profit)
        net_before_buy = net_after_fees(sell_price, fee_rate, ship_out, packaging) - parts - extra_costs

        all_items = items_by_console.get(console_id, [])
        df_live = live_to_df(all_items)

        # Capture raw snapshot before any filtering
        df_raw_snapshot = df_live.copy()
        raw_dfs.append(df_raw_snapshot)
        filter_log.append({"Stage": "Raw API results", "Kept": len(df_live), "Dropped": 0})

        # 1. Enforce auction ending window
        prev = len(df_live)
        df_live = _enforce_auction_window(df_live, ending_hours)
        filter_log.append({"Stage": "After auction window", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        # BIN mode: exclude auctions (incl auction+BIN) after mode labeling
        if scan_mode == "bin" and "mode" in df_live.columns:
            df_live = df_live[df_live["mode"] != "Auction"].copy()

        if df_live.empty:
            continue

        # 2. Must-include words
        prev = len(df_live)
        df_live = title_must_include_filter(df_live, c.get("must_include_any", []))
        filter_log.append({"Stage": "After must-include words", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        # 3. Exclude words (console + profile) and fault_query as local title filter
        prev = len(df_live)
        df_live = title_exclude_filter(df_live, c.get("exclude_words", []))
        df_live = title_exclude_filter(df_live, p.get("exclude_words", []))
        if fault_q:
            df_live = title_must_include_filter(df_live, [fault_q])
        filter_log.append({"Stage": "After exclude words", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        # 4. Dedupe
        prev = len(df_live)
        if "url" in df_live.columns and not df_live.empty:
            def _iid_from_url(u: str) -> str:
                u = str(u or "")
                m = re.search(r"/itm/(?:[^/]+/)?(\d{9,15})", u)
                return m.group(1) if m else u.split("?", 1)[0].lower()
            df_live = df_live.copy()
            df_live["_iid"] = df_live["url"].apply(_iid_from_url)
            df_live = df_live.drop_duplicates(subset=["_iid"]).drop(columns=["_iid"])
        filter_log.append({"Stage": "After dedupe", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        # 5. min_buy_total
        prev = len(df_live)
        df_live = min_buy_total_filter(df_live, float(c.get("min_buy_total", 0.0)))
        filter_log.append({"Stage": "After min_buy_total", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        # 6. Condition filter
        prev = len(df_live)
        if apply_condition_filter:
            df_live = condition_filter(df_live, selected_conditions)
        filter_log.append({"Stage": "After condition filter", "Kept": len(df_live), "Dropped": prev - len(df_live)})

        if df_live.empty:
            continue

        # 7. below_max_buy (optional)
        if only_below_max_buy:
            prev = len(df_live)
            df_live = below_max_buy_filter(df_live, mx_buy)
            filter_log.append({"Stage": "After below_max_buy", "Kept": len(df_live), "Dropped": prev - len(df_live)})
            if df_live.empty:
                continue

        # 8. Compute max_buy, est_profit, good_buy
        df_live["console"] = c.get("name", console_id)
        df_live["profile"] = prof_name
        df_live["max_buy"] = float(mx_buy)
        df_live["est_profit"] = net_before_buy - df_live["buy_total"]
        df_live["good_buy"] = df_live["buy_total"] <= mx_buy

        all_rows.append(df_live)

    # Store debug data in session_state
    try:
        if raw_dfs:
            st.session_state["last_scan_raw_df"] = pd.concat(raw_dfs, ignore_index=True)
        else:
            st.session_state["last_scan_raw_df"] = pd.DataFrame()
        st.session_state["last_scan_filter_log"] = filter_log
    except Exception:
        pass

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    # Global dedupe (Option A): one row per eBay item across ALL profiles/bases
    if "url" in df.columns:
        def _iid_from_url(u: str) -> str:
            u = str(u or "")
            m = re.search(r"/itm/(?:[^/]+/)?(\\d{9,15})", u)
            return m.group(1) if m else u.split("?", 1)[0].lower()

        df["_iid"] = df["url"].apply(_iid_from_url)
        df = df.drop_duplicates(subset=["_iid"]).drop(columns=["_iid"])

    return df.sort_values(["good_buy", "est_profit"], ascending=[False, False])


def run_rare_scan(
    rare_items: Dict[str, Any],
    selected_items: List[str],
    offline_mode: bool,
    marketplace: str,
    scan_mode_label: str,
    scan_mode: str,
    ending_hours: int,
    only_below_max_buy: bool,
    apply_condition_filter: bool,
    selected_conditions: List[str],
) -> pd.DataFrame:
    if not selected_items:
        return pd.DataFrame()

    token = None
    if not offline_mode:
        client_id = os.getenv("EBAY_CLIENT_ID")
        client_secret = os.getenv("EBAY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("Missing EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in .env.")
        token = get_app_token(client_id, client_secret)

    all_rows = []

    for item_id in selected_items:
        r = rare_items.get(item_id, {})
        query = str(r.get("search_query", "")).strip() or str(r.get("name", "")).strip()
        if not query:
            continue

        sell_price = float(r.get("sell_price", 0.0))
        fee_rate = float(r.get("fee_rate", 0.13))
        ship_out = float(r.get("ship_out", 0.0))
        packaging = float(r.get("packaging", 0.0))
        extra_costs = float(r.get("extra_costs", 0.0))
        target_profit = float(r.get("target_profit", 0.0))
        category_id = str(r.get("category_id", "")).strip()

        mx_buy = net_after_fees(sell_price, fee_rate, ship_out, packaging) - extra_costs - target_profit
        net_before_buy = net_after_fees(sell_price, fee_rate, ship_out, packaging) - extra_costs

        if offline_mode:
            df_live = pd.DataFrame(columns=["title","price","shipping_in","buy_total","condition","url","make_offer","mode"])
        else:
            if scan_mode == "bin":
                items = search_live_bin(token, marketplace, query, FETCH_LIMIT, category_id)
            elif scan_mode == "auctions_ending":
                items = search_live_auctions_ending(token, marketplace, query, FETCH_LIMIT, ending_hours, category_id)
            else:
                items = search_live_both(token, marketplace, query, FETCH_LIMIT, ending_hours, category_id)

            df_live = live_to_df(items)
            df_live = _enforce_auction_window(df_live, ending_hours)

            # BIN mode: exclude auctions (incl auction+BIN) after mode labeling
            if scan_mode == "bin" and "mode" in df_live.columns:
                df_live = df_live[df_live["mode"] != "Auction"].copy()


        if df_live.empty:
            continue

        # Filters
        df_live = title_must_include_filter(df_live, r.get("must_include_any", []))
        df_live = title_exclude_filter(df_live, r.get("exclude_words", []))
        df_live = min_buy_total_filter(df_live, float(r.get("min_buy_total", 0.0)))

        if apply_condition_filter:
            df_live = condition_filter(df_live, selected_conditions)

        if df_live.empty:
            continue

        if only_below_max_buy:
            df_live = below_max_buy_filter(df_live, mx_buy)
            if df_live.empty:
                continue

        df_live["item"] = r.get("name", item_id)
        df_live["max_buy"] = float(mx_buy)
        df_live["est_profit"] = net_before_buy - df_live["buy_total"]
        df_live["good_buy"] = df_live["buy_total"] <= mx_buy

        all_rows.append(df_live)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    # Global dedupe (Option A): one row per eBay item across ALL profiles/bases
    if "url" in df.columns and not df.empty:
        def _iid_from_url(u: str) -> str:
            u = str(u or "")
            m = re.search(r"/itm/(?:[^/]+/)?(\d{9,15})", u)
            return m.group(1) if m else u.split("?", 1)[0].lower()

        df = df.copy()
        df["_iid"] = df["url"].apply(_iid_from_url)
        df = df.drop_duplicates(subset=["_iid"]).drop(columns=["_iid"])

    return df.sort_values(["good_buy", "est_profit"], ascending=[False, False])


# -----------------------------
# UI
# -----------------------------

st.markdown("""
<style>
/* Search base remove icon button */
button[kind="secondary"] {
  padding: 0.25rem 0.5rem !important;
  min-height: 2.4rem !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.trash-wrap {
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.trash-wrap button {
  height: 36px !important;
  width: 36px !important;
  padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Scan products (eBay UK)")

cfg = load_config()
consoles: Dict[str, Any] = cfg["consoles"]
profiles: Dict[str, Any] = cfg["profiles"]
rare_items: Dict[str, Any] = cfg.get("rare_items", {})
workshop_jobs: Dict[str, Any] = cfg.get("workshop_jobs", {})
rare_finds: Dict[str, Any] = cfg.get("rare_finds", {})

COMMON_CONDITIONS = [
    "New",
    "New other (see details)",
    "New with defects",
    "Manufacturer refurbished",
    "Seller refurbished",
    "Used",
    "For parts or not working",
    "Open box",
]

# Session state
if "notified_urls" not in st.session_state:
    st.session_state["notified_urls"] = set()
if "last_scan_df" not in st.session_state:
    st.session_state["last_scan_df"] = None
if "last_scan_ts" not in st.session_state:
    st.session_state["last_scan_ts"] = None



# Persist "first seen" timestamps per listing URL (persisted on disk)
if "first_seen_map" not in st.session_state:
    st.session_state["first_seen_map"] = load_first_seen_map()

def apply_first_seen(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable first_seen timestamp per listing and persist it across restarts."""
    if df is None or df.empty or "url" not in df.columns:
        return df

    m = st.session_state.get("first_seen_map", {})
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    changed = False
    for u in df["url"].fillna("").astype(str).tolist():
        if u and u not in m:
            m[u] = now_ts
            changed = True

    st.session_state["first_seen_map"] = m
    if changed:
        save_first_seen_map(m)

    df = df.copy()
    df["first_seen"] = df["url"].astype(str).map(lambda u: m.get(u, ""))
    return df

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    # DealFinder branding
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.caption("DealFinder")

    # API / Rate-limit (pro)
    st.subheader("API / Rate-limit")
    _api = load_api_counter()
    used = int(_api.get("count", 0) or 0)
    st.metric("Calls today", f"{used} / {EBAY_DAILY_LIMIT}")
    try:
        st.progress(min(1.0, used / float(EBAY_DAILY_LIMIT)))
    except Exception:
        pass
    st.caption(f"Counter date: {_api.get('date')} • Resets (local): {_api.get('resets_at_local')}")
    if _api.get("last_429_at"):
        ra = _api.get("last_429_retry_after")
        st.warning(f"Last 429: {_api.get('last_429_at')} (Retry-After: {ra})", icon="⚠️")
        if _api.get("last_429_url"):
            st.caption(f"Last 429 URL: {_api.get('last_429_url')}")


    st.header("Mode")
    offline_mode = st.toggle("Offline mode (no eBay keys needed)", value=False, key="offline_mode")

    st.divider()
    st.header("Scan settings")

    # Persist scan settings across restarts
    _ui_scan = load_ui_settings()
    if "_ui_loaded_scan_settings" not in st.session_state:
        if isinstance(_ui_scan.get("marketplace_id"), str) and _ui_scan.get("marketplace_id").strip():
            st.session_state["marketplace_id"] = _ui_scan.get("marketplace_id").strip()
        if _ui_scan.get("scan_mode_label") in ["Buy It Now only", "Auctions ending < X hours", "Both"]:
            st.session_state["scan_mode_label"] = _ui_scan.get("scan_mode_label")
        try:
            if _ui_scan.get("ending_hours") is not None:
                st.session_state["ending_hours"] = int(_ui_scan.get("ending_hours"))
        except Exception:
            pass
        if isinstance(_ui_scan.get("only_below_max_buy"), bool):
            st.session_state["only_below_max_buy"] = _ui_scan.get("only_below_max_buy")
        st.session_state["_ui_loaded_scan_settings"] = True

    marketplace = st.text_input(
        "Marketplace ID",
        value=str(st.session_state.get("marketplace_id", os.getenv("EBAY_MARKETPLACE_ID", "EBAY_GB"))),
        key="marketplace_id",
        on_change=lambda: persist_ui_key("marketplace_id"),
    )

    _scan_modes = ["Buy It Now only", "Auctions ending < X hours", "Both"]
    _saved_mode = st.session_state.get("scan_mode_label", "Both")
    _mode_index = _scan_modes.index(_saved_mode) if _saved_mode in _scan_modes else 2

    scan_mode_label = st.selectbox(
        "Scan mode",
        _scan_modes,
        index=_mode_index,
        key="scan_mode_label",
        on_change=lambda: persist_ui_key("scan_mode_label"),
    )
    scan_mode = {"Buy It Now only": "bin", "Auctions ending < X hours": "auctions_ending", "Both": "both"}[scan_mode_label]

    ending_hours = st.slider(
        "Auction ending window (hours)",
        1, 72,
        int(st.session_state.get("ending_hours", 24)),
        key="ending_hours",
        on_change=lambda: persist_ui_key("ending_hours"),
    )

    only_below_max_buy = st.toggle(
        "Only show listings at/below max-buy (incl. shipping in)",
        value=bool(st.session_state.get("only_below_max_buy", True)),
        key="only_below_max_buy",
        on_change=lambda: persist_ui_key("only_below_max_buy"),
    )

    st.divider()
    st.header("Condition filter")
    apply_condition_filter = st.toggle("Only show selected conditions", value=bool(load_ui_settings().get("apply_condition_filter", False)), key="apply_condition_filter", on_change=lambda: persist_ui_key("apply_condition_filter"))
    _ui_cond = load_ui_settings()
    _saved_conds = _ui_cond.get("selected_conditions")
    if not isinstance(_saved_conds, list):
        _saved_conds = ["Used", "For parts or not working"]
    else:
        _saved_conds = [c for c in _saved_conds if c in COMMON_CONDITIONS]
        if not _saved_conds:
            _saved_conds = ["Used", "For parts or not working"]
    selected_conditions = st.multiselect(
        "Conditions to include",
        options=COMMON_CONDITIONS,
        default=_saved_conds,
        help="Enable the toggle above to apply this filter.",
        key="selected_conditions",
        on_change=lambda: persist_ui_key("selected_conditions")
    )

    st.divider()
    st.header("Select profiles to scan")
    # Load persisted profile selection once per session, then let the widget own state.
    if "_ui_loaded_profile_sel" not in st.session_state:
        _ui = load_ui_settings()
        _saved = _ui.get("selected_profiles")
        if isinstance(_saved, list):
            st.session_state["selected_profiles"] = _saved
        else:
            # First run default: all profiles
            st.session_state["selected_profiles"] = list(profiles.keys())
        st.session_state["_ui_loaded_profile_sel"] = True

    # Keep selection valid if profiles are renamed/deleted
    _curp = st.session_state.get("selected_profiles")
    if not isinstance(_curp, list):
        _curp = list(profiles.keys())
    else:
        _curp = [p for p in _curp if p in profiles]
    st.session_state["selected_profiles"] = _curp

    selected_profiles = st.multiselect(
        "Profiles",
        options=list(profiles.keys()),
        key="selected_profiles",
        on_change=lambda: persist_ui_key("selected_profiles"),
    )




    st.divider()
    st.header("Select rare items to scan")
    # Load persisted rare selection once per session, then let the widget own state.
    if "_ui_loaded_rare_sel" not in st.session_state:
        _ui = load_ui_settings()
        _saved = _ui.get("selected_rare_items_sidebar")
        if isinstance(_saved, list):
            st.session_state["selected_rare_items_sidebar"] = _saved
        st.session_state["_ui_loaded_rare_sel"] = True

    # Ensure the selection is always a valid list of existing rare IDs
    _cur = st.session_state.get("selected_rare_items_sidebar")
    if not isinstance(_cur, list):
        _cur = list(rare_items.keys())
        st.session_state["selected_rare_items_sidebar"] = _cur
    else:
        _cur = [r for r in _cur if r in rare_items]
        if not _cur and list(rare_items.keys()):
            _cur = list(rare_items.keys())
        st.session_state["selected_rare_items_sidebar"] = _cur

    selected_rare_items = st.multiselect(
        "Rare items",
        options=list(rare_items.keys()),
        format_func=lambda x: rare_items.get(x, {}).get("name", x),
        key="selected_rare_items_sidebar",
        on_change=lambda: persist_ui_key("selected_rare_items_sidebar"),
    )

tabs = st.tabs(["Scan", "Products", "Profiles", "Rare Scan", "Rare Items", "Workshop", "Rare Finds"])

# -----------------------------
# Scan tab
# -----------------------------
with tabs[0]:
    st.subheader("Scan results")
    scan_running = scan_lock_exists()
    manual_run = st.button("Scan now", key="scan_now_btn", disabled=scan_running)
    if scan_running:
        st.warning("Scanner currently running — manual scan disabled.")

    should_scan = bool(manual_run) and (not scan_running)
    if should_scan:
        if not acquire_scan_lock():
            st.warning("Another scan is already running.")
        else:
            try:
                df_all = run_scan(
                    consoles=consoles,
                    profiles=profiles,
                    selected_profiles=selected_profiles,
                    offline_mode=offline_mode,
                    marketplace=marketplace,
                    scan_mode_label=scan_mode_label,
                    scan_mode=scan_mode,
                    ending_hours=ending_hours,
                    only_below_max_buy=only_below_max_buy,
                    apply_condition_filter=apply_condition_filter,
                    selected_conditions=selected_conditions,
                )
            except RuntimeError as e:
                st.error(str(e))
                df_all = pd.DataFrame()
            finally:
                release_scan_lock()

            df_all = apply_first_seen(df_all)
            st.session_state["last_scan_df"] = df_all
            st.session_state["last_scan_ts"] = datetime.now()

    df_all = st.session_state.get("last_scan_df")
    last_ts = st.session_state.get("last_scan_ts")
    if last_ts:
        st.caption(f"Last scan: {last_ts:%Y-%m-%d %H:%M:%S}")

    _raw_df = st.session_state.get("last_scan_raw_df")
    _filter_log = st.session_state.get("last_scan_filter_log")
    if _raw_df is not None:
        with st.expander("🔍 Raw results (before filtering)", expanded=False):
            if _raw_df.empty:
                st.warning("No listings returned from eBay API — check credentials, search terms, or API limits")
            else:
                _raw_cols = [col for col in ["title", "price", "condition", "mode", "end_date", "url"] if col in _raw_df.columns]
                st.dataframe(_raw_df[_raw_cols], use_container_width=True)
    if _filter_log:
        with st.expander("📊 Filter breakdown", expanded=False):
            st.dataframe(pd.DataFrame(_filter_log))

    if df_all is None:
        st.info("Run a scan to see results.")
    elif df_all.empty:
        st.warning("No results (or all filtered out).")
    else:
        good_count = int(df_all["good_buy"].sum())
        total_count = int(len(df_all))
        best_profit = float(df_all["est_profit"].max())
        worst_profit = float(df_all["est_profit"].min())
        avg_profit = float(df_all["est_profit"].mean())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Good buys", good_count)
        c2.metric("Results shown", total_count)
        c3.metric("Best est. profit", f"£{best_profit:.2f}")
        c4.metric("Worst est. profit", f"£{worst_profit:.2f}")
        c5.metric("Avg est. profit", f"£{avg_profit:.2f}")

        df_table = df_all.copy()
        df_table["deal"] = df_table["good_buy"].apply(lambda x: "✅" if bool(x) else "•")
        df_table["offer"] = df_table.get("make_offer", False).apply(lambda x: "✅" if bool(x) else "•")
        df_table["open"] = df_table["url"]

        df_table = df_table[
            ["deal", "offer", "mode", "console", "profile",
             "seller_feedback", "seller_positive",
             "first_seen",
             "price", "shipping_in", "buy_total", "max_buy", "est_profit",
             "condition", "title", "open"]
        ].copy()

        st.dataframe(
            df_table,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config={
                "deal": st.column_config.TextColumn("Deal", width="small"),
                "offer": st.column_config.TextColumn("Make offer", width="small"),
                "first_seen": st.column_config.TextColumn("First seen", width="small"),
                "seller_feedback": st.column_config.NumberColumn("Seller fb", width="small"),
                "seller_positive": st.column_config.NumberColumn("Seller %", width="small", format="%.1f"),
                "price": st.column_config.NumberColumn("Price", format="£%.2f"),
                "shipping_in": st.column_config.NumberColumn("Ship in", format="£%.2f"),
                "buy_total": st.column_config.NumberColumn("Buy total", format="£%.2f"),
                "max_buy": st.column_config.NumberColumn("Max buy", format="£%.2f"),
                "est_profit": st.column_config.NumberColumn("Est profit", format="£%.2f"),
                "open": st.column_config.LinkColumn("Open", display_text="Open"),
            },
        )

        csv_bytes = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="dealfinder_scan.csv",
            mime="text/csv",
            key="download_csv_btn"
        )

# -----------------------------
# Products tab
# -----------------------------
with tabs[1]:
    st.subheader("Products")

    with st.expander("➕ Add new product", expanded=False):
        new_name = st.text_input("New product name", value="", key="add_console_name")
        st.markdown("**Search bases** (multiple names sellers might use)")
        new_bases = render_search_bases_editor("add_console_search_bases", [""])
        new_search_base = (new_bases[0] if new_bases else "")
        new_must = st.text_input("Must-include words (ANY match – OR logic, comma separated)", value="", key="add_console_must")
        new_excl = st.text_input("Exclude words (comma separated)", value="", key="add_console_excl")

        colA, colB, colC = st.columns(3)
        with colA:
            new_sell = st.number_input("Default sell price (£)", value=0.0, step=5.0, format="%.2f", key="add_console_sell")
            new_fee = st.number_input("Fee rate", value=0.13, step=0.01, format="%.2f", key="add_console_fee")
        with colB:
            new_ship_out = st.number_input("Ship out (£)", value=0.0, step=0.50, format="%.2f", key="add_console_ship_out")
            new_pack = st.number_input("Packaging (£)", value=0.0, step=0.50, format="%.2f", key="add_console_pack")
        with colC:
            new_min_buy = st.number_input("Min buy_total (£)", value=0.0, step=5.0, format="%.2f", key="add_console_min_buy")

        if st.button("Create product", key="add_console_btn"):
            if not new_name.strip():
                st.error("Please enter a product name.")
            else:
                cid = next_console_id(consoles, new_name)
                consoles[cid] = {
                    "name": new_name.strip(),
                    "search_base": new_search_base.strip() or new_name.strip(),
                    "search_bases": [b for b in new_bases if b.strip()] or [new_search_base.strip() or new_name.strip()],
                    "default_sell_price": float(new_sell),
                    "fee_rate": float(new_fee),
                    "ship_out": float(new_ship_out),
                    "packaging": float(new_pack),
                    "must_include_any": [x.strip().lower() for x in new_must.split(",") if x.strip()],
                    "min_buy_total": float(new_min_buy),
                    "exclude_words": [x.strip().lower() for x in new_excl.split(",") if x.strip()],
                }
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success(f"Created console: {cid}")
                st.rerun()

    st.divider()

    if not consoles:
        st.warning("No products in config.")
    else:
        console_id = st.selectbox("Select product to edit", options=list(consoles.keys()), key="console_select")
        c = copy.deepcopy(consoles[console_id])
        k = f"console::{console_id}::"

        col1, col2 = st.columns(2)
        with col1:
            c["name"] = st.text_input("Product name", value=c.get("name", ""), key=k + "name")
            st.markdown("**Search bases**")
            existing_bases = c.get("search_bases") or ([c.get("search_base","")] if str(c.get("search_base","")).strip() else [""])
            c["search_bases"] = render_search_bases_editor(k + "search_bases", existing_bases)
            c["search_base"] = c["search_bases"][0] if c["search_bases"] else ""
            c["default_sell_price"] = st.number_input("Default sell price (£)", value=float(c.get("default_sell_price", 0.0)), step=5.0, format="%.2f", key=k + "sell")
            c["min_buy_total"] = st.number_input("Min buy_total (£) (item + shipping in)", value=float(c.get("min_buy_total", 0.0)), step=5.0, format="%.2f", key=k + "min_buy_total")
        with col2:
            c["fee_rate"] = st.number_input("Fee rate", value=float(c.get("fee_rate", 0.13)), step=0.01, format="%.2f", key=k + "fee_rate")
            c["ship_out"] = st.number_input("Shipping out (£)", value=float(c.get("ship_out", 0.0)), step=0.50, format="%.2f", key=k + "ship_out")
            c["packaging"] = st.number_input("Packaging (£)", value=float(c.get("packaging", 0.0)), step=0.50, format="%.2f", key=k + "packaging")

        must_str = ", ".join(c.get("must_include_any", []))
        must_str = st.text_input("Must-include words (ANY match – OR logic, comma separated)", value=must_str, key=k + "must_include")
        c["must_include_any"] = [x.strip().lower() for x in must_str.split(",") if x.strip()]

        ex_str = ", ".join(c.get("exclude_words", []))
        ex_str = st.text_input("Product exclude words (comma separated)", value=ex_str, key=k + "exclude")
        c["exclude_words"] = [x.strip().lower() for x in ex_str.split(",") if x.strip()]

        if st.button("Save product changes", key=k + "save"):
            consoles[console_id] = copy.deepcopy(c)
            save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
            st.success("Product saved.")

        # Delete Product (safe)
        st.divider()
        dependent_profiles = [
            pname for pname, p in profiles.items()
            if p.get("console_id") == console_id
        ]

        if dependent_profiles:
            st.warning(
                "Cannot delete this product because these profiles still use it:\n\n"
                + "\n".join(f"- {pname}" for pname in dependent_profiles)
            )
        else:
            confirm_del_console = st.checkbox(
                f"Confirm delete console: {console_id}",
                key=k + "confirm_delete_console"
            )
            if confirm_del_console:
                if st.button("❌ Delete product", key=k + "delete_console"):
                    del consoles[console_id]
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Product deleted.")
                    st.rerun()

# -----------------------------
# Profiles tab
# -----------------------------
with tabs[2]:
    st.subheader("Profiles (fault lanes)")

    if not consoles:
        st.warning("Create a product first, then you can add profiles.")
    else:
        with st.expander("➕ Add new profile", expanded=False):
            new_prof_name = st.text_input("Profile name (display)", value="", key="add_profile_name")
            new_prof_console = st.selectbox("Link to product", options=list(consoles.keys()), key="add_profile_console")
            new_fault_query = st.text_input("Fault query (extra words)", value="", key="add_profile_fault_query")
            new_prof_excl = st.text_input("Profile exclude words (comma separated)", value="", key="add_profile_excl")

            colA, colB, colC = st.columns(3)
            with colA:
                new_parts = st.number_input("Parts (£)", value=0.0, step=1.0, format="%.2f", key="add_profile_parts")
            with colB:
                new_extra = st.number_input("Extra costs (£)", value=0.0, step=5.0, format="%.2f", key="add_profile_extra")
            with colC:
                new_target = st.number_input("Target profit (£)", value=0.0, step=5.0, format="%.2f", key="add_profile_target")

            new_override = st.number_input("Sell price override (£) (0 = none)", value=0.0, step=5.0, format="%.2f", key="add_profile_override")

            if st.button("Create profile", key="add_profile_btn"):
                if not new_prof_name.strip():
                    st.error("Please enter a profile name.")
                elif new_prof_name in profiles:
                    st.error("That profile name already exists.")
                else:
                    profiles[new_prof_name] = {
                        "console_id": new_prof_console,
                        "fault_query": new_fault_query.strip(),
                        "exclude_words": [x.strip().lower() for x in new_prof_excl.split(",") if x.strip()],
                        "parts": float(new_parts),
                        "extra_costs": float(new_extra),
                        "target_profit": float(new_target),
                        "sell_price_override": None if float(new_override) <= 0 else float(new_override),
                    }
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Profile created.")
                    st.rerun()

        st.divider()

        if not profiles:
            st.warning("No profiles in config.")
        else:
            prof_name = st.selectbox("Select profile to edit", options=list(profiles.keys()), key="profile_select")
            p = copy.deepcopy(profiles[prof_name])
            kp = f"profile::{prof_name}::"

            col1, col2 = st.columns(2)
            with col1:
                p["console_id"] = st.selectbox(
                    "Linked console",
                    options=list(consoles.keys()),
                    index=list(consoles.keys()).index(p.get("console_id", list(consoles.keys())[0])),
                    key=kp + "console_id"
                )
                p["fault_query"] = st.text_input("Fault query", value=p.get("fault_query", ""), key=kp + "fault_query")

                prof_ex_str = ", ".join(p.get("exclude_words", []))
                prof_ex_str = st.text_input("Profile exclude words (comma separated)", value=prof_ex_str, key=kp + "exclude")
                p["exclude_words"] = [x.strip().lower() for x in prof_ex_str.split(",") if x.strip()]

                override_val = float(p.get("sell_price_override") or 0.0)
                override_val = st.number_input(
                    "Sell price override (£) (0 = use console default)",
                    value=override_val,
                    step=5.0,
                    format="%.2f",
                    key=kp + "sell_override"
                )
                p["sell_price_override"] = None if override_val <= 0 else override_val

            with col2:
                p["parts"] = st.number_input("Parts/consumables (£)", value=float(p.get("parts", 0.0)), step=1.0, format="%.2f", key=kp + "parts")
                p["extra_costs"] = st.number_input("Extra costs (£)", value=float(p.get("extra_costs", 0.0)), step=5.0, format="%.2f", key=kp + "extra_costs")
                p["target_profit"] = st.number_input("Target profit (£)", value=float(p.get("target_profit", 0.0)), step=5.0, format="%.2f", key=kp + "target_profit")

            if st.button("Save profile changes", key=kp + "save"):
                profiles[prof_name] = copy.deepcopy(p)
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success("Profile saved.")

            # Delete Profile
            st.divider()
            confirm_del_profile = st.checkbox(
                f"Confirm delete profile: {prof_name}",
                key=kp + "confirm_delete_profile"
            )
            if confirm_del_profile:
                if st.button("❌ Delete profile", key=kp + "delete_profile"):
                    del profiles[prof_name]
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Profile deleted.")
                    st.rerun()

# -----------------------------
# Rare Scan tab
# -----------------------------
with tabs[3]:
    st.subheader("Rare scan results")

    selected_rare = st.session_state.get("selected_rare_items_sidebar", [])
    if not selected_rare:
        st.info("Select rare items in the sidebar to scan.")
    else:
        run_rare = st.button("Scan rare items now", key="scan_rare_now")

        if run_rare:
            try:
                df_rare = run_rare_scan(
                    rare_items=rare_items,
                    selected_items=selected_rare,
                    offline_mode=offline_mode,
                    marketplace=marketplace,
                    scan_mode_label=scan_mode_label,
                    scan_mode=scan_mode,
                    ending_hours=ending_hours,
                    only_below_max_buy=only_below_max_buy,
                    apply_condition_filter=apply_condition_filter,
                    selected_conditions=selected_conditions,
                )
            except RuntimeError as e:
                st.error(str(e))
                df_rare = pd.DataFrame()

            df_rare = apply_first_seen(df_rare)

            st.session_state["last_scan_rare_df"] = df_rare
            st.session_state["last_scan_rare_ts"] = datetime.now()

        df_rare = st.session_state.get("last_scan_rare_df")
        ts_rare = st.session_state.get("last_scan_rare_ts")
        if ts_rare:
            st.caption(f"Last rare scan: {ts_rare:%Y-%m-%d %H:%M:%S}")

        if df_rare is None:
            st.info("Run a rare scan to see results.")
        elif df_rare.empty:
            st.warning("No rare results (or all filtered out).")
        else:
            # Pretty table like main scan
            df_table = df_rare.copy()
            df_table["deal"] = df_table["good_buy"].apply(lambda x: "✅" if bool(x) else "•")
            df_table["offer"] = df_table.get("make_offer", False).apply(lambda x: "✅" if bool(x) else "•")
            df_table["open"] = df_table["url"]

            df_table = df_table[
                ["deal", "offer", "mode", "item",
                 "first_seen",
                 "price", "shipping_in", "buy_total", "max_buy", "est_profit",
                 "condition", "title", "open"]
            ].copy()

            st.dataframe(
                df_table,
                use_container_width=True,
                height=600,
                hide_index=True,
                column_config={
                    "deal": st.column_config.TextColumn("Deal", width="small"),
                    "offer": st.column_config.TextColumn("Make offer", width="small"),
                "first_seen": st.column_config.TextColumn("First seen", width="small"),
                "seller_feedback": st.column_config.NumberColumn("Seller fb", width="small"),
                "seller_positive": st.column_config.NumberColumn("Seller %", width="small", format="%.1f"),
                    "price": st.column_config.NumberColumn("Price", format="£%.2f"),
                    "shipping_in": st.column_config.NumberColumn("Ship in", format="£%.2f"),
                    "buy_total": st.column_config.NumberColumn("Buy total", format="£%.2f"),
                    "max_buy": st.column_config.NumberColumn("Max buy", format="£%.2f"),
                    "est_profit": st.column_config.NumberColumn("Est profit", format="£%.2f"),
                    "open": st.column_config.LinkColumn("Open", display_text="Open"),
                },
            )

            csv_bytes = df_rare.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download rare results CSV",
                data=csv_bytes,
                file_name="dealfinder_rare_scan.csv",
                mime="text/csv",
                key="download_rare_csv_btn"
            )

# -----------------------------
# Rare Items tab
# -----------------------------
with tabs[4]:
    st.subheader("Rare items (collectibles)")

    # eBay Category Lookup tool
    with st.expander("🔍 Browse eBay Categories"):
        cat_query = st.text_input(
            "Search category name (e.g. nintendo 64, coins, video games)",
            key="cat_lookup_query"
        )
        if st.button("Look up", key="cat_lookup_btn"):
            if not cat_query.strip():
                st.warning("Please enter a search term.")
            elif offline_mode:
                st.info("Category lookup is not available in offline mode.")
            else:
                _cat_client_id = os.getenv("EBAY_CLIENT_ID", "").strip()
                _cat_client_secret = os.getenv("EBAY_CLIENT_SECRET", "").strip()
                if not _cat_client_id or not _cat_client_secret:
                    st.error("eBay credentials not configured.")
                else:
                    try:
                        if "ebay_category_tree" not in st.session_state:
                            _cat_token = get_app_token(_cat_client_id, _cat_client_secret)
                            _cat_r = requests.get(
                                "https://api.ebay.com/commerce/taxonomy/v1/category_tree/3",
                                headers={"Authorization": f"Bearer {_cat_token}"},
                                timeout=30,
                            )
                            _cat_r.raise_for_status()
                            st.session_state["ebay_category_tree"] = _cat_r.json()
                        _tree = st.session_state["ebay_category_tree"]

                        def _search_tree(root, kw, results):
                            stack = [(root, "")]
                            while stack:
                                node, parent_name = stack.pop()
                                if not isinstance(node, dict):
                                    continue
                                cat = node.get("category", {})
                                name = cat.get("categoryName", "")
                                cid = cat.get("categoryId", "")
                                if kw in name.lower():
                                    results.append({
                                        "Category Name": name,
                                        "Category ID": cid,
                                        "Parent Category": parent_name,
                                    })
                                for child in (node.get("childCategoryTreeNodes") or []):
                                    stack.append((child, name))

                        _results = []
                        _kw = cat_query.strip().lower()
                        _search_tree(_tree.get("rootCategoryNode", {}), _kw, _results)

                        if _results:
                            st.dataframe(pd.DataFrame(_results), use_container_width=True)
                        else:
                            st.info("No matching categories found.")
                    except Exception:
                        st.error("Could not fetch category tree from eBay. Check credentials.")

    # Add new rare item
    with st.expander("➕ Add new rare item", expanded=False):
        new_item_name = st.text_input("Item name", value="", key="add_rare_name")
        new_item_query = st.text_input("Search query (eBay)", value="", key="add_rare_query")
        new_item_category = st.text_input(
            "eBay Category ID (optional — restricts search to this category)",
            value="",
            key="add_rare_category",
            help="e.g. 139973 for Video Games. Leave blank to search all categories."
        )
        new_item_must = st.text_input("Must-include words (ANY match – OR logic, comma separated)", value="", key="add_rare_must")
        new_item_excl = st.text_input("Exclude words (comma separated)", value="", key="add_rare_excl")

        colA, colB, colC = st.columns(3)
        with colA:
            new_sell = st.number_input("Sell price (£)", value=0.0, step=5.0, format="%.2f", key="add_rare_sell")
            new_fee = st.number_input("Fee rate", value=0.13, step=0.01, format="%.2f", key="add_rare_fee")
        with colB:
            new_ship_out = st.number_input("Ship out (£)", value=0.0, step=0.50, format="%.2f", key="add_rare_ship_out")
            new_pack = st.number_input("Packaging (£)", value=0.0, step=0.50, format="%.2f", key="add_rare_pack")
        with colC:
            new_extra = st.number_input("Extra costs (£)", value=0.0, step=1.0, format="%.2f", key="add_rare_extra")
            new_target = st.number_input("Target profit (£)", value=0.0, step=5.0, format="%.2f", key="add_rare_target")

        new_min_buy = st.number_input("Min buy_total (£) (item + shipping in)", value=0.0, step=5.0, format="%.2f", key="add_rare_min_buy")

        if st.button("Create rare item", key="add_rare_btn"):
            if not new_item_name.strip():
                st.error("Please enter an item name.")
            else:
                rid = next_console_id(rare_items, new_item_name)  # reuse ID helper
                rare_items[rid] = {
                    "name": new_item_name.strip(),
                    "search_query": new_item_query.strip() or new_item_name.strip(),
                    "category_id": new_item_category.strip(),
                    "sell_price": float(new_sell),
                    "fee_rate": float(new_fee),
                    "ship_out": float(new_ship_out),
                    "packaging": float(new_pack),
                    "extra_costs": float(new_extra),
                    "target_profit": float(new_target),
                    "must_include_any": [x.strip().lower() for x in new_item_must.split(",") if x.strip()],
                    "exclude_words": [x.strip().lower() for x in new_item_excl.split(",") if x.strip()],
                    "min_buy_total": float(new_min_buy),
                }
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success(f"Created rare item: {rid}")
                st.rerun()

    st.divider()

    if not rare_items:
        st.info("No rare items yet. Add one above.")
    else:
        rid = st.selectbox(
            "Select rare item to edit",
            options=list(rare_items.keys()),
            format_func=lambda x: rare_items.get(x, {}).get("name", x),
            key="rare_select"
        )

        r = copy.deepcopy(rare_items[rid])
        kr = f"rare::{rid}::"

        col1, col2 = st.columns(2)
        with col1:
            r["name"] = st.text_input("Item name", value=r.get("name", ""), key=kr + "name")
            r["search_query"] = st.text_input("Search query", value=r.get("search_query", ""), key=kr + "search_query")
            r["category_id"] = st.text_input(
                "eBay Category ID (optional — restricts search to this category)",
                value=str(r.get("category_id", "")),
                key=kr + "category_id",
                help="e.g. 139973 for Video Games. Leave blank to search all categories."
            )
            r["sell_price"] = st.number_input("Sell price (£)", value=float(r.get("sell_price", 0.0)), step=5.0, format="%.2f", key=kr + "sell")
            r["min_buy_total"] = st.number_input("Min buy_total (£) (item + shipping in)", value=float(r.get("min_buy_total", 0.0)), step=5.0, format="%.2f", key=kr + "min_buy_total")
        with col2:
            r["fee_rate"] = st.number_input("Fee rate", value=float(r.get("fee_rate", 0.13)), step=0.01, format="%.2f", key=kr + "fee_rate")
            r["ship_out"] = st.number_input("Ship out (£)", value=float(r.get("ship_out", 0.0)), step=0.50, format="%.2f", key=kr + "ship_out")
            r["packaging"] = st.number_input("Packaging (£)", value=float(r.get("packaging", 0.0)), step=0.50, format="%.2f", key=kr + "packaging")
            r["extra_costs"] = st.number_input("Extra costs (£)", value=float(r.get("extra_costs", 0.0)), step=1.0, format="%.2f", key=kr + "extra_costs")
            r["target_profit"] = st.number_input("Target profit (£)", value=float(r.get("target_profit", 0.0)), step=5.0, format="%.2f", key=kr + "target_profit")

        must_str = ", ".join(r.get("must_include_any", []))
        must_str = st.text_input("Must-include words (ANY match – OR logic, comma separated)", value=must_str, key=kr + "must_include")
        r["must_include_any"] = [x.strip().lower() for x in must_str.split(",") if x.strip()]

        ex_str = ", ".join(r.get("exclude_words", []))
        ex_str = st.text_input("Exclude words (comma separated)", value=ex_str, key=kr + "exclude")
        r["exclude_words"] = [x.strip().lower() for x in ex_str.split(",") if x.strip()]

        if st.button("Save rare item changes", key=kr + "save"):
            rare_items[rid] = copy.deepcopy(r)
            save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
            st.success("Rare item saved.")

        # Delete rare item
        st.divider()
        confirm_del_rare = st.checkbox(f"Confirm delete rare item: {r.get('name', rid)}", key=kr + "confirm_delete")
        if confirm_del_rare:
            if st.button("❌ Delete rare item", key=kr + "delete"):
                del rare_items[rid]
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success("Rare item deleted.")
                st.rerun()

# -----------------------------
# Workshop tab
# -----------------------------
with tabs[5]:
    st.subheader("Workshop — Repair & Resell Tracker")

    # ── Section 1: Add new job ──────────────────────────────────────────────
    with st.expander("➕ Log new repair job", expanded=False):
        wk_device = st.text_input("Device name (required)", value="", key="wk_add_device")
        console_options = [""] + list(consoles.keys())
        wk_console_id = st.selectbox(
            "Link to product (optional)",
            options=console_options,
            format_func=lambda x: consoles[x]["name"] if x else "— none —",
            key="wk_add_console_id",
        )
        wk_ebay_url = st.text_input("eBay listing URL", value="", key="wk_add_ebay_url")
        wk_date_purchased = st.date_input(
            "Date purchased",
            value=datetime.now().date(),
            key="wk_add_date_purchased",
        )

        _add_linked_c = consoles.get(wk_console_id, {}) if wk_console_id else {}
        _add_default_sell = float(_add_linked_c.get("default_sell_price", 0.0))
        _add_default_fee = float(_add_linked_c.get("fee_rate", 0.13)) * 100.0 if _add_linked_c else 13.0
        _add_default_ship = float(_add_linked_c.get("ship_out", 0.0))
        _add_default_pack = float(_add_linked_c.get("packaging", 0.0))

        wk_expected_sell_price = st.number_input(
            "Expected sell price (£)",
            value=_add_default_sell,
            step=1.0, format="%.2f",
            key="wk_add_expected_sell_price",
            help="Pre-filled from linked product. Edit to override.",
        )

        colA, colB, colC = st.columns(3)
        with colA:
            wk_buy_price = st.number_input("Buy price (£)", value=0.0, step=1.0, format="%.2f", key="wk_add_buy_price")
            wk_parts_cost = st.number_input("Parts cost (£)", value=0.0, step=1.0, format="%.2f", key="wk_add_parts_cost")
        with colB:
            wk_extra_costs = st.number_input("Extra costs (£)", value=0.0, step=1.0, format="%.2f", key="wk_add_extra_costs")
            wk_fee_rate_pct = st.number_input("Fee rate (%)", value=_add_default_fee, step=0.5, format="%.1f", key="wk_add_fee_rate")
        with colC:
            wk_ship_out = st.number_input("Postage out (£)", value=_add_default_ship, step=0.50, format="%.2f", key="wk_add_ship_out")
            wk_packaging = st.number_input("Packaging (£)", value=_add_default_pack, step=0.50, format="%.2f", key="wk_add_packaging")

        _add_nr = float(wk_expected_sell_price) * (1 - float(wk_fee_rate_pct) / 100.0) - float(wk_ship_out) - float(wk_packaging)
        _add_tc = float(wk_buy_price) + float(wk_parts_cost) + float(wk_extra_costs)
        _add_ep = _add_nr - _add_tc
        st.info(f"💡 Est. profit at these costs: £{_add_ep:.2f}")

        wk_notes = st.text_area("Notes", value="", key="wk_add_notes")

        if st.button("Log job", key="wk_add_btn"):
            if not wk_device.strip():
                st.error("Please enter a device name.")
            else:
                job_id = next_console_id(workshop_jobs, wk_device.strip())
                workshop_jobs[job_id] = {
                    "device_name": wk_device.strip(),
                    "console_id": wk_console_id,
                    "ebay_url": wk_ebay_url.strip(),
                    "buy_price": float(wk_buy_price),
                    "parts_cost": float(wk_parts_cost),
                    "extra_costs": float(wk_extra_costs),
                    "notes": wk_notes.strip(),
                    "date_purchased": wk_date_purchased.strftime("%Y-%m-%d"),
                    "status": "in_progress",
                    "sell_price": 0.0,
                    "date_sold": "",
                    "fee_rate": float(wk_fee_rate_pct) / 100.0,
                    "ship_out": float(wk_ship_out),
                    "packaging": float(wk_packaging),
                    "expected_sell_price": float(wk_expected_sell_price),
                }
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success(f"Job logged: {job_id}")
                st.rerun()

    st.divider()

    # ── Section 2: Active jobs (in_progress) ───────────────────────────────
    st.markdown("### 🔧 Active Jobs (In Progress)")

    in_progress = {
        jid: j for jid, j in workshop_jobs.items() if j.get("status") == "in_progress"
    }

    if not in_progress:
        st.info("No active jobs. Log one above.")
    else:
        # Build display table
        ip_rows = []
        for jid, j in in_progress.items():
            linked = consoles.get(j.get("console_id", ""), {}).get("name", "—") if j.get("console_id") else "—"
            total_costs = float(j.get("buy_price", 0.0)) + float(j.get("parts_cost", 0.0)) + float(j.get("extra_costs", 0.0))
            _esp = float(j.get("expected_sell_price", 0.0))
            _fr = float(j.get("fee_rate", 0.13))
            _so = float(j.get("ship_out", 0.0))
            _pk = float(j.get("packaging", 0.0))
            _nr = _esp * (1.0 - _fr) - _so - _pk
            _ep = _nr - total_costs
            ip_rows.append({
                "job_id": jid,
                "device_name": j.get("device_name", ""),
                "product": linked,
                "date_purchased": j.get("date_purchased", ""),
                "buy_price": float(j.get("buy_price", 0.0)),
                "parts_cost": float(j.get("parts_cost", 0.0)),
                "extra_costs": float(j.get("extra_costs", 0.0)),
                "total_costs": total_costs,
                "expected_sell_price": _esp,
                "est_profit": _ep,
                "notes": str(j.get("notes", ""))[:40],
                "ebay_url": j.get("ebay_url", ""),
            })
        ip_rows.sort(key=lambda r: r["date_purchased"], reverse=True)
        df_ip = pd.DataFrame(ip_rows)

        st.dataframe(
            df_ip.drop(columns=["job_id"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "buy_price": st.column_config.NumberColumn("Buy price", format="£%.2f"),
                "parts_cost": st.column_config.NumberColumn("Parts cost", format="£%.2f"),
                "extra_costs": st.column_config.NumberColumn("Extra costs", format="£%.2f"),
                "total_costs": st.column_config.NumberColumn("Total costs", format="£%.2f"),
                "expected_sell_price": st.column_config.NumberColumn("Exp. sell", format="£%.2f"),
                "est_profit": st.column_config.NumberColumn("Est. profit", format="£%.2f"),
                "ebay_url": st.column_config.LinkColumn("eBay listing", display_text="Open"),
            },
        )

        st.markdown("#### Edit / Mark as Sold")
        for jid in [r["job_id"] for r in ip_rows]:
            j = copy.deepcopy(workshop_jobs[jid])
            k = f"workshop::{jid}::"
            label = j.get("device_name", jid)
            with st.expander(f"✏️ {label}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    j["device_name"] = st.text_input("Device name", value=j.get("device_name", ""), key=k + "device_name")
                    console_opts = [""] + list(consoles.keys())
                    cur_cid = j.get("console_id", "")
                    cid_idx = console_opts.index(cur_cid) if cur_cid in console_opts else 0
                    j["console_id"] = st.selectbox(
                        "Link to product",
                        options=console_opts,
                        index=cid_idx,
                        format_func=lambda x: consoles[x]["name"] if x else "— none —",
                        key=k + "console_id",
                    )
                    j["ebay_url"] = st.text_input("eBay listing URL", value=j.get("ebay_url", ""), key=k + "ebay_url")
                    _dp_str = j.get("date_purchased") or datetime.now().strftime("%Y-%m-%d")
                    try:
                        dp_default = datetime.strptime(_dp_str, "%Y-%m-%d").date()
                    except Exception:
                        dp_default = datetime.now().date()
                    dp = st.date_input("Date purchased", value=dp_default, key=k + "date_purchased")
                    j["date_purchased"] = dp.strftime("%Y-%m-%d")
                with col2:
                    _linked_c = consoles.get(j.get("console_id", ""), {}) if j.get("console_id") else {}
                    _console_sell = float(_linked_c.get("default_sell_price", 0.0))
                    _console_fee = float(_linked_c.get("fee_rate", 0.13))
                    _console_ship = float(_linked_c.get("ship_out", 0.0))
                    _console_pack = float(_linked_c.get("packaging", 0.0))
                    j["buy_price"] = st.number_input("Buy price (£)", value=float(j.get("buy_price", 0.0)), step=1.0, format="%.2f", key=k + "buy_price")
                    j["parts_cost"] = st.number_input("Parts cost (£)", value=float(j.get("parts_cost", 0.0)), step=1.0, format="%.2f", key=k + "parts_cost")
                    j["extra_costs"] = st.number_input("Extra costs (£)", value=float(j.get("extra_costs", 0.0)), step=1.0, format="%.2f", key=k + "extra_costs")
                    j["fee_rate"] = st.number_input("Fee rate (%)", value=float(j.get("fee_rate", _console_fee)) * 100.0, step=0.5, format="%.1f", key=k + "fee_rate") / 100.0
                    _stored_ship = float(j.get("ship_out", 0.0))
                    _stored_pack = float(j.get("packaging", 0.0))
                    _ship_default = _stored_ship if _stored_ship > 0 else _console_ship
                    _pack_default = _stored_pack if _stored_pack > 0 else _console_pack
                    _ship_key = k + "ship_out"
                    _pack_key = k + "packaging"
                    if _ship_default > 0 and float(st.session_state.get(_ship_key) or 0.0) <= 0:
                        st.session_state[_ship_key] = _ship_default
                    if _pack_default > 0 and float(st.session_state.get(_pack_key) or 0.0) <= 0:
                        st.session_state[_pack_key] = _pack_default
                    j["ship_out"] = st.number_input("Postage out (£)", value=_ship_default, step=0.50, format="%.2f", key=_ship_key)
                    j["packaging"] = st.number_input("Packaging (£)", value=_pack_default, step=0.50, format="%.2f", key=_pack_key)

                _stored_esp = float(j.get("expected_sell_price") or 0.0)
                _esp_default = _stored_esp if _stored_esp > 0 else float(_console_sell or 0.0)
                _esp_key = k + "expected_sell_price"
                if _esp_default > 0 and float(st.session_state.get(_esp_key) or 0.0) <= 0:
                    st.session_state[_esp_key] = _esp_default
                j["expected_sell_price"] = st.number_input(
                    "Expected sell price (£)",
                    value=_esp_default,
                    step=1.0, format="%.2f",
                    key=_esp_key,
                    help="Pre-filled from linked product. Edit to override.",
                )

                j["notes"] = st.text_area("Notes", value=j.get("notes", ""), key=k + "notes")

                _esp = float(j.get("expected_sell_price", 0.0))
                _fr = float(j.get("fee_rate", 0.13))
                _so = float(j.get("ship_out", 0.0))
                _pk = float(j.get("packaging", 0.0))
                _tc = float(j.get("buy_price", 0.0)) + float(j.get("parts_cost", 0.0)) + float(j.get("extra_costs", 0.0))
                _nr = _esp * (1.0 - _fr) - _so - _pk
                _ep = _nr - _tc
                if _esp > 0:
                    _colour = "🟢" if _ep >= 0 else "🔴"
                    st.info(f"{_colour} Prospective profit at £{_esp:.2f} sell: **£{_ep:.2f}** (net rev £{_nr:.2f} − costs £{_tc:.2f})")

                if st.button("💾 Save changes", key=k + "save"):
                    workshop_jobs[jid] = copy.deepcopy(j)
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Job updated.")
                    st.rerun()

                st.markdown("**Mark as Sold**")
                sell_col1, sell_col2 = st.columns(2)
                with sell_col1:
                    sold_price = st.number_input("Sell price (£)", value=float(j.get("expected_sell_price", 0.0)), step=1.0, format="%.2f", key=k + "sold_price")
                with sell_col2:
                    sold_date = st.date_input("Date sold", value=datetime.now().date(), key=k + "date_sold")

                if st.button("✅ Mark as Sold", key=k + "mark_sold"):
                    workshop_jobs[jid] = copy.deepcopy(j)
                    workshop_jobs[jid]["status"] = "sold"
                    workshop_jobs[jid]["sell_price"] = float(sold_price)
                    workshop_jobs[jid]["date_sold"] = sold_date.strftime("%Y-%m-%d")
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Job marked as sold.")
                    st.rerun()

                st.divider()
                confirm_del = st.checkbox(f"Confirm delete job: {label}", key=k + "confirm_delete")
                if confirm_del:
                    if st.button("🗑️ Delete job", key=k + "delete"):
                        del workshop_jobs[jid]
                        save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                        st.success("Job deleted.")
                        st.rerun()

    st.divider()

    # ── Section 3: Profit Sheet (Sold jobs) ────────────────────────────────
    st.markdown("### 📊 Profit Sheet (Sold Jobs)")

    sold_jobs = {jid: j for jid, j in workshop_jobs.items() if j.get("status") == "sold"}

    if not sold_jobs:
        st.info("No sold jobs yet.")
    else:
        sold_rows = []
        for jid, j in sold_jobs.items():
            linked = consoles.get(j.get("console_id", ""), {}).get("name", "—") if j.get("console_id") else "—"
            buy_price = float(j.get("buy_price", 0.0))
            parts_cost = float(j.get("parts_cost", 0.0))
            extra_costs = float(j.get("extra_costs", 0.0))
            total_costs = buy_price + parts_cost + extra_costs
            sell_price = float(j.get("sell_price", 0.0))
            fee_rate = float(j.get("fee_rate", 0.13))
            ship_out = float(j.get("ship_out", 0.0))
            packaging = float(j.get("packaging", 0.0))
            fees = sell_price * fee_rate
            net_revenue = sell_price * (1.0 - fee_rate) - ship_out - packaging
            profit = net_revenue - total_costs
            sold_rows.append({
                "date_sold": j.get("date_sold", ""),
                "device_name": j.get("device_name", ""),
                "product": linked,
                "buy_price": buy_price,
                "parts_cost": parts_cost,
                "extra_costs": extra_costs,
                "total_costs": total_costs,
                "sell_price": sell_price,
                "fees": fees,
                "ship_out": ship_out,
                "packaging": packaging,
                "net_revenue": net_revenue,
                "profit": profit,
                "notes": j.get("notes", ""),
                "ebay_url": j.get("ebay_url", ""),
            })
        sold_rows.sort(key=lambda r: r["date_sold"], reverse=True)
        df_sold = pd.DataFrame(sold_rows)

        # Aggregate summary
        total_jobs = len(df_sold)
        total_revenue = float(df_sold["net_revenue"].sum())
        total_costs_sum = float(df_sold["total_costs"].sum())
        total_profit = float(df_sold["profit"].sum())

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total jobs sold", total_jobs)
        sc2.metric("Total net revenue", f"£{total_revenue:.2f}")
        sc3.metric("Total costs", f"£{total_costs_sum:.2f}")
        sc4.metric("Total profit", f"£{total_profit:.2f}")

        st.dataframe(
            df_sold,
            use_container_width=True,
            hide_index=True,
            column_config={
                "buy_price": st.column_config.NumberColumn("Buy price", format="£%.2f"),
                "parts_cost": st.column_config.NumberColumn("Parts cost", format="£%.2f"),
                "extra_costs": st.column_config.NumberColumn("Extra costs", format="£%.2f"),
                "total_costs": st.column_config.NumberColumn("Total costs", format="£%.2f"),
                "sell_price": st.column_config.NumberColumn("Sell price", format="£%.2f"),
                "fees": st.column_config.NumberColumn("Fees", format="£%.2f"),
                "ship_out": st.column_config.NumberColumn("Ship out", format="£%.2f"),
                "packaging": st.column_config.NumberColumn("Packaging", format="£%.2f"),
                "net_revenue": st.column_config.NumberColumn("Net revenue", format="£%.2f"),
                "profit": st.column_config.NumberColumn("Profit", format="£%.2f"),
                "ebay_url": st.column_config.LinkColumn("eBay listing", display_text="Open"),
            },
        )

        csv_bytes_workshop = df_sold.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download profit sheet CSV",
            data=csv_bytes_workshop,
            file_name="dealfinder_workshop.csv",
            mime="text/csv",
            key="download_workshop_csv_btn",
        )

        # Per-product summary
        st.markdown("#### Per-product summary")
        df_by_product = (
            df_sold.groupby("product")
            .agg(
                jobs=("profit", "count"),
                total_costs=("total_costs", "sum"),
                total_net_revenue=("net_revenue", "sum"),
                total_profit=("profit", "sum"),
            )
            .reset_index()
            .sort_values("total_profit", ascending=False)
        )
        st.dataframe(
            df_by_product,
            use_container_width=True,
            hide_index=True,
            column_config={
                "total_costs": st.column_config.NumberColumn("Total costs", format="£%.2f"),
                "total_net_revenue": st.column_config.NumberColumn("Total net revenue", format="£%.2f"),
                "total_profit": st.column_config.NumberColumn("Total profit", format="£%.2f"),
            },
        )

with tabs[6]:
    st.subheader("Rare Finds — Buy Cheap, Relist at Market Value")

    # ── Section 1: Log new rare find ──────────────────────────────────────
    with st.expander("➕ Log new rare find", expanded=False):
        rf_item_name = st.text_input("Item name (required)", value="", key="rf_add_item_name")
        rf_ebay_buy_url = st.text_input("eBay listing URL (bought from)", value="", key="rf_add_ebay_buy_url")
        rf_date_purchased = st.date_input(
            "Date purchased",
            value=datetime.now().date(),
            key="rf_add_date_purchased",
        )
        rf_condition = st.selectbox(
            "Condition",
            options=["CIB", "Loose", "Sealed", "Box Only", "Manual Only", "Other"],
            key="rf_add_condition",
        )
        rf_expected_sell_price = st.number_input(
            "Expected sell price (£)",
            value=0.0,
            step=1.0, format="%.2f",
            key="rf_add_expected_sell_price",
        )

        rfcA, rfcB, rfcC = st.columns(3)
        with rfcA:
            rf_fee_rate_pct = st.number_input("Fee rate (%)", value=13.0, step=0.5, format="%.1f", key="rf_add_fee_rate")
        with rfcB:
            rf_ship_out = st.number_input("Ship out (£)", value=0.0, step=0.50, format="%.2f", key="rf_add_ship_out")
        with rfcC:
            rf_packaging = st.number_input("Packaging (£)", value=0.0, step=0.50, format="%.2f", key="rf_add_packaging")

        rf_buy_price = st.number_input("Buy price (£)", value=0.0, step=1.0, format="%.2f", key="rf_add_buy_price")

        _rf_add_ep = float(rf_expected_sell_price) * (1.0 - float(rf_fee_rate_pct) / 100.0) - float(rf_ship_out) - float(rf_packaging) - float(rf_buy_price)
        st.info(f"💡 Est. profit: £{_rf_add_ep:.2f}")

        rf_notes = st.text_area("Notes", value="", key="rf_add_notes")

        if st.button("Log find", key="rf_add_btn"):
            if not rf_item_name.strip():
                st.error("Please enter an item name.")
            else:
                rf_find_id = next_console_id(rare_finds, rf_item_name.strip())
                rare_finds[rf_find_id] = {
                    "item_name": rf_item_name.strip(),
                    "ebay_buy_url": rf_ebay_buy_url.strip(),
                    "buy_price": float(rf_buy_price),
                    "date_purchased": rf_date_purchased.strftime("%Y-%m-%d"),
                    "condition": rf_condition,
                    "expected_sell_price": float(rf_expected_sell_price),
                    "fee_rate": float(rf_fee_rate_pct) / 100.0,
                    "ship_out": float(rf_ship_out),
                    "packaging": float(rf_packaging),
                    "notes": rf_notes.strip(),
                    "status": "in_stock",
                    "sell_price": 0.0,
                    "date_sold": "",
                }
                save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                st.success(f"Find logged: {rf_find_id}")
                st.rerun()

    st.divider()

    # ── Section 2: Active finds (in_stock + listed) ───────────────────────
    st.markdown("### 📦 Active Finds (In Stock / Listed)")

    active_finds = {
        fid: f for fid, f in rare_finds.items() if f.get("status") in ("in_stock", "listed")
    }

    if not active_finds:
        st.info("No active finds. Log one above.")
    else:
        af_rows = []
        for fid, f in active_finds.items():
            _esp = float(f.get("expected_sell_price", 0.0))
            _fr = float(f.get("fee_rate", 0.13))
            _so = float(f.get("ship_out", 0.0))
            _pk = float(f.get("packaging", 0.0))
            _bp = float(f.get("buy_price", 0.0))
            _ep = _esp * (1.0 - _fr) - _so - _pk - _bp
            af_rows.append({
                "find_id": fid,
                "item_name": f.get("item_name", ""),
                "condition": f.get("condition", ""),
                "date_purchased": f.get("date_purchased", ""),
                "buy_price": _bp,
                "expected_sell_price": _esp,
                "est_profit": _ep,
                "status": f.get("status", ""),
                "notes": str(f.get("notes", ""))[:40],
                "ebay_buy_url": f.get("ebay_buy_url", ""),
            })
        af_rows.sort(key=lambda r: r["date_purchased"], reverse=True)
        df_af = pd.DataFrame(af_rows)

        st.dataframe(
            df_af.drop(columns=["find_id"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "buy_price": st.column_config.NumberColumn("Buy price", format="£%.2f"),
                "expected_sell_price": st.column_config.NumberColumn("Exp. sell", format="£%.2f"),
                "est_profit": st.column_config.NumberColumn("Est. profit", format="£%.2f"),
                "ebay_buy_url": st.column_config.LinkColumn("eBay listing", display_text="Open"),
            },
        )

        st.markdown("#### Edit / Mark as Sold")
        for fid in [r["find_id"] for r in af_rows]:
            f = copy.deepcopy(rare_finds[fid])
            rk = f"rarefind::{fid}::"
            label = f.get("item_name", fid)
            with st.expander(f"✏️ {label}", expanded=False):
                f["item_name"] = st.text_input("Item name", value=f.get("item_name", ""), key=rk + "item_name")
                f["ebay_buy_url"] = st.text_input("eBay listing URL", value=f.get("ebay_buy_url", ""), key=rk + "ebay_buy_url")
                _rf_dp_str = f.get("date_purchased") or datetime.now().strftime("%Y-%m-%d")
                try:
                    _rf_dp_default = datetime.strptime(_rf_dp_str, "%Y-%m-%d").date()
                except Exception:
                    _rf_dp_default = datetime.now().date()
                rf_dp = st.date_input("Date purchased", value=_rf_dp_default, key=rk + "date_purchased")
                f["date_purchased"] = rf_dp.strftime("%Y-%m-%d")

                _cond_options = ["CIB", "Loose", "Sealed", "Box Only", "Manual Only", "Other"]
                _cond_cur = f.get("condition", "CIB")
                _cond_idx = _cond_options.index(_cond_cur) if _cond_cur in _cond_options else 0
                f["condition"] = st.selectbox("Condition", options=_cond_options, index=_cond_idx, key=rk + "condition")

                f["expected_sell_price"] = st.number_input(
                    "Expected sell price (£)",
                    value=float(f.get("expected_sell_price", 0.0)),
                    step=1.0, format="%.2f",
                    key=rk + "expected_sell_price",
                )

                rfc1, rfc2, rfc3 = st.columns(3)
                with rfc1:
                    f["fee_rate"] = st.number_input("Fee rate (%)", value=float(f.get("fee_rate", 0.13)) * 100.0, step=0.5, format="%.1f", key=rk + "fee_rate") / 100.0
                with rfc2:
                    f["ship_out"] = st.number_input("Ship out (£)", value=float(f.get("ship_out", 0.0)), step=0.50, format="%.2f", key=rk + "ship_out")
                with rfc3:
                    f["packaging"] = st.number_input("Packaging (£)", value=float(f.get("packaging", 0.0)), step=0.50, format="%.2f", key=rk + "packaging")

                f["buy_price"] = st.number_input("Buy price (£)", value=float(f.get("buy_price", 0.0)), step=1.0, format="%.2f", key=rk + "buy_price")

                _status_options = ["in_stock", "listed", "sold"]
                _status_cur = f.get("status", "in_stock")
                _status_idx = _status_options.index(_status_cur) if _status_cur in _status_options else 0
                f["status"] = st.selectbox("Status", options=_status_options, index=_status_idx, key=rk + "status")

                _rf_esp = float(f.get("expected_sell_price", 0.0))
                _rf_fr = float(f.get("fee_rate", 0.13))
                _rf_so = float(f.get("ship_out", 0.0))
                _rf_pk = float(f.get("packaging", 0.0))
                _rf_bp = float(f.get("buy_price", 0.0))
                _rf_ep = _rf_esp * (1.0 - _rf_fr) - _rf_so - _rf_pk - _rf_bp
                st.info(f"💡 Est. profit: £{_rf_ep:.2f}")

                f["notes"] = st.text_area("Notes", value=f.get("notes", ""), key=rk + "notes")

                if st.button("💾 Save changes", key=rk + "save"):
                    rare_finds[fid] = copy.deepcopy(f)
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Find updated.")
                    st.rerun()

                st.markdown("**✅ Mark as Sold**")
                rf_sell_col1, rf_sell_col2 = st.columns(2)
                with rf_sell_col1:
                    rf_sold_price = st.number_input("Sell price (£)", value=float(f.get("expected_sell_price", 0.0)), step=1.0, format="%.2f", key=rk + "sold_price")
                with rf_sell_col2:
                    rf_sold_date = st.date_input("Date sold", value=datetime.now().date(), key=rk + "date_sold")

                if st.button("✅ Mark as Sold", key=rk + "mark_sold"):
                    rare_finds[fid] = copy.deepcopy(f)
                    rare_finds[fid]["status"] = "sold"
                    rare_finds[fid]["sell_price"] = float(rf_sold_price)
                    rare_finds[fid]["date_sold"] = rf_sold_date.strftime("%Y-%m-%d")
                    save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                    st.success("Find marked as sold.")
                    st.rerun()

                st.divider()
                rf_confirm_del = st.checkbox(f"Confirm delete find: {label}", key=rk + "confirm_delete")
                if rf_confirm_del:
                    if st.button("🗑️ Delete find", key=rk + "delete"):
                        del rare_finds[fid]
                        save_all_config(consoles, profiles, rare_items, workshop_jobs=workshop_jobs, rare_finds=rare_finds)
                        st.success("Find deleted.")
                        st.rerun()

    st.divider()

    # ── Section 3: Profit Sheet (sold finds) ─────────────────────────────
    st.markdown("### 📊 Profit Sheet (Sold Finds)")

    sold_finds = {fid: f for fid, f in rare_finds.items() if f.get("status") == "sold"}

    if not sold_finds:
        st.info("No sold finds yet.")
    else:
        sf_rows = []
        for fid, f in sold_finds.items():
            buy_price = float(f.get("buy_price", 0.0))
            sell_price = float(f.get("sell_price", 0.0))
            fee_rate = float(f.get("fee_rate", 0.13))
            ship_out = float(f.get("ship_out", 0.0))
            packaging = float(f.get("packaging", 0.0))
            net_revenue = sell_price * (1.0 - fee_rate) - ship_out - packaging
            actual_profit = net_revenue - buy_price
            sf_rows.append({
                "date_purchased": f.get("date_purchased", ""),
                "item_name": f.get("item_name", ""),
                "condition": f.get("condition", ""),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "net_revenue": net_revenue,
                "actual_profit": actual_profit,
                "date_sold": f.get("date_sold", ""),
                "notes": f.get("notes", ""),
            })
        sf_rows.sort(key=lambda r: r["date_sold"], reverse=True)
        df_sf = pd.DataFrame(sf_rows)

        total_sf_sold = len(df_sf)
        total_sf_profit = float(df_sf["actual_profit"].sum())
        avg_sf_profit = total_sf_profit / total_sf_sold if total_sf_sold > 0 else 0.0

        sfc1, sfc2, sfc3 = st.columns(3)
        sfc1.metric("Total sold", total_sf_sold)
        sfc2.metric("Total profit", f"£{total_sf_profit:.2f}")
        sfc3.metric("Avg profit per find", f"£{avg_sf_profit:.2f}")

        st.dataframe(
            df_sf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "buy_price": st.column_config.NumberColumn("Buy price", format="£%.2f"),
                "sell_price": st.column_config.NumberColumn("Sell price", format="£%.2f"),
                "net_revenue": st.column_config.NumberColumn("Net revenue", format="£%.2f"),
                "actual_profit": st.column_config.NumberColumn("Actual profit", format="£%.2f"),
            },
        )
