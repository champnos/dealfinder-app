
import os
import json
import time
import atexit
import re
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv

# -----------------------------
# Paths / env
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(SCRIPT_DIR, "dealfinder_scan.lock")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "dealfinder_config.json")
NOTIFIED_PATH = os.path.join(SCRIPT_DIR, "notified_item_ids.json")
API_COUNTER_PATH = os.path.join(SCRIPT_DIR, "dealfinder_api_calls.json")

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

load_dotenv()

def ebay_item_id(it: Dict[str, Any]) -> str:
    """Stable key for an eBay item. Prefer API itemId; fallback to /itm/<id> in URL; else normalized URL."""
    if it.get("itemId"):
        return str(it["itemId"])
    url = str(it.get("itemWebUrl") or "")
    m = re.search(r"/itm/(?:[^/]+/)?(\d{9,15})", url)
    if m:
        return m.group(1)
    return url.split("?", 1)[0].lower()

def _norm_words_csv(val: Any) -> List[str]:
    """Normalize comma-separated string or list into lowercase words."""
    if val is None:
        return []
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(',')]
        return [p.lower() for p in parts if p]
    if isinstance(val, list):
        return [str(p).strip().lower() for p in val if str(p).strip()]
    s = str(val).strip()
    return [s.lower()] if s else []

def _passes_word_filters(title: str, must_any: List[str], exclude_any: List[str]) -> bool:
    s = str(title or '').lower()
    if exclude_any and any(w in s for w in exclude_any):
        return False
    if must_any and not any(w in s for w in must_any):
        return False
    return True


# -----------------------------
# File helpers
# -----------------------------
def load_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_atomic(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def load_config() -> Dict[str, Any]:
    cfg = load_json(CONFIG_PATH, None)
    if not isinstance(cfg, dict) or ("products" not in cfg and "consoles" not in cfg) or "profiles" not in cfg:
        raise RuntimeError("Missing/invalid dealfinder_config.json. Run the Streamlit app once to create it.")
    return cfg

def load_notified_ids() -> set:
    data = load_json(NOTIFIED_PATH, [])
    if isinstance(data, list):
        return set(str(x) for x in data)
    return set()

def save_notified_ids(s: set) -> None:
    save_json_atomic(NOTIFIED_PATH, sorted(list(s)))


# -----------------------------
# Shared API calls counter (app + scanner)
# -----------------------------
def _api_counter_load() -> Dict[str, Any]:
    data = load_json(API_COUNTER_PATH, None)
    if isinstance(data, dict):
        return data
    return {
        "date": None,
        "count": 0,
        "last_429_at": None,
        "last_429_retry_after": None,
        "last_429_url": None,
        "last_429_status": None,
    }

def _api_counter_reset_if_new_day(data: Dict[str, Any]) -> Dict[str, Any]:
    today = datetime.now().date().isoformat()
    if data.get("date") != today:
        data["date"] = today
        data["count"] = 0
        data["last_429_at"] = None
        data["last_429_retry_after"] = None
        data["last_429_url"] = None
        data["last_429_status"] = None
    return data

def api_counter_increment(n: int = 1) -> None:
    try:
        data = _api_counter_load()
        data = _api_counter_reset_if_new_day(data)
        data["count"] = int(data.get("count", 0) or 0) + int(n)
        save_json_atomic(API_COUNTER_PATH, data)
    except Exception:
        pass

def api_counter_note_429(status: int, url: str, retry_after: str = "") -> None:
    try:
        data = _api_counter_load()
        data = _api_counter_reset_if_new_day(data)
        data["last_429_at"] = datetime.now().isoformat(timespec="seconds")
        data["last_429_status"] = int(status)
        data["last_429_url"] = str(url)[:300]
        data["last_429_retry_after"] = str(retry_after or "")
        save_json_atomic(API_COUNTER_PATH, data)
    except Exception:
        pass

# -----------------------------
# Active hours
# -----------------------------
def is_within_active_hours(now_local: datetime, start_hour: int, end_hour: int) -> bool:
    h = now_local.hour
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= h < end_hour
    return (h >= start_hour) or (h < end_hour)

# -----------------------------
# Telegram
# -----------------------------
def telegram_send(bot_token: str, chat_id: str, text: str) -> Tuple[bool, Dict[str, Any]]:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=20)
        j = r.json() if "application/json" in r.headers.get("Content-Type", "") else {"raw": r.text}
        ok = (r.status_code == 200) and bool(j.get("ok", False))
        return ok, j
    except Exception as e:
        return False, {"exception": str(e)}

# -----------------------------
# eBay auth + search
# -----------------------------
def get_app_token(client_id: str, client_secret: str) -> str:
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(EBAY_OAUTH_URL, auth=auth, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def search_live(token: str, marketplace_id: str, q: str, limit: int, mode: str, category_id: str = "") -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace_id}
    items = []

    def _get(params):
        time.sleep(0.25)  # polite throttle for background scanner
        api_counter_increment(1)
        r = requests.get(BROWSE_SEARCH_URL, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            api_counter_note_429(r.status_code, r.url, r.headers.get("Retry-After", ""))
        r.raise_for_status()
        return r.json().get("itemSummaries", [])

    if mode in ("bin", "both"):
        params = {"q": q, "limit": min(limit, 200), "filter": "buyingOptions:{FIXED_PRICE}"}
        if category_id:
            params["category_ids"] = category_id
        items += _get(params)

    if mode in ("auctions_ending", "both"):
        params = {
            "q": q,
            "limit": min(limit, 200),
            "filter": "buyingOptions:{AUCTION}",
        }
        if category_id:
            params["category_ids"] = category_id
        items += _get(params)

    # de-dupe within scan by itemId (stable)
    seen = set()
    merged = []
    for it in items:
        item_id = it.get("itemId")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        merged.append(it)
    return merged

def detect_make_offer(item: Dict[str, Any]) -> bool:
    opts = item.get("buyingOptions") or []
    if isinstance(opts, str):
        opts = [opts]
    opts = {str(o).strip().upper() for o in opts}
    return ("BEST_OFFER" in opts) or ("MAKE_OFFER" in opts)

def item_buy_total(item: Dict[str, Any]) -> Tuple[float, float, float]:
    price_val = (item.get("price") or {}).get("value")
    if price_val is None:
        return (0.0, 0.0, 0.0)
    price = float(price_val)

    ship_in = 0.0
    try:
        ship_in = float((item.get("shippingOptions") or [{}])[0].get("shippingCost", {}).get("value", 0.0))
    except Exception:
        ship_in = 0.0

    return (price, ship_in, price + ship_in)

# -----------------------------
# Maths + filters
# -----------------------------
def net_after_fees(sell_price: float, fee_rate: float, ship_out: float, packaging: float) -> float:
    return sell_price * (1.0 - fee_rate) - ship_out - packaging

def max_buy_price(sell_price: float, fee_rate: float, ship_out: float, packaging: float, parts: float, extra_costs: float, target_profit: float) -> float:
    return net_after_fees(sell_price, fee_rate, ship_out, packaging) - parts - extra_costs - target_profit

def title_has_any(title: str, words: List[str]) -> bool:
    t = (title or "").lower()
    return any((w or "").strip().lower() in t for w in words if str(w).strip())

def title_has_none(title: str, words: List[str]) -> bool:
    t = (title or "").lower()
    return not any((w or "").strip().lower() in t for w in words if str(w).strip())

# -----------------------------
# One-shot scan run
# -----------------------------

# -----------------------------
# Single-instance shared scan lock (prevents overlap with Streamlit manual scan)
# -----------------------------
def acquire_lock() -> bool:
    try:
        if os.path.exists(LOCK_PATH):
            # Stale lock handling: if lock older than 2 hours, remove it
            try:
                age = time.time() - os.path.getmtime(LOCK_PATH)
                if age > 2 * 3600:
                    os.remove(LOCK_PATH)
                else:
                    return False
            except Exception:
                return False
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True
    except Exception:
        return False

def release_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass

def main():
    if not acquire_lock():
        print('[scanner] Another scan instance is already running. Exiting.')
        return
    atexit.register(release_lock)

    ACTIVE_START = int(os.getenv("ACTIVE_START_HOUR", "8"))
    ACTIVE_END = int(os.getenv("ACTIVE_END_HOUR", "24"))
    SCAN_MODE = os.getenv("SCAN_MODE", "both")
    PER_PROFILE_LIMIT = int(os.getenv("PER_PROFILE_LIMIT", "60"))
    ONLY_BELOW_MAXBUY = os.getenv("ONLY_BELOW_MAXBUY", "1").strip() not in ("0", "false", "False")

    now_local = datetime.now()
    if not is_within_active_hours(now_local, ACTIVE_START, ACTIVE_END):
        print(f"[scanner] Outside active hours ({ACTIVE_START}-{ACTIVE_END}). Exiting.")
        return

    cfg = load_config()
    products_raw = cfg.get("products") or cfg.get("consoles") or []
    if isinstance(products_raw, dict):
        products = products_raw
    elif isinstance(products_raw, list):
        products = {c.get("id"): c for c in products_raw if isinstance(c, dict) and c.get("id")}
    else:
        products = {}
    profiles = cfg["profiles"]

    client_id = os.getenv("EBAY_CLIENT_ID", "").strip()
    client_secret = os.getenv("EBAY_CLIENT_SECRET", "").strip()
    marketplace = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_GB").strip()

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not client_id or not client_secret:
        raise RuntimeError("Missing EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in .env")
    if not tg_token or not tg_chat:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env")

    notified_ids = load_notified_ids()

    # Global Option A dedupe across the whole run (prevents duplicate alerts across profiles/bases)
    seen_ids_global = set()

    token = get_app_token(client_id, client_secret)

    sent = 0
    checked = 0

    for prof_name, p in profiles.items():
        console_id = p.get("console_id")
        if console_id not in products:
            continue
        c = products[console_id]

        fault_q = str(p.get("fault_query","") or "").strip()
        bases = c.get("search_bases") or []
        if isinstance(bases, str):
            bases = [bases]
        bases = [str(b).strip() for b in bases if str(b).strip()]
        if not bases:
            bases = [str(c.get("search_base","")).strip()]
        bases = [b for b in bases if b]

        queries = [f"{b} {fault_q}".strip() for b in bases] if fault_q else bases

        all_items: List[Dict[str, Any]] = []
        for q in queries:
            if not q:
                continue
            all_items += search_live(token, marketplace, q, PER_PROFILE_LIMIT, SCAN_MODE)


        sell_price = float(p.get("sell_price_override") or c.get("default_sell_price", 0.0))
        fee_rate = float(c.get("fee_rate", 0.13))
        ship_out = float(c.get("ship_out", 0.0))
        packaging = float(c.get("packaging", 0.0))
        parts = float(p.get("parts", 0.0))
        extra_costs = float(p.get("extra_costs", 0.0))
        target_profit = float(p.get("target_profit", 0.0))

        mx_buy = max_buy_price(sell_price, fee_rate, ship_out, packaging, parts, extra_costs, target_profit)
        net_before_buy = net_after_fees(sell_price, fee_rate, ship_out, packaging) - parts - extra_costs

        must_any = c.get("must_include_any", []) or []
        console_excl = c.get("exclude_words", []) or []
        profile_excl = p.get("exclude_words", []) or []
        min_buy_total = float(c.get("min_buy_total", 0.0) or 0.0)

        # Word filters first, then dedupe (matches app behaviour)
        must_words = _norm_words_csv(must_any)
        excl_words = _norm_words_csv(console_excl) + _norm_words_csv(profile_excl)

        filtered_items: List[Dict[str, Any]] = []
        for _it in all_items:
            if _passes_word_filters(_it.get("title",""), must_words, excl_words):
                filtered_items.append(_it)

        deduped_items: List[Dict[str, Any]] = []
        for _it in filtered_items:
            iid = ebay_item_id(_it)
            if not iid or iid in seen_ids_global:
                continue
            seen_ids_global.add(iid)
            deduped_items.append(_it)

        for it in deduped_items:
            item_id = it.get("itemId")
            if not item_id:
                continue

            if item_id in notified_ids:
                continue

            title = it.get("title") or ""
            cond = (it.get("condition") or "").strip().lower()
            if cond == "new":
                continue


            price, ship_in, buy_total = item_buy_total(it)
            if buy_total <= 0 or buy_total < min_buy_total:
                continue

            checked += 1

            if ONLY_BELOW_MAXBUY and buy_total > mx_buy:
                continue

            est_profit = net_before_buy - buy_total
            offer_txt = "Yes" if detect_make_offer(it) else "No"
            url = it.get("itemWebUrl", "")

            msg = (
                f"✅ DealFinder: NEW GOOD BUY\n"
                f"{c.get('name', console_id)} | {prof_name}\n"
                f"Buy total: £{buy_total:.2f}\n"
                f"Est profit: £{est_profit:.2f}\n"
                f"Make offer: {offer_txt}\n"
                f"{title}\n"
                f"{url}"
            )

            ok, _info = telegram_send(tg_token, tg_chat, msg)
            if ok:
                notified_ids.add(item_id)
                sent += 1

    save_notified_ids(notified_ids)
    print(f"[scanner] Done. checked={checked}, alerts_sent={sent}, notified_total={len(notified_ids)}")

    # ── Rare items scan ─────────────────────────────────────────────────────
    rare_items = cfg.get("rare_items", {})
    for item_id, r in rare_items.items():
        query = str(r.get("search_query", "")).strip() or str(r.get("name", "")).strip()
        if not query:
            continue

        category_id = str(r.get("category_id", "")).strip()
        sell_price = float(r.get("sell_price", 0.0))
        fee_rate = float(r.get("fee_rate", 0.13))
        ship_out = float(r.get("ship_out", 0.0))
        packaging = float(r.get("packaging", 0.0))
        extra_costs = float(r.get("extra_costs", 0.0))
        target_profit = float(r.get("target_profit", 0.0))
        min_buy_total = float(r.get("min_buy_total", 0.0))

        must_any = _norm_words_csv(r.get("must_include_any", []))
        excl_words = _norm_words_csv(r.get("exclude_words", []))

        mx_buy = sell_price * (1.0 - fee_rate) - ship_out - packaging - extra_costs - target_profit
        net_before_buy = sell_price * (1.0 - fee_rate) - ship_out - packaging - extra_costs

        rare_items_raw = search_live(token, marketplace, query, PER_PROFILE_LIMIT, SCAN_MODE, category_id)

        for it in rare_items_raw:
            it_id = ebay_item_id(it)
            if not it_id or it_id in notified_ids:
                continue

            title = it.get("title") or ""
            if not _passes_word_filters(title, must_any, excl_words):
                continue

            price_val = (it.get("price") or {}).get("value")
            if price_val is None:
                continue
            price = float(price_val)
            try:
                ship_in = float((it.get("shippingOptions") or [{}])[0].get("shippingCost", {}).get("value", 0.0))
            except Exception:
                ship_in = 0.0
            buy_total = price + ship_in

            if buy_total <= 0 or buy_total < min_buy_total:
                continue

            checked += 1

            if ONLY_BELOW_MAXBUY and buy_total > mx_buy:
                continue

            est_profit = net_before_buy - buy_total
            offer_txt = "Yes" if detect_make_offer(it) else "No"
            url = it.get("itemWebUrl", "")

            msg = (
                f"✅ DealFinder: RARE ITEM FOUND\n"
                f"{r.get('name', item_id)}\n"
                f"Buy total: £{buy_total:.2f}\n"
                f"Est profit: £{est_profit:.2f}\n"
                f"Make offer: {offer_txt}\n"
                f"{title}\n"
                f"{url}"
            )

            ok, _info = telegram_send(tg_token, tg_chat, msg)
            if ok:
                notified_ids.add(it_id)
                sent += 1

    save_notified_ids(notified_ids)
    print(f"[scanner] Rare scan done. checked={checked}, alerts_sent={sent}, notified_total={len(notified_ids)}")

if __name__ == "__main__":
    main()