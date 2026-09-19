"""
BOSWaves Swing Structure Forecast Chart — v8.3
───────────────────────────────────────────────
เพิ่มจาก v8.2:
  [+35] pause ระบบจาก Supabase
  [+36] RSI alert mode: always / once_per_day / off
  [+37] Watchlist — แจ้ง RSI เฉพาะ ไม่ส่งรูป
  [+38] บันทึก log การส่งลง Supabase
  [+39] สรุปสัปดาห์ส่งทุกวันเสาร์
  [+40] font Linux fix
"""

import os, sys, time, logging, warnings, json, urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime, date
from logging.handlers import TimedRotatingFileHandler

warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")

from dotenv import load_dotenv
_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")

BOT_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL   = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY   = os.getenv("SUPABASE_ANON_KEY", "")

# ═══════════════════════════════════════════════════════════════════
#  SUPABASE HELPERS
# ═══════════════════════════════════════════════════════════════════
def _sb_get(path: str) -> list:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers={"apikey": SUPABASE_KEY,
                     "Authorization": f"Bearer {SUPABASE_KEY}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[WARN] Supabase GET {path}: {e}")
        return []

def _sb_post(path: str, body: dict) -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}
    try:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/{path}",
            data=data, method="POST",
            headers={"apikey": SUPABASE_KEY,
                     "Authorization": f"Bearer {SUPABASE_KEY}",
                     "Content-Type":  "application/json",
                     "Prefer":        "return=minimal"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return {}
    except Exception as e:
        print(f"[WARN] Supabase POST {path}: {e}")
        return {}

def _sb_delete(path: str) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/{path}",
            method="DELETE",
            headers={"apikey": SUPABASE_KEY,
                     "Authorization": f"Bearer {SUPABASE_KEY}"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[WARN] Supabase DELETE {path}: {e}")

# ═══════════════════════════════════════════════════════════════════
#  LOAD CONFIG FROM SUPABASE
# ═══════════════════════════════════════════════════════════════════
def _load_supabase_config() -> dict:
    defaults = {
        "tickers"        : ["VOO", "NVDA", "JEPQ"],
        "schedules"      : [],
        "rsi_overbought" : 70,
        "rsi_oversold"   : 30,
        "rsi_alert_mode" : "once_per_day",  # always / once_per_day / off
        "gemini"         : {"translate": True, "sentiment": True},
        "chat_ids"       : [],
        "recipients"     : [],
        "paused"         : False,
        "watchlist"      : [],
        "weekly_summary" : True,
        "bot_token"      : "",
    }
    rows = _sb_get("boswaves_config?id=eq.1&select=*")
    if rows:
        c = rows[0]
        for k in ["tickers","schedules","rsi_overbought","rsi_oversold",
                  "rsi_alert_mode","gemini","paused","watchlist",
                  "weekly_summary","bot_token"]:
            if c.get(k) is not None:
                defaults[k] = c[k]

    recips = _sb_get("boswaves_recipients?active=eq.true&select=*")
    defaults["recipients"] = recips
    defaults["chat_ids"]   = [r["chat_id"] for r in recips]
    return defaults

_cfg           = _load_supabase_config()
TICKERS        = _cfg["tickers"]
WATCHLIST      = _cfg.get("watchlist", [])
PAUSED         = _cfg.get("paused", False)
WEEKLY_SUMMARY = _cfg.get("weekly_summary", True)
RSI_ALERT_MODE = _cfg.get("rsi_alert_mode", "once_per_day")
RECIPIENTS     = _cfg.get("recipients", [])
CHAT_IDS       = _cfg["chat_ids"] or [
    c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS",
    os.getenv("TELEGRAM_CHAT_ID", "")).split(",") if c.strip()
]
CHAT_ID        = CHAT_IDS[0] if CHAT_IDS else ""
MAIN_BOT_TOKEN = _cfg.get("bot_token") or BOT_TOKEN

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION (ค่าคงที่)
# ═══════════════════════════════════════════════════════════════════
# [+41] รับ inputs จาก GitHub Actions workflow_dispatch
SEND_CHART = os.getenv("SEND_CHART", "true").lower() == "true"
SEND_NEWS  = os.getenv("SEND_NEWS",  "true").lower() == "true"
SEND_DAILY = os.getenv("SEND_DAILY", "true").lower() == "true"

DATA_PERIOD       = "1y"
SWING_LENGTH      = 16
ATR_PERIOD        = 14
ZONE_WIDTH_FACTOR = 0.4
FIB_EXTENSIONS    = [1.0, 1.272, 1.618, 2.0]
EMA_SHORT         = 50
EMA_LONG          = 200
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR", str(_SCRIPT_DIR / "output")))
LOG_DIR           = OUTPUT_DIR / "logs"
DOWNLOAD_RETRIES  = 3
DOWNLOAD_DELAY    = 5
TG_TIMEOUT        = 30
NEWS_PER_TICKER   = 3

# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_DIR.mkdir(parents=True, exist_ok=True)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                          datefmt="%d/%m/%Y %H:%M:%S")
_fh  = TimedRotatingFileHandler(LOG_DIR / "boswaves.log",
                                 when="midnight", backupCount=30,
                                 encoding="utf-8")
_fh.setFormatter(_fmt)
_ch  = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
log  = logging.getLogger("boswaves")
log.setLevel(logging.DEBUG)
log.addHandler(_fh)
log.addHandler(_ch)

# ── guard ──────────────────────────────────────────────────────────
if not BOT_TOKEN:
    log.error("TELEGRAM_TOKEN ไม่พบ — ยกเลิก"); sys.exit(1)
if not CHAT_IDS:
    log.error("ไม่พบ Chat ID — ยกเลิก"); sys.exit(1)

# [+35] ตรวจสอบ pause
if PAUSED:
    log.info("ระบบถูก pause จาก config — ไม่รันครับ")
    sys.exit(0)

log.info(f"ผู้รับ Telegram: {len(CHAT_IDS)} คน | RSI mode: {RSI_ALERT_MODE}")
if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY ไม่พบ — ข้ามการแปลข่าว")

import requests
import feedparser
import yfinance as yf
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════
#  GEMINI — auto-detect model
# ═══════════════════════════════════════════════════════════════════
_gemini            = None
_gemini_model_name = None
_gemini_error      = None

def _pick_gemini_model(preferred: list) -> str | None:
    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for name in preferred:
            if name in available:
                return name
        flash = [m for m in available
                 if "flash" in m and "image" not in m and "tts" not in m]
        return flash[0] if flash else None
    except Exception as e:
        log.warning(f"  list_models error: {e}")
        raise

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _MODEL_PRIORITY = ["gemini-3.6-flash","gemini-2.5-flash",
                       "gemini-2.0-flash","gemini-1.5-flash"]
    try:
        _gemini_model_name = _pick_gemini_model(_MODEL_PRIORITY)
        if _gemini_model_name:
            _gemini = genai.GenerativeModel(_gemini_model_name)
            log.info(f"  Gemini model: {_gemini_model_name}")
        else:
            _gemini_error = "ไม่พบ Gemini model"
    except Exception as _e:
        _gemini_error = f"Gemini init error: {_e}"

# ═══════════════════════════════════════════════════════════════════
#  FONT SETUP — รองรับ Windows และ Linux
# ═══════════════════════════════════════════════════════════════════
_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\tahomabd.ttf",  "Tahoma"),
    (r"C:\Windows\Fonts\tahoma.ttf",    "Tahoma"),
    (r"C:\Windows\Fonts\leelawdb.ttf",  "Leelawadee UI"),
    (r"C:\Windows\Fonts\leelawad.ttf",  "Leelawadee UI"),
    ("/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",  "Garuda"),
    ("/usr/share/fonts/truetype/tlwg/Garuda.ttf",       "Garuda"),
    ("/usr/share/fonts/truetype/tlwg/Loma.ttf",         "Loma"),
    ("/usr/share/fonts/truetype/tlwg/Norasi.ttf",       "Norasi"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
]
_loaded_families = []
for _fp, _fm in _FONT_CANDIDATES:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        if _fm not in _loaded_families:
            _loaded_families.append(_fm)
        log.debug(f"  font loaded: {_fm}")

plt.rcParams.update({
    'font.family'     : _loaded_families + ["DejaVu Sans", "sans-serif"],
    'font.size'       : 9,
    'axes.facecolor'  : '#0B0E14',
    'figure.facecolor': '#0B0E14',
    'mathtext.default': 'regular',
    'text.usetex'     : False,
})

# ═══════════════════════════════════════════════════════════════════
#  HELPERS — คำนวณ
# ═══════════════════════════════════════════════════════════════════
def compute_atr(high, low, close, period=14):
    n = len(high); tr = np.empty(n); tr[0] = high[0]-low[0]
    for i in range(1,n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.empty(n); atr[0] = tr[0]
    for i in range(1,n): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr

def compute_rsi(close, period=14):
    delta = np.diff(close)
    gain  = np.where(delta>0, delta, 0.0)
    loss  = np.where(delta<0,-delta, 0.0)
    ag = np.zeros(len(close)); al = np.zeros(len(close))
    ag[period] = gain[:period].mean(); al[period] = loss[:period].mean()
    for i in range(period+1, len(close)):
        ag[i] = (ag[i-1]*(period-1)+gain[i-1])/period
        al[i] = (al[i-1]*(period-1)+loss[i-1])/period
    rs = np.divide(ag, al, out=np.zeros_like(ag), where=al!=0)
    return 100-(100/(1+rs))

def compute_ema(close, period):
    ema = np.empty(len(close)); ema[0] = close[0]; k = 2.0/(period+1)
    for i in range(1,len(close)): ema[i] = close[i]*k+ema[i-1]*(1-k)
    return ema

def find_swings(high, low, length=16):
    n = len(high); sh = np.zeros(n,dtype=bool); sl = np.zeros(n,dtype=bool)
    half = length//2
    for i in range(half, n-half):
        win = range(i-half, i+half+1)
        if all(high[i]>=high[j] for j in win if j!=i): sh[i]=True
        if all(low[i] <=low[j]  for j in win if j!=i): sl[i]=True
    return sh, sl

def trend_bias(close, ema50, ema200, rsi):
    if ema50[-1]>ema200[-1] and close[-1]>ema50[-1] and rsi[-1]>50: return "BULLISH","#3ECF8E"
    if ema50[-1]<ema200[-1] and close[-1]<ema50[-1] and rsi[-1]<50: return "BEARISH","#FF5A5A"
    return "NEUTRAL","#F6C90E"

def download_with_retry(ticker, period, retries=3, delay=5):
    for attempt in range(1, retries+1):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            df = df[~df.index.duplicated(keep='first')].dropna()
            if not df.empty: return df
        except Exception as e:
            log.warning(f"  attempt {attempt} error: {e}")
        if attempt < retries: time.sleep(delay)
    return None

def fetch_extra_info(ticker: str) -> dict:
    defaults = {"day_change_pct":None,"week52_high":None,"week52_low":None,"div_yield":None}
    try:
        info = yf.Ticker(ticker).fast_info
        defaults["week52_high"] = getattr(info,"year_high",None)
        defaults["week52_low"]  = getattr(info,"year_low",None)
        prev = getattr(info,"previous_close",None); last = getattr(info,"last_price",None)
        if prev and last and prev!=0:
            defaults["day_change_pct"] = float((last-prev)/prev*100)
        full = yf.Ticker(ticker).info
        raw_dy = full.get("dividendYield")
        if raw_dy is not None:
            defaults["div_yield"] = float(raw_dy/100 if raw_dy>1.0 else raw_dy)
    except Exception as e: log.warning(f"  fetch_extra_info {ticker}: {e}")
    return defaults

# ═══════════════════════════════════════════════════════════════════
#  FEAR & GREED
# ═══════════════════════════════════════════════════════════════════
def fetch_fear_greed() -> dict:
    result = {"value":None,"label_th":"N/A"}
    def _lbl(v):
        if v<=25: return "กลัวมาก 😱"
        elif v<=45: return "กลัว 😟"
        elif v<=55: return "เป็นกลาง 😐"
        elif v<=75: return "โลภ 😏"
        else: return "โลภมาก 🤑"
    try:
        r = requests.get("https://api.alternative.me/fng/",timeout=8,
                         headers={"User-Agent":"Mozilla/5.0"})
        val = int(r.json()["data"][0]["value"])
        result.update({"value":val,"label_th":_lbl(val)}); return result
    except: pass
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                         timeout=8,headers={"User-Agent":"Mozilla/5.0"})
        val = round(r.json()["fear_and_greed"]["score"],1)
        result.update({"value":val,"label_th":_lbl(val)})
    except: pass
    return result

# ═══════════════════════════════════════════════════════════════════
#  VIX + S&P500 + USD/THB
# ═══════════════════════════════════════════════════════════════════
def fetch_market_context() -> dict:
    result = {"vix":None,"sp500_chg":None,"sp500_last":None,"usdthb":None}
    try:
        v = yf.download("^VIX",period="2d",progress=False,auto_adjust=True)
        if not v.empty: result["vix"] = round(float(v['Close'].squeeze().iloc[-1]),2)
        s = yf.download("^GSPC",period="2d",progress=False,auto_adjust=True)
        if len(s)>=2:
            c = s['Close'].squeeze().values
            result["sp500_last"] = round(float(c[-1]),2)
            result["sp500_chg"]  = round((c[-1]-c[-2])/c[-2]*100,2)
        t = yf.download("THB=X",period="2d",progress=False,auto_adjust=True)
        if not t.empty: result["usdthb"] = round(float(t['Close'].squeeze().iloc[-1]),2)
    except Exception as e: log.warning(f"  market_context: {e}")
    return result

# ═══════════════════════════════════════════════════════════════════
#  GOLD PRICE
# ═══════════════════════════════════════════════════════════════════
def fetch_gold_price() -> dict:
    result = {"price":None,"change_pct":None}
    try:
        req = urllib.request.Request("https://api.gold-api.com/price/XAU",
                                     headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req,timeout=10) as resp:
            data = json.loads(resp.read())
        price = data.get("price"); prev = data.get("prev_close_price")
        if price: result["price"] = float(price)
        if price and prev and float(prev)!=0:
            result["change_pct"] = (float(price)-float(prev))/float(prev)*100
    except Exception as e: log.warning(f"  gold: {e}")
    return result

# ═══════════════════════════════════════════════════════════════════
#  Yahoo Finance RSS
# ═══════════════════════════════════════════════════════════════════
def fetch_yahoo_news(ticker: str, n: int=3) -> list[str]:
    headlines = []
    try:
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        for entry in feed.entries[:n]:
            headlines.append(entry.title.strip())
    except Exception as e: log.warning(f"  yahoo_news {ticker}: {e}")
    return headlines

# ═══════════════════════════════════════════════════════════════════
#  GEMINI — แปลข่าว
# ═══════════════════════════════════════════════════════════════════
def translate_news_gemini(ticker: str, headlines: list[str]) -> tuple[list[str], str|None]:
    if not _gemini or not headlines:
        return headlines, (_gemini_error or "ไม่มี Gemini")
    prompt = (
        f"พาดหัวข่าวหุ้น {ticker} {len(headlines)} ข่าว:\n\n"
        + "\n".join(f"{i+1}. {h}" for i,h in enumerate(headlines))
        + "\n\nแปลและสรุปเป็นภาษาไทย สั้นกระชับ ตอบแค่รายการ 1. 2. 3."
    )
    try:
        resp = _gemini.generate_content(prompt)
        lines = [l.strip() for l in resp.text.strip().splitlines()
                 if l.strip() and l.strip()[0].isdigit()]
        cleaned = []
        for l in lines:
            parts = l.split(". ",1)
            cleaned.append(parts[1] if len(parts)>1 else l)
        return (cleaned if cleaned else headlines), None
    except Exception as e:
        err = f"Gemini error ({_gemini_model_name}): {e}"
        log.warning(f"  {err}")
        return headlines, err

# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM — รองรับหลาย recipient / หลาย bot token
# ═══════════════════════════════════════════════════════════════════
def tg_send_photo(image_path: Path, caption: str) -> bool:
    ok_count = 0
    for r in RECIPIENTS:
        token = (r.get("bot_token") or "").strip() or MAIN_BOT_TOKEN
        cid   = r["chat_id"]
        url   = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(image_path,"rb") as f:
                resp = requests.post(url,
                    cap = caption[:1020] + '…' if len(caption) > 1024 else caption
                    data={"chat_id":cid,"caption":cap,"parse_mode":"HTML"},
                    
                    files={"photo":f}, timeout=TG_TIMEOUT)
            if resp.status_code==200 and resp.json().get("ok"): ok_count+=1
            else: log.error(f"  TG photo [{cid}] status={resp.status_code}: {resp.text[:200]}")
        except Exception as e: log.error(f"  TG photo [{cid}]: {e}")
    return ok_count>0

def tg_send_text(text: str) -> None:
    for r in RECIPIENTS:
        token = (r.get("bot_token") or "").strip() or MAIN_BOT_TOKEN
        cid   = r["chat_id"]
        url   = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, data={"chat_id":cid,"text":text,"parse_mode":"HTML"},
                          timeout=TG_TIMEOUT)
        except: pass

# ═══════════════════════════════════════════════════════════════════
#  RSI HELPERS
# ═══════════════════════════════════════════════════════════════════
RSI_OVERBOUGHT = int(_cfg.get("rsi_overbought", 70))
RSI_OVERSOLD   = int(_cfg.get("rsi_oversold",   30))

def _rsi_label(rsi: float) -> str:
    if rsi>=RSI_OVERBOUGHT:
        return f"🔥 <b>{rsi:.1f} — ราคาร้อนแรงเกินไป</b> ระวังปรับตัวลง"
    elif rsi<=RSI_OVERSOLD:
        return f"💎 <b>{rsi:.1f} — ราคาถูกผิดปกติ</b> อาจเป็นจังหวะน่าสนใจ"
    return f"{rsi:.1f} — ปกติ"

def _rsi_tag_short(rsi: float) -> str:
    if rsi>=RSI_OVERBOUGHT: return f"  🔥 RSI {rsi:.0f} ร้อนแรง"
    elif rsi<=RSI_OVERSOLD: return f"  💎 RSI {rsi:.0f} ถูกผิดปกติ"
    return ""

def tg_rsi_alert(ticker: str, rsi: float, price: float) -> None:
    """[+36] ส่งตามโหมดที่ตั้งไว้"""
    if RSI_ALERT_MODE == "off":
        return
    if rsi < RSI_OVERBOUGHT and rsi > RSI_OVERSOLD:
        return

    # once_per_day: เช็คว่าส่งวันนี้แล้วหรือยัง
    if RSI_ALERT_MODE == "once_per_day":
        today = date.today().isoformat()
        rows = _sb_get(f"boswaves_rsi_sent?ticker=eq.{ticker}&sent_date=eq.{today}&select=id")
        if rows:
            log.info(f"  RSI alert {ticker} — ส่งแล้ววันนี้ ข้าม")
            return
        # บันทึกว่าส่งแล้ว
        _sb_post("boswaves_rsi_sent", {"ticker":ticker,"sent_date":today,"rsi_value":rsi})

    if rsi >= RSI_OVERBOUGHT:
        text = (f"⚠️ <b>สัญญาณ {ticker}</b>\n\n"
                f"🔥 <b>ราคาร้อนแรงเกินไปแล้ว!</b>\n"
                f"ตัวเลข RSI อยู่ที่ {rsi:.1f} (เกิน {RSI_OVERBOUGHT})\n\n"
                f"ราคาปัจจุบัน: <b>${price:.2f}</b>\n"
                f"👉 ยังไม่ควรรีบซื้อเพิ่ม รอให้ราคาเย็นลงก่อนครับ\n\n"
                f"<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>")
    else:
        text = (f"💎 <b>โอกาส {ticker}</b>\n\n"
                f"💎 <b>ราคาถูกผิดปกติแล้ว!</b>\n"
                f"ตัวเลข RSI อยู่ที่ {rsi:.1f} (ต่ำกว่า {RSI_OVERSOLD})\n\n"
                f"ราคาปัจจุบัน: <b>${price:.2f}</b>\n"
                f"👉 อาจเป็นจังหวะที่น่าสนใจ ดูสัญญาณอื่นประกอบด้วยครับ\n\n"
                f"<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>")
    tg_send_text(text)
    log.info(f"  ✓ RSI alert ส่งแล้ว ({ticker} RSI={rsi:.1f})")

# ═══════════════════════════════════════════════════════════════════
#  [+38] LOG การส่งลง Supabase
# ═══════════════════════════════════════════════════════════════════
def log_run(tickers: list, success: bool, recipients: int,
            duration_s: int, note: str="") -> None:
    _sb_post("boswaves_logs", {
        "tickers"   : tickers,
        "success"   : success,
        "recipients": recipients,
        "duration_s": duration_s,
        "note"      : note,
    })

# ═══════════════════════════════════════════════════════════════════
#  DAILY SUMMARY CHART
# ═══════════════════════════════════════════════════════════════════
def build_daily_chart(ticker_data: dict, date_str: str) -> Path | None:
    try:
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(11,9),facecolor='#0B0E14')
        fig.subplots_adjust(hspace=0.40,top=0.92,bottom=0.07,left=0.09,right=0.95)
        colors = {'VOO':'#3ECF8E','NVDA':'#F97316','JEPQ':'#60A5FA'}
        for ax in [ax1,ax2]:
            ax.set_facecolor('#0B0E14'); ax.grid(True,alpha=0.08,linestyle='--')
            ax.tick_params(colors='#888888',labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            for sp in ax.spines.values(): sp.set_edgecolor('#333333')

        # กราฟบน: ราคาจริง
        for ticker in TICKERS:
            df = ticker_data.get(ticker)
            if df is None: continue
            closes = df['Close'].squeeze().values.astype(float)
            dates_num = mdates.date2num(df.index)
            col = colors.get(ticker,'#FFFFFF')
            last = closes[-1]; prev = closes[-2] if len(closes)>=2 else last
            chg  = (last-prev)/prev*100; sign = '+' if chg>=0 else ''
            ax1.plot(dates_num,closes,color=col,linewidth=1.5,
                     label=f'{ticker}  ${last:.2f} ({sign}{chg:.2f}%)')
            ax1.annotate(f'${last:.2f}',xy=(dates_num[-1],last),
                        xytext=(6,0),textcoords='offset points',
                        color=col,fontsize=9,fontweight='bold',va='center')

        ax1.set_title('ราคาจริง (USD)',color='white',fontsize=10,fontweight='bold',loc='left',pad=6)
        ax1.set_ylabel('ราคา (USD)',color='white',fontsize=8)
        ax1.legend(facecolor='#1A1F2B',edgecolor='#444444',labelcolor='white',fontsize=9,loc='upper left')

        # กราฟล่าง: Normalized
        ax2.axhline(100,color='#555555',linestyle='--',linewidth=0.9,alpha=0.7)
        for ticker in TICKERS:
            df = ticker_data.get(ticker)
            if df is None: continue
            closes = df['Close'].squeeze().values.astype(float)
            dates_num = mdates.date2num(df.index)
            col  = colors.get(ticker,'#FFFFFF')
            norm = closes/closes[0]*100; last_n = norm[-1]
            chg  = last_n-100; sign = '+' if chg>=0 else ''
            ax2.plot(dates_num,norm,color=col,linewidth=1.8,label=f'{ticker}  {sign}{chg:.1f}%')
            ax2.fill_between(dates_num,100,norm,alpha=0.08,color=col)
            ax2.annotate(f'{sign}{chg:.1f}%',xy=(dates_num[-1],last_n),
                        xytext=(6,0),textcoords='offset points',
                        color=col,fontsize=10,fontweight='bold',va='center')

        ax2.set_title('Normalized — % การเติบโตเปรียบเทียบ (เริ่มต้น=100)',
                      color='white',fontsize=10,fontweight='bold',loc='left',pad=6)
        ax2.set_ylabel('ผลตอบแทน (base=100)',color='white',fontsize=8)
        ax2.legend(facecolor='#1A1F2B',edgecolor='#444444',labelcolor='white',fontsize=9,loc='upper left')
        fig.suptitle(f'Daily Summary — {now_str}',color='white',fontsize=11,fontweight='bold')
        out = OUTPUT_DIR / f"daily_summary_{date_str}.png"
        fig.savefig(str(out),dpi=110,facecolor='#0B0E14',edgecolor='none',bbox_inches='tight')
        plt.close(fig)
        return out
    except Exception as e:
        log.error(f"  Daily chart error: {e}"); return None

def build_daily_caption(ticker_data: dict, market_ctx: dict,
                        fear_greed: dict, gold: dict) -> str:
    lines = [f"<b>📊 Daily Summary — {now_str}</b>\n"]
    for ticker in TICKERS:
        df = ticker_data.get(ticker)
        if df is None: lines.append(f"{ticker}  N/A"); continue
        closes = df['Close'].squeeze().values.astype(float)
        last   = closes[-1]; prev = closes[-2] if len(closes)>=2 else last
        chg    = (last-prev)/prev*100; arrow = "▲" if chg>=0 else "▼"
        rsi_now = compute_rsi(closes)[-1]
        rsi_tag = _rsi_tag_short(rsi_now)
        lines.append(f"<b>{ticker}</b>  ${last:.2f}  {arrow} {abs(chg):.2f}%{rsi_tag}")
    lines.append("━━━━━━━━━━━━━━━━")
    sp_chg  = market_ctx.get("sp500_chg"); sp_last = market_ctx.get("sp500_last")
    vix     = market_ctx.get("vix");       usdthb  = market_ctx.get("usdthb")
    sp_str  = (f"${sp_last:,.2f} ({'+' if (sp_chg or 0)>=0 else ''}{sp_chg:.2f}%)" if sp_last else "N/A")
    vix_str = f"{vix:.1f}" if vix else "N/A"
    thb_str = f"{usdthb:.2f}" if usdthb else "N/A"
    fg_val  = fear_greed.get("value"); fg_lbl = fear_greed.get("label_th","N/A")
    fg_str  = f"{fg_val:.0f} ({fg_lbl})" if fg_val is not None else "N/A"
    gold_p  = gold.get("price"); gold_chg = gold.get("change_pct")
    gold_str = f"${gold_p:,.2f}" if gold_p else "N/A"
    gold_chg_str = (f" (+{gold_chg:.2f}%)" if gold_chg and gold_chg>=0
                    else f" ({gold_chg:.2f}%)" if gold_chg else "")
    lines.append(f"🌍 S&P500: {sp_str}  |  VIX: {vix_str}")
    lines.append(f"😱 Fear&Greed: {fg_str}")
    lines.append(f"🥇 ทองคำ: {gold_str}{gold_chg_str}")
    lines.append(f"💱 USD/THB: {thb_str}")
    lines.append("\n<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
#  BUILD CAPTION BOSWaves
# ═══════════════════════════════════════════════════════════════════
def build_caption(ticker, close_price, bias_label, rsi, atr,
                  entry, sl, tp, rr, extra, news_th: list[str]) -> str:
    bias_tag = "🟢" if bias_label=="BULLISH" else "🔴" if bias_label=="BEARISH" else "🟡"
    bias_th  = "แนวโน้มขาขึ้น" if bias_label=="BULLISH" else "แนวโน้มขาลง" if bias_label=="BEARISH" else "แนวโน้มทรงตัว"
    chg = extra.get("day_change_pct")
    chg_str  = (f"+{chg:.2f}%" if chg and chg>=0 else f"{chg:.2f}%" if chg is not None else "N/A")
    chg_icon = "📈" if (chg or 0)>=0 else "📉"
    w52h = extra.get("week52_high"); w52l = extra.get("week52_low")
    w52_str = f"${w52l:.2f} – ${w52h:.2f}" if w52h and w52l else "N/A"
    dy = extra.get("div_yield"); dy_str = f"{dy*100:.2f}%" if dy else "–"
    news_block = ""
    if news_th:
        news_block = "━━━━━━━━━━━━━━━━\n📰 ข่าวล่าสุด:\n" + "\n".join(f"• {h}" for h in news_th) + "\n"

    # [+42] ประเมินสถานการณ์ rule-based
    score = 0
    if bias_label == "BULLISH":               score += 3
    elif bias_label == "BEARISH":             score -= 3
    if rsi < RSI_OVERBOUGHT and rsi > 45:     score += 1
    if rsi > RSI_OVERSOLD  and rsi < 55:      score -= 1
    if rsi <= RSI_OVERSOLD:                   score += 2
    if rsi >= RSI_OVERBOUGHT:                 score -= 2
    if rr and rr >= 3.0:                      score += 2
    elif rr and rr >= 2.0:                    score += 1

    score = max(-5, min(5, score))  # clamp -5 ถึง 5

    up_pct   = 30 + score * 8
    down_pct = 30 - score * 8
    hold_pct = 100 - up_pct - down_pct
    up_pct   = max(5,  min(85, up_pct))
    down_pct = max(5,  min(85, down_pct))
    hold_pct = max(5,  min(50, hold_pct))

    if score >= 2:      rec = "🟢 น่าสนใจ — รอซื้อที่แนวรับ"
    elif score <= -2:   rec = "🔴 ระวัง — ยังไม่ควรเข้า"
    else:               rec = "🟡 รอดู — ถือหรือสังเกตก่อน"

    assess_block = (
        f"━━━━━━━━━━━━━━━━\n"
        f"🎲 ประเมินสถานการณ์\n"
        f"📈 โอกาสขึ้น: <b>{up_pct}%</b>\n"
        f"📉 โอกาสลง: <b>{down_pct}%</b>\n"
        f"⏸ ทรงตัว: <b>{hold_pct}%</b>\n"
        f"💡 {rec}\n"
    )

    return (
        f"<b>{ticker} — วิเคราะห์โครงสร้างราคา</b>\n"
        f"{bias_tag} {bias_th}  |  ราคา: <b>${close_price:.2f}</b>\n"
        f"{chg_icon} วันนี้: <b>{chg_str}</b>  |  ปันผล: {dy_str}\n"
        f"📊 52w: {w52_str}\n"
        f"RSI: {_rsi_label(rsi)}  |  ATR: ${atr:.2f}\n"
        f"{news_block}"
        f"{assess_block}" 
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 เข้าซื้อ: <b>${entry:.2f}</b>\n"
        f"🛑 ตัดขาดทุน: <b>${sl:.2f}</b>\n"
        f"🎯 เป้ากำไร: <b>${tp:.2f}</b>  (R:R 1:{rr:.1f})\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>\n"
        f"<i>⚠️ RSI — ถ้า > 70 มักปรับตัวลง, < 30 มักเด้งขึ้น</i>\n"
        f"<i>⚠️ EMA 50/200 — ถ้า EMA50 ตัดขึ้น EMA200 = สัญญาณขาขึ้น</i>\n"
        f"<i>⚠️ R:R — คำนวณจุดที่คุ้มเสี่ยงที่สุด</i>\n"
    )

# ═══════════════════════════════════════════════════════════════════
#  [+39] สรุปสัปดาห์ — ส่งวันเสาร์
# ═══════════════════════════════════════════════════════════════════
def send_weekly_summary(ticker_data: dict) -> None:
    if not WEEKLY_SUMMARY: return
    if datetime.now().weekday() != 5: return  # 5 = เสาร์
    log.info("--- Weekly Summary ---")
    lines = [f"<b>📅 สรุปสัปดาห์ — {now_str}</b>\n"]
    for ticker in TICKERS:
        df = ticker_data.get(ticker)
        if df is None: lines.append(f"{ticker}: ไม่มีข้อมูล"); continue
        closes = df['Close'].squeeze().values.astype(float)
        # เปรียบเทียบ 5 วันทำการ
        week_ago = closes[-6] if len(closes)>=6 else closes[0]
        last     = closes[-1]
        chg_5d   = (last-week_ago)/week_ago*100
        chg_1y   = (last-closes[0])/closes[0]*100
        arrow    = "▲" if chg_5d>=0 else "▼"
        rsi_now  = compute_rsi(closes)[-1]
        rsi_tag  = _rsi_tag_short(rsi_now)
        lines.append(
            f"<b>{ticker}</b>  ${last:.2f}\n"
            f"   สัปดาห์นี้: {arrow} {abs(chg_5d):.2f}%  |  "
            f"ตั้งแต่ต้นปี: {'▲' if chg_1y>=0 else '▼'} {abs(chg_1y):.1f}%"
            f"{rsi_tag}"
        )
    lines.append("\n<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>")
    tg_send_text("\n".join(lines))
    log.info("  ✓ Weekly summary ส่งแล้ว")

# ═══════════════════════════════════════════════════════════════════
#  [+37] WATCHLIST — แจ้ง RSI เฉพาะ ไม่ส่งรูป
# ═══════════════════════════════════════════════════════════════════
def process_watchlist() -> None:
    if not WATCHLIST: return
    log.info(f"--- Watchlist: {WATCHLIST} ---")
    for ticker in WATCHLIST:
        try:
            df = download_with_retry(ticker, "5d", 2, 3)
            if df is None: continue
            closes  = df['Close'].squeeze().values.astype(float)
            rsi_now = float(compute_rsi(closes)[-1])
            price   = float(closes[-1])
            log.info(f"  Watchlist {ticker}: RSI={rsi_now:.1f}")
            tg_rsi_alert(ticker, rsi_now, price)
        except Exception as e:
            log.warning(f"  Watchlist {ticker}: {e}")

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
now_str   = datetime.now().strftime("%d/%m/%Y %H:%M")
date_str  = datetime.now().strftime("%Y%m%d")
run_start = datetime.now()
failed    = []

log.info("="*55)
log.info(f"BOSWaves v8.3 เริ่มทำงาน — {now_str}")
log.info(f"tickers: {TICKERS} | watchlist: {WATCHLIST}")

# ── ดึงข้อมูลตลาดรวม ───────────────────────────────────────────
log.info("ดึงข้อมูลตลาดรวม...")
fear_greed = fetch_fear_greed()
market_ctx = fetch_market_context()
gold       = fetch_gold_price()
log.info(f"  Fear&Greed: {fear_greed} | Market: {market_ctx}")

# ── ดาวน์โหลดราคา ──────────────────────────────────────────────
ticker_df = {}
for ticker in TICKERS:
    df = download_with_retry(ticker, DATA_PERIOD, DOWNLOAD_RETRIES, DOWNLOAD_DELAY)
    if df is not None:
        ticker_df[ticker] = df
        log.info(f"  {ticker}: {len(df)} bars")
    else:
        log.error(f"  {ticker}: download ล้มเหลว")
        failed.append(ticker)

# ── [+39] สรุปสัปดาห์วันเสาร์ ──────────────────────────────────
send_weekly_summary(ticker_df)

# ── [+37] Watchlist ─────────────────────────────────────────────
process_watchlist()

# ── Daily Summary ───────────────────────────────────────────────
if SEND_DAILY:
    log.info("--- Daily Summary ---")
    try:
        daily_out = build_daily_chart(ticker_df, date_str)
        if daily_out:
            ok = tg_send_photo(daily_out, build_daily_caption(ticker_df, market_ctx, fear_greed, gold))
            log.info("  ✓ Daily chart ส่งสำเร็จ" if ok else "  ✗ Daily chart ส่งล้มเหลว")
    except Exception as e:
        log.error(f"  Daily chart: {e}")
        tg_send_text(f"⚠️ <b>Daily Chart error</b>\n{e}")
else:
    log.info("  ข้าม Daily Summary (SEND_DAILY=false)")

# ── BOSWaves แต่ละ ticker ───────────────────────────────────────
gemini_notified = False
for ticker in TICKERS:
    if ticker in failed: continue
    log.info(f"--- {ticker} ---")
    t0 = datetime.now()
    try:
        df    = ticker_df[ticker]
        extra = fetch_extra_info(ticker)

        if SEND_NEWS:
            log.info(f"  ดึงข่าว...")
            headlines = fetch_yahoo_news(ticker, NEWS_PER_TICKER)
            news_th, gemini_err = translate_news_gemini(ticker, headlines)
            if gemini_err and not gemini_notified:
                tg_send_text(f"⚠️ <b>Gemini API มีปัญหา</b>\n<code>{gemini_err}</code>\n📰 ข่าวเป็นภาษาอังกฤษแทน")
                gemini_notified = True
        else:
            news_th = []; log.info(f"  ข้ามข่าว (SEND_NEWS=false)")

        close  = df['Close'].squeeze().values.astype(float)
        high   = df['High'].squeeze().values.astype(float)
        low    = df['Low'].squeeze().values.astype(float)
        volume = df['Volume'].squeeze().values.astype(float)
        dates  = df.index
        n      = len(close)
        dates_num = mdates.date2num(dates)
        bar_w  = float(np.diff(dates_num).mean()) if n>1 else 1.0

        atr_period_safe = min(ATR_PERIOD, n//2)
        current_atr     = float(compute_atr(high,low,close,atr_period_safe)[-1])
        rsi_vals        = compute_rsi(close,14)

        # [+36] RSI alert
        tg_rsi_alert(ticker, float(rsi_vals[-1]), float(close[-1]))

        ema50  = compute_ema(close, EMA_SHORT)
        ema200 = compute_ema(close, EMA_LONG)
        bias_label, bias_color = trend_bias(close, ema50, ema200, rsi_vals)
        bias_icon = "▲" if bias_label=="BULLISH" else "▼" if bias_label=="BEARISH" else "◆"

        swing_high, swing_low = find_swings(high, low, SWING_LENGTH)
        pivot_highs = [(i,float(high[i])) for i in range(n) if swing_high[i]]
        pivot_lows  = [(i,float(low[i]))  for i in range(n) if swing_low[i]]

        # ── Figure ──────────────────────────────────────────────
        fig = plt.figure(figsize=(14,12))
        gs  = fig.add_gridspec(3,1,height_ratios=[3.0,0.8,0.8],hspace=0.18)
        ax     = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1],sharex=ax)
        ax_vol = fig.add_subplot(gs[2],sharex=ax)

        ax.plot(dates_num,close,color='#3ECF8E',linewidth=1.5,alpha=0.9)
        ax.fill_between(dates_num,low,high,alpha=0.07,color='#3ECF8E')
        ax.plot(dates_num,ema50, color='#F6C90E',linewidth=1.0,alpha=0.8)
        ax.plot(dates_num,ema200,color='#60A5FA',linewidth=1.0,alpha=0.8,linestyle='--')

        for p_idx,p_price in (pivot_lows[-3:] if pivot_lows else []):
            d_s = dates_num[max(0,p_idx-5)]
            d_w = max(bar_w,dates_num[min(n-1,p_idx+5)]-d_s)
            ax.add_patch(Rectangle((d_s,p_price-ZONE_WIDTH_FACTOR*current_atr),
                d_w,ZONE_WIDTH_FACTOR*current_atr*2,
                facecolor='#3ECF8E',alpha=0.20,edgecolor='#3ECF8E',linewidth=0.8,zorder=2))
        for p_idx,p_price in (pivot_highs[-3:] if pivot_highs else []):
            d_s = dates_num[max(0,p_idx-5)]
            d_w = max(bar_w,dates_num[min(n-1,p_idx+5)]-d_s)
            ax.add_patch(Rectangle((d_s,p_price-ZONE_WIDTH_FACTOR*current_atr),
                d_w,ZONE_WIDTH_FACTOR*current_atr*2,
                facecolor='#FF5A5A',alpha=0.20,edgecolor='#FF5A5A',linewidth=0.8,zorder=2))

        for idx,price in pivot_highs[-6:]:
            ax.scatter(dates_num[idx],price,color='#FF5A5A',s=45,
                      marker='v',edgecolor='white',linewidth=0.5,zorder=5)
            if idx>=n-90:
                ax.annotate(f'PH\n{dates[idx].strftime("%d/%m")}',
                    (dates_num[idx],price),fontsize=7.5,color='#FF5A5A',
                    ha='center',va='bottom',xytext=(0,4),textcoords='offset points')
        for idx,price in pivot_lows[-6:]:
            ax.scatter(dates_num[idx],price,color='#3ECF8E',s=45,
                      marker='^',edgecolor='white',linewidth=0.5,zorder=5)
            if idx>=n-90:
                ax.annotate(f'PL\n{dates[idx].strftime("%d/%m")}',
                    (dates_num[idx],price),fontsize=7.5,color='#3ECF8E',
                    ha='center',va='top',xytext=(0,-4),textcoords='offset points')

        if pivot_highs:
            last_ph = pivot_highs[-1]; xmax = dates_num[-1]+bar_w*10
            ax.hlines(y=last_ph[1],xmin=dates_num[last_ph[0]],xmax=xmax,
                     color='#FF5A5A',linestyle='--',linewidth=0.8,alpha=0.7)
            ax.text(dates_num[-1]+bar_w*1.5,last_ph[1],' แนวต้าน BOS',
                   color='#FF5A5A',fontsize=8,va='center')

        entry_price = sl_price = tp_price = rr_ratio = None
        if pivot_lows and pivot_highs:
            fib_low=pivot_lows[-1][1]; fib_high=pivot_highs[-1][1]
            fib_range=fib_high-fib_low
            x_fs=dates_num[pivot_lows[-1][0]]; x_fe=dates_num[-1]+bar_w*10
            fib_cols=['#F6C90E','#F97316','#A855F7','#60A5FA']
            for ratio,col in zip(FIB_EXTENSIONS,fib_cols):
                lvl=fib_low+fib_range*ratio
                ax.hlines(y=lvl,xmin=x_fs,xmax=x_fe,color=col,linestyle=':',linewidth=0.8,alpha=0.6)
                ax.text(x_fe+bar_w*0.3,lvl,f' Fib {ratio:.3f}  ${lvl:.2f}',color=col,fontsize=7,va='center')
            entry_price=fib_low+0.1*current_atr; sl_price=fib_low-1.5*current_atr
            tp_price=fib_high; rr_ratio=(tp_price-entry_price)/max(0.01,entry_price-sl_price)

        ax.text(dates_num[-1],close[-1],f' ปัจจุบัน ${close[-1]:.2f}',
               color='white',fontsize=10,fontweight='bold',va='center',
               bbox=dict(boxstyle='round,pad=0.3',facecolor='#1A1F2B',edgecolor='#3ECF8E'))
        panel_text = (f"  [{ticker}  สรุปโครงสร้าง]\n"
                      f"  ราคา: ${close[-1]:.2f}   ATR({atr_period_safe}): ${current_atr:.2f}\n"
                      f"  RSI: {rsi_vals[-1]:.1f}   EMA{EMA_SHORT}: ${ema50[-1]:.2f}   EMA{EMA_LONG}: ${ema200[-1]:.2f}\n"
                      f"  แนวโน้ม: {bias_icon} {bias_label}")
        ax.text(0.015,0.97,panel_text,transform=ax.transAxes,fontsize=8.5,color='white',
               va='top',ha='left',zorder=10,
               bbox=dict(boxstyle='round,pad=0.5',facecolor='#1A1F2B',edgecolor=bias_color,alpha=0.93))

        legend_elements = [
            Rectangle((0,0),1,1,facecolor='#3ECF8E',alpha=0.25,edgecolor='#3ECF8E',label='โซนแนวรับ (Demand)'),
            Rectangle((0,0),1,1,facecolor='#FF5A5A',alpha=0.25,edgecolor='#FF5A5A',label='โซนแนวต้าน (Supply)'),
            Line2D([0],[0],marker='^',color='w',markerfacecolor='#3ECF8E',markersize=7,label='Pivot Low'),
            Line2D([0],[0],marker='v',color='w',markerfacecolor='#FF5A5A',markersize=7,label='Pivot High'),
            Line2D([0],[0],color='#F6C90E',linewidth=1.2,label=f'EMA {EMA_SHORT}'),
            Line2D([0],[0],color='#60A5FA',linewidth=1.2,linestyle='--',label=f'EMA {EMA_LONG}'),
        ]
        leg = ax.legend(handles=legend_elements,loc='upper left',
                       bbox_to_anchor=(0.015,0.73),fontsize=8,
                       facecolor='#1A1F2B',edgecolor='#3ECF8E',
                       labelcolor='white',framealpha=0.92,ncol=3)
        leg.set_zorder(10)

        ax_rsi.plot(dates_num,rsi_vals,color='#A855F7',linewidth=1.2)
        ax_rsi.axhline(70,color='#FF5A5A',linestyle=':',alpha=0.5)
        ax_rsi.axhline(50,color='#888888',linestyle=':',alpha=0.3)
        ax_rsi.axhline(30,color='#3ECF8E',linestyle=':',alpha=0.5)
        ax_rsi.fill_between(dates_num,rsi_vals,70,where=(rsi_vals>=70),color='#FF5A5A',alpha=0.25)
        ax_rsi.fill_between(dates_num,rsi_vals,30,where=(rsi_vals<=30),color='#3ECF8E',alpha=0.25)
        ax_rsi.text(dates_num[0],72,' Overbought > 70',color='#FF5A5A',fontsize=7.5)
        ax_rsi.text(dates_num[0],22,' Oversold < 30',  color='#3ECF8E',fontsize=7.5)
        ax_rsi.set_ylim(10,90); ax_rsi.set_ylabel('RSI (14)',color='white',fontsize=8.5)
        ax_rsi.tick_params(colors='white',labelsize=8); ax_rsi.grid(True,alpha=0.07,linestyle='--')

        vol_colors = np.where(np.concatenate(([0],np.diff(close)))>=0,'#3ECF8E','#FF5A5A')
        ax_vol.bar(dates_num,volume,width=bar_w*0.8,color=vol_colors,alpha=0.7)
        ax_vol.plot(dates_num,compute_ema(volume,20),color='#F6C90E',linewidth=0.9,alpha=0.8)
        ax_vol.set_ylabel('Volume',color='white',fontsize=8.5)
        ax_vol.tick_params(colors='white',labelsize=8); ax_vol.grid(True,alpha=0.07,linestyle='--')
        ax_vol.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x,_: f'{x/1e6:.1f}M' if x>=1e6 else f'{x/1e3:.0f}K'))

        ax.set_title(f'{ticker} — โครงสร้างราคาและทิศทางแนวโน้ม (BOSWaves v8.3)',
                    color='white',fontsize=13,fontweight='bold',pad=12)
        ax.text(0.998,0.995,f'สร้างเมื่อ {now_str}',
               transform=ax.transAxes,fontsize=7,color='#777777',ha='right',va='top')
        ax.set_ylabel('ราคา (USD)',color='white',fontsize=10); ax.tick_params(colors='white')
        ax.grid(True,alpha=0.07,linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xlim(dates_num[0]-bar_w*2,dates_num[-1]+bar_w*18)
        plt.setp(ax.get_xticklabels(),visible=False)
        plt.setp(ax_rsi.get_xticklabels(),visible=False)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_vol.tick_params(axis='x',colors='white',labelsize=8)

        if entry_price is not None:
            footer = (f"[แผนการเทรด {ticker}]  Entry: ${entry_price:.2f}   |   "
                      f"SL: ${sl_price:.2f}   |   TP: ${tp_price:.2f}  (R:R = 1:{rr_ratio:.1f})\n"
                      f"คำเตือน: หากราคาหลุด ${sl_price:.2f} โครงสร้างขาขึ้นสิ้นสุด")
        else:
            footer = f"[{ticker}]: ข้อมูลไม่เพียงพอสำหรับสร้างแผนการเทรด"
            entry_price=sl_price=tp_price=rr_ratio=0.0

        fig.text(0.5,0.045,footer,ha='center',va='center',fontsize=8,color='#3ECF8E',
                bbox=dict(boxstyle='round,pad=0.5',facecolor='#1A1F2B',edgecolor='#3ECF8E',alpha=0.95))
        fig.text(0.98,0.007,"* การวิเคราะห์นี้สร้างจากอัลกอริทึมทางเทคนิค ไม่ใช่คำแนะนำทางการเงิน",
                ha='right',va='bottom',fontsize=7,color='#555555')
        fig.subplots_adjust(bottom=0.12)
        out = OUTPUT_DIR / f"{ticker}_BOSWaves_{date_str}.png"
        plt.savefig(str(out),dpi=120,facecolor='#0B0E14',edgecolor='none',bbox_inches='tight')
        plt.close(fig)
        log.info(f"  บันทึกรูปสำเร็จ: {out.name}")

        caption = build_caption(ticker,close[-1],bias_label,rsi_vals[-1],current_atr,
                                entry_price,sl_price,tp_price,rr_ratio,extra,news_th)
        if SEND_CHART:
            ok = tg_send_photo(out, caption)
            if not ok: raise RuntimeError("tg_send_photo คืน False")
        else:
            # ส่งแค่ข้อความ ไม่มีรูป
            tg_send_text(f"<b>{ticker}</b>\n{caption}")
            log.info(f"  ส่งแค่ข้อความ (SEND_CHART=false)")
        log.info(f"  ✓ {ticker} เสร็จ ({(datetime.now()-t0).seconds}s)")

    except Exception as exc:
        log.error(f"  ✗ {ticker} ล้มเหลว: {exc}")
        failed.append(ticker)
        tg_send_text(f"⚠️ <b>BOSWaves error</b>\n{ticker}: {exc}")

# ═══════════════════════════════════════════════════════════════════
#  RUN SUMMARY + LOG
# ═══════════════════════════════════════════════════════════════════
elapsed_total = (datetime.now()-run_start).seconds
ok_tickers    = [t for t in TICKERS if t not in failed]
success       = len(failed)==0
log.info("="*55)
log.info(f"สรุป: สำเร็จ {len(ok_tickers)}/{len(TICKERS)} | ล้มเหลว: {','.join(failed) or '-'} | {elapsed_total}s")
log.info("="*55)

# [+38] บันทึก log ลง Supabase
log_run(ok_tickers, success, len(CHAT_IDS), elapsed_total,
        note=f"failed: {','.join(failed)}" if failed else "")

sys.exit(1 if failed else 0)
