"""
BOSWaves Swing Structure Forecast Chart — v8 (News + Market Context Edition)
─────────────────────────────────────────────────────────────────────────────
เพิ่มจาก v7.1:
  [+23] ข่าว Yahoo Finance RSS (ฟรี ไม่ต้อง key)
  [+24] Fear & Greed Index จาก CNN API
  [+25] VIX + S&P500 % วันนี้ (yfinance)
  [+26] แปลและสรุปข่าวเป็นไทยด้วย Gemini API (gemini-1.5-flash)
  [+27] caption Telegram รวม market context + ข่าวแปลไทย

.env ที่ต้องมี:
  BOT_TOKEN=...
  CHAT_ID=...
  GEMINI_API_KEY=...
"""

import os, sys, time, logging, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# ⚠️ อย่า hardcode token ที่นี่ — ใช้ .env หรือ GitHub Secrets เท่านั้น


warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")

from dotenv import load_dotenv
_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")   # โหลด .env ข้างๆ ไฟล์ (local)
# หมายเหตุ: บน GitHub Actions ตัวแปรจะมาจาก Secrets โดยตรง ไม่ต้องมี .env

BOT_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
TICKERS           = ["VOO", "NVDA", "JEPQ"]
DATA_PERIOD       = "1y"
SWING_LENGTH      = 16
ATR_PERIOD        = 14
ZONE_WIDTH_FACTOR = 0.4
FIB_EXTENSIONS    = [1.0, 1.272, 1.618, 2.0]
EMA_SHORT         = 50
EMA_LONG          = 200
# รองรับทั้ง local (Windows) และ GitHub Actions (Linux)
# กำหนดด้วย env var OUTPUT_DIR หรือใช้ folder "output" ข้างๆ ไฟล์
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR", _SCRIPT_DIR / "output"))
LOG_DIR           = OUTPUT_DIR / "logs"
DOWNLOAD_RETRIES  = 3
DOWNLOAD_DELAY    = 5
TG_TIMEOUT        = 30
NEWS_PER_TICKER   = 3          # จำนวนข่าวต่อ ticker

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

# guard
if not BOT_TOKEN or not CHAT_ID:
    log.error("BOT_TOKEN หรือ CHAT_ID ไม่พบใน .env — ยกเลิก")
    sys.exit(1)
if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY ไม่พบ — จะข้ามการแปลข่าว")

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
#  GEMINI — auto-detect model ที่ใช้งานได้จริง
# ═══════════════════════════════════════════════════════════════════
_gemini            = None
_gemini_model_name = None
_gemini_error      = None   # เก็บ error ไว้แจ้ง Telegram หลัง bot/chat พร้อม

def _pick_gemini_model(preferred: list) -> str | None:
    """เลือก model แรกที่มีอยู่จริงและรองรับ generateContent"""
    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        log.info(f"  Gemini models available: {available}")
        for name in preferred:
            if name in available:
                log.info(f"  เลือก Gemini model: {name}")
                return name
        # fallback: เอาตัวแรกที่มีคำว่า "flash" และไม่ใช่ image/tts
        flash = [m for m in available
                 if "flash" in m and "image" not in m and "tts" not in m]
        if flash:
            log.info(f"  Gemini fallback model: {flash[0]}")
            return flash[0]
        log.warning("  ไม่พบ Gemini flash model เลย")
    except Exception as e:
        log.warning(f"  list_models error: {e}")
        raise
    return None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _MODEL_PRIORITY = [
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    try:
        _gemini_model_name = _pick_gemini_model(_MODEL_PRIORITY)
        if _gemini_model_name:
            _gemini = genai.GenerativeModel(_gemini_model_name)
        else:
            _gemini_error = "ไม่พบ Gemini model ที่รองรับใน API"
            log.warning(f"  {_gemini_error}")
    except Exception as _e:
        _gemini_error = f"Gemini API error ตอน init: {_e}"
        log.warning(f"  {_gemini_error}")
else:
    _gemini_error = "ไม่มี GEMINI_API_KEY"

# ═══════════════════════════════════════════════════════════════════
#  FONT SETUP
# ═══════════════════════════════════════════════════════════════════
_FONT_CANDIDATES = [
    # Windows paths (local)
    (r"C:\Windows\Fonts\tahomabd.ttf", "Tahoma"),
    (r"C:\Windows\Fonts\tahoma.ttf",   "Tahoma"),
    (r"C:\Windows\Fonts\leelawdb.ttf", "Leelawadee UI"),
    (r"C:\Windows\Fonts\leelawad.ttf", "Leelawadee UI"),
    (r"C:\Windows\Fonts\angsa.ttf",    "Angsana New"),
    (r"C:\Windows\Fonts\angsab.ttf",   "Angsana New"),
    # Linux paths (GitHub Actions) — ติดตั้งผ่าน apt: fonts-thai-tlwg
    ("/usr/share/fonts/truetype/tlwg/Garuda.ttf",      "Garuda"),
    ("/usr/share/fonts/truetype/tlwg/Loma.ttf",        "Loma"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","DejaVu Sans"),
]
_loaded_families = []
for _fp, _fm in _FONT_CANDIDATES:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        if _fm not in _loaded_families:
            _loaded_families.append(_fm)

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
    n = len(high); tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.empty(n); atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr

def compute_rsi(close, period=14):
    delta = np.diff(close)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(len(close)); al = np.zeros(len(close))
    ag[period] = gain[:period].mean()
    al[period] = loss[:period].mean()
    for i in range(period+1, len(close)):
        ag[i] = (ag[i-1]*(period-1) + gain[i-1]) / period
        al[i] = (al[i-1]*(period-1) + loss[i-1]) / period
    rs = np.divide(ag, al, out=np.zeros_like(ag), where=al != 0)
    return 100 - (100 / (1 + rs))

def compute_ema(close, period):
    ema = np.empty(len(close)); ema[0] = close[0]
    k   = 2.0 / (period + 1)
    for i in range(1, len(close)):
        ema[i] = close[i]*k + ema[i-1]*(1-k)
    return ema

def find_swings(high, low, length=16):
    n = len(high)
    sh = np.zeros(n, dtype=bool); sl = np.zeros(n, dtype=bool)
    half = length // 2
    for i in range(half, n-half):
        win = range(i-half, i+half+1)
        if all(high[i] >= high[j] for j in win if j != i): sh[i] = True
        if all(low[i]  <= low[j]  for j in win if j != i): sl[i] = True
    return sh, sl

def trend_bias(close, ema50, ema200, rsi):
    if ema50[-1] > ema200[-1] and close[-1] > ema50[-1] and rsi[-1] > 50:
        return "BULLISH", "#3ECF8E"
    if ema50[-1] < ema200[-1] and close[-1] < ema50[-1] and rsi[-1] < 50:
        return "BEARISH", "#FF5A5A"
    return "NEUTRAL", "#F6C90E"

def download_with_retry(ticker, period, retries=3, delay=5):
    for attempt in range(1, retries+1):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            df = df[~df.index.duplicated(keep='first')].dropna()
            if not df.empty:
                return df
            log.warning(f"  attempt {attempt}: ได้ข้อมูลเปล่า")
        except Exception as e:
            log.warning(f"  attempt {attempt} error: {e}")
        if attempt < retries:
            time.sleep(delay)
    return None

def fetch_extra_info(ticker: str) -> dict:
    defaults = {"day_change_pct": None, "week52_high": None,
                "week52_low": None, "div_yield": None}
    try:
        info = yf.Ticker(ticker).fast_info
        defaults["week52_high"] = getattr(info, "year_high", None)
        defaults["week52_low"]  = getattr(info, "year_low",  None)
        prev = getattr(info, "previous_close", None)
        last = getattr(info, "last_price",     None)
        if prev and last and prev != 0:
            defaults["day_change_pct"] = float((last - prev) / prev * 100)
        full = yf.Ticker(ticker).info
        raw_dy = full.get("dividendYield", None)
        if raw_dy is not None:
            defaults["div_yield"] = float(
                raw_dy / 100 if raw_dy > 1.0 else raw_dy)
    except Exception as e:
        log.warning(f"  fetch_extra_info {ticker}: {e}")
    return defaults

# ═══════════════════════════════════════════════════════════════════
#  [+24] FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════════════
def fetch_fear_greed() -> dict:
    result = {"value": None, "label_th": "N/A"}

    def _label_th(val):
        if   val <= 25: return "กลัวมาก 😱"
        elif val <= 45: return "กลัว 😟"
        elif val <= 55: return "เป็นกลาง 😐"
        elif val <= 75: return "โลภ 😏"
        else:           return "โลภมาก 🤑"

    # แหล่งที่ 1: alternative.me
    try:
        r   = requests.get("https://api.alternative.me/fng/",
                           timeout=8,
                           headers={"User-Agent": "Mozilla/5.0"})
        val = int(r.json()["data"][0]["value"])
        result["value"]    = val
        result["label_th"] = _label_th(val)
        log.info(f"  Fear&Greed: {val} ({result['label_th']})")
        return result
    except Exception as e:
        log.warning(f"  fear_greed alternative.me: {e}")

    # fallback: CNN
    try:
        r   = requests.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        val = round(r.json()["fear_and_greed"]["score"], 1)
        result["value"]    = val
        result["label_th"] = _label_th(val)
    except Exception as e:
        log.warning(f"  fear_greed CNN: {e}")

    return result

# ═══════════════════════════════════════════════════════════════════
#  [+25] VIX + S&P500
# ═══════════════════════════════════════════════════════════════════
def fetch_market_context() -> dict:
    """ดึง VIX และ S&P500 % วันนี้"""
    result = {"vix": None, "sp500_chg": None}
    try:
        vix_data = yf.download("^VIX", period="2d", progress=False,
                               auto_adjust=True)
        if not vix_data.empty:
            result["vix"] = round(float(
                vix_data['Close'].squeeze().iloc[-1]), 2)

        sp_data = yf.download("^GSPC", period="2d", progress=False,
                              auto_adjust=True)
        if len(sp_data) >= 2:
            c = sp_data['Close'].squeeze().values
            result["sp500_chg"] = round((c[-1] - c[-2]) / c[-2] * 100, 2)
    except Exception as e:
        log.warning(f"  market_context error: {e}")
    return result

# ═══════════════════════════════════════════════════════════════════
#  [+23] Yahoo Finance RSS
# ═══════════════════════════════════════════════════════════════════
def fetch_yahoo_news(ticker: str, n: int = 3) -> list[str]:
    """ดึงพาดหัวข่าวล่าสุด n ข่าวจาก Yahoo Finance RSS"""
    headlines = []
    try:
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        for entry in feed.entries[:n]:
            headlines.append(entry.title.strip())
    except Exception as e:
        log.warning(f"  yahoo_news {ticker}: {e}")
    return headlines

# ═══════════════════════════════════════════════════════════════════
#  [+26] GEMINI — แปลและสรุปข่าว
# ═══════════════════════════════════════════════════════════════════
def translate_news_gemini(ticker: str, headlines: list[str]) -> tuple[list[str], str | None]:
    """ส่งพาดหัวข่าวให้ Gemini แปลและสรุปเป็นภาษาไทยสั้นๆ
    คืน (รายการข่าว, error_message หรือ None ถ้าสำเร็จ)"""
    if not _gemini or not headlines:
        err = _gemini_error or "ไม่มี Gemini / ไม่มีข่าว"
        return headlines, err

    prompt = (
        f"ต่อไปนี้คือพาดหัวข่าวหุ้น {ticker} จาก Yahoo Finance "
        f"จำนวน {len(headlines)} ข่าว\n\n"
        + "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
        + "\n\nกรุณาแปลและสรุปแต่ละข่าวเป็นภาษาไทย "
          "สั้นกระชับไม่เกิน 1 บรรทัดต่อข่าว "
          "ตอบแค่รายการ 1. 2. 3. เท่านั้น ไม่ต้องมีคำนำหรือคำอธิบายเพิ่ม"
    )
    try:
        resp  = _gemini.generate_content(prompt)
        lines = [l.strip() for l in resp.text.strip().splitlines()
                 if l.strip() and l.strip()[0].isdigit()]
        cleaned = []
        for l in lines:
            parts = l.split(". ", 1)
            cleaned.append(parts[1] if len(parts) > 1 else l)
        return (cleaned if cleaned else headlines), None   # สำเร็จ
    except Exception as e:
        err = f"Gemini translate error ({_gemini_model_name}): {e}"
        log.warning(f"  {err}")
        return headlines, err   # fallback + error

# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM HELPERS
# ═══════════════════════════════════════════════════════════════════
def tg_send_photo(image_path: Path, caption: str) -> bool:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption,
                      "parse_mode": "HTML"},
                files={"photo": f},
                timeout=TG_TIMEOUT,
            )
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        log.error(f"  Telegram error: {resp.text}")
        return False
    except Exception as e:
        log.error(f"  Telegram exception: {e}")
        return False

def tg_send_text(text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url,
                      data={"chat_id": CHAT_ID, "text": text,
                            "parse_mode": "HTML"},
                      timeout=TG_TIMEOUT)
    except Exception:
        pass

# ── [+27] build_caption ───────────────────────────────────────────
def build_caption(ticker, close_price, bias_label,
                  rsi, atr, entry, sl, tp, rr,
                  extra, fear_greed, market_ctx,
                  news_th: list[str]) -> str:

    bias_tag = ("🟢" if bias_label == "BULLISH" else
                "🔴" if bias_label == "BEARISH" else "🟡")
    bias_th  = ("แนวโน้มขาขึ้น" if bias_label == "BULLISH" else
                "แนวโน้มขาลง"  if bias_label == "BEARISH" else "แนวโน้มทรงตัว")

    chg = extra.get("day_change_pct")
    chg_str  = (f"+{chg:.2f}%" if chg and chg >= 0
                else f"{chg:.2f}%" if chg is not None else "N/A")
    chg_icon = "📈" if (chg or 0) >= 0 else "📉"

    w52h = extra.get("week52_high")
    w52l = extra.get("week52_low")
    w52_str  = (f"${w52l:.2f} – ${w52h:.2f}" if w52h and w52l else "N/A")

    dy = extra.get("div_yield")
    dy_str = f"{dy*100:.2f}%" if dy else "–"

    # Fear & Greed
    fg_val   = fear_greed.get("value")
    fg_label = fear_greed.get("label_th", "N/A")
    fg_str   = f"{fg_val:.0f} ({fg_label})" if fg_val is not None else "N/A"

    # Market
    sp_chg = market_ctx.get("sp500_chg")
    sp_str = (f"+{sp_chg:.2f}%" if sp_chg and sp_chg >= 0
              else f"{sp_chg:.2f}%" if sp_chg is not None else "N/A")
    vix    = market_ctx.get("vix")
    vix_str = f"{vix:.1f}" if vix else "N/A"

    # News
    news_block = ""
    if news_th:
        lines = "\n".join(f"• {h}" for h in news_th)
        news_block = f"━━━━━━━━━━━━━━━━\n📰 ข่าวล่าสุด:\n{lines}\n"

    return (
        f"<b>{ticker} — วิเคราะห์โครงสร้างราคา</b>\n"
        f"{bias_tag} {bias_th}  |  ราคา: <b>${close_price:.2f}</b>\n"
        f"{chg_icon} วันนี้: <b>{chg_str}</b>  |  ปันผล: {dy_str}\n"
        f"📊 52w: {w52_str}\n"
        f"RSI: {rsi:.1f}  |  ATR: ${atr:.2f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🌍 ตลาด: S&P500 {sp_str}  |  VIX {vix_str}\n"
        f"😱 Fear&Greed: {fg_str}\n"
        f"{news_block}"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 เข้าซื้อ: <b>${entry:.2f}</b>\n"
        f"🛑 ตัดขาดทุน: <b>${sl:.2f}</b>\n"
        f"🎯 เป้ากำไร: <b>${tp:.2f}</b>  (R:R 1:{rr:.1f})\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ ไม่ใช่คำแนะนำทางการเงิน</i>"
    )

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
now_str   = datetime.now().strftime("%d/%m/%Y %H:%M")
run_start = datetime.now()
failed    = []

log.info("=" * 55)
log.info(f"BOSWaves v8 เริ่มทำงาน — {now_str}")
log.info(f"tickers: {TICKERS}")

# ── ดึงข้อมูลตลาดรวม (ดึงครั้งเดียว ใช้ทุก ticker) ────────────
log.info("ดึงข้อมูลตลาดรวม...")
fear_greed  = fetch_fear_greed()
market_ctx  = fetch_market_context()
log.info(f"  Fear&Greed: {fear_greed}")
log.info(f"  Market: {market_ctx}")

for ticker in TICKERS:
    log.info(f"--- {ticker} ---")
    t0 = datetime.now()

    try:
        # Download price
        df = download_with_retry(ticker, DATA_PERIOD,
                                 DOWNLOAD_RETRIES, DOWNLOAD_DELAY)
        if df is None:
            raise ValueError(f"download ล้มเหลวทุก {DOWNLOAD_RETRIES} attempt")
        log.info(f"  download OK — {len(df)} bars")

        extra = fetch_extra_info(ticker)

        # [+23][+26] ข่าว + แปลด้วย Gemini
        log.info(f"  ดึงข่าว {ticker}...")
        headlines = fetch_yahoo_news(ticker, NEWS_PER_TICKER)
        log.info(f"  ได้ข่าว {len(headlines)} ข่าว")
        news_th, gemini_err = translate_news_gemini(ticker, headlines)
        if gemini_err:
            log.warning(f"  Gemini ไม่ทำงาน: {gemini_err}")
            # แจ้ง Telegram ว่า Gemini มีปัญหา (แจ้งแค่ ticker แรก ไม่แจ้งซ้ำ)
            if ticker == TICKERS[0]:
                tg_send_text(
                    f"⚠️ <b>BOSWaves — Gemini API มีปัญหา</b>\n"
                    f"📌 Model: <code>{_gemini_model_name or 'ไม่พบ'}</code>\n"
                    f"❌ Error: <code>{gemini_err}</code>\n"
                    f"📰 ข่าวจะแสดงเป็นภาษาอังกฤษแทน"
                )
        else:
            log.info(f"  แปลข่าวเสร็จ (model: {_gemini_model_name})")

        close  = df['Close'].squeeze().values.astype(float)
        high   = df['High'].squeeze().values.astype(float)
        low    = df['Low'].squeeze().values.astype(float)
        volume = df['Volume'].squeeze().values.astype(float)
        dates  = df.index
        n      = len(close)
        dates_num = mdates.date2num(dates)
        bar_w  = float(np.diff(dates_num).mean()) if n > 1 else 1.0

        atr_period_safe = min(ATR_PERIOD, n // 2)
        current_atr     = float(compute_atr(high, low, close, atr_period_safe)[-1])
        rsi_vals        = compute_rsi(close, 14)
        ema50           = compute_ema(close, EMA_SHORT)
        ema200          = compute_ema(close, EMA_LONG)
        bias_label, bias_color = trend_bias(close, ema50, ema200, rsi_vals)
        bias_icon = ("▲" if bias_label == "BULLISH" else
                     "▼" if bias_label == "BEARISH" else "◆")

        swing_high, swing_low = find_swings(high, low, SWING_LENGTH)
        pivot_highs = [(i, float(high[i])) for i in range(n) if swing_high[i]]
        pivot_lows  = [(i, float(low[i]))  for i in range(n) if swing_low[i]]

        # ── Figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 12))
        gs  = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.8, 0.8], hspace=0.18)
        ax     = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1], sharex=ax)
        ax_vol = fig.add_subplot(gs[2], sharex=ax)

        ax.plot(dates_num, close, color='#3ECF8E', linewidth=1.5, alpha=0.9)
        ax.fill_between(dates_num, low, high, alpha=0.07, color='#3ECF8E')
        ax.plot(dates_num, ema50,  color='#F6C90E', linewidth=1.0, alpha=0.8)
        ax.plot(dates_num, ema200, color='#60A5FA', linewidth=1.0,
                alpha=0.8, linestyle='--')

        for p_idx, p_price in (pivot_lows[-3:] if pivot_lows else []):
            d_s = dates_num[max(0, p_idx-5)]
            d_w = max(bar_w, dates_num[min(n-1, p_idx+5)] - d_s)
            ax.add_patch(Rectangle((d_s, p_price - ZONE_WIDTH_FACTOR*current_atr),
                                   d_w, ZONE_WIDTH_FACTOR*current_atr*2,
                                   facecolor='#3ECF8E', alpha=0.20,
                                   edgecolor='#3ECF8E', linewidth=0.8, zorder=2))

        for p_idx, p_price in (pivot_highs[-3:] if pivot_highs else []):
            d_s = dates_num[max(0, p_idx-5)]
            d_w = max(bar_w, dates_num[min(n-1, p_idx+5)] - d_s)
            ax.add_patch(Rectangle((d_s, p_price - ZONE_WIDTH_FACTOR*current_atr),
                                   d_w, ZONE_WIDTH_FACTOR*current_atr*2,
                                   facecolor='#FF5A5A', alpha=0.20,
                                   edgecolor='#FF5A5A', linewidth=0.8, zorder=2))

        for idx, price in pivot_highs[-6:]:
            ax.scatter(dates_num[idx], price, color='#FF5A5A', s=45,
                       marker='v', edgecolor='white', linewidth=0.5, zorder=5)
            if idx >= n - 90:
                ax.annotate(f'PH\n{dates[idx].strftime("%d/%m")}',
                            (dates_num[idx], price), fontsize=7.5,
                            color='#FF5A5A', ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points')

        for idx, price in pivot_lows[-6:]:
            ax.scatter(dates_num[idx], price, color='#3ECF8E', s=45,
                       marker='^', edgecolor='white', linewidth=0.5, zorder=5)
            if idx >= n - 90:
                ax.annotate(f'PL\n{dates[idx].strftime("%d/%m")}',
                            (dates_num[idx], price), fontsize=7.5,
                            color='#3ECF8E', ha='center', va='top',
                            xytext=(0, -4), textcoords='offset points')

        if pivot_highs:
            last_ph = pivot_highs[-1]
            xmax    = dates_num[-1] + bar_w * 10
            ax.hlines(y=last_ph[1], xmin=dates_num[last_ph[0]], xmax=xmax,
                      color='#FF5A5A', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(dates_num[-1] + bar_w*1.5, last_ph[1],
                    ' แนวต้าน BOS', color='#FF5A5A', fontsize=8, va='center')

        entry_price = sl_price = tp_price = rr_ratio = None
        if pivot_lows and pivot_highs:
            fib_low   = pivot_lows[-1][1]
            fib_high  = pivot_highs[-1][1]
            fib_range = fib_high - fib_low
            x_fs = dates_num[pivot_lows[-1][0]]
            x_fe = dates_num[-1] + bar_w * 10
            fib_cols = ['#F6C90E', '#F97316', '#A855F7', '#60A5FA']
            for ratio, col in zip(FIB_EXTENSIONS, fib_cols):
                lvl = fib_low + fib_range * ratio
                ax.hlines(y=lvl, xmin=x_fs, xmax=x_fe,
                          color=col, linestyle=':', linewidth=0.8, alpha=0.6)
                ax.text(x_fe + bar_w*0.3, lvl,
                        f' Fib {ratio:.3f}  ${lvl:.2f}',
                        color=col, fontsize=7, va='center')
            entry_price = fib_low + 0.1 * current_atr
            sl_price    = fib_low - 1.5 * current_atr
            tp_price    = fib_high
            rr_ratio    = (tp_price - entry_price) / max(0.01, entry_price - sl_price)

        ax.text(dates_num[-1], close[-1], f' ปัจจุบัน ${close[-1]:.2f}',
                color='white', fontsize=10, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A1F2B',
                          edgecolor='#3ECF8E'))

        panel_text = (
            f"  [{ticker}  สรุปโครงสร้าง]\n"
            f"  ราคา: ${close[-1]:.2f}   ATR({atr_period_safe}): ${current_atr:.2f}\n"
            f"  RSI: {rsi_vals[-1]:.1f}   "
            f"EMA{EMA_SHORT}: ${ema50[-1]:.2f}   EMA{EMA_LONG}: ${ema200[-1]:.2f}\n"
            f"  แนวโน้ม: {bias_icon} {bias_label}"
        )
        ax.text(0.015, 0.97, panel_text,
                transform=ax.transAxes, fontsize=8.5, color='white',
                va='top', ha='left', zorder=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1F2B',
                          edgecolor=bias_color, alpha=0.93))

        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#3ECF8E', alpha=0.25,
                      edgecolor='#3ECF8E', label='โซนแนวรับ (Demand)'),
            Rectangle((0,0),1,1, facecolor='#FF5A5A', alpha=0.25,
                      edgecolor='#FF5A5A', label='โซนแนวต้าน (Supply)'),
            Line2D([0],[0], marker='^', color='w', markerfacecolor='#3ECF8E',
                   markersize=7, label='Pivot Low'),
            Line2D([0],[0], marker='v', color='w', markerfacecolor='#FF5A5A',
                   markersize=7, label='Pivot High'),
            Line2D([0],[0], color='#F6C90E', linewidth=1.2, label=f'EMA {EMA_SHORT}'),
            Line2D([0],[0], color='#60A5FA', linewidth=1.2,
                   linestyle='--', label=f'EMA {EMA_LONG}'),
        ]
        leg = ax.legend(handles=legend_elements, loc='upper left',
                        bbox_to_anchor=(0.015, 0.73), fontsize=8,
                        facecolor='#1A1F2B', edgecolor='#3ECF8E',
                        labelcolor='white', framealpha=0.92, ncol=3)
        leg.set_zorder(10)

        ax_rsi.plot(dates_num, rsi_vals, color='#A855F7', linewidth=1.2)
        ax_rsi.axhline(70, color='#FF5A5A', linestyle=':', alpha=0.5)
        ax_rsi.axhline(50, color='#888888', linestyle=':', alpha=0.3)
        ax_rsi.axhline(30, color='#3ECF8E', linestyle=':', alpha=0.5)
        ax_rsi.fill_between(dates_num, rsi_vals, 70,
                            where=(rsi_vals >= 70), color='#FF5A5A', alpha=0.25)
        ax_rsi.fill_between(dates_num, rsi_vals, 30,
                            where=(rsi_vals <= 30), color='#3ECF8E', alpha=0.25)
        ax_rsi.text(dates_num[0], 72, ' Overbought > 70',
                    color='#FF5A5A', fontsize=7.5)
        ax_rsi.text(dates_num[0], 22, ' Oversold < 30',
                    color='#3ECF8E', fontsize=7.5)
        ax_rsi.set_ylim(10, 90)
        ax_rsi.set_ylabel('RSI (14)', color='white', fontsize=8.5)
        ax_rsi.tick_params(colors='white', labelsize=8)
        ax_rsi.grid(True, alpha=0.07, linestyle='--')

        vol_colors = np.where(
            np.concatenate(([0], np.diff(close))) >= 0, '#3ECF8E', '#FF5A5A')
        ax_vol.bar(dates_num, volume, width=bar_w*0.8,
                   color=vol_colors, alpha=0.7)
        ax_vol.plot(dates_num, compute_ema(volume, 20),
                    color='#F6C90E', linewidth=0.9, alpha=0.8)
        ax_vol.set_ylabel('Volume', color='white', fontsize=8.5)
        ax_vol.tick_params(colors='white', labelsize=8)
        ax_vol.grid(True, alpha=0.07, linestyle='--')
        ax_vol.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

        ax.set_title(
            f'{ticker} — โครงสร้างราคาและทิศทางแนวโน้ม (BOSWaves v8)',
            color='white', fontsize=13, fontweight='bold', pad=12)
        ax.text(0.998, 0.995, f'สร้างเมื่อ {now_str}',
                transform=ax.transAxes, fontsize=7, color='#777777',
                ha='right', va='top')
        ax.set_ylabel('ราคา (USD)', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.07, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xlim(dates_num[0] - bar_w*2, dates_num[-1] + bar_w*18)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_vol.tick_params(axis='x', colors='white', labelsize=8)

        if entry_price is not None:
            footer = (
                f"[แผนการเทรด {ticker}]  "
                f"Entry: ${entry_price:.2f}   |   SL: ${sl_price:.2f}   |   "
                f"TP: ${tp_price:.2f}  (R:R = 1:{rr_ratio:.1f})\n"
                f"คำเตือน: หากราคาหลุด ${sl_price:.2f} โครงสร้างขาขึ้นสิ้นสุด"
            )
        else:
            footer = f"[{ticker}]: ข้อมูลไม่เพียงพอสำหรับสร้างแผนการเทรด"
            entry_price = sl_price = tp_price = rr_ratio = 0.0

        fig.text(0.5, 0.045, footer, ha='center', va='center',
                 fontsize=8, color='#3ECF8E',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1F2B',
                           edgecolor='#3ECF8E', alpha=0.95))
        fig.text(0.98, 0.007,
                 "* การวิเคราะห์นี้สร้างจากอัลกอริทึมทางเทคนิค "
                 "ไม่ใช่คำแนะนำทางการเงิน ผู้ลงทุนควรบริหารความเสี่ยงด้วยตนเอง",
                 ha='right', va='bottom', fontsize=7, color='#555555')

        fig.subplots_adjust(bottom=0.12)
        out = OUTPUT_DIR / f"{ticker}_BOSWaves_v8.png"
        plt.savefig(str(out), dpi=120, facecolor='#0B0E14',
                    edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        log.info(f"  บันทึกรูปสำเร็จ: {out.name}")

        # ── ส่ง Telegram ──────────────────────────────────────────
        caption = build_caption(
            ticker, close[-1], bias_label,
            rsi_vals[-1], current_atr,
            entry_price, sl_price, tp_price, rr_ratio,
            extra, fear_greed, market_ctx, news_th)

        log.info(f"  ส่ง Telegram...")
        ok = tg_send_photo(out, caption)
        if ok:
            elapsed = (datetime.now() - t0).seconds
            log.info(f"  ✓ {ticker} เสร็จ ({elapsed}s)")
        else:
            raise RuntimeError("tg_send_photo คืน False")

    except Exception as exc:
        log.error(f"  ✗ {ticker} ล้มเหลว: {exc}")
        failed.append(ticker)
        tg_send_text(f"⚠️ <b>BOSWaves cronjob error</b>\n{ticker}: {exc}")

# ═══════════════════════════════════════════════════════════════════
#  RUN SUMMARY
# ═══════════════════════════════════════════════════════════════════
elapsed_total = (datetime.now() - run_start).seconds
ok_tickers    = [t for t in TICKERS if t not in failed]
log.info("=" * 55)
log.info(f"สรุป: สำเร็จ {len(ok_tickers)}/{len(TICKERS)} "
         f"({', '.join(ok_tickers) or '-'}) | "
         f"ล้มเหลว: {', '.join(failed) or '-'} | "
         f"ใช้เวลา {elapsed_total}s")
log.info("=" * 55)

sys.exit(1 if failed else 0)
