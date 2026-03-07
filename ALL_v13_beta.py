# ==========================================
# 強勢股戰情室 V12
# V2: Bug修復 + 評分系統 + 大盤狀態 + 產業泡泡圖
# V3: 長紅K回溯天數、均量基準、量能門檻可調
# V4: 回測深度過濾
# V5: 表格移除整理振幅、量能倍數、評分欄位
# V6: 下載加速（period縮短至300d + chunk_size擴大至150）
# V7: 加入 10MA > 20MA > 60MA 多頭排列濾網
# V8: 移除市場潛力名單
# V10: 整合處置股策略（第二分頁）+ 底量濾網
# V11: 修正處置股爬蟲 SSL 憑證驗證問題 (加上 verify=False 與關閉警告)
# V12: 整合處置股掃描按鈕至左側欄，與主策略一鍵執行；修改狙擊手量能預設值為0.7倍
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import requests
from bs4 import BeautifulSoup
import time
import random
import importlib
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# 加入這兩行來關閉 SSL 憑證警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 📋 日誌設定
# ==========================================
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 🔥 匯入外部細產業資料庫 (強化版)
# ==========================================
SECTOR_DB = {}

try:
    import sector_data
    importlib.reload(sector_data)
    if hasattr(sector_data, 'CUSTOM_SECTOR_MAP'):
        raw_map = sector_data.CUSTOM_SECTOR_MAP
        SECTOR_DB = {str(k).strip(): v for k, v in raw_map.items()}
except ImportError:
    pass
except Exception as e:
    st.error(f"❌ `sector_data.py` 載入錯誤: {e}")
    logger.error(f"sector_data.py 載入錯誤: {e}")

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(
    page_title="強勢股戰情室 V12",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import twstock
except ImportError:
    st.error("❌ 缺少 `twstock` 套件，請輸入 `pip install twstock` 安裝")
    st.stop()

# Google Sheets（可選）
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🕐 交易時段判斷
# ==========================================
def is_trading_hours():
    """判斷目前是否在台股交易時段 (09:00~13:30)"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # 週六日
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

def get_cache_ttl():
    return 300 if is_trading_hours() else 3600

# ==========================================
# 🧠 策略核心邏輯類別 (Backtesting用)
# ==========================================
def SMA(array, n):
    return pd.Series(array).rolling(window=n).mean()

class SniperStrategy(Strategy):
    ma_trend_period = 60
    ma_long_period = 240
    ma_base_exit = 20
    ma_fast_exit = 10
    vol_ma_period = 5
    big_candle_pct = 0.05
    min_volume_shares = 2000000
    lookback_window = 10
    use_year_line = True
    defense_buffer = 0.01

    def init(self):
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)
        self.ma_trend = self.I(SMA, close, self.ma_trend_period)
        self.ma_base = self.I(SMA, close, self.ma_base_exit)
        self.ma_fast = self.I(SMA, close, self.ma_fast_exit)
        self.vol_ma = self.I(SMA, volume, self.vol_ma_period)
        if self.use_year_line:
            self.ma_long = self.I(SMA, close, self.ma_long_period)
        self.setup_active = False
        self.setup_bar_index = 0
        self.setup_low_price = 0
        self.defense_price = 0

    def next(self):
        price = self.data.Close[-1]
        prev_high = self.data.High[-2]

        if self.position:
            if price < self.defense_price:
                self.position.close()
                return
            current_profit_pct = self.position.pl_pct
            exit_line = self.ma_fast[-1] if current_profit_pct > 0.15 else self.ma_base[-1]
            if price < exit_line:
                self.position.close()
            return

        triggered_buy = False
        days_since_setup = len(self.data) - self.setup_bar_index

        if self.setup_active:
            if days_since_setup > self.lookback_window:
                self.setup_active = False
            elif price < self.defense_price:
                self.setup_active = False
            elif price > prev_high:
                self.buy()
                self.setup_active = False
                triggered_buy = True
                return

        if not triggered_buy:
            if self.data.Volume[-1] < self.min_volume_shares: return
            is_trend_up = (price > self.ma_trend[-1]) and (self.ma_trend[-1] > self.ma_trend[-2])
            if self.use_year_line and (pd.isna(self.ma_long[-1]) or price < self.ma_long[-1]): return

            prev_close = self.data.Close[-2]
            open_price = self.data.Open[-1]
            change_pct = (price - prev_close) / prev_close
            is_big = change_pct > self.big_candle_pct
            is_vol = self.data.Volume[-1] > self.vol_ma[-1]
            is_red = price > open_price

            if is_trend_up and is_big and is_vol and is_red:
                self.setup_active = True
                self.setup_bar_index = len(self.data)
                self.setup_low_price = self.data.Low[-1]
                prev_high_setup = self.data.High[-2]
                prev_close_setup = self.data.Close[-2]
                if self.data.Low[-1] > prev_high_setup:
                    base_val = prev_close_setup
                else:
                    base_val = self.data.Low[-1]
                self.defense_price = base_val * (1 - self.defense_buffer)

# ==========================================
# 🛠️ 輔助函式與資料庫
# ==========================================
def get_detailed_sector(code, standard_group=None, custom_db=None):
    code_str = str(code).strip()
    if custom_db and code_str in custom_db:
        return str(custom_db[code_str])
    if standard_group and str(standard_group) not in ['nan', 'None', '', 'NaN']:
        return str(standard_group)
    try:
        if code_str in twstock.codes:
            group = twstock.codes[code_str].group
            if group and str(group) not in ['nan', 'None', '', 'NaN']:
                return group
    except Exception as e:
        logger.warning(f"get_detailed_sector 錯誤 [{code}]: {e}")
    return "其他"

@st.cache_data(ttl=3600)
def get_stock_info_map():
    try:
        stock_map = {}
        for code, info in twstock.twse.items():
            if len(code) == 4:
                stock_map[code] = {
                    'name': f"{code} {info.name}",
                    'symbol': f"{code}.TW",
                    'short_name': info.name,
                    'group': getattr(info, 'group', '其他')
                }
        for code, info in twstock.tpex.items():
            if len(code) == 4:
                stock_map[code] = {
                    'name': f"{code} {info.name}",
                    'symbol': f"{code}.TWO",
                    'short_name': info.name,
                    'group': getattr(info, 'group', '其他')
                }
        return stock_map
    except Exception as e:
        logger.error(f"get_stock_info_map 錯誤: {e}")
        return {}

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)
def fetch_history_data(symbol, start_date=None, end_date=None, period="2y"):
    try:
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        if df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.warning(f"fetch_history_data 錯誤 [{symbol}]: {e}")
        return None

def get_stock_data_with_realtime(code, symbol, analysis_date_str):
    df = fetch_history_data(symbol)
    if df is None or df.empty:
        return None

    last_dt = df.index[-1].strftime('%Y-%m-%d')
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')

    if analysis_date_str == today_str and last_dt != today_str:
        try:
            realtime = twstock.realtime.get(code)
            if realtime['success'] and realtime['realtime']['latest_trade_price'] != '-':
                rt = realtime['realtime']
                new_row = pd.Series({
                    'Open': float(rt['open']), 'High': float(rt['high']),
                    'Low': float(rt['low']), 'Close': float(rt['latest_trade_price']),
                    'Volume': float(rt['accumulate_trade_volume']) * 1000
                }, name=pd.Timestamp(today_str))
                df = pd.concat([df, new_row.to_frame().T])
        except Exception as e:
            logger.warning(f"get_stock_data_with_realtime 即時資料錯誤 [{code}]: {e}")
    return df

# ==========================================
# 🐂 BULL_v7 Google Sheets 持倉同步
# ==========================================
BULL_SHEET_COLS = ["symbol","name","進場日期","進場價","上次加碼價","持倉最高價","加碼次數","加碼紀錄"]
BULL_SCOPES     = ["https://www.googleapis.com/auth/spreadsheets"]

def bull_use_gsheet() -> bool:
    if not GSPREAD_AVAILABLE:
        return False
    try:
        return ("gcp_service_account" in st.secrets and "sheets" in st.secrets)
    except Exception:
        return False

def bull_get_ws():
    """取得 BULL 持倉的 worksheet"""
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=BULL_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["sheets"]["sheet_id"])
        try:
            ws = sh.worksheet("bull_positions")
        except Exception:
            ws = sh.add_worksheet(title="bull_positions", rows=1000, cols=20)
            ws.append_row(BULL_SHEET_COLS)
        return ws
    except Exception as e:
        logger.warning(f"bull_get_ws 失敗: {e}")
        return None

def bull_gs_load() -> pd.DataFrame:
    ws = bull_get_ws()
    if ws is None:
        return pd.DataFrame(columns=BULL_SHEET_COLS)
    try:
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=BULL_SHEET_COLS)
        df = pd.DataFrame(data)
        df["加碼次數"]  = pd.to_numeric(df["加碼次數"],  errors="coerce").fillna(0).astype(int)
        df["進場價"]    = pd.to_numeric(df["進場價"],    errors="coerce").fillna(0)
        df["上次加碼價"]= pd.to_numeric(df["上次加碼價"],errors="coerce").fillna(0)
        df["持倉最高價"]= pd.to_numeric(df["持倉最高價"],errors="coerce").fillna(0)
        return df[BULL_SHEET_COLS]
    except Exception:
        return pd.DataFrame(columns=BULL_SHEET_COLS)

def bull_gs_save(df: pd.DataFrame):
    ws = bull_get_ws()
    if ws is None:
        return
    try:
        ws.clear()
        ws.append_row(BULL_SHEET_COLS)
        if len(df) > 0:
            rows = df[BULL_SHEET_COLS].fillna("").values.tolist()
            ws.append_rows(rows, value_input_option="USER_ENTERED")
    except Exception as e:
        logger.warning(f"bull_gs_save 失敗: {e}")


# ==========================================
# 🐂 BULL_v7 布林滾雪球策略（掃描器用）
# ==========================================
BULL_PARAMS = {
    "boll_period"    : 15,
    "boll_std"       : 2.1,
    "squeeze_n"      : 15,
    "squeeze_lookback": 5,
    "vol_ma_days"    : 5,
    "vol_ratio"      : 1.5,
    "sma_trend_days" : 3,
    "min_vol_shares" : 1_000_000,
    "vol_ma20_days"  : 20,
    "vol_heavy_days" : 5,
    "vol_shrink_days": 15,
    "init_position"  : 0.5,
    "add_position"   : 0.5,
    "addon_b_profit" : 0.10,
}

def bull_calc_indicators(df, params):
    """計算 BULL_v7 所需指標"""
    df = df.copy()
    # 統一欄位名稱
    df.columns = [c.capitalize() for c in df.columns]
    needed = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[needed].copy()

    close  = df["Close"]
    volume = df["Volume"]
    bp     = params["boll_period"]

    df["SMA"]        = close.rolling(bp).mean()
    df["Std"]        = close.rolling(bp).std()
    df["Upper"]      = df["SMA"] + df["Std"] * params["boll_std"]
    df["Lower"]      = df["SMA"] - df["Std"] * params["boll_std"]
    df["Bandwidth"]  = df["Upper"] - df["Lower"]
    df["Vol_MA"]     = volume.rolling(params["vol_ma_days"]).mean()

    sq_n = params["squeeze_n"]
    df["BW_Min"]         = df["Bandwidth"].rolling(sq_n).min()
    df["Is_Squeeze"]     = df["Bandwidth"] == df["BW_Min"]
    lb = params["squeeze_lookback"]
    df["Squeeze_Recent"] = df["Is_Squeeze"].shift(1).rolling(lb).max() == 1

    df["SMA_Up"]     = df["SMA"] > df["SMA"].shift(params["sma_trend_days"])
    df["Vol_MA20"]   = volume.rolling(params["vol_ma20_days"]).mean()
    df["Vol_Heavy"]  = volume.rolling(params["vol_heavy_days"]).mean()
    df["Vol_Shrink"] = volume.rolling(params["vol_shrink_days"]).mean()

    return df


def bull_check_entry(df, params):
    """今日是否符合 BULL_v7 進場條件"""
    if len(df) < params["boll_period"] + params["squeeze_n"]:
        return False
    r = df.iloc[-1]
    if pd.isna(r["Squeeze_Recent"]) or pd.isna(r["Vol_MA20"]):
        return False
    squeeze  = bool(r["Squeeze_Recent"])
    sma_up   = bool(r["SMA_Up"])
    breakout = float(r["Close"]) > float(r["Upper"])
    vol_ok   = float(r["Volume"]) > float(r["Vol_MA"]) * params["vol_ratio"]
    min_vol  = float(r["Vol_MA20"]) > params["min_vol_shares"]
    return squeeze and sma_up and breakout and vol_ok and min_vol


def bull_scan_one(code, info, df_raw, params, pos_map, scan_date_ts):
    """掃描單支股票，回傳進場/加碼A/加碼B/出場訊號"""
    result = {"entry": None, "addon_a": None, "addon_b": None, "exit": None, "high_update": None}
    try:
        df = bull_calc_indicators(df_raw, params)
        # 依掃描日期截斷
        df = df[df.index <= scan_date_ts]
        if len(df) < 50:
            return result

        sym    = info["symbol"]
        name   = info["short_name"]
        r      = df.iloc[-1]
        r1     = df.iloc[-2]
        c      = float(r["Close"])
        sma_i  = float(r["SMA"])
        lo_i   = float(r["Lower"])
        l_i    = float(r["Low"])
        vol_i  = float(r["Volume"])
        vheavy = float(r["Vol_Heavy"]) if not pd.isna(r["Vol_Heavy"]) else 0

        in_pos = sym in pos_map

        if not in_pos:
            # 進場訊號
            if bull_check_entry(df, params):
                result["entry"] = {
                    "代號": code, "名稱": name, "symbol": sym,
                    "收盤價": round(c, 2),
                    "上軌": round(float(r["Upper"]), 2),
                    "15MA": round(sma_i, 2),
                    "成交量": int(vol_i),
                    "5MA量": int(r["Vol_MA"]) if not pd.isna(r["Vol_MA"]) else 0,
                }
        else:
            pos = pos_map[sym]
            entry_price      = float(pos["進場價"])
            last_addon_price = float(pos["上次加碼價"])
            peak_price       = float(pos["持倉最高價"])

            # 更新最高價
            if c > peak_price:
                result["high_update"] = (sym, round(c, 2))

            # 出場優先：出量跌破15MA 或 最低碰下軌
            heavy_break = (c < sma_i) and (vol_i >= vheavy) and vheavy > 0
            low_break   = l_i <= lo_i
            if heavy_break or low_break:
                reason = "出量跌破15MA" if heavy_break else "最低碰下軌"
                result["exit"] = {
                    "代號": code, "名稱": name, "symbol": sym,
                    "收盤價": round(c, 2), "15MA": round(sma_i, 2),
                    "進場價": round(entry_price, 2),
                    "損益%": round((c - entry_price) / entry_price * 100, 2),
                    "加碼次數": int(pos.get("加碼次數", 0)),
                    "出場原因": reason,
                }
                return result

            # 加碼A：昨日量縮跌破15MA，今日站回
            vshrink = float(r1["Vol_Shrink"]) if not pd.isna(r1["Vol_Shrink"]) else 0
            y_below   = float(r1["Close"]) < float(r1["SMA"])
            y_vol_low = float(r1["Volume"]) < vshrink if vshrink > 0 else False
            if y_below and y_vol_low and c >= sma_i:
                result["addon_a"] = {
                    "代號": code, "名稱": name, "symbol": sym,
                    "收盤價": round(c, 2), "15MA": round(sma_i, 2),
                    "加碼次數": int(pos.get("加碼次數", 0)),
                    "加碼類型": "A 回測站回",
                }

            # 加碼B：突破持倉最高價 且 距上次加碼>=10%
            profit_from_last = (c - last_addon_price) / last_addon_price if last_addon_price > 0 else 0
            if c > peak_price and profit_from_last >= params["addon_b_profit"]:
                result["addon_b"] = {
                    "代號": code, "名稱": name, "symbol": sym,
                    "收盤價": round(c, 2),
                    "持倉最高價": round(peak_price, 2),
                    "距上次加碼": f"+{profit_from_last*100:.1f}%",
                    "加碼次數": int(pos.get("加碼次數", 0)),
                    "加碼類型": "B 突破新高",
                }
    except Exception as e:
        logger.warning(f"bull_scan_one 錯誤 [{code}]: {e}")
    return result


def bull_run_scan(stock_map, all_data, scan_date_str, bull_positions, bull_params, max_workers=16):
    """執行 BULL_v7 全市場掃描"""
    scan_date_ts = pd.Timestamp(scan_date_str)
    pos_map = {row["symbol"]: row for _, row in bull_positions.iterrows()} if len(bull_positions) > 0 else {}

    entry_list  = []
    addon_a_list = []
    addon_b_list = []
    exit_list   = []
    high_updates = []

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for code, df_raw in all_data.items():
            if code not in stock_map:
                continue
            info = stock_map[code]
            sym  = info["symbol"]
            futures[executor.submit(bull_scan_one, code, info, df_raw, bull_params, pos_map, scan_date_ts)] = code

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["entry"]     : entry_list.append(res["entry"])
            if res["addon_a"]   : addon_a_list.append(res["addon_a"])
            if res["addon_b"]   : addon_b_list.append(res["addon_b"])
            if res["exit"]      : exit_list.append(res["exit"])
            if res["high_update"]: high_updates.append(res["high_update"])

    return {
        "entry": entry_list, "addon_a": addon_a_list,
        "addon_b": addon_b_list, "exit": exit_list,
        "high_updates": high_updates
    }


# ==========================================
# 🚀 批量下載加速模組
# ==========================================
def fetch_data_batch(stock_map, period="300d", chunk_size=150):
    all_symbols = [info['symbol'] for info in stock_map.values()]
    data_store = {}
    symbol_to_code = {v['symbol']: k for k, v in stock_map.items()}

    total_chunks = (len(all_symbols) // chunk_size) + 1
    progress_text = st.empty()
    bar = st.progress(0)

    for i in range(0, len(all_symbols), chunk_size):
        chunk = all_symbols[i:i + chunk_size]
        if not chunk:
            continue

        chunk_idx = (i // chunk_size) + 1
        progress_text.text(f"📥 正在批量下載歷史資料... (批次 {chunk_idx}/{total_chunks})")
        bar.progress(chunk_idx / total_chunks)

        try:
            tickers_str = " ".join(chunk)
            batch_df = yf.download(tickers_str, period=period, group_by='ticker', threads=True, auto_adjust=True, progress=False)

            if not batch_df.empty:
                if isinstance(batch_df.columns, pd.MultiIndex):
                    for symbol in chunk:
                        try:
                            if symbol in batch_df:
                                stock_df = batch_df[symbol].dropna()
                                if not stock_df.empty:
                                    if stock_df.index.tz is not None:
                                        stock_df.index = stock_df.index.tz_localize(None)
                                    code = symbol_to_code.get(symbol)
                                    if code:
                                        data_store[code] = stock_df
                        except Exception as e:
                            logger.warning(f"批次解析錯誤 [{symbol}]: {e}")
                else:
                    symbol = chunk[0]
                    stock_df = batch_df.dropna()
                    if not stock_df.empty:
                        if stock_df.index.tz is not None:
                            stock_df.index = stock_df.index.tz_localize(None)
                        code = symbol_to_code.get(symbol)
                        if code:
                            data_store[code] = stock_df
            time.sleep(1)
        except Exception as e:
            logger.error(f"批次下載錯誤 (batch {chunk_idx}): {e}")
            st.toast(f"批次下載錯誤: {e}")
            continue

    progress_text.empty()
    bar.empty()
    return data_store

def fetch_realtime_batch(codes_list, chunk_size=50):
    """
    即時報價批量取得。
    優先用證交所/櫃買 STOCK_DAY_ALL API（一次回傳全部，極速）。
    若 API 失敗才 fallback 到 twstock（逐批，較慢）。
    """
    realtime_data = {}
    progress_text = st.empty()

    # ── 方案A：證交所 STOCK_DAY_ALL（上市，一次全拿）──
    try:
        progress_text.text("⚡ 取得即時報價（證交所）...")
        import urllib.request, json as _json
        twse_url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL?response=json"
        req = urllib.request.Request(twse_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = _json.loads(r.read())
        if d.get("stat") == "OK" and d.get("data"):
            # fields: 證券代號,證券名稱,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
            fields = d["fields"]
            fi = {f: i for i, f in enumerate(fields)}
            for row in d["data"]:
                try:
                    c = row[fi["證券代號"]].strip()
                    def _n(s):
                        return float(str(s).replace(",","").replace("+","").replace("--","0")) if str(s) not in ("--","") else 0.0
                    realtime_data[c] = {
                        "latest_trade_price": str(_n(row[fi["收盤價"]])),
                        "open":   str(_n(row[fi["開盤價"]])),
                        "high":   str(_n(row[fi["最高價"]])),
                        "low":    str(_n(row[fi["最低價"]])),
                        "accumulate_trade_volume": str(int(_n(row[fi["成交股數"]]) / 1000)),
                    }
                except Exception:
                    continue
        logger.info(f"證交所即時報價取得 {len(realtime_data)} 筆")
    except Exception as e:
        logger.warning(f"證交所 STOCK_DAY_ALL 失敗: {e}，改用 twstock")

    # ── 方案B：櫃買中心（上櫃，一次全拿）──
    try:
        progress_text.text("⚡ 取得即時報價（櫃買中心）...")
        import urllib.request, json as _json
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        req = urllib.request.Request(tpex_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            rows = _json.loads(r.read())
        for row in rows:
            try:
                c = str(row.get("SecuritiesCompanyCode","")).strip()
                def _n2(s):
                    return float(str(s).replace(",","")) if str(s) not in ("--","","N/A") else 0.0
                realtime_data[c] = {
                    "latest_trade_price": str(_n2(row.get("Close","0"))),
                    "open":   str(_n2(row.get("Open","0"))),
                    "high":   str(_n2(row.get("High","0"))),
                    "low":    str(_n2(row.get("Low","0"))),
                    "accumulate_trade_volume": str(int(_n2(row.get("TradeVolume","0")) / 1000)),
                }
            except Exception:
                continue
        logger.info(f"櫃買即時報價合計 {len(realtime_data)} 筆")
    except Exception as e:
        logger.warning(f"櫃買 tpex_mainboard_quotes 失敗: {e}")

    # ── 方案C：若兩個 API 都失敗，才 fallback twstock（逐批）──
    if not realtime_data:
        logger.warning("即時 API 全部失敗，fallback 到 twstock（速度較慢）")
        codes_needed = [c for c in codes_list if c not in realtime_data]
        for i in range(0, len(codes_needed), chunk_size):
            chunk = codes_needed[i:i + chunk_size]
            progress_text.text(f"⚡ twstock fallback... ({i}/{len(codes_needed)})")
            try:
                stocks = twstock.realtime.get(chunk)
                if stocks:
                    if 'success' in stocks:
                        if stocks['success']:
                            realtime_data[stocks['info']['code']] = stocks['realtime']
                    else:
                        for code, data in stocks.items():
                            if isinstance(data, dict) and data.get('success'):
                                realtime_data[code] = data['realtime']
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"twstock fallback 錯誤: {e}")

    progress_text.empty()
    return realtime_data

# ==========================================
# 📈 大盤狀態判斷
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_status(analysis_date_str):
    try:
        tw = yf.Ticker("^TWII")
        df = tw.history(period="1y")
        if df.empty or df.index.tz is not None:
            df.index = df.index.tz_localize(None) if not df.empty else df.index
        if df.empty:
            return {'strong': True, 'score': 5, 'label': '無法取得大盤資料'}

        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values:
            latest_idx = -1
        else:
            latest_idx = df.index.get_loc(pd.Timestamp(analysis_date_str))

        close = df['Close']
        ma20  = close.rolling(20).mean()
        ma60  = close.rolling(60).mean()
        c     = close.iloc[latest_idx]
        m20   = ma20.iloc[latest_idx]
        m60   = ma60.iloc[latest_idx]

        score = 0
        if c > m20:  score += 4
        if c > m60:  score += 3
        if m20 > m60: score += 3

        if score >= 7:
            label = "🟢 大盤強勢"
        elif score >= 4:
            label = "🟡 大盤偏弱"
        else:
            label = "🔴 大盤弱勢"

        return {'strong': score >= 7, 'score': score, 'label': label,
                'close': round(c, 0), 'ma20': round(m20, 0), 'ma60': round(m60, 0)}
    except Exception as e:
        logger.warning(f"fetch_market_status 錯誤: {e}")
        return {'strong': True, 'score': 5, 'label': '大盤資料異常'}

# ==========================================
# 🎯 處置股策略模組
# ==========================================
def is_valid_stock_code(code):
    if not code or not code.isdigit():
        return False
    if len(code) != 4:
        return False
    num = int(code)
    if num >= 9000:
        return False
    return True

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_disposal_stocks():
    disposal = {}

    def roc_to_date(s):
        s = s.strip()
        parts = s.replace('/', '-').split('-')
        y = int(parts[0].strip()) + 1911
        m = int(parts[1].strip())
        d = int(parts[2].strip())
        return datetime.date(y, m, d)

    try:
        today_dt = datetime.date.today()
        start_query = today_dt - datetime.timedelta(days=30)
        end_query   = today_dt + datetime.timedelta(days=30)
        url = (
            f"https://www.twse.com.tw/rwd/zh/announcement/punish"
            f"?response=json"
            f"&startDate={start_query.strftime('%Y%m%d')}"
            f"&endDate={end_query.strftime('%Y%m%d')}"
        )
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, */*',
            'Referer': 'https://www.twse.com.tw/',
        }
        res = requests.get(url, headers=headers, timeout=15, verify=False)
        data = res.json()

        if data.get('stat') == 'OK':
            twse_temp = {}
            for row in data.get('data', []):
                try:
                    code = str(row[2]).strip()
                    name = str(row[3]).strip()
                    period_str = str(row[6]).strip()

                    if '～' in period_str:
                        parts = period_str.split('～')
                    elif '~' in period_str:
                        parts = period_str.split('~')
                    else:
                        continue

                    start_dt = roc_to_date(parts[0].strip())
                    end_dt   = roc_to_date(parts[1].strip())

                    disposal_content = ''
                    if len(row) > 7: disposal_content += str(row[7])
                    if len(row) > 8: disposal_content += str(row[8])
                    is_5min = '每五分鐘' in disposal_content or '每5分鐘' in disposal_content

                    if is_valid_stock_code(code) and is_5min:
                        if code not in twse_temp or end_dt > twse_temp[code]['end']:
                            twse_temp[code] = {
                                'name': name,
                                'start': start_dt,
                                'end':   end_dt,
                                'market': 'twse',
                                'symbol': f"{code}.TW",
                                'match_type': '5分鐘撮合'
                            }
                except Exception as e:
                    pass
            disposal.update(twse_temp)
    except Exception as e:
        logger.warning(f"上市處置股查詢異常: {e}")

    tpex_urls = [
        "https://www.tpex.org.tw/web/stock/aftertrading/disposal_stock/dispost_result.php?l=zh-tw",
        "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information",
    ]
    for url in tpex_urls:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/html, */*',
                'Referer': 'https://www.tpex.org.tw/',
            }
            res = requests.get(url, headers=headers, timeout=15, verify=False)
            res.encoding = 'utf-8'

            if res.headers.get('content-type','').startswith('application/json') or res.text.strip().startswith('['):
                rows = res.json()
                if isinstance(rows, list):
                    for row in rows:
                        try:
                            code = str(row.get('SecuritiesCompanyCode', row.get('code',''))).strip()
                            name = str(row.get('CompanyName', row.get('name',''))).strip()
                            period = str(row.get('DispositionPeriod', '')).strip()
                            if '~' in period:
                                parts = period.split('~')
                                start_s = parts[0].strip()
                                end_s   = parts[1].strip()
                                def roc8_to_date(s):
                                    s = s.strip()
                                    y = int(s[0:3]) + 1911
                                    m = int(s[3:5])
                                    d = int(s[5:7])
                                    return datetime.date(y, m, d)
                                start_dt = roc8_to_date(start_s)
                                end_dt   = roc8_to_date(end_s)
                            else:
                                start_s = str(row.get('DisposalStartDate', row.get('start',''))).strip()
                                end_s   = str(row.get('DisposalEndDate',   row.get('end',''))).strip()
                                start_dt = roc_to_date(start_s)
                                end_dt   = roc_to_date(end_s)

                            disposal_content = str(row.get('DisposalCondition', row.get('DispositionReasons', '')))
                            is_5min = '每5分鐘' in disposal_content or '每五分鐘' in disposal_content
                            if is_valid_stock_code(code) and is_5min:
                                disposal[code] = {
                                    'name': name,
                                    'start': start_dt,
                                    'end':   end_dt,
                                    'market': 'tpex',
                                    'symbol': f"{code}.TWO",
                                    'match_type': '5分鐘撮合'
                                }
                        except Exception as e:
                            pass
                    break

            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.find('table')
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = [c.get_text(strip=True) for c in row.find_all('td')]
                    if len(cols) >= 4:
                        try:
                            code = cols[0].strip()
                            name = cols[1].strip()
                            if is_valid_stock_code(code):
                                disposal[code] = {
                                    'name': name,
                                    'start': roc_to_date(cols[2]),
                                    'end':   roc_to_date(cols[3]),
                                    'market': 'tpex',
                                    'symbol': f"{code}.TWO"
                                }
                        except Exception as e:
                            pass
                break
        except Exception as e:
            continue

    return disposal

def analyze_disposal_stock(code, info, analysis_date, min_vol=0):
    try:
        symbol = info['symbol']
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", auto_adjust=True)

        if df.empty or len(df) < 10:
            return None

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if min_vol > 0:
            _s = pd.Timestamp(info['start'])
            _e = pd.Timestamp(analysis_date)
            _dv = df[(_s <= df.index) & (df.index <= _e)]
            if not _dv.empty and _dv['Volume'].min() < min_vol * 1000:
                return None

        df['ma5']  = df['Close'].rolling(5).mean()
        df['ma10'] = df['Close'].rolling(10).mean()

        start_ts = pd.Timestamp(info['start'])
        end_ts   = pd.Timestamp(analysis_date)
        df_period = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

        if len(df_period) < 2:
            return None

        analysis_date_str = analysis_date.strftime('%Y-%m-%d')
        position = None
        trades   = []
        latest_signal = None

        for i in range(1, len(df_period)):
            today      = df_period.index[i]
            today_str  = today.strftime('%Y-%m-%d')
            c_today    = df_period['Close'].iloc[i]
            prev_high  = df_period['High'].iloc[i-1]
            ma5_today  = df_period['ma5'].iloc[i]
            ma10_today = df_period['ma10'].iloc[i]
            pct_today  = (c_today - df_period['Close'].iloc[i-1]) / df_period['Close'].iloc[i-1] * 100

            if position is None:
                if c_today > prev_high:
                    position = {
                        'entry_date':  today_str,
                        'entry_price': c_today,
                    }
                    latest_signal = {
                        'status': '🟢 持有中',
                        'entry_date':  today_str,
                        'entry_price': f"{c_today:.2f}",
                        'exit_date':   '-',
                        'exit_price':  '-',
                        'profit_pct':  '-',
                        'exit_reason': '-',
                        'signal_date': today_str,
                        'signal_close': f"{c_today:.2f}",
                        'signal_prev_high': f"{prev_high:.2f}",
                        'signal_pct': f"{pct_today:+.2f}%",
                    }
            else:
                entry_price = position['entry_price']
                profit_pct  = (c_today - entry_price) / entry_price * 100

                if profit_pct >= 15.0:
                    exit_ma    = ma5_today
                    exit_label = '5MA'
                else:
                    exit_ma    = ma10_today
                    exit_label = '10MA'

                should_exit = pd.notna(exit_ma) and c_today < exit_ma

                if should_exit:
                    trades.append({
                        'entry_date':  position['entry_date'],
                        'entry_price': entry_price,
                        'exit_date':   today_str,
                        'exit_price':  c_today,
                        'profit_pct':  profit_pct,
                        'exit_reason': f"跌破{exit_label}",
                    })
                    position = None
                    latest_signal = None
                else:
                    latest_signal = {
                        'status': '🟢 持有中',
                        'entry_date':  position['entry_date'],
                        'entry_price': f"{entry_price:.2f}",
                        'exit_date':   '-',
                        'exit_price':  '-',
                        'profit_pct':  f"{profit_pct:+.2f}%",
                        'exit_reason': f"出場條件：跌破{'5MA' if profit_pct >= 15 else '10MA'}",
                        'signal_date': position['entry_date'],
                        'signal_close': f"{entry_price:.2f}",
                        'signal_prev_high': '-',
                        'signal_pct': '-',
                    }

        today_date_str = analysis_date_str
        is_new_signal_today = (
            latest_signal is not None and
            latest_signal.get('entry_date') == today_date_str and
            latest_signal.get('status') == '🟢 持有中'
        )

        if latest_signal is None and not trades:
            return None

        last_trade = trades[-1] if trades else None

        if latest_signal:
            status = latest_signal['status']
        elif last_trade:
            status = '⚫ 已出場'
        else:
            status = '-'

        days_left = (info['end'] - analysis_date).days

        return {
            'code':             code,
            'name':             info['name'],
            'market':           '上市' if info['market'] == 'twse' else '上櫃',
            'disposal_start':   info['start'].strftime('%Y-%m-%d'),
            'disposal_end':     info['end'].strftime('%Y-%m-%d'),
            'days_left':        max(days_left, 0),
            'status':           status,
            'is_new_today':     is_new_signal_today,
            'entry_date':       latest_signal['entry_date'] if latest_signal else (last_trade['entry_date'] if last_trade else '-'),
            'entry_price':      latest_signal['entry_price'] if latest_signal else (f"{last_trade['entry_price']:.2f}" if last_trade else '-'),
            'exit_date':        latest_signal.get('exit_date', '-') if latest_signal else (last_trade['exit_date'] if last_trade else '-'),
            'exit_price':       latest_signal.get('exit_price', '-') if latest_signal else (f"{last_trade['exit_price']:.2f}" if last_trade else '-'),
            'profit_pct':       latest_signal.get('profit_pct', '-') if latest_signal else (f"{last_trade['profit_pct']:+.2f}%" if last_trade else '-'),
            'exit_reason':      latest_signal.get('exit_reason', '-') if latest_signal else (last_trade['exit_reason'] if last_trade else '-'),
            'signal_date':      latest_signal.get('signal_date', '-') if latest_signal else '-',
            'signal_close':     latest_signal.get('signal_close', '-') if latest_signal else '-',
            'signal_prev_high': latest_signal.get('signal_prev_high', '-') if latest_signal else '-',
            'signal_pct':       latest_signal.get('signal_pct', '-') if latest_signal else '-',
            'total_trades':     len(trades),
        }

    except Exception as e:
        logger.warning(f"analyze_disposal_stock 錯誤 [{code}]: {e}")
        return None

def show_disposal_table(data):
    if not data:
        return
    df = pd.DataFrame(data)

    col_map = {
        'code':             '代號',
        'name':             '名稱',
        'market':           '市場',
        'disposal_start':   '處置開始',
        'disposal_end':     '處置結束',
        'days_left':        '剩餘天數',
        'status':           '狀態',
        'entry_date':       '進場日',
        'entry_price':      '進場價',
        'exit_date':        '出場日',
        'exit_price':       '出場價',
        'profit_pct':       '損益',
        'exit_reason':      '出場條件',
        'signal_date':      '訊號日',
        'signal_close':     '訊號收盤',
        'signal_prev_high': '前日高點',
        'signal_pct':       '訊號漲幅',
    }

    show_cols = [c for c in col_map if c in df.columns]
    df = df[show_cols].rename(columns=col_map)
    st.dataframe(df, width='stretch', hide_index=True)

# ==========================================
# 🏆 N字品質評分
# ==========================================
def calc_sniper_score(c_today, prev_h, defense_price, s_high, s_low, s_close,
                      setup_idx, idx, high, low, volume, op, market_score, pullback_depth=0):
    score = 0
    details = {}

    if c_today > 0:
        risk_pct = (c_today - defense_price) / c_today * 100
        details['風險距離'] = round(risk_pct, 1)
        if risk_pct <= 3.0:   score += 35
        elif risk_pct <= 5.0: score += 25
        elif risk_pct <= 8.0: score += 15
        else:                 score += 0
    else:
        details['風險距離'] = 0

    if setup_idx > 0 and idx > setup_idx:
        consolidation_vol = volume.iloc[setup_idx:idx].mean()
        today_vol = volume.iloc[idx]
        vol_ratio = today_vol / consolidation_vol if consolidation_vol > 0 else 0
        details['量能倍數'] = round(vol_ratio, 1)
        if 2.0 <= vol_ratio <= 4.0:   score += 25
        elif 1.5 <= vol_ratio < 2.0:  score += 18
        elif vol_ratio >= 4.0:        score += 12
        elif vol_ratio >= 1.0:        score += 8
        else:                         score += 0
    else:
        details['量能倍數'] = 0

    h_today = high.iloc[idx]
    l_today = low.iloc[idx]
    candle_range = h_today - l_today
    if candle_range > 0:
        close_strength = (c_today - l_today) / candle_range
        details['收盤強度'] = round(close_strength, 2)
        if close_strength >= 0.85:   score += 20
        elif close_strength >= 0.70: score += 14
        elif close_strength >= 0.50: score += 7
        else:                        score += 0
    else:
        details['收盤強度'] = 0

    if setup_idx > 0:
        seg_high = high.iloc[setup_idx:idx].max()
        seg_low  = low.iloc[setup_idx:idx].min()
        if seg_low > 0:
            consolidation_range = (seg_high - seg_low) / seg_low * 100
            details['整理振幅'] = round(consolidation_range, 1)
            if consolidation_range <= 8:    score += 15
            elif consolidation_range <= 12: score += 10
            elif consolidation_range <= 18: score += 5
            else:                           score += 0
        else:
            details['整理振幅'] = 0
    else:
        details['整理振幅'] = 0

    details['回測深度'] = round(pullback_depth, 1)
    if pullback_depth <= 20:    score += 10
    elif pullback_depth <= 38:  score += 7
    elif pullback_depth <= 50:  score += 3

    market_pts = round(market_score / 10 * 5)
    score += market_pts
    details['大盤分'] = market_pts

    return min(score, 100), details

# ==========================================
# 🧠 綜合分析引擎
# ==========================================
def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None, market_score=5):
    try:
        if pre_loaded_df is not None:
            df = pre_loaded_df.copy()
        else:
            df = get_stock_data_with_realtime(code, info['symbol'], analysis_date_str)

        if df is None or df.empty:
            return "無法取得資料"
        if len(df) < 200:
            return "資料長度不足 (<200天)"

        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values:
            return f"無 {analysis_date_str} 交易資料"
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        op = df['Open']
        stock_name = info['short_name']
        sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db)

        result_sniper = None
        result_day = None

        s_ma_trend = params['s_ma_trend']
        s_use_year = params['s_use_year']
        s_big_candle = params['s_big_candle']
        s_min_vol = params['s_min_vol']
        s_setup_lookback = params.get('s_setup_lookback', 25)
        s_vol_ma_days = params.get('s_vol_ma_days', 20)
        s_vol_ratio = params.get('s_vol_ratio', 0.7)
        s_pullback_max = params.get('s_pullback_max', 50)

        ma_t = close.rolling(window=s_ma_trend).mean()
        ma_y = close.rolling(window=240).mean()
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()
        ma60 = close.rolling(window=60).mean()
        vol_ma = volume.rolling(window=5).mean()
        vol_ma_setup = volume.rolling(window=s_vol_ma_days).mean()

        is_sniper_candidate = True
        if volume.iloc[idx] < s_min_vol: is_sniper_candidate = False
        if s_use_year and len(ma_y) > idx and (pd.isna(ma_y.iloc[idx]) or close.iloc[idx] < ma_y.iloc[idx]): is_sniper_candidate = False
        if not (close.iloc[idx] > ma_t.iloc[idx] and ma_t.iloc[idx] > ma_t.iloc[idx-1]): is_sniper_candidate = False
        if not (pd.notna(ma10.iloc[idx]) and pd.notna(ma20.iloc[idx]) and pd.notna(ma60.iloc[idx]) and
                close.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma60.iloc[idx]): is_sniper_candidate = False

        if is_sniper_candidate:
            is_setup = ((close.iloc[idx] - close.iloc[idx-1]) / close.iloc[idx-1] > s_big_candle and
                        volume.iloc[idx] > vol_ma_setup.iloc[idx] * s_vol_ratio and close.iloc[idx] > op.iloc[idx])

            setup_found = False
            s_high = 0; s_low = 0; s_close = 0; s_date = ""; setup_idx = -1
            defense_price = 0

            for k in range(1, s_setup_lookback + 1):
                b_idx = idx - k
                if b_idx < 1: break

                if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > s_big_candle and
                    volume.iloc[b_idx] > vol_ma_setup.iloc[b_idx] * s_vol_ratio and close.iloc[b_idx] > op.iloc[b_idx]):

                    setup_found = True; setup_idx = b_idx
                    s_low = low.iloc[b_idx]
                    s_high = high.iloc[b_idx]
                    s_close = close.iloc[b_idx]
                    s_date = df.index[b_idx].strftime('%Y-%m-%d')

                    prev_high_setup = high.iloc[b_idx-1]
                    prev_close_setup = close.iloc[b_idx-1]
                    prev_open_setup = op.iloc[b_idx-1]

                    if s_low > prev_high_setup:
                        if prev_close_setup >= prev_open_setup:
                            base_val = prev_close_setup
                        else:
                            base_val = prev_open_setup
                    else:
                        base_val = s_low

                    defense_price = base_val * 0.99
                    break

            c_today = close.iloc[idx]
            prev_close_today = close.iloc[idx-1]
            prev_h = high.iloc[idx-1]
            daily_pct = (c_today - prev_close_today) / prev_close_today * 100

            if setup_found:
                is_broken = False; dropped_below_high = False
                for k in range(setup_idx + 1, idx + 1):
                    c_k = close.iloc[k]
                    if c_k < defense_price: is_broken = True; break
                    if c_k < s_high: dropped_below_high = True

                if setup_idx + 1 < idx:
                    consolidation_close_min = close.iloc[setup_idx + 1 : idx].min()
                else:
                    consolidation_close_min = s_close
                pullback_depth = (s_close - consolidation_close_min) / s_close * 100
                is_pullback_ok = pullback_depth <= s_pullback_max

                if not is_broken and is_pullback_ok:
                    is_breakout = c_today > prev_h
                    is_gap_breakout = (op.iloc[idx] > high.iloc[idx-1]) and (close.iloc[idx] > op.iloc[idx])

                    if not dropped_below_high:
                        pct_from_setup = (c_today - s_close) / s_close
                        if pct_from_setup <= 0.10:
                            if is_breakout:
                                score, score_details = calc_sniper_score(
                                    c_today, prev_h, defense_price, s_high, s_low, s_close,
                                    setup_idx, idx, high, low, volume, op, market_score, pullback_depth)
                                risk_pct = score_details.get('風險距離', 0)
                                risk_tag = "✅" if risk_pct <= 5.0 else "⚠️"
                                pb_depth = score_details.get('回測深度', 0)
                                pb_tag = "🔒" if pb_depth <= 38 else "📊"
                                result_sniper = ("triggered", {
                                    "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                    "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                    "狀態": "🚀 強勢突破", "訊號日": s_date,
                                    "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}",
                                    "風險距離": f"{risk_tag}{risk_pct:.1f}%",
                                    "回測深度": f"{pb_tag}{pb_depth:.1f}%",
                                    "sort_pct": daily_pct,
                                    "_score": score,
                                    "_risk_pct": risk_pct,
                                    "_setup_date": s_date, "_defense": defense_price,
                                    "_signal_high": s_high, "_signal_low": s_low
                                })
                            else:
                                result_sniper = ("watching", {
                                    "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                    "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                    "狀態": "💪 強勢整理", "訊號日": s_date,
                                    "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}",
                                    "sort_pct": daily_pct,
                                    "_setup_date": s_date, "_defense": defense_price,
                                    "_signal_high": s_high, "_signal_low": s_low
                                })
                    else:
                        prev_close_valid = close.iloc[idx-1] <= (s_high * 1.02)
                        if (is_breakout and prev_close_valid) or is_gap_breakout:
                            status_str = "🚀 N字跳空" if is_gap_breakout else "🎯 N字突破"
                            score, score_details = calc_sniper_score(
                                c_today, prev_h, defense_price, s_high, s_low, s_close,
                                setup_idx, idx, high, low, volume, op, market_score, pullback_depth)
                            risk_pct = score_details.get('風險距離', 0)
                            risk_tag = "✅" if risk_pct <= 5.0 else "⚠️"
                            pb_depth = score_details.get('回測深度', 0)
                            pb_tag = "🔒" if pb_depth <= 38 else "📊"
                            result_sniper = ("triggered", {
                                "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                "狀態": status_str, "訊號日": s_date,
                                "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}",
                                "風險距離": f"{risk_tag}{risk_pct:.1f}%",
                                "回測深度": f"{pb_tag}{pb_depth:.1f}%",
                                "sort_pct": daily_pct,
                                "_score": score,
                                "_risk_pct": risk_pct,
                                "_setup_date": s_date, "_defense": defense_price,
                                "_signal_high": s_high, "_signal_low": s_low
                            })
                        else:
                            result_sniper = ("watching", {
                                "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                "狀態": "📉 回檔整理", "訊號日": s_date,
                                "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}",
                                "sort_pct": daily_pct,
                                "_setup_date": s_date, "_defense": defense_price,
                                "_signal_high": s_high, "_signal_low": s_low
                            })

            elif is_setup:
                prev_c = close.iloc[idx-1]
                pct_chg = (c_today - prev_c) / prev_c * 100
                result_sniper = ("new_setup", {
                    "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                    "產業": sector_name, "狀態": "🔥 剛起漲",
                    "漲幅": f"{pct_chg:+.2f}%", "sort_pct": pct_chg,
                    "_setup_date": df.index[idx].strftime('%Y-%m-%d'),
                    "_defense": defense_price,
                    "_signal_high": high.iloc[idx], "_signal_low": low.iloc[idx]
                })

        d_period = params['d_period']
        d_threshold = params['d_threshold']
        d_min_vol = params['d_min_vol']
        d_min_pct = params['d_min_pct']

        d_close = close.iloc[idx]; d_open = op.iloc[idx]; d_high = high.iloc[idx]
        d_volume = volume.iloc[idx]; d_prev_close = close.iloc[idx-1]

        is_red = d_close > d_open
        upper_shadow = (d_high - d_close) / d_close
        is_strong_close = upper_shadow < 0.01
        pct_chg_val = (d_close - d_prev_close) / d_prev_close
        is_momentum_ok = (pct_chg_val > d_min_pct/100) and (pct_chg_val < 0.095)
        is_vol_ok = (d_volume / 1000) > d_min_vol

        if idx >= d_period:
            prev_period_high = high.iloc[idx-d_period : idx].max()
            threshold_factor = 1 - (d_threshold / 100)
            is_near_high = d_close >= (prev_period_high * threshold_factor)
            is_not_new_high = d_high <= prev_period_high

            if is_red and is_strong_close and is_momentum_ok and is_vol_ok and is_near_high and is_not_new_high:
                dist_to_high = (d_close - prev_period_high) / prev_period_high * 100
                result_day = {
                    "代號": code, "名稱": stock_name, "收盤": f"{d_close:.2f}", "產業": sector_name,
                    "漲幅": f"{(pct_chg_val*100):.2f}%", "成交量": int(d_volume/1000),
                    "前波高點": f"{prev_period_high:.2f}", "距離高點": f"{dist_to_high:+.2f}%", "狀態": "⚡ 蓄勢待發"
                }

        return {'sniper': result_sniper, 'day': result_day}

    except Exception as e:
        logger.error(f"analyze_combined_strategy 錯誤 [{code}]: {e}", exc_info=True)
        return f"程式執行錯誤: {str(e)}"

# ==========================================
# 🔥 全展開表格顯示函式
# ==========================================
def display_full_table(df):
    if df is not None and not df.empty:
        display_cols = [c for c in df.columns if not c.startswith('_')]
        display_df = df[display_cols]
        height = (len(display_df) * 35) + 38
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=height)
    else:
        st.info("無")

# ==========================================
# 📊 個股診斷強化版
# ==========================================
def run_diagnosis(stock_input, analysis_date_str, params):
    symbol = f"{stock_input}.TW"
    df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
    if df is None:
        symbol = f"{stock_input}.TWO"
        df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
    return df, symbol

def plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info=None):
    s_ma_trend = params['s_ma_trend']
    df = df.copy()
    df['MA_Trend'] = df['Close'].rolling(window=s_ma_trend).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA240'] = df['Close'].rolling(window=240).mean()

    plot_df = df.tail(250)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("K線圖", "成交量")
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name='K線', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['MA_Trend'],
        line=dict(color='blue', width=1.5), name=f'{s_ma_trend}MA'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['MA20'],
        line=dict(color='orange', width=1.2), name='20MA'
    ), row=1, col=1)
    if plot_df['MA240'].notna().any():
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['MA240'],
            line=dict(color='purple', width=1.2, dash='dash'), name='240MA'
        ), row=1, col=1)

    if sniper_info:
        setup_date = sniper_info.get('_setup_date')
        defense_price = sniper_info.get('_defense')
        signal_high = sniper_info.get('_signal_high')
        status = sniper_info.get('狀態', '')

        if setup_date:
            try:
                setup_ts = pd.Timestamp(setup_date)
                if setup_ts in plot_df.index:
                    fig.add_vline(
                        x=setup_ts, line_width=2,
                        line_dash="dash", line_color="gold",
                        annotation_text=f"📍訊號日 {setup_date}",
                        annotation_position="top right"
                    )
            except Exception as e:
                logger.warning(f"標記訊號日錯誤: {e}")

        if defense_price and defense_price > 0:
            fig.add_hline(
                y=defense_price,
                line_width=2, line_dash="dot", line_color="red",
                annotation_text=f"🛡️ 防守 {defense_price:.2f}",
                annotation_position="bottom right"
            )

        if signal_high and signal_high > 0:
            fig.add_hline(
                y=signal_high,
                line_width=1.5, line_dash="dot", line_color="orange",
                annotation_text=f"⚡ 長紅高 {signal_high:.2f}",
                annotation_position="top right"
            )

        fig.add_annotation(
            text=f"策略狀態：{status}",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(50,50,50,0.7)",
            bordercolor="gray",
            borderwidth=1
        )

    colors = ['red' if c >= o else 'green'
              for c, o in zip(plot_df['Close'], plot_df['Open'])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df['Volume'],
        name='成交量', marker_color=colors
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=650,
        title_text=f"{stock_input} 個股診斷圖",
        template="plotly_dark"
    )
    return fig

def run_backtest_ui(df, stock_input, params):
    try:
        bt_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
        bt_df.index = pd.to_datetime(bt_df.index)

        bt = Backtest(
            bt_df,
            SniperStrategy,
            cash=100000,
            commission=0.001425 * 2,
            trade_on_close=False
        )

        stats = bt.run(
            ma_trend_period=params['s_ma_trend'],
            big_candle_pct=params['s_big_candle'],
            min_volume_shares=params['s_min_vol'],
            use_year_line=params['s_use_year']
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("總報酬率", f"{stats['Return [%]']:.1f}%")
        col2.metric("最大回撤", f"{stats['Max. Drawdown [%]']:.1f}%")
        col3.metric("勝率", f"{stats['Win Rate [%]']:.1f}%")
        col4.metric("交易次數", f"{stats['# Trades']}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("夏普比率", f"{stats['Sharpe Ratio']:.2f}")
        col6.metric("平均獲利", f"{stats['Avg. Trade [%]']:.2f}%")
        col7.metric("買入持有報酬", f"{stats['Buy & Hold Return [%]']:.1f}%")
        col8.metric("Calmar 比率", f"{stats['Calmar Ratio']:.2f}" if 'Calmar Ratio' in stats else "N/A")

        trades = stats['_trades']
        if not trades.empty:
            st.markdown("#### 📋 交易明細")
            trades_display = trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']].copy()
            trades_display.columns = ['進場時間', '出場時間', '進場價', '出場價', '損益', '報酬%']
            trades_display['報酬%'] = (trades_display['報酬%'] * 100).round(2)
            trades_display['損益'] = trades_display['損益'].round(2)
            st.dataframe(trades_display, hide_index=True, use_container_width=True)

        equity = stats['_equity_curve']['Equity']
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            fill='tozeroy', line=dict(color='cyan'), name='權益曲線'
        ))
        fig_eq.update_layout(
            title="📈 策略權益曲線",
            height=300,
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    except Exception as e:
        logger.error(f"回測執行錯誤 [{stock_input}]: {e}", exc_info=True)
        st.error(f"回測錯誤：{e}")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室 V12")
st.sidebar.caption("波段與短線的極致整合")

cache_mode = "🟢 盤中模式 (5分鐘快取)" if is_trading_hours() else "🔵 盤後模式 (1小時快取)"
st.sidebar.caption(cache_mode)

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

# ================= V12 統一按鈕 =================
start_scan = st.sidebar.button("🚀 開始全域掃描 (極速版)", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.empty()

st.sidebar.divider()

with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=True):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA) 濾網", value=True)
    s_big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 5.0, 0.5) / 100
    s_min_vol = st.number_input("波段最小量 (張)", value=1000) * 1000
    s_setup_lookback = st.slider("長紅K回溯天數", 5, 30, 25, 5)
    s_vol_ma_days = st.slider("長紅K量能基準 (MA天數)", 5, 20, 20, 5)
    # V12修改：預設值改為 0.7
    s_vol_ratio = st.slider("長紅K量能門檻 (倍)", 0.5, 1.5, 0.7, 0.1)
    s_pullback_max = st.slider("整理回測深度上限 (%)", 20, 70, 50, 5)

with st.sidebar.expander("🎯 處置股策略參數", expanded=True):
    d_show_all = st.checkbox("顯示全部（含無訊號）", value=False)
    d_vol_filter = st.checkbox("啟用底量濾網", value=True)
    d_min_vol_disp = st.number_input("底量門檻（張）", min_value=0, max_value=10000, value=1000, step=100, disabled=not d_vol_filter)

# BULL GSheet 狀態（expander 外，永遠可見）
if bull_use_gsheet():
    st.sidebar.success("🟢 BULL 持倉已連線 Google Sheets")
elif GSPREAD_AVAILABLE:
    st.sidebar.warning("⚠️ BULL 持倉：未設定 Sheets Secrets")
else:
    st.sidebar.info("ℹ️ BULL 持倉：暫存模式（未安裝 gspread）")

with st.sidebar.expander("🐂 BULL_v7 布林滾雪球參數", expanded=False):
    b_boll_period  = st.number_input("布林週期", value=15, min_value=5, max_value=60, key="b_bp")
    b_boll_std     = st.number_input("布林標準差", value=2.1, min_value=1.0, max_value=3.0, step=0.1, key="b_bs")
    b_squeeze_n    = st.number_input("壓縮天數", value=15, min_value=5, max_value=60, key="b_sn")
    b_squeeze_lb   = st.number_input("壓縮回溯天數", value=5, min_value=1, max_value=15, key="b_sl")
    b_vol_ratio    = st.number_input("爆量倍數", value=1.5, min_value=1.0, max_value=5.0, step=0.1, key="b_vr")
    b_sma_days     = st.number_input("中軌趨勢天數", value=3, min_value=1, max_value=10, key="b_sd")
    b_min_vol      = st.number_input("20日均量門檻(張)", value=1000, min_value=100, step=100, key="b_mv") * 1000
    b_addon_b_pct  = st.number_input("加碼B門檻(%)", value=10, min_value=1, max_value=50, key="b_ab") / 100
    st.markdown("---")
    if bull_use_gsheet():
        st.success("🟢 持倉已連線 Google Sheets")
    elif GSPREAD_AVAILABLE:
        st.warning("⚠️ 未設定 Sheets Secrets，使用暫存模式")
    else:
        st.info("ℹ️ 未安裝 gspread，使用暫存模式")

with st.sidebar.expander("⚡ 隔日沖策略參數 (短線)", expanded=True):
    d_period = st.slider("追蹤波段天數 (N)", 10, 120, 60, 5)
    d_threshold = st.slider("高點容許誤差 (%)", 0.0, 5.0, 1.0, 0.1)
    d_min_pct = st.slider("當日最低漲幅 (%)", 3.0, 9.0, 5.0, 0.1)
    d_min_vol = st.number_input("隔日沖最小量 (張)", value=1000, step=500)

st.sidebar.divider()
max_workers_input = st.sidebar.slider("策略運算效能 (執行緒數)", 1, 32, 16)

params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year,
    's_big_candle': s_big_candle, 's_min_vol': s_min_vol,
    's_setup_lookback': s_setup_lookback, 's_vol_ma_days': s_vol_ma_days, 's_vol_ratio': s_vol_ratio, 's_pullback_max': s_pullback_max,
    'd_period': d_period, 'd_threshold': d_threshold,
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol,
    'disp_show_all': d_show_all, 'disp_vol_filter': d_vol_filter, 'disp_min_vol': d_min_vol_disp
}

bull_params = {
    **BULL_PARAMS,
    "boll_period"    : int(b_boll_period),
    "boll_std"       : float(b_boll_std),
    "squeeze_n"      : int(b_squeeze_n),
    "squeeze_lookback": int(b_squeeze_lb),
    "vol_ratio"      : float(b_vol_ratio),
    "sma_trend_days" : int(b_sma_days),
    "min_vol_shares" : int(b_min_vol),
    "addon_b_profit" : float(b_addon_b_pct),
}

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🟢 狙擊手波段", "🎯 處置股策略", "🐂 BULL滾雪球", "⚡ 隔日沖雷達", "📊 個股診斷"])

# ==========================================
# Session State 管理
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None
if 'bull_results' not in st.session_state:
    st.session_state['bull_results'] = None
if 'bull_positions' not in st.session_state:
    st.session_state['bull_positions'] = pd.DataFrame(columns=BULL_SHEET_COLS)
if 'bull_gs_loaded' not in st.session_state:
    st.session_state['bull_gs_loaded'] = False
# 首次啟動從 Google Sheets 載入 BULL 持倉
if not st.session_state['bull_gs_loaded'] and bull_use_gsheet():
    df_gs = bull_gs_load()
    if len(df_gs) > 0:
        st.session_state['bull_positions'] = df_gs
    st.session_state['bull_gs_loaded'] = True
if 'disposal_results' not in st.session_state:
    st.session_state['disposal_results'] = None
if 'disposal_active' not in st.session_state:
    st.session_state['disposal_active'] = None
if 'scan_params' not in st.session_state:
    st.session_state['scan_params'] = None
if 'scan_date' not in st.session_state:
    st.session_state['scan_date'] = None
if 'market_status' not in st.session_state:
    st.session_state['market_status'] = None

def params_changed():
    if st.session_state['scan_params'] is None:
        return False
    return (st.session_state['scan_params'] != params or
            st.session_state['scan_date'] != analysis_date_str)

# ==========================================
# 執行整合掃描
# ==========================================
if start_scan:
    # ---------------- 1. 狙擊手與隔日沖 ----------------
    stock_map = get_stock_info_map()

    status_text.text("📈 正在判斷大盤狀態...")
    market_status = fetch_market_status(analysis_date_str)
    st.session_state['market_status'] = market_status

    sniper_triggered = []
    sniper_setup = []
    sniper_watching = []
    day_candidates = []
    failed_list = []

    status_text.text("🔄 正在批量下載歷史資料 (yfinance)...")
    history_data_store = fetch_data_batch(stock_map)
    # 存入 session_state 供 BULL_v7 掃描共用（不重複下載）
    st.session_state['all_data_cache']   = history_data_store
    st.session_state['stock_map_cache']  = stock_map

    # 即時盤更新已移除：twstock/yfinance 免費版同樣有 15 分鐘延遲
    # 省掉這段等待，直接用 yfinance 下載的資料（結果相同）
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    status_text.text("🧠 正在進行大盤與波段策略運算...")
    progress_bar.progress(0)

    tasks_data = {code: df for code, df in history_data_store.items()}

    total = len(tasks_data)
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {
            executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df, market_status.get('score', 5)): code
            for code, df in tasks_data.items()
        }
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total:
                progress_bar.progress(done / max(total, 1))
                status_text.text(f"波段策略運算中: {done}/{total}")
            res = future.result()
            if isinstance(res, dict):
                if res['sniper']:
                    typ, data = res['sniper']
                    if typ == "triggered": sniper_triggered.append(data)
                    elif typ == "new_setup": sniper_setup.append(data)
                    elif typ == "watching": sniper_watching.append(data)
                if res['day']:
                    day_candidates.append(res['day'])
            else:
                current_code = futures[future]
                stock_name = stock_map[current_code]['short_name']
                failed_list.append(f"{current_code} {stock_name} : {res}")

    st.session_state['scan_results'] = {
        'sniper_triggered': sniper_triggered,
        'sniper_setup': sniper_setup,
        'sniper_watching': sniper_watching,
        'day_candidates': day_candidates,
        'failed_list': failed_list
    }

    # ---------------- 2. 處置股策略 ----------------
    status_text.text("📋 正在取得並分析處置股清單...")
    progress_bar.progress(0)
    
    disposal_map = fetch_disposal_stocks()
    if not disposal_map:
        st.warning("⚠️ 無法自動取得處置股清單")
        disposal_map = {}
        
    lookahead = analysis_date_input + datetime.timedelta(days=3)
    d_active        = {k: v for k, v in disposal_map.items() if v['start'] <= lookahead and v['end'] >= analysis_date_input}
    d_active_now    = {k: v for k, v in d_active.items() if v['start'] <= analysis_date_input}
    d_active_coming = {k: v for k, v in d_active.items() if v['start'] > analysis_date_input}

    d_results = []
    total_d = len(d_active_now)
    for i, (code, info) in enumerate(d_active_now.items()):
        status_text.text(f"🔍 處置股分析中... {code} {info['name']} ({i+1}/{total_d})")
        progress_bar.progress((i+1) / max(total_d, 1))
        
        result = analyze_disposal_stock(code, info, analysis_date_input, min_vol=d_min_vol_disp if d_vol_filter else 0)
        
        if result:
            d_results.append(result)
        elif d_show_all:
            d_results.append({
                'code': code, 'name': info['name'],
                'market': '上市' if info['market'] == 'twse' else '上櫃',
                'disposal_start': info['start'].strftime('%Y-%m-%d'),
                'disposal_end':   info['end'].strftime('%Y-%m-%d'),
                'days_left': max((info['end'] - analysis_date_input).days, 0),
                'status': '⚪ 無訊號', 'is_new_today': False,
                'entry_date': '-', 'entry_price': '-',
                'exit_date': '-', 'exit_price': '-',
                'profit_pct': '-', 'exit_reason': '-',
                'signal_date': '-', 'signal_close': '-',
                'signal_prev_high': '-', 'signal_pct': '-',
                'total_trades': 0,
            })
        time.sleep(0.1)

    st.session_state['disposal_results'] = d_results
    st.session_state['disposal_active'] = {
        'now': d_active_now,
        'coming': d_active_coming
    }

    progress_bar.progress(1.0)
    status_text.success("✅ 全域掃描（波段、隔日沖、處置股）已完成！")

    st.session_state['scan_params'] = params.copy()
    st.session_state['scan_date'] = analysis_date_str


results = st.session_state['scan_results']

if results is not None and params_changed():
    st.warning("⚠️ 您已修改策略參數或分析日期，目前顯示的結果為**上次掃描**的資料，請重新執行掃描以取得最新結果。")


# ==========================================
# 🐂 BULL_v7 掃描（共用 all_data，不重複下載）
# ==========================================
if start_scan and results and 'all_data_cache' in st.session_state:
    status_text.text("🐂 BULL_v7 布林滾雪球掃描中...")
    try:
        bull_res = bull_run_scan(
            stock_map         = st.session_state['stock_map_cache'],
            all_data          = st.session_state['all_data_cache'],
            scan_date_str     = analysis_date_str,
            bull_positions    = st.session_state['bull_positions'],
            bull_params       = bull_params,
            max_workers       = max_workers_input,
        )
        st.session_state['bull_results'] = bull_res
    except Exception as e:
        logger.error(f"BULL 掃描錯誤: {e}")

# ==========================================
# 介面顯示：Tab 1 (狙擊手波段)
# ==========================================
with tab1:
    st.header("🟢 狙擊手波段策略")
    st.caption(f"基準日: {analysis_date_str} | 策略：趨勢 + 實體長紅 + 型態確認 (防守點含 1% 誤差)")

    mkt = st.session_state.get('market_status')
    if mkt:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("大盤狀態", mkt['label'])
        mc2.metric("加權指數", f"{mkt.get('close', '-'):,.0f}")
        mc3.metric("20MA", f"{mkt.get('ma20', '-'):,.0f}")
        mc4.metric("60MA", f"{mkt.get('ma60', '-'):,.0f}")
        if not mkt['strong']:
            st.warning("⚠️ 大盤目前偏弱，建議降低持倉比例，訊號可信度下降。")
        st.divider()

    if results:
        if 'failed_list' in results and results['failed_list']:
            with st.expander(f"⚠️ 掃描失敗/無資料清單 ({len(results['failed_list'])})"):
                st.write(", ".join(results['failed_list']))

        s_trig = results['sniper_triggered']
        trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]
        trig_n      = [x for x in s_trig if "N字" in x['狀態']]

        s_watch       = results['sniper_watching']
        watch_strong   = [x for x in s_watch if "強勢整理" in x['狀態']]
        watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]

        if trig_strong or trig_n:
            all_triggered = trig_strong + trig_n
            st.markdown("### 🎯 買點觸發訊號 (Actionable)")
            st.caption("依漲幅排序，⚠️風險距離 > 5% 建議跳過")

            if trig_strong:
                st.markdown(f"#### 🚀 強勢突破 ({len(trig_strong)})")
                df_ts = pd.DataFrame(trig_strong)
                df_ts = df_ts.sort_values(by='sort_pct', ascending=False)
                drop_cols = [c for c in ['sort_pct', '_score', '_risk_pct'] if c in df_ts.columns]
                display_full_table(df_ts.drop(columns=drop_cols))

            if trig_n:
                st.markdown(f"#### 🎯 N字突破 ({len(trig_n)})")
                df_tn = pd.DataFrame(trig_n)
                df_tn = df_tn.sort_values(by='sort_pct', ascending=False)
                drop_cols = [c for c in ['sort_pct', '_score', '_risk_pct'] if c in df_tn.columns]
                display_full_table(df_tn.drop(columns=drop_cols))

            st.divider()
            st.markdown("### 🫧 產業強度分析")
            st.caption("右上角 = 廣度與強度兼具的強勢產業，泡泡大小代表訊號數")

            all_sig = trig_strong + trig_n + results.get('sniper_setup', []) + watch_strong
            if all_sig:
                df_all = pd.DataFrame(all_sig)
                df_all['漲幅_num'] = df_all['漲幅'].str.replace('%','').str.replace('+','').astype(float)
                sector_grp = df_all.groupby('產業').agg(
                    訊號數=('代號', 'count'),
                    平均漲幅=('漲幅_num', 'mean')
                ).reset_index()

                fig_bubble = go.Figure(go.Scatter(
                    x=sector_grp['訊號數'],
                    y=sector_grp['平均漲幅'],
                    mode='markers+text',
                    text=sector_grp['產業'],
                    textposition='top center',
                    marker=dict(
                        size=sector_grp['訊號數'] * 8 + 10,
                        color=sector_grp['平均漲幅'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='平均漲幅%'),
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "訊號數: %{x}<br>"
                        "平均漲幅: %{y:.2f}%<extra></extra>"
                    )
                ))
                fig_bubble.update_layout(
                    height=420, template="plotly_dark",
                    xaxis_title="訊號數（廣度）",
                    yaxis_title="平均漲幅（強度）",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

# ==========================================
# 介面顯示：Tab 2 (處置股策略)
# ==========================================
with tab2:
    st.markdown("### 🎯 處置股策略")
    st.caption("僅限每5分鐘撮合處置股｜進場：收盤站上前日最高點｜出場：獲利<15%跌破10MA，獲利≥15%跌破5MA")
    
    if st.session_state['disposal_results'] is not None:
        d_results = st.session_state['disposal_results']
        d_active_now = st.session_state['disposal_active']['now']
        d_active_coming = st.session_state['disposal_active']['coming']

        st.info(f"📋 處置中 **{len(d_active_now)}** 支｜即將開始（3天內）**{len(d_active_coming)}** 支")

        d_new_today = [r for r in d_results if r.get('is_new_today')]
        d_holding   = [r for r in d_results if r.get('status') == '🟢 持有中' and not r.get('is_new_today')]
        d_exited    = [r for r in d_results if r.get('status') == '⚫ 已出場']
        d_no_signal = [r for r in d_results if r.get('status') == '⚪ 無訊號']

        st.markdown(f"#### 🔴 今日新訊號 ({len(d_new_today)} 支)")
        if d_new_today:
            show_disposal_table(d_new_today)
        else:
            if d_active_coming:
                st.info(f"今日無訊號，以下 {len(d_active_coming)} 支即將進入處置期間：")
                st.dataframe(pd.DataFrame([{
                    '代號': k, '名稱': v['name'],
                    '市場': '上市' if v['market'] == 'twse' else '上櫃',
                    '處置開始': v['start'].strftime('%Y-%m-%d'),
                    '處置結束': v['end'].strftime('%Y-%m-%d')
                } for k, v in d_active_coming.items()]), use_container_width=True, hide_index=True)
            else:
                st.info(f"基準日 {analysis_date_str} 無今日訊號")

        if d_holding:
            st.divider()
            st.markdown(f"#### 🟢 持有中 ({len(d_holding)} 支)")
            show_disposal_table(d_holding)

        if d_exited:
            st.divider()
            st.markdown(f"#### ⚫ 已出場 ({len(d_exited)} 支)")
            show_disposal_table(d_exited)

        if d_show_all and d_no_signal:
            st.divider()
            st.markdown(f"#### ⚪ 無訊號 ({len(d_no_signal)} 支)")
            show_disposal_table(d_no_signal)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("處置中", f"{len(d_active_now)} 支")
        c2.metric("今日新訊號", f"{len(d_new_today)} 支")
        c3.metric("持有中", f"{len(d_holding) + len(d_new_today)} 支")
        c4.metric("已出場", f"{len(d_exited)} 支")
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

# ==========================================
# 介面顯示：Tab 3 (BULL_v7 布林滾雪球)
# ==========================================
with tab3:
    st.header("🐂 BULL_v7 布林滾雪球掃描器")
    bull_res = st.session_state.get('bull_results')
    bull_pos = st.session_state['bull_positions']

    # 持倉操作函數
    def bull_add_pos(sym, name, price, date_str):
        pos = st.session_state['bull_positions']
        if sym in pos['symbol'].values:
            return
        new_row = pd.DataFrame([{
            "symbol": sym, "name": name, "進場日期": date_str,
            "進場價": round(price, 2), "上次加碼價": round(price, 2),
            "持倉最高價": round(price, 2), "加碼次數": 0, "加碼紀錄": ""
        }])
        updated = pd.concat([pos, new_row], ignore_index=True)
        st.session_state['bull_positions'] = updated
        if bull_use_gsheet(): bull_gs_save(updated)

    def bull_do_addon(sym, addon_type, price, date_str):
        pos  = st.session_state['bull_positions']
        mask = pos['symbol'] == sym
        if not mask.any(): return
        idx  = pos[mask].index[-1]
        pos.loc[idx, '上次加碼價'] = round(price, 2)
        pos.loc[idx, '加碼次數']   = int(pos.loc[idx, '加碼次數']) + 1
        tag  = f"{date_str}({addon_type})"
        prev = str(pos.loc[idx, '加碼紀錄'])
        pos.loc[idx, '加碼紀錄']   = (prev + " → " + tag).strip(" → ") if prev else tag
        st.session_state['bull_positions'] = pos
        if bull_use_gsheet(): bull_gs_save(pos)

    def bull_remove_pos(sym):
        pos     = st.session_state['bull_positions']
        updated = pos[pos['symbol'] != sym].reset_index(drop=True)
        st.session_state['bull_positions'] = updated
        if bull_use_gsheet(): bull_gs_save(updated)

    # 更新持倉最高價
    if bull_res:
        pos = st.session_state['bull_positions']
        for sym_, price_ in bull_res.get('high_updates', []):
            mask = pos['symbol'] == sym_
            if mask.any():
                idx_ = pos[mask].index[-1]
                if price_ > float(pos.loc[idx_, '持倉最高價']):
                    pos.loc[idx_, '持倉最高價'] = price_
        st.session_state['bull_positions'] = pos
        bull_pos = pos

    entry_n  = len(bull_res['entry'])   if bull_res else 0
    addon_a_n= len(bull_res['addon_a']) if bull_res else 0
    addon_b_n= len(bull_res['addon_b']) if bull_res else 0
    exit_n   = len(bull_res['exit'])    if bull_res else 0
    pos_n    = len(bull_pos)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 持倉", pos_n)
    c2.metric("🟢 進場", entry_n)
    c3.metric("🔵 加碼B", addon_b_n)
    c4.metric("🟡 加碼A", addon_a_n)
    c5.metric("🔴 出場", exit_n)

    if not bull_res:
        st.info("👈 點擊左側「開始全域掃描」後即可看到 BULL_v7 訊號。")
    else:
        bt1, bt2, bt3, bt4, bt5 = st.tabs([
            f"🟢 進場({entry_n})", f"🔵 加碼B({addon_b_n})",
            f"🟡 加碼A({addon_a_n})", f"🔴 出場({exit_n})", f"📋 持倉({pos_n})"
        ])

        with bt1:
            if not bull_res['entry']:
                st.info("今日無進場訊號")
            else:
                df_e = pd.DataFrame(bull_res['entry'])
                st.dataframe(df_e, use_container_width=True, hide_index=True)
                cols = st.columns(min(len(bull_res['entry']), 6))
                for i, r in enumerate(bull_res['entry']):
                    with cols[i % 6]:
                        if st.button(f"✅ 加入 {r['代號']}", key=f"bull_add_{r['symbol']}"):
                            bull_add_pos(r['symbol'], r['名稱'], r['收盤價'], analysis_date_str)
                            st.success(f"{r['代號']} 已加入持倉")
                            st.rerun()

        with bt2:
            if not bull_res['addon_b']:
                st.info("今日無加碼B訊號")
            else:
                df_b = pd.DataFrame(bull_res['addon_b'])
                st.dataframe(df_b, use_container_width=True, hide_index=True)
                cols = st.columns(min(len(bull_res['addon_b']), 6))
                for i, r in enumerate(bull_res['addon_b']):
                    with cols[i % 6]:
                        if st.button(f"🔵 加碼B {r['代號']}", key=f"bull_addb_{r['symbol']}"):
                            bull_do_addon(r['symbol'], "突破新高", r['收盤價'], analysis_date_str)
                            st.success(f"{r['代號']} 加碼B完成")
                            st.rerun()

        with bt3:
            if not bull_res['addon_a']:
                st.info("今日無加碼A訊號")
            else:
                df_a = pd.DataFrame(bull_res['addon_a'])
                st.dataframe(df_a, use_container_width=True, hide_index=True)
                cols = st.columns(min(len(bull_res['addon_a']), 6))
                for i, r in enumerate(bull_res['addon_a']):
                    with cols[i % 6]:
                        if st.button(f"🟡 加碼A {r['代號']}", key=f"bull_adda_{r['symbol']}"):
                            bull_do_addon(r['symbol'], "回測站回", r['收盤價'], analysis_date_str)
                            st.success(f"{r['代號']} 加碼A完成")
                            st.rerun()

        with bt4:
            if not bull_res['exit']:
                st.info("今日無出場訊號")
            else:
                df_x = pd.DataFrame(bull_res['exit'])
                st.dataframe(df_x, use_container_width=True, hide_index=True)
                cols = st.columns(min(len(bull_res['exit']), 6))
                for i, r in enumerate(bull_res['exit']):
                    with cols[i % 6]:
                        if st.button(f"🚪 出場 {r['代號']}", key=f"bull_exit_{r['symbol']}"):
                            bull_remove_pos(r['symbol'])
                            st.success(f"{r['代號']} 已出場")
                            st.rerun()

        with bt5:
            if len(bull_pos) == 0:
                st.info("目前無持倉")
            else:
                st.dataframe(bull_pos, use_container_width=True, hide_index=True)
                st.markdown("**手動出場：**")
                bcols = st.columns(min(len(bull_pos), 8))
                for i, (_, row) in enumerate(bull_pos.iterrows()):
                    label = str(row['symbol']).replace('.TW','').replace('.TWO','')
                    with bcols[i % 8]:
                        if st.button(f"🚪 {label}", key=f"bull_m_exit_{row['symbol']}"):
                            bull_remove_pos(row['symbol'])
                            st.rerun()
                st.markdown("---")
                csv = bull_pos.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("⬇️ 匯出持倉 CSV", data=csv,
                                   file_name=f"BULL_positions_{analysis_date_str}.csv",
                                   mime="text/csv")
                uploaded = st.file_uploader("📂 匯入持倉 CSV", type="csv", key="bull_upload")
                if uploaded:
                    try:
                        df_imp = pd.read_csv(uploaded, encoding='utf-8-sig')
                        st.session_state['bull_positions'] = df_imp
                        st.success(f"✅ 已匯入 {len(df_imp)} 筆持倉")
                        st.rerun()
                    except Exception as e:
                        st.error(f"匯入失敗：{e}")

# ==========================================
# 介面顯示：Tab 4 (隔日沖雷達)
# ==========================================
with tab4:
    st.header("⚡ 隔日沖雷達")
    if results:
        day_list = results['day_candidates']
        if day_list:
            df_day = pd.DataFrame(day_list)
            df_day['sort_val'] = df_day['距離高點'].str.rstrip('%').astype(float)
            df_day = df_day.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
            display_full_table(df_day)
        else:
            st.info("今日無符合隔日沖策略之標的。")
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

# ==========================================
# 介面顯示：Tab 5 (個股 K 線診斷)
# ==========================================
with tab5:
    st.header("📊 個股 K 線診斷")

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        stock_input = st.text_input("輸入代號", value="2330")
    with col_btn:
        diag_btn = st.button("診斷")

    run_bt = st.checkbox("同時執行回測", value=False, help="勾選後診斷時會一併執行 SniperStrategy 回測，需要較長時間")

    if diag_btn:
        with st.spinner("載入資料中..."):
            df, symbol = run_diagnosis(stock_input, analysis_date_str, params)

        if df is not None:
            sniper_info = None
            if results:
                all_sniper = (
                    results.get('sniper_triggered', []) +
                    results.get('sniper_watching', []) +
                    results.get('sniper_setup', [])
                )
                for item in all_sniper:
                    if item.get('代號') == stock_input:
                        sniper_info = item
                        break

            if sniper_info:
                st.success(f"✅ 此股命中策略：{sniper_info.get('狀態', '')}　|　訊號日：{sniper_info.get('訊號日', sniper_info.get('_setup_date', 'N/A'))}")
            else:
                st.info("ℹ️ 此股在最近一次掃描中未命中任何策略訊號（或尚未執行掃描）")

            fig = plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info)
            st.plotly_chart(fig, use_container_width=True)

            if run_bt:
                st.markdown("---")
                st.markdown("### 📈 SniperStrategy 回測結果")
                st.caption(f"使用當前側邊欄參數 | 初始資金 NT$100,000 | 手續費 0.1425% x2")
                with st.spinner("回測運算中..."):
                    run_backtest_ui(df, stock_input, params)
        else:
            st.error("查無資料，請確認代號是否正確")