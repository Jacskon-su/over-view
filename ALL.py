# ==========================================
# 強勢股戰情室 V18 (純永豐強護版)
# V2~V16: 歷史更新與 Bug 修正
# V17_beta: 新增「🔧 系統診斷」分頁
# V18: 🚀 終極進化版
#      - 完全移除 twstock 與 證交所盤後 API 依賴。
#      - 全面導入 永豐金 Shioaji API 作為 24H 唯一報價引擎。
#      - (修復) 修正大盤無資料時的字串格式化 ValueError。
#      - (修復) 解決 Parquet 型態與 API 型態不一致導致股票被清空的 Bug。
#      - (修復) 加入 Pandas Index 去重防呆，完美解決盤中快照寫入失敗/報錯的問題。
# ==========================================
import streamlit as st
import os
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import requests
from bs4 import BeautifulSoup
import time
import importlib
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# 加入這兩行來關閉 SSL 憑證警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 📋 日誌與外部套件設定
# ==========================================
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 載入永豐金 Shioaji API
try:
    import shioaji as sj
except ImportError:
    st.error("❌ 缺少 `shioaji` 套件，請輸入 `pip install shioaji` 安裝")
    st.stop()

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
    logger.error(f"sector_data.py 載入錯誤: {e}")

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(
    page_title="強勢股戰情室 V18",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 強制台灣時區 ---
import pytz as _pytz_main
_TW_TZ = _pytz_main.timezone("Asia/Taipei")
def _now_tw():
    """取得台灣當前時間，解決 Streamlit Cloud UTC 時差問題"""
    import datetime as _dtt, pytz as _ptz
    return _dtt.datetime.now(_ptz.timezone("Asia/Taipei"))

# Google Sheets
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
# 🚀 永豐金 Shioaji API 初始化與核心服務
# ==========================================
@st.cache_resource
def get_shioaji_api():
    """初始化並登入永豐金 Shioaji API"""
    if "shioaji" not in st.secrets:
        st.error("❌ 找不到永豐金 API 金鑰！請在 `.streamlit/secrets.toml` 中設定 `[shioaji]` 的 `api_key` 與 `secret_key`。")
        st.stop()
        
    api = sj.Shioaji(simulation=False)
    try:
        api.login(
            api_key=st.secrets["shioaji"]["api_key"],
            secret_key=st.secrets["shioaji"]["secret_key"],
            fetch_contract=True
        )
        return api
    except Exception as e:
        st.error(f"🔴 永豐金 API 登入失敗: {e}")
        st.stop()

api = get_shioaji_api()

def is_trading_hours():
    now = _now_tw()
    if now.weekday() >= 5: return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

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
        close = pd.Series(self.data.Close); volume = pd.Series(self.data.Volume)
        self.ma_trend = self.I(SMA, close, self.ma_trend_period)
        self.ma_base = self.I(SMA, close, self.ma_base_exit)
        self.ma_fast = self.I(SMA, close, self.ma_fast_exit)
        self.vol_ma = self.I(SMA, volume, self.vol_ma_period)
        if self.use_year_line: self.ma_long = self.I(SMA, close, self.ma_long_period)
        self.setup_active = False; self.setup_bar_index = 0; self.setup_low_price = 0; self.defense_price = 0

    def next(self):
        price = self.data.Close[-1]; prev_high = self.data.High[-2]
        if self.position:
            if price < self.defense_price:
                self.position.close(); return
            exit_line = self.ma_fast[-1] if self.position.pl_pct > 0.15 else self.ma_base[-1]
            if price < exit_line: self.position.close()
            return

        triggered_buy = False
        if self.setup_active:
            if len(self.data) - self.setup_bar_index > self.lookback_window or price < self.defense_price: self.setup_active = False
            elif price > prev_high: self.buy(); self.setup_active = False; triggered_buy = True; return

        if not triggered_buy:
            if self.data.Volume[-1] < self.min_volume_shares: return
            if not ((price > self.ma_trend[-1]) and (self.ma_trend[-1] > self.ma_trend[-2])): return
            if self.use_year_line and (pd.isna(self.ma_long[-1]) or price < self.ma_long[-1]): return

            is_big = (price - self.data.Close[-2]) / self.data.Close[-2] > self.big_candle_pct
            if is_big and (self.data.Volume[-1] > self.vol_ma[-1]) and (price > self.data.Open[-1]):
                self.setup_active = True; self.setup_bar_index = len(self.data); self.setup_low_price = self.data.Low[-1]
                self.defense_price = (self.data.Close[-2] if self.data.Low[-1] > self.data.High[-2] else self.data.Low[-1]) * (1 - self.defense_buffer)

# ==========================================
# 🛠️ 輔助函式與資料庫 (Shioaji 版)
# ==========================================
def get_detailed_sector(code, standard_group=None, custom_db=None):
    code_str = str(code).strip()
    if custom_db and code_str in custom_db: return str(custom_db[code_str])
    if standard_group and str(standard_group) not in ['nan', 'None', '', 'NaN']: return str(standard_group)
    return "其他"

@st.cache_data(ttl=3600*24)
def get_stock_info_map(_api_instance):
    stock_map = {}
    try:
        for exchange in [_api_instance.Contracts.Stocks.TSE, _api_instance.Contracts.Stocks.OTC]:
            for contract in exchange:
                code = contract.code
                if len(code) == 4 and code.isdigit():
                    stock_map[str(code)] = {
                        'name': f"{code} {contract.name}", 
                        'symbol': f"{code}{'.TW' if contract.exchange == 'TSE' else '.TWO'}", 
                        'short_name': contract.name, 
                        'group': getattr(contract, 'category', '其他')
                    }
    except Exception as e: logger.error(f"Shioaji 取得股票清單失敗: {e}")
    return stock_map

def get_stock_data_with_realtime(code, symbol, analysis_date_str, _api_instance):
    try:
        df = yf.Ticker(symbol).history(period="2y")
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
    except: return None

    today_str = _now_tw().strftime('%Y-%m-%d')
    if analysis_date_str == today_str and df.index[-1].strftime('%Y-%m-%d') != today_str:
        try:
            contract = _api_instance.Contracts.Stocks.get(str(code))
            if contract:
                snapshots = _api_instance.snapshots([contract])
                if snapshots:
                    snap = snapshots[0]
                    c_p = snap.close if snap.close > 0 else snap.previous_close
                    if c_p > 0:
                        new_row = pd.Series({'Open': snap.open if snap.open > 0 else c_p, 'High': snap.high if snap.high > 0 else c_p, 'Low': snap.low if snap.low > 0 else c_p, 'Close': c_p, 'Volume': snap.total_volume * 1000}, name=pd.Timestamp(today_str))
                        df = pd.concat([df, new_row.to_frame().T])
        except Exception: pass
    return df

# ==========================================
# 🐂 BULL_v7 Google Sheets 持倉同步
# ==========================================
BULL_SHEET_COLS = ["symbol","name","進場日期","進場價","上次加碼價","持倉最高價","加碼次數","加碼紀錄"]
BULL_SCOPES     = ["https://www.googleapis.com/auth/spreadsheets"]

def bull_use_gsheet() -> bool:
    if not GSPREAD_AVAILABLE: return False
    try: return ("gcp_service_account" in st.secrets and "sheets" in st.secrets)
    except Exception: return False

def bull_get_ws():
    if not GSPREAD_AVAILABLE: return None
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=BULL_SCOPES)
        sh = gspread.authorize(creds).open_by_key(st.secrets["sheets"]["sheet_id"])
        try: return sh.worksheet("bull_positions")
        except Exception:
            ws = sh.add_worksheet(title="bull_positions", rows=1000, cols=20)
            ws.append_row(BULL_SHEET_COLS); return ws
    except Exception: return None

def bull_gs_load() -> pd.DataFrame:
    ws = bull_get_ws()
    if ws is None: return pd.DataFrame(columns=BULL_SHEET_COLS)
    try:
        data = ws.get_all_records()
        if not data: return pd.DataFrame(columns=BULL_SHEET_COLS)
        df = pd.DataFrame(data)
        for col in ["加碼次數", "進場價", "上次加碼價", "持倉最高價"]: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df[BULL_SHEET_COLS]
    except Exception: return pd.DataFrame(columns=BULL_SHEET_COLS)

def bull_gs_save(df: pd.DataFrame):
    if ws := bull_get_ws():
        try: ws.clear(); ws.append_row(BULL_SHEET_COLS); ws.append_rows(df[BULL_SHEET_COLS].fillna("").values.tolist(), value_input_option="USER_ENTERED")
        except Exception: pass

# ==========================================
# 🐂 BULL_v7 布林滾雪球策略（掃描器用）
# ==========================================
BULL_PARAMS = {"boll_period": 15, "boll_std": 2.1, "squeeze_n": 15, "squeeze_lookback": 5, "vol_ma_days": 5, "vol_ratio": 1.5, "sma_trend_days": 3, "min_vol_shares": 1_000_000, "vol_ma20_days": 20, "vol_heavy_days": 5, "vol_shrink_days": 15, "init_position": 0.5, "add_position": 0.5, "addon_b_profit": 0.10}

def bull_calc_indicators(df, params):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    df = df[[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]].copy()
    close  = df["Close"]; volume = df["Volume"]; bp = params["boll_period"]
    df["SMA"] = close.rolling(bp).mean(); df["Std"] = close.rolling(bp).std()
    df["Upper"] = df["SMA"] + df["Std"] * params["boll_std"]; df["Lower"] = df["SMA"] - df["Std"] * params["boll_std"]
    df["Bandwidth"] = df["Upper"] - df["Lower"]; df["Vol_MA"] = volume.rolling(params["vol_ma_days"]).mean()
    df["Is_Squeeze"] = df["Bandwidth"] == df["Bandwidth"].rolling(params["squeeze_n"]).min()
    df["Squeeze_Recent"] = df["Is_Squeeze"].shift(1).rolling(params["squeeze_lookback"]).max() == 1
    df["SMA_Up"] = df["SMA"] > df["SMA"].shift(params["sma_trend_days"])
    df["Vol_MA20"] = volume.rolling(params["vol_ma20_days"]).mean()
    df["Vol_Heavy"] = volume.rolling(params["vol_heavy_days"]).mean()
    df["Vol_Shrink"] = volume.rolling(params["vol_shrink_days"]).mean()
    return df

def bull_scan_one(code, info, df_raw, params, pos_map, scan_date_ts):
    result = {"entry": None, "addon_a": None, "addon_b": None, "exit": None, "high_update": None}
    try:
        df = bull_calc_indicators(df_raw, params)
        df = df[df.index <= scan_date_ts]
        if len(df) < 50: return result

        sym = info["symbol"]; name = info["short_name"]; r = df.iloc[-1]; r1 = df.iloc[-2]
        c = float(r["Close"]); sma_i = float(r["SMA"]); lo_i = float(r["Lower"]); l_i = float(r["Low"])
        vol_i = float(r["Volume"]); vheavy = float(r["Vol_Heavy"]) if not pd.isna(r["Vol_Heavy"]) else 0

        if sym not in pos_map:
            if not (pd.isna(r["Squeeze_Recent"]) or pd.isna(r["Vol_MA20"])):
                if bool(r["Squeeze_Recent"]) and bool(r["SMA_Up"]) and (c > float(r["Upper"])) and (vol_i > float(r["Vol_MA"]) * params["vol_ratio"]) and (float(r["Vol_MA20"]) > params["min_vol_shares"]):
                    result["entry"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "上軌": round(float(r["Upper"]), 2), "15MA": round(sma_i, 2), "成交量": int(vol_i), "5MA量": int(r["Vol_MA"]) if not pd.isna(r["Vol_MA"]) else 0}
        else:
            pos = pos_map[sym]; ep = float(pos["進場價"]); lap = float(pos["上次加碼價"]); pp = float(pos["持倉最高價"])
            if c > pp: result["high_update"] = (sym, round(c, 2))

            heavy_break = (c < sma_i) and (vol_i >= vheavy) and vheavy > 0
            if heavy_break or (l_i <= lo_i):
                result["exit"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "15MA": round(sma_i, 2), "進場價": round(ep, 2), "損益%": round((c - ep) / ep * 100, 2), "加碼次數": int(pos.get("加碼次數", 0)), "出場原因": "出量跌破15MA" if heavy_break else "最低碰下軌"}
                return result

            vshrink = float(r1["Vol_Shrink"]) if not pd.isna(r1["Vol_Shrink"]) else 0
            if (float(r1["Close"]) < float(r1["SMA"])) and (float(r1["Volume"]) < vshrink if vshrink > 0 else False) and (c >= sma_i):
                result["addon_a"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "15MA": round(sma_i, 2), "加碼次數": int(pos.get("加碼次數", 0)), "加碼類型": "A 回測站回"}

            profit = (c - lap) / lap if lap > 0 else 0
            if c > pp and profit >= params["addon_b_profit"]:
                result["addon_b"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "持倉最高價": round(pp, 2), "距上次加碼": f"+{profit*100:.1f}%", "加碼次數": int(pos.get("加碼次數", 0)), "加碼類型": "B 突破新高"}
    except Exception: pass
    return result

def bull_run_scan(stock_map, all_data, scan_date_str, bull_positions, bull_params, max_workers=16, status_text=None, progress_bar=None):
    scan_date_ts = pd.Timestamp(scan_date_str)
    pos_map = {row["symbol"]: row for _, row in bull_positions.iterrows()} if len(bull_positions) > 0 else {}
    res = {"entry": [], "addon_a": [], "addon_b": [], "exit": [], "high_updates": []}
    valid_codes = {str(c): df for c, df in all_data.items() if str(c) in stock_map}
    
    total = len(valid_codes); done = 0
    if status_text: status_text.text(f"🐂 BULL_v7 策略運算中... (共 {total} 支)")
    if progress_bar: progress_bar.progress(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(bull_scan_one, code, stock_map[code], df, bull_params, pos_map, scan_date_ts): code for code, df in valid_codes.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 100 == 0 or done == total:
                if progress_bar: progress_bar.progress(done / max(total, 1), text=f"🐂 BULL_v7 策略運算 {done}/{total}...")
            r = future.result()
            for k in res.keys(): 
                if r.get(k): res[k].append(r[k])

    if status_text: status_text.success(f"✅ 全部掃描完成｜進場 {len(res['entry'])} | 加碼A {len(res['addon_a'])} | 加碼B {len(res['addon_b'])} | 出場 {len(res['exit'])}")
    if progress_bar: progress_bar.empty()
    return res

# ==========================================
# 📂 歷史資料管理
# ==========================================
PARQUET_PATH = "data/history.parquet"

def load_history_parquet():
    if not os.path.exists(PARQUET_PATH): return None, f"找不到 {PARQUET_PATH}，請確認 data/ 資料夾已上傳"
    try:
        df_all = pd.read_parquet(PARQUET_PATH)
        df_all["date"] = pd.to_datetime(df_all["date"])
        data_store = {}
        for code, grp in df_all.groupby("code"):
            grp = grp.sort_values("date").set_index("date")
            grp = grp[["Open","High","Low","Close","Volume"]].copy()
            grp.index.name = None
            data_store[str(code)] = grp # 🔧 修復1: 強制將 Parquet 讀出的 code 轉為字串，確保與 API 一致
        last_date = df_all["date"].max().strftime("%Y-%m-%d")
        return data_store, f"✅ 歷史資料載入完成：{len(data_store)} 支股票，最新日期 {last_date}"
    except Exception as e: return None, f"❌ 讀取 {PARQUET_PATH} 失敗：{e}"

def fetch_data_batch(stock_map, period="300d", chunk_size=150):
    all_symbols = [info['symbol'] for info in stock_map.values()]
    data_store = {}; symbol_to_code = {v['symbol']: str(k) for k, v in stock_map.items()}
    total_chunks = (len(all_symbols) // chunk_size) + 1
    progress_text = st.empty(); bar = st.progress(0)

    for i in range(0, len(all_symbols), chunk_size):
        chunk = all_symbols[i:i + chunk_size]; chunk_idx = (i // chunk_size) + 1
        progress_text.text(f"📥 正在批量下載歷史資料... (批次 {chunk_idx}/{total_chunks})"); bar.progress(chunk_idx / total_chunks)
        try:
            batch_df = yf.download(" ".join(chunk), period=period, group_by='ticker', threads=True, auto_adjust=True, progress=False)
            if not batch_df.empty:
                if isinstance(batch_df.columns, pd.MultiIndex):
                    for symbol in chunk:
                        if symbol in batch_df and not (stock_df := batch_df[symbol].dropna()).empty:
                            if stock_df.index.tz is not None: stock_df.index = stock_df.index.tz_localize(None)
                            if c := symbol_to_code.get(symbol): data_store[c] = stock_df
                elif not (stock_df := batch_df.dropna()).empty:
                    if stock_df.index.tz is not None: stock_df.index = stock_df.index.tz_localize(None)
                    if c := symbol_to_code.get(chunk[0]): data_store[c] = stock_df
            time.sleep(1)
        except Exception: continue
    bar.empty()
    return data_store

# ==========================================
# ⚡ 全時段即時報價 (100% 純 Shioaji API)
# ==========================================
def fetch_realtime_batch(codes_list, _api_instance, status_text=None):
    """不分盤中或盤後，一律使用永豐金 Shioaji API 獲取極速快照"""
    realtime_data = {}
    _txt = status_text if status_text else st.sidebar.empty()
    rt_bar = st.sidebar.progress(0)
    
    try:
        contracts = []
        for c in codes_list:
            c_str = str(c)
            # 🔧 增強: 更安全的合約抓取方式
            contract = _api_instance.Contracts.Stocks.get(c_str)
            if contract: contracts.append(contract)
            
        total_c = len(contracts)
        if total_c > 0:
            _txt.text(f"⚡ 永豐金快照擷取中 (共 {total_c} 支)...")
            for i in range(0, total_c, 300):
                chunk = contracts[i:i + 300]
                for snap in _api_instance.snapshots(chunk):
                    c_price = snap.close if snap.close > 0 else snap.previous_close
                    realtime_data[str(snap.code)] = {
                        "latest_trade_price": str(c_price), "open": str(snap.open if snap.open > 0 else c_price),
                        "high": str(snap.high if snap.high > 0 else c_price), "low": str(snap.low if snap.low > 0 else c_price),
                        "accumulate_trade_volume": str(int(snap.total_volume))
                    }
                rt_bar.progress(min(100, int((i + 300) / total_c * 100)))
            
        rt_bar.progress(100); _txt.text(f"✅ 報價快照 (永豐金) 更新完成｜成功取得 {len(realtime_data)} 支")
    except Exception as e:
        logger.error(f"Shioaji 快照抓取錯誤: {e}"); _txt.text(f"❌ 永豐金連線異常: {e}"); rt_bar.progress(100)
    return realtime_data

# ==========================================
# 📈 大盤狀態與處置股模組 
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_status(analysis_date_str, _api_instance=None):
    try:
        df = yf.Ticker("^TWII").history(period="1y")
        if df.empty or df.index.tz is not None: df.index = df.index.tz_localize(None) if not df.empty else df.index
        if df.empty: return {'strong': True, 'score': 5, 'label': '無法取得大盤資料', 'close': '-', 'ma20': '-', 'ma60': '-'}
        
        today_str = _now_tw().strftime('%Y-%m-%d')
        df = df[~df.index.duplicated(keep='last')].copy() # 確保歷史庫索引乾淨
        
        if _api_instance and analysis_date_str == today_str:
            try:
                if snap := _api_instance.snapshots([_api_instance.Contracts.Indices.TSE.TSE001]):
                    rt_c = snap[0].close if snap[0].close > 0 else snap[0].previous_close
                    today_ts = pd.Timestamp(today_str)
                    if today_ts in df.index: df.loc[today_ts, 'Close'] = rt_c
                    else: df = pd.concat([df, pd.Series({'Close': rt_c}, name=today_ts).to_frame().T])
            except Exception: pass

        latest_idx = df.index.get_loc(pd.Timestamp(analysis_date_str)) if pd.Timestamp(analysis_date_str) in df.index else -1
        close = df['Close'].ffill(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
        c = close.iloc[latest_idx]; m20 = ma20.iloc[latest_idx]; m60 = ma60.iloc[latest_idx]
        
        score = sum([4 if c > m20 else 0, 3 if c > m60 else 0, 3 if m20 > m60 else 0])
        return {'strong': score >= 7, 'score': score, 'label': "🟢 大盤強勢" if score >= 7 else ("🟡 大盤偏弱" if score >= 4 else "🔴 大盤弱勢"), 'close': round(c, 0), 'ma20': round(m20, 0), 'ma60': round(m60, 0)}
    except Exception: return {'strong': True, 'score': 5, 'label': '大盤資料異常', 'close': '-', 'ma20': '-', 'ma60': '-'}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_disposal_stocks():
    disposal = {}
    def roc_to_date(s): parts = s.strip().replace('/', '-').split('-'); return datetime.date(int(parts[0].strip()) + 1911, int(parts[1].strip()), int(parts[2].strip()))
    try:
        today_dt = _now_tw().date(); url = f"https://www.twse.com.tw/rwd/zh/announcement/punish?response=json&startDate={(today_dt - datetime.timedelta(days=30)).strftime('%Y%m%d')}&endDate={(today_dt + datetime.timedelta(days=30)).strftime('%Y%m%d')}"
        if (res := requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15, verify=False)).json().get('stat') == 'OK':
            for r in res.json().get('data', []):
                try:
                    c = str(r[2]).strip(); p = str(r[6]); d = (str(r[7]) if len(r)>7 else '') + (str(r[8]) if len(r)>8 else '')
                    if c.isdigit() and len(c)==4 and int(c)<9000 and ('每五分鐘' in d or '每5分鐘' in d) and ('～' in p or '~' in p):
                        end_dt = roc_to_date(p.split('～')[1] if '～' in p else p.split('~')[1])
                        if c not in disposal or end_dt > disposal[c]['end']: disposal[c] = {'name': str(r[3]).strip(), 'start': roc_to_date(p.split('～')[0] if '～' in p else p.split('~')[0]), 'end': end_dt, 'market': 'twse', 'symbol': f"{c}.TW", 'match_type': '5分鐘撮合'}
                except Exception: pass
    except Exception: pass
    
    for url in ["https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"]:
        try:
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15, verify=False)
            if res.text.strip().startswith('['):
                for r in res.json():
                    try:
                        c = str(r.get('SecuritiesCompanyCode', '')).strip(); p = str(r.get('DispositionPeriod', '')).strip(); d = str(r.get('DisposalCondition', ''))
                        if c.isdigit() and len(c)==4 and int(c)<9000 and ('每5分鐘' in d or '每五分鐘' in d) and '~' in p:
                            s, e = p.split('~')
                            disposal[c] = {'name': str(r.get('CompanyName', '')).strip(), 'start': datetime.date(int(s[0:3])+1911, int(s[3:5]), int(s[5:7])), 'end': datetime.date(int(e[0:3])+1911, int(e[3:5]), int(e[5:7])), 'market': 'tpex', 'symbol': f"{c}.TWO", 'match_type': '5分鐘撮合'}
                    except Exception: pass
        except Exception: pass
    return disposal

def analyze_disposal_stock(code, info, analysis_date, min_vol=0):
    try:
        df = yf.Ticker(info['symbol']).history(period="60d", auto_adjust=True)
        if df.empty or len(df) < 10: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if min_vol > 0 and not (dv := df[(pd.Timestamp(info['start']) <= df.index) & (df.index <= pd.Timestamp(analysis_date))]).empty and dv['Volume'].min() < min_vol * 1000: return None
        df['ma5'] = df['Close'].rolling(5).mean(); df['ma10'] = df['Close'].rolling(10).mean()
        df_period = df[(df.index >= pd.Timestamp(info['start'])) & (df.index <= pd.Timestamp(analysis_date))].copy()
        if len(df_period) < 2: return None

        position = None; trades = []; latest_signal = None
        for i in range(1, len(df_period)):
            ts = df_period.index[i].strftime('%Y-%m-%d'); c = df_period['Close'].iloc[i]; ph = df_period['High'].iloc[i-1]
            if position is None:
                if c > ph:
                    position = {'entry_date': ts, 'entry_price': c}
                    latest_signal = {'status': '🟢 持有中', 'entry_date': ts, 'entry_price': f"{c:.2f}", 'exit_date': '-', 'exit_price': '-', 'profit_pct': '-', 'exit_reason': '-', 'signal_date': ts, 'signal_close': f"{c:.2f}", 'signal_prev_high': f"{ph:.2f}", 'signal_pct': f"{(c - df_period['Close'].iloc[i-1]) / df_period['Close'].iloc[i-1] * 100:+.2f}%"}
            else:
                pct = (c - position['entry_price']) / position['entry_price'] * 100
                if pd.notna(exit_ma := df_period['ma5'].iloc[i] if pct >= 15.0 else df_period['ma10'].iloc[i]) and c < exit_ma:
                    trades.append({'entry_date': position['entry_date'], 'entry_price': position['entry_price'], 'exit_date': ts, 'exit_price': c, 'profit_pct': pct, 'exit_reason': f"跌破{'5MA' if pct >= 15 else '10MA'}"}); position = None; latest_signal = None
                else: latest_signal = {'status': '🟢 持有中', 'entry_date': position['entry_date'], 'entry_price': f"{position['entry_price']:.2f}", 'exit_date': '-', 'exit_price': '-', 'profit_pct': f"{pct:+.2f}%", 'exit_reason': f"出場條件：跌破{'5MA' if pct >= 15 else '10MA'}", 'signal_date': position['entry_date'], 'signal_close': f"{position['entry_price']:.2f}", 'signal_prev_high': '-', 'signal_pct': '-'}
        
        if not latest_signal and not trades: return None
        return {'code': code, 'name': info['name'], 'market': '上市' if info['market'] == 'twse' else '上櫃', 'disposal_start': info['start'].strftime('%Y-%m-%d'), 'disposal_end': info['end'].strftime('%Y-%m-%d'), 'days_left': max((info['end'] - analysis_date).days, 0), 'status': latest_signal['status'] if latest_signal else '⚫ 已出場', 'is_new_today': bool(latest_signal and latest_signal.get('entry_date') == analysis_date.strftime('%Y-%m-%d') and latest_signal['status'] == '🟢 持有中'), 'entry_date': latest_signal['entry_date'] if latest_signal else trades[-1]['entry_date'], 'entry_price': latest_signal['entry_price'] if latest_signal else f"{trades[-1]['entry_price']:.2f}", 'exit_date': latest_signal.get('exit_date', '-') if latest_signal else trades[-1]['exit_date'], 'exit_price': latest_signal.get('exit_price', '-') if latest_signal else f"{trades[-1]['exit_price']:.2f}", 'profit_pct': latest_signal.get('profit_pct', '-') if latest_signal else f"{trades[-1]['profit_pct']:+.2f}%", 'exit_reason': latest_signal.get('exit_reason', '-') if latest_signal else trades[-1]['exit_reason'], 'signal_date': latest_signal.get('signal_date', '-') if latest_signal else '-', 'signal_close': latest_signal.get('signal_close', '-') if latest_signal else '-', 'signal_prev_high': latest_signal.get('signal_prev_high', '-') if latest_signal else '-', 'signal_pct': latest_signal.get('signal_pct', '-') if latest_signal else '-', 'total_trades': len(trades)}
    except Exception: return None

def show_disposal_table(data):
    if not data: return
    df = pd.DataFrame(data)
    col_map = {'code': '代號', 'name': '名稱', 'market': '市場', 'disposal_start': '處置開始', 'disposal_end': '處置結束', 'days_left': '剩餘天數', 'status': '狀態', 'entry_date': '進場日', 'entry_price': '進場價', 'exit_date': '出場日', 'exit_price': '出場價', 'profit_pct': '損益', 'exit_reason': '出場條件', 'signal_date': '訊號日', 'signal_close': '訊號收盤', 'signal_prev_high': '前日高點', 'signal_pct': '訊號漲幅'}
    st.dataframe(df[[c for c in col_map if c in df.columns]].rename(columns=col_map), width='stretch', hide_index=True)

# ==========================================
# 🏆 綜合分析引擎與評分
# ==========================================
def calc_sniper_score(c_today, prev_h, defense_price, s_high, s_low, s_close, setup_idx, idx, high, low, volume, op, market_score, pullback_depth=0):
    score = 0; details = {}
    risk_pct = (c_today - defense_price) / c_today * 100 if c_today > 0 else 0
    details['風險距離'] = round(risk_pct, 1)
    if risk_pct <= 3.0: score += 35
    elif risk_pct <= 5.0: score += 25
    elif risk_pct <= 8.0: score += 15

    vol_ratio = volume.iloc[idx] / volume.iloc[setup_idx:idx].mean() if setup_idx > 0 and idx > setup_idx and volume.iloc[setup_idx:idx].mean() > 0 else 0
    details['量能倍數'] = round(vol_ratio, 1)
    if 2.0 <= vol_ratio <= 4.0: score += 25
    elif 1.5 <= vol_ratio < 2.0: score += 18
    elif vol_ratio >= 4.0: score += 12
    elif vol_ratio >= 1.0: score += 8

    close_strength = (c_today - low.iloc[idx]) / (high.iloc[idx] - low.iloc[idx]) if (high.iloc[idx] - low.iloc[idx]) > 0 else 0
    details['收盤強度'] = round(close_strength, 2)
    if close_strength >= 0.85: score += 20
    elif close_strength >= 0.70: score += 14
    elif close_strength >= 0.50: score += 7

    details['回測深度'] = round(pullback_depth, 1)
    if pullback_depth <= 20: score += 10
    elif pullback_depth <= 38: score += 7
    elif pullback_depth <= 50: score += 3

    score += round(market_score / 10 * 5)
    return min(score, 100), details

def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None, market_score=5):
    try:
        df = pre_loaded_df.copy() if pre_loaded_df is not None else get_stock_data_with_realtime(code, info['symbol'], analysis_date_str, api)
        if df is None or df.empty or len(df) < 200: return "資料長度不足 (<200天)"
        
        # 🔧 修復2: 徹底防堵 Pandas Index 重複報錯
        df = df.loc[~df.index.duplicated(keep='last')].copy()
        
        if pd.Timestamp(analysis_date_str) not in df.index: return f"無 {analysis_date_str} 交易資料"
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
        stock_name = info['short_name']; sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db)
        result_sniper = None; result_day = None

        s_ma_trend = params['s_ma_trend']; s_use_year = params['s_use_year']; s_big_candle = params['s_big_candle']; s_min_vol = params['s_min_vol']
        ma_t = close.rolling(window=s_ma_trend).mean(); ma_y = close.rolling(window=240).mean(); ma10 = close.rolling(window=10).mean(); ma20 = close.rolling(window=20).mean(); ma60 = close.rolling(window=60).mean()
        vol_ma_setup = volume.rolling(window=params.get('s_vol_ma_days', 20)).mean()

        if (volume.iloc[idx] >= s_min_vol) and (not s_use_year or not (pd.isna(ma_y.iloc[idx]) or close.iloc[idx] < ma_y.iloc[idx])) and (close.iloc[idx] > ma_t.iloc[idx] > ma_t.iloc[idx-1]) and (pd.notna(ma60.iloc[idx]) and close.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma60.iloc[idx]):
            is_setup = ((close.iloc[idx] - close.iloc[idx-1]) / close.iloc[idx-1] > s_big_candle and volume.iloc[idx] > vol_ma_setup.iloc[idx] * params.get('s_vol_ratio', 0.7) and close.iloc[idx] > op.iloc[idx])
            setup_found = False; s_high = 0; s_low = 0; s_close = 0; s_date = ""; setup_idx = -1; defense_price = 0

            for k in range(1, params.get('s_setup_lookback', 25) + 1):
                if (b_idx := idx - k) < 1: break
                if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > s_big_candle and volume.iloc[b_idx] > vol_ma_setup.iloc[b_idx] * params.get('s_vol_ratio', 0.7) and close.iloc[b_idx] > op.iloc[b_idx]):
                    setup_found = True; setup_idx = b_idx; s_low = low.iloc[b_idx]; s_high = high.iloc[b_idx]; s_close = close.iloc[b_idx]; s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    defense_price = (close.iloc[b_idx-1] if close.iloc[b_idx-1] >= op.iloc[b_idx-1] else op.iloc[b_idx-1]) * 0.99 if s_low > high.iloc[b_idx-1] else s_low * 0.99
                    break

            c_today = close.iloc[idx]; prev_h = high.iloc[idx-1]; daily_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1] * 100

            if setup_found:
                is_broken = False; dropped_below_high = False
                for k in range(setup_idx + 1, idx + 1):
                    if close.iloc[k] < defense_price: is_broken = True; break
                    if close.iloc[k] < s_high: dropped_below_high = True

                if not is_broken and (pullback_depth := (s_close - (close.iloc[setup_idx + 1 : idx].min() if setup_idx + 1 < idx else s_close)) / s_close * 100) <= params.get('s_pullback_max', 50):
                    is_breakout = c_today > prev_h; is_gap_breakout = (op.iloc[idx] > high.iloc[idx-1]) and (close.iloc[idx] > op.iloc[idx])
                    if not dropped_below_high:
                        if (c_today - s_close) / s_close <= 0.10:
                            score, score_details = calc_sniper_score(c_today, prev_h, defense_price, s_high, s_low, s_close, setup_idx, idx, high, low, volume, op, market_score, pullback_depth)
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🚀 強勢突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}", "風險距離": f"{'✅' if score_details.get('風險距離',0)<=5.0 else '⚠️'}{score_details.get('風險距離',0):.1f}%", "回測深度": f"{'🔒' if score_details.get('回測深度',0)<=38 else '📊'}{score_details.get('回測深度',0):.1f}%", "sort_pct": daily_pct, "_score": score, "_risk_pct": score_details.get('風險距離',0), "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low}) if is_breakout else ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "💪 強勢整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
                    else:
                        if (is_breakout and close.iloc[idx-1] <= (s_high * 1.02)) or is_gap_breakout:
                            score, score_details = calc_sniper_score(c_today, prev_h, defense_price, s_high, s_low, s_close, setup_idx, idx, high, low, volume, op, market_score, pullback_depth)
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🚀 N字跳空" if is_gap_breakout else "🎯 N字突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}", "風險距離": f"{'✅' if score_details.get('風險距離',0)<=5.0 else '⚠️'}{score_details.get('風險距離',0):.1f}%", "回測深度": f"{'🔒' if score_details.get('回測深度',0)<=38 else '📊'}{score_details.get('回測深度',0):.1f}%", "sort_pct": daily_pct, "_score": score, "_risk_pct": score_details.get('風險距離',0), "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
                        else: result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "📉 回檔整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
            elif is_setup:
                result_sniper = ("new_setup", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🔥 剛起漲", "漲幅": f"{daily_pct:+.2f}%", "sort_pct": daily_pct, "_setup_date": df.index[idx].strftime('%Y-%m-%d'), "_defense": defense_price, "_signal_high": high.iloc[idx], "_signal_low": low.iloc[idx]})

        d_period = params['d_period']; d_threshold = params['d_threshold']; d_min_vol = params['d_min_vol']; d_min_pct = params['d_min_pct']
        d_close = close.iloc[idx]; d_open = op.iloc[idx]; d_high = high.iloc[idx]; d_volume = volume.iloc[idx]; d_prev_close = close.iloc[idx-1]
        if idx >= d_period and (d_close > d_open) and ((d_high - d_close) / d_close < 0.01) and (d_min_pct/100 < (pct_chg_val := (d_close - d_prev_close) / d_prev_close) < 0.095) and ((d_volume / 1000) > d_min_vol) and (d_close >= ((prev_period_high := high.iloc[idx-d_period : idx].max()) * (1 - (d_threshold / 100)))) and (d_high <= prev_period_high):
            result_day = {"代號": code, "名稱": stock_name, "收盤": f"{d_close:.2f}", "產業": sector_name, "漲幅": f"{(pct_chg_val*100):.2f}%", "成交量": int(d_volume/1000), "前波高點": f"{prev_period_high:.2f}", "距離高點": f"{(d_close - prev_period_high) / prev_period_high * 100:+.2f}%", "狀態": "⚡ 蓄勢待發"}

        return {'sniper': result_sniper, 'day': result_day}
    except Exception as e: return f"程式執行錯誤: {str(e)}"

def display_full_table(df):
    if df is not None and not df.empty:
        st.dataframe(df[[c for c in df.columns if not c.startswith('_')]], hide_index=True, use_container_width=True, height=(len(df) * 35) + 38)
    else: st.info("無")

# ==========================================
# 📊 個股診斷強化版
# ==========================================
def run_diagnosis(stock_input, analysis_date_str, params):
    df = get_stock_data_with_realtime(stock_input, f"{stock_input}.TW", analysis_date_str, api)
    if df is None: df = get_stock_data_with_realtime(stock_input, f"{stock_input}.TWO", analysis_date_str, api)
    return df, f"{stock_input}.TW"

def plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info=None):
    df = df.copy(); df['MA_Trend'] = df['Close'].rolling(window=params['s_ma_trend']).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean(); df['MA240'] = df['Close'].rolling(window=240).mean()
    plot_df = df.tail(250)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], subplot_titles=("K線圖", "成交量"))
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線', increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Trend'], line=dict(color='blue', width=1.5), name=f"{params['s_ma_trend']}MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], line=dict(color='orange', width=1.2), name='20MA'), row=1, col=1)
    if plot_df['MA240'].notna().any(): fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA240'], line=dict(color='purple', width=1.2, dash='dash'), name='240MA'), row=1, col=1)

    if sniper_info:
        if sniper_info.get('_setup_date') and pd.Timestamp(sniper_info.get('_setup_date')) in plot_df.index:
            fig.add_vline(x=pd.Timestamp(sniper_info.get('_setup_date')), line_width=2, line_dash="dash", line_color="gold", annotation_text=f"📍訊號日 {sniper_info.get('_setup_date')}", annotation_position="top right")
        if sniper_info.get('_defense'): fig.add_hline(y=sniper_info.get('_defense'), line_width=2, line_dash="dot", line_color="red", annotation_text=f"🛡️ 防守 {sniper_info.get('_defense'):.2f}", annotation_position="bottom right")
        if sniper_info.get('_signal_high'): fig.add_hline(y=sniper_info.get('_signal_high'), line_width=1.5, line_dash="dot", line_color="orange", annotation_text=f"⚡ 長紅高 {sniper_info.get('_signal_high'):.2f}", annotation_position="top right")
        fig.add_annotation(text=f"策略狀態：{sniper_info.get('狀態', '')}", xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, font=dict(size=14, color="white"), bgcolor="rgba(50,50,50,0.7)", bordercolor="gray", borderwidth=1)

    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量', marker_color=['red' if c >= o else 'green' for c, o in zip(plot_df['Close'], plot_df['Open'])]), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=650, title_text=f"{stock_input} 個股診斷圖", template="plotly_dark")
    return fig

def run_backtest_ui(df, stock_input, params):
    try:
        bt_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
        bt_df.index = pd.to_datetime(bt_df.index)
        bt = Backtest(bt_df, SniperStrategy, cash=100000, commission=0.001425 * 2, trade_on_close=False)
        stats = bt.run(ma_trend_period=params['s_ma_trend'], big_candle_pct=params['s_big_candle'], min_volume_shares=params['s_min_vol'], use_year_line=params['s_use_year'])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總報酬率", f"{stats['Return [%]']:.1f}%"); c2.metric("最大回撤", f"{stats['Max. Drawdown [%]']:.1f}%"); c3.metric("勝率", f"{stats['Win Rate [%]']:.1f}%"); c4.metric("交易次數", f"{stats['# Trades']}")
        if not stats['_trades'].empty:
            trades = stats['_trades'][['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']].copy()
            trades.columns = ['進場時間', '出場時間', '進場價', '出場價', '損益', '報酬%']
            trades['報酬%'] = (trades['報酬%'] * 100).round(2); trades['損益'] = trades['損益'].round(2)
            st.dataframe(trades, hide_index=True, use_container_width=True)
        
        fig_eq = go.Figure(go.Scatter(x=stats['_equity_curve']['Equity'].index, y=stats['_equity_curve']['Equity'].values, fill='tozeroy', line=dict(color='cyan'), name='權益曲線'))
        fig_eq.update_layout(title="📈 策略權益曲線", height=300, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_eq, use_container_width=True)
    except Exception as e: st.error(f"回測錯誤：{e}")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室 V18")
st.sidebar.caption("純永豐金 API 企業級 24H 連線版")

if os.path.exists(PARQUET_PATH):
    try:
        size_mb = os.path.getsize(PARQUET_PATH) / 1024 / 1024
        st.sidebar.success(f"📂 歷史資料：Parquet ({size_mb:.1f} MB)")
    except Exception: st.sidebar.success("📂 歷史資料：Parquet ✅")
else: st.sidebar.warning("⚠️ 未找到 data/history.parquet，將使用 yfinance 下載")

st.sidebar.success("🟢 Shioaji 全時段連線中")

analysis_date_input = st.sidebar.date_input("分析基準日", _now_tw().date())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

start_scan = st.sidebar.button("🚀 開始全域掃描 (極速版)", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)

st.sidebar.divider()

with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=False):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA) 濾網", value=True)
    s_big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 5.0, 0.5) / 100
    s_min_vol = st.number_input("波段最小量 (張)", value=1000) * 1000
    s_setup_lookback = st.slider("長紅K回溯天數", 5, 30, 25, 5)
    s_vol_ma_days = st.slider("長紅K量能基準 (MA天數)", 5, 20, 20, 5)
    s_vol_ratio = st.slider("長紅K量能門檻 (倍)", 0.5, 1.5, 0.7, 0.1)
    s_pullback_max = st.slider("整理回測深度上限 (%)", 20, 70, 50, 5)

with st.sidebar.expander("🎯 處置股策略參數", expanded=False):
    d_show_all = st.checkbox("顯示全部（含無訊號）", value=True) # 🔧 已預設為 True
    d_vol_filter = st.checkbox("啟用底量濾網", value=False)
    d_min_vol_disp = st.number_input("底量門檻（張）", min_value=0, max_value=10000, value=1000, step=100, disabled=not d_vol_filter)

if bull_use_gsheet(): st.sidebar.success("🟢 BULL 持倉已連線 Google Sheets")
elif GSPREAD_AVAILABLE: st.sidebar.warning("⚠️ BULL 持倉：未設定 Sheets Secrets")
else: st.sidebar.info("ℹ️ BULL 持倉：暫存模式（未安裝 gspread）")

with st.sidebar.expander("🐂 BULL_v7 布林滾雪球參數", expanded=False):
    b_boll_period  = st.number_input("布林週期", value=15, min_value=5, max_value=60, key="b_bp")
    b_boll_std     = st.number_input("布林標準差", value=2.1, min_value=1.0, max_value=3.0, step=0.1, key="b_bs")
    b_squeeze_n    = st.number_input("壓縮天數", value=15, min_value=5, max_value=60, key="b_sn")
    b_squeeze_lb   = st.number_input("壓縮回溯天數", value=5, min_value=1, max_value=15, key="b_sl")
    b_vol_ratio    = st.number_input("爆量倍數", value=1.5, min_value=1.0, max_value=5.0, step=0.1, key="b_vr")
    b_sma_days     = st.number_input("中軌趨勢天數", value=3, min_value=1, max_value=10, key="b_sd")
    b_min_vol      = st.number_input("20日均量門檻(張)", value=1000, min_value=100, step=100, key="b_mv") * 1000
    b_addon_b_pct  = st.number_input("加碼B門檻(%)", value=10, min_value=1, max_value=50, key="b_ab") / 100

with st.sidebar.expander("⚡ 隔日沖策略參數 (短線)", expanded=False):
    d_period = st.slider("追蹤波段天數 (N)", 10, 120, 60, 5)
    d_threshold = st.slider("高點容許誤差 (%)", 0.0, 5.0, 1.0, 0.1)
    d_min_pct = st.slider("當日最低漲幅 (%)", 3.0, 9.0, 5.0, 0.1)
    d_min_vol = st.number_input("隔日沖最小量 (張)", value=1000, step=500)

st.sidebar.divider()
max_workers_input = st.sidebar.slider("策略運算效能 (執行緒數)", 1, 32, 16)

params = {'s_ma_trend': s_ma_trend, 's_use_year': s_use_year, 's_big_candle': s_big_candle, 's_min_vol': s_min_vol, 's_setup_lookback': s_setup_lookback, 's_vol_ma_days': s_vol_ma_days, 's_vol_ratio': s_vol_ratio, 's_pullback_max': s_pullback_max, 'd_period': d_period, 'd_threshold': d_threshold, 'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol, 'disp_show_all': d_show_all, 'disp_vol_filter': d_vol_filter, 'disp_min_vol': d_min_vol_disp}
bull_params = {**BULL_PARAMS, "boll_period": int(b_boll_period), "boll_std": float(b_boll_std), "squeeze_n": int(b_squeeze_n), "squeeze_lookback": int(b_squeeze_lb), "vol_ratio": float(b_vol_ratio), "sma_trend_days": int(b_sma_days), "min_vol_shares": int(b_min_vol), "addon_b_profit": float(b_addon_b_pct)}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🟢 狙擊手波段", "🎯 處置股策略", "🐂 BULL滾雪球", "⚡ 隔日沖雷達", "📊 個股診斷", "🔧 系統診斷"])

# Session State 管理
for k in ['scan_results', 'bull_results', 'disposal_results', 'disposal_active', 'scan_params', 'scan_date', 'market_status']:
    if k not in st.session_state: st.session_state[k] = None
if 'bull_positions' not in st.session_state: st.session_state['bull_positions'] = pd.DataFrame(columns=BULL_SHEET_COLS)
if 'bull_gs_loaded' not in st.session_state: st.session_state['bull_gs_loaded'] = False

if not st.session_state['bull_gs_loaded'] and bull_use_gsheet():
    df_gs = bull_gs_load()
    if len(df_gs) > 0: st.session_state['bull_positions'] = df_gs
    st.session_state['bull_gs_loaded'] = True

def params_changed():
    return st.session_state['scan_params'] is not None and (st.session_state['scan_params'] != params or st.session_state['scan_date'] != analysis_date_str)

# ==========================================
# 執行整合掃描
# ==========================================
if start_scan:
    stock_map = get_stock_info_map(api)
    status_text.text("📈 正在判斷大盤狀態...")
    market_status = fetch_market_status(analysis_date_str, api)
    st.session_state['market_status'] = market_status

    sniper_triggered = []; sniper_setup = []; sniper_watching = []; day_candidates = []; failed_list = []

    status_text.text("📂 正在讀取歷史資料 (data/history.parquet)...")
    history_data_store, load_msg = load_history_parquet()

    if history_data_store is None:
        status_text.text("⚠️ Parquet 不存在，改用 yfinance 下載...")
        history_data_store = fetch_data_batch(stock_map)
    else:
        status_text.text(load_msg)
        # 🔧 防護：確保 history_data_store 的 Key 與 stock_map 的 Key 同為字串，避免被過濾空！
        history_data_store = {str(k): v for k, v in history_data_store.items() if str(k) in stock_map}

    st.session_state['all_data_cache']   = history_data_store
    st.session_state['stock_map_cache']  = stock_map

    today_str = _now_tw().strftime('%Y-%m-%d')
    realtime_map = {}
    
    if analysis_date_str == today_str:
        status_text.text("⚡ 正在請求永豐金 Shioaji 報價快照...")
        realtime_map = fetch_realtime_batch(list(history_data_store.keys()), api, status_text=status_text)

    status_text.text("🧠 正在進行大盤與波段策略運算...")
    progress_bar.progress(0)

    tasks_data = {}
    def safe_f(val, default=0.0):
        try: return float(str(val).replace(',', '')) if val not in ('-', '', None) else default
        except: return default

    for code, df in history_data_store.items():
        code_str = str(code)
        # 🔧 防護：清除歷史庫中殘留的「今天」，避免 Index 重複造成 Pandas 報錯
        df = df.loc[~df.index.duplicated(keep='first')].copy()
        
        if code_str in realtime_map and realtime_map[code_str].get('latest_trade_price', '-') != '-':
            try:
                rt = realtime_map[code_str]
                c_p = safe_f(rt.get('latest_trade_price'))
                o_p = safe_f(rt.get('open')) or c_p
                h_p = safe_f(rt.get('high')) or c_p
                l_p = safe_f(rt.get('low')) or c_p
                v_p = safe_f(rt.get('accumulate_trade_volume')) * 1000
                
                if c_p > 0:
                    today_ts = pd.Timestamp(today_str)
                    new_row = pd.Series({'Open': o_p, 'High': h_p, 'Low': l_p, 'Close': c_p, 'Volume': v_p}, name=today_ts)
                    
                    # 🔧 防護：使用 loc 精準更新或寫入今天的資料列
                    if today_ts in df.index:
                        df.loc[today_ts] = new_row
                    else:
                        df = pd.concat([df, new_row.to_frame().T])
            except Exception as e: logger.warning(f"更新即時資料失敗 {code_str}: {e}")
        tasks_data[code_str] = df

    st.session_state['all_data_cache'] = tasks_data
    total = len(tasks_data); done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df, market_status.get('score', 5)): code for code, df in tasks_data.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total:
                progress_bar.progress(done / max(total, 1)); status_text.text(f"波段策略運算中: {done}/{total}")
            res = future.result()
            if isinstance(res, dict):
                if res['sniper']:
                    typ, data = res['sniper']
                    if typ == "triggered": sniper_triggered.append(data)
                    elif typ == "new_setup": sniper_setup.append(data)
                    elif typ == "watching": sniper_watching.append(data)
                if res['day']: day_candidates.append(res['day'])
            else:
                current_code = futures[future]
                failed_list.append(f"{current_code} {stock_map[current_code]['short_name']} : {res}")

    st.session_state['scan_results'] = {'sniper_triggered': sniper_triggered, 'sniper_setup': sniper_setup, 'sniper_watching': sniper_watching, 'day_candidates': day_candidates, 'failed_list': failed_list}

    # 處置股策略
    status_text.text("📋 正在取得並分析處置股清單...")
    progress_bar.progress(0)
    disposal_map = fetch_disposal_stocks()
    if not disposal_map: st.warning("⚠️ 無法自動取得處置股清單"); disposal_map = {}
    lookahead = analysis_date_input + datetime.timedelta(days=3)
    d_active = {k: v for k, v in disposal_map.items() if v['start'] <= lookahead and v['end'] >= analysis_date_input}
    d_active_now = {k: v for k, v in d_active.items() if v['start'] <= analysis_date_input}
    d_active_coming = {k: v for k, v in d_active.items() if v['start'] > analysis_date_input}

    d_results = []; total_d = len(d_active_now)
    for i, (code, info) in enumerate(d_active_now.items()):
        status_text.text(f"🔍 處置股分析中... {code} {info['name']} ({i+1}/{total_d})")
        progress_bar.progress((i+1) / max(total_d, 1))
        result = analyze_disposal_stock(code, info, analysis_date_input, min_vol=d_min_vol_disp if d_vol_filter else 0)
        if result: d_results.append(result)
        elif d_show_all: d_results.append({'code': code, 'name': info['name'], 'market': '上市' if info['market'] == 'twse' else '上櫃', 'disposal_start': info['start'].strftime('%Y-%m-%d'), 'disposal_end': info['end'].strftime('%Y-%m-%d'), 'days_left': max((info['end'] - analysis_date_input).days, 0), 'status': '⚪ 無訊號', 'is_new_today': False, 'entry_date': '-', 'entry_price': '-', 'exit_date': '-', 'exit_price': '-', 'profit_pct': '-', 'exit_reason': '-', 'signal_date': '-', 'signal_close': '-', 'signal_prev_high': '-', 'signal_pct': '-', 'total_trades': 0})
        time.sleep(0.1)

    st.session_state['disposal_results'] = d_results
    st.session_state['disposal_active'] = {'now': d_active_now, 'coming': d_active_coming}

    progress_bar.progress(1.0)
    status_text.success("✅ 全域掃描（波段、隔日沖、處置股）已完成！")
    st.session_state['scan_params'] = params.copy()
    st.session_state['scan_date'] = analysis_date_str

results = st.session_state['scan_results']
if results is not None and params_changed(): st.warning("⚠️ 您已修改策略參數或分析日期，目前顯示的結果為**上次掃描**的資料，請重新執行掃描以取得最新結果。")

if start_scan and 'all_data_cache' in st.session_state:
    try:
        st.session_state['bull_results'] = bull_run_scan(stock_map=st.session_state['stock_map_cache'], all_data=st.session_state['all_data_cache'], scan_date_str=analysis_date_str, bull_positions=st.session_state['bull_positions'], bull_params=bull_params, max_workers=max_workers_input, status_text=status_text, progress_bar=progress_bar)
    except Exception as e:
        logger.error(f"BULL 掃描錯誤: {e}")
        status_text.text("⚠️ BULL 掃描發生錯誤，請查看 log")

# ==========================================
# 介面顯示：Tab 1~6
# ==========================================
with tab1:
    st.header("🟢 狙擊手波段策略")
    st.caption(f"基準日: {analysis_date_str} | 策略：趨勢 + 實體長紅 + 型態確認 (防守點含 1% 誤差)")
    if mkt := st.session_state.get('market_status'):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("大盤狀態", mkt['label'])
        
        # 🔧 防護：確保當 API 獲取失敗時 (顯示 '-' )，排版不會崩潰
        c_val = mkt.get('close', '-')
        m20_val = mkt.get('ma20', '-')
        m60_val = mkt.get('ma60', '-')
        mc2.metric("加權指數", f"{c_val:,.0f}" if isinstance(c_val, (int, float)) else str(c_val))
        mc3.metric("20MA", f"{m20_val:,.0f}" if isinstance(m20_val, (int, float)) else str(m20_val))
        mc4.metric("60MA", f"{m60_val:,.0f}" if isinstance(m60_val, (int, float)) else str(m60_val))
        
        if not mkt['strong']: st.warning("⚠️ 大盤目前偏弱，建議降低持倉比例，訊號可信度下降。")
        st.divider()

    if results:
        if results.get('failed_list'):
            with st.expander(f"⚠️ 掃描失敗/無資料清單 ({len(results['failed_list'])})"): st.write(", ".join(results['failed_list']))
        s_trig = results['sniper_triggered']; trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]; trig_n = [x for x in s_trig if "N字" in x['狀態']]
        s_watch = results['sniper_watching']; watch_strong = [x for x in s_watch if "強勢整理" in x['狀態']]; watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]

        if trig_strong or trig_n:
            st.markdown("### 🎯 買點觸發訊號 (Actionable)"); st.caption("依漲幅排序，⚠️風險距離 > 5% 建議跳過")
            if trig_strong:
                st.markdown(f"#### 🚀 強勢突破 ({len(trig_strong)})")
                display_full_table(pd.DataFrame(trig_strong).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_score', '_risk_pct'] if c in pd.DataFrame(trig_strong).columns]))
            if trig_n:
                st.markdown(f"#### 🎯 N字突破 ({len(trig_n)})")
                display_full_table(pd.DataFrame(trig_n).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_score', '_risk_pct'] if c in pd.DataFrame(trig_n).columns]))
            st.divider(); st.markdown("### 🫧 產業強度分析"); st.caption("右上角 = 廣度與強度兼具的強勢產業，泡泡大小代表訊號數")

            if all_sig := trig_strong + trig_n + results.get('sniper_setup', []) + watch_strong:
                df_all = pd.DataFrame(all_sig)
                df_all['漲幅_num'] = df_all['漲幅'].str.replace('%','').str.replace('+','').astype(float)
                sector_grp = df_all.groupby('產業').agg(訊號數=('代號', 'count'), 平均漲幅=('漲幅_num', 'mean')).reset_index()
                fig_bubble = go.Figure(go.Scatter(x=sector_grp['訊號數'], y=sector_grp['平均漲幅'], mode='markers+text', text=sector_grp['產業'], textposition='top center', marker=dict(size=sector_grp['訊號數'] * 8 + 10, color=sector_grp['平均漲幅'], colorscale='RdYlGn', showscale=True, colorbar=dict(title='平均漲幅%'), line=dict(width=1, color='white')), hovertemplate="<b>%{text}</b><br>訊號數: %{x}<br>平均漲幅: %{y:.2f}%<extra></extra>"))
                fig_bubble.update_layout(height=420, template="plotly_dark", xaxis_title="訊號數（廣度）", yaxis_title="平均漲幅（強度）", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_bubble, use_container_width=True)
    else: st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab2:
    st.markdown("### 🎯 處置股策略"); st.caption("僅限每5分鐘撮合處置股｜進場：收盤站上前日最高點｜出場：獲利<15%跌破10MA，獲利≥15%跌破5MA")
    if st.session_state['disposal_results'] is not None:
        d_results = st.session_state['disposal_results']; d_active_now = st.session_state['disposal_active']['now']; d_active_coming = st.session_state['disposal_active']['coming']
        st.info(f"📋 處置中 **{len(d_active_now)}** 支｜即將開始（3天內）**{len(d_active_coming)}** 支")

        d_new_today = [r for r in d_results if r.get('is_new_today')]; d_holding = [r for r in d_results if r.get('status') == '🟢 持有中' and not r.get('is_new_today')]; d_exited = [r for r in d_results if r.get('status') == '⚫ 已出場']; d_no_signal = [r for r in d_results if r.get('status') == '⚪ 無訊號']

        st.markdown(f"#### 🔴 今日新訊號 ({len(d_new_today)} 支)")
        if d_new_today: show_disposal_table(d_new_today)
        else:
            if d_active_coming:
                st.info(f"今日無訊號，以下 {len(d_active_coming)} 支即將進入處置期間：")
                st.dataframe(pd.DataFrame([{'代號': k, '名稱': v['name'], '市場': '上市' if v['market'] == 'twse' else '上櫃', '處置開始': v['start'].strftime('%Y-%m-%d'), '處置結束': v['end'].strftime('%Y-%m-%d')} for k, v in d_active_coming.items()]), use_container_width=True, hide_index=True)
            else: st.info(f"基準日 {analysis_date_str} 無今日訊號")

        if d_holding: st.divider(); st.markdown(f"#### 🟢 持有中 ({len(d_holding)} 支)"); show_disposal_table(d_holding)
        if d_exited: st.divider(); st.markdown(f"#### ⚫ 已出場 ({len(d_exited)} 支)"); show_disposal_table(d_exited)
        if d_show_all and d_no_signal: st.divider(); st.markdown(f"#### ⚪ 無訊號 ({len(d_no_signal)} 支)"); show_disposal_table(d_no_signal)
        st.divider(); c1, c2, c3, c4 = st.columns(4); c1.metric("處置中", f"{len(d_active_now)} 支"); c2.metric("今日新訊號", f"{len(d_new_today)} 支"); c3.metric("持有中", f"{len(d_holding) + len(d_new_today)} 支"); c4.metric("已出場", f"{len(d_exited)} 支")
    else: st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab3:
    st.header("🐂 BULL_v7 布林滾雪球掃描器")
    bull_res = st.session_state.get('bull_results'); bull_pos = st.session_state['bull_positions']

    def bull_add_pos(sym, name, price, date_str):
        pos = st.session_state['bull_positions']
        if sym in pos['symbol'].values: return
        new_row = pd.DataFrame([{"symbol": sym, "name": name, "進場日期": date_str, "進場價": round(price, 2), "上次加碼價": round(price, 2), "持倉最高價": round(price, 2), "加碼次數": 0, "加碼紀錄": ""}])
        updated = pd.concat([pos, new_row], ignore_index=True); st.session_state['bull_positions'] = updated
        if bull_use_gsheet(): bull_gs_save(updated)

    def bull_do_addon(sym, addon_type, price, date_str):
        pos = st.session_state['bull_positions']
        if not (mask := pos['symbol'] == sym).any(): return
        idx = pos[mask].index[-1]; pos.loc[idx, '上次加碼價'] = round(price, 2); pos.loc[idx, '加碼次數'] = int(pos.loc[idx, '加碼次數']) + 1
        tag = f"{date_str}({addon_type})"; prev = str(pos.loc[idx, '加碼紀錄']); pos.loc[idx, '加碼紀錄'] = (prev + " → " + tag).strip(" → ") if prev else tag
        st.session_state['bull_positions'] = pos
        if bull_use_gsheet(): bull_gs_save(pos)

    def bull_remove_pos(sym):
        updated = st.session_state['bull_positions'][st.session_state['bull_positions']['symbol'] != sym].reset_index(drop=True)
        st.session_state['bull_positions'] = updated
        if bull_use_gsheet(): bull_gs_save(updated)

    if bull_res:
        pos = st.session_state['bull_positions']
        for sym_, price_ in bull_res.get('high_updates', []):
            if (mask := pos['symbol'] == sym_).any():
                idx_ = pos[mask].index[-1]
                if price_ > float(pos.loc[idx_, '持倉最高價']): pos.loc[idx_, '持倉最高價'] = price_
        st.session_state['bull_positions'] = pos; bull_pos = pos

    entry_n = len(bull_res['entry']) if bull_res else 0; addon_a_n = len(bull_res['addon_a']) if bull_res else 0; addon_b_n = len(bull_res['addon_b']) if bull_res else 0; exit_n = len(bull_res['exit']) if bull_res else 0; pos_n = len(bull_pos)
    c1, c2, c3, c4, c5 = st.columns(5); c1.metric("📋 持倉", pos_n); c2.metric("🟢 進場", entry_n); c3.metric("🔵 加碼B", addon_b_n); c4.metric("🟡 加碼A", addon_a_n); c5.metric("🔴 出場", exit_n)

    if not bull_res: st.info("👈 點擊左側「開始全域掃描」後即可看到 BULL_v7 訊號。")
    else:
        bt1, bt2, bt3, bt4, bt5 = st.tabs([f"🟢 進場({entry_n})", f"🔵 加碼B({addon_b_n})", f"🟡 加碼A({addon_a_n})", f"🔴 出場({exit_n})", f"📋 持倉({pos_n})"])
        with bt1:
            if not bull_res['entry']: st.info("今日無進場訊號")
            else:
                st.dataframe(pd.DataFrame(bull_res['entry']), use_container_width=True, hide_index=True); cols = st.columns(min(len(bull_res['entry']), 6))
                for i, r in enumerate(bull_res['entry']):
                    with cols[i % 6]:
                        if st.button(f"✅ 加入 {r['代號']}", key=f"bull_add_{r['symbol']}"): bull_add_pos(r['symbol'], r['名稱'], r['收盤價'], analysis_date_str); st.success(f"{r['代號']} 已加入持倉"); st.rerun()
        with bt2:
            if not bull_res['addon_b']: st.info("今日無加碼B訊號")
            else:
                st.dataframe(pd.DataFrame(bull_res['addon_b']), use_container_width=True, hide_index=True); cols = st.columns(min(len(bull_res['addon_b']), 6))
                for i, r in enumerate(bull_res['addon_b']):
                    with cols[i % 6]:
                        if st.button(f"🔵 加碼B {r['代號']}", key=f"bull_addb_{r['symbol']}"): bull_do_addon(r['symbol'], "突破新高", r['收盤價'], analysis_date_str); st.success(f"{r['代號']} 加碼B完成"); st.rerun()
        with bt3:
            if not bull_res['addon_a']: st.info("今日無加碼A訊號")
            else:
                st.dataframe(pd.DataFrame(bull_res['addon_a']), use_container_width=True, hide_index=True); cols = st.columns(min(len(bull_res['addon_a']), 6))
                for i, r in enumerate(bull_res['addon_a']):
                    with cols[i % 6]:
                        if st.button(f"🟡 加碼A {r['代號']}", key=f"bull_adda_{r['symbol']}"): bull_do_addon(r['symbol'], "回測站回", r['收盤價'], analysis_date_str); st.success(f"{r['代號']} 加碼A完成"); st.rerun()
        with bt4:
            if not bull_res['exit']: st.info("今日無出場訊號")
            else:
                st.dataframe(pd.DataFrame(bull_res['exit']), use_container_width=True, hide_index=True); cols = st.columns(min(len(bull_res['exit']), 6))
                for i, r in enumerate(bull_res['exit']):
                    with cols[i % 6]:
                        if st.button(f"🚪 出場 {r['代號']}", key=f"bull_exit_{r['symbol']}"): bull_remove_pos(r['symbol']); st.success(f"{r['代號']} 已出場"); st.rerun()
        with bt5:
            if len(bull_pos) == 0: st.info("目前無持倉")
            else:
                st.dataframe(bull_pos, use_container_width=True, hide_index=True); st.markdown("**手動出場：**"); bcols = st.columns(min(len(bull_pos), 8))
                for i, (_, row) in enumerate(bull_pos.iterrows()):
                    with bcols[i % 8]:
                        if st.button(f"🚪 {str(row['symbol']).replace('.TW','').replace('.TWO','')}", key=f"bull_m_exit_{row['symbol']}"): bull_remove_pos(row['symbol']); st.rerun()
                st.markdown("---")
                st.download_button("⬇️ 匯出持倉 CSV", data=bull_pos.to_csv(index=False, encoding='utf-8-sig'), file_name=f"BULL_positions_{analysis_date_str}.csv", mime="text/csv")
                if uploaded := st.file_uploader("📂 匯入持倉 CSV", type="csv", key="bull_upload"):
                    try: st.session_state['bull_positions'] = pd.read_csv(uploaded, encoding='utf-8-sig'); st.success(f"✅ 已匯入"); st.rerun()
                    except Exception as e: st.error(f"匯入失敗：{e}")

with tab4:
    st.header("⚡ 隔日沖雷達")
    if results:
        if day_list := results['day_candidates']:
            df_day = pd.DataFrame(day_list); df_day['sort_val'] = df_day['距離高點'].str.rstrip('%').astype(float)
            display_full_table(df_day.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val']))
        else: st.info("今日無符合隔日沖策略之標的。")
    else: st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab5:
    st.header("📊 個股 K 線診斷")
    col_in, col_btn = st.columns([3, 1])
    with col_in: stock_input = st.text_input("輸入代號", value="2330")
    with col_btn: diag_btn = st.button("診斷")
    run_bt = st.checkbox("同時執行回測", value=False, help="勾選後診斷時會一併執行 SniperStrategy 回測，需要較長時間")

    if diag_btn:
        with st.spinner("載入資料中..."): df, symbol = run_diagnosis(stock_input, analysis_date_str, params)
        if df is not None:
            sniper_info = next((item for item in (results.get('sniper_triggered', []) + results.get('sniper_watching', []) + results.get('sniper_setup', [])) if item.get('代號') == stock_input), None) if results else None
            if sniper_info: st.success(f"✅ 此股命中策略：{sniper_info.get('狀態', '')}　|　訊號日：{sniper_info.get('訊號日', sniper_info.get('_setup_date', 'N/A'))}")
            else: st.info("ℹ️ 此股在最近一次掃描中未命中任何策略訊號（或尚未執行掃描）")
            st.plotly_chart(plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info), use_container_width=True)
            if run_bt:
                st.markdown("---"); st.markdown("### 📈 SniperStrategy 回測結果"); st.caption("使用當前側邊欄參數 | 初始資金 NT$100,000 | 手續費 0.1425% x2")
                with st.spinner("回測運算中..."): run_backtest_ui(df, stock_input, params)
        else: st.error("查無資料，請確認代號是否正確")

with tab6:
    st.header("🔧 系統診斷"); st.caption("確認各資料來源與套件運作狀態")
    if st.button("🔄 執行診斷", type="primary"): st.session_state["diag_done"] = True

    if not st.session_state.get("diag_done", False): st.info("點擊「執行診斷」開始檢查")
    else:
        st.markdown("---")
        
        # 1. 永豐金 Shioaji 狀態
        st.subheader("1. 永豐金 Shioaji API 狀態")
        col_a, col_b = st.columns([3, 1])
        with col_a: st.caption("強制送出 API 查詢快照，驗證連線狀況")
        with col_b: st.button("🔁 強制測試 Shioaji", key="force_sj")
        
        with st.spinner("檢查 Shioaji 連線與快照抓取..."):
            try:
                # 拿取 2330 與 2317 測試
                contracts = [api.Contracts.Stocks["2330"], api.Contracts.Stocks["2317"]]
                snapshots = api.snapshots(contracts)
                
                if snapshots:
                    st.success("✅ Shioaji API 連線與資料獲取皆正常！")
                    rows_diag = []
                    for snap in snapshots:
                        c_p = snap.close if snap.close > 0 else snap.previous_close
                        rows_diag.append({
                            "代號": snap.code,
                            "最新價": c_p,
                            "開盤價": snap.open,
                            "最高價": snap.high,
                            "總成交量(張)": int(snap.total_volume),
                            "資料時間": str(pd.Timestamp(snap.ts))
                        })
                    st.dataframe(pd.DataFrame(rows_diag), use_container_width=True, hide_index=True)
                else:
                    st.error("❌ 登入成功但無法取得快照資料")
            except Exception as e:
                st.error(f"❌ Shioaji 發生錯誤: {e}")

        # 2. Parquet 歷史資料
        st.subheader("2. 歷史資料 data/history.parquet")
        if os.path.exists(PARQUET_PATH):
            try:
                df_check = pd.read_parquet(PARQUET_PATH); df_check["date"] = pd.to_datetime(df_check["date"])
                st.success(f"✅ Parquet 正常 | {df_check['code'].nunique()} 支  最新: {df_check['date'].max().strftime('%Y-%m-%d')}  大小: {round(os.path.getsize(PARQUET_PATH) / 1024 / 1024, 1)} MB")
            except Exception as e: st.error(f"❌ Parquet 讀取失敗: {e}")
        else: st.error(f"❌ 找不到 {PARQUET_PATH}，請先確認 GitHub Action 已執行")