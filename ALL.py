# ==========================================
# 強勢股戰情室 V20 (原汁原味 + 創高拉回引擎)
# ==========================================
import streamlit as st
import os
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import requests
import time
import importlib
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 📋 日誌與外部套件設定
# ==========================================
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

try:
    import shioaji as sj
except ImportError:
    st.error("❌ 缺少 `shioaji` 套件，請安裝")
    st.stop()

SECTOR_DB = {}
try:
    import sector_data
    importlib.reload(sector_data)
    if hasattr(sector_data, 'CUSTOM_SECTOR_MAP'):
        SECTOR_DB = {str(k).strip(): v for k, v in sector_data.CUSTOM_SECTOR_MAP.items()}
except: pass
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="強勢股戰情室 V20+", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

import pytz as _pytz_main
_TW_TZ = _pytz_main.timezone("Asia/Taipei")
def _now_tw():
    import datetime as _dtt, pytz as _ptz
    return _dtt.datetime.now(_ptz.timezone("Asia/Taipei"))

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError: GSPREAD_AVAILABLE = False

st.markdown("""<style>.stDataFrame {font-size: 1.1rem;} [data-testid="stMetricValue"] {font-size: 1.5rem;}</style>""", unsafe_allow_html=True)

# ==========================================
# 🚀 永豐金 Shioaji API 初始化
# ==========================================
@st.cache_resource
def get_shioaji_api():
    if "shioaji" not in st.secrets:
        st.error("❌ 找不到永豐金 API 金鑰！")
        st.stop()
    is_sim = st.secrets["shioaji"].get("simulation", True)
    api = sj.Shioaji(simulation=is_sim)
    try:
        api.login(api_key=st.secrets["shioaji"]["api_key"], secret_key=st.secrets["shioaji"]["secret_key"], fetch_contract=True)
    except Exception as e:
        st.error(f"🔴 永豐金 API 登入失敗: {e}"); st.stop()
    return api

api = get_shioaji_api()

# ==========================================
# ☁️ 雲端資料庫 URL 設定
# ==========================================
# 👇 法人籌碼資料庫 URL
HF_CHIP_URL = "https://huggingface.co/datasets/4340P/institutional_investors_parquet_by_stock/resolve/main/all_institutional_investors_2020_2026.parquet"

# 👇 歷史報價資料庫 URL (⚠️ 請把下方替換成你新開的 Repo 名稱)
HF_HISTORY_URL = "https://huggingface.co/datasets/4340P/history/resolve/main/history.parquet"

# ==========================================
# 🧠 策略核心邏輯類別 (Backtesting)
# ==========================================
def SMA(array, n): return pd.Series(array).rolling(window=n).mean()

class SniperStrategy(Strategy):
    ma_trend_period = 60; ma_long_period = 240; ma_base_exit = 20; ma_fast_exit = 10; vol_ma_period = 5; big_candle_pct = 0.05; min_volume_shares = 2000000; lookback_window = 10; use_year_line = True; defense_buffer = 0.01
    def init(self):
        close = pd.Series(self.data.Close); volume = pd.Series(self.data.Volume)
        self.ma_trend = self.I(SMA, close, self.ma_trend_period); self.ma_base = self.I(SMA, close, self.ma_base_exit); self.ma_fast = self.I(SMA, close, self.ma_fast_exit); self.vol_ma = self.I(SMA, volume, self.vol_ma_period)
        if self.use_year_line: self.ma_long = self.I(SMA, close, self.ma_long_period)
        self.setup_active = False; self.setup_bar_index = 0; self.setup_low_price = 0; self.defense_price = 0
    def next(self):
        price = self.data.Close[-1]; prev_high = self.data.High[-2]
        if self.position:
            if price < self.defense_price: self.position.close(); return
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
# 🛠️ 輔助函式與資料庫
# ==========================================
def get_detailed_sector(code, standard_group=None, custom_db=None):
    code_str = str(code).strip()
    if custom_db and code_str in custom_db: return str(custom_db[code_str])
    if standard_group and str(standard_group) not in ['nan', 'None', '', 'NaN']: return str(standard_group)
    return "其他"

@st.cache_data(ttl=3600*12, show_spinner=False)
def load_cloud_chip_data(url):
    """載入雲端法人籌碼資料"""
    url = url.replace("/blob/main/", "/resolve/main/")
    temp_file = "temp_chip_data.parquet"
    try:
        resp = requests.get(url, timeout=45) 
        if resp.status_code != 200: return pd.DataFrame()
        with open(temp_file, "wb") as f: f.write(resp.content)
        df_chip = pd.read_parquet(temp_file)
        if not df_chip.empty:
            df_chip.columns = [str(c).lower() for c in df_chip.columns]
            if not all(col in df_chip.columns for col in ['date', 'stock_id', 'name', 'buy', 'sell']): return df_chip
            df_chip['stock_id'] = df_chip['stock_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_chip['name'] = df_chip['name'].astype(str).str.strip()
            df_chip = df_chip[df_chip['name'].str.contains('Foreign_Investor|外資', case=False, na=False)].copy()
            df_chip['date'] = pd.to_datetime(df_chip['date']).dt.strftime('%Y-%m-%d')
            df_chip['net_buy'] = ((df_chip['buy'] - df_chip['sell']) / 1000).astype('float32')
            df_chip = df_chip[['date', 'stock_id', 'net_buy']]
        return df_chip
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*12, show_spinner=False)
def load_cloud_history_data(url):
    """載入雲端歷史報價資料"""
    url = url.replace("/blob/main/", "/resolve/main/")
    try:
        df_all = pd.read_parquet(url)
        df_all["date"] = pd.to_datetime(df_all["date"])
        data_store = {}
        for code, grp in df_all.groupby("code"):
            grp = grp.sort_values("date").set_index("date")[["Open","High","Low","Close","Volume"]].copy()
            grp.index.name = None
            data_store[str(code).strip()] = grp 
        last_date = df_all["date"].max().strftime("%Y-%m-%d")
        return data_store, f"✅ 雲端歷史資料載入完成：{len(data_store)} 支 (最新至 {last_date})"
    except Exception as e:
        return None, f"❌ 讀取雲端歷史資料庫失敗: {e}"

def evaluate_chip_filter(code_str, analysis_date_str, df_kline, chip_params, df_chip_main):
    chip_summary = "無籌碼資料"
    passed_chip_filter = True 
    if chip_params.get('c_use_filter', True) and df_chip_main is not None and not df_chip_main.empty:
        target_date_ts = pd.Timestamp(analysis_date_str)
        lookback_days = chip_params.get('c_lookback_days', 30)
        threshold_shares = chip_params.get('c_threshold_shares', 10000)
        valid_dates = df_kline[df_kline.index <= target_date_ts].tail(lookback_days).index.strftime('%Y-%m-%d').tolist()
        if valid_dates:
            mask = (df_chip_main['stock_id'] == code_str) & (df_chip_main['date'].isin(valid_dates))
            recent_foreign_chip = df_chip_main[mask]
            if not recent_foreign_chip.empty:
                total_foreign_buy = recent_foreign_chip['net_buy'].sum()
                chip_summary = f"近{lookback_days}日外資: {total_foreign_buy:+.0f}張"
                if total_foreign_buy < threshold_shares: passed_chip_filter = False
            else:
                chip_summary = f"近{lookback_days}日無外資進出"
                passed_chip_filter = False
    else:
        if df_chip_main is not None and not df_chip_main.empty:
             day_chip = df_chip_main[(df_chip_main['date'] == analysis_date_str) & (df_chip_main['stock_id'] == code_str)]
             if not day_chip.empty: chip_summary = f"外資當日: {day_chip['net_buy'].sum():+.0f}張"
    return passed_chip_filter, chip_summary

@st.cache_data(ttl=3600*12)
def get_stock_info_map(_api_instance):
    stock_map = {}
    valid_codes = set()
    try:
        df = pd.read_parquet(HF_HISTORY_URL, columns=["code"])
        valid_codes.update(df["code"].astype(str).unique())
    except: pass
    if len(valid_codes) < 1000:
        valid_codes.update([str(i).zfill(4) for i in range(1101, 10000)])
        valid_codes.update([f"00{str(i).zfill(3)}" for i in range(1, 1000)])

    for c_str in valid_codes:
        c_str = c_str.strip()
        if len(c_str) < 4: continue
        try:
            contract = _api_instance.Contracts.Stocks[c_str]
            if contract:
                stock_map[c_str] = {
                    'name': f"{c_str} {contract.name}", 
                    'symbol': f"{c_str}{'.TW' if contract.exchange == 'TSE' else '.TWO'}", 
                    'short_name': contract.name, 
                    'group': getattr(contract, 'category', '其他')
                }
        except: pass
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
            contract = _api_instance.Contracts.Stocks[str(code).strip()]
            if contract:
                if snaps := _api_instance.snapshots([contract]):
                    snap = snaps[0]
                    c_p = snap.close if snap.close > 0 else (snap.close - getattr(snap, 'change_price', 0))
                    if c_p > 0:
                        new_row = pd.Series({'Open': snap.open if snap.open > 0 else c_p, 'High': snap.high if snap.high > 0 else c_p, 'Low': snap.low if snap.low > 0 else c_p, 'Close': c_p, 'Volume': snap.total_volume * 1000}, name=pd.Timestamp(today_str))
                        df = pd.concat([df, new_row.to_frame().T])
        except: pass
    return df

# ==========================================
# 🐂 BULL_v7 模組
# ==========================================
BULL_SHEET_COLS = ["symbol","name","進場日期","進場價","上次加碼價","持倉最高價","加碼次數","加碼紀錄"]
BULL_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
def bull_use_gsheet() -> bool: return GSPREAD_AVAILABLE and ("gcp_service_account" in st.secrets and "sheets" in st.secrets)
def bull_get_ws():
    if not GSPREAD_AVAILABLE: return None
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=BULL_SCOPES)
        sh = gspread.authorize(creds).open_by_key(st.secrets["sheets"]["sheet_id"])
        try: return sh.worksheet("bull_positions")
        except: ws = sh.add_worksheet(title="bull_positions", rows=1000, cols=20); ws.append_row(BULL_SHEET_COLS); return ws
    except: return None
def bull_gs_load() -> pd.DataFrame:
    ws = bull_get_ws()
    if ws is None: return pd.DataFrame(columns=BULL_SHEET_COLS)
    try:
        if not (data := ws.get_all_records()): return pd.DataFrame(columns=BULL_SHEET_COLS)
        df = pd.DataFrame(data)
        for col in ["加碼次數", "進場價", "上次加碼價", "持倉最高價"]: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df[BULL_SHEET_COLS]
    except: return pd.DataFrame(columns=BULL_SHEET_COLS)
def bull_gs_save(df: pd.DataFrame):
    if ws := bull_get_ws():
        try: ws.clear(); ws.append_row(BULL_SHEET_COLS); ws.append_rows(df[BULL_SHEET_COLS].fillna("").values.tolist(), value_input_option="USER_ENTERED")
        except: pass
BULL_PARAMS = {"boll_period": 15, "boll_std": 2.1, "squeeze_n": 15, "squeeze_lookback": 5, "vol_ma_days": 5, "vol_ratio": 1.5, "sma_trend_days": 3, "min_vol_shares": 1_000_000, "vol_ma20_days": 20, "vol_heavy_days": 5, "vol_shrink_days": 15, "init_position": 0.5, "add_position": 0.5, "addon_b_profit": 0.10}
def bull_calc_indicators(df, params):
    df = df.copy(); df.columns = [c.capitalize() for c in df.columns]; df = df[[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]].copy()
    close = df["Close"]; volume = df["Volume"]; bp = params["boll_period"]
    df["SMA"] = close.rolling(bp).mean(); df["Std"] = close.rolling(bp).std()
    df["Upper"] = df["SMA"] + df["Std"] * params["boll_std"]; df["Lower"] = df["SMA"] - df["Std"] * params["boll_std"]
    df["Bandwidth"] = df["Upper"] - df["Lower"]; df["Vol_MA"] = volume.rolling(params["vol_ma_days"]).mean()
    df["Is_Squeeze"] = df["Bandwidth"] == df["Bandwidth"].rolling(params["squeeze_n"]).min(); df["Squeeze_Recent"] = df["Is_Squeeze"].shift(1).rolling(params["squeeze_lookback"]).max() == 1
    df["SMA_Up"] = df["SMA"] > df["SMA"].shift(params["sma_trend_days"]); df["Vol_MA20"] = volume.rolling(params["vol_ma20_days"]).mean()
    df["Vol_Heavy"] = volume.rolling(params["vol_heavy_days"]).mean(); df["Vol_Shrink"] = volume.rolling(params["vol_shrink_days"]).mean()
    return df

def bull_scan_one(code, info, df_raw, params, pos_map, scan_date_ts, df_chip_main, chip_params):
    res = {"entry": None, "addon_a": None, "addon_b": None, "exit": None, "high_update": None}
    try:
        passed_chip, chip_summary = evaluate_chip_filter(str(code).strip(), scan_date_ts.strftime('%Y-%m-%d'), df_raw, chip_params, df_chip_main)
        df = bull_calc_indicators(df_raw, params); df = df[df.index <= scan_date_ts]
        if len(df) < 50: return res
        sym = info["symbol"]; name = info["short_name"]; r = df.iloc[-1]; r1 = df.iloc[-2]
        c = float(r["Close"]); sma_i = float(r["SMA"]); lo_i = float(r["Lower"]); l_i = float(r["Low"]); vol_i = float(r["Volume"]); vheavy = float(r["Vol_Heavy"]) if not pd.isna(r["Vol_Heavy"]) else 0
        if sym not in pos_map:
            if passed_chip:
                if not (pd.isna(r["Squeeze_Recent"]) or pd.isna(r["Vol_MA20"])):
                    if bool(r["Squeeze_Recent"]) and bool(r["SMA_Up"]) and (c > float(r["Upper"])) and (vol_i > float(r["Vol_MA"]) * params["vol_ratio"]) and (float(r["Vol_MA20"]) > params["min_vol_shares"]):
                        res["entry"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "上軌": round(float(r["Upper"]), 2), "15MA": round(sma_i, 2), "成交量": int(vol_i), "5MA量": int(r["Vol_MA"]) if not pd.isna(r["Vol_MA"]) else 0, "法人籌碼": chip_summary}
        else:
            pos = pos_map[sym]; ep = float(pos["進場價"]); lap = float(pos["上次加碼價"]); pp = float(pos["持倉最高價"])
            if c > pp: res["high_update"] = (sym, round(c, 2))
            heavy_break = (c < sma_i) and (vol_i >= vheavy) and vheavy > 0
            if heavy_break or (l_i <= lo_i):
                res["exit"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "15MA": round(sma_i, 2), "進場價": round(ep, 2), "損益%": round((c - ep) / ep * 100, 2), "加碼次數": int(pos.get("加碼次數", 0)), "出場原因": "出量跌破15MA" if heavy_break else "最低碰下軌", "法人籌碼": chip_summary}
                return res
            vshrink = float(r1["Vol_Shrink"]) if not pd.isna(r1["Vol_Shrink"]) else 0
            if (float(r1["Close"]) < float(r1["SMA"])) and (float(r1["Volume"]) < vshrink if vshrink > 0 else False) and (c >= sma_i): res["addon_a"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "15MA": round(sma_i, 2), "加碼次數": int(pos.get("加碼次數", 0)), "加碼類型": "A 回測站回", "法人籌碼": chip_summary}
            profit = (c - lap) / lap if lap > 0 else 0
            if c > pp and profit >= params["addon_b_profit"]: res["addon_b"] = {"代號": code, "名稱": name, "symbol": sym, "收盤價": round(c, 2), "持倉最高價": round(pp, 2), "距上次加碼": f"+{profit*100:.1f}%", "加碼次數": int(pos.get("加碼次數", 0)), "加碼類型": "B 突破新高", "法人籌碼": chip_summary}
    except: pass
    return res

def bull_run_scan(stock_map, all_data, scan_date_str, bull_positions, bull_params, max_workers=16, status_text=None, progress_bar=None, df_chip_main=None, chip_params=None):
    scan_date_ts = pd.Timestamp(scan_date_str); pos_map = {row["symbol"]: row for _, row in bull_positions.iterrows()} if len(bull_positions) > 0 else {}
    res = {"entry": [], "addon_a": [], "addon_b": [], "exit": [], "high_updates": []}
    valid_codes = {str(c): df for c, df in all_data.items() if str(c) in stock_map}
    total = len(valid_codes); done = 0
    if status_text: status_text.text(f"🐂 BULL_v7 策略運算中... (共 {total} 支)")
    if progress_bar: progress_bar.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(bull_scan_one, code, stock_map[code], df, bull_params, pos_map, scan_date_ts, df_chip_main, chip_params): code for code, df in valid_codes.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 100 == 0 or done == total:
                if progress_bar: progress_bar.progress(done / max(total, 1), text=f"🐂 BULL_v7 策略運算 {done}/{total}...")
            r = future.result()
            for k in res.keys(): 
                if r.get(k): res[k].append(r[k])
    if status_text: status_text.success(f"✅ 全部掃描完成｜進場 {len(res['entry'])} | 出場 {len(res['exit'])}")
    if progress_bar: progress_bar.empty()
    return res

def fetch_data_batch(stock_map, period="300d", chunk_size=150):
    all_symbols = [info['symbol'] for info in stock_map.values()]
    data_store = {}; symbol_to_code = {v['symbol']: str(k) for k, v in stock_map.items()}
    total_chunks = (len(all_symbols) // chunk_size) + 1
    progress_text = st.empty(); bar = st.progress(0)
    for i in range(0, len(all_symbols), chunk_size):
        chunk = all_symbols[i:i + chunk_size]; chunk_idx = (i // chunk_size) + 1
        progress_text.text(f"📥 下載歷史資料... (批次 {chunk_idx}/{total_chunks})"); bar.progress(chunk_idx / total_chunks)
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
        except: continue
    bar.empty()
    return data_store

# ==========================================
# ⚡ 全時段即時報價
# ==========================================
def fetch_realtime_batch(codes_list, _api_instance, status_text=None):
    realtime_data = {}; _txt = status_text if status_text else st.sidebar.empty(); rt_bar = st.sidebar.progress(0)
    try:
        contracts = []
        for c in codes_list:
            c_str = str(c).strip()
            try:
                if contract := _api_instance.Contracts.Stocks[c_str]: contracts.append(contract)
            except: pass
        total_c = len(contracts)
        if total_c == 0: _txt.text("❌ 沒有需要抓取的合約"); return realtime_data
        _txt.text(f"⚡ 快照擷取中 (共 {total_c} 支)...")
        chunk_size = 50 
        for i in range(0, total_c, chunk_size):
            chunk = contracts[i:i + chunk_size]
            for _ in range(2):
                try:
                    if snaps := _api_instance.snapshots(chunk):
                        for snap in snaps:
                            c_p = snap.close if snap.close > 0 else (snap.close - getattr(snap, 'change_price', 0))
                            realtime_data[str(snap.code)] = {"latest_trade_price": str(c_p), "open": str(snap.open if snap.open > 0 else c_p), "high": str(snap.high if snap.high > 0 else c_p), "low": str(snap.low if snap.low > 0 else c_p), "accumulate_trade_volume": str(int(snap.total_volume))}
                        break 
                    else: time.sleep(0.5)
                except: time.sleep(0.5)
            time.sleep(0.1); rt_bar.progress(min(80, int((i + chunk_size) / total_c * 80)))
        missing = [c for c in contracts if str(c.code) not in realtime_data]
        if missing:
            _txt.text(f"⚠️ 針對遺漏 {len(missing)} 支進行補考...")
            retry_chunk = 20
            for i in range(0, len(missing), retry_chunk):
                chunk = missing[i:i + retry_chunk]
                for _ in range(3):
                    try:
                        if snaps := _api_instance.snapshots(chunk):
                            for snap in snaps:
                                c_p = snap.close if snap.close > 0 else (snap.close - getattr(snap, 'change_price', 0))
                                realtime_data[str(snap.code)] = {"latest_trade_price": str(c_p), "open": str(snap.open if snap.open > 0 else c_p), "high": str(snap.high if snap.high > 0 else c_p), "low": str(snap.low if snap.low > 0 else c_p), "accumulate_trade_volume": str(int(snap.total_volume))}
                            break
                        else: time.sleep(1)
                    except: time.sleep(1)
                time.sleep(0.3); rt_bar.progress(80 + min(20, int((i + retry_chunk) / len(missing) * 20)))
        rt_bar.progress(100); _txt.text(f"✅ 快照更新完成｜成功取得 {len(realtime_data)} 支")
    except Exception as e: logger.error(f"Shioaji 錯誤: {e}"); _txt.text(f"❌ 永豐連線異常: {e}"); rt_bar.progress(100)
    return realtime_data

# ==========================================
# 📈 策略模組
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_status(analysis_date_str, _api_instance=None):
    try:
        df = yf.Ticker("^TWII").history(period="1y")
        if df.empty or df.index.tz is not None: df.index = df.index.tz_localize(None) if not df.empty else df.index
        if df.empty: return {'strong': True, 'score': 5, 'label': '無大盤資料', 'close': '-', 'ma20': '-', 'ma60': '-'}
        today_str = _now_tw().strftime('%Y-%m-%d'); df = df[~df.index.duplicated(keep='last')].copy()
        if _api_instance and analysis_date_str == today_str:
            try:
                if snap := _api_instance.snapshots([_api_instance.Contracts.Indices.TSE.TSE001]):
                    rt_c = snap[0].close if snap[0].close > 0 else (snap[0].close - getattr(snap[0], 'change_price', 0)); today_ts = pd.Timestamp(today_str)
                    if today_ts in df.index: df.loc[today_ts, 'Close'] = rt_c
                    else: df = pd.concat([df, pd.Series({'Close': rt_c}, name=today_ts).to_frame().T])
            except: pass
        latest_idx = df.index.get_loc(pd.Timestamp(analysis_date_str)) if pd.Timestamp(analysis_date_str) in df.index else -1
        close = df['Close'].ffill(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
        c = close.iloc[latest_idx]; m20 = ma20.iloc[latest_idx]; m60 = ma60.iloc[latest_idx]
        score = sum([4 if c > m20 else 0, 3 if c > m60 else 0, 3 if m20 > m60 else 0])
        return {'strong': score >= 7, 'score': score, 'label': "🟢 大盤強勢" if score >= 7 else ("🟡 大盤偏弱" if score >= 4 else "🔴 大盤弱勢"), 'close': round(c, 0), 'ma20': round(m20, 0), 'ma60': round(m60, 0)}
    except: return {'strong': True, 'score': 5, 'label': '大盤資料異常', 'close': '-', 'ma20': '-', 'ma60': '-'}

def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None, market_score=5, df_chip_main=None):
    try:
        code_str = str(code).strip()

        df = pre_loaded_df.copy() if pre_loaded_df is not None else get_stock_data_with_realtime(code, info['symbol'], analysis_date_str, api)
        if df is None or df.empty or len(df) < 200: return "資料長度不足 (<200天)"
        df = df.loc[~df.index.duplicated(keep='last')].copy()
        if pd.Timestamp(analysis_date_str) not in df.index: return f"無 {analysis_date_str} 交易資料"
        
        passed_chip_filter, chip_summary = evaluate_chip_filter(code_str, analysis_date_str, df, params, df_chip_main)
        if params.get('c_use_filter', True) and not passed_chip_filter: return {'sniper': None, 'day': None}
        
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
        stock_name = info['short_name']; sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db); result_sniper = None; result_day = None
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
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🚀 強勢突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}", "法人籌碼": chip_summary, "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low}) if is_breakout else ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "💪 強勢整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "法人籌碼": chip_summary, "長紅高": f"{s_high:.2f}", "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
                    else:
                        if (is_breakout and close.iloc[idx-1] <= (s_high * 1.02)) or is_gap_breakout:
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🚀 N字跳空" if is_gap_breakout else "🎯 N字突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}", "法人籌碼": chip_summary, "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
                        else: result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "📉 回檔整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "法人籌碼": chip_summary, "長紅高": f"{s_high:.2f}", "sort_pct": daily_pct, "_setup_date": s_date, "_defense": defense_price, "_signal_high": s_high, "_signal_low": s_low})
            elif is_setup:
                result_sniper = ("new_setup", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🔥 剛起漲", "漲幅": f"{daily_pct:+.2f}%", "法人籌碼": chip_summary, "sort_pct": daily_pct, "_setup_date": df.index[idx].strftime('%Y-%m-%d'), "_defense": defense_price, "_signal_high": high.iloc[idx], "_signal_low": low.iloc[idx]})
        d_period = params['d_period']; d_threshold = params['d_threshold']; d_min_vol = params['d_min_vol']; d_min_pct = params['d_min_pct']
        d_close = close.iloc[idx]; d_open = op.iloc[idx]; d_high = high.iloc[idx]; d_volume = volume.iloc[idx]; d_prev_close = close.iloc[idx-1]
        if idx >= d_period and (d_close > d_open) and ((d_high - d_close) / d_close < 0.01) and (d_min_pct/100 < (pct_chg_val := (d_close - d_prev_close) / d_prev_close) < 0.095) and ((d_volume / 1000) > d_min_vol) and (d_close >= ((prev_period_high := high.iloc[idx-d_period : idx].max()) * (1 - (d_threshold / 100)))) and (d_high <= prev_period_high):
            result_day = {"代號": code, "名稱": stock_name, "收盤": f"{d_close:.2f}", "產業": sector_name, "漲幅": f"{(pct_chg_val*100):.2f}%", "成交量": int(d_volume/1000), "前波高點": f"{prev_period_high:.2f}", "法人籌碼": chip_summary, "距離高點": f"{(d_close - prev_period_high) / prev_period_high * 100:+.2f}%", "狀態": "⚡ 蓄勢待發"}
        
        # ==================================
        # 🏆 創高拉回佈局策略 (Dip-Buyer)
        # ==================================
        result_dip = None
        high_250 = high.rolling(250).max()
        peak_60 = high.iloc[max(0, idx-60):idx+1].max()
        
        ma240 = close.rolling(window=240).mean()
        
        trend_pass = (pd.notna(ma240.iloc[idx]) and ma240.iloc[idx] > ma240.iloc[idx-5]) and (c_today > ma240.iloc[idx])
        high_pass = (peak_60 >= high_250.iloc[idx]) and pd.notna(peak_60)
        
        if trend_pass and high_pass:
            pullback_pct = (c_today - peak_60) / peak_60 if peak_60 > 0 else 0
            # 使用 sidebar 參數
            db_p_min = params.get('db_p_min', -25) / 100
            db_p_max = params.get('db_p_max', -10) / 100
            pullback_pass = (db_p_min <= pullback_pct <= db_p_max)
            
            vol_ma20 = volume.rolling(window=20).mean()
            vol_shrink = v_today < vol_ma20.iloc[idx]
            
            trigger_pass = (c_today > ma10.iloc[idx]) and (close.iloc[idx-1] <= ma10.iloc[idx-1])
            
            if pullback_pass and vol_shrink and trigger_pass:
                result_dip = {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🏆 創高拉回止跌", "拉回幅度": f"{pullback_pct*100:.2f}%", "一年高點": f"{peak_60:.2f}", "法人籌碼": chip_summary}

        return {'sniper': result_sniper, 'day': result_day, 'dip': result_dip}
    except Exception as e: return f"執行錯誤: {e}"

def display_full_table(df):
    if df is not None and not df.empty:
        st.dataframe(df[[c for c in df.columns if not c.startswith('_')]], hide_index=True, use_container_width=True, height=(len(df) * 35) + 38)
    else: st.info("無")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室 V20+")
st.sidebar.caption("純永豐金 API 企業級 24H 連線版")
st.sidebar.success("☁️ 歷史資料：Hugging Face 雲端連線中")
st.sidebar.success("🟢 Shioaji 全時段即時報價中")

analysis_date_input = st.sidebar.date_input("分析基準日", _now_tw().date())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')
start_scan = st.sidebar.button("🚀 開始全域掃描 (極速版)", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)
st.sidebar.divider()

# 👇 法人過濾
with st.sidebar.expander("🛡️ 法人籌碼濾網", expanded=False):
    c_use_filter = st.checkbox("啟用籌碼過濾 (僅篩選外資買超)", value=True)
    st.caption("啟用後，未達標股票將不會顯示在結果中")
    c_lookback_days = st.number_input("最近 N 天", min_value=1, max_value=250, value=30)
    c_threshold_shares = st.number_input("外資買超大於 (張)", min_value=0, max_value=1000000, value=10000, step=1000)

# 👇 創高拉回策略參數
with st.sidebar.expander("🏆 創高拉回策略 (波段)", expanded=False):
    st.caption("過濾創一年高後，量縮拉回之強勢股")
    db_p_min = st.slider("最大拉回幅度(%)", -40, -10, -25)
    db_p_max = st.slider("最小拉回幅度(%)", -30, -5, -10)

with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=False):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60); s_use_year = st.checkbox("啟用年線", value=True); s_big_candle = st.slider("長紅漲幅(%)", 2.0, 10.0, 5.0) / 100; s_min_vol = st.number_input("波段最小量(張)", value=1000) * 1000; s_setup_lookback = st.slider("回溯天數", 5, 30, 25); s_vol_ma_days = st.slider("量能MA", 5, 20, 20); s_vol_ratio = st.slider("量能門檻", 0.5, 1.5, 0.7); s_pullback_max = st.slider("回測深度(%)", 20, 70, 50)
with st.sidebar.expander("🐂 BULL_v7 滾雪球參數", expanded=False):
    b_boll_period = st.number_input("布林週期", value=15, key="b_bp"); b_boll_std = st.number_input("布林標準差", value=2.1, key="b_bs"); b_squeeze_n = st.number_input("壓縮天數", value=15, key="b_sn"); b_squeeze_lb = st.number_input("回溯", value=5, key="b_sl"); b_vol_ratio = st.number_input("爆量倍數", value=1.5, key="b_vr"); b_sma_days = st.number_input("趨勢天數", value=3, key="b_sd"); b_min_vol = st.number_input("均量門檻(張)", value=1000, key="b_mv") * 1000; b_addon_b_pct = st.number_input("加碼B(%)", value=10, key="b_ab") / 100
with st.sidebar.expander("⚡ 隔日沖雷達參數", expanded=False):
    d_period = st.slider("追蹤波段(N)", 10, 120, 60); d_threshold = st.slider("高點誤差(%)", 0.0, 5.0, 1.0); d_min_pct = st.slider("最低漲幅(%)", 3.0, 9.0, 5.0); d_min_vol = st.number_input("最小量(張)", value=1000, step=500)
max_workers_input = st.sidebar.slider("運算效能(執行緒)", 1, 32, 16)

# 組合所有參數
params = {'c_use_filter': c_use_filter, 'c_lookback_days': c_lookback_days, 'c_threshold_shares': c_threshold_shares, 's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 's_big_candle': s_big_candle, 's_min_vol': s_min_vol, 's_setup_lookback': s_setup_lookback, 's_vol_ma_days': s_vol_ma_days, 's_vol_ratio': s_vol_ratio, 's_pullback_max': s_pullback_max, 'd_period': d_period, 'd_threshold': d_threshold, 'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol, 'db_p_min': db_p_min, 'db_p_max': db_p_max}
bull_params = {**BULL_PARAMS, "boll_period": int(b_boll_period), "boll_std": float(b_boll_std), "squeeze_n": int(b_squeeze_n), "squeeze_lookback": int(b_squeeze_lb), "vol_ratio": float(b_vol_ratio), "sma_trend_days": int(b_sma_days), "min_vol_shares": int(b_min_vol), "addon_b_profit": float(b_addon_b_pct)}
chip_params = {'c_use_filter': c_use_filter, 'c_lookback_days': c_lookback_days, 'c_threshold_shares': c_threshold_shares}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🟢 狙擊手波段", "🏆 創高拉回", "🐂 BULL滾雪球", "⚡ 隔日沖雷達", "📊 個股診斷", "🔧 系統診斷"])

for k in ['scan_results', 'bull_results', 'scan_params', 'scan_date', 'market_status']:
    if k not in st.session_state: st.session_state[k] = None
if 'bull_positions' not in st.session_state: st.session_state['bull_positions'] = pd.DataFrame(columns=BULL_SHEET_COLS)
if 'bull_gs_loaded' not in st.session_state: st.session_state['bull_gs_loaded'] = False
if not st.session_state['bull_gs_loaded'] and bull_use_gsheet():
    df_gs = bull_gs_load()
    if len(df_gs) > 0: st.session_state['bull_positions'] = df_gs
    st.session_state['bull_gs_loaded'] = True
def params_changed(): return st.session_state['scan_params'] is not None and (st.session_state['scan_params'] != params or st.session_state['scan_date'] != analysis_date_str)

# ==========================================
# 執行整合掃描
# ==========================================
if start_scan:
    stock_map = get_stock_info_map(api)
    status_text.text("📈 正在判斷大盤狀態...")
    market_status = fetch_market_status(analysis_date_str, api)
    st.session_state['market_status'] = market_status
    
    # 載入籌碼庫 (Hugging Face)
    df_chip_main = None
    if c_use_filter or True: 
        status_text.text("☁️ 正在載入雲端法人籌碼...")
        df_chip_main = load_cloud_chip_data(HF_CHIP_URL)
        if df_chip_main is not None and not df_chip_main.empty:
            st.sidebar.caption(f"✅ 籌碼庫載入完畢 (最新至: **{df_chip_main['date'].max()}**)")

    sniper_triggered = []; sniper_setup = []; sniper_watching = []; day_candidates = []; dip_candidates = []; failed_list = []
    
    # 載入歷史報價資料庫 (Hugging Face)
    status_text.text("☁️ 正在載入雲端歷史報價...")
    history_data_store, load_msg = load_cloud_history_data(HF_HISTORY_URL)

    if history_data_store is None:
        status_text.text("⚠️ 雲端歷史報價載入失敗，改用 yfinance 即時下載 (會較慢)...")
        history_data_store = fetch_data_batch(stock_map)
    else:
        st.sidebar.caption(load_msg)
        history_data_store = {str(k).strip(): v for k, v in history_data_store.items() if str(k).strip() in stock_map}

    st.session_state['all_data_cache'] = history_data_store
    st.session_state['stock_map_cache'] = stock_map

    today_str = _now_tw().strftime('%Y-%m-%d')
    realtime_map = {}
    
    if analysis_date_str == today_str:
        status_text.text("⚡ 正在請求永豐金即時報價快照...")
        realtime_map = fetch_realtime_batch(list(history_data_store.keys()), api, status_text=status_text)

    status_text.text("🧠 正在進行技術與籌碼綜合掃描...")
    progress_bar.progress(0)
    tasks_data = {}
    def safe_f(val, default=0.0):
        try: return float(str(val).replace(',', '')) if val not in ('-', '', None) else default
        except: return default

    for code, df in history_data_store.items():
        code_str = str(code).strip()
        df = df.loc[~df.index.duplicated(keep='first')].copy()
        if code_str in realtime_map and realtime_map[code_str].get('latest_trade_price', '-') != '-':
            try:
                rt = realtime_map[code_str]
                c_p = safe_f(rt.get('latest_trade_price')); o_p = safe_f(rt.get('open')) or c_p; h_p = safe_f(rt.get('high')) or c_p; l_p = safe_f(rt.get('low')) or c_p; v_p = safe_f(rt.get('accumulate_trade_volume')) * 1000
                if c_p > 0:
                    today_ts = pd.Timestamp(today_str); new_row = pd.Series({'Open': o_p, 'High': h_p, 'Low': l_p, 'Close': c_p, 'Volume': v_p}, name=today_ts)
                    if today_ts in df.index: df.loc[today_ts] = new_row
                    else: df = pd.concat([df, new_row.to_frame().T])
            except: pass
        tasks_data[code_str] = df

    st.session_state['all_data_cache'] = tasks_data
    total = len(tasks_data); done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df, market_status.get('score', 5), df_chip_main): code for code, df in tasks_data.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total: progress_bar.progress(done / max(total, 1)); status_text.text(f"綜合掃描中: {done}/{total}")
            res = future.result()
            if isinstance(res, dict):
                if res.get('sniper'):
                    typ, data = res['sniper']
                    if typ == "triggered": sniper_triggered.append(data)
                    elif typ == "new_setup": sniper_setup.append(data)
                    elif typ == "watching": sniper_watching.append(data)
                if res.get('day'): day_candidates.append(res['day'])
                if res.get('dip'): dip_candidates.append(res['dip'])
            else:
                current_code = futures[future]
                failed_list.append(f"{current_code}: {res}")

    st.session_state['scan_results'] = {'sniper_triggered': sniper_triggered, 'sniper_setup': sniper_setup, 'sniper_watching': sniper_watching, 'day_candidates': day_candidates, 'dip_candidates': dip_candidates, 'failed_list': failed_list}
    progress_bar.progress(1.0)
    status_text.success("✅ 全域掃描（含籌碼濾網）已完成！")
    st.session_state['scan_params'] = params.copy()
    st.session_state['scan_date'] = analysis_date_str
    
    try:
        st.session_state['bull_results'] = bull_run_scan(stock_map=st.session_state['stock_map_cache'], all_data=st.session_state['all_data_cache'], scan_date_str=analysis_date_str, bull_positions=st.session_state['bull_positions'], bull_params=bull_params, max_workers=max_workers_input, status_text=status_text, progress_bar=progress_bar, df_chip_main=df_chip_main, chip_params=chip_params)
    except Exception as e: logger.error(f"BULL 錯誤: {e}")

results = st.session_state['scan_results']
if results is not None and params_changed(): st.warning("⚠️ 參數已更改，請重新掃描。")

# ==========================================
# 介面顯示：Tab 1~6
# ==========================================
with tab1:
    st.header("🟢 狙擊手波段策略")
    if mkt := st.session_state.get('market_status'):
        c1, c2, c3, c4 = st.columns(4); c1.metric("大盤", mkt['label']); c2.metric("指數", f"{mkt.get('close','-'):,.0f}" if isinstance(mkt.get('close'),(int,float)) else str(mkt.get('close','-'))); c3.metric("20MA", f"{mkt.get('ma20','-'):,.0f}" if isinstance(mkt.get('ma20'),(int,float)) else str(mkt.get('ma20','-'))); c4.metric("60MA", f"{mkt.get('ma60','-'):,.0f}" if isinstance(mkt.get('ma60'),(int,float)) else str(mkt.get('ma60','-')))
        st.divider()
    if results:
        if results.get('failed_list'):
            with st.expander(f"⚠️ 掃描失敗清單 ({len(results['failed_list'])})"): st.write(", ".join(results['failed_list']))
        s_trig = results['sniper_triggered']; trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]; trig_n = [x for x in s_trig if "N字" in x['狀態']]
        if trig_strong or trig_n:
            if trig_strong: st.markdown(f"#### 🚀 強勢突破 ({len(trig_strong)})"); display_full_table(pd.DataFrame(trig_strong).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_setup_date', '_defense', '_signal_high', '_signal_low'] if c in pd.DataFrame(trig_strong).columns]))
            if trig_n: st.markdown(f"#### 🎯 N字突破 ({len(trig_n)})"); display_full_table(pd.DataFrame(trig_n).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_setup_date', '_defense', '_signal_high', '_signal_low'] if c in pd.DataFrame(trig_n).columns]))
    else: st.info("👈 請點擊左側「開始全域掃描」")

with tab2:
    st.header("🏆 創高拉回佈局")
    st.caption("符合外資買超 + 年線上彎 + 創一年高 + 量縮拉回 10~25% + 重新站回 10MA 之標的")
    if results and results.get('dip_candidates'):
        display_full_table(pd.DataFrame(results['dip_candidates']))
    else: st.info("今日無訊號")

with tab3:
    st.header("🐂 BULL_v7 滾雪球")
    bull_res = st.session_state.get('bull_results'); pos_n = len(st.session_state['bull_positions'])
    st.metric("📋 持倉數量", pos_n)
    if not bull_res: st.info("請先執行全域掃描。")
    else:
        entry_list = bull_res.get('entry', [])
        addon_a_list = bull_res.get('addon_a', [])
        addon_b_list = bull_res.get('addon_b', [])
        exit_list = bull_res.get('exit', [])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 進場訊號", len(entry_list))
        c2.metric("🔺 加碼A (回測站回)", len(addon_a_list))
        c3.metric("🚀 加碼B (突破新高)", len(addon_b_list))
        c4.metric("🔴 出場訊號", len(exit_list))
        st.divider()

        if entry_list:
            st.markdown(f"#### 🟢 布林突破進場訊號 ({len(entry_list)} 筆)")
            df_entry = pd.DataFrame(entry_list).sort_values(by='成交量', ascending=False)
            display_full_table(df_entry)
        else:
            st.info("今日無布林突破進場訊號。")

        if addon_a_list:
            st.markdown(f"#### 🔺 加碼A：回測站回 ({len(addon_a_list)} 筆)")
            display_full_table(pd.DataFrame(addon_a_list))

        if addon_b_list:
            st.markdown(f"#### 🚀 加碼B：突破新高 ({len(addon_b_list)} 筆)")
            display_full_table(pd.DataFrame(addon_b_list))

        if exit_list:
            st.markdown(f"#### 🔴 出場訊號 ({len(exit_list)} 筆)")
            display_full_table(pd.DataFrame(exit_list))

with tab4:
    st.header("⚡ 隔日沖雷達")
    if results and results.get('day_candidates'): display_full_table(pd.DataFrame(results['day_candidates']))
    else: st.info("請先執行全域掃描，或今日無訊號。")

def plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info=None):
    df = df.copy(); df['MA_Trend'] = df['Close'].rolling(window=params.get('s_ma_trend', 60)).mean(); df['MA20'] = df['Close'].rolling(window=20).mean(); df['MA240'] = df['Close'].rolling(window=240).mean(); plot_df = df.tail(250)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], subplot_titles=("K線", "成交量", "外資每日買賣超(張)"))
    
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線', increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Trend'], line=dict(color='blue', width=1.5), name=f"{params.get('s_ma_trend', 60)}MA"), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量', marker_color='gray'), row=2, col=1)
    
    try:
        df_chip = load_cloud_chip_data(HF_CHIP_URL)
        stock_chip = df_chip[(df_chip['stock_id'] == str(stock_input).strip())].copy()
        if not stock_chip.empty:
            stock_chip.index = pd.to_datetime(stock_chip['date'])
            stock_chip = stock_chip.reindex(plot_df.index).fillna(0)
            colors = ['red' if x >= 0 else 'green' for x in stock_chip['net_buy']]
            fig.add_trace(go.Bar(x=plot_df.index, y=stock_chip['net_buy'], name='外資買賣超', marker_color=colors), row=3, col=1)
    except: pass

    fig.update_layout(xaxis_rangeslider_visible=False, height=800, template="plotly_dark"); return fig

with tab5:
    st.header("📊 個股診斷")
    col_in, col_btn = st.columns([3, 1])
    with col_in: stock_input = st.text_input("輸入代號", value="2330")
    if col_btn.button("診斷"):
        df = get_stock_data_with_realtime(stock_input, f"{stock_input}.TW", analysis_date_str, api)
        if df is not None: st.plotly_chart(plot_diagnosis_chart(df, stock_input, analysis_date_str, params), use_container_width=True)
        else: st.error("查無資料")

with tab6:
    st.header("🔧 系統診斷")
    
    st.subheader("1. 測試連線")
    if st.button("測試抓取 2330"):
        try:
            snaps = api.snapshots([api.Contracts.Stocks["2330"]])
            if snaps: st.success(f"✅ 2330 最新價: {snaps[0].close}")
        except Exception as e: st.error(f"錯誤: {e}")
        
    st.subheader("2. 合約防呆檢查")
    try:
        safe_map = get_stock_info_map(api)
        st.info(f"成功安全載入目標股票清單: **{len(safe_map)}** 支")
    except Exception as e:
        st.error(f"合約檢查異常: {e}")

    st.divider()
    
    st.subheader("3. ☁️ 雲端歷史報價資料庫檢測")
    if st.button("測試讀取雲端報價庫"):
        with st.spinner("正在從 Hugging Face 下載歷史報價庫..."):
            try:
                df_check = pd.read_parquet(HF_HISTORY_URL, columns=["date", "code"])
                df_check["date"] = pd.to_datetime(df_check["date"])
                parquet_last_date = df_check["date"].max()
                parquet_last_str = parquet_last_date.strftime("%Y-%m-%d")
                total_stocks = df_check["code"].nunique()

                today_tw = _now_tw().date()
                check_day = today_tw - datetime.timedelta(days=1)
                for _ in range(7):
                    if check_day.weekday() < 5: break
                    check_day -= datetime.timedelta(days=1)
                latest_trading_day = check_day

                days_lag = (latest_trading_day - parquet_last_date.date()).days

                st.success("✅ 成功連線 Hugging Face 歷史報價庫！")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📅 庫內最新日期", parquet_last_str)
                col_b.metric("📅 預估最近交易日", str(latest_trading_day))
                col_c.metric("⏳ 落後天數", f"{days_lag} 天" if days_lag >= 0 else "超前0天")
                st.info(f"📦 股票涵蓋數量：{total_stocks} 支")
            except Exception as e:
                st.error(f"❌ 雲端資料庫檢查失敗，請確認你的 URL 與 Hugging Face 權限：{e}")

    st.divider()
    
    st.subheader("4. ☁️ 雲端籌碼資料庫檢測 (X 光機)")
    if st.button("測試讀取雲端籌碼資料"):
        with st.spinner("正在從 Hugging Face 下載並解析資料..."):
            test_df = load_cloud_chip_data(HF_CHIP_URL)
            if test_df is not None and not test_df.empty:
                st.success("✅ 成功讀取籌碼資料庫！")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("總筆數 (僅限外資)", f"{len(test_df):,}")
                c2.metric("涵蓋股票檔數", f"{test_df['stock_id'].nunique():,}")
                c3.metric("資料最新日期", test_df['date'].max())
                
                st.write(f"📅 資料期間：{test_df['date'].min()} ~ {test_df['date'].max()}")
                st.write("🔍 資料前 5 筆預覽：")
                st.dataframe(test_df.head())
                
                tsmc = test_df[test_df['stock_id'] == '2330']
                if not tsmc.empty:
                    st.info(f"✅ 成功找到台積電 (2330) 籌碼，最新一筆：{tsmc['date'].max()} / 外資淨買超：{tsmc['net_buy'].iloc[-1]:.0f} 張")
                else:
                    st.warning("⚠️ 在資料庫中找不到 2330 (台積電) 的資料，請確認股票代號格式是否異常。")
            else:
                st.error("❌ 無法讀取資料，或資料庫為空。請檢查上面的報錯訊息。")
