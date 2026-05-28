# ==========================================
# 強勢股戰情室 V19 (三引擎版 / 解除乖離封印)
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
st.set_page_config(page_title="強勢股戰情室 V19+", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

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
HF_CHIP_URL = "https://huggingface.co/datasets/4340P/institutional_investors_parquet_by_stock/resolve/main/all_institutional_investors_2020_2026.parquet"
HF_HISTORY_URL = "https://huggingface.co/datasets/4340P/history/resolve/main/history.parquet"

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
            # 拔除 is_overheated 條件
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(bull_scan_one, code, stock_map[code], df, bull_params, pos_map, scan_date_ts, df_chip_main, chip_params): code for code, df in valid_codes.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            r = future.result()
            for k in res.keys(): 
                if r.get(k): res[k].append(r[k])
    return res

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
        if total_c == 0: return realtime_data
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
            rt_bar.progress(min(80, int((i + chunk_size) / total_c * 80)))
        rt_bar.progress(100); _txt.text(f"✅ 快照更新完成")
    except Exception as e: _txt.text(f"❌ 永豐連線異常"); rt_bar.progress(100)
    return realtime_data

# ==========================================
# 📈 綜合策略模組 (Sniper + Dip-Buyer)
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_status(analysis_date_str, _api_instance=None):
    try:
        df = yf.Ticker("^TWII").history(period="1y")
        if df.empty or df.index.tz is not None: df.index = df.index.tz_localize(None) if not df.empty else df.index
        if df.empty: return {'strong': True, 'score': 5, 'label': '無大盤資料'}
        today_str = _now_tw().strftime('%Y-%m-%d'); df = df[~df.index.duplicated(keep='last')].copy()
        latest_idx = df.index.get_loc(pd.Timestamp(analysis_date_str)) if pd.Timestamp(analysis_date_str) in df.index else -1
        close = df['Close'].ffill(); ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
        c = close.iloc[latest_idx]; m20 = ma20.iloc[latest_idx]; m60 = ma60.iloc[latest_idx]
        score = sum([4 if c > m20 else 0, 3 if c > m60 else 0, 3 if m20 > m60 else 0])
        return {'strong': score >= 7, 'score': score, 'label': "🟢 大盤強勢" if score >= 7 else ("🟡 大盤偏弱" if score >= 4 else "🔴 大盤弱勢"), 'close': round(c, 0), 'ma20': round(m20, 0), 'ma60': round(m60, 0)}
    except: return {'strong': True, 'score': 5, 'label': '大盤資料異常'}

def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None, market_score=5, df_chip_main=None):
    try:
        code_str = str(code).strip()
        df = pre_loaded_df.copy() if pre_loaded_df is not None else get_stock_data_with_realtime(code, info['symbol'], analysis_date_str, api)
        if df is None or df.empty or len(df) < 250: return None
        df = df.loc[~df.index.duplicated(keep='last')].copy()
        if pd.Timestamp(analysis_date_str) not in df.index: return None
        
        passed_chip_filter, chip_summary = evaluate_chip_filter(code_str, analysis_date_str, df, params, df_chip_main)
        if params.get('c_use_filter', True) and not passed_chip_filter: return {'sniper': None, 'dip': None}
        
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
        stock_name = info['short_name']; sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db)
        
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()
        ma60 = close.rolling(window=60).mean()
        ma240 = close.rolling(window=240).mean()
        vol_ma20 = volume.rolling(window=20).mean()
        
        c_today = close.iloc[idx]; v_today = volume.iloc[idx]; prev_h = high.iloc[idx-1]; daily_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1] * 100
        
        result_sniper = None; result_dip = None
        
        # ==================================
        # 🟢 1. 狙擊手波段策略 (拔除乖離限制)
        # ==================================
        if (v_today >= params['s_min_vol']) and (c_today > ma10.iloc[idx] > ma20.iloc[idx] > ma60.iloc[idx]) and (ma60.iloc[idx] > ma60.iloc[idx-1]):
            setup_found = False; s_high = 0; s_low = 0; s_close = 0; s_date = ""; defense_price = 0
            for k in range(1, 26):
                if (b_idx := idx - k) < 1: break
                if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > params['s_big_candle'] and volume.iloc[b_idx] > vol_ma20.iloc[b_idx] * 0.7 and close.iloc[b_idx] > op.iloc[b_idx]):
                    setup_found = True; s_low = low.iloc[b_idx]; s_high = high.iloc[b_idx]; s_close = close.iloc[b_idx]; s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    defense_price = (close.iloc[b_idx-1] if close.iloc[b_idx-1] >= op.iloc[b_idx-1] else op.iloc[b_idx-1]) * 0.99 if s_low > high.iloc[b_idx-1] else s_low * 0.99
                    break
            
            if setup_found:
                is_broken = False
                for k in range(b_idx + 1, idx + 1):
                    if close.iloc[k] < defense_price: is_broken = True; break
                
                if not is_broken:
                    is_breakout = c_today > prev_h
                    if is_breakout:
                        result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🚀 強勢突破", "防守價": f"{defense_price:.2f}", "法人籌碼": chip_summary, "sort_pct": daily_pct})
        
        # ==================================
        # 🏆 2. 創高拉回佈局策略 (Dip-Buyer)
        # ==================================
        high_250 = high.rolling(250).max()
        peak_60 = high.iloc[max(0, idx-60):idx+1].max()
        
        trend_pass = (ma240.iloc[idx] > ma240.iloc[idx-5]) and (c_today > ma240.iloc[idx])
        high_pass = (peak_60 >= high_250.iloc[idx]) and pd.notna(peak_60)
        
        if trend_pass and high_pass:
            pullback_pct = (c_today - peak_60) / peak_60 if peak_60 > 0 else 0
            pullback_pass = (params.get('db_p_min', -25) / 100 <= pullback_pct <= params.get('db_p_max', -10) / 100)
            vol_shrink = v_today < vol_ma20.iloc[idx]
            trigger_pass = (c_today > ma10.iloc[idx]) and (close.iloc[idx-1] <= ma10.iloc[idx-1])
            
            if pullback_pass and vol_shrink and trigger_pass:
                result_dip = {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🏆 創高拉回止跌", "拉回幅度": f"{pullback_pct*100:.2f}%", "一年高點": f"{peak_60:.2f}", "法人籌碼": chip_summary}

        return {'sniper': result_sniper, 'dip': result_dip}
    except Exception as e: return None

def display_full_table(df):
    if df is not None and not df.empty:
        st.dataframe(df[[c for c in df.columns if not c.startswith('_')]], hide_index=True, use_container_width=True, height=(len(df) * 35) + 38)
    else: st.info("無")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室 V19+")
st.sidebar.caption("純永豐金 API 企業級三引擎版")
st.sidebar.success("☁️ 雙雲端資料庫連線中")

analysis_date_input = st.sidebar.date_input("分析基準日", _now_tw().date())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')
start_scan = st.sidebar.button("🚀 開始全域掃描 (極速版)", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)
st.sidebar.divider()

with st.sidebar.expander("🛡️ 法人籌碼濾網", expanded=False):
    c_use_filter = st.checkbox("啟用籌碼過濾", value=True)
    c_lookback_days = st.number_input("最近 N 天", min_value=1, value=30)
    c_threshold_shares = st.number_input("外資買超大於 (張)", min_value=0, value=10000, step=1000)

with st.sidebar.expander("🏆 創高拉回策略 (波段)", expanded=True):
    st.caption("過濾創一年高後，量縮拉回之強勢股")
    db_p_min = st.slider("最大拉回幅度(%)", -40, -10, -25)
    db_p_max = st.slider("最小拉回幅度(%)", -30, -5, -10)

with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=False):
    s_big_candle = st.slider("長紅漲幅(%)", 2.0, 10.0, 5.0) / 100
    s_min_vol = st.number_input("波段最小量(張)", value=1000) * 1000
    
with st.sidebar.expander("🐂 BULL_v7 滾雪球參數", expanded=False):
    b_boll_period = st.number_input("布林週期", value=15, key="b_bp"); b_boll_std = st.number_input("布林標準差", value=2.1, key="b_bs"); b_squeeze_n = st.number_input("壓縮天數", value=15, key="b_sn"); b_squeeze_lb = st.number_input("回溯", value=5, key="b_sl"); b_vol_ratio = st.number_input("爆量倍數", value=1.5, key="b_vr"); b_sma_days = st.number_input("趨勢天數", value=3, key="b_sd"); b_min_vol = st.number_input("均量門檻(張)", value=1000, key="b_mv") * 1000; b_addon_b_pct = st.number_input("加碼B(%)", value=10, key="b_ab") / 100

max_workers_input = 16

params = {'c_use_filter': c_use_filter, 'c_lookback_days': c_lookback_days, 'c_threshold_shares': c_threshold_shares, 's_big_candle': s_big_candle, 's_min_vol': s_min_vol, 'db_p_min': db_p_min, 'db_p_max': db_p_max}
bull_params = {**BULL_PARAMS, "boll_period": int(b_boll_period), "boll_std": float(b_boll_std), "squeeze_n": int(b_squeeze_n), "squeeze_lookback": int(b_squeeze_lb), "vol_ratio": float(b_vol_ratio), "sma_trend_days": int(b_sma_days), "min_vol_shares": int(b_min_vol), "addon_b_profit": float(b_addon_b_pct)}
chip_params = {'c_use_filter': c_use_filter, 'c_lookback_days': c_lookback_days, 'c_threshold_shares': c_threshold_shares}

tab1, tab2, tab3, tab5, tab6 = st.tabs(["🟢 狙擊手突破", "🏆 創高拉回", "🐂 BULL滾雪球", "📊 個股診斷", "🔧 系統診斷"])

for k in ['scan_results', 'bull_results', 'scan_params', 'scan_date', 'market_status']:
    if k not in st.session_state: st.session_state[k] = None
if 'bull_positions' not in st.session_state: st.session_state['bull_positions'] = pd.DataFrame(columns=BULL_SHEET_COLS)

if start_scan:
    stock_map = get_stock_info_map(api)
    market_status = fetch_market_status(analysis_date_str, api)
    st.session_state['market_status'] = market_status
    
    df_chip_main = load_cloud_chip_data(HF_CHIP_URL) if c_use_filter else None
    history_data_store, load_msg = load_cloud_history_data(HF_HISTORY_URL)
    
    if history_data_store: history_data_store = {str(k).strip(): v for k, v in history_data_store.items() if str(k).strip() in stock_map}
    
    today_str = _now_tw().strftime('%Y-%m-%d')
    realtime_map = fetch_realtime_batch(list(history_data_store.keys()), api, status_text=status_text) if analysis_date_str == today_str else {}

    progress_bar.progress(0)
    tasks_data = {}
    for code, df in history_data_store.items():
        code_str = str(code).strip()
        df = df.loc[~df.index.duplicated(keep='first')].copy()
        if code_str in realtime_map and realtime_map[code_str].get('latest_trade_price', '-') != '-':
            try:
                rt = realtime_map[code_str]
                c_p = float(rt.get('latest_trade_price')); o_p = float(rt.get('open')) or c_p; h_p = float(rt.get('high')) or c_p; l_p = float(rt.get('low')) or c_p; v_p = float(rt.get('accumulate_trade_volume')) * 1000
                if c_p > 0:
                    today_ts = pd.Timestamp(today_str); new_row = pd.Series({'Open': o_p, 'High': h_p, 'Low': l_p, 'Close': c_p, 'Volume': v_p}, name=today_ts)
                    if today_ts in df.index: df.loc[today_ts] = new_row
                    else: df = pd.concat([df, new_row.to_frame().T])
            except: pass
        tasks_data[code_str] = df

    sniper_triggered = []; dip_candidates = []
    total = len(tasks_data); done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df, market_status.get('score', 5), df_chip_main): code for code, df in tasks_data.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total: progress_bar.progress(done / max(total, 1)); status_text.text(f"綜合掃描中: {done}/{total}")
            res = future.result()
            if isinstance(res, dict):
                if res.get('sniper') and res['sniper'][0] == "triggered": sniper_triggered.append(res['sniper'][1])
                if res.get('dip'): dip_candidates.append(res['dip'])

    st.session_state['scan_results'] = {'sniper_triggered': sniper_triggered, 'dip_candidates': dip_candidates}
    try: st.session_state['bull_results'] = bull_run_scan(stock_map, tasks_data, analysis_date_str, st.session_state['bull_positions'], bull_params, max_workers_input, status_text, progress_bar, df_chip_main, chip_params)
    except: pass
    
    progress_bar.progress(1.0)
    status_text.success("✅ 全域掃描完成！")

results = st.session_state['scan_results']

# 🟢 Tab 1: Sniper
with tab1:
    st.header("🟢 狙擊手突破")
    if results and results.get('sniper_triggered'):
        display_full_table(pd.DataFrame(results['sniper_triggered']).sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct']))
    else: st.info("無訊號或請點擊開始掃描")

# 🏆 Tab 2: Dip-Buyer
with tab2:
    st.header("🏆 創高拉回佈局")
    st.caption("符合外資買超 + 年線上彎 + 創一年高 + 量縮拉回 10~25% + 重新站回 10MA 之極品標的")
    if results and results.get('dip_candidates'):
        display_full_table(pd.DataFrame(results['dip_candidates']))
    else: st.info("今日無訊號")

# 🐂 Tab 3: BULL
with tab3:
    st.header("🐂 BULL_v7 滾雪球")
    bull_res = st.session_state.get('bull_results')
    if bull_res and bull_res.get('entry'):
        display_full_table(pd.DataFrame(bull_res['entry']))
    else: st.info("今日無進場訊號")

# 📊 Tab 5: 診斷
with tab5:
    st.header("📊 個股診斷")
    col_in, col_btn = st.columns([3, 1])
    with col_in: stock_input = st.text_input("輸入代號", value="2330")
    if col_btn.button("診斷"):
        df = get_stock_data_with_realtime(stock_input, f"{stock_input}.TW", analysis_date_str, api)
        if df is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume']), row=2, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)
        else: st.error("查無資料")

# 🔧 Tab 6: 系統
with tab6:
    st.header("🔧 系統連線狀態")
    st.success("✅ 雙雲端資料庫已鎖定。")
