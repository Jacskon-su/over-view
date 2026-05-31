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
st.markdown("""<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.metric-box { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
.stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
</style>""", unsafe_allow_html=True)

# ==========================================
# ☁️ 雲端與本機路徑設定
# ==========================================
HF_HISTORY_URL = "https://huggingface.co/datasets/4340P/history/resolve/main/history.parquet"
HF_CHIP_URL = "https://huggingface.co/datasets/4340P/institutional_investors_parquet_by_stock/resolve/main/all_institutional_investors_2020_2026.parquet"
LOCAL_HISTORY_FILE = "history_local_cache.parquet"
LOCAL_CHIP_FILE = "chip_local_cache.parquet"
BULL_GSHEET_URL = "https://docs.google.com/spreadsheets/d/1B7jM0W3PihLz9XUaV09oP2y69x3p5WbZgK1D7l97jR8/edit"

# ==========================================
# 🛠️ 輔助函式區
# ==========================================
def _now_tw():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=8)

@st.cache_resource(show_spinner=False)
def get_shioaji_api():
    try:
        api_key = st.secrets["SHIOAJI_API_KEY"]
        secret_key = st.secrets["SHIOAJI_SECRET_KEY"]
        api = sj.Shioaji(simulation=False)
        api.login(api_key=api_key, secret_key=secret_key, contracts_cb=lambda x: None)
        return api
    except Exception as e:
        st.warning(f"Shioaji 登入失敗: {e}")
        return None

api = get_shioaji_api()

def SMA(array, n):
    return pd.Series(array).rolling(window=n).mean()

class SniperStrategy(Strategy):
    ma_trend_period = 60; ma_long_period = 240; ma_base_exit = 20; ma_fast_exit = 10
    stop_loss_pct = 0.08; time_stop_days = 5; time_stop_profit = 0.02; take_profit_pct = 0.20
    def init(self):
        c = self.data.Close
        self.ma_trend = self.I(SMA, c, self.ma_trend_period)
        self.ma_long = self.I(SMA, c, self.ma_long_period)
        self.ma_base = self.I(SMA, c, self.ma_base_exit)
        self.ma_fast = self.I(SMA, c, self.ma_fast_exit)
    def next(self):
        if not self.position:
            self.buy()
        else:
            profit = (self.data.Close[-1] - self.position.entry_price) / self.position.entry_price
            bars_held = len(self.data) - self.position.entry_bar
            is_hard_stop = profit <= -self.stop_loss_pct
            is_time_stop = (bars_held >= self.time_stop_days) and (profit < self.time_stop_profit)
            is_tp = False
            if profit > self.take_profit_pct: is_tp = self.data.Close[-1] < self.ma_fast[-1]
            else: is_tp = self.data.Close[-1] < self.ma_base[-1]
            if is_hard_stop or is_time_stop or is_tp: self.position.close()

def get_detailed_sector(code, standard_group=None, custom_db=None):
    code_str = str(code).strip()
    if custom_db and code_str in custom_db: return custom_db[code_str]
    return standard_group if pd.notna(standard_group) and standard_group else "其他"

@st.cache_data(ttl=3600*12)
def load_cloud_chip_data(url):
    try:
        urllib.request.urlretrieve(url, LOCAL_CHIP_FILE)
        df_c = pd.read_parquet(LOCAL_CHIP_FILE, columns=['date', 'stock_id', 'name', 'buy', 'sell'])
        df_c.columns = [str(c).lower() for c in df_c.columns]
        df_c['stock_id'] = df_c['stock_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_c['name'] = df_c['name'].astype(str).str.strip()
        df_c = df_c[df_c['name'].str.contains('Foreign_Investor|外資', case=False, na=False)].copy()
        df_c['date'] = pd.to_datetime(df_c['date']).dt.tz_localize(None).dt.normalize()
        df_c['net_buy'] = ((df_c['buy'] - df_c['sell']) / 1000).astype('float32')
        return df_c
    except Exception as e:
        logger.error(f"下載/解析雲端籌碼失敗: {e}")
        return None

@st.cache_data(ttl=3600*12)
def load_cloud_history_data(url):
    try:
        urllib.request.urlretrieve(url, LOCAL_HISTORY_FILE)
        df = pd.read_parquet(LOCAL_HISTORY_FILE, columns=['date', 'code', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        return df
    except Exception as e:
        logger.error(f"下載/解析歷史報價失敗: {e}")
        return None

def evaluate_chip_filter(code_str, analysis_date_str, df, params, df_chip_main):
    passed = True; chip_summary = "無籌碼資料"
    if df_chip_main is not None and code_str in df_chip_main.columns:
        chip_series = df_chip_main[code_str].reindex(df.index).fillna(0)
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        days = params.get('c_chip_days', 10)
        if idx >= days - 1:
            period_sum = chip_series.iloc[idx-days+1 : idx+1].sum()
            chip_summary = f"{int(period_sum):,}張"
            if params.get('c_use_filter', True) and period_sum < params.get('c_chip_min', 0): passed = False
    return passed, chip_summary

def get_stock_info_map(api_obj):
    if not api_obj: return {}
    try:
        m = {}
        for exchange in [api_obj.Contracts.Stocks.TSE, api_obj.Contracts.Stocks.OTC]:
            for contract in exchange:
                if len(contract.code) == 4: m[contract.code] = {'symbol': contract.code, 'short_name': contract.name, 'group': getattr(contract, 'category', '其他')}
        return m
    except: return {}

def get_stock_data_with_realtime(code, symbol, target_date_str, api_obj):
    try:
        df = yf.download(f"{symbol}.TW", period="1y", interval="1d", progress=False)
        if df.empty: df = yf.download(f"{symbol}.TWO", period="1y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None).normalize()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if target_date_str == _now_tw().strftime('%Y-%m-%d') and api_obj:
            contract = api_obj.Contracts.Stocks.get(symbol)
            if contract:
                snap = api_obj.snapshots([contract])
                if snap:
                    s = snap[0]
                    if s.volume > 0:
                        rt_ts = pd.Timestamp(target_date_str)
                        if rt_ts in df.index: df.loc[rt_ts] = [s.open, s.high, s.low, s.close, s.volume]
                        else: df.loc[rt_ts] = pd.Series({'Open': s.open, 'High': s.high, 'Low': s.low, 'Close': s.close, 'Volume': s.volume})
        return df
    except: return None

# ==========================================
# 📈 策略與掃描核心
# ==========================================
def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None, market_score=5, df_chip_main=None):
    try:
        code_str = str(code).strip()

        df = pre_loaded_df.copy() if pre_loaded_df is not None else get_stock_data_with_realtime(code, info['symbol'], analysis_date_str, api)
        # ⚠️ 將資料長度要求提高到 250 天，確保年線可以正確計算
        if df is None or df.empty or len(df) < 250: return "資料長度不足 (<250天)"
        df = df.loc[~df.index.duplicated(keep='last')].copy()
        if pd.Timestamp(analysis_date_str) not in df.index: return f"無 {analysis_date_str} 交易資料"
        
        passed_chip_filter, chip_summary = evaluate_chip_filter(code_str, analysis_date_str, df, params, df_chip_main)
        if params.get('c_use_filter', True) and not passed_chip_filter: return {'sniper': None, 'day': None, 'dip': None}
        
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
        
        stock_name = info['short_name']
        sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db)
        result_sniper = None
        result_day = None
        result_dip = None
        
        c_today = close.iloc[idx]
        v_today = volume.iloc[idx]
        yesterday_c = close.iloc[idx-1] if idx > 0 else c_today
        daily_pct = (c_today - yesterday_c) / yesterday_c * 100
        
        # ==================================
        # 🎯 真・均線起漲狙擊手 (30日籌碼 + 破底防守版)
        # ==================================
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()
        ma60 = close.rolling(window=60).mean()
        ma240 = close.rolling(window=240).mean()
        v_ma5 = volume.rolling(window=5).mean()
        
        chip_30d_sum = 0
        if df_chip_main is not None and code_str in df_chip_main.columns:
            chip_series = df_chip_main[code_str].reindex(df.index).fillna(0)
            chip_30d_sum = chip_series.iloc[max(0, idx-29):idx+1].sum()
        
        if v_today >= 1000000 and chip_30d_sum >= 5000:
            ma20_is_rising = ma20.iloc[idx] >= ma20.iloc[idx-1] if idx > 0 else False
            above_60ma = c_today > ma60.iloc[idx]
            above_240ma = pd.notna(ma240.iloc[idx]) and (c_today > ma240.iloc[idx])
            
            if ma20_is_rising and above_60ma and above_240ma:
                lowest_20d = low.iloc[max(0, idx-19):idx+1].min()
                not_extended = c_today <= (lowest_20d * 1.15)
                
                if not_extended:
                    recent_high = high.iloc[max(0, idx-6):idx].max() if idx >= 6 else high.iloc[0:idx].max()
                    recent_low = low.iloc[max(0, idx-6):idx].min() if idx >= 6 else low.iloc[0:idx].min()
                    yesterday_ma20 = ma20.iloc[idx-1] if idx > 0 else 0
                    
                    near_ma20 = abs(yesterday_c - yesterday_ma20) / yesterday_ma20 < 0.05 if yesterday_ma20 else False
                    tight_range = (recent_high - recent_low) / recent_low < 0.08 if recent_low else False
                    
                    if near_ma20 and tight_range:
                        price_surge = (c_today - yesterday_c) / yesterday_c >= 0.04
                        vol_surge = v_today > (v_ma5.iloc[idx-1] * 2) if idx > 0 else False
                        
                        if price_surge and vol_surge:
                            recent_15d_low = low.iloc[max(0, idx-14):idx+1].min()
                            calculated_stop = recent_15d_low * 0.98
                            max_loss_price = c_today * 0.90 
                            defense_price = max(calculated_stop, max_loss_price)
                            
                            # 🔥 狀態名稱保留 '強勢突破' 四個字，確保顯示在表格內
                            result_sniper = ("triggered", {
                                "代號": code, 
                                "名稱": stock_name, 
                                "收盤": f"{c_today:.2f}", 
                                "漲幅": f"{daily_pct:+.2f}%", 
                                "產業": sector_name, 
                                "狀態": "🚀 強勢突破(真狙擊)", 
                                "外資30日": f"{int(chip_30d_sum)}張",
                                "防守價": f"{defense_price:.2f}", 
                                "法人籌碼": chip_summary, 
                                "sort_pct": daily_pct
                            })

        # ==================================
        # ⚡ Day Trade 蓄勢待發策略
        # ==================================
        d_period = params.get('d_period', 20)
        d_threshold = params.get('d_threshold', 2)
        d_min_vol = params.get('d_min_vol', 1000)
        d_min_pct = params.get('d_min_pct', 2)
        
        d_close = close.iloc[idx]; d_open = op.iloc[idx]; d_high = high.iloc[idx]; d_volume = volume.iloc[idx]; d_prev_close = close.iloc[idx-1]
        if idx >= d_period and (d_close > d_open) and ((d_high - d_close) / d_close < 0.01) and (d_min_pct/100 < (pct_chg_val := (d_close - d_prev_close) / d_prev_close) < 0.095) and ((d_volume / 1000) > d_min_vol) and (d_close >= ((prev_period_high := high.iloc[idx-d_period : idx].max()) * (1 - (d_threshold / 100)))) and (d_high <= prev_period_high):
            result_day = {"代號": code, "名稱": stock_name, "收盤": f"{d_close:.2f}", "產業": sector_name, "漲幅": f"{(pct_chg_val*100):.2f}%", "成交量": int(d_volume/1000), "前波高點": f"{prev_period_high:.2f}", "法人籌碼": chip_summary, "距離高點": f"{(d_close - prev_period_high) / prev_period_high * 100:+.2f}%", "狀態": "⚡ 蓄勢待發"}
        
        # ==================================
        # 🏆 創高拉回佈局策略 (Dip-Buyer)
        # ==================================
        high_250 = high.rolling(250).max()
        peak_60 = high.iloc[max(0, idx-60):idx+1].max()
        
        trend_pass = (pd.notna(ma240.iloc[idx]) and ma240.iloc[idx] > ma240.iloc[idx-5]) and (c_today > ma240.iloc[idx])
        high_pass = (peak_60 >= high_250.iloc[idx]) and pd.notna(peak_60)
        
        if trend_pass and high_pass:
            pullback_pct = (c_today - peak_60) / peak_60 if peak_60 > 0 else 0
            db_p_min = params.get('db_p_min', -25) / 100
            db_p_max = params.get('db_p_max', -10) / 100
            pullback_pass = (db_p_min <= pullback_pct <= db_p_max)
            
            vol_ma20 = volume.rolling(window=20).mean()
            vol_shrink = v_today < vol_ma20.iloc[idx]
            
            trigger_pass = (c_today > ma10.iloc[idx]) and (yesterday_c <= ma10.iloc[idx-1])
            
            if pullback_pass and vol_shrink and trigger_pass:
                result_dip = {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name, "狀態": "🏆 創高拉回止跌", "拉回幅度": f"{pullback_pct*100:.2f}%", "一年高點": f"{peak_60:.2f}", "法人籌碼": chip_summary}

        return {'sniper': result_sniper, 'day': result_day, 'dip': result_dip}
    except Exception as e: 
        return f"執行錯誤: {e}"


def fetch_data_batch(codes, info_map, target_date_str, api_obj):
    dfs = {}
    def _fetch(code):
        df = get_stock_data_with_realtime(code, info_map[code]['symbol'], target_date_str, api_obj)
        return code, df
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch, code): code for code in codes}
        for future in concurrent.futures.as_completed(futures):
            code, df = future.result()
            if df is not None and not df.empty: dfs[code] = df
    return dfs

def fetch_realtime_batch(codes, info_map, api_obj):
    updates = {}
    if not api_obj: return updates
    contracts = [api_obj.Contracts.Stocks.get(info_map[c]['symbol']) for c in codes if info_map.get(c)]
    contracts = [c for c in contracts if c]
    try:
        snaps = api_obj.snapshots(contracts)
        for s, contract in zip(snaps, contracts):
            if s.volume > 0: updates[contract.code] = {'Open': s.open, 'High': s.high, 'Low': s.low, 'Close': s.close, 'Volume': s.volume}
    except: pass
    return updates

def display_full_table(df):
    if df is not None and not df.empty:
        st.dataframe(df[[c for c in df.columns if not c.startswith('_')]], hide_index=True, use_container_width=True, height=(len(df) * 35) + 38)
    else: st.info("無")

# ==========================================
# 📊 UI 介面與主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室設定")
analysis_date = st.sidebar.date_input("掃描日期", _now_tw().date())
analysis_date_str = analysis_date.strftime('%Y-%m-%d')

st.sidebar.markdown("### 🎯 Sniper 均線起漲參數")
s_ma_trend = st.sidebar.number_input("趨勢均線 (MA)", 20, 120, 60, 10)
s_use_year = st.sidebar.checkbox("需站上年線", value=True)
s_big_candle = st.sidebar.number_input("長紅判定(>%)", 1.0, 9.9, 4.0, 0.5) / 100
s_vol_ratio = st.sidebar.number_input("爆量倍數 (vs 月均量)", 0.5, 5.0, 0.7, 0.1)
s_min_vol = st.sidebar.number_input("起漲最低成交量", 500, 5000, 1000, 500)
s_setup_lookback = st.sidebar.number_input("N字尋找長紅範圍(天)", 10, 60, 25, 5)
s_pullback_max = st.sidebar.number_input("N字回檔極限(%)", 10, 80, 50, 5)

st.sidebar.markdown("### ⚡ Day 蓄勢待發參數")
d_period = st.sidebar.number_input("前高尋找期間", 10, 60, 20, 5)
d_threshold = st.sidebar.number_input("距離前高(%)", 0.5, 5.0, 2.0, 0.5)
d_min_vol = st.sidebar.number_input("今日最低成交量(千股)", 500, 5000, 1000, 500)
d_min_pct = st.sidebar.number_input("今日最低漲幅(%)", 0.5, 5.0, 2.0, 0.5)

st.sidebar.markdown("### 💰 籌碼濾網 (共用)")
c_use_filter = st.sidebar.checkbox("啟用籌碼濾網", value=True)
c_chip_days = st.sidebar.number_input("累積天數", 1, 60, 10, 1)
c_chip_min = st.sidebar.number_input("最低買超張數", 0, 20000, 3000, 1000)

st.sidebar.markdown("### 🏆 創高拉回參數")
db_p_min = st.sidebar.number_input("拉回最小幅度 (%)", -50, -5, -25, 1)
db_p_max = st.sidebar.number_input("拉回最大幅度 (%)", -50, -5, -10, 1)

params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 's_big_candle': s_big_candle, 's_vol_ratio': s_vol_ratio,
    's_min_vol': s_min_vol, 's_setup_lookback': s_setup_lookback, 's_pullback_max': s_pullback_max,
    'd_period': d_period, 'd_threshold': d_threshold, 'd_min_vol': d_min_vol, 'd_min_pct': d_min_pct,
    'c_use_filter': c_use_filter, 'c_chip_days': c_chip_days, 'c_chip_min': c_chip_min,
    'db_p_min': db_p_min, 'db_p_max': db_p_max
}

tab1, tab2, tab3 = st.tabs(["📊 每日掃描戰情室", "🏆 創高拉回佈局", "⚙️ 系統狀態與雲端資料庫"])

with tab3:
    st.header("⚙️ 系統狀態與資料庫檢測")
    st.info("💡 系統啟動時會自動讀取雲端 Parquet 歷史報價與籌碼資料。")
    
    st.subheader("1. 🔗 API 連線狀態")
    if api: st.success("✅ Shioaji API 登入成功")
    else: st.error("❌ Shioaji API 登入失敗")

    st.divider()

    st.subheader("2. ☁️ 雲端報價資料庫檢測 (X 光機)")
    if st.button("測試讀取雲端報價資料"):
        with st.spinner("正在從 Hugging Face 下載並解析資料..."):
            test_df = load_cloud_history_data(HF_HISTORY_URL)
            if test_df is not None and not test_df.empty:
                st.success("✅ 成功讀取報價資料庫！")
                total_stocks = test_df['code'].nunique()
                parquet_last_date = test_df['date'].max()
                parquet_last_str = parquet_last_date.strftime('%Y-%m-%d')
                
                tz = datetime.timezone(datetime.timedelta(hours=8))
                now = datetime.datetime.now(tz)
                latest_trading_day = now.date() if now.hour >= 15 else (now - datetime.timedelta(days=1)).date()
                if latest_trading_day.weekday() >= 5: latest_trading_day -= datetime.timedelta(days=latest_trading_day.weekday() - 4)
                
                days_lag = (latest_trading_day - parquet_last_date.date()).days
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📅 庫內最新日期", parquet_last_str)
                col_b.metric("📅 預估最近交易日", str(latest_trading_day))
                col_c.metric("⏳ 落後天數", f"{days_lag} 天" if days_lag >= 0 else "超前0天")
                st.info(f"📦 股票涵蓋數量：{total_stocks} 支")
            else:
                st.error("❌ 雲端資料庫檢查失敗。")

    st.divider()
    
    st.subheader("3. ☁️ 雲端籌碼資料庫檢測 (X 光機)")
    if st.button("測試讀取雲端籌碼資料"):
        with st.spinner("正在從 Hugging Face 下載並解析資料..."):
            test_df = load_cloud_chip_data(HF_CHIP_URL)
            if test_df is not None and not test_df.empty:
                st.success("✅ 成功讀取籌碼資料庫！")
                c1, c2, c3 = st.columns(3)
                c1.metric("總筆數 (僅限外資)", f"{len(test_df):,}")
                c2.metric("涵蓋股票檔數", f"{test_df['stock_id'].nunique():,}")
                c3.metric("資料最新日期", test_df['date'].max().strftime('%Y-%m-%d'))
            else:
                st.error("❌ 雲端資料庫檢查失敗。")

with tab1:
    st.header("📊 每日全域掃描 (雲端 Parquet 引擎)")
    col1, col2 = st.columns([1, 4])
    with col1:
        start_scan = st.button("🚀 開始全域掃描", type="primary", use_container_width=True)
    
    if start_scan:
        st.session_state['scan_results'] = None
        info_map = get_stock_info_map(api)
        if not info_map: st.error("❌ 無法取得股票清單")
        else:
            with st.spinner("📦 正在載入雲端巨量資料庫..."):
                df_hist_main = load_cloud_history_data(HF_HISTORY_URL)
                df_chip_main = load_cloud_chip_data(HF_CHIP_URL)
                
                if df_chip_main is not None and not df_chip_main.empty:
                    st.session_state['df_chip_pivot'] = df_chip_main.pivot_table(index='date', columns='stock_id', values='net_buy', aggfunc='sum').fillna(0)
                else: st.session_state['df_chip_pivot'] = None
                
            if df_hist_main is None or df_hist_main.empty:
                st.error("❌ 無法載入歷史報價資料庫，掃描中斷。")
            else:
                st.success(f"✅ 雲端資料載入完畢 (最新至: **{df_hist_main['date'].max().strftime('%Y-%m-%d')}**)")
                sniper_triggered = []; sniper_setup = []; sniper_watching = []; day_candidates = []; dip_candidates = []; failed_list = []
                
                # 若今天日期大於 Parquet 最新日期，且在盤中/盤後，嘗試抓取今日即時資料填補
                parquet_max_date = df_hist_main['date'].max()
                need_realtime = False
                if pd.Timestamp(analysis_date_str) > parquet_max_date:
                    need_realtime = True
                    st.info(f"🔄 目標掃描日 ({analysis_date_str}) 新於資料庫，將自動合併即時 API 報價 (較耗時)...")

                target_codes = list(info_map.keys())
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if not need_realtime:
                    for i, code in enumerate(target_codes):
                        if i % 100 == 0:
                            progress_bar.progress((i + 1) / len(target_codes))
                            status_text.text(f"掃描中... {i+1} / {len(target_codes)}")
                        
                        df_stock = df_hist_main[df_hist_main['code'] == str(code)].copy()
                        if df_stock.empty: continue
                        df_stock.set_index('date', inplace=True)
                        df_stock.sort_index(inplace=True)
                        
                        res = analyze_combined_strategy(code, info_map[code], analysis_date_str, params, SECTOR_DB, pre_loaded_df=df_stock, df_chip_main=st.session_state['df_chip_pivot'])
                        if isinstance(res, dict):
                            if res.get('sniper'):
                                typ, data = res['sniper']
                                if typ == "triggered": sniper_triggered.append(data)
                                elif typ == "new_setup": sniper_setup.append(data)
                                elif typ == "watching": sniper_watching.append(data)
                            if res.get('day'): day_candidates.append(res['day'])
                            if res.get('dip'): dip_candidates.append(res['dip'])
                        else: failed_list.append(f"{code}: {res}")
                else:
                    batch_size = 100
                    for i in range(0, len(target_codes), batch_size):
                        batch_codes = target_codes[i:i+batch_size]
                        progress_bar.progress((i + 1) / len(target_codes))
                        status_text.text(f"混合即時資料掃描中... {i+1} / {len(target_codes)}")
                        
                        rt_updates = fetch_realtime_batch(batch_codes, info_map, api)
                        for code in batch_codes:
                            df_stock = df_hist_main[df_hist_main['code'] == str(code)].copy()
                            if not df_stock.empty:
                                df_stock.set_index('date', inplace=True)
                                df_stock.sort_index(inplace=True)
                            
                            if code in rt_updates:
                                rt = rt_updates[code]
                                df_stock.loc[pd.Timestamp(analysis_date_str)] = [code, rt['Open'], rt['High'], rt['Low'], rt['Close'], rt['Volume']]
                            
                            if not df_stock.empty:
                                res = analyze_combined_strategy(code, info_map[code], analysis_date_str, params, SECTOR_DB, pre_loaded_df=df_stock, df_chip_main=st.session_state['df_chip_pivot'])
                                if isinstance(res, dict):
                                    if res.get('sniper'):
                                        typ, data = res['sniper']
                                        if typ == "triggered": sniper_triggered.append(data)
                                        elif typ == "new_setup": sniper_setup.append(data)
                                        elif typ == "watching": sniper_watching.append(data)
                                    if res.get('day'): day_candidates.append(res['day'])
                                    if res.get('dip'): dip_candidates.append(res['dip'])
                                else: failed_list.append(f"{code}: {res}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ 掃描完成！")
                st.session_state['scan_results'] = {'sniper_triggered': sniper_triggered, 'sniper_setup': sniper_setup, 'sniper_watching': sniper_watching, 'day_candidates': day_candidates, 'dip_candidates': dip_candidates, 'failed_list': failed_list}

    results = st.session_state.get('scan_results')
    if results:
        s_trig = results['sniper_triggered']
        
        # 🔥 UI 這裡就是靠找這幾個字來分類表格的，現在一定找得到了！
        trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]
        trig_n = [x for x in s_trig if "N字" in x['狀態']]
        
        if trig_strong or trig_n:
            if trig_strong: 
                st.markdown(f"#### 🚀 強勢突破 ({len(trig_strong)})")
                display_full_table(pd.DataFrame(trig_strong).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_setup_date', '_defense', '_signal_high', '_signal_low'] if c in pd.DataFrame(trig_strong).columns]))
            if trig_n: 
                st.markdown(f"#### 🎯 N字突破 ({len(trig_n)})")
                display_full_table(pd.DataFrame(trig_n).sort_values(by='sort_pct', ascending=False).drop(columns=[c for c in ['sort_pct', '_setup_date', '_defense', '_signal_high', '_signal_low'] if c in pd.DataFrame(trig_n).columns]))
        else:
            st.info("今天沒有出現強勢突破或 N 字突破的標的。")

with tab2:
    st.header("🏆 創高拉回佈局")
    st.caption("符合外資買超 + 年線上彎 + 創一年高 + 量縮拉回 10~25% + 重新站回 10MA 之標的")
    if results and results.get('dip_candidates'):
        display_full_table(pd.DataFrame(results['dip_candidates']))
    elif results:
        st.info("今天沒有符合創高拉回條件的標的。")
    else:
        st.info("👈 請點擊左側「開始全域掃描」")
