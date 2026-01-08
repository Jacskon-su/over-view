import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(
    page_title="強勢股戰情室",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 依賴檢查 ---
try:
    import twstock
except ImportError:
    st.error("❌ 缺少 `twstock` 套件，請輸入 `pip install twstock` 安裝")
    st.stop()

# 自訂 CSS
st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 策略核心邏輯類別 (Backtesting用 - 僅供診斷分頁)
# ==========================================
def SMA(array, n):
    return pd.Series(array).rolling(window=n).mean()

class SniperStrategy(Strategy):
    ma_trend_period = 60
    ma_long_period = 240
    ma_base_exit = 20
    ma_fast_exit = 10
    vol_ma_period = 5
    big_candle_pct = 0.03
    min_volume_shares = 2000000
    lookback_window = 10
    use_year_line = True 
    
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

    def next(self):
        price = self.data.Close[-1]
        prev_high = self.data.High[-2]
        
        if self.position:
            if price < self.setup_low_price:
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
            elif price < self.setup_low_price:
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

# ==========================================
# 🛠️ 輔助函式與資料庫
# ==========================================
CUSTOM_SECTOR_MAP = {
    '2317': 'AI伺服器', '2382': 'AI伺服器', '3231': 'AI伺服器', '2356': 'AI伺服器', '6669': 'AI伺服器', '2376': 'AI伺服器',
    '3017': '散熱模組', '3324': '散熱模組', '2421': '散熱模組', '3653': '散熱模組',
    '1513': '重電綠能', '1519': '重電綠能', '1503': '重電綠能', '1504': '重電綠能', '1609': '重電綠能',
    '3661': 'IP/ASIC', '3443': 'IP/ASIC', '3035': 'IP/ASIC', '3529': 'IP/ASIC', '6531': 'IP/ASIC',
    '2603': '貨櫃航運', '2609': '貨櫃航運', '2615': '貨櫃航運',
    '2368': 'PCB/CCL', '3037': 'PCB/CCL', '6213': 'PCB/CCL', '6274': 'PCB/CCL',
    '2330': '半導體', '3711': '半導體封測'
}

def get_detailed_sector(code):
    """取得細分產業"""
    if code in CUSTOM_SECTOR_MAP: return CUSTOM_SECTOR_MAP[code]
    try:
        if code in twstock.codes: return twstock.codes[code].group
    except: pass
    return "其他"

@st.cache_data(ttl=3600)
def get_stock_info_map():
    """取得上市櫃股票資訊表"""
    try:
        stock_map = {}
        for code, info in twstock.twse.items():
            if len(code) == 4: stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TW", 'short_name': info.name}
        for code, info in twstock.tpex.items():
            if len(code) == 4: stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TWO", 'short_name': info.name}
        return stock_map
    except:
        return {}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history_data(symbol, start_date=None, end_date=None, period="2y"):
    """下載數據 (快取)"""
    try:
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return None

def get_stock_data_with_realtime(code, symbol, analysis_date_str):
    """取得資料並補即時盤"""
    df = fetch_history_data(symbol)
    if df is None or df.empty: return None
    
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
        except: pass
    return df

# ==========================================
# 🧠 綜合分析引擎 (一次運算雙策略)
# ==========================================
def analyze_combined_strategy(code, info, analysis_date_str, params):
    """
    整合分析函式
    回傳字典: {'sniper': result_dict_or_none, 'day_trading': result_dict_or_none}
    """
    try:
        # 1. 取得資料
        df = get_stock_data_with_realtime(code, info['symbol'], analysis_date_str)
        if df is None or len(df) < 250: return None # 至少要有約一年資料供長線判斷

        # 定位日期
        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values: return None
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        op = df['Open']
        stock_name = info['short_name']

        # 初始化結果
        result_sniper = None
        result_day = None

        # ==========================================
        # 🟢 策略 A: 狙擊手波段 (Sniper Strategy)
        # ==========================================
        # 參數
        s_ma_trend = params['s_ma_trend']
        s_use_year = params['s_use_year']
        s_big_candle = params['s_big_candle']
        s_min_vol = params['s_min_vol'] # 使用較寬鬆的量 (若有不同)

        # 指標
        ma_t = close.rolling(window=s_ma_trend).mean()
        ma_y = close.rolling(window=240).mean()
        vol_ma = volume.rolling(window=5).mean()

        # 狙擊手基礎濾網
        is_sniper_candidate = True
        if volume.iloc[idx] < s_min_vol: is_sniper_candidate = False
        if s_use_year and close.iloc[idx] < ma_y.iloc[idx]: is_sniper_candidate = False
        if not (close.iloc[idx] > ma_t.iloc[idx] and ma_t.iloc[idx] > ma_t.iloc[idx-1]): is_sniper_candidate = False

        if is_sniper_candidate:
            # 判斷是否為 Setup (長紅)
            is_setup = (
                (close.iloc[idx] - close.iloc[idx-1]) / close.iloc[idx-1] > s_big_candle and
                volume.iloc[idx] > vol_ma.iloc[idx] and
                close.iloc[idx] > op.iloc[idx]
            )
            
            # 回溯尋找 Setup
            setup_found = False
            s_high = 0 
            s_low = 0
            s_close = 0
            s_date = ""
            setup_idx = -1
            
            for k in range(1, 11): # 回溯10天
                b_idx = idx - k
                if b_idx < 0: break
                if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > s_big_candle and
                    volume.iloc[b_idx] > vol_ma.iloc[b_idx] and
                    close.iloc[b_idx] > op.iloc[b_idx]):
                    setup_found = True
                    setup_idx = b_idx
                    s_low = low.iloc[b_idx]
                    s_high = high.iloc[b_idx]
                    s_close = close.iloc[b_idx]
                    s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    break
            
            c_today = close.iloc[idx]
            prev_h = high.iloc[idx-1]
            
            if setup_found:
                # 檢查從 Setup 後到今天 (idx) 的路徑狀態
                is_broken = False # 是否跌破長紅低點
                dropped_below_high = False # 是否曾經跌破長紅高點

                # 檢查區間：Setup後第一天 ~ 今天 (包含今天)
                for k in range(setup_idx + 1, idx + 1):
                    c_k = close.iloc[k]
                    if c_k < s_low:
                        is_broken = True
                        break
                    if c_k < s_high:
                        dropped_below_high = True

                if not is_broken:
                    # 路徑 1: 強勢路徑 (從未跌破長紅高)
                    if not dropped_below_high:
                        # 漲幅限制：距離 Setup 收盤 10% 內 (避免追高)
                        pct_from_setup = (c_today - s_close) / s_close
                        
                        if pct_from_setup <= 0.10:
                            # 判斷：突破昨日高點 -> 強勢突破 / 否則 -> 強勢整理
                            if c_today > prev_h:
                                result_sniper = ("triggered", {
                                    "代號": code, "名稱": stock_name, 
                                    "收盤": f"{c_today:.2f}", 
                                    "狀態": "🚀 強勢突破", 
                                    "訊號日": s_date, "突破價": f"{prev_h:.2f}"
                                })
                            else:
                                curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                                result_sniper = ("watching", {
                                    "代號": code, "名稱": stock_name, 
                                    "收盤": f"{c_today:.2f}", 
                                    "狀態": "💪 強勢整理", 
                                    "訊號日": s_date, "防守": f"{s_low:.2f}", 
                                    "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"
                                })
                    
                    # 路徑 2: 回檔路徑 (曾經跌破長紅高，但守住長紅低)
                    else:
                        # 判斷：突破昨日高點 -> N字突破 / 否則 -> 回檔整理
                        if c_today > prev_h:
                            result_sniper = ("triggered", {
                                "代號": code, "名稱": stock_name, 
                                "收盤": f"{c_today:.2f}", 
                                "狀態": "🎯 N字突破", 
                                "訊號日": s_date, "突破價": f"{prev_h:.2f}"
                            })
                        else:
                            curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                            result_sniper = ("watching", {
                                "代號": code, "名稱": stock_name, 
                                "收盤": f"{c_today:.2f}", 
                                "狀態": "📉 回檔整理", 
                                "訊號日": s_date, "防守": f"{s_low:.2f}", 
                                "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"
                            })
            
            elif is_setup:
                # 剛起漲
                prev_c = close.iloc[idx-1]
                pct_chg = (c_today - prev_c) / prev_c * 100
                stock_group = get_detailed_sector(code)
                result_sniper = ("new_setup", {
                    "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", 
                    "狀態": "🔥 剛起漲", "漲幅": f"{pct_chg:+.2f}%", "族群": stock_group
                })

        # ==========================================
        # ⚡ 策略 B: 隔日沖雷達 (Day Trading Strategy)
        # ==========================================
        # 參數
        d_period = params['d_period']
        d_threshold = params['d_threshold']
        d_min_vol = params['d_min_vol']
        d_min_pct = params['d_min_pct']

        # 基礎變數
        d_close = close.iloc[idx]
        d_open = op.iloc[idx]
        d_high = high.iloc[idx]
        d_volume = volume.iloc[idx]
        d_prev_close = close.iloc[idx-1]
        
        # 隔日沖邏輯
        # 1. 實體紅K
        is_red = d_close > d_open
        # 2. 收盤極強 (上影線 < 1%)
        upper_shadow = (d_high - d_close) / d_close
        is_strong_close = upper_shadow < 0.01
        # 3. 漲幅過濾 (min_pct < 漲幅 < 9.5%)
        pct_chg_val = (d_close - d_prev_close) / d_prev_close
        is_momentum_ok = (pct_chg_val > d_min_pct/100) and (pct_chg_val < 0.095)
        # 4. 成交量濾網 (張)
        is_vol_ok = (d_volume / 1000) > d_min_vol
        # 5. 蓄勢待發 (接近前 N 日高點 1% 內，但今日未創新高)
        # 取前 N 日 (不含今日)
        if idx >= d_period:
            prev_period_high = high.iloc[idx-d_period : idx].max()
            threshold_factor = 1 - (d_threshold / 100)
            is_near_high = d_close >= (prev_period_high * threshold_factor)
            is_not_new_high = d_high <= prev_period_high
            
            if is_red and is_strong_close and is_momentum_ok and is_vol_ok and is_near_high and is_not_new_high:
                # 計算距離
                dist_to_high = (d_close - prev_period_high) / prev_period_high * 100
                result_day = {
                    "代號": code,
                    "名稱": stock_name,
                    "收盤": f"{d_close:.2f}",
                    "漲幅": f"{(pct_chg_val*100):.2f}%",
                    "成交量": int(d_volume/1000),
                    "前波高點": f"{prev_period_high:.2f}",
                    "距離高點": f"{dist_to_high:+.2f}%",
                    "狀態": "⚡ 蓄勢待發"
                }

        return {'sniper': result_sniper, 'day': result_day}

    except Exception as e:
        # print(e) # Debug use
        return None

# 🔥 全展開表格顯示函式
def display_full_table(df):
    if df is not None and not df.empty:
        height = (len(df) * 35) + 38
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)
    else:
        st.info("無")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室")
st.sidebar.caption("波段與短線的極致整合")

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

# 掃描按鈕 (全域) - 移至最上方
start_scan = st.sidebar.button("🚀 開始全域掃描", type="primary")

# 佔位元件：進度條與狀態文字 (初始化為空)
status_text = st.sidebar.empty()
progress_bar = st.sidebar.empty()

st.sidebar.divider()

# --- 參數設定區 (整合) ---
with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=True):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA) 濾網", value=True)
    s_big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 3.0, 0.5) / 100
    s_min_vol = st.number_input("波段最小量 (張)", value=1000) * 1000

with st.sidebar.expander("⚡ 隔日沖策略參數 (短線)", expanded=True):
    d_period = st.slider("追蹤波段天數 (N)", 10, 120, 60, 5)
    d_threshold = st.slider("高點容許誤差 (%)", 0.0, 5.0, 1.0, 0.1)
    d_min_pct = st.slider("當日最低漲幅 (%)", 3.0, 9.0, 5.0, 0.1)
    d_min_vol = st.number_input("隔日沖最小量 (張)", value=1000, step=500)

st.sidebar.divider()
max_workers_input = st.sidebar.slider("系統效能 (執行緒數)", 1, 32, 8)

# 整合參數封包
params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 
    's_big_candle': s_big_candle, 's_min_vol': s_min_vol,
    'd_period': d_period, 'd_threshold': d_threshold, 
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

# --- 主畫面 Tab ---
tab1, tab2, tab3 = st.tabs(["🟢 狙擊手波段", "⚡ 隔日沖雷達", "📊 個股診斷"])

# 初始化 Session State 以儲存結果
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None

if start_scan:
    stock_map = get_stock_info_map()
    # 預設掃描全台股
    scan_codes = list(stock_map.keys())
    # 測試用：若要加速測試可限制數量，正式版請註解下一行
    # scan_codes = scan_codes[:100] 

    # 容器初始化
    sniper_triggered = []
    sniper_setup = []
    sniper_watching = []
    day_candidates = []

    # 顯示初始進度 (使用已建立的佔位元件)
    progress_bar.progress(0)
    
    total = len(scan_codes)
    done = 0
    status_text.text(f"啟動雙策略引擎... ({total} 檔)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params): code for code in scan_codes}
        
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 20 == 0 or done == total:
                progress_bar.progress(done / total)
                status_text.text(f"掃描中: {done}/{total}")
            
            res = future.result()
            if res:
                # 分類狙擊手結果
                if res['sniper']:
                    typ, data = res['sniper']
                    if typ == "triggered": sniper_triggered.append(data)
                    elif typ == "new_setup": sniper_setup.append(data)
                    elif typ == "watching": sniper_watching.append(data)
                
                # 分類隔日沖結果
                if res['day']:
                    day_candidates.append(res['day'])
    
    progress_bar.progress(1.0)
    status_text.success("掃描完成！")
    
    # 存入 Session State
    st.session_state['scan_results'] = {
        'sniper_triggered': sniper_triggered,
        'sniper_setup': sniper_setup,
        'sniper_watching': sniper_watching,
        'day_candidates': day_candidates
    }

# --- 顯示邏輯 ---
results = st.session_state['scan_results']

with tab1:
    st.header("🟢 狙擊手波段策略")
    st.caption(f"基準日: {analysis_date_str} | 策略：趨勢 + 實體長紅 + 型態確認 (強勢路徑 / 回檔路徑)")
    
    if results:
        # 分類 Triggered
        s_trig = results['sniper_triggered']
        trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]
        trig_n = [x for x in s_trig if "N字突破" in x['狀態']]
        
        # 分類 Watching
        s_watch = results['sniper_watching']
        watch_strong = [x for x in s_watch if "強勢整理" in x['狀態']]
        watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]
        
        # 顯示 Triggered
        if trig_strong or trig_n:
            st.markdown("### 🎯 買點觸發訊號 (Actionable)")
            if trig_strong:
                st.subheader(f"🚀 強勢突破 ({len(trig_strong)})")
                display_full_table(pd.DataFrame(trig_strong))
            if trig_n:
                st.subheader(f"🎯 N字突破 ({len(trig_n)})")
                display_full_table(pd.DataFrame(trig_n))
        
        # 顯示 Monitoring
        if results['sniper_setup'] or watch_strong or watch_pullback:
            if trig_strong or trig_n: st.divider()
            st.markdown("### 👀 市場潛力名單 (Monitoring)")
            
            if results['sniper_setup']:
                st.subheader(f"🔥 今日剛起漲 ({len(results['sniper_setup'])})")
                df_new = pd.DataFrame(results['sniper_setup'])
                if "族群" in df_new.columns:
                    sector_counts = df_new['族群'].value_counts().reset_index()
                    sector_counts.columns = ['族群', '數量']
                    top_sectors = [f"{row['族群']}({row['數量']})" for i, row in sector_counts.head(5).iterrows()]
                    st.success("📊 熱門族群: " + " | ".join(top_sectors))
                display_full_table(df_new)
            
            if watch_strong:
                st.subheader(f"💪 強勢整理 ({len(watch_strong)})")
                display_full_table(pd.DataFrame(watch_strong))
            
            if watch_pullback:
                st.subheader(f"📉 回檔整理 ({len(watch_pullback)})")
                display_full_table(pd.DataFrame(watch_pullback))
        
        if not (s_trig or results['sniper_setup'] or s_watch):
            st.info("今日無符合狙擊手策略之標的。")
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab2:
    st.header("⚡ 隔日沖雷達")
    st.caption(f"基準日: {analysis_date_str} | 策略：蓄勢待發 + 強勢動能 (> {d_min_pct}%) + 未漲停")
    
    if results:
        day_list = results['day_candidates']
        if day_list:
            df_day = pd.DataFrame(day_list)
            # 排序：距離高點越近 (數值越大，負數越接近0) 排前面
            df_day['sort_val'] = df_day['距離高點'].str.rstrip('%').astype(float)
            df_day = df_day.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
            
            st.subheader(f"⚡ 蓄勢待發清單 ({len(day_list)})")
            display_full_table(df_day)
        else:
            st.info("今日無符合隔日沖策略之標的。")
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab3:
    st.header("📊 個股 K 線診斷")
    col_in, col_btn = st.columns([3, 1])
    with col_in: stock_input = st.text_input("輸入代號", value="2330")
    with col_btn: diag_btn = st.button("診斷")
    
    if diag_btn:
        try:
            symbol = f"{stock_input}.TW"
            df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
            if df is None:
                symbol = f"{stock_input}.TWO"
                df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
            
            if df is not None:
                # 這裡僅作簡單 K 線圖展示，若要詳細回測可再擴充
                # 繪製 K 線 + MA
                df['MA_Trend'] = df['Close'].rolling(window=s_ma_trend).mean()
                df['MA_Base'] = df['Close'].rolling(window=20).mean()
                
                plot_df = df.tail(250) # 只畫最近一年
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Trend'], line=dict(color='blue'), name=f'{s_ma_trend}MA'), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Base'], line=dict(color='orange'), name='20MA'), row=1, col=1)
                fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量'), row=2, col=1)
                
                fig.update_layout(xaxis_rangeslider_visible=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("查無資料")
        except:
            st.error("發生錯誤")