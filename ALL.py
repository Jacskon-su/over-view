import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import importlib 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# ==========================================
# 🔥 匯入外部細產業資料庫 (強化版)
# ==========================================
SECTOR_DB = {}

try:
    import sector_data
    importlib.reload(sector_data) # 強制重載，確保編輯後生效
    
    # 🌟 關鍵修正：將所有鍵值轉為字串並去除空白，防止比對失敗
    if hasattr(sector_data, 'CUSTOM_SECTOR_MAP'):
        raw_map = sector_data.CUSTOM_SECTOR_MAP
        SECTOR_DB = {str(k).strip(): v for k, v in raw_map.items()}
    else:
        pass # 靜默處理
        
except ImportError:
    pass # 靜默處理
except Exception as e:
    st.error(f"❌ `sector_data.py` 載入錯誤: {e}")

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
                    self.defense_price = prev_close_setup
                else:
                    self.defense_price = self.data.Low[-1]

# ==========================================
# 🛠️ 輔助函式與資料庫
# ==========================================
def get_detailed_sector(code, standard_group=None, custom_db=None):
    """
    取得細分產業 (修正版)：
    1. 強制檢查 custom_db (細產業)
    2. 若無，則查詢官方分類
    """
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
    except: pass
    return "其他"

@st.cache_data(ttl=3600)
def get_stock_info_map():
    """取得上市櫃股票資訊表"""
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
# 🧠 綜合分析引擎 (含錯誤回報機制)
# ==========================================
def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db):
    try:
        df = get_stock_data_with_realtime(code, info['symbol'], analysis_date_str)
        if df is None or df.empty: return "無法取得資料 (yfinance/網路)"
        if len(df) < 250: return "資料長度不足 (<250天)"

        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values: return f"無 {analysis_date_str} 交易資料"
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

        # --- 策略 A: 狙擊手 ---
        s_ma_trend = params['s_ma_trend']
        s_use_year = params['s_use_year']
        s_big_candle = params['s_big_candle']
        s_min_vol = params['s_min_vol']

        ma_t = close.rolling(window=s_ma_trend).mean()
        ma_y = close.rolling(window=240).mean()
        vol_ma = volume.rolling(window=5).mean()

        is_sniper_candidate = True
        if volume.iloc[idx] < s_min_vol: is_sniper_candidate = False
        if s_use_year and close.iloc[idx] < ma_y.iloc[idx]: is_sniper_candidate = False
        if not (close.iloc[idx] > ma_t.iloc[idx] and ma_t.iloc[idx] > ma_t.iloc[idx-1]): is_sniper_candidate = False

        if is_sniper_candidate:
            is_setup = ((close.iloc[idx] - close.iloc[idx-1]) / close.iloc[idx-1] > s_big_candle and
                        volume.iloc[idx] > vol_ma.iloc[idx] and close.iloc[idx] > op.iloc[idx])
            
            setup_found = False
            s_high = 0; s_low = 0; s_close = 0; s_date = ""; setup_idx = -1
            defense_price = 0
            
            for k in range(1, 11):
                b_idx = idx - k
                if b_idx < 1: break
                
                if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > s_big_candle and
                    volume.iloc[b_idx] > vol_ma.iloc[b_idx] and close.iloc[b_idx] > op.iloc[b_idx]):
                    
                    setup_found = True; setup_idx = b_idx
                    s_low = low.iloc[b_idx]
                    s_high = high.iloc[b_idx]
                    s_close = close.iloc[b_idx]
                    s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    
                    prev_high_setup = high.iloc[b_idx-1]
                    prev_close_setup = close.iloc[b_idx-1]
                    
                    if s_low > prev_high_setup:
                        defense_price = prev_close_setup
                    else:
                        defense_price = s_low
                    break
            
            c_today = close.iloc[idx]
            prev_h = high.iloc[idx-1]
            
            if setup_found:
                is_broken = False; dropped_below_high = False
                for k in range(setup_idx + 1, idx + 1):
                    c_k = close.iloc[k]
                    if c_k < defense_price: is_broken = True; break
                    if c_k < s_high: dropped_below_high = True

                if not is_broken:
                    is_breakout = c_today > prev_h

                    if not dropped_below_high:
                        pct_from_setup = (c_today - s_close) / s_close
                        if pct_from_setup <= 0.10:
                            if is_breakout:
                                result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🚀 強勢突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}"})
                            else:
                                curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                                result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "💪 強勢整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"})
                    else:
                        if is_breakout:
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🎯 N字突破", "訊號日": s_date, "突破價": f"{prev_h:.2f}"})
                        else:
                            curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                            result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "📉 回檔整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"})
            
            elif is_setup:
                prev_c = close.iloc[idx-1]
                pct_chg = (c_today - prev_c) / prev_c * 100
                result_sniper = ("new_setup", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🔥 剛起漲", "漲幅": f"{pct_chg:+.2f}%"})

        # --- 策略 B: 隔日沖 ---
        d_period = params['d_period']
        d_threshold = params['d_threshold']
        d_min_vol = params['d_min_vol']
        d_min_pct = params['d_min_pct']

        d_close = close.iloc[idx]; d_open = op.iloc[idx]; d_high = high.iloc[idx]; d_volume = volume.iloc[idx]; d_prev_close = close.iloc[idx-1]
        
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
                    "代號": code,
                    "名稱": stock_name,
                    "收盤": f"{d_close:.2f}",
                    "產業": sector_name,
                    "漲幅": f"{(pct_chg_val*100):.2f}%",
                    "成交量": int(d_volume/1000),
                    "前波高點": f"{prev_period_high:.2f}",
                    "距離高點": f"{dist_to_high:+.2f}%",
                    "狀態": "⚡ 蓄勢待發"
                }

        # 成功回傳字典
        return {'sniper': result_sniper, 'day': result_day}

    except Exception as e: return f"程式執行錯誤: {str(e)}"

# 🔥 全展開表格顯示函式
def display_full_table(df):
    if df is not None and not df.empty:
        height = (len(df) * 35) + 38
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)
    else: st.info("無")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室")
st.sidebar.caption("波段與短線的極致整合")

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

start_scan = st.sidebar.button("🚀 開始全域掃描", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.empty()

st.sidebar.divider()

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

params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 
    's_big_candle': s_big_candle, 's_min_vol': s_min_vol,
    'd_period': d_period, 'd_threshold': d_threshold, 
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

tab1, tab2, tab3 = st.tabs(["🟢 狙擊手波段", "⚡ 隔日沖雷達", "📊 個股診斷"])

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None

if start_scan:
    stock_map = get_stock_info_map()
    scan_codes = list(stock_map.keys())

    sniper_triggered = []
    sniper_setup = []
    sniper_watching = []
    day_candidates = []
    failed_list = [] # 儲存失敗的股票

    progress_bar.progress(0)
    total = len(scan_codes)
    done = 0
    status_text.text(f"啟動雙策略引擎... ({total} 檔)")

    # 傳遞 SECTOR_DB 給分析函式
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB): code for code in scan_codes}
        
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 20 == 0 or done == total:
                progress_bar.progress(done / total)
                status_text.text(f"掃描中: {done}/{total}")
            
            res = future.result()
            
            # 判斷回傳是否為字典 (成功) 或是字串 (失敗)
            if isinstance(res, dict):
                if res['sniper']:
                    typ, data = res['sniper']
                    if typ == "triggered": sniper_triggered.append(data)
                    elif typ == "new_setup": sniper_setup.append(data)
                    elif typ == "watching": sniper_watching.append(data)
                
                if res['day']:
                    day_candidates.append(res['day'])
            else:
                # 若為字串，則視為錯誤訊息
                current_code = futures[future]
                stock_name = stock_map[current_code]['short_name']
                reason = res if isinstance(res, str) else "未知錯誤"
                failed_list.append(f"{current_code} {stock_name} : {reason}")
    
    progress_bar.progress(1.0)
    status_text.success(f"掃描完成！ (成功: {total - len(failed_list)} / 失敗: {len(failed_list)})")
    
    st.session_state['scan_results'] = {
        'sniper_triggered': sniper_triggered,
        'sniper_setup': sniper_setup,
        'sniper_watching': sniper_watching,
        'day_candidates': day_candidates,
        'failed_list': failed_list
    }

results = st.session_state['scan_results']

with tab1:
    st.header("🟢 狙擊手波段策略")
    st.caption(f"基準日: {analysis_date_str} | 策略：趨勢 + 實體長紅 + 型態確認 (強勢路徑 / 回檔路徑)")
    
    if results:
        # 顯示失敗清單
        if 'failed_list' in results and results['failed_list']:
             with st.expander(f"⚠️ 掃描失敗/無資料清單 ({len(results['failed_list'])})"):
                st.write(", ".join(results['failed_list']))
                
        s_trig = results['sniper_triggered']
        trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]
        trig_n = [x for x in s_trig if "N字突破" in x['狀態']]
        
        s_watch = results['sniper_watching']
        watch_strong = [x for x in s_watch if "強勢整理" in x['狀態']]
        watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]
        
        if trig_strong or trig_n:
            st.markdown("### 🎯 買點觸發訊號 (Actionable)")
            if trig_strong:
                st.markdown(f"### 🚀 強勢突破 ({len(trig_strong)})") 
                display_full_table(pd.DataFrame(trig_strong))
            if trig_n:
                st.markdown(f"### 🎯 N字突破 ({len(trig_n)})")
                display_full_table(pd.DataFrame(trig_n))
        
        if results['sniper_setup'] or watch_strong or watch_pullback:
            if trig_strong or trig_n: st.divider()
            st.markdown("### 👀 市場潛力名單 (Monitoring)")
            
            if results['sniper_setup']:
                st.markdown(f"### 🔥 今日剛起漲 ({len(results['sniper_setup'])})")
                display_full_table(pd.DataFrame(results['sniper_setup']))
            
            if watch_strong:
                st.markdown(f"### 💪 強勢整理 ({len(watch_strong)})")
                display_full_table(pd.DataFrame(watch_strong))
            
            if watch_pullback:
                st.markdown(f"### 📉 回檔整理 ({len(watch_pullback)})")
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
            df_day['sort_val'] = df_day['距離高點'].str.rstrip('%').astype(float)
            df_day = df_day.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
            
            st.markdown(f"### ⚡ 蓄勢待發清單 ({len(day_list)})")
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
                df['MA_Trend'] = df['Close'].rolling(window=s_ma_trend).mean()
                df['MA_Base'] = df['Close'].rolling(window=20).mean()
                plot_df = df.tail(250)
                
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