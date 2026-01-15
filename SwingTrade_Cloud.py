import streamlit as st
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import importlib 
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# ==========================================
# 🔧 設定與全域變數
# ==========================================
DATA_DIR = "stock_data"  # 本地資料庫資料夾名稱 (對應 Fly.io Volume)

# 確保資料夾存在 (雖然不下載，但讀取時需要此路徑)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ==========================================
# 🔥 匯入外部細產業資料庫
# ==========================================
SECTOR_DB = {}
try:
    import sector_data
    importlib.reload(sector_data)
    if hasattr(sector_data, 'CUSTOM_SECTOR_MAP'):
        raw_map = sector_data.CUSTOM_SECTOR_MAP
        SECTOR_DB = {str(k).strip(): v for k, v in raw_map.items()}
except: pass

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(
    page_title="強勢股戰情室 (純掃描版)",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import twstock
except ImportError:
    st.error("❌ 缺少 `twstock` 套件，請輸入 `pip install twstock` 安裝")
    st.stop()

st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ 輔助函式 (資料庫核心)
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_info_map():
    try:
        stock_map = {}
        for code, info in twstock.twse.items():
            if len(code) == 4: 
                stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TW", 'short_name': info.name, 'group': getattr(info, 'group', '其他')}
        for code, info in twstock.tpex.items():
            if len(code) == 4: 
                stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TWO", 'short_name': info.name, 'group': getattr(info, 'group', '其他')}
        return stock_map
    except: return {}

def get_detailed_sector(code, standard_group=None, custom_db=None):
    code_str = str(code).strip() 
    if custom_db and code_str in custom_db: return str(custom_db[code_str])
    if standard_group and str(standard_group) not in ['nan', 'None', '', 'NaN']: return str(standard_group)
    return "其他"

def load_dataframe(code):
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        except: return None
    return None

def get_database_status():
    """取得資料庫的最新日期狀態"""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files: return "無資料", 0
    
    target_file = "2330.csv" if "2330.csv" in files else random.choice(files)
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, target_file), index_col=0)
        last_date = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d')
        return last_date, len(files)
    except:
        return "讀取錯誤", len(files)

# ==========================================
# 🧠 策略核心
# ==========================================
def SMA(array, n):
    return pd.Series(array).rolling(window=n).mean()

def analyze_from_local(code, info, analysis_date_str, params, custom_sector_db):
    try:
        df = load_dataframe(code)
        
        if df is None or df.empty: return "無本地資料"
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values: 
            last_dt = df.index[-1]
            target_dt = pd.Timestamp(analysis_date_str)
            if (target_dt - last_dt).days > 5:
                return f"資料過舊 (最後: {last_dt.strftime('%Y-%m-%d')})"
            else:
                if analysis_date_str not in df['DateStr'].values: return f"無 {analysis_date_str} 資料"

        if len(df) < 250: return "長度不足"
        
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
        stock_name = info['short_name']
        sector_name = get_detailed_sector(code, standard_group=info.get('group'), custom_db=custom_sector_db)

        result_sniper = None
        result_day = None

        # --- 策略 A: 狙擊手 (最終完整版) ---
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
                    s_low = low.iloc[b_idx]; s_high = high.iloc[b_idx]; s_close = close.iloc[b_idx]
                    s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    
                    prev_high_setup = high.iloc[b_idx-1]; prev_close_setup = close.iloc[b_idx-1]
                    
                    if s_low > prev_high_setup:
                        defense_price = prev_close_setup 
                    else:
                        defense_price = s_low 
                    break
            
            c_today = close.iloc[idx]
            prev_h = high.iloc[idx-1] 
            
            if setup_found:
                is_broken_defense = False
                for k in range(setup_idx + 1, idx + 1):
                    if close.iloc[k] < defense_price: 
                        is_broken_defense = True; break
                
                if not is_broken_defense:
                    has_broken_before = False
                    for k in range(setup_idx + 1, idx): 
                        if close.iloc[k] > high.iloc[k-1]:
                            has_broken_before = True
                            break
                    
                    # 🔥 N字突破判斷
                    is_today_breakout = c_today > prev_h
                    
                    # 🔥 跳空且收紅檢查 (開高走低會被降級)
                    is_valid_gap = (op.iloc[idx] > prev_h) and (c_today > op.iloc[idx])
                    gap_str = "🚀 跳空" if is_valid_gap else "🎯 "
                    reentry_gap_str = "🚀 跳空" if is_valid_gap else "🚀 "
                    
                    if is_today_breakout:
                        if not has_broken_before:
                            status_str = f"{gap_str}N字突破"
                            result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": status_str, "訊號日": s_date, "突破價": f"{prev_h:.2f}"})
                        else:
                            was_yesterday_strong = close.iloc[idx-1] > high.iloc[idx-2]
                            if not was_yesterday_strong:
                                status_str = f"{reentry_gap_str}N字續漲"
                                result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": status_str, "訊號日": s_date, "突破價": f"{prev_h:.2f}"})
                            else:
                                result_sniper = ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "🔥 強勢續漲", "訊號日": s_date, "突破價": f"{prev_h:.2f}"})
                    else:
                        curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                        
                        # 修正：跌破 Setup 最高價即視為回檔
                        if c_today < s_high:
                             result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "📉 回檔整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"})
                        else:
                             result_sniper = ("watching", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": "💪 強勢整理", "訊號日": s_date, "防守": f"{defense_price:.2f}", "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"})

            elif is_setup:
                prev_c = close.iloc[idx-1]
                pct_chg = (c_today - prev_c) / prev_c * 100
                # 🔥 剛起漲的跳空顯示 (也加上防開高走低)
                is_gap_start = (low.iloc[idx] > high.iloc[idx-1]) and (c_today > op.iloc[idx])
                status_str = "🚀 跳空起漲" if is_gap_start else "🔥 剛起漲"
                result_sniper = ("new_setup", {"代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}", "產業": sector_name, "狀態": status_str, "漲幅": f"{pct_chg:+.2f}%"})

        # --- 策略 B: 隔日沖 ---
        d_period = params['d_period']; d_threshold = params['d_threshold']
        d_min_vol = params['d_min_vol']; d_min_pct = params['d_min_pct']
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
                    "代號": code, "名稱": stock_name, "收盤": f"{d_close:.2f}", "產業": sector_name,
                    "漲幅": f"{(pct_chg_val*100):.2f}%", "成交量": int(d_volume/1000),
                    "前波高點": f"{prev_period_high:.2f}", "距離高點": f"{dist_to_high:+.2f}%", "狀態": "⚡ 蓄勢待發"
                }

        return {'sniper': result_sniper, 'day': result_day}

    except Exception as e: return f"Error: {str(e)}"

# 🔥 全展開表格顯示
def display_full_table(df):
    if df is not None and not df.empty:
        height = (len(df) + 1) * 35
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)
    else: st.info("無")

# ==========================================
# 🖥️ 介面主程式
# ==========================================
st.sidebar.title("🔥 強勢股戰情室")
st.sidebar.caption("純掃描版 (需外部資料源)")

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

stock_map = get_stock_info_map()
db_last_date, db_file_count = get_database_status()

st.sidebar.divider()
st.sidebar.markdown(f"**📦 資料庫狀態**")
st.sidebar.text(f"檔案數: {db_file_count} 檔")
st.sidebar.text(f"最新資料: {db_last_date}")

if db_last_date != "無資料" and db_last_date < analysis_date_str:
    st.sidebar.warning(f"⚠️ 資料庫滯後 (最新: {db_last_date})")
elif db_last_date == analysis_date_str:
    st.sidebar.success("✅ 資料庫已是最新")

start_scan = st.sidebar.button("🚀 執行策略掃描", type="primary")
scan_status = st.sidebar.empty()
scan_progress = st.sidebar.empty()

st.sidebar.divider()

with st.sidebar.expander("🟢 狙擊手策略參數", expanded=True):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA)", value=True)
    s_big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 3.0, 0.5) / 100
    s_min_vol = st.number_input("波段最小量 (張)", value=1000) * 1000

with st.sidebar.expander("⚡ 隔日沖策略參數", expanded=False):
    d_period = st.slider("追蹤波段 (N)", 10, 120, 60, 5)
    d_threshold = st.slider("高點誤差 (%)", 0.0, 5.0, 1.0, 0.1)
    d_min_pct = st.slider("最低漲幅 (%)", 3.0, 9.0, 5.0, 0.1)
    d_min_vol = st.number_input("隔日沖最小量", value=1000, step=500)

params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 
    's_big_candle': s_big_candle, 's_min_vol': s_min_vol,
    'd_period': d_period, 'd_threshold': d_threshold, 
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

tab1, tab2, tab3 = st.tabs(["🟢 波段策略", "⚡ 短線雷達", "📊 個股診斷"])

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None

if start_scan:
    if db_file_count < 100:
        st.error("⚠️ 資料庫為空！請確認 `stock_data` 資料夾內有數據。")
    else:
        # 只掃描資料庫中存在的檔案
        db_files = [f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        scan_codes = list(set(db_files) & set(stock_map.keys()))
        scan_codes.sort()
        
        excluded_count = len(stock_map) - len(scan_codes)
        if excluded_count > 0:
            st.toast(f"ℹ️ 已自動排除 {excluded_count} 檔無資料/下市股票", icon="🗑️")

        sniper_triggered = []; sniper_setup = []; sniper_watching = []; day_candidates = []; failed_list = []

        scan_status.text(f"讀取硬碟資料並運算中... (共 {len(scan_codes)} 檔)")
        scan_progress.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(analyze_from_local, code, stock_map[code], analysis_date_str, params, SECTOR_DB): code for code in scan_codes}
            
            total = len(scan_codes); done = 0
            for future in concurrent.futures.as_completed(futures):
                done += 1
                if done % 100 == 0: scan_progress.progress(done / total)
                
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
                    stock_name = stock_map[current_code]['short_name']
                    failed_list.append(f"{current_code} {stock_name} : {res}")

        scan_progress.progress(1.0)
        scan_status.success(f"掃描完成！ (成功: {len(scan_codes) - len(failed_list)} / 失敗: {len(failed_list)})")
        
        st.session_state['scan_results'] = {
            'sniper_triggered': sniper_triggered,
            'sniper_setup': sniper_setup,
            'sniper_watching': sniper_watching,
            'day_candidates': day_candidates,
            'failed_list': failed_list
        }

results = st.session_state['scan_results']

with tab1:
    st.header("🟢 狙擊手波段")
    if results:
        if results['failed_list']:
             with st.expander(f"⚠️ 掃描失敗/無資料清單 ({len(results['failed_list'])})", expanded=False):
                st.write(", ".join(results['failed_list']))
        
        s_trig = results['sniper_triggered']
        
        trig_strong = [x for x in s_trig if "N字突破" in x['狀態']]
        trig_reentry = [x for x in s_trig if "N字續漲" in x['狀態']]
        trig_continue = [x for x in s_trig if "強勢續漲" in x['狀態']]

        if trig_strong: 
            st.markdown(f"### 🎯 N字突破 ({len(trig_strong)})")
            display_full_table(pd.DataFrame(trig_strong))
        
        if trig_reentry: 
            st.markdown(f"### 🚀 N字續漲 (回測再攻) ({len(trig_reentry)})")
            display_full_table(pd.DataFrame(trig_reentry))
            
        if trig_continue: 
            st.markdown(f"### 🔥 強勢續漲 (持倉) ({len(trig_continue)})")
            display_full_table(pd.DataFrame(trig_continue))
        
        s_setup = results['sniper_setup']
        if s_setup:
            st.markdown(f"### 🔥 剛起漲 ({len(s_setup)})")
            display_full_table(pd.DataFrame(s_setup))
            
        s_watch = results['sniper_watching']
        if s_watch:
            watch_strong = [x for x in s_watch if "強勢整理" in x['狀態']]
            watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]

            if watch_strong:
                st.markdown(f"### 💪 強勢整理 ({len(watch_strong)})")
                display_full_table(pd.DataFrame(watch_strong))
            
            if watch_pullback:
                st.markdown(f"### 📉 回檔整理 ({len(watch_pullback)})")
                display_full_table(pd.DataFrame(watch_pullback))

with tab2:
    st.header("⚡ 隔日沖雷達")
    if results and results['day_candidates']:
        df_day = pd.DataFrame(results['day_candidates'])
        df_day['sort_val'] = df_day['距離高點'].str.rstrip('%').astype(float)
        df_day = df_day.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
        st.markdown(f"### ⚡ 蓄勢待發 ({len(df_day)})")
        display_full_table(df_day)

with tab3:
    st.header("📊 個股 K 線診斷")
    col_in, col_btn = st.columns([3, 1])
    with col_in: stock_input = st.text_input("輸入代號", value="2330")
    with col_btn: diag_btn = st.button("診斷")
    
    if diag_btn:
        df = load_dataframe(stock_input)
        if df is None:
             st.warning("⚠️ 本地無資料，無法診斷")
        
        if df is not None and not df.empty:
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df['MA_Trend'] = df['Close'].rolling(window=s_ma_trend).mean()
            df['MA_Base'] = df['Close'].rolling(window=20).mean()
            plot_df = df.tail(250)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Trend'], line=dict(color='blue'), name=f'{s_ma_trend}MA'), row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='成交量'), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            if df is not None: st.error("查無資料")