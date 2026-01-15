import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import importlib 
import os
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 🔧 設定
# ==========================================
DATA_DIR = "stock_data"
ZIP_FILE = "stock_data.zip"

st.set_page_config(page_title="強勢股戰情室", page_icon="🔥", layout="wide")
warnings.filterwarnings("ignore")

# 嘗試載入細產業資料
SECTOR_DB = {}
try:
    import sector_data
    if hasattr(sector_data, 'CUSTOM_SECTOR_MAP'):
        SECTOR_DB = {str(k).strip(): v for k, v in sector_data.CUSTOM_SECTOR_MAP.items()}
except: pass

# ==========================================
# 📦 雲端自動解壓縮 (只在啟動時執行一次)
# ==========================================
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 檢查是否需要解壓 (如果資料夾是空的，且有 zip 檔)
if len(os.listdir(DATA_DIR)) < 100 and os.path.exists(ZIP_FILE):
    with st.spinner("📦 正在還原雲端資料庫..."):
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.toast(f"✅ 資料庫還原成功！", icon="📂")
        except Exception as e:
            st.error(f"解壓縮失敗: {e}")

# ==========================================
# 🛠️ 輔助函式
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_info_map():
    try:
        import twstock
        stock_map = {}
        for c, i in twstock.twse.items():
            if len(c) == 4: stock_map[c] = {'name': f"{c} {i.name}", 'symbol': f"{c}.TW", 'short_name': i.name, 'group': getattr(i, 'group', '其他')}
        for c, i in twstock.tpex.items():
            if len(c) == 4: stock_map[c] = {'name': f"{c} {i.name}", 'symbol': f"{c}.TWO", 'short_name': i.name, 'group': getattr(i, 'group', '其他')}
        return stock_map
    except: return {}

def get_detailed_sector(code, standard_group=None):
    if code in SECTOR_DB: return SECTOR_DB[code]
    return standard_group if standard_group else "其他"

def load_data(code):
    """讀取本地 CSV"""
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            return df
        except: return None
    return None

# ==========================================
# 🧠 核心分析引擎 (含即時盤補丁)
# ==========================================
def analyze_stock(code, info, params, fetch_realtime=False):
    # 1. 讀取歷史資料
    df = load_data(code)
    if df is None or len(df) < 200: return None # 資料不足

    # 2. 盤中即時補丁 (Real-time Patch)
    # 如果使用者勾選「補齊今日資料」，則嘗試抓取最新報價並合併
    if fetch_realtime:
        try:
            # 抓取最近 2 天 (確保包含今天)
            rt_df = yf.download(info['symbol'], period="5d", interval="1d", progress=False, auto_adjust=True)
            if not rt_df.empty:
                if rt_df.index.tz is not None: rt_df.index = rt_df.index.tz_localize(None)
                # 合併：用新的覆蓋舊的
                df = pd.concat([df, rt_df])
                df = df[~df.index.duplicated(keep='last')]
        except: pass

    # 3. 準備數據
    close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']; op = df['Open']
    # 取得最後一天 (可能是今天盤中，也可能是昨天收盤)
    idx = -1 
    current_date = df.index[idx].strftime('%Y-%m-%d')
    
    # 排除無量
    if volume.iloc[idx] < params['s_min_vol']: return None

    # --- 策略運算 ---
    # 均線
    ma_trend = close.rolling(window=params['s_ma_trend']).mean()
    ma_long = close.rolling(window=240).mean()
    vol_ma = volume.rolling(window=5).mean()

    # 趨勢濾網
    is_trend_up = (close.iloc[idx] > ma_trend.iloc[idx]) and (ma_trend.iloc[idx] > ma_trend.iloc[idx-1])
    if params['s_use_year'] and (close.iloc[idx] < ma_long.iloc[idx]): is_trend_up = False
    
    if not is_trend_up: return None

    # 尋找 Setup Bar (回溯 10 天)
    setup_found = False
    s_high = 0; s_low = 0; s_close = 0; s_date = ""; setup_idx = -1
    defense_price = 0
    
    for k in range(2, 12): # 從昨天往前推 (idx-1 是昨天)
        b_idx = idx - k + 1
        if b_idx < 0: break
        
        # 條件：漲幅 > 3% 且 爆量 且 收紅
        prev_c = close.iloc[b_idx-1]
        is_big = (close.iloc[b_idx] - prev_c) / prev_c > params['s_big_candle']
        is_vol = volume.iloc[b_idx] > vol_ma.iloc[b_idx]
        is_red = close.iloc[b_idx] > op.iloc[b_idx]
        
        if is_big and is_vol and is_red:
            setup_found = True
            s_high = high.iloc[b_idx]
            s_low = low.iloc[b_idx]
            s_close = close.iloc[b_idx]
            s_date = df.index[b_idx].strftime('%Y-%m-%d')
            setup_idx = b_idx
            
            # 跳空判斷 (設定防守價)
            prev_high_setup = high.iloc[b_idx-1]
            prev_close_setup = close.iloc[b_idx-1]
            if s_low > prev_high_setup:
                defense_price = prev_close_setup # 守缺口
            else:
                defense_price = s_low # 守低點
            break
    
    result_sniper = None
    result_day = None
    sector_name = get_detailed_sector(code, info.get('group'))

    # --- 策略 A: 狙擊手 ---
    if setup_found:
        # 檢查是否破防守
        is_broken = False
        for k in range(setup_idx + 1, len(df)):
            if close.iloc[k] < defense_price: 
                is_broken = True; break
        
        if not is_broken:
            # 歷史回溯：檢查 Setup 後到昨天為止，是否曾經突破過
            has_broken_before = False
            # 範圍：Setup隔天 ~ 昨天 (idx-1)
            for k in range(setup_idx + 1, len(df) - 1):
                if close.iloc[k] > high.iloc[k-1]:
                    has_broken_before = True; break
            
            # 今日數據
            c_today = close.iloc[idx]
            prev_h = high.iloc[idx-1]
            today_open = op.iloc[idx]
            
            # 判定今日是否突破
            is_today_breakout = c_today > prev_h
            
            # 跳空檢查 (開盤 > 昨高) 且 (收紅 - 避免開高走低)
            is_gap = (today_open > prev_h) and (c_today > today_open)
            gap_tag = "🚀 跳空" if is_gap else "🎯 "
            reentry_tag = "🚀 跳空" if is_gap else "🚀 "
            
            if is_today_breakout:
                if not has_broken_before:
                    # 第一次突破
                    status = f"{gap_tag}N字突破"
                    result_sniper = {"狀態": status, "日期": current_date}
                else:
                    # 曾突破過，檢查昨天狀態
                    yesterday_c = close.iloc[idx-1]
                    yesterday_prev_h = high.iloc[idx-2]
                    was_strong = yesterday_c > yesterday_prev_h
                    
                    if not was_strong:
                        # 昨天弱，今天強 -> 續漲 (回馬槍)
                        status = f"{reentry_tag}N字續漲"
                        result_sniper = {"狀態": status, "日期": current_date}
                    else:
                        # 昨天強，今天強 -> 強勢續漲
                        result_sniper = {"狀態": "🔥 強勢續漲", "日期": current_date}
            else:
                # 沒突破 -> 觀察名單
                # 跌破 Setup 最高價視為回檔
                state_str = "📉 回檔整理" if c_today < s_high else "💪 強勢整理"
                curr_pct = (c_today - close.iloc[idx-1]) / close.iloc[idx-1]
                result_sniper = {"狀態": state_str, "日期": current_date, "漲幅": f"{curr_pct:.2%}"}

    # 剛起漲偵測
    elif idx > 0:
        prev_c = close.iloc[idx-1]
        is_big = (close.iloc[idx] - prev_c) / prev_c > params['s_big_candle']
        is_vol = volume.iloc[idx] > vol_ma.iloc[idx]
        is_red = close.iloc[idx] > op.iloc[idx]
        if is_big and is_vol and is_red:
            # 檢查跳空 (防開高走低)
            is_gap_start = (low.iloc[idx] > high.iloc[idx-1]) and (close.iloc[idx] > op.iloc[idx])
            status = "🚀 跳空起漲" if is_gap_start else "🔥 剛起漲"
            pct = (close.iloc[idx] - prev_c) / prev_c
            result_sniper = {"狀態": status, "日期": current_date, "漲幅": f"{pct:.2%}"}

    # --- 策略 B: 隔日沖 ---
    # (保留原邏輯: 收紅, 上影線短, 漲幅3~9.5%, 近前高)
    if result_sniper is None: # 簡化: 若符合狙擊手就不重複報
        d_close = close.iloc[idx]
        d_open = op.iloc[idx]
        d_high = high.iloc[idx]
        upper_shadow = (d_high - d_close) / d_close
        pct_val = (d_close - close.iloc[idx-1]) / close.iloc[idx-1]
        
        is_red = d_close > d_open
        is_strong_close = upper_shadow < 0.01
        is_momentum = 0.03 < pct_val < 0.095
        
        if is_red and is_strong_close and is_momentum:
            # 檢查是否近前高 (60日)
            past_60_high = high.iloc[idx-60:idx].max()
            if d_close >= past_60_high * 0.98 and d_high <= past_60_high: # 逼近但未過
                 dist = (d_close - past_60_high) / past_60_high
                 result_day = {
                     "狀態": "⚡ 蓄勢待發", "距離前高": f"{dist:.2%}", 
                     "日期": current_date
                 }

    # 整合回傳
    final_res = {}
    if result_sniper:
        final_res['sniper'] = result_sniper
        final_res['sniper'].update({"代號": code, "名稱": stock_name, "收盤": f"{close.iloc[idx]:.2f}", "產業": sector_name})
    if result_day:
        final_res['day'] = result_day
        final_res['day'].update({"代號": code, "名稱": stock_name, "收盤": f"{close.iloc[idx]:.2f}", "產業": sector_name})
        
    return final_res if final_res else None

# ==========================================
# 🖥️ 介面
# ==========================================
st.sidebar.title("🔥 強勢股戰情室")
st.sidebar.caption("Github 雲端部署版")

# 參數設定
with st.sidebar.expander("策略參數", expanded=False):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA)", value=True)
    s_big_candle = st.slider("長紅漲幅 (%)", 0.03, 0.1, 0.03)
    s_min_vol = st.number_input("最小量 (張)", value=1000) * 1000
    
    d_period = 60
    d_threshold = 1.0
    d_min_pct = 3.0
    d_min_vol = 1000

params = {
    's_ma_trend': s_ma_trend, 's_use_year': s_use_year, 
    's_big_candle': s_big_candle, 's_min_vol': s_min_vol,
    'd_period': d_period, 'd_threshold': d_threshold, 
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

# 按鈕區
col_btn1, col_btn2 = st.columns([1,2])
with col_btn1:
    fetch_realtime = st.checkbox("盤中即時補資料", value=True, help="勾選後，掃描時會嘗試抓取每檔股票的當日最新報價合併計算。")
with col_btn2:
    start_scan = st.button("🚀 執行策略掃描", type="primary")

# 顯示資料庫狀態
file_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
st.info(f"📚 資料庫狀態：{file_count} 檔 (來自 ZIP)")

if start_scan:
    if file_count < 100:
        st.error("⚠️ 資料庫為空！請確認 GitHub 上傳了 stock_data.zip。")
    else:
        stock_map = get_stock_info_map()
        # 1. 讀取現有檔案列表
        db_files = [f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        # 2. 取交集 (確保有資料且在清單內)
        scan_codes = list(set(db_files) & set(stock_map.keys()))
        scan_codes.sort()
        
        results_s = []
        results_d = []
        
        bar = st.progress(0)
        status = st.empty()
        
        # 平行運算
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(analyze_stock, code, stock_map[code], params, fetch_realtime): code for code in scan_codes}
            
            total = len(scan_codes); done = 0
            for future in concurrent.futures.as_completed(futures):
                done += 1
                if done % 50 == 0: 
                    bar.progress(done / total)
                    status.text(f"掃描中... {done}/{total}")
                
                res = future.result()
                if res:
                    if 'sniper' in res: results_s.append(res['sniper'])
                    if 'day' in res: results_d.append(res['day'])
        
        bar.progress(1.0)
        status.success("掃描完成！")
        
        # --- 顯示結果 ---
        tab1, tab2 = st.tabs(["🟢 波段策略", "⚡ 隔日沖"])
        
        with tab1:
            if results_s:
                df_s = pd.DataFrame(results_s)
                # 分類顯示
                for status_key in ["N字突破", "N字續漲", "強勢續漲", "剛起漲", "強勢整理", "回檔整理"]:
                    # 過濾包含該關鍵字的狀態
                    df_part = df_s[df_s['狀態'].str.contains(status_key)]
                    if not df_part.empty:
                        st.subheader(f"{status_key} 清單 ({len(df_part)})")
                        st.dataframe(df_part, hide_index=True, use_container_width=True)
            else:
                st.info("無符合標的")
                
        with tab2:
            if results_d:
                df_d = pd.DataFrame(results_d)
                st.dataframe(df_d, hide_index=True, use_container_width=True)
            else:
                st.info("無符合標的")