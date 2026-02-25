import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import importlib
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# ==========================================
# 📋 日誌設定 (修改項目 6：改善錯誤處理)
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
    page_title="強勢股戰情室",
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
# 🕐 交易時段判斷 (修改項目 2：動態快取)
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
    """
    動態決定快取時間：
    - 交易時段中：300秒 (5分鐘)，讓資料更新更頻繁
    - 盤後/盤前：3600秒 (1小時)，節省流量
    """
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

# 修改項目 2：動態 TTL 快取，交易時段 5 分鐘，盤後 1 小時
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
# 🚀 批量下載加速模組
# ==========================================
def fetch_data_batch(stock_map, period="1y", chunk_size=100):
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
                    # 單一股票時 yfinance 不回傳 MultiIndex，直接使用
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
    realtime_data = {}
    progress_text = st.empty()

    for i in range(0, len(codes_list), chunk_size):
        chunk = codes_list[i:i + chunk_size]
        progress_text.text(f"⚡ 正在批量更新即時盤... ({i}/{len(codes_list)})")
        try:
            stocks = twstock.realtime.get(chunk)
            if stocks:
                if 'success' in stocks:
                    if stocks['success']:
                        realtime_data[stocks['info']['code']] = stocks['realtime']
                else:
                    for code, data in stocks.items():
                        if data['success']:
                            realtime_data[code] = data['realtime']
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"fetch_realtime_batch 錯誤 (offset {i}): {e}")
    progress_text.empty()
    return realtime_data

# ==========================================
# 🧠 綜合分析引擎
# ==========================================
def analyze_combined_strategy(code, info, analysis_date_str, params, custom_sector_db, pre_loaded_df=None):
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
        if s_use_year and len(ma_y) > idx and (pd.isna(ma_y.iloc[idx]) or close.iloc[idx] < ma_y.iloc[idx]): is_sniper_candidate = False
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
                    prev_open_setup = op.iloc[b_idx-1]

                    if s_low > prev_high_setup:
                        # 跳空情況：用前一天實體上緣定義缺口下緣（排除上影線干擾）
                        if prev_close_setup >= prev_open_setup:  # 前一天是紅K
                            base_val = prev_close_setup
                        else:  # 前一天是黑K
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

                if not is_broken:
                    is_breakout = c_today > prev_h
                    is_gap_breakout = (op.iloc[idx] > high.iloc[idx-1]) and (close.iloc[idx] > op.iloc[idx])

                    if not dropped_below_high:
                        pct_from_setup = (c_today - s_close) / s_close
                        if pct_from_setup <= 0.10:
                            if is_breakout:
                                result_sniper = ("triggered", {
                                    "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                    "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                    "狀態": "🚀 強勢突破", "訊號日": s_date,
                                    "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}",
                                    "sort_pct": daily_pct,
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
                            result_sniper = ("triggered", {
                                "代號": code, "名稱": stock_name, "收盤": f"{c_today:.2f}",
                                "漲幅": f"{daily_pct:+.2f}%", "產業": sector_name,
                                "狀態": status_str, "訊號日": s_date,
                                "突破價": f"{prev_h:.2f}", "防守價": f"{defense_price:.2f}",
                                "sort_pct": daily_pct,
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

        # --- 策略 B: 隔日沖 ---
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
        # 過濾掉內部用的 _ 開頭欄位
        display_cols = [c for c in df.columns if not c.startswith('_')]
        display_df = df[display_cols]
        height = (len(display_df) * 35) + 38
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=height)
    else:
        st.info("無")

# ==========================================
# 📊 個股診斷強化版 (修改項目 3：標記訊號 + 修改項目 4：回測整合)
# ==========================================
def run_diagnosis(stock_input, analysis_date_str, params):
    """執行個股診斷，回傳 (df, symbol, info_dict) 或 None"""
    symbol = f"{stock_input}.TW"
    df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
    if df is None:
        symbol = f"{stock_input}.TWO"
        df = get_stock_data_with_realtime(stock_input, symbol, analysis_date_str)
    return df, symbol

def plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info=None):
    """
    修改項目 3：強化個股診斷圖表
    - 標記訊號日 (長紅K那天)
    - 標記防守價位線
    - 顯示策略狀態標籤
    """
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

    # K線
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name='K線', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)

    # 均線
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

    # 修改項目 3：標記訊號日與防守價
    if sniper_info:
        setup_date = sniper_info.get('_setup_date')
        defense_price = sniper_info.get('_defense')
        signal_high = sniper_info.get('_signal_high')
        status = sniper_info.get('狀態', '')

        # 標記訊號日垂直線
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

        # 標記防守價水平線
        if defense_price and defense_price > 0:
            fig.add_hline(
                y=defense_price,
                line_width=2, line_dash="dot", line_color="red",
                annotation_text=f"🛡️ 防守 {defense_price:.2f}",
                annotation_position="bottom right"
            )

        # 標記長紅K最高點
        if signal_high and signal_high > 0:
            fig.add_hline(
                y=signal_high,
                line_width=1.5, line_dash="dot", line_color="orange",
                annotation_text=f"⚡ 長紅高 {signal_high:.2f}",
                annotation_position="top right"
            )

        # 策略狀態標籤
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

    # 成交量
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
    """
    修改項目 4：回測功能接入 UI
    執行 SniperStrategy 回測並顯示績效報告
    """
    try:
        bt_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
        bt_df.index = pd.to_datetime(bt_df.index)

        bt = Backtest(
            bt_df,
            SniperStrategy,
            cash=100000,
            commission=0.001425 * 2,  # 台股手續費雙邊
            trade_on_close=False
        )

        # 套用側邊欄參數
        stats = bt.run(
            ma_trend_period=params['s_ma_trend'],
            big_candle_pct=params['s_big_candle'],
            min_volume_shares=params['s_min_vol'],
            use_year_line=params['s_use_year']
        )

        # 顯示關鍵指標
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

        # 顯示交易明細
        trades = stats['_trades']
        if not trades.empty:
            st.markdown("#### 📋 交易明細")
            trades_display = trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']].copy()
            trades_display.columns = ['進場時間', '出場時間', '進場價', '出場價', '損益', '報酬%']
            trades_display['報酬%'] = (trades_display['報酬%'] * 100).round(2)
            trades_display['損益'] = trades_display['損益'].round(2)
            st.dataframe(trades_display, hide_index=True, use_container_width=True)

        # 權益曲線
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
st.sidebar.title("🔥 強勢股戰情室")
st.sidebar.caption("波段與短線的極致整合")

# 修改項目 2：顯示目前快取模式
cache_mode = "🟢 盤中模式 (5分鐘快取)" if is_trading_hours() else "🔵 盤後模式 (1小時快取)"
st.sidebar.caption(cache_mode)

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

start_scan = st.sidebar.button("🚀 開始全域掃描 (極速版)", type="primary")
status_text = st.sidebar.empty()
progress_bar = st.sidebar.empty()

st.sidebar.divider()

with st.sidebar.expander("🟢 狙擊手策略參數 (波段)", expanded=True):
    s_ma_trend = st.number_input("趨勢線 (MA)", value=60)
    s_use_year = st.checkbox("啟用年線 (240MA) 濾網", value=True)
    s_big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 5.0, 0.5) / 100
    s_min_vol = st.number_input("波段最小量 (張)", value=1000) * 1000

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
    'd_period': d_period, 'd_threshold': d_threshold,
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

tab1, tab2, tab3 = st.tabs(["🟢 狙擊手波段", "⚡ 隔日沖雷達", "📊 個股診斷"])

# ==========================================
# 修改項目 7：Session State 管理 + 參數變更警告
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None
if 'scan_params' not in st.session_state:
    st.session_state['scan_params'] = None
if 'scan_date' not in st.session_state:
    st.session_state['scan_date'] = None

# 檢查參數是否與上次掃描時不同
def params_changed():
    if st.session_state['scan_params'] is None:
        return False
    return (st.session_state['scan_params'] != params or
            st.session_state['scan_date'] != analysis_date_str)

if start_scan:
    stock_map = get_stock_info_map()

    sniper_triggered = []
    sniper_setup = []
    sniper_watching = []
    day_candidates = []
    failed_list = []

    status_text.text("🔄 正在批量下載歷史資料 (yfinance)...")
    history_data_store = fetch_data_batch(stock_map)

    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    realtime_map = {}
    if analysis_date_str == today_str:
        status_text.text("⚡ 正在批量更新即時盤 (twstock)...")
        realtime_map = fetch_realtime_batch(list(history_data_store.keys()))

    status_text.text("🧠 正在進行策略運算...")
    progress_bar.progress(0)

    tasks_data = {}
    for code, df in history_data_store.items():
        if code in realtime_map and realtime_map[code]['latest_trade_price'] != '-':
            try:
                rt = realtime_map[code]
                new_row = pd.Series({
                    'Open': float(rt['open']), 'High': float(rt['high']),
                    'Low': float(rt['low']), 'Close': float(rt['latest_trade_price']),
                    'Volume': float(rt['accumulate_trade_volume']) * 1000
                }, name=pd.Timestamp(today_str))
                if df.index[-1].strftime('%Y-%m-%d') == today_str:
                    df.iloc[-1] = new_row
                else:
                    df = pd.concat([df, new_row.to_frame().T])
            except Exception as e:
                logger.warning(f"即時資料合併錯誤 [{code}]: {e}")
        tasks_data[code] = df

    total = len(tasks_data)
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_input) as executor:
        futures = {
            executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df): code
            for code, df in tasks_data.items()
        }
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total:
                progress_bar.progress(done / total)
                status_text.text(f"策略運算中: {done}/{total}")
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

    progress_bar.progress(1.0)
    status_text.success(f"掃描完成！ (成功: {len(tasks_data)} / 失敗: {len(failed_list)})")

    # 修改項目 7：儲存掃描當下的參數與日期
    st.session_state['scan_results'] = {
        'sniper_triggered': sniper_triggered,
        'sniper_setup': sniper_setup,
        'sniper_watching': sniper_watching,
        'day_candidates': day_candidates,
        'failed_list': failed_list
    }
    st.session_state['scan_params'] = params.copy()
    st.session_state['scan_date'] = analysis_date_str

results = st.session_state['scan_results']

# 修改項目 7：參數變更提醒
if results is not None and params_changed():
    st.warning("⚠️ 您已修改策略參數或分析日期，目前顯示的結果為**上次掃描**的資料，請重新執行掃描以取得最新結果。")

with tab1:
    st.header("🟢 狙擊手波段策略")
    st.caption(f"基準日: {analysis_date_str} | 策略：趨勢 + 實體長紅 + 型態確認 (防守點含 1% 誤差)")

    if results:
        if 'failed_list' in results and results['failed_list']:
            with st.expander(f"⚠️ 掃描失敗/無資料清單 ({len(results['failed_list'])})"):
                st.write(", ".join(results['failed_list']))

        s_trig = results['sniper_triggered']
        trig_strong = [x for x in s_trig if "強勢突破" in x['狀態']]
        trig_n = [x for x in s_trig if "N字" in x['狀態']]

        s_watch = results['sniper_watching']
        watch_strong = [x for x in s_watch if "強勢整理" in x['狀態']]
        watch_pullback = [x for x in s_watch if "回檔整理" in x['狀態']]

        if trig_strong or trig_n:
            st.markdown("### 🎯 買點觸發訊號 (Actionable)")
            if trig_strong:
                st.markdown(f"### 🚀 強勢突破 ({len(trig_strong)})")
                df = pd.DataFrame(trig_strong)
                if 'sort_pct' in df.columns: df = df.sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct'])
                display_full_table(df)
            if trig_n:
                st.markdown(f"### 🎯 N字突破 ({len(trig_n)})")
                df = pd.DataFrame(trig_n)
                if 'sort_pct' in df.columns: df = df.sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct'])
                display_full_table(df)

        if results['sniper_setup'] or watch_strong or watch_pullback:
            if trig_strong or trig_n: st.divider()
            st.markdown("### 👀 市場潛力名單 (Monitoring)")
            if results['sniper_setup']:
                st.markdown(f"### 🔥 今日剛起漲 ({len(results['sniper_setup'])})")
                df = pd.DataFrame(results['sniper_setup'])
                if 'sort_pct' in df.columns: df = df.sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct'])
                display_full_table(df)
            if watch_strong:
                st.markdown(f"### 💪 強勢整理 ({len(watch_strong)})")
                df = pd.DataFrame(watch_strong)
                if 'sort_pct' in df.columns: df = df.sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct'])
                display_full_table(df)
            if watch_pullback:
                st.markdown(f"### 📉 回檔整理 ({len(watch_pullback)})")
                df = pd.DataFrame(watch_pullback)
                if 'sort_pct' in df.columns: df = df.sort_values(by='sort_pct', ascending=False).drop(columns=['sort_pct'])
                display_full_table(df)
    else:
        st.info("👈 請點擊左側「開始全域掃描」按鈕。")

with tab2:
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

with tab3:
    st.header("📊 個股 K 線診斷")

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        stock_input = st.text_input("輸入代號", value="2330")
    with col_btn:
        diag_btn = st.button("診斷")

    # 修改項目 4：回測按鈕
    run_bt = st.checkbox("同時執行回測", value=False, help="勾選後診斷時會一併執行 SniperStrategy 回測，需要較長時間")

    if diag_btn:
        with st.spinner("載入資料中..."):
            df, symbol = run_diagnosis(stock_input, analysis_date_str, params)

        if df is not None:
            # 修改項目 3：嘗試從掃描結果中找出訊號資訊
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

            # 顯示策略命中狀態
            if sniper_info:
                st.success(f"✅ 此股命中策略：{sniper_info.get('狀態', '')}　|　訊號日：{sniper_info.get('訊號日', sniper_info.get('_setup_date', 'N/A'))}")
            else:
                st.info("ℹ️ 此股在最近一次掃描中未命中任何策略訊號（或尚未執行掃描）")

            # 繪製圖表
            fig = plot_diagnosis_chart(df, stock_input, analysis_date_str, params, sniper_info)
            st.plotly_chart(fig, use_container_width=True)

            # 修改項目 4：執行回測
            if run_bt:
                st.markdown("---")
                st.markdown("### 📈 SniperStrategy 回測結果")
                st.caption(f"使用當前側邊欄參數 | 初始資金 NT$100,000 | 手續費 0.1425% x2")
                with st.spinner("回測運算中..."):
                    run_backtest_ui(df, stock_input, params)
        else:
            st.error("查無資料，請確認代號是否正確")
