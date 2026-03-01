# ==========================================
# 強勢股戰情室 V10
# V2: Bug修復 + 評分系統 + 大盤狀態 + 產業泡泡圖
# V3: 長紅K回溯天數、均量基準、量能門檻可調
# V4: 回測深度過濾
# V5: 表格移除整理振幅、量能倍數、評分欄位
# V6: 下載加速（period縮短至300d + chunk_size擴大至150）
# V7: 加入 10MA > 20MA > 60MA 多頭排列濾網
# V8: 移除市場潛力名單
# V10: 整合處置股策略（第二分頁）+ 底量濾網
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
    page_title="強勢股戰情室 V10",
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
# 📈 大盤狀態判斷
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)

# ==========================================
# 🎯 處置股策略模組
# ==========================================
def is_valid_stock_code(code):
    """
    只接受一般股票代號：
    - 上市：4碼純數字，不含可轉債(第5碼起有字母)
    - 上櫃：4碼純數字，1開頭到8開頭，排除衍生商品
    可轉債代號通常 > 9000 或包含字母
    """
    if not code or not code.isdigit():
        return False
    if len(code) != 4:
        return False
    num = int(code)
    # 排除可轉債（通常 > 9000）、ETF受益憑證等特殊代號
    if num >= 9000:
        return False
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_disposal_stocks():
    """
    從證交所與櫃買中心爬取當前處置股清單
    回傳 dict: { code: { 'name', 'start', 'end', 'market', 'symbol' } }
    """
    disposal = {}

    def roc_to_date(s):
        s = s.strip()
        parts = s.replace('/', '-').split('-')
        y = int(parts[0].strip()) + 1911
        m = int(parts[1].strip())
        d = int(parts[2].strip())
        return datetime.date(y, m, d)

    # --- 上市（證交所）---
    # 格式：row[2]=代號, row[3]=名稱, row[6]='115/03/02～115/03/13'（全形波浪號）
    try:
        today_dt = datetime.date.today()
        start_query = today_dt - datetime.timedelta(days=30)
        end_query   = today_dt + datetime.timedelta(days=30)
        # 證交所使用西元年 YYYYMMDD 格式
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
        logger.warning(f"上市處置股查詢 URL: {url}")
        res = requests.get(url, headers=headers, timeout=15)
        logger.warning(f"上市處置股 HTTP status: {res.status_code}")
        data = res.json()
        logger.warning(f"上市處置股 stat:{data.get('stat')} total:{data.get('total')} count:{data.get('count')}")

        if data.get('stat') == 'OK':
            twse_temp = {}
            total_rows = len(data.get('data', []))
            logger.warning(f"上市處置股 API 回傳 {total_rows} 筆 | title:{data.get('title','')} | total:{data.get('total','')}")
            for row in data.get('data', []):
                try:
                    code = str(row[2]).strip()
                    name = str(row[3]).strip()
                    period_str = str(row[6]).strip()

                    # 支援全形 ～ 和半形 ~ 兩種分隔符
                    if '～' in period_str:
                        parts = period_str.split('～')
                    elif '~' in period_str:
                        parts = period_str.split('~')
                    else:
                        logger.warning(f"上市 {code} 日期格式無法解析: {period_str}")
                        continue

                    start_dt = roc_to_date(parts[0].strip())
                    end_dt   = roc_to_date(parts[1].strip())

                    # 只保留每5分鐘撮合的處置股
                    # 上市：row[7]=處置措施 row[8]=處置內容，兩個都檢查
                    disposal_content = ''
                    if len(row) > 7: disposal_content += str(row[7])
                    if len(row) > 8: disposal_content += str(row[8])
                    is_5min = '每五分鐘' in disposal_content or '每5分鐘' in disposal_content

                    logger.warning(f"上市 {code} {name} | 期間:{period_str} | 5min:{is_5min} | valid:{is_valid_stock_code(code)}")

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
                    logger.warning(f"上市處置股解析錯誤: {row} -> {e}")
            logger.warning(f"上市處置股篩選後 {len(twse_temp)} 筆")
            disposal.update(twse_temp)
        else:
            st.warning(f"⚠️ 上市處置股查詢異常: {data.get('stat')}")
    except Exception as e:
        st.warning(f"⚠️ 無法取得上市處置股清單: {e}")

    # --- 上櫃（櫃買中心）---
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
            res = requests.get(url, headers=headers, timeout=15)
            res.encoding = 'utf-8'

            # JSON 格式
            if res.headers.get('content-type','').startswith('application/json') or res.text.strip().startswith('['):
                rows = res.json()
                if isinstance(rows, list):
                    for row in rows:
                        try:
                            code = str(row.get('SecuritiesCompanyCode', row.get('code',''))).strip()
                            name = str(row.get('CompanyName', row.get('name',''))).strip()

                            # 處理 DispositionPeriod 格式：'1150204~1150226'
                            period = str(row.get('DispositionPeriod', '')).strip()
                            if '~' in period:
                                parts = period.split('~')
                                start_s = parts[0].strip()  # '1150204'
                                end_s   = parts[1].strip()  # '1150226'
                                # 格式轉換：1150204 → 民國115年02月04日
                                def roc8_to_date(s):
                                    s = s.strip()
                                    y = int(s[0:3]) + 1911
                                    m = int(s[3:5])
                                    d = int(s[5:7])
                                    return datetime.date(y, m, d)
                                start_dt = roc8_to_date(start_s)
                                end_dt   = roc8_to_date(end_s)
                            else:
                                # 備用：嘗試獨立欄位
                                start_s = str(row.get('DisposalStartDate', row.get('start',''))).strip()
                                end_s   = str(row.get('DisposalEndDate',   row.get('end',''))).strip()
                                start_dt = roc_to_date(start_s)
                                end_dt   = roc_to_date(end_s)

                            # 只保留每5分鐘撮合的處置股
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
                            logger.warning(f"上櫃處置股JSON解析錯誤: {row} -> {e}")
                    break

            # HTML 格式
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
                            logger.warning(f"上櫃處置股HTML解析錯誤: {cols} -> {e}")
                break
        except Exception as e:
            logger.warning(f"上櫃處置股端點失敗 {url}: {e}")
            continue

    return disposal


def analyze_disposal_stock(code, info, analysis_date, min_vol=0):
    """
    下載日K，執行進出場邏輯
    回傳包含最新訊號狀態的 dict
    """
    try:
        symbol = info['symbol']
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", auto_adjust=True)

        if df.empty or len(df) < 10:
            return None

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 量能濾網
        if min_vol > 0:
            _s = pd.Timestamp(info['start'])
            _e = pd.Timestamp(analysis_date)
            _dv = df[(_s <= df.index) & (df.index <= _e)]
            if not _dv.empty and _dv['Volume'].min() < min_vol * 1000:
                return None

        # 計算 MA（用全部資料確保均線準確）
        df['ma5']  = df['Close'].rolling(5).mean()
        df['ma10'] = df['Close'].rolling(10).mean()

        # 只保留處置期間到分析基準日的資料
        start_ts = pd.Timestamp(info['start'])
        end_ts   = pd.Timestamp(analysis_date)
        df_period = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

        if len(df_period) < 2:
            return None

        analysis_date_str = analysis_date.strftime('%Y-%m-%d')

        # ==========================================
        # 掃描進出場訊號
        # ==========================================
        position = None   # 持倉資訊 { entry_date, entry_price }
        trades   = []     # 完成的交易紀錄
        latest_signal = None  # 最新狀態

        for i in range(1, len(df_period)):
            today      = df_period.index[i]
            today_str  = today.strftime('%Y-%m-%d')
            c_today    = df_period['Close'].iloc[i]
            prev_high  = df_period['High'].iloc[i-1]
            ma5_today  = df_period['ma5'].iloc[i]
            ma10_today = df_period['ma10'].iloc[i]
            pct_today  = (c_today - df_period['Close'].iloc[i-1]) / df_period['Close'].iloc[i-1] * 100

            if position is None:
                # 沒有持倉 → 找進場訊號
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
                # 有持倉 → 判斷出場條件
                entry_price = position['entry_price']
                profit_pct  = (c_today - entry_price) / entry_price * 100

                # 決定用哪條 MA 出場
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
                    position = None  # 出場後重置
                    latest_signal = None
                else:
                    # 更新持有中狀態
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

        # ==========================================
        # 整理回傳結果
        # ==========================================
        today_date_str = analysis_date_str

        # 判斷今天有沒有新進場訊號
        is_new_signal_today = (
            latest_signal is not None and
            latest_signal.get('entry_date') == today_date_str and
            latest_signal.get('status') == '🟢 持有中'
        )

        # 有持倉或今天有訊號才回傳
        if latest_signal is None and not trades:
            return None

        # 最近一筆已出場交易
        last_trade = trades[-1] if trades else None

        # 目前狀態
        if latest_signal:
            status = latest_signal['status']
        elif last_trade:
            status = '⚫ 已出場'
        else:
            status = '-'

        # 計算處置剩餘天數
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
            # 進場資訊
            'entry_date':       latest_signal['entry_date'] if latest_signal else (last_trade['entry_date'] if last_trade else '-'),
            'entry_price':      latest_signal['entry_price'] if latest_signal else (f"{last_trade['entry_price']:.2f}" if last_trade else '-'),
            # 出場資訊
            'exit_date':        latest_signal.get('exit_date', '-') if latest_signal else (last_trade['exit_date'] if last_trade else '-'),
            'exit_price':       latest_signal.get('exit_price', '-') if latest_signal else (f"{last_trade['exit_price']:.2f}" if last_trade else '-'),
            'profit_pct':       latest_signal.get('profit_pct', '-') if latest_signal else (f"{last_trade['profit_pct']:+.2f}%" if last_trade else '-'),
            'exit_reason':      latest_signal.get('exit_reason', '-') if latest_signal else (last_trade['exit_reason'] if last_trade else '-'),
            # 訊號K棒資訊
            'signal_date':      latest_signal.get('signal_date', '-') if latest_signal else '-',
            'signal_close':     latest_signal.get('signal_close', '-') if latest_signal else '-',
            'signal_prev_high': latest_signal.get('signal_prev_high', '-') if latest_signal else '-',
            'signal_pct':       latest_signal.get('signal_pct', '-') if latest_signal else '-',
            # 統計
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



def fetch_market_status(analysis_date_str):
    """
    下載加權指數，判斷大盤強弱
    回傳 dict: { 'strong': bool, 'score': int(0~10), 'label': str }
    """
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
# 🏆 N字品質評分
# ==========================================
def calc_sniper_score(c_today, prev_h, defense_price, s_high, s_low, s_close,
                      setup_idx, idx, high, low, volume, op, market_score, pullback_depth=0):
    """
    評分維度（滿分100）：
    1. 風險報酬比    35分
    2. 量能品質      25分
    3. 收盤強度      20分
    4. 趨勢距離      15分
    5. 大盤環境       5分
    """
    score = 0
    details = {}

    # --- 1. 風險距離 (35分) ---
    # 風險距離 = (收盤 - 防守價) / 收盤，越小越好
    if c_today > 0:
        risk_pct = (c_today - defense_price) / c_today * 100
        details['風險距離'] = round(risk_pct, 1)
        if risk_pct <= 3.0:   score += 35   # 極佳
        elif risk_pct <= 5.0: score += 25   # 良好
        elif risk_pct <= 8.0: score += 15   # 尚可
        else:                 score += 0    # 風險距離過大
    else:
        details['風險距離'] = 0

    # --- 2. 量能品質 (25分) ---
    if setup_idx > 0 and idx > setup_idx:
        consolidation_vol = volume.iloc[setup_idx:idx].mean()
        today_vol = volume.iloc[idx]
        vol_ratio = today_vol / consolidation_vol if consolidation_vol > 0 else 0
        details['量能倍數'] = round(vol_ratio, 1)
        if 2.0 <= vol_ratio <= 4.0:   score += 25
        elif 1.5 <= vol_ratio < 2.0:  score += 18
        elif vol_ratio >= 4.0:        score += 12  # 量太大可能是出貨
        elif vol_ratio >= 1.0:        score += 8
        else:                         score += 0
    else:
        details['量能倍數'] = 0

    # --- 3. 收盤強度 (20分) ---
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

    # --- 4. 整理振幅（越小越好）(15分) ---
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

    # --- 5. 回測深度 (加分項，最高+10）---
    # 回測越淺代表籌碼越穩，給予額外加分
    details['回測深度'] = round(pullback_depth, 1)
    if pullback_depth <= 20:    score += 10   # 極淺，籌碼鎖死
    elif pullback_depth <= 38:  score += 7    # 黃金回測
    elif pullback_depth <= 50:  score += 3    # 尚可
    # > 50% 已被過濾，不會到這裡

    # --- 6. 大盤環境 (5分) ---
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

        # --- 策略 A: 狙擊手 ---
        s_ma_trend = params['s_ma_trend']
        s_use_year = params['s_use_year']
        s_big_candle = params['s_big_candle']
        s_min_vol = params['s_min_vol']
        s_setup_lookback = params.get('s_setup_lookback', 25)
        s_vol_ma_days = params.get('s_vol_ma_days', 20)
        s_vol_ratio = params.get('s_vol_ratio', 0.8)
        s_pullback_max = params.get('s_pullback_max', 50)

        ma_t = close.rolling(window=s_ma_trend).mean()
        ma_y = close.rolling(window=240).mean()
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()
        ma60 = close.rolling(window=60).mean()
        vol_ma = volume.rolling(window=5).mean()          # 保留5日均量供其他用途
        vol_ma_setup = volume.rolling(window=s_vol_ma_days).mean()  # 長紅K判斷用

        is_sniper_candidate = True
        if volume.iloc[idx] < s_min_vol: is_sniper_candidate = False
        if s_use_year and len(ma_y) > idx and (pd.isna(ma_y.iloc[idx]) or close.iloc[idx] < ma_y.iloc[idx]): is_sniper_candidate = False
        if not (close.iloc[idx] > ma_t.iloc[idx] and ma_t.iloc[idx] > ma_t.iloc[idx-1]): is_sniper_candidate = False
        # V7: 10MA > 20MA > 60MA 多頭排列
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

                # --- 回測深度過濾 ---
                # 整理期間（長紅K隔天到突破前一天）收盤最低點
                if setup_idx + 1 < idx:
                    consolidation_close_min = close.iloc[setup_idx + 1 : idx].min()
                else:
                    consolidation_close_min = s_close
                # 回測深度 = (長紅K收盤 - 整理期收盤最低) / 長紅K收盤
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
st.sidebar.title("🔥 強勢股戰情室 V10")
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
    s_setup_lookback = st.slider("長紅K回溯天數", 5, 30, 25, 5)
    st.caption("預設25天，整理期較長的型態建議拉高")
    s_vol_ma_days = st.slider("長紅K量能基準 (MA天數)", 5, 20, 20, 5)
    s_vol_ratio = st.slider("長紅K量能門檻 (倍)", 0.5, 1.5, 0.8, 0.1)
    st.caption("量 > 基準均量 × 門檻倍數才算長紅K")
    s_pullback_max = st.slider("整理回測深度上限 (%)", 20, 70, 50, 5)
    st.caption("整理期收盤最低點回測長紅K收盤超過此比例則過濾")

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
    'd_min_pct': d_min_pct, 'd_min_vol': d_min_vol
}

tab1, tab2, tab3, tab4 = st.tabs(["🟢 狙擊手波段", "🎯 處置股策略", "⚡ 隔日沖雷達", "📊 個股診斷"])

# ==========================================
# 修改項目 7：Session State 管理 + 參數變更警告
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None
if 'scan_params' not in st.session_state:
    st.session_state['scan_params'] = None
if 'scan_date' not in st.session_state:
    st.session_state['scan_date'] = None
if 'market_status' not in st.session_state:
    st.session_state['market_status'] = None

# 檢查參數是否與上次掃描時不同
def params_changed():
    if st.session_state['scan_params'] is None:
        return False
    return (st.session_state['scan_params'] != params or
            st.session_state['scan_date'] != analysis_date_str)

if start_scan:
    stock_map = get_stock_info_map()

    # 大盤狀態
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
            executor.submit(analyze_combined_strategy, code, stock_map[code], analysis_date_str, params, SECTOR_DB, df, market_status.get('score', 5)): code
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

    # --- 大盤狀態列 ---
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

        # --- 買點觸發 ---
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

            # --- 產業強度泡泡圖 ---
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

with tab2:
    st.markdown("### 🎯 處置股策略")
    st.caption("僅限每5分鐘撮合處置股｜進場：收盤站上前日最高點｜出場：獲利<15%跌破10MA，獲利≥15%跌破5MA")

    col_d1, col_d2 = st.columns([2, 2])
    with col_d1:
        d_analysis_date = st.date_input("分析基準日", datetime.date.today(), key="disposal_date")
    with col_d2:
        d_show_all = st.checkbox("顯示全部（含無訊號）", value=False, key="disposal_show_all")

    col_d3, col_d4 = st.columns([1, 3])
    with col_d3:
        d_vol_filter = st.checkbox("啟用底量濾網", value=True, key="disposal_vol_filter")
    with col_d4:
        d_min_vol = st.number_input("底量門檻（張）", min_value=0, max_value=10000,
                                     value=1000, step=100, key="disposal_min_vol",
                                     disabled=not d_vol_filter)

    d_scan_btn = st.button("🔍 開始掃描處置股", type="primary", key="disposal_scan")

    if d_scan_btn:
        with st.spinner("📋 正在取得處置股清單..."):
            disposal_map = fetch_disposal_stocks()

        if not disposal_map:
            st.error("❌ 無法自動取得處置股清單")
            manual_input = st.text_input("手動輸入代號（逗號分隔）", placeholder="3234,6271", key="disposal_manual")
            if manual_input:
                for _c in manual_input.replace('，', ',').split(','):
                    _c = _c.strip()
                    if len(_c) == 4 and _c.isdigit():
                        disposal_map[_c] = {
                            'name': _c, 'market': 'twse',
                            'start': d_analysis_date - datetime.timedelta(days=30),
                            'end':   d_analysis_date + datetime.timedelta(days=30),
                            'symbol': f"{_c}.TW"
                        }
        else:
            lookahead = d_analysis_date + datetime.timedelta(days=3)
            d_active        = {k: v for k, v in disposal_map.items()
                               if v['start'] <= lookahead and v['end'] >= d_analysis_date}
            d_active_now    = {k: v for k, v in d_active.items() if v['start'] <= d_analysis_date}
            d_active_coming = {k: v for k, v in d_active.items() if v['start'] > d_analysis_date}

            st.info(f"📋 處置中 **{len(d_active_now)}** 支｜即將開始（3天內）**{len(d_active_coming)}** 支")

            if not d_active:
                st.warning("目前沒有符合條件的處置股")
            else:
                d_results = []
                d_progress = st.progress(0)
                d_status = st.empty()

                for i, (code, info) in enumerate(d_active_now.items()):
                    d_status.text(f"🔍 分析中... {code} {info['name']} ({i+1}/{len(d_active_now)})")
                    d_progress.progress((i+1) / max(len(d_active_now), 1))
                    result = analyze_disposal_stock(code, info, d_analysis_date,
                                                    min_vol=d_min_vol if d_vol_filter else 0)
                    if result:
                        d_results.append(result)
                    elif d_show_all:
                        d_results.append({
                            'code': code, 'name': info['name'],
                            'market': '上市' if info['market'] == 'twse' else '上櫃',
                            'disposal_start': info['start'].strftime('%Y-%m-%d'),
                            'disposal_end':   info['end'].strftime('%Y-%m-%d'),
                            'days_left': max((info['end'] - d_analysis_date).days, 0),
                            'status': '⚪ 無訊號', 'is_new_today': False,
                            'entry_date': '-', 'entry_price': '-',
                            'exit_date': '-', 'exit_price': '-',
                            'profit_pct': '-', 'exit_reason': '-',
                            'signal_date': '-', 'signal_close': '-',
                            'signal_prev_high': '-', 'signal_pct': '-',
                            'total_trades': 0,
                        })
                    time.sleep(0.3)

                d_progress.empty()
                d_status.empty()

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
                        st.info(f"基準日 {d_analysis_date} 無今日訊號")

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
        st.info("👈 點擊「開始掃描處置股」取得今日訊號")

with tab3:
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

with tab4:
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
