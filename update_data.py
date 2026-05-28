# update_data.py
# 台股歷史資料每日自動更新 (Hugging Face 雲端版 - 永久保留歷史資料)
import yfinance as yf
import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta
from huggingface_hub import HfApi
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 🔐 系統參數與環境變數
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 👇 請把這裡改成你新開的 Hugging Face 資料庫名稱！(例如 "4340P/tw_stock_history")
HF_REPO = "4340P/history" 
FILE_NAME = "history.parquet"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{FILE_NAME}"

CHUNK_SIZE  = 100
SLEEP_SEC   = 2
NEW_STOCK_DAYS = 800 # 當發現新上市的股票時，預設往前抓取幾天的歷史資料

def get_stock_map():
    stock_map = {}
    sj_api_key    = os.environ.get("SJ_API_KEY", "")
    sj_secret_key = os.environ.get("SJ_SECRET_KEY", "")

    if sj_api_key and sj_secret_key:
        try:
            import shioaji as sj
            api = sj.Shioaji(simulation=True)
            api.login(api_key=sj_api_key, secret_key=sj_secret_key, fetch_contract=True)
            for c in api.Contracts.Stocks.TSE:
                if len(c.code) == 4 and c.code.isdigit():
                    stock_map[c.code] = {"symbol": c.code + ".TW", "name": c.name, "market": "TWSE"}
            for c in api.Contracts.Stocks.OTC:
                if len(c.code) == 4 and c.code.isdigit():
                    stock_map[c.code] = {"symbol": c.code + ".TWO", "name": c.name, "market": "TPEX"}
            try: api.logout()
            except: pass
            print("   永豐合約：共 " + str(len(stock_map)) + " 支")
            return stock_map
        except Exception as e:
            print("   永豐登入失敗（" + str(e) + "），改用 OpenAPI fallback")

    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        for row in r.json():
            code = str(row.get("Code", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_map[code] = {"symbol": code + ".TW", "name": row.get("Name", ""), "market": "TWSE"}
    except Exception as e: print("   證交所 OpenAPI 失敗: " + str(e))
    
    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        for row in r.json():
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_map[code] = {"symbol": code + ".TWO", "name": row.get("CompanyName", ""), "market": "TPEX"}
    except Exception as e: print("   櫃買 OpenAPI 失敗: " + str(e))

    print("   OpenAPI fallback：共 " + str(len(stock_map)) + " 支")
    return stock_map

def download_batch(symbols, start, end):
    result = {}
    try:
        raw = yf.download(" ".join(symbols), start=start, end=end, group_by="ticker", auto_adjust=True, threads=True, progress=False)
        if raw.empty: return result
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    df = raw[sym].dropna(how="all")
                    if not df.empty:
                        df = df[["Open","High","Low","Close","Volume"]].copy()
                        df.index = pd.to_datetime(df.index)
                        if df.index.tz is not None: df.index = df.index.tz_localize(None)
                        result[sym] = df
                except: continue
        else:
            sym = symbols[0]; df = raw.dropna(how="all")
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                result[sym] = df
    except Exception as e: print("  批次下載錯誤: " + str(e))
    return result

def build_frames(all_dfs, sym_to_code, stock_map):
    frames = []
    for sym, df in all_dfs.items():
        code = sym_to_code.get(sym, sym); df = df.copy()
        df["code"] = code; df["symbol"] = sym; df["name"] = stock_map.get(code, {}).get("name", ""); df["market"] = stock_map.get(code, {}).get("market", "")
        df.index.name = "date"
        frames.append(df.reset_index())
    return frames

def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 56)
    print("  台股歷史資料每日更新 (Hugging Face 雲端無刪除版)")
    print("  執行時間：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 56)

    if not HF_TOKEN:
        print("❌ 找不到 HF_TOKEN 環境變數！請在 GitHub Secrets 中設定。")
        exit(1)

    print("\n1. 讀取現有資料 (從 Hugging Face)...")
    try:
        existing = pd.read_parquet(HF_URL)
        existing["date"] = pd.to_datetime(existing["date"])
        last_date = existing["date"].max()
        existing_codes = set(existing["code"].unique())
        print("   ✅ 現有資料最新日期：" + str(last_date.date()))
        print("   ✅ 現有筆數：" + str(len(existing)))
    except Exception as e:
        print("   ⚠️ 無法從 HF 讀取舊資料。請確認你是否已經手動上傳了第一份 history.parquet！")
        print(f"   (錯誤細節: {e})")
        exit(1) 

    print("\n2. 取得股票清單...")
    stock_map   = get_stock_map()
    symbols     = [info["symbol"] for info in stock_map.values()]
    sym_to_code = {v["symbol"]: k for k, v in stock_map.items()}
    new_codes   = [c for c in stock_map if c not in existing_codes]
    print("   合計 " + str(len(stock_map)) + " 支，其中新股 " + str(len(new_codes)) + " 支")

    all_dfs = {}

    if new_codes:
        new_symbols = [stock_map[c]["symbol"] for c in new_codes]
        cutoff_str  = (datetime.today() - timedelta(days=NEW_STOCK_DAYS)).strftime("%Y-%m-%d")
        end_new     = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\n3. 補齊新股歷史（{len(new_codes)} 支，往前抓取 {NEW_STOCK_DAYS} 天）...")
        chunks_new = [new_symbols[i:i+CHUNK_SIZE] for i in range(0, len(new_symbols), CHUNK_SIZE)]
        for idx, batch in enumerate(chunks_new, 1):
            print("  [" + str(idx) + "/" + str(len(chunks_new)) + "] 下載新股...", end="  ")
            batch_data = download_batch(batch, cutoff_str, end_new)
            all_dfs.update(batch_data)
            print("取得 " + str(len(batch_data)) + " 支")
            if idx < len(chunks_new): time.sleep(SLEEP_SEC)

    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date <= today:
        print("\n4. 下載增量資料（" + start_date + " 至 " + today + "）...")
        chunks  = [symbols[i:i+CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
        total_c = len(chunks)
        for idx, batch in enumerate(chunks, 1):
            print("  [" + str(idx) + "/" + str(total_c) + "] 下載中...", end="  ")
            batch_data = download_batch(batch, start_date, end_date)
            all_dfs.update(batch_data)
            print("取得 " + str(len(batch_data)) + " 支")
            if idx < total_c: time.sleep(SLEEP_SEC)

    if not all_dfs:
        print("\n今日無新資料（可能是假日或尚未收盤）")
        return

    print("\n5. 整合資料...")
    frames = build_frames(all_dfs, sym_to_code, stock_map)
    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])
    
    combined = pd.concat([existing, new_data], ignore_index=True)
    
    # 去除重複：如果今天抓了兩次，只保留最新的一筆，確保資料精準乾淨
    combined = combined.drop_duplicates(subset=["code","date"], keep="last")
    combined = combined.sort_values(["code","date"]).reset_index(drop=True)

    combined["Open"]   = combined["Open"].astype("float32")
    combined["High"]   = combined["High"].astype("float32")
    combined["Low"]    = combined["Low"].astype("float32")
    combined["Close"]  = combined["Close"].astype("float32")
    combined["Volume"] = combined["Volume"].astype("int64")

    print(f"   合併後總筆數：{len(combined)}")
    
    print("\n6. 儲存並上傳至 Hugging Face...")
    temp_file = "temp_history_updated.parquet"
    combined.to_parquet(temp_file, index=False, compression="snappy")
    
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=FILE_NAME,
            repo_id=HF_REPO,
            repo_type="dataset"
        )
        print("   ✅ 成功覆蓋推送到 Hugging Face 資料庫！")
    except Exception as e:
        print(f"   ❌ 上傳失敗: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()
