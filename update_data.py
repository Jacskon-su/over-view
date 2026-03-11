# update_data.py
# 台股歷史資料每日自動更新
# 股票清單：永豐金 Shioaji 合約（fallback: 證交所/櫃買 OpenAPI）
# 歷史資料：yfinance
# GitHub Actions 每天 14:30（台灣時間）自動執行

import yfinance as yf
import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta

KEEP_DAYS   = 800
CHUNK_SIZE  = 100
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "history.parquet")
LOG_FILE    = os.path.join(OUTPUT_DIR, "last_update.txt")
SLEEP_SEC   = 2

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
            try:
                api.logout()
            except Exception:
                pass
            print("   永豐合約：共 " + str(len(stock_map)) + " 支")
            return stock_map
        except Exception as e:
            print("   永豐登入失敗（" + str(e) + "），改用 OpenAPI fallback")

    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        for row in r.json():
            code = str(row.get("Code", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_map[code] = {"symbol": code + ".TW", "name": row.get("Name", ""), "market": "TWSE"}
    except Exception as e:
        print("   證交所 OpenAPI 失敗: " + str(e))
    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        for row in r.json():
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_map[code] = {"symbol": code + ".TWO", "name": row.get("CompanyName", ""), "market": "TPEX"}
    except Exception as e:
        print("   櫃買 OpenAPI 失敗: " + str(e))

    print("   OpenAPI fallback：共 " + str(len(stock_map)) + " 支")
    return stock_map

def download_batch(symbols, start, end):
    result = {}
    try:
        raw = yf.download(
            " ".join(symbols),
            start=start, end=end,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False
        )
        if raw.empty:
            return result
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    df = raw[sym].dropna(how="all")
                    if df.empty:
                        continue
                    df = df[["Open","High","Low","Close","Volume"]].copy()
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    result[sym] = df
                except Exception:
                    continue
        else:
            sym = symbols[0]
            df = raw.dropna(how="all")
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result[sym] = df
    except Exception as e:
        print("  批次下載錯誤: " + str(e))
    return result

def build_frames(all_dfs, sym_to_code, stock_map):
    frames = []
    for sym, df in all_dfs.items():
        code = sym_to_code.get(sym, sym)
        df = df.copy()
        df["code"]   = code
        df["symbol"] = sym
        df["name"]   = stock_map.get(code, {}).get("name", "")
        df["market"] = stock_map.get(code, {}).get("market", "")
        df.index.name = "date"
        frames.append(df.reset_index())
    return frames

def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 56)
    print("  台股歷史資料每日更新")
    print("  執行時間：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  滾動視窗：保留最近 " + str(KEEP_DAYS) + " 天")
    print("=" * 56)

    if not os.path.exists(OUTPUT_FILE):
        print("找不到 " + OUTPUT_FILE + "，請先執行 init_data.py")
        exit(1)

    print("\n讀取現有資料...")
    existing = pd.read_parquet(OUTPUT_FILE)
    existing["date"] = pd.to_datetime(existing["date"])
    last_date = existing["date"].max()
    existing_codes = set(existing["code"].unique())
    print("   現有資料最新日期：" + str(last_date.date()))
    print("   現有筆數：" + str(len(existing)))
    print("   現有股票：" + str(len(existing_codes)) + " 支")

    print("\n取得股票清單...")
    stock_map   = get_stock_map()
    symbols     = [info["symbol"] for info in stock_map.values()]
    sym_to_code = {v["symbol"]: k for k, v in stock_map.items()}
    new_codes   = [c for c in stock_map if c not in existing_codes]
    print("   合計 " + str(len(stock_map)) + " 支，其中新股 " + str(len(new_codes)) + " 支（parquet 沒有）")

    all_dfs = {}

    if new_codes:
        new_symbols = [stock_map[c]["symbol"] for c in new_codes]
        cutoff_str  = (datetime.today() - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
        end_new     = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        print("\n補齊新股歷史（" + str(len(new_codes)) + " 支，從 " + cutoff_str + " 起）...")
        chunks_new = [new_symbols[i:i+CHUNK_SIZE] for i in range(0, len(new_symbols), CHUNK_SIZE)]
        for idx, batch in enumerate(chunks_new, 1):
            print("  [" + str(idx) + "/" + str(len(chunks_new)) + "] 下載 " + str(len(batch)) + " 支新股...", end="  ")
            batch_data = download_batch(batch, cutoff_str, end_new)
            all_dfs.update(batch_data)
            print("取得 " + str(len(batch_data)) + " 支")
            if idx < len(chunks_new):
                time.sleep(SLEEP_SEC)

    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date <= today:
        print("\n下載增量資料（" + start_date + " 至 " + today + "）...")
        chunks  = [symbols[i:i+CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
        total_c = len(chunks)
        for idx, batch in enumerate(chunks, 1):
            pct = round(idx / total_c * 100, 1)
            print("  [" + str(idx) + "/" + str(total_c) + "] " + str(pct) + "%  下載 " + str(len(batch)) + " 支...", end="  ")
            batch_data = download_batch(batch, start_date, end_date)
            all_dfs.update(batch_data)
            print("取得 " + str(len(batch_data)) + " 支")
            if idx < total_c:
                time.sleep(SLEEP_SEC)

    if not all_dfs:
        print("\n今日無新資料（可能是假日或尚未收盤）")
        return

    print("\n整合資料（" + str(len(all_dfs)) + " 支）...")
    frames = build_frames(all_dfs, sym_to_code, stock_map)
    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])
    print("   新增筆數：" + str(len(new_data)))

    print("合併資料...")
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code","date"], keep="last")
    combined = combined.sort_values(["code","date"]).reset_index(drop=True)

    cutoff   = pd.Timestamp.today() - pd.Timedelta(days=KEEP_DAYS)
    before   = len(combined)
    combined = combined[combined["date"] >= cutoff].reset_index(drop=True)
    trimmed  = before - len(combined)
    if trimmed > 0:
        print("   刪除 " + str(trimmed) + " 筆過期資料（" + str(cutoff.date()) + " 以前）")

    combined["Open"]   = combined["Open"].astype("float32")
    combined["High"]   = combined["High"].astype("float32")
    combined["Low"]    = combined["Low"].astype("float32")
    combined["Close"]  = combined["Close"].astype("float32")
    combined["Volume"] = combined["Volume"].astype("int64")

    print("   合併後總筆數：" + str(len(combined)))
    print("   涵蓋股票：" + str(combined["code"].nunique()) + " 支")
    print("   日期範圍：" + str(combined["date"].min().date()) + " 至 " + str(combined["date"].max().date()))

    print("\n儲存至 " + OUTPUT_FILE + "...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print("   檔案大小：" + str(round(size_mb, 1)) + " MB")
    if size_mb > 25:
        print("   超過 GitHub 25MB 限制！考慮縮短 KEEP_DAYS")

    with open(LOG_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("\n更新完成！最新資料日期：" + str(combined["date"].max().date()))

if __name__ == "__main__":
    main()
