# ==========================================
# update_data.py
# 台股歷史資料每日自動更新
# 由 GitHub Actions 每天 14:30（台灣時間）自動執行
# 只下載最新缺少的日期，合併後套用滾動視窗
# ==========================================

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta

try:
    import twstock
except ImportError:
    print("❌ 請先安裝 twstock：pip install twstock")
    exit(1)

# ==========================================
# 設定
# ==========================================
KEEP_DAYS   = 800          # 滾動視窗：永遠只保留最近 800 個日曆天
CHUNK_SIZE  = 100
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "history.parquet")
LOG_FILE    = os.path.join(OUTPUT_DIR, "last_update.txt")
SLEEP_SEC   = 2

# ==========================================
# 取得全台股清單
# ==========================================
def get_stock_map():
    stock_map = {}
    for code, info in twstock.twse.items():
        if len(code) == 4 and code.isdigit():
            stock_map[code] = {"symbol": f"{code}.TW",  "name": info.name, "market": "TWSE"}
    for code, info in twstock.tpex.items():
        if len(code) == 4 and code.isdigit():
            stock_map[code] = {"symbol": f"{code}.TWO", "name": info.name, "market": "TPEX"}
    return stock_map

# ==========================================
# 批量下載
# ==========================================
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
        print(f"  ⚠️  批次下載錯誤: {e}")
    return result

# ==========================================
# 主程式
# ==========================================
def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 56)
    print("  台股歷史資料每日更新")
    print(f"  執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  滾動視窗：保留最近 {KEEP_DAYS} 天")
    print("=" * 56)

    # ── 確認 history.parquet 存在 ──
    if not os.path.exists(OUTPUT_FILE):
        print(f"❌ 找不到 {OUTPUT_FILE}，請先執行 init_data.py")
        exit(1)

    # ── 讀取現有資料 ──
    print(f"\n📂 讀取現有資料...")
    existing = pd.read_parquet(OUTPUT_FILE)
    existing["date"] = pd.to_datetime(existing["date"])
    last_date = existing["date"].max()
    print(f"   現有資料最新日期：{last_date.date()}")
    print(f"   現有筆數：{len(existing):,}")

    # ── 取得股票清單 ──
    print("\n📋 取得股票清單...")
    stock_map   = get_stock_map()
    symbols     = [info["symbol"] for info in stock_map.values()]
    sym_to_code = {v["symbol"]: k for k, v in stock_map.items()}
    existing_codes = set(existing["code"].unique())
    new_codes = [code for code in stock_map if code not in existing_codes]
    print(f"   共 {len(symbols)} 支，其中新股 {len(new_codes)} 支（parquet 沒有）")

    all_dfs = {}

    # ── 補齊新股完整歷史 ──
    if new_codes:
        new_symbols = [stock_map[c]["symbol"] for c in new_codes]
        cutoff_str  = (datetime.today() - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
        end_date_new = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\n🆕 補齊新股歷史（{len(new_codes)} 支，從 {cutoff_str} 起）...")
        chunks_new = [new_symbols[i:i+CHUNK_SIZE] for i in range(0, len(new_symbols), CHUNK_SIZE)]
        for idx, batch in enumerate(chunks_new, 1):
            print(f"  [{idx}/{len(chunks_new)}] 下載 {len(batch)} 支新股...", end="  ")
            batch_data = download_batch(batch, cutoff_str, end_date_new)
            all_dfs.update(batch_data)
            print(f"取得 {len(batch_data)} 支")
            if idx < len(chunks_new):
                time.sleep(SLEEP_SEC)

    # ── 判斷需要下載的日期增量 ──
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date > today and not new_codes:
        print(f"\n✅ 資料已是最新（{last_date.date()}），不需要更新")
        return

    if start_date <= today:
        print(f"\n📥 下載增量資料（{start_date} ～ {today}，每批 {CHUNK_SIZE} 支）...")
        chunks  = [symbols[i:i+CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
        total_c = len(chunks)
        for idx, batch in enumerate(chunks, 1):
            pct = idx / total_c * 100
            print(f"  [{idx:3d}/{total_c}] {pct:5.1f}%  下載 {len(batch)} 支...", end="  ")
            batch_data = download_batch(batch, start_date, end_date)
            all_dfs.update(batch_data)
            print(f"取得 {len(batch_data)} 支")
            if idx < total_c:
                time.sleep(SLEEP_SEC)

    if not all_dfs:
        print("\n⚠️  今日無新資料（可能是假日或收盤資料尚未更新）")
        return

    # ── 整合新資料 ──
    print(f"\n🔧 整合新資料（{len(all_dfs)} 支）...")
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

    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])
    print(f"   新增筆數：{len(new_data):,}")
    print(f"   新資料日期：{new_data['date'].min().date()} ～ {new_data['date'].max().date()}")

    # ── 合併並去重 ──
    print("\n🔗 合併資料...")
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code","date"], keep="last")
    combined = combined.sort_values(["code","date"]).reset_index(drop=True)

    # ── 滾動視窗：刪除超過 KEEP_DAYS 的舊資料 ──
    cutoff    = pd.Timestamp.today() - pd.Timedelta(days=KEEP_DAYS)
    before    = len(combined)
    combined  = combined[combined["date"] >= cutoff].reset_index(drop=True)
    trimmed   = before - len(combined)
    if trimmed > 0:
        print(f"   🗑️  刪除 {trimmed:,} 筆過期資料（{cutoff.date()} 以前）")

    # 型別轉換
    combined["Open"]   = combined["Open"].astype("float32")
    combined["High"]   = combined["High"].astype("float32")
    combined["Low"]    = combined["Low"].astype("float32")
    combined["Close"]  = combined["Close"].astype("float32")
    combined["Volume"] = combined["Volume"].astype("int64")

    print(f"   合併後總筆數：{len(combined):,}")
    print(f"   涵蓋股票：{combined['code'].nunique()} 支")
    print(f"   日期範圍：{combined['date'].min().date()} ～ {combined['date'].max().date()}")

    # ── 儲存 ──
    print(f"\n💾 儲存至 {OUTPUT_FILE}...")
    combined.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"   檔案大小：{size_mb:.1f} MB")

    if size_mb > 25:
        print(f"   ⚠️  超過 GitHub 建議的 25MB！考慮縮短 KEEP_DAYS（目前 {KEEP_DAYS}）")
    else:
        print(f"   ✅ 在 GitHub 25MB 限制以內")

    # ── 更新時間記錄 ──
    with open(LOG_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n✅ 更新完成！最新資料日期：{combined['date'].max().date()}")

if __name__ == "__main__":
    main()
