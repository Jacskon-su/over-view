# update_chip.py
# 每日自動抓取 FinMind 籌碼資料，並上傳至 Hugging Face
# GitHub Actions 每天 21:37（台灣時間）自動執行

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from huggingface_hub import HfApi

# ==========================================
# 🔐 系統參數與環境變數 (絕對不寫死 Token)
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# 你的 Hugging Face 資料庫設定
HF_REPO = "4340P/institutional_investors_parquet_by_stock"
FILE_NAME = "all_institutional_investors_2020_2026.parquet"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{FILE_NAME}"

def get_stock_list():
    """利用 OpenAPI 抓取最新上市櫃股票清單"""
    stock_list = []
    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=15)
        for row in r.json():
            code = str(row.get("Code", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_list.append(code)
        
        r2 = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", timeout=15)
        for row in r2.json():
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if len(code) == 4 and code.isdigit():
                stock_list.append(code)
    except Exception as e:
        print(f"取得股票清單失敗: {e}")
    return list(set(stock_list))

def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 56)
    print("  法人籌碼每日自動更新 (Hugging Face)")
    print("  執行時間：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 56)

    if not HF_TOKEN:
        print("❌ 找不到 HF_TOKEN 環境變數！請在 GitHub Secrets 中設定。")
        exit(1)

    api = HfApi(token=HF_TOKEN)

    print("\n1. 正在從 Hugging Face 下載現有籌碼資料庫...")
    try:
        # 由於已設為 Public，可直接讀取免帶 Token
        df_old = pd.read_parquet(HF_URL)
        print(f"   ✅ 下載成功！目前資料庫筆數: {len(df_old):,}")
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        exit(1)

    print("\n2. 取得全市場股票清單...")
    stocks = get_stock_list()
    print(f"   共 {len(stocks)} 檔股票待抓取")

    print("\n3. 開始抓取 FinMind 今日籌碼資料...")
    finmind_url = "https://api.finmindtrade.com/api/v4/data"
    today_data = []

    # 智慧限速機制：有 Token 則全速抓取，無 Token 則拉長間隔避免被鎖 IP
    sleep_time = 0.5 if FINMIND_TOKEN else 5.0

    for i, code in enumerate(stocks):
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": code,
            "start_date": today,
            "end_date": today,
        }
        if FINMIND_TOKEN:
            params["token"] = FINMIND_TOKEN

        # 斷線重試機制 (最多重試 3 次)
        for retry in range(3):
            try:
                resp = requests.get(finmind_url, params=params, timeout=15)
                data = resp.json().get("data", [])
                if data:
                    today_data.extend(data)
                break
            except Exception as e:
                print(f"     [重試 {retry+1}] 抓取 {code} 發生錯誤: {e}")
                time.sleep(3)
        
        time.sleep(sleep_time)

        # 每 100 檔印出一次進度
        if (i + 1) % 100 == 0:
            print(f"   已處理 {i + 1} / {len(stocks)} 檔...")

    if not today_data:
        print("\n今日無新籌碼資料 (可能為假日或尚未更新完畢)。")
        return

    df_today = pd.DataFrame(today_data)
    print(f"\n   ✅ 今日共抓取到 {len(df_today):,} 筆籌碼資料")

    print("\n4. 合併新舊資料並進行滑動視窗清理...")
    df_all = pd.concat([df_old, df_today], ignore_index=True)
    
    # 確保格式一致
    df_all['date'] = pd.to_datetime(df_all['date']).dt.strftime('%Y-%m-%d')
    df_all['stock_id'] = df_all['stock_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # 去除重複資料 (以防重複執行導致抓到兩份同一天的)
    df_all = df_all.drop_duplicates(subset=['date', 'stock_id', 'name'], keep='last')

    # 滑動視窗：保留近 6 年資料
    six_years_ago = (datetime.today() - timedelta(days=6*365)).strftime('%Y-%m-%d')
    df_all = df_all[df_all['date'] >= six_years_ago]
    df_all = df_all.sort_values(by=['date', 'stock_id']).reset_index(drop=True)

    print(f"   更新後總筆數: {len(df_all):,}")

    print("\n5. 儲存暫存檔並上傳至 Hugging Face...")
    temp_file = "temp_chip_updated.parquet"
    df_all.to_parquet(temp_file, index=False, compression="snappy")

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
            os.remove(temp_file) # 刪除暫存檔釋放空間

if __name__ == "__main__":
    main()
