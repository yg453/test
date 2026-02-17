# -*- coding: utf-8 -*-
"""
probe_db.py
数据库库存透视探针
"""
import sqlite3
import pandas as pd

try:
    from config import DB_PATH
except ImportError:
    DB_PATH = "stocks.db"

def main():
    print(f"正在扫描数据库: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        # 查询数据库中究竟有哪些独特的股票代码，以及它们的数据条数
        query = "SELECT code, COUNT(*) as days_count FROM daily_kline GROUP BY code"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            print("[警告] 数据库是空的，没有任何股票数据！")
            return
            
        print(f"\n✅ 扫描完毕！当前数据库中共包含 {len(df)} 只股票。")
        print("\n[库存清单 (前 20 只)]:")
        print(df.head(20).to_string(index=False))
        
        # 检查你的目标股票
        target = 'sz.000158'
        if target in df['code'].values:
            days = df[df['code'] == target]['days_count'].values[0]
            print(f"\n🎯 发现目标 {target}！共有 {days} 天的数据。")
            if days < 100:
                print("   -> 警告：数据量太少，不足以支撑 5 年的回测训练，请重新下载！")
        else:
            print(f"\n❌ 确诊：数据库中根本没有 {target} 的数据。")
            
    except Exception as e:
        print(f"数据库读取失败，请检查路径是否正确: {e}")

if __name__ == "__main__":
    main()