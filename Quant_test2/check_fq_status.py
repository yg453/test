# -*- coding: utf-8 -*-
"""
check_fq_status.py
A股底层数据库复权状态诊断探针
"""

import os
import sqlite3
import pandas as pd
import numpy as np

try:
    from config import DB_PATH
except ImportError:
    print("[警告] 无法导入 config.DB_PATH，尝试使用当前目录降级路径...")
    DB_PATH = "stock_data.db"

class DataIntegrityChecker:
    def __init__(self, db_path):
        self.db_path = db_path
        self._check_logic()
        
    def _check_logic(self):
        """运行时自检：确保数据库文件物理存在"""
        assert os.path.exists(self.db_path), f"[致命错误] 找不到数据库文件: {self.db_path}"

    def check_target_stock(self, code='sh.600000'):
        """
        核心诊断逻辑：通过 A 股涨跌停极限反推数据复权状态
        """
        print(f"=== 开始对 {code} 进行复权状态物理诊断 ===")
        print(f"-> 目标数据库: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT date, code, close 
            FROM daily_kline 
            WHERE code='{code}' 
            ORDER BY date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        assert not df.empty, f"[逻辑崩溃] 数据库中没有找到 {code} 的数据！"
        assert 'close' in df.columns, "[逻辑崩溃] 数据缺失核心字段 'close'"
        
        # 计算基于相邻交易日的真实价格跳空比例 (收益率)
        df['prev_close'] = df['close'].shift(1)
        df['pct_change'] = (df['close'] / df['prev_close'] - 1.0) * 100.0
        
        # 排除第一行的 NaN
        df.dropna(inplace=True)
        
        # 寻找极端异常点 (A股正常单日跌幅极难超过 21%)
        extreme_drops = df[df['pct_change'] < -21.0]
        extreme_ups = df[df['pct_change'] > 21.0]
        
        print(f"-> 共载入 {len(df)} 个交易日数据.")
        print(f"-> 历史最大单日涨幅: {df['pct_change'].max():.2f}%")
        print(f"-> 历史最大单日跌幅: {df['pct_change'].min():.2f}%")
        
        print("\n=== 诊断结论 ===")
        if not extreme_drops.empty:
            print("[!!! 警报 !!!] 发现未复权数据！(Unadjusted Data Detected)")
            print("原因：检测到违背 A 股物理规则的断崖式下跌缺口。极大概率是发生过分红/送转。")
            print("这会导致神经网络计算出的收益率和波动率彻底畸变！")
            print("\n[异常样本清单 (Top 5 暴跌日)]:")
            # 打印最严重的5次暴跌
            worst_5 = extreme_drops.sort_values(by='pct_change', ascending=True).head(5)
            for _, row in worst_5.iterrows():
                print(f"  * {row['date']}: 昨收={row['prev_close']:.2f}, 今收={row['close']:.2f} (跌幅 {row['pct_change']:.2f}%)")
        else:
            print("[安全通过] 恭喜！当前数据大概率为【前复权】或已平滑数据。")
            print("原因：未发现突破 A 股常理的断崖式价格衰减，神经网络可安全食用。")
            if not extreme_ups.empty:
                print("\n[备注]: 虽然没有暴跌，但存在单日暴涨 > 21% 的记录 (可能是新股上市首日或长期停牌复牌):")
                top_ups = extreme_ups.sort_values(by='pct_change', ascending=False).head(3)
                for _, row in top_ups.iterrows():
                    print(f"  * {row['date']}: 昨收={row['prev_close']:.2f}, 今收={row['close']:.2f} (涨幅 {row['pct_change']:.2f}%)")

if __name__ == "__main__":
    checker = DataIntegrityChecker(DB_PATH)
    # 测试你一直在用的浦发银行 (经常有分红)
    checker.check_target_stock('sh.600000')