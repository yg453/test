import sqlite3
import baostock as bs
import pandas as pd
import datetime
from tqdm import tqdm
import os
import time
import signal

# 定义超时异常
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Baostock query timed out")

class StockDBManager:
    def __init__(self, db_path='stocks.db'):
        self.db_path = db_path
        # 定义板块过滤规则
        self.BOARD_CONFIG = {
            'sh_main': {'enabled': True, 'prefixes': ['sh.600', 'sh.601', 'sh.603', 'sh.605']},  # 沪市主板
            'sz_main': {'enabled': True, 'prefixes': ['sz.000', 'sz.001', 'sz.002', 'sz.003']},  # 深市主板 (含原中小板)
            'chi_next': {'enabled': False, 'prefixes': ['sz.300']},                               # 创业板
            'star': {'enabled': False, 'prefixes': ['sh.688']},                                   # 科创板
            'bj': {'enabled': False, 'prefixes': ['bj.']},                                        # 北交所
            'etf': {'enabled': False, 'prefixes': ['sh.51', 'sh.58', 'sz.15', 'sz.16']},         # ETF
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_tables()
        self._migrate_tables()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def _init_tables(self):
        """初始化数据库表结构"""
        # 股票基础信息表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_basic (
            code TEXT PRIMARY KEY,
            code_name TEXT,
            tradeStatus TEXT, -- 1:正常, 0:停牌
            listing_status TEXT, -- 上市, 退市, 停牌
            is_st INTEGER,  -- 1: 是ST, 0: 否
            last_update_date TEXT -- 记录该股票最后一次更新行情的日期
        )
        ''')

        # 日线行情表
        # 联合主键：日期 + 股票代码
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_kline (
            date TEXT,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            preclose REAL,
            volume INTEGER,
            amount REAL,
            adjustflag INTEGER,
            turn REAL,
            pctChg REAL,
            peTTM REAL,
            pbMRQ REAL,
            psTTM REAL,
            pcfNcfTTM REAL,
            is_st INTEGER,
            PRIMARY KEY (date, code)
        )
        ''')
        
        # 创建索引加速查询
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_kline (date)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_code ON daily_kline (code)')
        self.conn.commit()

    def _migrate_tables(self):
        """检查并迁移数据库表结构"""
        # 检查 stock_basic 表的列
        self.cursor.execute("PRAGMA table_info(stock_basic)")
        columns = [info[1] for info in self.cursor.fetchall()]
        
        # 需要确保存在的列
        required_columns = {
            'tradeStatus': 'TEXT',
            'listing_status': 'TEXT',
            'last_update_date': 'TEXT',
            'is_st': 'INTEGER'
        }
        
        for col, dtype in required_columns.items():
            if col not in columns:
                print(f"正在迁移数据库: 添加列 {col}...")
                try:
                    self.cursor.execute(f"ALTER TABLE stock_basic ADD COLUMN {col} {dtype}")
                except Exception as e:
                    print(f"添加列 {col} 失败: {e}")
        
        # 检查 daily_kline 表的列
        self.cursor.execute("PRAGMA table_info(daily_kline)")
        kline_columns = [info[1] for info in self.cursor.fetchall()]
        
        # 检查关键列 is_st
        if 'is_st' not in kline_columns:
             print("正在迁移数据库: daily_kline 添加列 is_st...")
             try:
                 self.cursor.execute("ALTER TABLE daily_kline ADD COLUMN is_st INTEGER")
             except Exception as e:
                 print(f"daily_kline 添加列 is_st 失败: {e}")
        
        self.conn.commit()

    def _login_baostock(self):
        """登录Baostock"""
        lg = bs.login()
        if lg.error_code != '0':
            raise Exception(f"Baostock login failed: {lg.error_msg}")
        print(f"Baostock login success: {lg.error_msg}")

    def _logout_baostock(self):
        bs.logout()

    def _is_target_board(self, code):
        """
        判断股票是否属于启用的板块
        """
        for board, config in self.BOARD_CONFIG.items():
            if config['enabled']:
                for prefix in config['prefixes']:
                    if code.startswith(prefix):
                        return True
        return False

    def _get_stock_list_at_date(self, date_str):
        """获取指定日期的股票列表"""
        rs = bs.query_all_stock(day=date_str)
        stocks = []
        while (rs.error_code == '0') and rs.next():
            stocks.append(rs.get_row_data())
        return pd.DataFrame(stocks, columns=rs.fields) if stocks else pd.DataFrame()

    def update_stock_list(self, lookback_years=5):
        """
        更新股票列表
        策略：为了捕获退市股票，我们查询过去N年每年的年初和年中的股票列表，
        然后与当日列表合并去重。
        """
        print("正在构建全量股票列表 (包含历史退市股票)...")
        
        all_dfs = []
        
        # 1. 获取最近的一个交易日的列表 (作为"上市"状态的基准)
        latest_valid_df = pd.DataFrame()
        check_date = datetime.datetime.now()
        # 向前回溯10天找最近的交易日数据
        for _ in range(10):
            d_str = check_date.strftime("%Y-%m-%d")
            # print(f"Trying date: {d_str}")
            df = self._get_stock_list_at_date(d_str)
            if not df.empty:
                latest_valid_df = df
                print(f"使用 {d_str} 的数据作为最新股票列表基准")
                break
            check_date -= datetime.timedelta(days=1)
            
        if not latest_valid_df.empty:
            all_dfs.append(latest_valid_df)
            latest_codes = set(latest_valid_df['code'].values)
            # 创建一个查找字典，用于快速获取最新交易状态
            # code -> tradeStatus
            latest_status_map = dict(zip(latest_valid_df['code'], latest_valid_df['tradeStatus']))
        else:
            print("❌ 严重错误：无法获取最近10天内的任何股票列表！")
            print("  为了防止错误地将所有股票标记为退市，程序将中止更新列表。")
            return []
            
        # 2. 采样历史列表
        current_year = datetime.datetime.now().year
        print("正在采样历史数据以捕获退市股票...")
        for year in range(current_year - lookback_years, current_year + 1):
            for date_suffix in ['-01-05', '-07-05']: 
                date_str = f"{year}{date_suffix}"
                if date_str >= check_date.strftime("%Y-%m-%d"): # 避免重复
                    continue
                df_hist = self._get_stock_list_at_date(date_str)
                if not df_hist.empty:
                    all_dfs.append(df_hist)
        
        if not all_dfs:
            print("无法获取任何股票列表！")
            return []

        # 3. 合并去重
        df_combined = pd.concat(all_dfs, ignore_index=True)
        # 按代码去重，保留第一次出现的信息（即最新的信息，因为latest_valid_df在最前）
        df_combined = df_combined.drop_duplicates(subset=['code'], keep='first')
        
        print(f"初步获取到 {len(df_combined)} 只历史/当前股票")

        # 4. 过滤板块
        df_combined['is_target'] = df_combined['code'].apply(self._is_target_board)
        df_target = df_combined[df_combined['is_target']].copy()
        
        # 5. 标记并过滤ST股票 (简单策略：最新名称含ST)
        df_target['is_st'] = df_target['code_name'].apply(lambda x: 1 if 'ST' in x else 0)
        
        # 仅保留非ST股票
        df_final = df_target[df_target['is_st'] == 0].copy()
        
        print(f"筛选出 {len(df_final)} 只符合条件股票 (沪深主板, 非ST)")
        
        # 6. 计算 Listing Status (上市, 退市, 停牌)
        def get_status(row):
            code = row['code']
            if code in latest_codes:
                # 在最新列表中，检查 tradeStatus
                # Baostock: 1=正常, 0=停牌
                t_status = latest_status_map.get(code, '1')
                if t_status == '1':
                    return '上市'
                else:
                    return '停牌'
            else:
                # 不在最新列表中 -> 退市
                return '退市'

        def get_trade_status(row):
            code = row['code']
            if code in latest_codes:
                 return latest_status_map.get(code, '1')
            else:
                # 退市股票 tradeStatus 设为 0
                return '0'

        df_final['listing_status'] = df_final.apply(get_status, axis=1)
        df_final['tradeStatus'] = df_final.apply(get_trade_status, axis=1)

        # 更新数据库基础表
        data_to_insert = []
        for _, row in df_final.iterrows():
            data_to_insert.append((
                row['code'], 
                row['code_name'], 
                row['tradeStatus'], 
                row['listing_status'],
                row['is_st']
            ))
            
        # 使用 Upsert (ON CONFLICT) 语法，避免覆盖 last_update_date
        self.cursor.executemany('''
        INSERT INTO stock_basic (code, code_name, tradeStatus, listing_status, is_st)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            code_name=excluded.code_name,
            tradeStatus=excluded.tradeStatus,
            listing_status=excluded.listing_status,
            is_st=excluded.is_st
        ''', data_to_insert)
        self.conn.commit()
        
        return df_final['code'].tolist()

    def get_latest_date(self, code):
        """查询数据库中某只股票的最新日期"""
        self.cursor.execute('SELECT MAX(date) FROM daily_kline WHERE code = ?', (code,))
        result = self.cursor.fetchone()
        return result[0] if result[0] else None

    def update_daily_data(self, lookback_years=5):
        """
        全量/增量更新日线数据
        """
        self._login_baostock()
        
        try:
            # 1. 更新股票列表
            stock_codes = self.update_stock_list(lookback_years)
            
            print("开始更新日线数据...")
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            # 默认最早开始时间
            default_start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
            
            pbar = tqdm(stock_codes)
            for code in pbar:
                pbar.set_description(f"Processing {code}")
                
                # 确定起始日期
                last_date = self.get_latest_date(code)
                if last_date:
                    # 如果有数据，从最后一天+1天开始
                    start_date = (datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    # 如果没数据，从5年前开始
                    start_date = default_start_date
                
                # 如果起始日期超过今天，说明是最新的，跳过
                if start_date > today:
                    continue
                
                # 获取数据
                fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
                
                # 设置超时报警 (30秒)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    rs = bs.query_history_k_data_plus(
                        code, fields,
                        start_date=start_date, end_date=today,
                        frequency="d", adjustflag="3" 
                    )
                    
                    # 取消超时报警
                    signal.alarm(0)
                    
                    if rs.error_code != '0':
                        continue

                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        row = rs.get_row_data()
                        # 转换空字符串为 None
                        row = [None if x == '' else x for x in row]
                        data_list.append(row)
                    
                    if data_list:
                        self.cursor.executemany(f'''
                        INSERT OR IGNORE INTO daily_kline 
                        ({fields.replace('isST', 'is_st')})
                        VALUES ({','.join(['?']*len(fields.split(',')))})
                        ''', data_list)
                        
                        # 更新 stock_basic 表的 last_update_date，标记该股已更新
                        latest_date = data_list[-1][0] # 假设第一列是date
                        self.cursor.execute('UPDATE stock_basic SET last_update_date = ? WHERE code = ?', (latest_date, code))
                        
                        self.conn.commit()
                        
                except TimeoutException:
                    tqdm.write(f"⚠️ Warning: Timeout processing {code}, skipping...")
                    # 重新登录以防连接死锁
                    try:
                        self._logout_baostock()
                    except:
                        pass
                    time.sleep(1)
                    try:
                        self._login_baostock()
                    except:
                        pass
                    continue
                except Exception as e:
                    signal.alarm(0) # 确保异常时也关闭alarm
                    tqdm.write(f"Error processing {code}: {e}")
                    continue
                finally:
                    signal.alarm(0) # 双重保险
                    
        finally:
            self._logout_baostock()


if __name__ == "__main__":
    db = StockDBManager()
    db.connect()
    try:
        db.update_daily_data(lookback_years=5)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()
