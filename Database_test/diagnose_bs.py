import baostock as bs
import pandas as pd
import datetime

def diagnose_baostock():
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Login failed: {lg.error_msg}")
        return

    print("="*40)
    print("诊断 query_all_stock (获取当日全量股票)")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    rs = bs.query_all_stock(day=date)
    print(f"Fields: {rs.fields}")
    if rs.next():
        print(f"Sample: {rs.get_row_data()}")
    
    print("\n" + "="*40)
    print("诊断 query_stock_basic (获取个股基本信息)")
    # 试探性查询一只股票
    rs_basic = bs.query_stock_basic(code="sh.600000")
    print(f"Fields: {rs_basic.fields}")
    if rs_basic.next():
        print(f"Sample: {rs_basic.get_row_data()}")
        
    bs.logout()

if __name__ == "__main__":
    diagnose_baostock()
