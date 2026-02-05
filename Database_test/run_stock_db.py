from stock_db_manager import StockDBManager
import os
import sys

def main():
    print("="*60)
    print("A股(沪深主板) 历史数据库更新程序")
    print("="*60)
    
    # 获取当前脚本所在目录的绝对路径，确保数据库生成在正确位置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'stocks.db')
    
    print(f"数据库路径: {db_path}")
    print("正在初始化/更新数据库，请勿关闭程序...")
    print("首次运行可能需要较长时间(约数十分钟)下载过去5年数据。")
    print("后续运行仅下载当日新数据，速度极快。")
    print("-" * 60)
    
    try:
        manager = StockDBManager(db_path)
        manager.connect()
        manager.update_daily_data(lookback_years=5)
        print("\n" + "="*60)
        print("✅ 更新完成！所有数据已保存至 stocks.db")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作。数据库状态安全，下次运行将断点续传。")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        if 'manager' in locals():
            manager.close()

if __name__ == "__main__":
    main()
