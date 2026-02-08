# run_pipeline.py
import os
import subprocess
import sys
from config import UPDATE_SCRIPT, DB_PATH

def step_1_update_data():
    print(">>> [Step 1] Updating A-Share Data...")
    # 调用外部 Python 脚本更新数据库
    if not os.path.exists(UPDATE_SCRIPT):
        print(f"Error: Update script not found at {UPDATE_SCRIPT}")
        return False
    
    try:
        # 使用当前 python 解释器运行
        subprocess.check_call([sys.executable, UPDATE_SCRIPT])
        print("✅ Data Update Completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data Update Failed: {e}")
        return False

def step_2_run_strategy():
    print(">>> [Step 2] Running Strategy Simulation...")
    # 运行 main.py
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    subprocess.check_call([sys.executable, main_script])

if __name__ == "__main__":
    # 询问用户是否更新数据
    user_input = input("Do you want to update stock data first? (y/n): ").lower()
    
    if user_input == 'y':
        success = step_1_update_data()
        if not success:
            sys.exit(1)
            
    step_2_run_strategy()