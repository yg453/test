#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据探索测试程序
用于测试和探索开源数据源的每日股票数据
"""

import pandas as pd
import json
from datetime import datetime, timedelta

def test_akshare():
    """测试AKShare数据源"""
    print("=" * 60)
    print("测试 AKShare 数据源")
    print("=" * 60)
    
    try:
        import akshare as ak
        
        # 1. 获取所有A股股票列表
        print("\n1. 获取A股股票列表...")
        stock_list = ak.stock_zh_a_spot_em()
        print(f"   获取到 {len(stock_list)} 只股票")
        print(f"   列名: {list(stock_list.columns)}")
        print("\n   示例数据 (前3行):")
        print(stock_list.head(3).to_string())
        
        # 2. 获取单只股票历史数据
        print("\n2. 获取单只股票历史日线数据...")
        # 使用平安银行作为示例 (000001)
        stock_code = "000001"
        stock_name = "平安银行"
        
        df_daily = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily",
            start_date="20240101",
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq"  # 前复权
        )
        
        print(f"   股票: {stock_name} ({stock_code})")
        print(f"   数据条数: {len(df_daily)}")
        print(f"   数据列: {list(df_daily.columns)}")
        print("\n   最近5天数据:")
        print(df_daily.tail(5).to_string())
        
        # 3. 获取股票基本信息
        print("\n3. 获取股票基本信息...")
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        print(f"   {stock_name} ({stock_code}) 基本信息:")
        for _, row in stock_info.iterrows():
            print(f"   {row['item']}: {row['value']}")
        
        # 4. 获取实时行情
        print("\n4. 获取实时行情快照...")
        realtime = ak.stock_zh_a_spot_em()
        # 筛选几只热门股票
        hot_stocks = ['000001', '000002', '600519', '000858']
        hot_data = realtime[realtime['代码'].isin(hot_stocks)]
        print("\n   热门股票实时行情:")
        print(hot_data[['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']].to_string(index=False))
        
        # 5. 数据字段总结
        print("\n5. 可用数据字段总结:")
        print("   日线数据字段说明:")
        field_descriptions = {
            '日期': '交易日期',
            '开盘': '开盘价',
            '收盘': '收盘价',
            '最高': '最高价',
            '最低': '最低价',
            '成交量': '成交量(手)',
            '成交额': '成交金额(元)',
            '振幅': '价格振幅(%)',
            '涨跌幅': '价格变动百分比(%)',
            '涨跌额': '价格变动金额',
            '换手率': '换手率(%)'
        }
        for field, desc in field_descriptions.items():
            if field in df_daily.columns:
                print(f"   - {field}: {desc}")
        
        return {
            'source': 'AKShare',
            'status': 'success',
            'total_stocks': len(stock_list),
            'daily_fields': list(df_daily.columns),
            'sample_data': df_daily.tail(5).to_dict('records')
        }
        
    except ImportError:
        print("   错误: 未安装AKShare")
        print("   请运行: pip install akshare")
        return {'source': 'AKShare', 'status': 'error', 'message': '未安装AKShare'}
    except Exception as e:
        print(f"   错误: {e}")
        return {'source': 'AKShare', 'status': 'error', 'message': str(e)}


def test_baostock():
    """测试Baostock数据源"""
    print("\n" + "=" * 60)
    print("测试 Baostock 数据源")
    print("=" * 60)
    
    try:
        import baostock as bs
        
        # 登录
        print("\n1. 登录Baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            print(f"   登录失败: {lg.error_msg}")
            return {'source': 'Baostock', 'status': 'error', 'message': lg.error_msg}
        
        print(f"   登录成功: {lg.error_msg}")
        
        # 2. 获取股票列表
        print("\n2. 获取A股股票列表...")
        rs = bs.query_all_stock(day="2024-01-02")
        stock_list = []
        while rs.error_code == '0' and rs.next():
            stock_list.append(rs.get_row_data())
        
        print(f"   获取到 {len(stock_list)} 只股票")
        print(f"   示例: {stock_list[:5]}")
        
        # 3. 获取历史数据
        print("\n3. 获取单只股票历史数据...")
        code = "sh.600000"  # 浦发银行
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
        
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date='2024-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency='d'
        )
        
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
        
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            print(f"   股票代码: {code}")
            print(f"   数据条数: {len(df)}")
            print(f"   数据字段: {rs.fields}")
            print("\n   最近5天数据:")
            print(df.tail(5).to_string())
        
        # 登出
        bs.logout()
        
        return {
            'source': 'Baostock',
            'status': 'success',
            'total_stocks': len(stock_list),
            'daily_fields': rs.fields if data_list else [],
            'sample_data': df.tail(5).to_dict('records') if data_list else []
        }
        
    except ImportError:
        print("   错误: 未安装baostock")
        print("   请运行: pip install baostock")
        return {'source': 'Baostock', 'status': 'error', 'message': '未安装baostock'}
    except Exception as e:
        print(f"   错误: {e}")
        return {'source': 'Baostock', 'status': 'error', 'message': str(e)}


def test_yfinance():
    """测试Yahoo Finance数据源"""
    print("\n" + "=" * 60)
    print("测试 Yahoo Finance (yfinance) 数据源")
    print("=" * 60)
    
    try:
        import yfinance as yf
        
        print("\n1. 获取A股数据...")
        # A股在Yahoo的代码格式: 股票代码.SS (上海) 或 .SZ (深圳)
        # 贵州茅台: 600519.SS
        ticker = yf.Ticker("600519.SS")
        
        # 获取历史数据
        hist = ticker.history(period="1mo")
        print(f"   股票: 贵州茅台 (600519)")
        print(f"   数据条数: {len(hist)}")
        print(f"   数据列: {list(hist.columns)}")
        print("\n   最近5天数据:")
        print(hist.tail(5).to_string())
        
        # 获取股票信息
        print("\n2. 获取股票基本信息...")
        info = ticker.info
        key_info = {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'pb_ratio': info.get('priceToBook', 'N/A')
        }
        for key, value in key_info.items():
            print(f"   {key}: {value}")
        
        return {
            'source': 'Yahoo Finance',
            'status': 'success',
            'daily_fields': list(hist.columns),
            'sample_data': hist.tail(5).reset_index().to_dict('records')
        }
        
    except ImportError:
        print("   错误: 未安装yfinance")
        print("   请运行: pip install yfinance")
        return {'source': 'Yahoo Finance', 'status': 'error', 'message': '未安装yfinance'}
    except Exception as e:
        print(f"   错误: {e}")
        return {'source': 'Yahoo Finance', 'status': 'error', 'message': str(e)}


def generate_data_summary(results):
    """生成数据总结报告"""
    print("\n" + "=" * 60)
    print("数据探索总结报告")
    "=" * 60
    
    summary = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sources_tested': len(results),
        'available_sources': [],
        'recommendations': []
    }
    
    for result in results:
        if result['status'] == 'success':
            summary['available_sources'].append(result['source'])
    
    print(f"\n✓ 测试时间: {summary['test_time']}")
    print(f"✓ 测试数据源: {summary['sources_tested']} 个")
    print(f"✓ 可用数据源: {', '.join(summary['available_sources'])}")
    
    print("\n推荐方案:")
    if 'AKShare' in summary['available_sources']:
        print("  1. AKShare - 推荐用于日常数据获取")
        print("     优点: 数据源丰富、完全免费、更新及时")
        print("     数据范围: 所有A股日线、分钟线、实时行情")
    
    if 'Baostock' in summary['available_sources']:
        print("\n  2. Baostock - 推荐用于历史数据批量下载")
        print("     优点: 数据稳定、适合批量获取、无需登录")
        print("     数据范围: 日线、周线、月线数据")
    
    print("\n可用数据字段 (日线数据):")
    print("  - 基础价格数据: 开盘价、收盘价、最高价、最低价")
    print("  - 成交量数据: 成交量、成交额")
    print("  - 衍生指标: 涨跌幅、涨跌额、振幅、换手率")
    print("  - 复权数据: 前复权、后复权价格")
    
    return summary


def main():
    """主函数"""
    print("A股数据探索测试程序")
    print("本程序将测试多种开源数据源的可用性和数据字段\n")
    
    results = []
    
    # 测试AKShare
    results.append(test_akshare())
    
    # 测试Baostock
    results.append(test_baostock())
    
    # 测试Yahoo Finance
    results.append(test_yfinance())
    
    # 生成总结
    summary = generate_data_summary(results)
    
    # 保存结果到文件
    output_file = f"stock_data_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'details': results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✓ 详细结果已保存到: {output_file}")
    print("\n建议下一步:")
    print("  1. 运行: pip install akshare baostock yfinance pandas")
    print("  2. 重新运行本程序获取完整数据")
    print("  3. 根据测试结果设计数据库结构")


if __name__ == "__main__":
    main()
