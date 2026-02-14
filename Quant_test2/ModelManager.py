# -*- coding: utf-8 -*-
"""
ModelManager.py
模型权重与种子管理中心 (炼丹炉与藏经阁)
"""
import os
import glob
import re
import torch

class ModelManager:
    def __init__(self, base_dir="trained_models"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def get_stock_dir(self, stock_code):
        path = os.path.join(self.base_dir, stock_code)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_top_models(self, stock_code):
        stock_dir = self.get_stock_dir(stock_code)
        files = glob.glob(os.path.join(stock_dir, "rank_*.pth"))
        models = []
        for f in files:
            basename = os.path.basename(f)
            m = re.match(r"rank_\d+_pnl_([\d\.\-]+)_seed_(\d+)\.pth", basename)
            if m:
                models.append({
                    'path': f, 'pnl': float(m.group(1)), 'seed': int(m.group(2))
                })
        models.sort(key=lambda x: x['pnl'], reverse=True)
        return models

    def save_if_top(self, stock_code, model, scaler, pnl, seed):
        models = self.get_top_models(stock_code)
        
        if len(models) < 5 or pnl > models[-1]['pnl']:
            print(f"\n[ModelManager] 🏆 触发入库条件! PnL: {pnl:.2f} 成功打榜.")
            stock_dir = self.get_stock_dir(stock_code)
            
            for m in models: m['checkpoint'] = torch.load(m['path'])
                
            new_checkpoint = {'model_state': model.state_dict(), 'scaler': scaler}
            models.append({'pnl': pnl, 'seed': seed, 'checkpoint': new_checkpoint})
            
            models.sort(key=lambda x: x['pnl'], reverse=True)
            models = models[:5]
            
            for f in glob.glob(os.path.join(stock_dir, "rank_*.pth")): os.remove(f)
                
            for i, m in enumerate(models):
                rank = i + 1
                new_name = f"rank_{rank}_pnl_{m['pnl']:.2f}_seed_{m['seed']}.pth"
                torch.save(m['checkpoint'], os.path.join(stock_dir, new_name))
                if m['seed'] == seed and m['pnl'] == pnl:
                    print(f"[ModelManager] 新模型已保存为第 {rank} 名: {new_name}")
        else:
            print(f"\n[ModelManager] ❌ 竞争失败. PnL: {pnl:.2f} 未能超越第 5 名 ({models[-1]['pnl']:.2f}).")

    def load_model(self, stock_code, rank):
        models = self.get_top_models(stock_code)
        if rank < 1 or rank > len(models):
            raise ValueError(f"找不到排名为 {rank} 的模型。目前库中仅有 {len(models)} 个。")
        target = models[rank - 1]
        print(f"[ModelManager] 成功加载排名第 {rank} 的模型 (Seed: {target['seed']}, 历史PnL: {target['pnl']:.2f})")
        return torch.load(target['path']), target['seed']