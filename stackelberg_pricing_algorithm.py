#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stackelberg定价博弈算法实现
对比算法A: Stackelberg 定价博弈(CP定价, CDN响应)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import copy
from dataclasses import dataclass
import time

@dataclass
class NodeParameters:
    """节点参数"""
    C_s: float  # 存储容量
    r_b: float  # 带宽容量
    r_c: float  # 计算容量
    c_s: float  # 存储成本
    c_b: float  # 带宽成本
    c_c: float  # 计算成本
    R: Dict[int, float]  # 传输速率 R_i,k
    Phi: float  # 计算效率 Φ_i

@dataclass
class ContentParameters:
    """内容参数"""
    S: float  # 内容大小 S_k
    gamma: float  # 计算复杂度 γ_k
    P: float  # 到达率 P_k

@dataclass
class CPValue:
    """CP价值参数"""
    v_hit: float  # 命中价值 v_k^hit
    v_tr: float   # 传输价值 v_k^tr

class IntegerPackSolver:
    """整数背包问题求解器"""
    
    def __init__(self):
        pass
    
    def solve(self, gain_stor: float, gain_hit: List[float], gain_tr: List[float],
              constraints: Dict, content_params: List[ContentParameters]) -> Tuple[List[float], List[int], List[int]]:
        """
        求解整数背包问题
        
        Args:
            gain_stor: 存储边际收益
            gain_hit: 命中边际收益列表
            gain_tr: 传输边际收益列表
            constraints: 约束条件
            content_params: 内容参数列表
            
        Returns:
            (t_i, n_i, m_i): 存储时间、命中次数、传输次数
        """
        K = len(content_params)
        
        # 初始化决策变量
        t_i = [0.0] * K  # 存储时间
        n_i = [0] * K    # 命中次数
        m_i = [0] * K    # 传输次数
        
        # 计算每个内容的单位收益
        items = []
        for k in range(K):
            content = content_params[k]
            
            # 存储收益
            if gain_stor > 0:
                items.append({
                    'type': 'storage',
                    'content_id': k,
                    'gain': gain_stor,
                    'size': content.S,
                    'constraint': 'storage'
                })
            
            # 命中收益
            if gain_hit[k] > 0:
                items.append({
                    'type': 'hit',
                    'content_id': k,
                    'gain': gain_hit[k],
                    'size': content.S / constraints['R'][k] if k in constraints['R'] else content.S,
                    'constraint': 'bandwidth'
                })
            
            # 传输收益
            if gain_tr[k] > 0:
                items.append({
                    'type': 'transmission',
                    'content_id': k,
                    'gain': gain_tr[k],
                    'size': content.gamma * content.S / constraints['Phi'],
                    'constraint': 'computation'
                })
        
        # 按收益密度排序（贪心策略）
        items.sort(key=lambda x: x['gain'] / x['size'], reverse=True)
        
        # 当前资源使用量
        used_storage = 0.0
        used_bandwidth = 0.0
        used_computation = 0.0
        
        # 贪心选择
        for item in items:
            content_id = item['content_id']
            content = content_params[content_id]
            
            if item['type'] == 'storage':
                # 检查存储约束
                if used_storage + content.S <= constraints['C_s']:
                    t_i[content_id] += 1.0
                    used_storage += content.S
                    
            elif item['type'] == 'hit':
                # 检查带宽约束
                bandwidth_needed = content.S / constraints['R'][content_id] if content_id in constraints['R'] else content.S
                if used_bandwidth + bandwidth_needed <= constraints['r_b']:
                    n_i[content_id] += 1
                    used_bandwidth += bandwidth_needed
                    
            elif item['type'] == 'transmission':
                # 检查计算约束
                computation_needed = content.gamma * content.S / constraints['Phi']
                if used_computation + computation_needed <= constraints['r_c']:
                    m_i[content_id] += 1
                    used_computation += computation_needed
        
        return t_i, n_i, m_i

class StackelbergPricingAlgorithm:
    """Stackelberg定价博弈算法"""
    
    def __init__(self, eta: float = 0.01, epsilon: float = 1e-3, max_iterations: int = 500):
        """
        初始化算法
        
        Args:
            eta: 学习步长
            epsilon: 收敛阈值
            max_iterations: 最大迭代次数
        """
        self.eta = eta
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.pack_solver = IntegerPackSolver()
        
    def stackelberg_pricing(self, 
                          node_params: List[NodeParameters],
                          content_params: List[ContentParameters],
                          cp_values: List[CPValue],
                          T: float,
                          target_stor_value: float = 100.0) -> Dict:
        """
        Stackelberg定价博弈主算法
        
        Args:
            node_params: 节点参数列表
            content_params: 内容参数列表
            cp_values: CP价值参数列表
            T: 统计窗口时间
            target_stor_value: 目标存储价值
        
        Returns:
            包含最终价格和分配结果的字典
        """
        I = len(node_params)  # 节点数量
        K = len(content_params)  # 内容数量
        
        # 初始化价格
        p_s = 1.0  # 存储价格
        p_hit = [1.0] * K  # 命中价格
        p_tr = [1.0] * K   # 传输价格
        
        # 存储历史价格用于收敛判断
        prev_prices = None
        
        print(f"开始Stackelberg定价博弈算法...")
        print(f"节点数量: {I}, 内容数量: {K}")
        print(f"学习步长: {self.eta}, 收敛阈值: {self.epsilon}")
        
        for iteration in range(self.max_iterations):
            # 存储当前迭代的分配结果
            allocations = {
                't': [[0.0] * K for _ in range(I)],  # 存储时间
                'n': [[0] * K for _ in range(I)],    # 命中次数
                'm': [[0] * K for _ in range(I)]     # 传输次数
            }
            
            # 内层循环：各节点给出最优响应
            for i in range(I):
                node = node_params[i]
                
                # 计算单位边际净收益
                gain_stor = p_s - node.c_s
                gain_hit = []
                gain_tr = []
                
                for k in range(K):
                    content = content_params[k]
                    # 命中边际收益
                    if k in node.R:
                        hit_gain = p_hit[k] - node.c_b * content.S / node.R[k]
                    else:
                        hit_gain = p_hit[k] - node.c_b * content.S / 1000.0  # 默认传输速率
                    gain_hit.append(hit_gain)
                    
                    # 传输边际收益
                    tr_gain = p_tr[k] - node.c_c * content.gamma * content.S / node.Phi
                    gain_tr.append(tr_gain)
                
                # 构建约束条件
            constraints = {
                    'C_s': node.C_s * T,
                    'r_b': node.r_b * T,
                    'r_c': node.r_c * T,
                    'R': node.R,
                    'Phi': node.Phi
            }
            
            # 求解整数背包问题
            t_i, n_i, m_i = self.pack_solver.solve(
                gain_stor, gain_hit, gain_tr, constraints, content_params
            )
            
            # 存储分配结果
            allocations['t'][i] = t_i
            allocations['n'][i] = n_i
            allocations['m'][i] = m_i
            
            # CP价格更新
            new_p_hit = []
            new_p_tr = []
            
            for k in range(K):
                # 计算总使用量
                total_hit = sum(allocations['n'][i][k] for i in range(I))
                total_tr = sum(allocations['m'][i][k] for i in range(I))
                
                # 更新命中价格
                new_hit_price = max(0, p_hit[k] + self.eta * (cp_values[k].v_hit - total_hit))
                new_p_hit.append(new_hit_price)
                
                # 更新传输价格
                new_tr_price = max(0, p_tr[k] + self.eta * (cp_values[k].v_tr - total_tr))
                new_p_tr.append(new_tr_price)
        
        # 更新存储价格
            total_storage = sum(sum(allocations['t'][i][k] for k in range(K)) for i in range(I))
            new_p_s = max(0, p_s + self.eta * (target_stor_value - total_storage))
            
            # 检查收敛性
            current_prices = [new_p_s] + new_p_hit + new_p_tr
            if prev_prices is not None:
                price_changes = [abs(current_prices[i] - prev_prices[i]) for i in range(len(current_prices))]
                max_change = max(price_changes)
                
                if max_change <= self.epsilon:
                    print(f"算法在第 {iteration + 1} 次迭代后收敛")
                    break
            
            # 更新价格
            p_s = new_p_s
            p_hit = new_p_hit
            p_tr = new_p_tr
            prev_prices = current_prices.copy()
            
            # 每100次迭代输出一次进度
            if (iteration + 1) % 100 == 0:
                total_storage = sum(sum(allocations['t'][i][k] for k in range(K)) for i in range(I))
                total_hits = sum(sum(allocations['n'][i][k] for k in range(K)) for i in range(I))
                total_transmissions = sum(sum(allocations['m'][i][k] for k in range(K)) for i in range(I))
                print(f"迭代 {iteration + 1}: p_s={p_s:.4f}, max_p_hit={max(p_hit):.4f}, max_p_tr={max(p_tr):.4f}")
                print(f"  分配量: 存储={total_storage:.1f}, 命中={total_hits}, 传输={total_transmissions}")
                if prev_prices is not None:
                    price_changes = [abs(current_prices[i] - prev_prices[i]) for i in range(len(current_prices))]
                    max_change = max(price_changes)
                    print(f"  最大价格变化: {max_change:.6f}")
        
        # 返回最终结果
        result = {
            'prices': {
                'p_s': p_s,
                'p_hit': p_hit,
                'p_tr': p_tr
            },
            'allocations': allocations,
            'iterations': iteration + 1,
            'converged': iteration < self.max_iterations - 1
        }
        
        return result

def create_test_data():
    """创建测试数据"""
    
    # 创建节点参数
    node_params = []
    for i in range(5):  # 5个节点
        R = {k: 1000.0 + k * 100 for k in range(10)}  # 传输速率
        node = NodeParameters(
            C_s=10000.0 + i * 1000,  # 存储容量
            r_b=100.0 + i * 10,      # 带宽容量
            r_c=50.0 + i * 5,        # 计算容量
            c_s=0.1 + i * 0.01,      # 存储成本
            c_b=0.5 + i * 0.05,      # 带宽成本
            c_c=1.0 + i * 0.1,       # 计算成本
            R=R,
            Phi=100.0 + i * 10       # 计算效率
        )
        node_params.append(node)
    
    # 创建内容参数
    content_params = []
    for k in range(10):  # 10个内容
        content = ContentParameters(
            S=1000.0 + k * 100,      # 内容大小
            gamma=1.0 + k * 0.1,     # 计算复杂度
            P=10.0 + k * 1.0         # 到达率
        )
        content_params.append(content)
    
    # 创建CP价值参数
    cp_values = []
    for k in range(10):
        cp_value = CPValue(
            v_hit=5.0 + k * 0.5,     # 命中价值
            v_tr=3.0 + k * 0.3       # 传输价值
        )
        cp_values.append(cp_value)
    
    return node_params, content_params, cp_values

def main():
    """主函数"""
    print("=== Stackelberg定价博弈算法测试 ===")
    
    # 创建测试数据
    node_params, content_params, cp_values = create_test_data()
    
    # 创建算法实例
    algorithm = StackelbergPricingAlgorithm(eta=0.01, epsilon=1e-3, max_iterations=500)
    
    # 运行算法
    start_time = time.time()
    result = algorithm.stackelberg_pricing(
        node_params=node_params,
        content_params=content_params,
        cp_values=cp_values,
        T=100.0,  # 统计窗口时间
        target_stor_value=100.0
    )
    end_time = time.time()
    
    # 输出结果
    print(f"\n算法运行时间: {end_time - start_time:.2f} 秒")
    print(f"迭代次数: {result['iterations']}")
    print(f"是否收敛: {result['converged']}")
    
    print(f"\n最终价格:")
    print(f"存储价格 p_s: {result['prices']['p_s']:.4f}")
    print(f"命中价格 p_hit: {[f'{p:.4f}' for p in result['prices']['p_hit']]}")
    print(f"传输价格 p_tr: {[f'{p:.4f}' for p in result['prices']['p_tr']]}")
    
    # 计算总分配量
    total_storage = sum(sum(result['allocations']['t'][i][k] for k in range(len(content_params))) 
                       for i in range(len(node_params)))
    total_hits = sum(sum(result['allocations']['n'][i][k] for k in range(len(content_params))) 
                    for i in range(len(node_params)))
    total_transmissions = sum(sum(result['allocations']['m'][i][k] for k in range(len(content_params))) 
                             for i in range(len(node_params)))
    
    print(f"\n总分配量:")
    print(f"总存储时间: {total_storage:.2f}")
    print(f"总命中次数: {total_hits}")
    print(f"总传输次数: {total_transmissions}")
    
    # 计算总收益
    total_revenue = (result['prices']['p_s'] * total_storage + 
                    sum(result['prices']['p_hit'][k] * sum(result['allocations']['n'][i][k] for i in range(len(node_params))) 
                        for k in range(len(content_params))) +
                    sum(result['prices']['p_tr'][k] * sum(result['allocations']['m'][i][k] for i in range(len(node_params))) 
                        for k in range(len(content_params))))
    
    print(f"总收益: {total_revenue:.2f}")

if __name__ == "__main__":
    main()