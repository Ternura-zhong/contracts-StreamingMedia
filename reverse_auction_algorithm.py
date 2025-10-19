"""
对比算法C: 多单元逆向拍卖 (统一清算价; 可选VCG)
Multi-unit Reverse Auction Algorithm with Uniform Clearing Price

基于图片中的伪代码实现多单元逆向拍卖算法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import json


class NodeParameters:
    """节点参数类"""
    def __init__(self, node_id: int, r_b: float, r_c: float, R: Dict[int, float], Phi: float):
        self.node_id = node_id
        self.r_b = r_b  # 带宽容量
        self.r_c = r_c  # 计算容量
        self.R = R      # 传输速率映射 {content_id: rate}
        self.Phi = Phi  # 计算效率


class ContentParameters:
    """内容参数类"""
    def __init__(self, content_id: int, S: float, gamma: float):
        self.content_id = content_id
        self.S = S      # 内容大小
        self.gamma = gamma  # 计算复杂度


class DemandParameters:
    """需求参数类"""
    def __init__(self, content_id: int, D_hit: float, D_tr: float):
        self.content_id = content_id
        self.D_hit = D_hit  # 命中服务需求
        self.D_tr = D_tr    # 转码任务需求


class BidParameters:
    """报价参数类"""
    def __init__(self, node_id: int, content_id: int, a_hit: float, cap_hit: float, 
                 a_tr: float, cap_tr: float):
        self.node_id = node_id
        self.content_id = content_id
        self.a_hit = a_hit      # 命中服务报价
        self.cap_hit = cap_hit  # 命中服务容量
        self.a_tr = a_tr        # 转码任务报价
        self.cap_tr = cap_tr    # 转码任务容量


class ReverseAuctionAlgorithm:
    """多单元逆向拍卖算法"""
    
    def __init__(self, T: float = 100.0, payment_mechanism: str = "uniform"):
        self.T = T  # 时间窗口
        self.payment_mechanism = payment_mechanism  # 支付机制: "uniform" 或 "vcg"
        self.used_band = {}  # 各节点已使用带宽
        self.used_comp = {}  # 各节点已使用计算资源
        
    def reset_resource_usage(self, node_params: List[NodeParameters]):
        """重置资源使用情况"""
        self.used_band = {node.node_id: 0.0 for node in node_params}
        self.used_comp = {node.node_id: 0.0 for node in node_params}
    
    def allocate_hit_services(self, 
                            content_params: List[ContentParameters],
                            demand_params: List[DemandParameters],
                            bid_params: List[BidParameters],
                            node_params: List[NodeParameters]) -> Tuple[Dict, Dict, Dict]:
        """
        命中服务分配算法
        
        Returns:
            allocations: {node_id: {content_id: allocation}}
            prices: {content_id: clearing_price}
            unmet_demands: {content_id: unmet_amount}
        """
        allocations = {}
        prices = {}
        unmet_demands = {}
        
        # 初始化分配结果
        for node in node_params:
            allocations[node.node_id] = {}
        
        for content in content_params:
            # 找到该内容的需求
            demand = next((d for d in demand_params if d.content_id == content.content_id), None)
            if not demand:
                continue
                
            D_k_hit = demand.D_hit
            
            # 构造候选列表 L_hit(k) = {(i, a_{i,k}^hit, cap_{i,k}^hit)}
            candidates = []
            for bid in bid_params:
                if bid.content_id == content.content_id:
                    node = next((n for n in node_params if n.node_id == bid.node_id), None)
                    if node:
                        candidates.append((bid.node_id, bid.a_hit, bid.cap_hit))
            
            # 按 a_{i,k}^hit 升序排序
            candidates.sort(key=lambda x: x[1])
            
            # 分配逻辑
            rem = D_k_hit
            p_k_hit = 0.0
            
            for node_id, a, cap in candidates:
                node = next((n for n in node_params if n.node_id == node_id), None)
                if not node:
                    continue
                
                # 计算最大可分配量
                # max_assign ← min( cap, floor( (r_i^(b)T - used_band_i) / (S_k / R_{i,k})))
                if content.content_id in node.R:
                    max_assign = min(cap, 
                                   int((node.r_b * self.T - self.used_band[node_id]) / 
                                       (content.S / node.R[content.content_id])))
                else:
                    max_assign = 0
                
                # 实际分配量
                take = min(rem, max_assign)
                
                if take > 0:
                    allocations[node_id][content.content_id] = take
                    
                    # 更新已使用带宽
                    if content.content_id in node.R:
                        self.used_band[node_id] += take * (content.S / node.R[content.content_id])
                    
                    # 更新剩余需求
                    rem -= take
                    
                    # 设置清算价
                    p_k_hit = a
                    
                    # 如果需求满足，跳出循环
                    if rem == 0:
                        break
            
            prices[content.content_id] = p_k_hit
            unmet_demands[content.content_id] = max(0, rem)
        
        return allocations, prices, unmet_demands
    
    def allocate_transcoding_tasks(self,
                                 content_params: List[ContentParameters],
                                 demand_params: List[DemandParameters],
                                 bid_params: List[BidParameters],
                                 node_params: List[NodeParameters]) -> Tuple[Dict, Dict, Dict]:
        """
        转码任务分配算法
        
        Returns:
            allocations: {node_id: {content_id: allocation}}
            prices: {content_id: clearing_price}
            unmet_demands: {content_id: unmet_amount}
        """
        allocations = {}
        prices = {}
        unmet_demands = {}
        
        # 初始化分配结果
        for node in node_params:
            allocations[node.node_id] = {}
        
        for content in content_params:
            # 找到该内容的需求
            demand = next((d for d in demand_params if d.content_id == content.content_id), None)
            if not demand:
                continue
                
            D_k_tr = demand.D_tr
            
            # 构造候选列表 L_tr(k) = {(i, a_{i,k}^tr, cap_{i,k}^tr)}
            candidates = []
            for bid in bid_params:
                if bid.content_id == content.content_id:
                    node = next((n for n in node_params if n.node_id == bid.node_id), None)
                    if node:
                        candidates.append((bid.node_id, bid.a_tr, bid.cap_tr))
            
            # 按 a_{i,k}^tr 升序排序
            candidates.sort(key=lambda x: x[1])
            
            # 分配逻辑
            rem = D_k_tr
            p_k_tr = 0.0
            
            for node_id, a, cap in candidates:
                node = next((n for n in node_params if n.node_id == node_id), None)
                if not node:
                    continue
                
                # 计算最大可分配量
                # max_assign ← min( cap, floor( (r_i^(c)T - used_comp_i) / (γ_k S_k / Φ_i) ))
                max_assign = min(cap, 
                               int((node.r_c * self.T - self.used_comp[node_id]) / 
                                   (content.gamma * content.S / node.Phi)))
                
                # 实际分配量
                take = min(rem, max_assign)
                
                if take > 0:
                    allocations[node_id][content.content_id] = take
                    
                    # 更新已使用计算资源
                    self.used_comp[node_id] += take * (content.gamma * content.S / node.Phi)
                    
                    # 更新剩余需求
                    rem -= take
                    
                    # 设置清算价
                    p_k_tr = a
                    
                    # 如果需求满足，跳出循环
                    if rem == 0:
                        break
            
            prices[content.content_id] = p_k_tr
            unmet_demands[content.content_id] = max(0, rem)
        
        return allocations, prices, unmet_demands
    
    def calculate_payments(self, 
                          hit_allocations: Dict,
                          tr_allocations: Dict,
                          hit_prices: Dict,
                          tr_prices: Dict,
                          bid_params: List[BidParameters] = None,
                          content_params: List[ContentParameters] = None,
                          demand_params: List[DemandParameters] = None,
                          node_params: List[NodeParameters] = None) -> Dict:
        """
        计算支付（支持统一清算价和VCG机制）
        
        对每个节点 i: 
        - 统一清算价: 支付_i ← Σ_k ( p_k^hit * x_{i,k} + p_k^tr * y_{i,k} )
        - VCG机制: 支付_i ← 边际贡献价值
        """
        if self.payment_mechanism == "vcg" and all(param is not None for param in [bid_params, content_params, demand_params, node_params]):
            return self._calculate_vcg_payments(hit_allocations, tr_allocations, 
                                              bid_params, content_params, demand_params, node_params)
        else:
            return self._calculate_uniform_payments(hit_allocations, tr_allocations, hit_prices, tr_prices)
    
    def _calculate_uniform_payments(self, hit_allocations: Dict, tr_allocations: Dict, 
                                   hit_prices: Dict, tr_prices: Dict) -> Dict:
        """计算统一清算价支付"""
        payments = {}
        
        # 获取所有节点ID
        all_nodes = set()
        for node_id in hit_allocations.keys():
            all_nodes.add(node_id)
        for node_id in tr_allocations.keys():
            all_nodes.add(node_id)
        
        for node_id in all_nodes:
            payment = 0.0
            
            # 命中服务支付
            if node_id in hit_allocations:
                for content_id, allocation in hit_allocations[node_id].items():
                    if content_id in hit_prices:
                        payment += hit_prices[content_id] * allocation
            
            # 转码任务支付
            if node_id in tr_allocations:
                for content_id, allocation in tr_allocations[node_id].items():
                    if content_id in tr_prices:
                        payment += tr_prices[content_id] * allocation
            
            payments[node_id] = payment
        
        return payments
    
    def _calculate_vcg_payments(self, hit_allocations: Dict, tr_allocations: Dict,
                               bid_params: List[BidParameters], content_params: List[ContentParameters],
                               demand_params: List[DemandParameters], node_params: List[NodeParameters]) -> Dict:
        """
        计算VCG支付
        VCG支付 = 其他节点的总价值损失
        """
        payments = {}
        all_nodes = set()
        
        # 获取所有节点ID
        for node_id in hit_allocations.keys():
            all_nodes.add(node_id)
        for node_id in tr_allocations.keys():
            all_nodes.add(node_id)
        
        # 计算总社会福利（包含所有节点）
        total_welfare = self._calculate_social_welfare(hit_allocations, tr_allocations, bid_params, demand_params)
        
        for target_node in all_nodes:
            # 计算排除该节点后的最优分配和社会福利
            filtered_bids = [bid for bid in bid_params if bid.node_id != target_node]
            filtered_nodes = [node for node in node_params if node.node_id != target_node]
            
            if filtered_bids and filtered_nodes:
                # 重新分配（排除目标节点）
                hit_alloc_without, tr_alloc_without, _, _ = self._allocate_without_node(
                    content_params, demand_params, filtered_bids, filtered_nodes)
                
                welfare_without = self._calculate_social_welfare(hit_alloc_without, tr_alloc_without, 
                                                               filtered_bids, demand_params)
                
                # VCG支付 = 其他节点的价值损失
                payments[target_node] = max(0, welfare_without - (total_welfare - self._get_node_welfare(
                    target_node, hit_allocations, tr_allocations, bid_params)))
            else:
                payments[target_node] = 0.0
        
        return payments
    
    def _calculate_social_welfare(self, hit_allocations: Dict, tr_allocations: Dict,
                                 bid_params: List[BidParameters], demand_params: List[DemandParameters]) -> float:
        """计算社会福利"""
        welfare = 0.0
        
        # 计算满足的需求价值（假设单位需求价值为100）
        for demand in demand_params:
            content_id = demand.content_id
            
            # 命中服务满足的需求
            hit_satisfied = sum(hit_allocations.get(node_id, {}).get(content_id, 0) 
                              for node_id in hit_allocations.keys())
            hit_satisfied = min(hit_satisfied, demand.D_hit)
            welfare += hit_satisfied * 100  # 假设单位命中价值为100
            
            # 转码任务满足的需求
            tr_satisfied = sum(tr_allocations.get(node_id, {}).get(content_id, 0) 
                             for node_id in tr_allocations.keys())
            tr_satisfied = min(tr_satisfied, demand.D_tr)
            welfare += tr_satisfied * 80   # 假设单位转码价值为80
        
        # 减去节点成本
        for bid in bid_params:
            node_id = bid.node_id
            content_id = bid.content_id
            
            hit_alloc = hit_allocations.get(node_id, {}).get(content_id, 0)
            tr_alloc = tr_allocations.get(node_id, {}).get(content_id, 0)
            
            welfare -= hit_alloc * bid.a_hit + tr_alloc * bid.a_tr
        
        return welfare
    
    def _get_node_welfare(self, node_id: int, hit_allocations: Dict, tr_allocations: Dict,
                         bid_params: List[BidParameters]) -> float:
        """计算特定节点的福利贡献"""
        welfare = 0.0
        
        for bid in bid_params:
            if bid.node_id == node_id:
                content_id = bid.content_id
                hit_alloc = hit_allocations.get(node_id, {}).get(content_id, 0)
                tr_alloc = tr_allocations.get(node_id, {}).get(content_id, 0)
                
                # 节点提供的价值减去成本
                welfare += hit_alloc * 100 + tr_alloc * 80 - (hit_alloc * bid.a_hit + tr_alloc * bid.a_tr)
        
        return welfare
    
    def _allocate_without_node(self, content_params: List[ContentParameters],
                              demand_params: List[DemandParameters],
                              bid_params: List[BidParameters],
                              node_params: List[NodeParameters]) -> Tuple[Dict, Dict, Dict, Dict]:
        """在排除特定节点的情况下重新分配"""
        # 重置资源使用
        self.reset_resource_usage(node_params)
        
        # 重新分配
        hit_allocations, hit_prices, _ = self.allocate_hit_services(
            content_params, demand_params, bid_params, node_params)
        tr_allocations, tr_prices, _ = self.allocate_transcoding_tasks(
            content_params, demand_params, bid_params, node_params)
        
        return hit_allocations, tr_allocations, hit_prices, tr_prices
    
    def reverse_auction(self,
                      node_params: List[NodeParameters],
                      content_params: List[ContentParameters],
                      demand_params: List[DemandParameters],
                      bid_params: List[BidParameters]) -> Dict:
        """
        执行多单元逆向拍卖算法
        
        Returns:
            result: {
                'hit_allocations': 命中服务分配结果,
                'tr_allocations': 转码任务分配结果,
                'hit_prices': 命中服务清算价,
                'tr_prices': 转码任务清算价,
                'payments': 各节点支付,
                'unmet_hit_demands': 未满足的命中需求,
                'unmet_tr_demands': 未满足的转码需求,
                'total_payment': 总支付,
                'satisfaction_rate': 需求满足率
            }
        """
        start_time = time.time()
        
        # 重置资源使用情况
        self.reset_resource_usage(node_params)
        
        # 1. 命中服务分配
        hit_allocations, hit_prices, unmet_hit_demands = self.allocate_hit_services(
            content_params, demand_params, bid_params, node_params)
        
        # 2. 转码任务分配
        tr_allocations, tr_prices, unmet_tr_demands = self.allocate_transcoding_tasks(
            content_params, demand_params, bid_params, node_params)
        
        # 3. 计算支付
        payments = self.calculate_payments(
            hit_allocations, tr_allocations, hit_prices, tr_prices,
            bid_params, content_params, demand_params, node_params)
        
        # 4. 计算统计信息
        total_payment = sum(payments.values())
        
        # 计算需求满足率
        total_hit_demand = sum(d.D_hit for d in demand_params)
        total_tr_demand = sum(d.D_tr for d in demand_params)
        total_unmet_hit = sum(unmet_hit_demands.values())
        total_unmet_tr = sum(unmet_tr_demands.values())
        
        hit_satisfaction_rate = (total_hit_demand - total_unmet_hit) / total_hit_demand if total_hit_demand > 0 else 1.0
        tr_satisfaction_rate = (total_tr_demand - total_unmet_tr) / total_tr_demand if total_tr_demand > 0 else 1.0
        overall_satisfaction_rate = (hit_satisfaction_rate + tr_satisfaction_rate) / 2
        
        execution_time = time.time() - start_time
        
        result = {
            'hit_allocations': hit_allocations,
            'tr_allocations': tr_allocations,
            'hit_prices': hit_prices,
            'tr_prices': tr_prices,
            'payments': payments,
            'unmet_hit_demands': unmet_hit_demands,
            'unmet_tr_demands': unmet_tr_demands,
            'total_payment': total_payment,
            'hit_satisfaction_rate': hit_satisfaction_rate,
            'tr_satisfaction_rate': tr_satisfaction_rate,
            'overall_satisfaction_rate': overall_satisfaction_rate,
            'execution_time': execution_time,
            'num_nodes': len(node_params),
            'num_contents': len(content_params)
        }
        
        return result


def load_test_data():
    """加载测试数据"""
    # 读取CDN节点数据
    cdn_df = pd.read_csv('模拟动态生成/cdn_nodes.csv')
    
    # 读取视频元数据
    video_df = pd.read_csv('模拟动态生成/video_metadata.csv')
    
    return cdn_df, video_df


def create_test_scenario(cdn_df: pd.DataFrame, video_df: pd.DataFrame, 
                        num_nodes: int = 5, num_contents: int = 10) -> Tuple:
    """创建测试场景"""
    
    # 选择前num_nodes个节点
    selected_nodes = cdn_df.head(num_nodes)
    
    # 选择前num_contents个视频
    selected_videos = video_df.head(num_contents)
    
    # 创建节点参数
    node_params = []
    for _, node in selected_nodes.iterrows():
        # 随机生成传输速率映射
        R = {}
        for _, video in selected_videos.iterrows():
            R[video['video_id']] = np.random.uniform(10, 100)  # 10-100 Mbps
        
        node_param = NodeParameters(
            node_id=int(node['node_id']),
            r_b=float(node['total_bandwidth']),
            r_c=float(node['available_computing']),
            R=R,
            Phi=np.random.uniform(0.8, 1.2)  # 计算效率
        )
        node_params.append(node_param)
    
    # 创建内容参数
    content_params = []
    for _, video in selected_videos.iterrows():
        content_param = ContentParameters(
            content_id=int(video['video_id']),
            S=float(video['estimated_size_mb']) * 8,  # 转换为Mbps
            gamma=np.random.uniform(0.5, 2.0)  # 计算复杂度
        )
        content_params.append(content_param)
    
    # 创建需求参数
    demand_params = []
    for _, video in selected_videos.iterrows():
        demand_param = DemandParameters(
            content_id=int(video['video_id']),
            D_hit=np.random.uniform(10, 50),  # 命中服务需求
            D_tr=np.random.uniform(5, 25)     # 转码任务需求
        )
        demand_params.append(demand_param)
    
    # 创建报价参数
    bid_params = []
    for node in node_params:
        for content in content_params:
            bid_param = BidParameters(
                node_id=node.node_id,
                content_id=content.content_id,
                a_hit=np.random.uniform(1, 10),      # 命中服务报价
                cap_hit=np.random.uniform(5, 30),   # 命中服务容量
                a_tr=np.random.uniform(2, 15),       # 转码任务报价
                cap_tr=np.random.uniform(3, 20)      # 转码任务容量
            )
            bid_params.append(bid_param)
    
    return node_params, content_params, demand_params, bid_params


def print_results(result: Dict):
    """打印结果"""
    print("=" * 60)
    print("多单元逆向拍卖算法结果")
    print("=" * 60)
    
    print(f"节点数量: {result['num_nodes']}")
    print(f"内容数量: {result['num_contents']}")
    print(f"执行时间: {result['execution_time']:.4f} 秒")
    print(f"总支付: {result['total_payment']:.2f}")
    print(f"命中服务满足率: {result['hit_satisfaction_rate']:.2%}")
    print(f"转码任务满足率: {result['tr_satisfaction_rate']:.2%}")
    print(f"整体满足率: {result['overall_satisfaction_rate']:.2%}")
    
    print("\n各节点支付:")
    for node_id, payment in result['payments'].items():
        print(f"  节点 {node_id}: {payment:.2f}")
    
    print("\n命中服务清算价:")
    for content_id, price in result['hit_prices'].items():
        print(f"  内容 {content_id}: {price:.2f}")
    
    print("\n转码任务清算价:")
    for content_id, price in result['tr_prices'].items():
        print(f"  内容 {content_id}: {price:.2f}")
    
    print("\n未满足需求统计:")
    print("命中服务未满足:")
    for content_id, unmet in result['unmet_hit_demands'].items():
        if unmet > 0:
            print(f"  内容 {content_id}: {unmet:.2f}")
    
    print("转码任务未满足:")
    for content_id, unmet in result['unmet_tr_demands'].items():
        if unmet > 0:
            print(f"  内容 {content_id}: {unmet:.2f}")


if __name__ == "__main__":
    # 加载测试数据
    cdn_df, video_df = load_test_data()
    
    # 创建测试场景
    node_params, content_params, demand_params, bid_params = create_test_scenario(
        cdn_df, video_df, num_nodes=5, num_contents=10)
    
    print("=" * 80)
    print("多单元逆向拍卖算法对比测试")
    print("=" * 80)
    
    # 测试统一清算价机制
    print("\n1. 统一清算价机制 (Uniform Clearing Price)")
    print("-" * 50)
    algorithm_uniform = ReverseAuctionAlgorithm(T=100.0, payment_mechanism="uniform")
    result_uniform = algorithm_uniform.reverse_auction(
        node_params, content_params, demand_params, bid_params)
    print_results(result_uniform)
    
    # 测试VCG支付机制
    print("\n2. VCG支付机制 (Vickrey-Clarke-Groves)")
    print("-" * 50)
    algorithm_vcg = ReverseAuctionAlgorithm(T=100.0, payment_mechanism="vcg")
    result_vcg = algorithm_vcg.reverse_auction(
        node_params, content_params, demand_params, bid_params)
    print_results(result_vcg)
    
    # 对比分析
    print("\n3. 支付机制对比分析")
    print("-" * 50)
    print(f"统一清算价总支付: {result_uniform['total_payment']:.2f}")
    print(f"VCG机制总支付: {result_vcg['total_payment']:.2f}")
    print(f"支付差异: {abs(result_vcg['total_payment'] - result_uniform['total_payment']):.2f}")
    print(f"统一清算价满足率: {result_uniform['overall_satisfaction_rate']:.2%}")
    print(f"VCG机制满足率: {result_vcg['overall_satisfaction_rate']:.2%}")
    
    # 激励机制分析
    print(f"\n激励机制强度对比:")
    uniform_payments = list(result_uniform['payments'].values())
    vcg_payments = list(result_vcg['payments'].values())
    print(f"统一清算价支付方差: {np.var(uniform_payments):.2f}")
    print(f"VCG机制支付方差: {np.var(vcg_payments):.2f}")
    print(f"VCG机制提供了{'更强' if np.var(vcg_payments) > np.var(uniform_payments) else '更弱'}的差异化激励")
