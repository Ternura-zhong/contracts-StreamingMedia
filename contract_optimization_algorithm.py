"""
合同优化算法实现
基于现有代码框架，实现合同理论优化的CDN资源分配算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import ast
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ContractConfig:
    """合同优化配置"""
    a: float = 2.0  # 奖励计算参数
    b1: float = 1.5  # 奖励计算参数
    b2: float = 5.0  # 奖励计算参数
    c_base: float = 3.5  # 价格基础参数
    c1: float = 5.0  # 价格计算参数
    c2: float = 0.1  # 价格计算参数
    alpha: float = 1.0  # 服务效用参数
    beta: float = 1.0  # 服务效用参数
    fairness_weight: float = 1.0  # 公平性权重

class ContractOptimizationAlgorithm:
    """合同优化算法主类"""
    
    def __init__(self, config: ContractConfig):
        self.config = config
        self.optimization_history = []
        
    def load_data(self, data_paths: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
        """加载实验数据"""
        df_users = pd.read_csv(data_paths['users'])
        df_requests = pd.read_csv(data_paths['requests'])
        df_cdn = pd.read_csv(data_paths['cdn_nodes'])
        df_videos = pd.read_csv(data_paths['videos'])
        df_segments = pd.read_csv(data_paths['segments'])
        
        # 数据预处理
        df_videos['available_quality_level'] = df_videos['available_quality_level'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        return df_users, df_requests, df_cdn, df_videos, df_segments
    
    def init_cdn_status(self, df_cdn: pd.DataFrame) -> pd.DataFrame:
        """初始化CDN状态"""
        df_cdn['available_bandwidth'] = df_cdn['total_bandwidth'].astype(float)
        df_cdn['available_computing'] = df_cdn['total_computing_resources'].astype(float)
        df_cdn['cache_hit_rate'] = 0.0
        return df_cdn
    
    def classify_cdn_nodes(self, df_cdn: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """CDN节点分类"""
        features = ['cache_capacity', 'total_bandwidth', 'total_computing_resources']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_cdn[features])
        df_scaled = pd.DataFrame(scaled, columns=features)
        
        df_cdn['score'] = (df_scaled['cache_capacity'] + 
                          df_scaled['total_bandwidth'] + 
                          df_scaled['total_computing_resources'])
        
        df_cdn['node_type'] = pd.qcut(df_cdn['score'], q=n_bins, labels=False)
        return df_cdn
    
    def compute_popularity_dict(self, df_requests: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        """计算流行度字典"""
        freq = df_requests.groupby(['video_id', 'bandwidth']).size().to_dict()
        total = sum(freq.values())
        pr_dict = {(vid, int(q)): count / total for (vid, q), count in freq.items()}
        return pr_dict
    
    def init_reward_price(self, df_videos: pd.DataFrame, pr_dict: Dict[Tuple[int, int], float]) -> Tuple[Dict, Dict]:
        """初始化奖励和价格矩阵"""
        reward_matrix, price_matrix = {}, {}
        
        for _, row in df_videos.iterrows():
            vid = row['video_id']
            quality_levels = row['available_quality_level']
            if isinstance(quality_levels, str):
                quality_levels = ast.literal_eval(quality_levels)
            
            for q in quality_levels:
                q = int(q)
                R = q / 1e6  # Mbps
                P = pr_dict.get((vid, q), 1.0)
                
                reward = self.config.a * R**self.config.b1 + self.config.b2 * P
                price = self.config.c_base + self.config.c1 * R**self.config.c2
                
                reward_matrix[(vid, q)] = reward
                price_matrix[(vid, q)] = price
        
        return reward_matrix, price_matrix
    
    def compute_cp_utility(self, df_requests: pd.DataFrame, reward_matrix: Dict, 
                          price_matrix: Dict, theta_map: Dict = None) -> float:
        """计算CP效用"""
        u_cp = 0.0
        
        for _, row in df_requests.iterrows():
            k = row['video_id']
            q = int(row['bandwidth'])
            R = q / 1e6  # Mbps
            
            theta = theta_map.get((k, q), 1.0) if theta_map else 1.0
            service_utility = self.config.alpha * np.log(1 + self.config.beta * R)
            
            reward = reward_matrix.get((k, q), 1.0)
            price = price_matrix.get((k, q), 0.2)
            
            total_reward = theta * service_utility
            u_cp += total_reward - price
        
        return u_cp
    
    def compute_cdn_utility(self, df_cdn: pd.DataFrame) -> float:
        """计算CDN效用"""
        total_utility = 0.0
        for _, row in df_cdn.iterrows():
            hit_reward = row['cache_hit_rate'] * 5
            used_bandwidth = row['total_bandwidth'] - row['available_bandwidth']
            cost = 0.01 * used_bandwidth
            total_utility += hit_reward - cost
        return total_utility
    
    def compute_social_welfare(self, u_cp: float, u_cdn: float, fairness_penalty: float = 0.0) -> float:
        """计算社会福利"""
        return u_cp + u_cdn - fairness_penalty
    
    def handle_request(self, req: pd.Series, df_cdn: pd.DataFrame) -> pd.DataFrame:
        """处理用户请求"""
        vid, q = req['video_id'], int(req['bandwidth'])
        required_bw = q / 1e6
        candidates = df_cdn[df_cdn['available_bandwidth'] >= required_bw]
        
        if not candidates.empty:
            best = candidates.sort_values(by=['node_type', 'available_bandwidth'], 
                                        ascending=[True, False]).iloc[0]
            node_id = best['node_id']
            df_cdn.loc[df_cdn['node_id'] == node_id, 'available_bandwidth'] -= required_bw
            df_cdn.loc[df_cdn['node_id'] == node_id, 'cache_hit_rate'] = min(
                1.0, df_cdn.loc[df_cdn['node_id'] == node_id, 'cache_hit_rate'].values[0] + 0.01)
        
        return df_cdn
    
    def optimize_parameters(self, df_videos: pd.DataFrame, df_requests: pd.DataFrame, 
                           df_cdn: pd.DataFrame, pr_dict: Dict) -> Tuple[float, float, float]:
        """优化算法参数"""
        def objective(params):
            a, b1, b2 = params
            reward_matrix, price_matrix = self._compute_matrices_with_params(
                df_videos, pr_dict, a, b1, b2)
            
            u_cp = self.compute_cp_utility(df_requests, reward_matrix, price_matrix)
            u_cdn = self.compute_cdn_utility(df_cdn)
            sw = self.compute_social_welfare(u_cp, u_cdn)
            
            self.optimization_history.append((a, b1, b2, sw))
            return -sw  # 最小化目标函数
        
        result = gp_minimize(
            func=objective,
            dimensions=[
                Real(0.1, 10.0, name='a'),
                Real(1.0, 3.0, name='b1'),
                Real(0.0, 10.0, name='b2')
            ],
            acq_func='EI',
            n_calls=30,
            random_state=42
        )
        
        return result.x
    
    def _compute_matrices_with_params(self, df_videos: pd.DataFrame, pr_dict: Dict,
                                    a: float, b1: float, b2: float) -> Tuple[Dict, Dict]:
        """使用指定参数计算矩阵"""
        reward_matrix, price_matrix = {}, {}
        
        for _, row in df_videos.iterrows():
            vid = row['video_id']
            quality_levels = row['available_quality_level']
            if isinstance(quality_levels, str):
                quality_levels = ast.literal_eval(quality_levels)
            
            for q in quality_levels:
                q = int(q)
                R = q / 1e6
                P = pr_dict.get((vid, q), 1.0)
                
                reward = a * R**b1 + b2 * P
                price = self.config.c_base + self.config.c1 * R**self.config.c2
                
                reward_matrix[(vid, q)] = reward
                price_matrix[(vid, q)] = price
        
        return reward_matrix, price_matrix
    
    def run_experiment(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """运行完整实验"""
        print("开始运行合同优化算法...")
        
        # 加载数据
        df_users, df_requests, df_cdn, df_videos, df_segments = self.load_data(data_paths)
        
        # 初始化CDN状态
        df_cdn = self.init_cdn_status(df_cdn)
        df_cdn = self.classify_cdn_nodes(df_cdn)
        
        # 计算流行度
        pr_dict = self.compute_popularity_dict(df_requests)
        
        # 优化参数
        print("正在优化算法参数...")
        best_a, best_b1, best_b2 = self.optimize_parameters(df_videos, df_requests, df_cdn, pr_dict)
        
        print(f"最优参数: a={best_a:.4f}, b1={best_b1:.4f}, b2={best_b2:.4f}")
        
        # 使用最优参数计算最终结果
        reward_matrix, price_matrix = self._compute_matrices_with_params(
            df_videos, pr_dict, best_a, best_b1, best_b2)
        
        # 处理请求
        for _, req in df_requests.iterrows():
            df_cdn = self.handle_request(req, df_cdn)
        
        # 计算最终效用
        u_cp = self.compute_cp_utility(df_requests, reward_matrix, price_matrix)
        u_cdn = self.compute_cdn_utility(df_cdn)
        welfare = self.compute_social_welfare(u_cp, u_cdn)
        
        print(f"最终结果: U_CP={u_cp:.2f}, U_CDN={u_cdn:.2f}, Social Welfare={welfare:.2f}")
        
        return {
            'optimal_params': (best_a, best_b1, best_b2),
            'utilities': (u_cp, u_cdn, welfare),
            'reward_matrix': reward_matrix,
            'price_matrix': price_matrix,
            'optimization_history': self.optimization_history,
            'cdn_status': df_cdn
        }
    
    def plot_optimization_history(self, save_path: str = None):
        """绘制优化历史"""
        if not self.optimization_history:
            print("没有优化历史数据")
            return
        
        history = np.array(self.optimization_history)
        
        plt.figure(figsize=(15, 5))
        
        # 参数变化
        plt.subplot(1, 3, 1)
        plt.plot(history[:, 0], label='a')
        plt.plot(history[:, 1], label='b1')
        plt.plot(history[:, 2], label='b2')
        plt.xlabel('迭代次数')
        plt.ylabel('参数值')
        plt.title('参数优化过程')
        plt.legend()
        plt.grid(True)
        
        # 社会福利变化
        plt.subplot(1, 3, 2)
        plt.plot(-history[:, 3], label='Social Welfare')
        plt.xlabel('迭代次数')
        plt.ylabel('社会福利')
        plt.title('社会福利优化过程')
        plt.legend()
        plt.grid(True)
        
        # 参数空间
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(history[:, 0], history[:, 1], c=-history[:, 3], cmap='viridis')
        plt.colorbar(scatter, label='Social Welfare')
        plt.xlabel('参数 a')
        plt.ylabel('参数 b1')
        plt.title('参数空间搜索')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reward_price_analysis(self, reward_matrix: Dict, price_matrix: Dict, 
                                  save_path: str = None):
        """绘制奖励价格分析"""
        # 提取数据
        qualities = sorted(set(int(k[1]) for k in reward_matrix.keys()))
        sample_vid = next(iter(set(k[0] for k in reward_matrix.keys())))
        
        rewards = [reward_matrix.get((sample_vid, q), 0) for q in qualities]
        prices = [price_matrix.get((sample_vid, q), 0) for q in qualities]
        cp_utils = [r - p for r, p in zip(rewards, prices)]
        
        plt.figure(figsize=(12, 4))
        
        # 奖励曲线
        plt.subplot(1, 3, 1)
        plt.plot(qualities, rewards, marker='o', color='blue', label='Reward')
        plt.xlabel("Quality Level (bps)")
        plt.ylabel("Reward Value")
        plt.title("Reward vs. Quality Level")
        plt.grid(True)
        plt.legend()
        
        # 价格曲线
        plt.subplot(1, 3, 2)
        plt.plot(qualities, prices, marker='s', color='orange', label='Price')
        plt.xlabel("Quality Level (bps)")
        plt.ylabel("Price Value")
        plt.title("Price vs. Quality Level")
        plt.grid(True)
        plt.legend()
        
        # CP效用曲线
        plt.subplot(1, 3, 3)
        plt.plot(qualities, cp_utils, marker='^', color='green', label='CP Utility')
        plt.xlabel("Quality Level (bps)")
        plt.ylabel("Utility Value")
        plt.title("CP Utility vs. Quality Level")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 创建配置
    config = ContractConfig()
    
    # 创建算法实例
    algorithm = ContractOptimizationAlgorithm(config)
    
    # 数据路径
    data_paths = {
        'users': 'user_devices.csv',
        'requests': 'requests_data.csv',
        'cdn_nodes': 'cdn_nodes.csv',
        'videos': 'video_metadata.csv',
        'segments': 'segments_data.csv'
    }
    
    # 运行实验
    results = algorithm.run_experiment(data_paths)
    
    # 绘制结果
    algorithm.plot_optimization_history('contract_optimization_history.png')
    algorithm.plot_reward_price_analysis(results['reward_matrix'], 
                                       results['price_matrix'],
                                       'contract_reward_price_analysis.png')
    
    return results

if __name__ == "__main__":
    results = main()







