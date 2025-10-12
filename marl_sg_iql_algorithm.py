"""
多智能体马尔可夫随机博弈 (独立Q学习版 MARL-SG)
Multi-Agent Markov Stochastic Game (Independent Q-learning version MARL-SG)

基于图片描述的算法实现，包含状态、动作、奖励函数和独立Q学习算法
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，绘图功能将不可用")


@dataclass
class State:
    """状态 s_t: 可观测指标的聚合"""
    cache_occupancy: Dict[int, float]  # 缓存占用 (各节点)
    bandwidth_remaining: Dict[int, float]  # 带宽剩余 (各节点)
    computation_remaining: Dict[int, float]  # 计算剩余 (各节点)
    content_service_counts: Dict[int, Dict[int, int]]  # 各内容待服务计数估计 (节点, 内容)
    link_rates: Dict[Tuple[int, int], float]  # 链路速率 R_{i,k}
    window_progress: float  # 统计窗口进度
    timestamp: int  # 时间戳


@dataclass
class Action:
    """动作 a_{i,t}: 节点i在下一微时隙的选择"""
    accept_hit_requests: Dict[int, bool]  # 是否接收命中请求(按内容k)
    perform_transcoding: Dict[int, bool]  # 是否执行转码(按内容k)
    cache_admission_eviction: Dict[int, bool]  # 是否准入/驱逐缓存(按内容k)


@dataclass
class Reward:
    """回报 r_{i,t}: 奖励函数"""
    storage_reward: float  # p_s * Δz_{i,t}^{(s)}
    hit_reward: float  # Σ_k p_k^{hit} * Δn_{i,k,t}
    transcoding_reward: float  # Σ_k p_k^{tr} * Δm_{i,k,t}
    cost_penalty: float  # -ΔC_i(Δz_{i,t}|θ_i)
    total_reward: float  # 总奖励


class MARLSGIQLAgent:
    """多智能体MARL-SG独立Q学习智能体"""
    
    def __init__(self, agent_id: int, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Q表初始化
        self.q_table = {}
        self.episode_rewards = []
        self.episode_steps = []
        
    def get_state_key(self, state: State) -> str:
        """将状态转换为Q表的键"""
        # 简化的状态编码，实际应用中可能需要更复杂的编码
        cache_sum = sum(state.cache_occupancy.values())
        bandwidth_sum = sum(state.bandwidth_remaining.values())
        computation_sum = sum(state.computation_remaining.values())
        
        # 离散化状态空间
        cache_bin = int(cache_sum * 10) // 1
        bandwidth_bin = int(bandwidth_sum * 10) // 1
        computation_bin = int(computation_sum * 10) // 1
        progress_bin = int(state.window_progress * 10) // 1
        
        return f"{cache_bin}_{bandwidth_bin}_{computation_bin}_{progress_bin}"
    
    def get_action_key(self, action: Action) -> str:
        """将动作转换为Q表的键"""
        hit_str = "".join([str(int(v)) for v in action.accept_hit_requests.values()])
        trans_str = "".join([str(int(v)) for v in action.perform_transcoding.values()])
        cache_str = "".join([str(int(v)) for v in action.cache_admission_eviction.values()])
        return f"{hit_str}_{trans_str}_{cache_str}"
    
    def get_q_value(self, state: State, action: Action) -> float:
        """获取Q值"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            
        return self.q_table[state_key][action_key]
    
    def update_q_value(self, state: State, action: Action, reward: float, next_state: State):
        """更新Q值"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Q学习更新规则
        current_q = self.q_table[state_key][action_key]
        max_next_q = self.get_max_q_value(next_state)
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action_key] = new_q
    
    def get_max_q_value(self, state: State) -> float:
        """获取状态下的最大Q值"""
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table or not self.q_table[state_key]:
            return 0.0
        
        return max(self.q_table[state_key].values())
    
    def get_best_action(self, state: State, feasible_actions: List[Action]) -> Action:
        """获取最佳动作"""
        if not feasible_actions:
            return self.get_random_action()
        
        best_action = None
        best_q_value = float('-inf')
        
        for action in feasible_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action if best_action else feasible_actions[0]
    
    def get_random_action(self) -> Action:
        """获取随机动作"""
        # 简化的随机动作生成
        return Action(
            accept_hit_requests={i: random.choice([True, False]) for i in range(3)},
            perform_transcoding={i: random.choice([True, False]) for i in range(3)},
            cache_admission_eviction={i: random.choice([True, False]) for i in range(3)}
        )
    
    def epsilon_greedy_action(self, state: State, feasible_actions: List[Action]) -> Action:
        """ε-贪心动作选择"""
        if random.random() < self.epsilon:
            return random.choice(feasible_actions) if feasible_actions else self.get_random_action()
        else:
            return self.get_best_action(state, feasible_actions)
    
    def decay_epsilon(self):
        """衰减ε值"""
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)


class MARLSGIQLEnvironment:
    """MARL-SG环境"""
    
    def __init__(self, num_nodes: int, num_contents: int, T: float = 100.0):
        self.num_nodes = num_nodes
        self.num_contents = num_contents
        self.T = T
        
        # 环境参数
        self.node_capacities = {
            'storage': {i: random.uniform(50, 100) for i in range(num_nodes)},
            'bandwidth': {i: random.uniform(20, 50) for i in range(num_nodes)},
            'computation': {i: random.uniform(10, 30) for i in range(num_nodes)}
        }
        
        self.content_params = {
            'size': {k: random.uniform(1, 10) for k in range(num_contents)},
            'complexity': {k: random.uniform(0.5, 2.0) for k in range(num_contents)},
            'arrival_rate': {k: random.uniform(0.1, 0.5) for k in range(num_contents)}
        }
        
        # 奖励参数
        self.reward_params = {
            'storage_price': 1.0,
            'hit_prices': {k: random.uniform(0.5, 2.0) for k in range(num_contents)},
            'transcoding_prices': {k: random.uniform(1.0, 3.0) for k in range(num_contents)},
            'cost_factor': 0.1
        }
        
        # 当前状态
        self.current_state = self.reset()
        
    def reset(self) -> State:
        """重置环境"""
        self.current_state = State(
            cache_occupancy={i: 0.0 for i in range(self.num_nodes)},
            bandwidth_remaining={i: self.node_capacities['bandwidth'][i] for i in range(self.num_nodes)},
            computation_remaining={i: self.node_capacities['computation'][i] for i in range(self.num_nodes)},
            content_service_counts={i: {k: 0 for k in range(self.num_contents)} for i in range(self.num_nodes)},
            link_rates={(i, k): random.uniform(1, 5) for i in range(self.num_nodes) for k in range(self.num_contents)},
            window_progress=0.0,
            timestamp=0
        )
        return self.current_state
    
    def get_feasible_actions(self, agent_id: int, state: State) -> List[Action]:
        """获取可行动作集合（满足C1-C3约束）"""
        feasible_actions = []
        
        # 生成所有可能的动作组合
        for hit_combo in self._generate_combinations(self.num_contents):
            for trans_combo in self._generate_combinations(self.num_contents):
                for cache_combo in self._generate_combinations(self.num_contents):
                    action = Action(
                        accept_hit_requests={k: hit_combo[k] for k in range(self.num_contents)},
                        perform_transcoding={k: trans_combo[k] for k in range(self.num_contents)},
                        cache_admission_eviction={k: cache_combo[k] for k in range(self.num_contents)}
                    )
                    
                    # 检查约束条件
                    if self._check_constraints(agent_id, state, action):
                        feasible_actions.append(action)
        
        return feasible_actions if feasible_actions else [self._get_default_action()]
    
    def _generate_combinations(self, num_items: int) -> List[List[bool]]:
        """生成布尔组合"""
        if num_items <= 3:  # 限制组合数量以避免组合爆炸
            combinations = []
            for i in range(2 ** num_items):
                combo = [(i >> j) & 1 for j in range(num_items)]
                combinations.append([bool(b) for b in combo])
            return combinations
        else:
            # 对于大量内容，只生成部分组合
            return [[random.choice([True, False]) for _ in range(num_items)] for _ in range(8)]
    
    def _check_constraints(self, agent_id: int, state: State, action: Action) -> bool:
        """检查约束条件C1-C3"""
        # C1: 存储约束
        storage_used = sum(
            self.content_params['size'][k] for k in range(self.num_contents)
            if action.cache_admission_eviction.get(k, False)
        )
        if storage_used > self.node_capacities['storage'][agent_id]:
            return False
        
        # C2: 带宽约束
        bandwidth_used = sum(
            self.content_params['size'][k] / self.current_state.link_rates.get((agent_id, k), 1.0)
            for k in range(self.num_contents)
            if action.accept_hit_requests.get(k, False)
        )
        if bandwidth_used > state.bandwidth_remaining[agent_id]:
            return False
        
        # C3: 计算约束
        computation_used = sum(
            self.content_params['complexity'][k] * self.content_params['size'][k]
            for k in range(self.num_contents)
            if action.perform_transcoding.get(k, False)
        )
        if computation_used > state.computation_remaining[agent_id]:
            return False
        
        return True
    
    def _get_default_action(self) -> Action:
        """获取默认动作"""
        return Action(
            accept_hit_requests={k: False for k in range(self.num_contents)},
            perform_transcoding={k: False for k in range(self.num_contents)},
            cache_admission_eviction={k: False for k in range(self.num_contents)}
        )
    
    def step(self, actions: Dict[int, Action]) -> Tuple[Dict[int, Reward], State]:
        """执行动作并返回奖励和新状态"""
        rewards = {}
        
        # 计算每个智能体的奖励
        for agent_id, action in actions.items():
            reward = self._calculate_reward(agent_id, action)
            rewards[agent_id] = reward
        
        # 更新环境状态
        next_state = self._update_state(actions)
        
        return rewards, next_state
    
    def _calculate_reward(self, agent_id: int, action: Action) -> Reward:
        """计算奖励函数"""
        # 存储奖励
        storage_delta = sum(
            self.content_params['size'][k] for k in range(self.num_contents)
            if action.cache_admission_eviction.get(k, False)
        )
        storage_reward = self.reward_params['storage_price'] * storage_delta
        
        # 命中奖励
        hit_reward = sum(
            self.reward_params['hit_prices'][k] for k in range(self.num_contents)
            if action.accept_hit_requests.get(k, False)
        )
        
        # 转码奖励
        transcoding_reward = sum(
            self.reward_params['transcoding_prices'][k] for k in range(self.num_contents)
            if action.perform_transcoding.get(k, False)
        )
        
        # 成本惩罚
        cost_penalty = self.reward_params['cost_factor'] * (
            storage_delta + hit_reward + transcoding_reward
        )
        
        total_reward = storage_reward + hit_reward + transcoding_reward - cost_penalty
        
        return Reward(
            storage_reward=storage_reward,
            hit_reward=hit_reward,
            transcoding_reward=transcoding_reward,
            cost_penalty=cost_penalty,
            total_reward=total_reward
        )
    
    def _update_state(self, actions: Dict[int, Action]) -> State:
        """更新环境状态"""
        new_state = State(
            cache_occupancy={},
            bandwidth_remaining={},
            computation_remaining={},
            content_service_counts={},
            link_rates=self.current_state.link_rates,
            window_progress=min(1.0, self.current_state.window_progress + 0.01),
            timestamp=self.current_state.timestamp + 1
        )
        
        # 更新各节点状态
        for agent_id in range(self.num_nodes):
            action = actions.get(agent_id, self._get_default_action())
            
            # 更新缓存占用
            cache_delta = sum(
                self.content_params['size'][k] for k in range(self.num_contents)
                if action.cache_admission_eviction.get(k, False)
            )
            new_state.cache_occupancy[agent_id] = min(
                self.node_capacities['storage'][agent_id],
                self.current_state.cache_occupancy[agent_id] + cache_delta
            )
            
            # 更新剩余带宽
            bandwidth_used = sum(
                self.content_params['size'][k] / self.current_state.link_rates.get((agent_id, k), 1.0)
                for k in range(self.num_contents)
                if action.accept_hit_requests.get(k, False)
            )
            new_state.bandwidth_remaining[agent_id] = max(
                0, self.current_state.bandwidth_remaining[agent_id] - bandwidth_used
            )
            
            # 更新剩余计算
            computation_used = sum(
                self.content_params['complexity'][k] * self.content_params['size'][k]
                for k in range(self.num_contents)
                if action.perform_transcoding.get(k, False)
            )
            new_state.computation_remaining[agent_id] = max(
                0, self.current_state.computation_remaining[agent_id] - computation_used
            )
            
            # 更新服务计数
            new_state.content_service_counts[agent_id] = {
                k: self.current_state.content_service_counts[agent_id][k] + 
                  (1 if action.accept_hit_requests.get(k, False) else 0)
                for k in range(self.num_contents)
            }
        
        self.current_state = new_state
        return new_state


class MARLSGIQLAlgorithm:
    """多智能体马尔可夫随机博弈独立Q学习算法"""
    
    def __init__(self, num_nodes: int, num_contents: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 T_steps: int = 100, T: float = 100.0):
        
        self.num_nodes = num_nodes
        self.num_contents = num_contents
        self.T_steps = T_steps
        self.T = T
        
        # 创建环境和智能体
        self.environment = MARLSGIQLEnvironment(num_nodes, num_contents, T)
        
        # 估算状态和动作空间大小
        state_size = num_nodes * 4 + num_contents  # 简化的状态空间
        action_size = 3 ** num_contents  # 简化的动作空间
        
        self.agents = [
            MARLSGIQLAgent(
                agent_id=i,
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay
            )
            for i in range(num_nodes)
        ]
        
        # 训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'convergence_metrics': [],
            'social_welfare': [],
            'hit_rates': []
        }
    
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """训练算法"""
        print(f"开始训练MARL-SG IQL算法，共{num_episodes}个回合...")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_steps = 0
            
            # 重置环境
            state = self.environment.reset()
            
            for step in range(self.T_steps):
                # 安全动作裁剪：剔除会违反(C1)-(C3)的动作
                actions = {}
                for agent in self.agents:
                    feasible_actions = self.environment.get_feasible_actions(agent.agent_id, state)
                    if feasible_actions:
                        action = agent.epsilon_greedy_action(state, feasible_actions)
                    else:
                        action = self.environment._get_default_action()
                    actions[agent.agent_id] = action
                
                # 执行联合动作
                rewards, next_state = self.environment.step(actions)
                
                # 更新Q值
                for agent in self.agents:
                    agent.update_q_value(state, actions[agent.agent_id], 
                                       rewards[agent.agent_id].total_reward, next_state)
                    episode_reward += rewards[agent.agent_id].total_reward
                
                # 状态转移
                state = next_state
                episode_steps += 1
            
            # 记录训练指标
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_steps'].append(episode_steps)
            
            # 计算社会福利和命中率
            social_welfare = episode_reward
            hit_rate = self._calculate_hit_rate(state)
            
            self.training_history['social_welfare'].append(social_welfare)
            self.training_history['hit_rates'].append(hit_rate)
            
            # 衰减ε值
            for agent in self.agents:
                agent.decay_epsilon()
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_hit_rate = np.mean(self.training_history['hit_rates'][-100:])
                print(f"回合 {episode + 1}: 平均奖励={avg_reward:.2f}, 平均命中率={avg_hit_rate:.3f}")
        
        training_time = time.time() - start_time
        
        # 导出策略
        policies = self._derive_policies()
        
        print(f"训练完成！用时: {training_time:.2f}秒")
        
        return {
            'policies': policies,
            'training_history': self.training_history,
            'training_time': training_time,
            'final_social_welfare': self.training_history['social_welfare'][-1],
            'final_hit_rate': self.training_history['hit_rates'][-1],
            'convergence_steps': len(self.training_history['episode_rewards'])
        }
    
    def _calculate_hit_rate(self, state: State) -> float:
        """计算命中率"""
        total_requests = sum(
            sum(node_counts.values()) for node_counts in state.content_service_counts.values()
        )
        if total_requests == 0:
            return 0.0
        
        hit_requests = sum(
            sum(node_counts.values()) for node_counts in state.content_service_counts.values()
        )
        return hit_requests / total_requests if total_requests > 0 else 0.0
    
    def _derive_policies(self) -> Dict[int, Any]:
        """导出策略"""
        policies = {}
        
        for agent in self.agents:
            policies[agent.agent_id] = {
                'q_table_size': len(agent.q_table),
                'total_q_entries': sum(len(actions) for actions in agent.q_table.values()),
                'epsilon': agent.epsilon
            }
        
        return policies
    
    def plot_training_results(self):
        """绘制训练结果"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法绘制训练结果")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('训练奖励曲线')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].grid(True)
        
        # 社会福利
        axes[0, 1].plot(self.training_history['social_welfare'])
        axes[0, 1].set_title('社会福利')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('社会福利')
        axes[0, 1].grid(True)
        
        # 命中率
        axes[1, 0].plot(self.training_history['hit_rates'])
        axes[1, 0].set_title('命中率')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].grid(True)
        
        # 收敛性
        if len(self.training_history['episode_rewards']) > 100:
            moving_avg = np.convolve(self.training_history['episode_rewards'], 
                                   np.ones(100)/100, mode='valid')
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title('收敛曲线 (100回合移动平均)')
            axes[1, 1].set_xlabel('回合')
            axes[1, 1].set_ylabel('平均奖励')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


def test_marl_sg_iql():
    """测试MARL-SG IQL算法"""
    print("=== MARL-SG IQL算法测试 ===")
    
    # 创建算法实例
    algorithm = MARLSGIQLAlgorithm(
        num_nodes=3,
        num_contents=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2,
        epsilon_decay=0.995,
        T_steps=50,
        T=100.0
    )
    
    # 训练算法
    result = algorithm.train(num_episodes=500)
    
    # 输出结果
    print(f"\n训练结果:")
    print(f"训练时间: {result['training_time']:.2f}秒")
    print(f"最终社会福利: {result['final_social_welfare']:.2f}")
    print(f"最终命中率: {result['final_hit_rate']:.3f}")
    print(f"收敛步数: {result['convergence_steps']}")
    
    # 策略信息
    print(f"\n策略信息:")
    for agent_id, policy_info in result['policies'].items():
        print(f"智能体 {agent_id}: Q表大小={policy_info['q_table_size']}, "
              f"总条目数={policy_info['total_q_entries']}, ε={policy_info['epsilon']:.3f}")
    
    # 绘制训练结果
    algorithm.plot_training_results()
    
    return result


if __name__ == "__main__":
    test_marl_sg_iql()
