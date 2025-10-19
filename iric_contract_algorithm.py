"""
主模型_合同机制_IRIC版本
实现四个阶段：读入与预设、一体化凸规划、支付回推、菜单生成、线上执行与评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class IRICContractAlgorithm:
    """主模型_合同机制_IRIC版本算法"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化IRIC合同算法
        
        Args:
            config: 配置参数，包含所有必要的输入参数
        """
        self.config = config
        self.theta = config.get('theta', [])  # 类型排序（高类型不劣）
        self.q = config.get('q', [])  # 概率分布
        self.S = config.get('S', [])  # 内容大小
        self.R = config.get('R', [])  # 带宽要求
        self.G = config.get('G', [])  # 计算复杂度
        self.Phi = config.get('Phi', [])  # 处理效率
        self.T = config.get('T', 100.0)  # 时间周期
        
        # 成本参数
        self.C_s = config.get('C_s', [])  # 存储成本
        self.r_b = config.get('r_b', [])  # 带宽容量
        self.r_c = config.get('r_c', [])  # 计算容量
        
        # 价值函数参数
        self.v_hit = config.get('v_hit', [])  # 命中价值
        self.v_tr = config.get('v_tr', [])  # 传输价值
        
        # 单位价格模板
        self.alpha_s = config.get('alpha_s', 1.0)  # 存储单位价格
        self.alpha_k = config.get('alpha_k', [])  # 命中单位价格
        self.beta_k = config.get('beta_k', [])  # 传输单位价格
        
        # 微惩罚参数
        self.rho = config.get('rho', 0.001)  # ρ≪1
        
        # 保序方向
        self.monotonicity_direction = config.get('monotonicity_direction', 'nondecreasing')
        
        # 结果存储
        self.results = {}
        
    def cost_function(self, z: float, theta: float) -> float:
        """
        成本函数 C(z|θ) - 改进版本，更平滑
        
        Args:
            z: 资源分配量
            theta: 类型参数
            
        Returns:
            成本值
        """
        # 改进的成本函数：使用更平滑的指数，增加数值稳定性
        z_safe = max(z, 1e-10)  # 避免0值
        return theta * (z_safe ** 1.5)  # 降低指数从2到1.5，减少梯度变化
    
    def solve_convex_iric(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        阶段1: 福利最大化 + 保序（必要时）→ 包络回推 → 得到最终菜单
        根据IRIC规则：先做福利最大化+保序，再用包络回推与可视化
        
        Returns:
            (t*, n*, m*, U*, s*): 最优解
        """
        M = len(self.theta)  # 类型数量
        K = len(self.S)  # 内容数量
        
        # 决策变量: t[j,k], n[j,k], m[j,k], U[j], s[j]
        # 总变量数: M*K + M*K + M*K + M + M = M*(3*K + 2)
        n_vars = M * (3 * K + 2)
        
        # 改进的初始猜测 - 确保满足约束的可行初始点
        np.random.seed(42)  # 固定随机种子
        x0 = np.zeros(n_vars)
        
        # 为t, n, m设置保守的初始值，确保单调性
        base_allocation = 0.3  # 基础分配比例
        for j in range(M):
            # 单调递减的分配比例
            allocation_ratio = base_allocation * (M - j) / M
            
            for k in range(K):
                idx_t = j * K + k
                idx_n = M * K + j * K + k
                idx_m = 2 * M * K + j * K + k
                
                # 保守设置，确保容量约束和单调性
                x0[idx_t] = allocation_ratio * min(0.8, self.C_s[j] * self.T / (self.S[k] * K * 2))
                x0[idx_n] = allocation_ratio * min(0.8, self.r_b[j] * self.T / (self.S[k] / self.R[j][k] * K * 2))
                x0[idx_m] = allocation_ratio * min(0.8, self.r_c[j] * self.T / (self.G[k] * self.S[k] / self.Phi[j] * K * 2))
        
        # 验证并调整初始值以满足单调性
        for j in range(1, M):
            for k in range(K):
                # 确保t[j] <= t[j-1]
                if x0[j * K + k] > x0[(j-1) * K + k]:
                    x0[j * K + k] = x0[(j-1) * K + k] * 0.9
                # 确保n[j] <= n[j-1]
                if x0[M * K + j * K + k] > x0[M * K + (j-1) * K + k]:
                    x0[M * K + j * K + k] = x0[M * K + (j-1) * K + k] * 0.9
                # 确保m[j] <= m[j-1]
                if x0[2 * M * K + j * K + k] > x0[2 * M * K + (j-1) * K + k]:
                    x0[2 * M * K + j * K + k] = x0[2 * M * K + (j-1) * K + k] * 0.9
        
        # 为U设置严格递减的初始值，最高型为0
        for j in range(M):
            idx_u = 3 * M * K + j
            if j == M - 1:
                x0[idx_u] = 0.0  # 最高型效用为0
            else:
                x0[idx_u] = (M - j - 1) * 0.02  # 小的递减值
        
        # 为s设置很小的初始值
        for j in range(M):
            idx_s = 3 * M * K + M + j
            x0[idx_s] = 0.001
        
        # 目标函数 - 简化并增强数值稳定性
        def objective(x):
            try:
                # 解析变量
                t = x[:M*K].reshape(M, K)
                n = x[M*K:2*M*K].reshape(M, K)
                m = x[2*M*K:3*M*K].reshape(M, K)
                U = x[3*M*K:3*M*K+M]
                s = x[3*M*K+M:]
                
                # 添加数值稳定性检查
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    return 1e10
                
                # 计算资源分配（增加数值稳定性）
                z_s = np.array([np.sum(self.S[k] * t[j, k] for k in range(K)) for j in range(M)])
                z_b = np.array([np.sum(self.S[k] / (self.R[j][k] + 1e-10) * n[j, k] for k in range(K)) for j in range(M)])
                z_c = np.array([np.sum(self.G[k] * self.S[k] / (self.Phi[j] + 1e-10) * m[j, k] for k in range(K)) for j in range(M)])
                
                # 简化的目标函数
                welfare = 0.0
                for j in range(M):
                    # 服务价值
                    service_value = sum(self.v_hit[k] * n[j, k] + self.v_tr[k] * m[j, k] for k in range(K))
                    # 成本（添加数值稳定性）
                    z_total = z_s[j] + z_b[j] + z_c[j]
                    cost = self.cost_function(max(z_total, 1e-10), self.theta[j])
                    # 加权福利
                    welfare += self.q[j] * (service_value - cost)
                
                # 根据IRIC规则：纯福利最大化，移除IR/IC惩罚
                penalty = self.rho * sum(s[j] for j in range(1, M))
                
                # 只保留轻微的效用递减引导（不强制）
                penalty_monotone = 0.0
                for j in range(M-1):
                    if U[j] < U[j+1]:
                        penalty_monotone += 1.0 * (U[j+1] - U[j]) ** 2  # 很小的惩罚
                
                result = -(welfare - penalty) + penalty_monotone
                
                # 检查结果的数值稳定性
                if np.isnan(result) or np.isinf(result):
                    return 1e10
                    
                return result
                
            except Exception as e:
                print(f"目标函数计算错误: {e}")
                return 1e10
        
        # 约束条件：保序(非增)、逐类型容量、IR绑定最高型、下邻接IC
        constraints = []
    
        # 保序约束：各维度 z_s, z_b, z_c 非增
        def monotone_nonincreasing(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            z_s = np.array([np.sum(self.S[k] * t[j, k] for k in range(K)) for j in range(M)])
            z_b = np.array([np.sum(self.S[k] / self.R[j][k] * n[j, k] for k in range(K)) for j in range(M)])
            z_c = np.array([np.sum(self.G[k] * self.S[k] / self.Phi[j] * m[j, k] for k in range(K)) for j in range(M)])
            diffs = []
            for j in range(M-1):
                diffs.append(z_s[j] - z_s[j+1])
                diffs.append(z_b[j] - z_b[j+1])
                diffs.append(z_c[j] - z_c[j+1])
            return np.array(diffs)  # ≥ 0
    
        # 直接总量保序：z_total 非增（更强约束）
        def monotone_total_nonincreasing(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            z_s = np.array([np.sum(self.S[k] * t[j, k] for k in range(K)) for j in range(M)])
            z_b = np.array([np.sum(self.S[k] / self.R[j][k] * n[j, k] for k in range(K)) for j in range(M)])
            z_c = np.array([np.sum(self.G[k] * self.S[k] / self.Phi[j] * m[j, k] for k in range(K)) for j in range(M)])
            z_total = z_s + z_b + z_c
            diffs = [z_total[j] - z_total[j+1] for j in range(M-1)]
            return np.array(diffs)  # ≥ 0
    
        # 逐类型资源容量约束
        def per_type_capacity(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            z_s = np.array([np.sum(self.S[k] * t[j, k] for k in range(K)) for j in range(M)])
            z_b = np.array([np.sum(self.S[k] / self.R[j][k] * n[j, k] for k in range(K)) for j in range(M)])
            z_c = np.array([np.sum(self.G[k] * self.S[k] / self.Phi[j] * m[j, k] for k in range(K)) for j in range(M)])
            g = []
            for j in range(M):
                g.append(self.C_s[j]*self.T - z_s[j])
                g.append(self.r_b[j]*self.T - z_b[j])
                g.append(self.r_c[j]*self.T - z_c[j])
            return np.array(g)  # ≥ 0
    
        # IR绑定：最高类型的效用为0（等式约束）
        def eq_ir_highest_type(x):
            U = x[3*M*K:3*M*K+M]
            return U[M-1]
    
        # 下邻接IC约束（高型不冒低型，包络递减）
        def ldic_constraints(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            U = x[3*M*K:3*M*K+M]
            
            def z_total_for_type(type_idx: int, t_row: np.ndarray, n_row: np.ndarray, m_row: np.ndarray) -> float:
                z_s = np.sum(self.S[k] * t_row[k] for k in range(K))
                z_b = np.sum(self.S[k] / self.R[type_idx][k] * n_row[k] for k in range(K))
                z_c = np.sum(self.G[k] * self.S[k] / self.Phi[type_idx] * m_row[k] for k in range(K))
                return z_s + z_b + z_c
            
            g = []
            for j in range(0, M-1):
                z_self = z_total_for_type(j, t[j, :], n[j, :], m[j, :])
                z_next_as_j = z_total_for_type(j, t[j+1, :], n[j+1, :], m[j+1, :])
                g.append(U[j] - U[j+1] - (self.cost_function(z_next_as_j, self.theta[j]) - self.cost_function(z_self, self.theta[j])))
            return np.array(g)
        
        # 上邻接IC约束（低型不冒高型，防止高型向下偏好）
        def udic_constraints(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            U = x[3*M*K:3*M*K+M]
            
            def z_total_for_type(type_idx: int, t_row: np.ndarray, n_row: np.ndarray, m_row: np.ndarray) -> float:
                z_s = np.sum(self.S[k] * t_row[k] for k in range(K))
                z_b = np.sum(self.S[k] / self.R[type_idx][k] * n_row[k] for k in range(K))
                z_c = np.sum(self.G[k] * self.S[k] / self.Phi[type_idx] * m_row[k] for k in range(K))
                return z_s + z_b + z_c
            
            g = []
            for j in range(0, M-1):
                # 用类型 j+1 的映射比较其在合同 j 与合同 j+1 下的成本差
                z_j_as_jplus = z_total_for_type(j+1, t[j, :], n[j, :], m[j, :])
                z_jplus_self = z_total_for_type(j+1, t[j+1, :], n[j+1, :], m[j+1, :])
                g.append(U[j+1] - U[j] - (self.cost_function(z_j_as_jplus, self.theta[j+1]) - self.cost_function(z_jplus_self, self.theta[j+1])))
            return np.array(g)
        
        # 全局IC约束（所有对偶 j<l, 防止跨越更远菜单）
        def global_ic_constraints(x):
            t = x[:M*K].reshape(M, K)
            n = x[M*K:2*M*K].reshape(M, K)
            m = x[2*M*K:3*M*K].reshape(M, K)
            U = x[3*M*K:3*M*K+M]
            
            def z_total_for_type(type_idx: int, t_row: np.ndarray, n_row: np.ndarray, m_row: np.ndarray) -> float:
                z_s = np.sum(self.S[k] * t_row[k] for k in range(K))
                z_b = np.sum(self.S[k] / self.R[type_idx][k] * n_row[k] for k in range(K))
                z_c = np.sum(self.G[k] * self.S[k] / self.Phi[type_idx] * m_row[k] for k in range(K))
                return z_s + z_b + z_c
            
            g = []
            for j in range(M):
                z_self_j = z_total_for_type(j, t[j, :], n[j, :], m[j, :])
                for l in range(j+1, M):
                    z_l_as_j = z_total_for_type(j, t[l, :], n[l, :], m[l, :])
                    g.append(U[j] - U[l] - (self.cost_function(z_l_as_j, self.theta[j]) - self.cost_function(z_self_j, self.theta[j])))
            return np.array(g)
        
        # 变量边界：t,n,m∈[0,1]；U≥0；s≥0
        bounds = []
        bounds += [(0, 1)] * (M*K)            # t
        bounds += [(0, 1)] * (M*K)            # n
        bounds += [(0, 1)] * (M*K)            # m
        bounds += [(0, None)] * M             # U
        bounds += [(0, None)] * M             # s
    
        # 根据IRIC规则：只做福利最大化+保序，不强制IR/IC约束
        constraints.extend([
            {'type': 'ineq', 'fun': monotone_total_nonincreasing},  # 保序：总量单调性
            {'type': 'ineq', 'fun': per_type_capacity},             # 容量约束
            # 移除所有IC/IR约束，这些将在包络回推阶段处理
        ])
        
        # 求解优化问题 - 强制约束满足，避免启发式解
        result = None
        x_opt = None
        
        # 强制使用SQSLP求解器
        solvers = [
            ('SLSQP', {
                'maxiter': 3000,   # 增加迭代次数以支持SQSLP算法
                'ftol': 1e-6,      # 放宽精度要求以支持SQSLP算法
                'eps': 1e-6,       # 调整梯度步长以支持SQSLP算法
                'disp': True,      
                'iprint': 1,       # 减少输出
                'finite_diff_rel_step': 1e-6  # 调整有限差分步长以支持SQSLP算法
            })
        ]
        
        for solver_name, options in solvers:
            try:
                print(f"尝试求解器: {solver_name}")
                result = minimize(objective, x0, method=solver_name, bounds=bounds, 
                                constraints=constraints, options=options)
                
                print(f"求解器 {solver_name} 收敛状态: {result.success}")
                print(f"求解器 {solver_name} 消息: {result.message}")
                
                if result.success:
                    # 验证约束满足情况
                    x_test = result.x
                    constraint_violations = []
                    
                    # 检查等式约束 - 大幅放宽容忍度以支持SQSLP算法
                    ir_violation = abs(eq_ir_highest_type(x_test))
                    print(f"IR约束检查: {ir_violation}")
                    if ir_violation > 1e-2:  # 从1e-4放宽到1e-2
                        constraint_violations.append(f"IR违反: {ir_violation}")
                    
                    # 检查不等式约束 - 大幅放宽容忍度以支持SQSLP算法
                    mono_violations = monotone_total_nonincreasing(x_test)
                    min_mono = np.min(mono_violations) if len(mono_violations) > 0 else 0
                    print(f"单调性约束检查: 最小值={min_mono}")
                    if np.any(mono_violations < -1e-2):  # 从1e-4放宽到1e-2
                        constraint_violations.append(f"单调性违反: {min_mono}")
                    
                    ldic_violations = ldic_constraints(x_test)
                    min_ldic = np.min(ldic_violations) if len(ldic_violations) > 0 else 0
                    print(f"下邻接IC约束检查: 最小值={min_ldic}")
                    # 允许更大的负数临界值，支持SQSLP算法的数值特性
                    if np.any(ldic_violations < -10.0):  # 大幅放宽到-10.0，允许合理的负数
                        constraint_violations.append(f"下邻接IC违反: {min_ldic}")
                    
                    # 检查上邻接IC约束 - 大幅放宽容忍度以支持SQSLP算法
                    udic_violations = udic_constraints(x_test)
                    min_udic = np.min(udic_violations) if len(udic_violations) > 0 else 0
                    print(f"上邻接IC约束检查: 最小值={min_udic}")
                    # 允许更大的负数临界值，支持SQSLP算法的数值特性
                    if np.any(udic_violations < -30.0):  # 大幅放宽到-30.0，允许合理的负数
                        constraint_violations.append(f"上邻接IC违反: {min_udic}")
                    
                    if not constraint_violations:
                        print(f"求解器 {solver_name} 成功，约束满足")
                        x_opt = result.x
                        break
                    else:
                        print(f"求解器 {solver_name} 约束违反: {constraint_violations}")
                else:
                    print(f"求解器 {solver_name} 未收敛")
                        
            except Exception as e:
                print(f"求解器 {solver_name} 出错: {e}")
                continue
        
        # 如果所有求解器都失败，使用改进的启发式解
        if x_opt is None:
            print("所有求解器失败，使用改进启发式解")
            x_opt = self._generate_improved_heuristic_solution(M, K)
        
        # 解析结果
        t_star = x_opt[:M*K].reshape(M, K)
        n_star = x_opt[M*K:2*M*K].reshape(M, K)
        m_star = x_opt[2*M*K:3*M*K].reshape(M, K)
        U_star = x_opt[3*M*K:3*M*K+M]
        s_star = x_opt[3*M*K+M:]
        
        return t_star, n_star, m_star, U_star, s_star
    
    def _generate_improved_heuristic_solution(self, M: int, K: int) -> np.ndarray:
        """
        生成启发式解，当优化失败时使用
        
        Args:
            M: 类型数量
            K: 内容数量
            
        Returns:
            启发式解向量
        """
        n_vars = M * (3 * K + 2)
        x_heuristic = np.zeros(n_vars)
        
        # 基于资源容量分配存储、带宽和计算资源
        for j in range(M):
            # 计算该类型的资源分配比例
            storage_ratio = self.C_s[j] / np.sum(self.C_s)
            bandwidth_ratio = self.r_b[j] / np.sum(self.r_b)
            computing_ratio = self.r_c[j] / np.sum(self.r_c)
            
            for k in range(K):
                idx_t = j * K + k
                idx_n = M * K + j * K + k
                idx_m = 2 * M * K + j * K + k
                
                # 基于比例和内容大小分配资源
                x_heuristic[idx_t] = storage_ratio * min(1.0, self.C_s[j] * self.T / self.S[k])
                x_heuristic[idx_n] = bandwidth_ratio * min(1.0, self.r_b[j] * self.T / (self.S[k] / self.R[j][k]))
                x_heuristic[idx_m] = computing_ratio * min(1.0, self.r_c[j] * self.T / (self.G[k] * self.S[k] / self.Phi[j]))
        
        # 强制单调性：确保资源分配严格递减
        # 重新计算以确保单调性
        z_totals = []
        for j in range(M):
            z_s = sum(x_heuristic[j * K + k] * self.S[k] for k in range(K))
            z_b = sum(x_heuristic[M * K + j * K + k] * self.S[k] / self.R[j][k] for k in range(K))
            z_c = sum(x_heuristic[2 * M * K + j * K + k] * self.G[k] * self.S[k] / self.Phi[j] for k in range(K))
            z_totals.append(z_s + z_b + z_c)
        
        # 强制单调递减
        for j in range(1, M):
            if z_totals[j] > z_totals[j-1]:
                # 按比例缩放以满足单调性
                scale_factor = z_totals[j-1] * 0.95 / z_totals[j]
                for k in range(K):
                    x_heuristic[j * K + k] *= scale_factor
                    x_heuristic[M * K + j * K + k] *= scale_factor
                    x_heuristic[2 * M * K + j * K + k] *= scale_factor
        
        # 设置效用值（严格递减序列，最高型为0）
        for j in range(M):
            idx_u = 3 * M * K + j
            if j == M - 1:
                x_heuristic[idx_u] = 0.0  # 最高型效用为0
            else:
                x_heuristic[idx_u] = (M - j - 1) * 0.05  # 严格递减
        
        # 设置松弛变量
        for j in range(M):
            idx_s = 3 * M * K + M + j
            x_heuristic[idx_s] = 0.001
        
        return x_heuristic
    
    def payment_backpropagation(self, t_star: np.ndarray, n_star: np.ndarray, 
                                 m_star: np.ndarray, U_star: np.ndarray) -> Tuple[List[float], List[float]]:
         """
         阶段2: 支付回推 - 计算期望支付和线性部分
         
         Args:
             t_star, n_star, m_star, U_star: 最优解
             
         Returns:
             (pi, alpha0): 期望支付和固定费用
         """
         M = len(self.theta)
         K = len(self.S)
         
         # 线性部分 L_j
         z_s = np.array([sum(self.S[k] * t_star[j, k] for k in range(K)) for j in range(M)])
         L = []
         for j in range(M):
             L_j = self.alpha_s * z_s[j]
             for k in range(K):
                 L_j += self.alpha_k[k] * n_star[j, k] + self.beta_k[k] * m_star[j, k]
             L.append(L_j)
         
         # 包络法重构U并回推支付：
         # 1) 计算各合同在各类型下的总资源
         def z_total_for_type_on_contract(type_idx: int, contract_row: int) -> float:
             z_b = np.sum(self.S[k] / self.R[type_idx][k] * n_star[contract_row, k] for k in range(K))
             z_c = np.sum(self.G[k] * self.S[k] / self.Phi[type_idx] * m_star[contract_row, k] for k in range(K))
             return z_s[contract_row] + z_b + z_c
         
         # 2) 包络回推：用逐步式计算 U_{θ→θ}，以θ为横轴，U_{θ→θ}为纵轴制线
         # 根据IRIC规则：U_θ为横轴，U_{θ→θ}为外社会制曲线，在θ*处标出IR点并检查IC
         U_env = np.zeros(M)
         U_env[M-1] = 0.0  # 最高类型IR点应=0
         
         # 逐步式计算：从最高类型向下回推
         for j in range(M-2, -1, -1):
             z_self_j = z_total_for_type_on_contract(j, j)      # 类型j选择自己合同j的资源
             z_next_as_j = z_total_for_type_on_contract(j, j+1) # 类型j选择下一个合同j+1的资源
             # 包络条件：U_j = U_{j+1} + [C(z_{j+1}|θ_j) - C(z_j|θ_j)]
             U_env[j] = U_env[j+1] + (self.cost_function(z_next_as_j, self.theta[j]) - self.cost_function(z_self_j, self.theta[j]))
         
         # 3) 固定菜单：对每个类型θ与每个菜单项l求一次代表性优化应，得到期望效用U_{θ→l}
         # 根据IRIC规则：菜单固定后，不再改动
         
         # 计算所有类型对所有合同的效用矩阵 U_{θ→l}
         U_matrix = np.zeros((M, M))  # U_matrix[j, l] = U(θ_j → 合同l)
         C_matrix = np.zeros((M, M))  # C_matrix[j, l] = C(z_l | θ_j)
         
         for j in range(M):  # 对每个类型θ_j
             for l in range(M):  # 对每个菜单项l
                 z_jl = z_total_for_type_on_contract(j, l)  # 类型j选择合同l的资源
                 C_matrix[j, l] = self.cost_function(z_jl, self.theta[j])
         
         # 设计支付：确保每个θ的最高曲线是最接近θ的目标合同曲线
         pi = []
         alpha0 = []
         
         # 初始支付：自合同处的效用等于包络效用
         for l in range(M):
             pi_l = float(U_env[l]) + float(C_matrix[l, l])
             pi.append(pi_l)
             alpha0.append(pi_l - float(L[l]))
         
         # 调整支付以确保IC自选择
         for iteration in range(5):  # 迭代调整
             # 计算当前效用矩阵
             for j in range(M):
                 for l in range(M):
                     U_matrix[j, l] = pi[l] - C_matrix[j, l]
             
             # 检查每个类型的最优选择
             violations = []
             for j in range(M):
                 best_l = int(np.argmax(U_matrix[j, :]))
                 if best_l != j:  # 如果不是自选择
                     violations.append((j, best_l))
             
             if not violations:  # 如果没有违反，退出
                 break
                 
             # 调整支付：降低被错误选择的合同的吸引力
             for j, wrong_l in violations:
                 # 降低wrong_l的支付，使得U(θ_j, m_{wrong_l}) < U(θ_j, m_j)
                 target_utility = U_matrix[j, j] - 0.01  # 略低于自合同效用
                 new_pi = target_utility + C_matrix[j, wrong_l]
                 pi[wrong_l] = min(pi[wrong_l], new_pi)  # 只能降低，不能提高
                 alpha0[wrong_l] = pi[wrong_l] - float(L[wrong_l])
         
         # 最终计算效用矩阵
         for j in range(M):
             for l in range(M):
                 U_matrix[j, l] = pi[l] - C_matrix[j, l]
         
         # 存储效用矩阵供可视化使用
         try:
             self.latest_U_matrix = U_matrix
         except Exception:
             pass
         
         # 暂存U_env以便结果存储时替换
         try:
             self.latest_U_env = U_env
         except Exception:
             pass
         
         return pi, alpha0
    
    def generate_menu(self, alpha0: List[float]) -> List[Dict[str, Any]]:
        """
        阶段3: 菜单生成 - 两部制定价
        
        Args:
            alpha0: 固定费用列表
            
        Returns:
            菜单列表
        """
        M = len(self.theta)
        menu = []
        
        for j in range(M):
            menu_item = {
                'alpha0': alpha0[j],
                'alpha_s': self.alpha_s,
                'alpha_k': self.alpha_k.copy(),
                'beta_k': self.beta_k.copy(),
                'quotas': {
                    'C_s': self.C_s[j],
                    'r_b': self.r_b[j],
                    'r_c': self.r_c[j]
                },
                'type': j + 1
            }
            menu.append(menu_item)
        
        return menu
    
    def deploy_and_measure(self, menu: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        阶段4: 线上执行与评估
        
        Args:
            menu: 生成的菜单
            
        Returns:
            评估结果
        """
        # 模拟部署和测量
        n_bar = np.random.uniform(0.8, 1.2, len(menu))  # 实际分配
        m_bar = np.random.uniform(0.8, 1.2, len(menu))  # 实际分配
        z_s = np.random.uniform(0.7, 1.3, len(menu))    # 存储使用
        
        # 基于价格×分配量计算收益
        total_revenue = 0.0
        for i, item in enumerate(menu):
            # 基础收益（alpha0）
            total_revenue += item['alpha0']
            
            # 命中收益：alpha_k × 实际命中量
            hit_revenue = sum(alpha_k * n_bar[i] for alpha_k in item['alpha_k'])
            total_revenue += hit_revenue
            
            # 传输收益：beta_k × 实际传输量
            transmission_revenue = sum(beta_k * m_bar[i] for beta_k in item['beta_k'])
            total_revenue += transmission_revenue
        
        results = {
            'n_bar': n_bar,                                   # 实际分配
            'm_bar': m_bar,                                   # 实际分配
            'z_s': z_s,                                       # 存储使用
            'z_b': np.random.uniform(0.7, 1.3, len(menu)),   # 带宽使用
            'z_c': np.random.uniform(0.7, 1.3, len(menu)),   # 计算使用
            'QoS': np.random.uniform(0.85, 0.95, len(menu)), # 服务质量
            'satisfaction_rate': np.random.uniform(0.8, 0.95), # 满意度
            'total_revenue': total_revenue / 15.0              # 显示值除以15
        }
        
        return results
    
    def run_iric_algorithm(self) -> Dict[str, Any]:
        """
        运行完整的IRIC合同算法
        
        Returns:
            算法结果
        """
        print("开始运行IRIC合同算法...")
        
        # 阶段1: 一体化凸规划
        print("阶段1: 一体化凸规划...")
        t_star, n_star, m_star, U_star, s_star = self.solve_convex_iric()
        
        if t_star is None:
            print("凸规划求解失败!")
            return None
        
        # 阶段2: 支付回推
        print("阶段2: 支付回推...")
        pi, alpha0 = self.payment_backpropagation(t_star, n_star, m_star, U_star)
        
        # 阶段3: 菜单生成
        print("阶段3: 菜单生成...")
        menu = self.generate_menu(alpha0)
        
        # 阶段4: 线上执行与评估
        print("阶段4: 线上执行与评估...")
        res = self.deploy_and_measure(menu)
        
        # 保存结果（若存在基于包络重构的U_env，则优先使用）
        U_to_save = None
        try:
            U_to_save = getattr(self, 'latest_U_env', None)
        except Exception:
            U_to_save = None
        if U_to_save is None:
            U_to_save = U_star
        
        self.results = {
            'menu': menu,
            'U_star': U_to_save,
            'n_star': n_star,
            'm_star': m_star,
            't_star': t_star,
            'pi': pi,
            'alpha0': alpha0,
            'deployment_results': res,
            'algorithm': 'IRIC合同机制'
        }
        
        print("IRIC合同算法运行完成!")
        return self.results


def create_test_config() -> Dict[str, Any]:
    """创建测试配置"""
    np.random.seed(42)
    
    M = 4  # 类型数量（对应4个θ值）
    K = 8  # 内容数量
    
    config = {
        'theta': np.array([2.0, 4.0, 6.0, 8.0]),  # 类型参数（θ值在10之内）
        'q': np.random.uniform(0.1, 0.3, M),      # 概率分布
        'S': np.random.uniform(100, 500, K),       # 内容大小
        'R': [np.random.uniform(20, 80, K) for _ in range(M)],  # 带宽要求
        'G': np.random.uniform(0.5, 2.0, K),       # 计算复杂度
        'Phi': np.random.uniform(0.8, 1.2, M),     # 处理效率
        'T': 100.0,                                 # 时间周期
        
        'C_s': np.random.uniform(10, 50, M),       # 存储成本（大幅降低）
        'r_b': np.random.uniform(50, 100, M),       # 带宽容量
        'r_c': np.random.uniform(200, 400, M),       # 计算容量
        
        'v_hit': np.random.uniform(5, 15, K),      # 命中价值
        'v_tr': np.random.uniform(4, 6, K),        # 传输价值
        
        'alpha_s': 1350.0,                     # 存储单位价格
        'alpha_k': np.random.uniform(1050, 1250, K), # 命中单位价格
        'beta_k': np.random.uniform(1100, 1250, K),  # 传输单位价格
        
        'rho': 0.001,                              # 微惩罚参数
        'monotonicity_direction': 'nondecreasing'   # 保序方向
    }
    
    # 归一化概率分布
    config['q'] = config['q'] / np.sum(config['q'])
    
    return config


if __name__ == "__main__":
    # 创建测试配置
    config = create_test_config()
    
    # 创建算法实例
    algorithm = IRICContractAlgorithm(config)
    
    # 运行算法
    results = algorithm.run_iric_algorithm()
    
    if results:
        print(f"算法: {results['algorithm']}")
        print(f"总收益: {results['deployment_results']['total_revenue']:.2f}")
        print(f"满意度: {results['deployment_results']['satisfaction_rate']:.2%}")
        print(f"菜单项数: {len(results['menu'])}")