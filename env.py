import numpy as np
import time
import random        
from uav import ChannelModel

class UAVEnv:
    def __init__(self, env_params, reward_params, channel_params):

        self.env_params = env_params
        self.reward_params = reward_params
        self.channel_params = channel_params

        self.map_size = self.env_params['map_size']
        self.max_steps = self.env_params['max_steps']
        self.normalization_factor = self.env_params['map_size']  # 归一化因子等于地图尺寸
        self.time_slot = self.env_params['time_slot']

        self.start = np.array(self.env_params['start_pos'])
        self.goal = np.array(self.env_params['goal_pos'])
        
        # 初始化合法节点
        self.legitimate_nodes = [np.array(pos) for pos in self.env_params['legitimate_nodes']]
        
        # 随机生成监管者
        self._generate_wardens()
        
        # 环境状态变量
        self.current_step = 0
        self.state = self.start.copy()
        
        # 动力学参数
        self.velocity_magnitude = self.env_params['velocity_magnitude']

        # 传递信道参数和环境参数
        self.channel_model = ChannelModel(channel_params=self.channel_params, env_params=self.env_params, seed=int(time.time()))
        
        # 性能指标记录
        self.dep_history = []
        self.transmission_rate_history = []
    
    def _generate_wardens(self):
        # 随机确定监管者数量
        min_count, max_count = self.env_params['warden_count_range']
        self.warden_count = random.randint(min_count, max_count)
        
        # 随机生成监管者位置
        min_pos, max_pos = self.env_params['warden_pos_range']
        self.wardens = []
        for _ in range(self.warden_count):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            self.wardens.append(np.array([x, y]))
        
        self.wardens = np.array(self.wardens)
        
    def _calculate_transmission_rates(self):

        rates_to_nodes = []
        
        for node_pos in self.legitimate_nodes:
            # 计算传输速率 R_l = log2(1 + SNR)
            transmission_rate = self.channel_model.calculate_transmission_rate(
                self.state[0], self.state[1], node_pos[0], node_pos[1]
            )
            rates_to_nodes.append(transmission_rate)
        
        total_transmission_rate = sum(rates_to_nodes)
        return total_transmission_rate, rates_to_nodes

    # 环境接口
    
    def get_state(self):
        map_center = self.normalization_factor / 2.0
        pos_norm = (self.state - map_center) / (self.normalization_factor / 2.0)
        pos_norm = np.clip(pos_norm, -1.0, 1.0)
        
        return pos_norm.astype(np.float32)

    def _get_obs(self):
        return self.get_state()

    def reset(self):
        self._generate_wardens()
        
        self.state = self.start.copy()
        self.current_step = 0
        self.dep_history = []
        self.transmission_rate_history = []
        
        # 计算初始DEP值
        if len(self.wardens) > 0:
            initial_dep = self.channel_model.calculate_dep(
                self.state[0], self.state[1], 
                self.wardens[0][0], self.wardens[0][1]
            )
            self.dep_history.append(initial_dep)
        else:
            self.dep_history.append(0.0)

        self.prev_dist_to_dest = np.linalg.norm(self.state - self.goal)
        self.position_history = [] # 重置位置历史，用于振荡检测
        
        return self._get_obs()

    def step(self, action):
        reward = 0.0
        # 将动作转换为速度向量，动作控制方向，速度大小恒定为V
        action_norm = np.linalg.norm(action)
        if action_norm > 1e-8:  # 避免除零
            velocity_direction = action / action_norm
        else:
            # 如果动作为零向量，保持当前方向或随机方向
            velocity_direction = np.array([1.0, 0.0])  # 默认向右
        velocity = velocity_direction * self.velocity_magnitude
        
        # 计算下一个位置
        next_state = self.state + self.time_slot * velocity

        # 边界碰撞
        boundary_violation = False
        if np.any(next_state < 0.0) or np.any(next_state > self.env_params['map_size']):
            # 限制在边界内
            self.state = np.clip(next_state, 0.0, self.env_params['map_size'])
            boundary_violation = True
            # 给予边界惩罚
            reward += self.reward_params['boundary_penalty']
        else:
            self.state = next_state
        
        # 边界惩罚场
        boundary_distance = min(
            self.state[0],  # 左边界
            self.state[1],  # 下边界
            self.env_params['map_size'] - self.state[0],  # 右边界
            self.env_params['map_size'] - self.state[1]   # 上边界
        )
        
        boundary_field_penalty = 0.0
        boundary_penalty_threshold = self.reward_params.get('boundary_penalty_threshold', 40.0)
        if boundary_distance < boundary_penalty_threshold:
            # 其实奖励场就是一个开口向下的二次函数
            boundary_penalty_strength = self.reward_params.get('boundary_penalty_strength', 0.01)
            boundary_field_penalty = -boundary_penalty_strength * (boundary_penalty_threshold - boundary_distance) ** 2
            reward += boundary_field_penalty

        # 更新环境状态
        self.current_step += 1

        # 当前距离
        dist_to_dest = np.linalg.norm(self.state - self.goal)
        
        # 到最近监管者的距离
        dist_to_warden = float('inf')
        closest_warden_idx = 0
        for i, warden in enumerate(self.wardens):
            dist = np.linalg.norm(self.state - warden)
            if dist < dist_to_warden:
                dist_to_warden = dist
                closest_warden_idx = i

        # 到最近合法节点的距离
        distances_to_legit = [np.linalg.norm(self.state - node) for node in self.legitimate_nodes]
        min_dist_to_legit = min(distances_to_legit)
        closest_legit_idx = distances_to_legit.index(min_dist_to_legit)
        closest_legit_node = self.legitimate_nodes[closest_legit_idx]
        
        # DEP
        DEP = self.channel_model.calculate_dep(
            uav_x=self.state[0],
            uav_y=self.state[1],
            warden_x=self.wardens[0][0],
            warden_y=self.wardens[0][1]
        )
        
        # 传输速率
        total_transmission_rate, rates_to_nodes = self._calculate_transmission_rates()
        
        # 保持向后兼容性，记录到最近节点的传输速率
        transmission_rate = self.channel_model.calculate_transmission_rate(
            uav_x=self.state[0],
            uav_y=self.state[1],
            ground_x=closest_legit_node[0],
            ground_y=closest_legit_node[1]
        )
        
        # 记录性能指标
        self.dep_history.append(DEP)
        self.transmission_rate_history.append(transmission_rate)
        
        # 传输速率奖励
        transmission_reward = total_transmission_rate * self.reward_params['transmission_weight']
        
        # 基础惩罚
        base_penalty = self.reward_params['base_penalty']

        covertness_penalty = 0.0
        if DEP < self.reward_params['covertness_threshold']:
            # 当DEP低于阈值时给予惩罚，因为隐蔽性不足
            covertness_penalty = self.reward_params['covert_penalty_weight'] * (self.reward_params['covertness_threshold'] - DEP)
        
        # 目标导向奖励
        goal_reward = 0.0
        goal_reached = False
        
        # 任务完成奖励
        if dist_to_dest <= self.reward_params['goal_threshold']:
            goal_reward = self.reward_params['goal_reward']
            goal_reached = True
        
        # 距离改善奖励，距离减少给奖励，增加给惩罚
        distance_improvement_reward = 0.0
        if hasattr(self, 'prev_dist_to_dest'):
            distance_change = self.prev_dist_to_dest - dist_to_dest
            distance_improvement_reward = distance_change * self.reward_params.get('distance_guidance_weight', 0.1)
        
        # 更新
        self.prev_dist_to_dest = dist_to_dest
        
        # 振荡惩罚
        oscillation_penalty = 0.0
        if not hasattr(self, 'position_history'):
            self.position_history = []
        
        # 记录位置
        self.position_history.append(self.state.copy())
        
        # 保持位置历史
        history_length = self.reward_params.get('oscillation_history_length', 20)
        if len(self.position_history) > history_length:
            self.position_history.pop(0)
        
        # 检测振荡，如果在最近N步内有相似位置出现可以视为震荡发生
        check_steps = self.reward_params.get('oscillation_check_steps', 10)
        distance_threshold = self.reward_params.get('oscillation_distance_threshold', 30.0)
        penalty_value = self.reward_params.get('oscillation_penalty', -50.0)
        
        if len(self.position_history) >= check_steps:
            current_pos = self.state
            recent_positions = self.position_history[-check_steps:]
            
            for past_pos in recent_positions[:-1]:  # 排除当前位置
                distance_to_past = np.linalg.norm(current_pos - past_pos)
                if distance_to_past < distance_threshold:
                    oscillation_penalty = penalty_value  # 给予振荡惩罚
                    break
        
        # 目标方向奖励，基于当前位置到目标的距离给予连续奖励，距离越近奖励越高
        max_distance = np.sqrt(self.map_size**2 + self.map_size**2)  # 对角线距离
        normalized_distance = dist_to_dest / max_distance
        proximity_reward = (1.0 - normalized_distance) * self.reward_params.get('proximity_reward_weight', 0.05)
        
        # 步数效率奖励，鼓励最短路径，随着步数增加给予递减的时间惩罚
        step_efficiency_penalty = -self.current_step * self.reward_params.get('step_penalty_weight', 0.01)
        
        # 合并所有奖励
        total_goal_reward = goal_reward + distance_improvement_reward + proximity_reward + step_efficiency_penalty
        
        # 基础约束惩罚
        constraint_penalty = 0.0

        # 边界惩罚
        constraint_penalty += boundary_field_penalty
        
        # 组合最终奖励，论文公式 + 目标导向改进 + 振荡惩罚
        reward = transmission_reward + base_penalty + covertness_penalty + total_goal_reward + constraint_penalty + oscillation_penalty

        # 检查终止条件
        done = False
        termination_reason = "ongoing"
        # 任务完成终止，即到达目标点
        if goal_reached:
            done = True
            termination_reason = "goal_reached"
        
        # 超过最大步数终止
        elif self.current_step >= self.max_steps:
            done = True
            termination_reason = "max_steps"

        # 构建字典
        info = {
            # 核心指标
            'DEP': DEP, 
            'throughput': total_transmission_rate, 
            'rates_to_nodes': rates_to_nodes,

            # 环境状态
            'dist_to_dest': dist_to_dest, 
            'dist_to_warden': dist_to_warden, 
            'velocity_magnitude': np.linalg.norm(velocity), 
            'step': self.current_step, 
            
            # 约束和终止条件
            'covertness_violation': DEP < self.reward_params['covertness_threshold'], # 隐蔽性违反
            'goal_reached': goal_reached,  
            'termination_reason': termination_reason,   
            'boundary_distance': boundary_distance, 
            'boundary_violation': boundary_violation, 
            
            # 奖励
            'transmission_reward': transmission_reward, 
            'base_penalty': base_penalty, 
            'covertness_penalty': covertness_penalty, 
            'goal_reward': goal_reward,  
            'distance_improvement_reward': distance_improvement_reward, 
            'proximity_reward': proximity_reward, 
            'step_efficiency_penalty': step_efficiency_penalty, 
            'total_goal_reward': total_goal_reward,  
            'constraint_penalty': constraint_penalty,     
            'oscillation_penalty': oscillation_penalty,   
        }
        
        return self._get_obs(), reward, done, info