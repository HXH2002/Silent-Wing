import random
from collections import deque
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Actor策略网络
class Actor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):

        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)    # 第一层全连接
        self.fc2 = nn.Linear(256, 256)          # 第二层全连接
        self.fc3 = nn.Linear(256, action_dim)   # 输出层

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) 


# 优先经验回放缓冲区
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, n_step, gamma, alpha, beta, beta_increment):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # 防止优先级为0

        # 存储经验的数组
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
        # 多步经验存储
        self.n_step_state = np.zeros((max_size, state_dim))
        self.n_step_action = np.zeros((max_size, action_dim))
        self.n_step_next_state = np.zeros((max_size, state_dim))
        self.n_step_reward = np.zeros((max_size, 1))
        self.n_step_done = np.zeros((max_size, 1))
        
        # 临时存储n步轨迹
        self.n_step_buffer = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, next_state, reward, done):
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, next_state, reward, done))
        # 存储当前单步经验
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        # 计算多步经验
        if len(self.n_step_buffer) >= self.n_step:
            # 计算n步回报
            n_step_reward = 0
            n_step_done = False
            
            for i in range(self.n_step):
                step_reward = self.n_step_buffer[i][3]
                step_done = self.n_step_buffer[i][4]
                n_step_reward += (self.gamma ** i) * step_reward
                if step_done:  # 终止状态，提前结束
                    n_step_done = True
                    break
            
            # 获取n步经验的始末状态
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            if n_step_done:
                # 如果在n步内遇到终止状态，使用终止状态
                _, _, last_next_state, _, _ = self.n_step_buffer[i]
            else:
                # 否则使用第n步的next_state
                _, _, last_next_state, _, _ = self.n_step_buffer[self.n_step-1]
            
            # 存储n步经验
            self.n_step_state[self.ptr] = first_state
            self.n_step_action[self.ptr] = first_action
            self.n_step_next_state[self.ptr] = last_next_state
            self.n_step_reward[self.ptr] = n_step_reward
            self.n_step_done[self.ptr] = n_step_done
        else:
            # 如果步数不够，使用单步经验作为多步经验的占位符
            self.n_step_state[self.ptr] = state
            self.n_step_action[self.ptr] = action
            self.n_step_next_state[self.ptr] = next_state
            self.n_step_reward[self.ptr] = reward
            self.n_step_done[self.ptr] = done
        
        # 计算初始优先级使用最大优先级
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        
        # 添加到优先级树
        experience = (self.ptr, 
                     self.state[self.ptr], self.action[self.ptr], self.next_state[self.ptr], 
                     self.reward[self.ptr], self.done[self.ptr],
                     self.n_step_state[self.ptr], self.n_step_action[self.ptr], 
                     self.n_step_next_state[self.ptr], self.n_step_reward[self.ptr], 
                     self.n_step_done[self.ptr])
        self.tree.add(max_priority, experience)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        # 如果缓冲区满了，移除最旧的经验
        if len(self.n_step_buffer) > self.n_step:
            self.n_step_buffer.pop(0)
        
        # 如果episode结束，处理剩余的n步经验
        if done:
            self._flush_n_step_buffer()
    
    def _flush_n_step_buffer(self):
        # 为缓冲区剩余的每个经验计算多步回报
        while len(self.n_step_buffer) > 1:
            # 移除第一个
            self.n_step_buffer.pop(0)
        
            if len(self.n_step_buffer) == 0:
                break 

            # 为剩余经验重新计算
            n_step_reward = 0
            n_step_done = False
            
            for i, (_, _, _, r, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:  # 终止状态，提前结束
                    n_step_done = True
                    break
            
            # 获取经验
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            if n_step_done:
                _, _, last_next_state, _, _ = self.n_step_buffer[i]
            else:
                _, _, last_next_state, _, _ = self.n_step_buffer[-1]
            
            # 找到对应的存储位置
            buffer_len = len(self.n_step_buffer)
            target_ptr = (self.ptr - buffer_len) % self.max_size
            
            # 更新经验
            self.n_step_state[target_ptr] = first_state
            self.n_step_action[target_ptr] = first_action
            self.n_step_next_state[target_ptr] = last_next_state
            self.n_step_reward[target_ptr] = n_step_reward
            self.n_step_done[target_ptr] = n_step_done

            experience = (target_ptr,
                         self.state[target_ptr], self.action[target_ptr], self.next_state[target_ptr],
                         self.reward[target_ptr], self.done[target_ptr],
                         self.n_step_state[target_ptr], self.n_step_action[target_ptr],
                         self.n_step_next_state[target_ptr], self.n_step_reward[target_ptr],
                         self.n_step_done[target_ptr])

            tree_idx = target_ptr + self.tree.capacity - 1
            if tree_idx < len(self.tree.data):
                self.tree.data[tree_idx] = experience
        
        # 清空缓存
        self.n_step_buffer.clear()

    
    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # 解包经验
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        n_states = []
        n_actions = []
        n_next_states = []
        n_rewards = []
        n_dones = []
        
        for experience in batch:
            ptr, state, action, next_state, reward, done, n_state, n_action, n_next_state, n_reward, n_done = experience
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            n_states.append(n_state)
            n_actions.append(n_action)
            n_next_states.append(n_next_state)
            n_rewards.append(n_reward)
            n_dones.append(n_done)
        
        # 单步经验
        single_step = (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(rewards).reshape(-1, 1).to(self.device),
            torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        )
        
        # 多步经验
        multi_step = (
            torch.FloatTensor(n_states).to(self.device),
            torch.FloatTensor(n_actions).to(self.device),
            torch.FloatTensor(n_next_states).to(self.device),
            torch.FloatTensor(n_rewards).reshape(-1, 1).to(self.device),
            torch.FloatTensor(n_dones).reshape(-1, 1).to(self.device)
        )
        
        return single_step, multi_step, idxs, torch.FloatTensor(is_weights).to(self.device)

    def update_priorities(self, idxs, errors):
        for i, idx in enumerate(idxs):
            priority = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)



# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_dim + action_dim, 256)  # 第一层：状态和动作拼接
        self.fc2_q1 = nn.Linear(256, 256)                     # 第二层
        self.fc3_q1 = nn.Linear(256, 1)                       # 输出层：Q值
        # Q2
        self.fc1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q2 = nn.Linear(256, 256)
        self.fc3_q2 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # 拼接状态和动作
        # Q1
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        # Q2
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        return q1


# TD3算法
class TD3:

    def __init__(self, state_dim: int, action_dim: int, td3_params: dict):

         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.action_dim = action_dim

         # 初始化网络
         self.actor = Actor(state_dim, action_dim).to(self.device)
         self.critic1 = Critic(state_dim, action_dim).to(self.device)
         self.critic2 = Critic(state_dim, action_dim).to(self.device)
         self.actor_target = Actor(state_dim, action_dim).to(self.device)
         self.critic1_target = Critic(state_dim, action_dim).to(self.device)
         self.critic2_target = Critic(state_dim, action_dim).to(self.device)
         
         # 目标网络参数同步
         self.actor_target.load_state_dict(self.actor.state_dict())
         self.critic1_target.load_state_dict(self.critic1.state_dict())
         self.critic2_target.load_state_dict(self.critic2.state_dict())

         # 初始化优化器
         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=td3_params['lr_actor'])
         self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=td3_params['lr_critic'])
         self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=td3_params['lr_critic'])

         # 初始化优先经验回放缓冲区，支持多步学习
         self.replay_buffer = PrioritizedReplayBuffer(
             state_dim=state_dim,
             action_dim=action_dim,
             max_size=td3_params['replay_buffer_size'],
             n_step=td3_params['n_step'],
             gamma=td3_params['multi_step_gamma'],
             alpha=td3_params['per_alpha'],
             beta=td3_params['per_beta'],
             beta_increment=td3_params['per_beta_increment']
         )

         # 算法超参
         self.batch_size = td3_params['batch_size']      # 批处理大小
         self.gamma = td3_params['gamma']                # 折扣因子
         self.tau = td3_params['tau']                    # 软更新系数
         self.policy_noise = td3_params['policy_noise']  # 目标策略噪声
         self.noise_clip = td3_params['noise_clip']      # 噪声裁剪范围
         self.policy_freq = td3_params['policy_freq']    # 策略更新频率
         self.max_action = td3_params['max_action']      # 最大动作值

    def select_action(self, state, noise: float = 0.1):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise > 0:
            # 高斯噪声
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1.0, 1.0)
        return action

    def store_transition(self, s, a, r, s_, d):
        self.replay_buffer.add(s, a, s_, r, d)

    def update(self, total_steps: int):
        if self.replay_buffer.size < self.batch_size:
            return
        
        # 采样单步和多步经验（基于优先级）
        single_step, multi_step, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = single_step
        n_states, n_actions, n_next_states, n_rewards, n_dones = multi_step

        # 单步TD
        noise = torch.clamp(
            torch.randn_like(actions) * self.policy_noise,
            -self.noise_clip, self.noise_clip,
        ).to(self.device)
        next_actions = self.actor_target(next_states) + noise
        next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

        target_q1, _ = self.critic1_target(next_states, next_actions)
        _, target_q2 = self.critic2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)  # 取较小的Q值（双Q学习）
        single_step_target = rewards + (1 - dones) * self.gamma * target_q

        # 多步TD
        n_noise = torch.clamp(
            torch.randn_like(n_actions) * self.policy_noise,
            -self.noise_clip, self.noise_clip
        )
        n_target_actions = torch.clamp(
            self.actor_target(n_next_states) + n_noise,
            -self.max_action, self.max_action
        )

        n_target_q1, _ = self.critic1_target(n_next_states, n_target_actions)
        _, n_target_q2 = self.critic2_target(n_next_states, n_target_actions)
        n_target_q = torch.min(n_target_q1, n_target_q2)  # 取较小的Q值（TD3双Q学习）
        multi_step_target = n_rewards + (1 - n_dones) * (self.gamma ** self.replay_buffer.n_step) * n_target_q

        # 混合损失
        # 当前Q值
        current_q1, current_q2 = self.critic1(states, actions)
        # 多步当前Q值
        n_current_q1, _ = self.critic1(n_states, n_actions)
        _, n_current_q2 = self.critic2(n_states, n_actions)

        # 计算混合TD误差，用于更新优先级
        # 单步TD误差
        single_td_errors1 = torch.abs(current_q1 - single_step_target.detach())
        single_td_errors2 = torch.abs(current_q2 - single_step_target.detach())
        single_td_errors = torch.max(single_td_errors1, single_td_errors2)
        # 多步TD误差
        multi_td_errors1 = torch.abs(n_current_q1 - multi_step_target.detach())
        multi_td_errors2 = torch.abs(n_current_q2 - multi_step_target.detach())
        multi_td_errors = torch.max(multi_td_errors1, multi_td_errors2)
        # 混合TD误差，加权平均，与损失函数保持一致
        td_errors = (0.5 * single_td_errors + 0.5 * multi_td_errors).cpu().data.numpy().flatten()
        # 混合单步和多步损失
        critic1_loss_single = F.mse_loss(current_q1, single_step_target.detach(), reduction='none')
        critic1_loss_multi = F.mse_loss(n_current_q1, multi_step_target.detach(), reduction='none')
        critic1_loss = (0.5 * critic1_loss_single + 0.5 * critic1_loss_multi) * is_weights.unsqueeze(1)
        critic1_loss = critic1_loss.mean()
        
        critic2_loss_single = F.mse_loss(current_q2, single_step_target.detach(), reduction='none')
        critic2_loss_multi = F.mse_loss(n_current_q2, multi_step_target.detach(), reduction='none')
        critic2_loss = (0.5 * critic2_loss_single + 0.5 * critic2_loss_multi) * is_weights.unsqueeze(1)
        critic2_loss = critic2_loss.mean()
        
        # 更新Critic网络
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # 更新优先级
        self.replay_buffer.update_priorities(idxs, td_errors)

        # 延迟更新策略网络
        if total_steps % self.policy_freq == 0:
            # 计算策略损失，最大化Q1值
            q1_value, _ = self.critic1(states, self.actor(states))
            actor_loss = -q1_value.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # 软更新
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)