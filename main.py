import time
import numpy as np
import torch
import shutil
import os
from typing import Tuple, List
from env import UAVEnv
from agent import TD3
from log import ExperimentManager
from visual import DynamicTrajectoryVisualizer

# 项目大字典（可视化过程还有一个小的配置字典放在visual.py里，其余全部参数都可以在这里调整）
CONFIG = {
    # 训练参数
    'TRAIN': {
        'num_episodes': 500,           # 训练次数
        'min_buffer_size': 1500,
        'noise_decay_factor': 1.5,     # 噪声衰减因子
        'min_noise': 0.08,             # 最小噪声水平
        'seed': int(time.time()),
    },
    
    # 测试参数
    'TEST': {
        'num_tests': 10,               # 测试次数
    },
    
    # 算法参数
    'TD3': {
        'state_dim': 2,
        'action_dim': 2,
        'batch_size': 32,               # 批处理大小
        'gamma': 0.99,                  # 折扣因子
        'tau': 1e-5,                    # 软更新系数
        'policy_noise': 1.0,            # 目标策略平滑化噪声
        'exploration_noise': 1.0,       # 探索噪声
        'noise_clip': 0.5,
        'policy_freq': 10,              # 网络更新频率
        'max_action': 1.0,
        'replay_buffer_size': 1000000,  # 经验回放缓冲区
        'lr_actor': 1e-4,               # Actor学习率
        'lr_critic': 1e-3,              # Critic学习率
        
        # 多步学习
        'n_step': 6,                    # 步数
        'multi_step_gamma': 0.99,       # 折扣因子

        # PER
        'per_alpha': 0.6,               # 优先级指数
        'per_beta': 0.4,                # 重要性采样指数
        'per_beta_increment': 0.001,    # beta增长率
    },
    
    # 环境参数
    'ENV': {
        'max_steps': 100,                   # 最大步数
        'velocity_magnitude': 50.0,         # 速度幅值 m/s
        'time_slot': 0.5,                   # 时间槽长度 s
        'map_size': 1000.0,                 # 地图尺寸 km^2

        'start_pos': [60.0, 200.0],         # 起点位置
        'goal_pos': [940.0, 600.0],         # 目标位置
        'warden_count_range': [1, 1],       # warden数量范围
        'warden_pos_range': [490, 510],     # warden位置范围
        'legitimate_nodes': [               # 合法节点
            [200.0, 300.0],
            [300.0, 800.0], 
            [700.0, 200.0],
            [800.0, 700.0]
        ],

        # ITU建筑分布
        'itu_alpha': 0.2,
        'itu_beta': 40, 
        'itu_gamma': 25, 
        'density_factor': 1.0, 
        
        # 建筑生成
        'height_range': [10, 70], 
        'width_range': [40, 100], 
        'length_range': [40, 100],               # 长宽高生成范围
        'min_distance': 5.0,                     # 最小间距
        'max_attempts': 10,
        'building_generation_range': [60, 940],  # 生成坐标范围
        'warden_exclusion_size': 50, # 因为建筑是随机生成，有过近遮挡的话效果不好，所以以监测者为中心的50*50不生成建筑

        # 隐蔽洞
        'num_rays': 180,                # 可视化精度
        'max_ray_length': 300.0,        # 半径        
        'min_building_height': 30,      # 高度阈值，低于此高度的建筑不参与遮挡计算

    },
    
    # 奖励函数
    'REWARD': {
        # 基础奖励r = ∑R_l - 1 + 离散惩罚
        'transmission_weight': 5.0,            # 传输速率权重
        'base_penalty': -1.0,                  # -1
        'covertness_threshold': 0.99,          # 隐蔽性阈值
        'covert_penalty_weight': -500.0,       # 隐蔽洞惩罚
        
        # 目标奖励
        'goal_reward': 4000.0,                 # 到达奖励
        'goal_threshold': 20.0,                # 到达阈值

        'distance_guidance_weight': 2,         # 距离改善奖励权重
        'proximity_reward_weight': 1,          # 接近目标奖励权重
        'step_penalty_weight': 0.5,            # 步数效率惩罚权重
        
        # 边界惩罚
        'boundary_penalty': -10000.0,          # 碰撞惩罚
        'boundary_penalty_threshold': 40.0,    # 场惩罚，作用距离
        'boundary_penalty_strength': 0.3,      # 场惩罚，强度系数
        
        # 振荡惩罚
        'oscillation_penalty': -100.0,         # 惩罚值
        'oscillation_history_length': 12,      # 历史记录步数
        'oscillation_check_steps': 6,          # 检测步数
        'oscillation_distance_threshold': 12,  # 检测距离
        
    },
    
    # 信道
    'CHANNEL': {
        # 信道参数
        'fc': 2e9,                  # 载波频率 2GHz
        'uav_height': 100,          # UAV高度
        'tx_power': 30,             # UAV功率 dBm
        'noise_power': -90,         # AWGN功率 dBm
        'noise_uncertainty': 3,     # 噪声不确定度 dB
        'nominal_noise': 1e-6,      # 标称噪声功率

        'location_error_var': 0.025,
        'num_evaluations': 1000,
        
        # 路径损耗参数
        'los_a': 22.0,
        'los_b': 28.0,
        'los_c': 20.0,
        'nlos_a': 36.7,
        'nlos_b': 22.7, 
        'nlos_c': 26.0, 
        'nlos_d': 0.3, 
    },
    
    # 日志
    'LOG': {
        'output_dir': 'figures', 
        'save_data': True,                 # 是否保存CSV
        'show_plots': False,               # 是否显示图表
        'show_dynamic_training': True,     # 是否显示动态可视化
        'dpi': 300,
        'figsize': (12, 8),                # 图像尺寸
    }
}




# 训练模块
def train() -> Tuple[TD3, List[float], 'DynamicTrajectoryVisualizer', 'ExperimentManager']:

    # 初始化
    env = UAVEnv(env_params=CONFIG['ENV'], reward_params=CONFIG['REWARD'], channel_params=CONFIG['CHANNEL'])
    agent = TD3(state_dim=CONFIG['TD3']['state_dim'], action_dim=CONFIG['TD3']['action_dim'], td3_params=CONFIG['TD3'])

    np.random.seed(CONFIG['TRAIN']['seed'])
    torch.manual_seed(CONFIG['TRAIN']['seed'])

    exp_manager = ExperimentManager(log_params=CONFIG['LOG'])
    dynamic_visualizer = None

    if CONFIG['LOG']['show_dynamic_training']:
        dynamic_visualizer = DynamicTrajectoryVisualizer(CONFIG, CONFIG['LOG']['output_dir'])
        dynamic_visualizer.initialize_plot(env)

    total_steps = 0 # 总步数计数器

    while len(agent.replay_buffer) < CONFIG['TRAIN']['min_buffer_size']:
        s = env.reset()
        done = False
        while not done:
            # 随机动作
            a = np.random.uniform(-1.0, 1.0, size=agent.action_dim)
            s_, r, done, info = env.step(a)
            agent.store_transition(s, a, r, s_, done)
            s = s_
            total_steps += 1

    # 开始正式训练
    for ep in range(CONFIG['TRAIN']['num_episodes']):

        s = env.reset()
        done = False
        ep_reward, ep_deps, ep_throughputs = 0.0, [], []
        ep_trajectory = [env.state.copy()]

        base_noise = np.sqrt(CONFIG['TD3']['exploration_noise'])
        noise = max(base_noise * np.exp(-CONFIG['TRAIN']['noise_decay_factor'] * ep / CONFIG['TRAIN']['num_episodes']),
                    CONFIG['TRAIN']['min_noise'])

        # episode内部循环
        while not done:
            a = agent.select_action(s, noise)
            s_, r, done, info = env.step(a)
            agent.store_transition(s, a, r, s_, done)
            s = s_

            ep_trajectory.append(env.state.copy())
            ep_reward += r
            ep_deps.append(info['DEP'])
            ep_throughputs.append(info['throughput'])

            exp_manager.logger.log_training_step(ep, env.current_step, env.state, r, info)

            total_steps += 1
            agent.update(total_steps)

        # 记录
        avg_dep = np.mean(ep_deps) if ep_deps else 0.0
        avg_throughput = np.mean(ep_throughputs) if ep_throughputs else 0.0
        exp_manager.log_training_episode(ep, ep_reward, avg_dep, avg_throughput)
        
        if CONFIG['LOG']['show_dynamic_training'] and dynamic_visualizer is not None:
            dynamic_visualizer.update_trajectory(np.asarray(ep_trajectory), ep_reward, ep + 1)

        # 训练进度
        print(f"Episode {ep + 1}: Reward={ep_reward:.2f}, DEP={avg_dep:.4f}, "f"Throughput={avg_throughput:.4f}, Noise={noise:.3f}")

    exp_manager.visualize_training()
    return agent, exp_manager.logger.episode_rewards, dynamic_visualizer, exp_manager, env


#  测试模块
def test(agent: TD3, exp_manager: ExperimentManager, env: UAVEnv, train_dynamic_visualizer=None):


    rewards_list, deps_list, thrpts_list = [], [], []
    all_trajectories = []
    all_deps, all_thrpts = [], []

    test_dynamic_visualizer = train_dynamic_visualizer

    for test_run in range(CONFIG['TEST']['num_tests']):

        s = env.reset()
        done = False
        traj, deps, thrpts, rs = [env.state.copy()], [], [], []
        step = 0

        while not done:
            a = agent.select_action(s, noise=0)  # 确定性策略
            s_, r, done, info = env.step(a)

            exp_manager.log_test_step(test_run + 1, step, env.state, r, info)

            traj.append(env.state.copy())
            deps.append(info['DEP'])
            thrpts.append(info['throughput'])
            rs.append(r)
            s = s_
            step += 1

        # 计算并记录
        total_reward = sum(rs)
        rewards_list.append(total_reward)
        deps_list.append(np.mean(deps))
        thrpts_list.append(sum(thrpts))

        trajectory_array = np.asarray(traj)
        all_trajectories.append((trajectory_array, total_reward))
        all_deps.append(deps)
        all_thrpts.append(thrpts)

        exp_manager.log_test_trajectory(trajectory_array, total_reward, deps, thrpts)

        # 测试进度
        print(f"Test {test_run + 1}: Reward={total_reward:.2f}, "
              f"DEP={np.mean(deps):.4f}, "
              f"Throughput={sum(thrpts):.4f}")

        if test_dynamic_visualizer is not None:
            test_dynamic_visualizer.update_trajectory(trajectory_array, total_reward, test_run + 1)
    
    exp_manager.visualize_testing(env)

    if test_dynamic_visualizer is not None:
        test_dynamic_visualizer._save_test_trajectory_image(CONFIG['TEST']['num_tests'])

    return rewards_list, deps_list, thrpts_list, all_trajectories, env


# 主程序
def main():

    figures_dir = "figures"
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    log_dir = "log"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print("已清空figures和log文件夹")
    
    print("\n开始训练...")
    agent, rewards, train_dynamic_visualizer, exp_manager, env = train()

    print("\n开始测试...")
    test(agent, exp_manager, env, train_dynamic_visualizer)

    print("训练与测试完成")


if __name__ == "__main__":
    main()