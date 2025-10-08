import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from typing import List, Tuple, Optional, Dict, Any

# 可视化小字典
VISUALIZATION_CONFIG = {
    # 颜色配置
    'building_colors': ['#FFFF99', '#CCFF66', '#99FF33', '#66FF00', '#33CC00', '#009900'],
    'reward_colors': ['#FF0000', '#FF6666', '#FFCCCC', '#FFFFFF', '#CCCCFF', '#6666FF', '#0000FF'],
    'building_edge_color': 'black',
    'building_edge_width': 0.8,
    
    # 透明度配置
    'alpha_transparent': 0.1,
    'alpha_semi': 0.6,
    'alpha_opaque': 0.8,
    
    # 字体配置
    'font_small': 7,
    'font_medium': 8,
    'font_normal': 10,
    'font_large': 12,
    'font_title': 14,
    
    # 网格和图形配置
    'grid_size': 100,
    'colormap_bins': 256,
    'grid_alpha': 0.3,
    'figure_size': (15, 10),
    'dpi_high': 300,
    
    # 颜色条配置
    'colorbar_width': 0.02,
    'colorbar_height': 0.45,
    'colorbar_y_start': 0.5,
    
    'building_colorbar_x': 0.815,
    'reward_colorbar_x': 0.90,
    
    # 轨迹配置
    'trajectory_limit': 10,
    'save_frequency': 50,
    'legend_bbox': (1.024, 0.0),
    
    # 性能指标图
    'dep_color': 'blue',
    'throughput_color': 'green',
    'reward_color': 'purple',
    'moving_avg_color': 'red',
    'moving_avg_window': 20, # 在第540行额外设置了最小5，最大20
}


class EnvironmentRenderer:
    def __init__(self, config: Dict[str, Any], map_size: float):

        self.config = config
        self.map_size = map_size
        self.reward_params = config.get('REWARD', {})
        
    def create_colorbar(self, fig, ax, cmap, norm, x_position, label, ticks=None):

        cax = fig.add_axes([
            x_position, 
            VISUALIZATION_CONFIG['colorbar_y_start'], 
            VISUALIZATION_CONFIG['colorbar_width'], 
            VISUALIZATION_CONFIG['colorbar_height']
        ])
        
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label(label, fontsize=VISUALIZATION_CONFIG['font_medium'], 
                    labelpad=2, rotation=90, va='top')
        cb.ax.tick_params(labelsize=VISUALIZATION_CONFIG['font_medium'])
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')
        
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels([
                f'{t:.0f}' if t == 0 else
                f'{t:.1f}' if isinstance(t, (int, float)) and t.is_integer() 
                else f'{t:.2f}' for t in ticks
            ])
        
        return cb
    
    def draw_buildings(self, fig, ax, env):

        if not (hasattr(env, 'channel_model') and hasattr(env.channel_model, 'buildings')):
            return
            
        buildings = env.channel_model.buildings
        if not buildings:
            return
            
        # 获取建筑高度范围
        heights = [b.building_height for b in buildings]
        min_height, max_height = min(heights), max(heights)
        
        # 创建颜色映射
        cmap = LinearSegmentedColormap.from_list('height_map', 
                                               VISUALIZATION_CONFIG['building_colors'], 
                                               N=VISUALIZATION_CONFIG['colormap_bins'])
        
        # 绘制建筑物
        for building in buildings:
            if max_height > min_height:
                norm_height = (building.building_height - min_height) / (max_height - min_height)
            else:
                norm_height = 0.5
            
            color = cmap(norm_height)
            
            rect = patches.Rectangle(
                (building.x_min, building.y_min),
                building.width, building.height,
                linewidth=VISUALIZATION_CONFIG['building_edge_width'], 
                edgecolor=VISUALIZATION_CONFIG['building_edge_color'], 
                facecolor=color, 
                alpha=VISUALIZATION_CONFIG['alpha_opaque'],
                zorder=10
            )
            ax.add_patch(rect)
            
            # 添加高度标注
            if building.width > 40 and building.height > 40:
                ax.text(building.x, building.y, f'{building.building_height:.0f}m',
                       ha='center', va='center', 
                       fontsize=VISUALIZATION_CONFIG['font_small'], 
                       color='black', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                       zorder=11)
        
        # 创建颜色条
        norm = Normalize(vmin=min_height, vmax=max_height)
        ticks = np.linspace(min_height, max_height, 5)
        self.create_colorbar(fig, ax, cmap, norm, VISUALIZATION_CONFIG['building_colorbar_x'], 
                           'Building Height (m)', ticks)
    
    def draw_reward_field(self, fig, ax, env):

        grid_size = VISUALIZATION_CONFIG['grid_size']
        cell_size = self.map_size / grid_size
        
        # 创建网格中心点坐标
        x_centers = np.linspace(cell_size/2, self.map_size - cell_size/2, grid_size)
        y_centers = np.linspace(cell_size/2, self.map_size - cell_size/2, grid_size)
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # 计算奖励值
        reward_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                x, y = X[i, j], Y[i, j]
                reward_grid[i, j] = self._calculate_position_reward(x, y, env)
        
        # 创建颜色映射
        cmap = LinearSegmentedColormap.from_list('reward_field', 
                                               VISUALIZATION_CONFIG['reward_colors'], 
                                               N=VISUALIZATION_CONFIG['colormap_bins'])
        
        # 设置颜色范围
        vmin, vmax = np.min(reward_grid), np.max(reward_grid)
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        
        # 绘制奖励场
        im = ax.imshow(reward_grid, extent=[0, self.map_size, 0, self.map_size], 
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, 
                      alpha=VISUALIZATION_CONFIG['alpha_semi'], zorder=1)
        
        # 创建颜色条
        norm = Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, 7)
        self.create_colorbar(fig, ax, cmap, norm, VISUALIZATION_CONFIG['reward_colorbar_x'], 
                           'Reward Value', ticks)
        
        return im
    
    def _calculate_position_reward(self, x, y, env):

        position = np.array([x, y])
        total_reward = 0.0
        
        # 边界惩罚（与env.py保持一致）
        boundary_distance = min(x, y, self.map_size - x, self.map_size - y)
        boundary_penalty_threshold = self.reward_params.get('boundary_penalty_threshold', 40.0)
        if boundary_distance < boundary_penalty_threshold:
            boundary_penalty_strength = self.reward_params.get('boundary_penalty_strength', 0.01)
            boundary_field_penalty = -boundary_penalty_strength * (boundary_penalty_threshold - boundary_distance) ** 2
            total_reward += boundary_field_penalty
        
        # 隐蔽洞惩罚（基于DEP计算）
        if hasattr(env, 'wardens') and len(env.wardens) > 0 and hasattr(env, 'channel_model'):
            # 计算对所有监管者的DEP
            min_dep = float('inf')
            for warden in env.wardens:
                dep = env.channel_model.calculate_dep(
                    uav_x=x,
                    uav_y=y,
                    warden_x=warden[0],
                    warden_y=warden[1]
                )
                min_dep = min(min_dep, dep)
            
            # 应用隐蔽性惩罚（与env.py保持一致）
            covertness_threshold = self.reward_params.get('covertness_threshold', 0.5)
            if min_dep < covertness_threshold:
                covert_penalty_weight = self.reward_params.get('covert_penalty_weight', -1000.0)
                covertness_penalty = covert_penalty_weight * (covertness_threshold - min_dep)
                total_reward += covertness_penalty

        # proximity奖励（距离目标越近奖励越高）
        if hasattr(env, 'goal'):
            dist_to_goal = np.linalg.norm(position - env.goal)
            max_distance = np.sqrt(self.map_size**2 + self.map_size**2)  # 对角线距离
            normalized_distance = dist_to_goal / max_distance
            proximity_reward_weight = self.reward_params.get('proximity_reward_weight', 0.05)
            proximity_reward = (1.0 - normalized_distance) * proximity_reward_weight
            total_reward += proximity_reward
        
        # 传输速率奖励（基于信道模型）
        if hasattr(env, 'legitimate_nodes') and len(env.legitimate_nodes) > 0 and hasattr(env, 'channel_model'):
            total_transmission_rate = 0.0
            for legit_node in env.legitimate_nodes:
                rate = env.channel_model.calculate_transmission_rate(
                    uav_x=x,
                    uav_y=y,
                    ground_x=legit_node[0],
                    ground_y=legit_node[1]
                )
                total_transmission_rate += rate
            
            transmission_weight = self.reward_params.get('transmission_weight', 1.0)
            transmission_reward = total_transmission_rate * transmission_weight
            total_reward += transmission_reward
        
        # 基础惩罚（每步的固定成本）
        base_penalty = self.reward_params.get('base_penalty', -1.0)
        total_reward += base_penalty
        
        return total_reward
    
    def draw_covert_holes(self, ax, env):

        if not (hasattr(env, 'wardens') and len(env.wardens) > 0 and 
                hasattr(env, 'channel_model')):
            return
        
        for warden_idx, (warden_x, warden_y) in enumerate(env.wardens):
            # 调用uav.py中的统一射线计算方法
            ray_endpoints = env.channel_model.calculate_ray_endpoints(
                warden_x, warden_y, self.map_size)
            
            # 绘制射线
            for end_x, end_y in ray_endpoints:
                # 绘制每条射线
                ax.plot([warden_x, end_x], [warden_y, end_y], 
                       color='gray', linewidth=0.8, linestyle='-', alpha=0.05, zorder=15)
            
            # 绘制隐蔽洞轮廓线
            if ray_endpoints:
                ray_endpoints.append(ray_endpoints[0])  # 闭合
                xs, ys = zip(*ray_endpoints)
                ax.plot(xs, ys, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, 
                       label='Covert Hole' if warden_idx == 0 else "", zorder=20)
    
    def draw_environment_markers(self, ax, env):

        # 绘制起点和终点
        if hasattr(env, 'start') and hasattr(env, 'goal'):
            ax.scatter(*env.start, c="green", s=100, marker="s", 
                      label="Start", zorder=100)
            ax.scatter(*env.goal, c="red", s=100, marker="s", 
                      label="Goal", zorder=100)
        
        # 绘制监管者
        if hasattr(env, 'wardens') and len(env.wardens) > 0:
            warden_x = [warden[0] for warden in env.wardens]
            warden_y = [warden[1] for warden in env.wardens]
            
            ax.scatter(warden_x, warden_y, c="black", s=100, marker="^", 
                      label=f"Wardens ({len(env.wardens)})", zorder=100)
            
            for i, warden in enumerate(env.wardens):
                ax.annotate(f'W{i+1}', (warden[0], warden[1]), xytext=(5, 5), 
                           textcoords='offset points', 
                           fontsize=VISUALIZATION_CONFIG['font_small'], 
                           color='white', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                   alpha=VISUALIZATION_CONFIG['alpha_semi']), zorder=100)
        
        # 绘制合法节点
        if hasattr(env, 'legitimate_nodes') and len(env.legitimate_nodes) > 0:
            legit_x = [node[0] for node in env.legitimate_nodes]
            legit_y = [node[1] for node in env.legitimate_nodes]
            
            ax.scatter(legit_x, legit_y, c="orange", s=80, marker="o", 
                      label=f"Nodes ({len(env.legitimate_nodes)})", zorder=100)


class UnifiedVisualizer:
    
    # 类变量：跟踪是否已保存初始环境图像
    _initial_env_saved = False
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "figures"):

        self.config = config
        self.output_dir = output_dir
        self.env_params = config.get('ENV', {})
        self.map_size = self.env_params.get('map_size', 1000.0)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 轨迹存储
        self.all_trajectories = []
        self.recent_trajectories = []
        self.step_count = 0
        
        # 图形对象
        self.fig = None
        self.ax = None
        
        # 环境渲染器
        self.env_renderer = EnvironmentRenderer(config, self.map_size)
    
    def initialize_plot(self, env):

        self._setup_plot(env)
        
        # 只在initial_env.png文件不存在时保存初始环境图，这步是为了防止测试时调用和训练过程同一个方法时重复生成一个新的初始环境图
        initial_env_path = os.path.join(self.output_dir, "initial_env.png")
        if not os.path.exists(initial_env_path):
            plt.tight_layout()
            plt.savefig(initial_env_path, dpi=150, bbox_inches='tight')
    
    def _setup_plot(self, env):

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 绘制环境要素
        self.env_renderer.draw_buildings(self.fig, self.ax, env)
        self.env_renderer.draw_reward_field(self.fig, self.ax, env)
        self.env_renderer.draw_covert_holes(self.ax, env)
        self.env_renderer.draw_environment_markers(self.ax, env)
        
        # 设置图表属性
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_xlabel("X Position (m)", fontsize=VISUALIZATION_CONFIG['font_large'])
        self.ax.set_ylabel("Y Position (m)", fontsize=VISUALIZATION_CONFIG['font_large'])
        self.ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        self.ax.set_title("Initial Environment", fontsize=VISUALIZATION_CONFIG['font_title'])
        
        # 设置图例
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
    
    def update_trajectory(self, trajectory: np.ndarray, reward: float, episode):

        self.all_trajectories.append((trajectory.copy(), reward))
        self.recent_trajectories.append((trajectory.copy(), reward))
        
        if len(self.recent_trajectories) > VISUALIZATION_CONFIG['trajectory_limit']:
            self.recent_trajectories.pop(0)
            
        self.step_count += 1
        
        # 按频率保存轨迹图像
        if self.step_count % VISUALIZATION_CONFIG['save_frequency'] == 0:
            self._save_trajectory_image(episode)
    
    def _save_trajectory_image(self, episode):

        if self.fig is None or self.ax is None:
            return
            
        # 清除之前的轨迹线
        for line in self.ax.lines[:]:
            if hasattr(line, '_trajectory_line'):
                line.remove()
                
        # 绘制历史轨迹
        other_trajs_labeled = False
        for trajectory, _ in self.all_trajectories[:-1]:
            if len(trajectory) > 1:
                label = "Other Trajs" if not other_trajs_labeled else ""
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                   color='black', 
                                   alpha=VISUALIZATION_CONFIG['alpha_transparent'], 
                                   linewidth=1, label=label, zorder=30)
                line._trajectory_line = True
                other_trajs_labeled = True
                    
        # 绘制最近10轮中的最佳轨迹
        if self.recent_trajectories:
            best_trajectory = None
            best_reward = float('-inf')
            for trajectory, reward in self.recent_trajectories:
                if reward > best_reward:
                    best_reward = reward
                    best_trajectory = trajectory
                    
            if best_trajectory is not None and len(best_trajectory) > 1:
                line, = self.ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 
                                   color='blue', linewidth=3, 
                                   label='Best Recent Traj', zorder=50)
                line._trajectory_line = True
                
        # 更新图例和标题
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
        
        self.ax.set_title(f"Training Trajectory - Ep {episode}", 
                         fontsize=VISUALIZATION_CONFIG['font_title'])
        filename = f"train_traj_ep_{episode}.png"
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, filename), 
                    dpi=150, bbox_inches='tight')
    
    def _save_test_trajectory_image(self, num_tests):

        if self.fig is None or self.ax is None:
            return
            
        # 清除之前的轨迹线
        for line in self.ax.lines[:]:
            if hasattr(line, '_trajectory_line'):
                line.remove()
                
        # 找到最佳测试轨迹
        if not self.all_trajectories:
            return
            
        best_trajectory = None
        best_reward = float('-inf')
        for trajectory, reward in self.all_trajectories:
            if reward > best_reward:
                best_reward = reward
                best_trajectory = trajectory
        
        # 绘制所有其余轨迹
        other_trajs_labeled = False
        for trajectory, reward in self.all_trajectories:
            if reward != best_reward and len(trajectory) > 1:
                label = "Other Trajs" if not other_trajs_labeled else ""
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                   color='gray', 
                                   alpha=VISUALIZATION_CONFIG['alpha_transparent'], 
                                   linewidth=1, label=label, zorder=30)
                line._trajectory_line = True
                other_trajs_labeled = True
                    
        # 绘制最佳轨迹
        if best_trajectory is not None and len(best_trajectory) > 1:
            line, = self.ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 
                               color='blue', linewidth=3, 
                               label='Best Recent Traj', zorder=50)
            line._trajectory_line = True
                
        # 更新图例和标题
        self.ax.legend(loc='lower left', bbox_to_anchor=VISUALIZATION_CONFIG['legend_bbox'], 
                      fontsize=VISUALIZATION_CONFIG['font_normal'], 
                      borderaxespad=0, frameon=False)
        
        self.ax.set_title(f"Testing Trajectory - Ep {num_tests}", 
                         fontsize=VISUALIZATION_CONFIG['font_title'])
        filename = f"test_traj_ep_{num_tests}.png"
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, filename), 
                    dpi=150, bbox_inches='tight')
    
    def plot_performance_metrics(self, deps: List[float], throughputs: List[float], 
                               episode_rewards: List[float] = None,
                               test_deps: List[float] = None,
                               test_throughputs: List[float] = None,
                               test_rewards: List[float] = None):

        fig, axes = plt.subplots(3, 1, figsize=VISUALIZATION_CONFIG['figure_size'])
        fig.suptitle('Performance Metrics', fontsize=VISUALIZATION_CONFIG['font_title'], y=0.98)
        train_episodes = range(1, len(deps) + 1)
        
        # DEP图
        axes[0].plot(train_episodes, deps, color=VISUALIZATION_CONFIG['dep_color'], linewidth=1, label='Training')
        
        # 添加测试数据
        if test_deps and len(test_deps) > 0:
            test_episodes = range(len(deps) + 1, len(deps) + len(test_deps) + 1)
            axes[0].plot(test_episodes, test_deps, color=VISUALIZATION_CONFIG['dep_color'], linewidth=2, label='Testing')
            # 添加竖向虚线分隔
            axes[0].axvline(x=len(deps) + 0.5, color='gray', linestyle='--', alpha=0.7)
            axes[0].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        
        axes[0].set_ylabel('DEP', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[0].set_title('Detection Error Probability', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[0].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        # 吞吐量图
        axes[1].plot(train_episodes, throughputs, color=VISUALIZATION_CONFIG['throughput_color'], linewidth=1, label='Training')
        
        # 添加测试数据
        if test_throughputs and len(test_throughputs) > 0:
            test_episodes = range(len(throughputs) + 1, len(throughputs) + len(test_throughputs) + 1)
            axes[1].plot(test_episodes, test_throughputs, color=VISUALIZATION_CONFIG['throughput_color'], linewidth=2, label='Testing')
            # 添加竖向虚线分隔
            axes[1].axvline(x=len(throughputs) + 0.5, color='gray', linestyle='--', alpha=0.7)
            axes[1].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        
        axes[1].set_ylabel('Throughput', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[1].set_title('Transmission Rate', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[1].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        # 学习曲线图
        if episode_rewards and len(episode_rewards) > 0:
            reward_episodes = range(1, len(episode_rewards) + 1)
            axes[2].plot(reward_episodes, episode_rewards, 
                        color=VISUALIZATION_CONFIG['reward_color'], 
                        linewidth=1, alpha=VISUALIZATION_CONFIG['alpha_semi'], label='Training')
            
            # 添加测试数据
            if test_rewards and len(test_rewards) > 0:
                test_reward_episodes = range(len(episode_rewards) + 1, len(episode_rewards) + len(test_rewards) + 1)
                axes[2].plot(test_reward_episodes, test_rewards, 
                            color=VISUALIZATION_CONFIG['reward_color'], linewidth=2, label='Testing')
                # 添加竖向虚线分隔
                axes[2].axvline(x=len(episode_rewards) + 0.5, color='gray', linestyle='--', alpha=0.7)
            
            # 添加移动平均线
            if len(episode_rewards) > 10:
                window_size = max(5, min(20, min(VISUALIZATION_CONFIG['moving_avg_window'], len(episode_rewards) // 10)))
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                moving_episodes = range(window_size, len(episode_rewards) + 1)
                axes[2].plot(moving_episodes, moving_avg, 
                            color=VISUALIZATION_CONFIG['moving_avg_color'], 
                            linewidth=1, label=f'Moving Average (window={window_size})')
            
            axes[2].legend(fontsize=VISUALIZATION_CONFIG['font_normal'])
        else:
            axes[2].plot(train_episodes, [0] * len(train_episodes), 
                        color=VISUALIZATION_CONFIG['reward_color'], linewidth=1)
        
        axes[2].set_ylabel('Total Reward', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].set_xlabel('Episode', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].set_title('Learning Curve', fontsize=VISUALIZATION_CONFIG['font_large'])
        axes[2].grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_metrics.png"), 
                   dpi=VISUALIZATION_CONFIG['dpi_high'], bbox_inches='tight')
        plt.close()
    
    def finalize_visualization(self):
        """完成可视化"""
        if self.fig is not None:
            plt.ioff()
            plt.savefig(os.path.join(self.output_dir, "final_trajectory_visualization.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close(self.fig)

# 保留原有类名的别名
DynamicTrajectoryVisualizer = UnifiedVisualizer
StaticVisualizationGenerator = UnifiedVisualizer
