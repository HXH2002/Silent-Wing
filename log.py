import os
from typing import List, Tuple, Optional
from visual import UnifiedVisualizer
import numpy as np


class ExperimentLogger:
    def __init__(self, log_params):

        self.output_dir = log_params['output_dir']
        self.save_csv = log_params.get('save_data', True)
        self.save_plots = log_params.get('show_plots', True)
        self.plot_dpi = log_params['dpi']
        self.figure_size = log_params['figsize']
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.coords_log = []
        self.episode_rewards = [] 
        self.all_deps = [] 
        self.all_throughputs = []

        self.test_coords_log = []
        self.test_trajectories = []
        self.test_rewards = []
        self.test_deps = []  
        self.test_throughputs = []  
    
    def log_training_step(self, episode: int, step: int, state: np.ndarray, 
                         reward: float, info: dict):
        # 记录坐标
        self.coords_log.append([episode, step, state[0], state[1]])
    
    def log_episode_end(self, episode: int, total_reward: float, avg_dep: float, avg_throughput: float):
        self.episode_rewards.append(total_reward)
        self.all_deps.append(avg_dep)
        self.all_throughputs.append(avg_throughput)
    
    def log_test_step(self, episode: int, step: int, state: np.ndarray, 
                     reward: float, info: dict):
        self.test_coords_log.append([episode, step, state[0], state[1]])
    
    def log_test_trajectory(self, trajectory: np.ndarray, reward: float, 
                           deps: List[float], throughputs: List[float]):
        self.test_trajectories.append((trajectory, reward))
        self.test_rewards.append(reward)
        self.test_deps.append(deps)
        self.test_throughputs.append(throughputs)
    
    def save_training_data(self):
        if not self.save_csv:
            return

        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存训练数据
        if self.coords_log:
            np.savetxt(os.path.join(log_dir, "train_coords.csv"), 
                      np.asarray(self.coords_log),
                      delimiter=",", header="episode,step,x,y", comments="", fmt="%.2f")
        
        if self.episode_rewards and self.all_deps and self.all_throughputs:

            avg_deps = [np.mean(deps) if deps else 0.0 for deps in self.all_deps]
            avg_throughputs = [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.all_throughputs]

            episodes = list(range(1, len(self.episode_rewards) + 1))
            combined_data = np.column_stack((episodes, self.episode_rewards, avg_deps, avg_throughputs))

            np.savetxt(os.path.join(log_dir, "train_data.csv"), 
                      combined_data,
                      delimiter=",", header="episode,total_reward,DEP,throughput", comments="", fmt="%.4f")
    
    def save_test_data(self, best_trajectory: np.ndarray = None):
        if not self.save_csv:
            return

        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存测试数据
        if self.test_coords_log:
            np.savetxt(os.path.join(log_dir, "test_coords.csv"), 
                      np.asarray(self.test_coords_log),
                      delimiter=",", header="episode,step,x,y", comments="", fmt="%.2f")

        if self.test_rewards and self.test_deps and self.test_throughputs:

            avg_deps = [np.mean(deps) if deps else 0.0 for deps in self.test_deps]
            avg_throughputs = [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.test_throughputs]

            episodes = list(range(1, len(self.test_rewards) + 1))
            combined_data = np.column_stack((episodes, self.test_rewards, avg_deps, avg_throughputs))

            np.savetxt(os.path.join(log_dir, "test_data.csv"), 
                      combined_data,
                      delimiter=",", header="episode,total_reward,DEP,throughput", comments="", fmt="%.4f")



class ExperimentManager:
    def __init__(self, log_params):
        self.logger = ExperimentLogger(log_params)
        self.output_dir = log_params['output_dir']
    
    def log_training_step(self, episode: int, step: int, state: np.ndarray, reward: float, info: dict):
        self.logger.log_training_step(episode, step, state, reward, info)
    
    def log_training_episode(self, episode: int, total_reward: float, avg_dep: float, avg_throughput: float):
        self.logger.log_episode_end(episode, total_reward, avg_dep, avg_throughput)
    
    def log_test_step(self, episode: int, step: int, state: np.ndarray, reward: float, info: dict):
        self.logger.log_test_step(episode, step, state, reward, info)
    
    def log_test_trajectory(self, trajectory: np.ndarray, reward: float, 
                           deps: List[float], throughputs: List[float]):
        self.logger.log_test_trajectory(trajectory, reward, deps, throughputs)
    
    def generate_all_visualizations(self, best_trajectory: np.ndarray, 
                                  all_trajectories: List[Tuple[np.ndarray, float]],
                                  env):
        

        static_viz = UnifiedVisualizer({}, self.output_dir)
        
        # 保存数据
        self.logger.save_training_data()
        self.logger.save_test_data(best_trajectory)

        # 生成性能指标图
        avg_deps = [np.mean(deps) if deps else 0.0 for deps in self.logger.all_deps]
        avg_throughputs = [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.logger.all_throughputs]
        static_viz.plot_performance_metrics(avg_deps, avg_throughputs, self.logger.episode_rewards)
        
        print(f"所有可视化结果已保存到 {self.output_dir}/ 目录")
    

    
    def visualize_training(self):
        static_viz = UnifiedVisualizer({}, self.output_dir)
        # 生成性能指标图
        if self.logger.all_deps and self.logger.all_throughputs:

            avg_deps = self.logger.all_deps if (self.logger.all_deps and isinstance(self.logger.all_deps[0], (int, float))) else [np.mean(deps) if deps else 0.0 for deps in self.logger.all_deps]
            avg_throughputs = self.logger.all_throughputs if (self.logger.all_throughputs and isinstance(self.logger.all_throughputs[0], (int, float))) else [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.logger.all_throughputs]

            test_deps = None
            test_throughputs = None
            test_rewards = None
            
            if self.logger.test_deps and self.logger.test_throughputs:

                test_deps = [np.mean(deps) if deps else 0.0 for deps in self.logger.test_deps]
                test_throughputs = [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.logger.test_throughputs]
                test_rewards = self.logger.test_rewards
            
            static_viz.plot_performance_metrics(avg_deps, avg_throughputs, self.logger.episode_rewards,test_deps, test_throughputs, test_rewards)
        
        # 保存训练数据
        self.logger.save_training_data()
        print(f"训练可视化结果已保存到 {self.output_dir}/ 目录")
    
    def visualize_testing(self, env):
        if self.logger.all_deps and self.logger.all_throughputs and self.logger.test_deps and self.logger.test_throughputs:

            visualizer = UnifiedVisualizer({}, self.output_dir)

            avg_deps = self.logger.all_deps if (self.logger.all_deps and isinstance(self.logger.all_deps[0], (int, float))) else [np.mean(deps) if deps else 0.0 for deps in self.logger.all_deps]
            avg_throughputs = self.logger.all_throughputs if (self.logger.all_throughputs and isinstance(self.logger.all_throughputs[0], (int, float))) else [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.logger.all_throughputs]

            test_deps = [np.mean(deps) if deps else 0.0 for deps in self.logger.test_deps]
            test_throughputs = [np.mean(thrpts) if thrpts else 0.0 for thrpts in self.logger.test_throughputs]
            test_rewards = self.logger.test_rewards

            visualizer.plot_performance_metrics(avg_deps, avg_throughputs, self.logger.episode_rewards,test_deps, test_throughputs, test_rewards)
        # 保存测试数据
        self.logger.save_test_data()
        print(f"测试可视化结果已保存到 {self.output_dir}/ 目录")