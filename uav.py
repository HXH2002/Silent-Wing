import numpy as np
import random
from typing import List, Tuple, Dict
from scipy.stats import chi2


class Building:

    def __init__(self, x: float, y: float, width: float, height: float, building_height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.building_height = building_height
        
        # 计算建筑物边界
        self.x_min = x - width / 2
        self.x_max = x + width / 2
        self.y_min = y - height / 2
        self.y_max = y + height / 2
    
    def intersects_line(self, x1: float, y1: float, x2: float, y2: float) -> bool:

        # 使用Liang-Barsky线段裁剪算法
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return False
            
        t_min = 0.0
        t_max = 1.0
        
        # 检查x方向
        if dx != 0:
            t1 = (self.x_min - x1) / dx
            t2 = (self.x_max - x1) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        else:
            if x1 < self.x_min or x1 > self.x_max:
                return False
                
        # 检查y方向
        if dy != 0:
            t1 = (self.y_min - y1) / dy
            t2 = (self.y_max - y1) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        else:
            if y1 < self.y_min or y1 > self.y_max:
                return False
                
        return True


class ChannelModel:
    
    def __init__(self, channel_params, env_params=None, area_size_km: float = 1.0, seed: int = 42):

         # 验证必需参数
         if channel_params is None:
             raise ValueError("channel_params is required and cannot be None")
         if env_params is None:
             raise ValueError("env_params is required and cannot be None")
         
         # 直接使用传入的参数
         self.channel_params = channel_params
         self.env_params = env_params
         
         # 从building_generation_range计算区域大小
         generation_range = self.env_params.get('building_generation_range', [0, 1000])
         range_min, range_max = generation_range
         self.area_size_m = range_max - range_min  # 区域大小等于建筑生成范围的跨度
         self.area_size_km = self.area_size_m / 1000  # 转换为公里
         
         # 设置随机种子确保可重现性
         np.random.seed(seed)
         random.seed(seed)
         
         # 生成建筑物分布
         self.buildings = self._generate_buildings()
    
    def _generate_buildings(self) -> List[Building]:
        buildings = []

        # 从env_params获取建筑生成参数
        itu_beta = self.env_params.get('itu_beta', 40)
        density_factor = self.env_params.get('density_factor', 1.0)
        height_range = self.env_params.get('height_range', [10, 70])
        width_range = self.env_params.get('width_range', [40, 100])
        length_range = self.env_params.get('length_range', [40, 100])
        min_distance = self.env_params.get('min_distance', 5.0)
        max_attempts = self.env_params.get('max_attempts', 10)
        
        # 获取建筑生成范围参数
        generation_range = self.env_params.get('building_generation_range', [0, 1000])
        range_min, range_max = generation_range
        
        # 计算建筑物数量：β * D^2 * density_factor
        base_num_buildings = int(itu_beta * (self.area_size_km ** 2))
        num_buildings = int(base_num_buildings * density_factor)
        
        for i in range(num_buildings):
            building_placed = False
            
            for attempt in range(max_attempts):
                # 随机生成建筑物尺寸
                width = np.random.uniform(*width_range)
                length = np.random.uniform(*length_range)

                # 在指定范围内随机生成建筑物位置
                x_min = max(range_min + width/2, width/2)
                x_max = min(range_max - width/2, self.area_size_m - width/2)
                y_min = max(range_min + length/2, length/2)
                y_max = min(range_max - length/2, self.area_size_m - length/2)

                # 检查有效生成区域
                if x_min >= x_max or y_min >= y_max:
                    continue
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
                # 生成建筑物高度
                building_height = np.random.uniform(*height_range)
                
                # 创建候选建筑物
                candidate = Building(x, y, width, length, building_height)
                
                # 检查是否与现有建筑物重合
                if self._is_building_valid(candidate, buildings, min_distance):
                    buildings.append(candidate)
                    building_placed = True
                    break
            
            # 如果建筑物放置失败，静默跳过
        return buildings
    
    def _is_building_valid(self, candidate: Building, existing_buildings: List[Building], min_distance: float) -> bool:

        # 检查是否与现有建筑物重合
        for existing in existing_buildings:
            # 计算两个建筑物边界之间的最小距离
            dx = max(0, max(existing.x_min - candidate.x_max, candidate.x_min - existing.x_max))
            dy = max(0, max(existing.y_min - candidate.y_max, candidate.y_min - existing.y_max))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                return False
        
        # 检查是否与监测者区域重合
        if self._is_in_warden_exclusion_zone(candidate):
            return False
        
        return True
    
    def _is_in_warden_exclusion_zone(self, candidate: Building) -> bool:

        # 获取监测者位置范围和排除区域大小
        warden_pos_range = self.env_params.get('warden_pos_range', [490, 510])
        exclusion_size = self.env_params.get('warden_exclusion_size', 100)
        
        # 计算监测者区域的中心位置（假设监测者在范围中心）
        warden_center_x = (warden_pos_range[0] + warden_pos_range[1]) / 2
        warden_center_y = (warden_pos_range[0] + warden_pos_range[1]) / 2
        
        # 计算监测者排除区域的边界
        exclusion_half_size = exclusion_size / 2
        exclusion_x_min = warden_center_x - exclusion_half_size
        exclusion_x_max = warden_center_x + exclusion_half_size
        exclusion_y_min = warden_center_y - exclusion_half_size
        exclusion_y_max = warden_center_y + exclusion_half_size
        
        # 检查建筑是否与排除区域重合（矩形重合检测）
        if (candidate.x_max >= exclusion_x_min and candidate.x_min <= exclusion_x_max and
            candidate.y_max >= exclusion_y_min and candidate.y_min <= exclusion_y_max):
            return True
        
        return False
    
    def is_los(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> bool:

        # 检查是否有建筑物阻挡
        for building in self.buildings:
            if building.intersects_line(uav_x, uav_y, ground_x, ground_y):
                # 进一步检查建筑物高度是否足以阻挡信号
                # 简化模型：如果建筑物高度超过UAV高度的一定比例则阻挡
                if building.building_height > self.channel_params['uav_height'] * 0.3:
                    return False
        return True
    
    def calculate_pathloss(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        # 计算3D距离
        horizontal_dist = np.sqrt((uav_x - ground_x)**2 + (uav_y - ground_y)**2)
        distance_3d = np.sqrt(horizontal_dist**2 + self.channel_params['uav_height']**2)
        
        # 频率 (GHz)
        fc_ghz = self.channel_params['fc'] / 1e9
        
        # 判断LoS/NLoS
        is_los_link = self.is_los(uav_x, uav_y, ground_x, ground_y)
        
        if is_los_link:
            # LoS路径损耗模型
            pathloss = (self.channel_params['los_a'] * np.log10(distance_3d) + 
                       self.channel_params['los_b'] + 
                       self.channel_params['los_c'] * np.log10(fc_ghz))
        else:
            # NLoS路径损耗模型
            pathloss = (self.channel_params['nlos_a'] * np.log10(distance_3d) + 
                       self.channel_params['nlos_b'] + 
                       self.channel_params['nlos_c'] * np.log10(fc_ghz) + 
                       self.channel_params['nlos_d'] * horizontal_dist)
        
        return pathloss
    
    def calculate_snr(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        pathloss_db = self.calculate_pathloss(uav_x, uav_y, ground_x, ground_y)
        received_power_db = self.channel_params['tx_power'] - pathloss_db
        snr_db = received_power_db - self.channel_params['noise_power']
        return snr_db
    
    def calculate_transmission_rate(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        snr_db = self.calculate_snr(uav_x, uav_y, ground_x, ground_y)
        snr_linear = 10 ** (snr_db / 10)
        rate = np.log2(1 + snr_linear)
        return rate
    

    
    def calculate_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # 使用射线辅助的DEP计算方法
        return self._calculate_ray_assisted_dep(uav_x, uav_y, warden_x, warden_y)

    def _calculate_exact_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # 关键修复：检查距离限制
        max_ray_length = self.env_params['max_ray_length']
        distance_to_warden = np.sqrt((warden_x - uav_x)**2 + (warden_y - uav_y)**2)
        
        # 如果距离超过射线长度，返回完全隐蔽（DEP=1.0）
        if distance_to_warden > max_ray_length:
            return 1.0
        
        # 获取参数
        P_u = 10 ** (self.channel_params['tx_power'] / 10)  # 线性功率 mW
        iota = self.channel_params['noise_uncertainty']  # dB
        iota_hat = self.channel_params['nominal_noise']  # 线性值
        
        # 计算路径损耗
        pathloss_w = self.calculate_pathloss(uav_x, uav_y, warden_x, warden_y)
    
        # 计算接收功率
        received_power = P_u * (10 ** (-pathloss_w / 10))
        
        # 计算噪声不确定性边界
        iota_linear = 10 ** (iota / 10)
        noise_upper = iota_linear * iota_hat
        noise_lower = iota_hat / iota_linear
        
        # 数值稳定性检查
        if received_power <= 1e-12:  # 接收功率极小
            return 1.0
        
        if noise_upper <= noise_lower:
            return 0.0
        
        # 计算DEP
        if received_power >= (noise_upper - noise_lower):
            return 0.0
        denominator = received_power + noise_lower
        if denominator <= 1e-12:
            return 1.0
        ratio = noise_upper / denominator
        if ratio <= 1.0:
            return 0.0

        try:
            log_ratio = np.log(ratio)
            log_base = 2 * iota * np.log(10) / 10  # 简化的对数计算
            
            if log_base <= 1e-12:
                return 1.0
            
            dep = log_ratio / log_base
            return np.clip(dep, 0.0, 1.0)
        except (ValueError, OverflowError, ZeroDivisionError):
            # 数值计算异常时的安全返回值
            return 1.0 if ratio > 1.0 else 0.0
    
    def is_covert_hole(self, uav_x: float, uav_y: float, warden_positions: List[Tuple[float, float]]) -> bool:
        return self._is_ray_assisted_covert_hole(uav_x, uav_y, warden_positions)
    
    def _calculate_ray_assisted_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # 计算DEP
        base_dep = self._calculate_exact_dep(uav_x, uav_y, warden_x, warden_y)
        
        # 计算射线遮挡因子
        obstruction_factor = self._calculate_ray_obstruction_factor(uav_x, uav_y, warden_x, warden_y)

        # 遮挡越严重，DEP越高
        adjusted_dep = base_dep + obstruction_factor * (1.0 - base_dep) * self.channel_params.get('obstruction_weight', 0.3)
        
        return np.clip(adjusted_dep, 0.0, 1.0)
    
    def _calculate_ray_obstruction_factor(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # 使用配置参数
        num_rays = self.env_params['num_rays']
        angle_step = 360.0 / num_rays
        max_ray_length = self.env_params['max_ray_length']
        
        # 计算UAV到监管者的方向
        dx_to_warden = warden_x - uav_x
        dy_to_warden = warden_y - uav_y
        distance_to_warden = np.sqrt(dx_to_warden**2 + dy_to_warden**2)
        
        if distance_to_warden < 1e-6:
            return 0.0 

        if distance_to_warden > max_ray_length:
            return 0.0
        
        # 监管者方向的角度
        warden_angle = np.degrees(np.arctan2(dy_to_warden, dx_to_warden))
        if warden_angle < 0:
            warden_angle += 360
        
        # 计算相关射线的遮挡情况
        relevant_rays = []
        obstruction_count = 0
        
        # 选择监管者方向附近的射线，±45度范围内
        for ray_idx in range(num_rays):
            ray_angle = ray_idx * angle_step
            angle_diff = abs(ray_angle - warden_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= 45:  # 监管者方向±45度内的射线
                relevant_rays.append(ray_idx)
                
                # 计算射线方向
                angle_rad = np.radians(ray_angle)
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)
                
                # 计算射线与建筑物的交点
                ray_length = self._calculate_ray_building_intersection_distance(
                    uav_x, uav_y, dx, dy, max_ray_length)
                
                # 如果射线在到达监管者之前被建筑物阻挡，则计为遮挡
                if ray_length < distance_to_warden:
                    obstruction_count += 1
        
        if not relevant_rays:
            return 0.0
        
        # 计算遮挡比例
        obstruction_ratio = obstruction_count / len(relevant_rays)
        
        # 应用距离衰减：距离越远，遮挡影响越小
        distance_factor = min(1.0, distance_to_warden / max_ray_length)
        
        return obstruction_ratio * (1.0 - distance_factor * 0.5)
    
    def _calculate_ray_building_intersection_distance(self, start_x: float, start_y: float, 
                                                    dx: float, dy: float, max_length: float) -> float:

        min_distance = max_length
        
        for building in self.buildings:
            # 只考虑足够高的建筑物
            min_height = self.env_params.get('min_building_height', 40)
            if hasattr(building, 'building_height') and building.building_height <= min_height:
                continue
                
            x_min, x_max = building.x_min, building.x_max
            y_min, y_max = building.y_min, building.y_max
            
            intersections = []
            
            # 检查四条边的交点
            if abs(dx) > 1e-10:  # 避免除零
                # 左边界
                t = (x_min - start_x) / dx
                if 0 < t < max_length:
                    y_intersect = start_y + t * dy
                    if y_min <= y_intersect <= y_max:
                        intersections.append(t)
                
                # 右边界
                t = (x_max - start_x) / dx
                if 0 < t < max_length:
                    y_intersect = start_y + t * dy
                    if y_min <= y_intersect <= y_max:
                        intersections.append(t)
            
            if abs(dy) > 1e-10:  # 避免除零
                # 下边界
                t = (y_min - start_y) / dy
                if 0 < t < max_length:
                    x_intersect = start_x + t * dx
                    if x_min <= x_intersect <= x_max:
                        intersections.append(t)
                
                # 上边界
                t = (y_max - start_y) / dy
                if 0 < t < max_length:
                    x_intersect = start_x + t * dx
                    if x_min <= x_intersect <= x_max:
                        intersections.append(t)
            
            if intersections:
                min_distance = min(min_distance, min(intersections))
        
        return min_distance

    def calculate_ray_endpoints(self, start_x: float, start_y: float, map_size: float) -> List[Tuple[float, float]]:

        # 使用配置参数
        num_rays = self.env_params['num_rays']
        angle_step = 360.0 / num_rays
        max_ray_length = self.env_params['max_ray_length'] 
        
        ray_endpoints = []
        
        for ray_idx in range(num_rays):
            angle_deg = ray_idx * angle_step
            angle_rad = np.radians(angle_deg)
            
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            
            # 计算建筑物交点距离
            building_distance = self._calculate_ray_building_intersection_distance(
                start_x, start_y, dx, dy, max_ray_length)
            
            # 射线长度由建筑物遮挡或max_ray_length决定
            ray_length = min(max_ray_length, building_distance)
            
            # 计算射线终点
            end_x = start_x + ray_length * dx
            end_y = start_y + ray_length * dy
            ray_endpoints.append((end_x, end_y))
        
        return ray_endpoints

    def _is_ray_assisted_covert_hole(self, uav_x: float, uav_y: float, 
                                   warden_positions: List[Tuple[float, float]]) -> bool:

        covert_threshold = self.env_params['covertness_threshold']
        
        for warden_x, warden_y in warden_positions:
            # 使用射线辅助的DEP计算
            dep = self._calculate_ray_assisted_dep(uav_x, uav_y, warden_x, warden_y)
            
            # 对任何一个监管者的DEP小于等于阈值，不是隐蔽洞
            if dep <= covert_threshold:
                return False
        
        # 只有对所有监管者的DEP都大于阈值时，才是隐蔽洞
        return True