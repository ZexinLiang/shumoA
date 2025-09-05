import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.optimize import differential_evolution
import functools
# --------------------------
# Constants from PDF
# --------------------------
g = 9.80  # m/s^2
cloud_descent_rate = 3.0  # m/s
cloud_radius = 10.0  # m
cloud_duration = 20.0  # s
missile_speed = 300.0  # m/s
real_target_radius = 7.0  # m
real_target_height = 10.0  # m

# Missile initial positions (from PDF)
M1_pos0 = np.array([20000.0, 0.0, 2000.0])
M2_pos0 = np.array([19000.0, 600.0, 2100.0])
M3_pos0 = np.array([18000.0, -600.0, 1900.0])

# UAV initial positions (FY1.FY2.FY3.FY4.FY5) -- from PDF
FY_pos = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0])
}

# Targets: fake (origin) and true (given)
fake_target = np.array([0.0, 0.0, 0.0])
true_target = np.array([0.0, 200.0, 0.0])  # 真目标下底心 (0,200,0)

# Constraints
uav_speed_bounds = (70.0, 140.0)
min_drop_interval = 1.0  # s between drops on same UAV
max_per_uav_problem5 = 3

# --------------------------
# Utility math functions
# --------------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v.copy()

def missile_velocity(missile_pos0: np.ndarray) -> np.ndarray:
    return unit(fake_target - missile_pos0) * missile_speed

def missile_pos(missile_pos0: np.ndarray, t: float) -> np.ndarray:
    return missile_pos0 + missile_velocity(missile_pos0) * t

def uav_velocity_from_heading(speed: float, heading_rad: float) -> np.ndarray:
    # heading absolute angle in XY plane; z component zero (等高度)
    return np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0]) * speed

def explosion_position(uav_pos0: np.ndarray, v_uav: np.ndarray, t_release: float, fuse_delay: float) -> Tuple[float, np.ndarray]:
    """
    Returns (t_explosion, explosion_pos)
    projectile initial vel = v_uav (equal to UAV horizontal vel), vertical initial vel = 0
    displacement during fuse_delay: horizontal v_uav * dt, vertical -0.5*g*dt^2
    """
    t_expl = float(t_release + fuse_delay)
    release_pos = uav_pos0 + v_uav * t_release
    dt = float(fuse_delay)
    disp = v_uav * dt + np.array([0.0, 0.0, -0.5 * g * dt * dt])
    return t_expl, release_pos + disp

def cloud_center_at(t: float, explosion_pos: np.ndarray, t_explosion: float) -> np.ndarray:
    # valid for t >= t_explosion
    return explosion_pos + np.array([0.0, 0.0, -cloud_descent_rate * (t - t_explosion)])

# --------------------------
# Segment-sphere intersection (vectorized over samples)
# Check segment P->Q intersects sphere (C,R)
# --------------------------
def segment_sphere_intersect_single(P: np.ndarray, Q: np.ndarray, C: np.ndarray, R: float) -> bool:
    d = Q - P
    f = P - C
    a = np.dot(d, d)
    if abs(a) < 1e-12:
        return np.dot(P - C, P - C) <= R * R
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - R * R
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sd = math.sqrt(disc)
    t1 = (-b - sd) / (2 * a)
    t2 = (-b + sd) / (2 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def segment_sphere_intersect_times(Ps: np.ndarray, Q: np.ndarray, Cs: np.ndarray, R: float) -> np.ndarray:
    """
    Vectorized check for many segments Pi->Q against many spheres with centers Ci and same radius R.
    Ps: (N,3) missile positions at times
    Qs: single Q (3,) true_target
    Cs: (N,3) cloud centers at times
    Returns boolean array length N
    """
    # d = Q - P
    d = Q[None, :] - Ps  # (N,3)
    f = Ps - Cs  # (N,3)
    a = np.einsum('ij,ij->i', d, d)  # (N,)
    b = 2.0 * np.einsum('ij,ij->i', f, d)
    c = np.einsum('ij,ij->i', f, f) - R * R
    flags = np.zeros(a.shape[0], dtype=bool)
    eps = 1e-12
    deg = a < eps
    if np.any(deg):
        flags[deg] = c[deg] <= 0.0
    nondeg_idx = np.where(~deg)[0]
    if nondeg_idx.size:
        a_nd = a[nondeg_idx]; b_nd = b[nondeg_idx]; c_nd = c[nondeg_idx]
        disc = b_nd * b_nd - 4.0 * a_nd * c_nd
        valid = disc >= 0.0
        if np.any(valid):
            idxs = nondeg_idx[valid]
            disc_v = disc[valid]
            sd = np.sqrt(disc_v)
            b_sub = b[idxs]; a_sub = a[idxs]
            t1 = (-b_sub - sd) / (2.0 * a_sub)
            t2 = (-b_sub + sd) / (2.0 * a_sub)
            flags[idxs] = ((t1 >= 0.0) & (t1 <= 1.0)) | ((t2 >= 0.0) & (t2 <= 1.0))
    return flags

# 采样圆柱体表面点
def sample_cylinder_surface(center, radius, height, n_theta=16, n_h=6):
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    hs = np.linspace(0, height, n_h)
    points = []
    for theta in thetas:
        for h in hs:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = center[2] + h
            points.append([x, y, z])
    return np.array(points)  # (n_theta*n_h, 3)

# --------------------------
# 计算遮蔽时间区间
# --------------------------
def find_shield_intervals_for_explosion(missile_pos0: np.ndarray, explosion_pos: np.ndarray, t_explosion: float,
                                        dt_sample: float = 0.02, radius: float = cloud_radius) -> List[Tuple[float,float]]:
    t0 = t_explosion
    t1 = t_explosion + cloud_duration
    times = np.arange(t0, t1 + dt_sample*0.5, dt_sample)
    if times.size == 0:
        return []
    # 计算所有采样时间点的导弹位置
    Ms = np.array([missile_pos(missile_pos0, t) for t in times])  # (T,3)
    # 计算所有采样时间点的云中心位置
    Cs = np.array([cloud_center_at(t, explosion_pos, t_explosion) for t in times])  # (T,3)
    #采样圆柱体表面点
    surface_points = sample_cylinder_surface(true_target, real_target_radius, real_target_height, n_theta=16, n_h=6)  # 可调整采样密度
    # 判断每个时间点的遮蔽情况
    # 将真实目标物视为圆柱体情况
    flags = np.zeros(len(times), dtype=bool)
    for i in range(len(times)):
        missile_p = Ms[i]
        cloud_c = Cs[i]
        all_shielded = True  # 只有所有表面点都被遮蔽才算遮蔽
        for q in surface_points:
            if not segment_sphere_intersect_single(missile_p, q, cloud_c, radius):
                all_shielded = False
                break
        flags[i] = all_shielded
    # 计算遮蔽时间区间
    intervals = []
    in_seg = False
    start = None
    for i, t in enumerate(times):
        if flags[i] and not in_seg:
            in_seg = True; start = t
        if (not flags[i]) and in_seg:
            in_seg = False; intervals.append((start, times[i-1]))
    if in_seg:
        intervals.append((start, times[-1]))
    # 找到的粗略区间再细化端点，使用二分法
    refined = []
    for (s, e) in intervals:
        # 位于 [s - dt, s]
        left = max(t0, s - dt_sample); right = s
        for _ in range(25):
            mid = 0.5 * (left + right)
            Pm = missile_pos(missile_pos0, mid)
            Cm = cloud_center_at(mid, explosion_pos, t_explosion)
            # 细化时也要对圆柱体表面所有点判断
            all_shielded = True
            for q in surface_points:
                if not segment_sphere_intersect_single(Pm, q, Cm, radius):
                    all_shielded = False
                    break
            if all_shielded:
                right = mid
            else:
                left = mid
        s_ref = right
        # 位于 [e, e + dt]
        left = e; right = min(t1, e + dt_sample)
        for _ in range(25):
            mid = 0.5 * (left + right)
            Pm = missile_pos(missile_pos0, mid)
            Cm = cloud_center_at(mid, explosion_pos, t_explosion)
            all_shielded = True
            for q in surface_points:
                if not segment_sphere_intersect_single(Pm, q, Cm, radius):
                    all_shielded = False
                    break
            if all_shielded:
                left = mid
            else:
                right = mid
        e_ref = left
        if e_ref > s_ref + 1e-9:
            refined.append((s_ref, e_ref))
    return refined
#视为质点情况
def find_shield_intervals_for_explosion_P(missile_pos0: np.ndarray, explosion_pos: np.ndarray, t_explosion: float,
                                        dt_sample: float = 0.02, radius: float = cloud_radius) -> List[Tuple[float,float]]:
    t0 = t_explosion
    t1 = t_explosion + cloud_duration
    times = np.arange(t0, t1 + dt_sample*0.5, dt_sample)
    if times.size == 0:
        return []
    # 计算所有采样时间点的导弹位置
    Ms = np.array([missile_pos(missile_pos0, t) for t in times])  # (T,3)
    # 计算所有采样时间点的云中心位置
    Cs = np.array([cloud_center_at(t, explosion_pos, t_explosion) for t in times])  # (T,3)
    flags = segment_sphere_intersect_times(Ms, true_target, Cs, radius)
    # 计算遮蔽时间区间
    intervals = []
    in_seg = False
    start = None
    for i, t in enumerate(times):
        if flags[i] and not in_seg:
            in_seg = True; start = t
        if (not flags[i]) and in_seg:
            in_seg = False; intervals.append((start, times[i-1]))
    if in_seg:
        intervals.append((start, times[-1]))
    # 找到的粗略区间再细化端点，使用二分法
    refined = []
    for (s, e) in intervals:
        # 位于 [s - dt, s]
        left = max(t0, s - dt_sample); right = s
        for _ in range(25):
            mid = 0.5 * (left + right)
            Pm = missile_pos(missile_pos0, mid)
            Cm = cloud_center_at(mid, explosion_pos, t_explosion)
            if segment_sphere_intersect_single(Pm, true_target, Cm, radius):
                right = mid
            else:
                left = mid
        s_ref = right
        # 位于 [e, e + dt]
        left = e; right = min(t1, e + dt_sample)
        for _ in range(25):
            mid = 0.5 * (left + right)
            Pm = missile_pos(missile_pos0, mid)
            Cm = cloud_center_at(mid, explosion_pos, t_explosion)
            if segment_sphere_intersect_single(Pm, true_target, Cm, radius):
                left = mid
            else:
                right = mid
        e_ref = left
        if e_ref > s_ref + 1e-9:
            refined.append((s_ref, e_ref))
    return refined

def merge_intervals(intervals: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = []
    s, e = ints[0]
    for ss, ee in ints[1:]:
        if ss <= e + 1e-9:
            e = max(e, ee)
        else:
            merged.append((s, e)); s, e = ss, ee
    merged.append((s, e))
    return merged

def total_time(intervals: List[Tuple[float,float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)

# --------------------------
# 问题1
# --------------------------
def problem1():
    speed = 120.0# 无人机速度
    t_release = 1.5 # 投放时间
    fuse_delay = 3.6 # 引信延时
    heading = math.atan2((fake_target - FY_pos["FY1"])[1], (fake_target - FY_pos["FY1"])[0])

    # 计算引爆位置和时间
    v_uav = uav_velocity_from_heading(speed, heading)
    t_expl, expl_pos = explosion_position(FY_pos["FY1"], v_uav, t_release, fuse_delay)
    #计算遮蔽时间区间
    intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
    tot = total_time(intervals)
    out = {
        "t_release": t_release,
        "fuse_delay": fuse_delay,
        "t_explosion": t_expl,
        "explosion_pos": expl_pos,
        "intervals": intervals,
        "total_shield_time": tot
    }
    return out

# --------------------------
# 问题2
# --------------------------
def problem2_pso_optimize(
    heading_span=math.pi*2/3,
    uav_name="FY1",
    iter_num=100,
    pop_size=100
):
    base_heading = math.atan2((fake_target - FY_pos[uav_name])[1], (fake_target - FY_pos[uav_name])[0])
    # heading_min = base_heading - heading_span / 2
    # heading_max = base_heading + heading_span / 2
    heading_min = -math.pi
    heading_max = math.pi
    # speed_min, speed_max = uav_speed_bounds
    speed_min, speed_max = 110.0, 140.0
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0.0, 40.0

    dim = 4
    x_max = np.array([heading_max, speed_max, t_release_max, fuse_max])
    x_min = np.array([heading_min, speed_min, t_release_min, fuse_min])
    max_vel = (x_max - x_min) / 2
    
    def fitness(x):
        heading, speed, t_release, fuse_delay = x[0]
        v_uav = uav_velocity_from_heading(speed, heading)
        t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, t_release, fuse_delay)
        # 如果爆炸点高度小于0，直接返回极差适应度
        if expl_pos[2] < 0:
            return 1e6
        intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
        tot = total_time(intervals)
        return -tot

    from pso import PSO
    pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness)
    fit_var_list, best_pos = pso_solver.update_ndim()
    best_heading, best_speed, best_t_release, best_fuse = best_pos[0]
    v_uav = uav_velocity_from_heading(best_speed, best_heading)
    t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, best_t_release, best_fuse)
    intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
    tot = total_time(intervals)
    return {
        "heading": best_heading,
        "speed": best_speed,
        "t_release": best_t_release,
        "fuse_delay": best_fuse,
        "t_explosion": t_expl,
        "explosion_pos": expl_pos,
        "intervals": intervals,
        "total_shield_time": tot
    }

#问题2 使用差分进化算法
def problem2_de_fitness(x, uav_name):
    heading, speed, t_release, fuse_delay = x
    v_uav = uav_velocity_from_heading(speed, heading)
    t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, t_release, fuse_delay)
    if expl_pos[2] < 0:
        return 1e6
    intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
    tot = total_time(intervals)
    return -tot
def problem2_de_optimize(
    heading_span=math.pi*2/3,
    uav_name="FY1",
    maxiter=200,
    popsize=30
):
    base_heading = math.atan2((fake_target - FY_pos[uav_name])[1], (fake_target - FY_pos[uav_name])[0])
    heading_min = -math.pi
    heading_max = math.pi
    speed_min, speed_max = 110.0, 140.0
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0.0, 40.0

    bounds = [
        (heading_min, heading_max),
        (speed_min, speed_max),
        (t_release_min, t_release_max),
        (fuse_min, fuse_max)
    ]

    # 用partial包装参数，保证是全局函数
    fit_func = functools.partial(problem2_de_fitness, uav_name=uav_name)

    result = differential_evolution(
        fit_func,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        workers=-1,
        updating='deferred'
    )
    best_heading, best_speed, best_t_release, best_fuse = result.x
    v_uav = uav_velocity_from_heading(best_speed, best_heading)
    t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, best_t_release, best_fuse)
    intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
    tot = total_time(intervals)
    return {
        "heading": best_heading,
        "speed": best_speed,
        "t_release": best_t_release,
        "fuse_delay": best_fuse,
        "t_explosion": t_expl,
        "explosion_pos": expl_pos,
        "intervals": intervals,
        "total_shield_time": tot
    }

    
# --------------------------
# 问题3
# --------------------------
def problem3_pso_FY1_three(
    heading_span=math.pi/2,
    uav_name="FY1",
    iter_num=1000,  # increase for higher dim
    pop_size=50,   # increase for higher dim
    dt_refine=0.05  # sampling dt for joint shielding
):
    """
    Return best plan for FY1 to drop 3 munitions to maximize M1 shielding using PSO.
    """
    base_heading = math.atan2((fake_target - FY_pos[uav_name])[1], (fake_target - FY_pos[uav_name])[0])
    heading_min = -math.pi
    heading_max = math.pi
    speed_min, speed_max = uav_speed_bounds
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0.0, 40.0

    dim = 8  # heading, speed, t_release1, fuse1, t_release2, fuse2, t_release3, fuse3
    x_max = np.array([heading_max, speed_max, t_release_max, fuse_max, t_release_max, fuse_max, t_release_max, fuse_max])
    x_min = np.array([heading_min, speed_min, t_release_min, fuse_min, t_release_min, fuse_min, t_release_min, fuse_min])
    max_vel = (x_max - x_min) / 2
    
    # Joint shielding function
    def joint_shield_time(explosions, dt_sample=dt_refine, radius=cloud_radius):
        if not explosions:
            return 0.0
        min_t = min(t_expl for t_expl, _ in explosions)
        max_t = max(t_expl + cloud_duration for t_expl, _ in explosions)
        times = np.arange(min_t, max_t + dt_sample/2, dt_sample)
        if times.size == 0:
            return 0.0
        surface_points = sample_cylinder_surface(true_target, real_target_radius, real_target_height)
        flags = np.zeros(len(times), dtype=bool)
        for i, t in enumerate(times):
            m_pos = missile_pos(M1_pos0, t)
            active_clouds = [cloud_center_at(t, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= t < t_expl + cloud_duration]
            if not active_clouds:
                continue
            all_shielded = True
            for q in surface_points:
                shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                if not shielded:
                    all_shielded = False
                    break
            flags[i] = all_shielded
        # Compute rough intervals
        intervals = []
        in_seg = False
        start = None
        for i, t in enumerate(times):
            if flags[i] and not in_seg:
                in_seg = True
                start = t
            if not flags[i] and in_seg:
                in_seg = False
                intervals.append((start, times[i-1]))
        if in_seg:
            intervals.append((start, times[-1]))
        # Refine intervals with binary search
        refined = []
        for s, e in intervals:
            # Refine start
            left = max(min_t, s - dt_sample)
            right = s
            for _ in range(25):
                mid = 0.5 * (left + right)
                m_pos = missile_pos(M1_pos0, mid)
                active_clouds = [cloud_center_at(mid, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= mid < t_expl + cloud_duration]
                all_shielded = True
                for q in surface_points:
                    shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                    if not shielded:
                        all_shielded = False
                        break
                if all_shielded:
                    right = mid
                else:
                    left = mid
            s_ref = right
            # Refine end
            left = e
            right = min(max_t, e + dt_sample)
            for _ in range(25):
                mid = 0.5 * (left + right)
                m_pos = missile_pos(M1_pos0, mid)
                active_clouds = [cloud_center_at(mid, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= mid < t_expl + cloud_duration]
                all_shielded = True
                for q in surface_points:
                    shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                    if not shielded:
                        all_shielded = False
                        break
                if all_shielded:
                    left = mid
                else:
                    right = mid
            e_ref = left
            if e_ref > s_ref + 1e-9:
                refined.append((s_ref, e_ref))
        merged = merge_intervals(refined)
        return total_time(merged)
    
    def fitness(x):
        heading, speed, tr1, fu1, tr2, fu2, tr3, fu3 = x[0]
        # sort t_releases to enforce order and check min_drop_interval
        trs = sorted([tr1, tr2, tr3])
        fus = [fu1, fu2, fu3]  # assign to sorted trs
        if (trs[1] - trs[0] < min_drop_interval - 1e-9) or (trs[2] - trs[1] < min_drop_interval - 1e-9):
            return 1e6  # penalty
        v_uav = uav_velocity_from_heading(speed, heading)
        explosions = []
        for tr, fu in zip(trs, fus):
            t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, tr, fu)
            if expl_pos[2] < 0:
                return 1e6  # penalty
            explosions.append((t_expl, expl_pos))
        tot = joint_shield_time(explosions)
        return -tot  # maximize

    from pso import PSO
    #pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=2, C2=2, W=1.3)
    pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=2, C2=2, W=1.18)
    
    fit_var_list, best_pos = pso_solver.update_ndim()
    best_heading, best_speed, tr1, fu1, tr2, fu2, tr3, fu3 = best_pos[0]
    # reconstruct best plan with sorted t_releases
    trs = sorted([tr1, tr2, tr3])
    fus = [fu1, fu2, fu3]
    v_uav = uav_velocity_from_heading(best_speed, best_heading)
    assigned = []
    explosions = []
    for i, (tr, fu) in enumerate(zip(trs, fus), start=1):
        t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, tr, fu)
        assigned.append({
            "drop_idx": i,
            "t_release": tr,
            "fuse_delay": fu,
            "t_explosion": t_expl,
            "explosion_pos": expl_pos
        })
        explosions.append((t_expl, expl_pos))
    # Compute joint intervals for output (individual intervals not needed, but total)
    total = joint_shield_time(explosions)
    union_intervals = []  # For consistency, but since joint, we can compute intervals here if needed
    best_plan = {"heading": best_heading, "speed": best_speed, "assigned": assigned, "union_intervals": union_intervals, "total": total}
    # save Excel result1.xlsx (adjust for no individual intervals)
    rows = []
    for a in best_plan["assigned"]:
        rows.append({
            "drop_idx": a["drop_idx"],
            "uav": uav_name,
            "t_release": a["t_release"],
            "fuse_delay": a["fuse_delay"],
            "t_explosion": a["t_explosion"],
            "explosion_x": float(a["explosion_pos"][0]),
            "explosion_y": float(a["explosion_pos"][1]),
            "explosion_z": float(a["explosion_pos"][2]),
            "intervals": ""  # No individual intervals; joint computed separately
        })
    df = pd.DataFrame(rows)
    df_summary = pd.DataFrame([{"total_shield": best_plan["total"], "heading_deg": math.degrees(best_plan["heading"]), "speed": best_plan["speed"]}])
    with pd.ExcelWriter("result1.xlsx") as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
    return best_plan
# --------------------------
# Problem 4: FY1,FY2,FY3 each drop 1 munition to interfere M1
# PSO: optimize heading, speed, t_release, fuse_delay for FY1, FY2, FY3 in one PSO to maximize joint union shielding
# Constraints: respect physical bounds and joint shielding computation
# --------------------------
def problem4_pso_three_uavs_one_each(
    uav_names=("FY1", "FY2", "FY3"),
    heading_span=math.pi/2,
    iter_num=200,  # increased for 12D optimization
    pop_size=10000,   # increased for 12D optimization
    dt_refine=0.01
):
    """
    Return best plan for FY1, FY2, FY3 to each drop 1 munition to maximize M1 shielding using a single PSO.
    """
    # Define bounds for all 12 variables
    base_headings = [math.atan2((fake_target - FY_pos[uav])[1], (fake_target - FY_pos[uav])[0]) for uav in uav_names]
    heading_mins = -math.pi * np.ones(3)  # allow full circle headings
    heading_maxs = math.pi * np.ones(3)
    speed_min, speed_max = uav_speed_bounds
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0, 40.0  # avoid fuse=0

    dim = 12  # 4 vars per UAV: heading, speed, t_release, fuse_delay
    x_max = np.array([heading_maxs[0], speed_max, t_release_max, fuse_max,
                     heading_maxs[1], speed_max, t_release_max, fuse_max,
                     heading_maxs[2], speed_max, t_release_max, fuse_max])
    x_min = np.array([heading_mins[0], speed_min, t_release_min, fuse_min,
                     heading_mins[1], speed_min, t_release_min, fuse_min,
                     heading_mins[2], speed_min, t_release_min, fuse_min])
    max_vel = (x_max - x_min) / 2

    # Joint shielding function
    def joint_shield_time(explosions, dt_sample=dt_refine, radius=cloud_radius):
        if not explosions:
            return 0.0
        min_t = min(t_expl for t_expl, _ in explosions)
        max_t = max(t_expl + cloud_duration for t_expl, _ in explosions)
        times = np.arange(min_t, max_t + dt_sample/2, dt_sample)
        if times.size == 0:
            return 0.0
        surface_points = sample_cylinder_surface(true_target, real_target_radius, real_target_height)
        flags = np.zeros(len(times), dtype=bool)
        for i, t in enumerate(times):
            m_pos = missile_pos(M1_pos0, t)
            active_clouds = [cloud_center_at(t, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= t < t_expl + cloud_duration]
            if not active_clouds:
                continue
            all_shielded = True
            for q in surface_points:
                shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                if not shielded:
                    all_shielded = False
                    break
            flags[i] = all_shielded
        # Compute rough intervals
        intervals = []
        in_seg = False
        start = None
        for i, t in enumerate(times):
            if flags[i] and not in_seg:
                in_seg = True
                start = t
            if not flags[i] and in_seg:
                in_seg = False
                intervals.append((start, times[i-1]))
        if in_seg:
            intervals.append((start, times[-1]))
        # Refine intervals with binary search
        refined = []
        for s, e in intervals:
            left = max(min_t, s - dt_sample)
            right = s
            for _ in range(25):
                mid = 0.5 * (left + right)
                m_pos = missile_pos(M1_pos0, mid)
                active_clouds = [cloud_center_at(mid, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= mid < t_expl + cloud_duration]
                all_shielded = True
                for q in surface_points:
                    shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                    if not shielded:
                        all_shielded = False
                        break
                if all_shielded:
                    right = mid
                else:
                    left = mid
            s_ref = right
            left = e
            right = min(max_t, e + dt_sample)
            for _ in range(25):
                mid = 0.5 * (left + right)
                m_pos = missile_pos(M1_pos0, mid)
                active_clouds = [cloud_center_at(mid, expl_pos, t_expl) for t_expl, expl_pos in explosions if t_expl <= mid < t_expl + cloud_duration]
                all_shielded = True
                for q in surface_points:
                    shielded = any(segment_sphere_intersect_single(m_pos, q, c_pos, radius) for c_pos in active_clouds)
                    if not shielded:
                        all_shielded = False
                        break
                if all_shielded:
                    left = mid
                else:
                    right = mid
            e_ref = left
            if e_ref > s_ref + 1e-9:
                refined.append((s_ref, e_ref))
        merged = merge_intervals(refined)
        return total_time(merged)

    def fitness(x):
        # Extract parameters for all 3 UAVs
        h1, s1, tr1, fu1, h2, s2, tr2, fu2, h3, s3, tr3, fu3 = x[0]
        explosions = []
        for uav, h, s, tr, fu in zip(uav_names, [h1, h2, h3], [s1, s2, s3], [tr1, tr2, tr3], [fu1, fu2, fu3]):
            v_uav = uav_velocity_from_heading(s, h)
            t_expl, expl_pos = explosion_position(FY_pos[uav], v_uav, tr, fu)
            if expl_pos[2] < 0:
                return 1e6  # penalty for invalid explosion height
            explosions.append((t_expl, expl_pos))
        tot = joint_shield_time(explosions)
        return -tot  # maximize total shield time

    from pso import PSO
    pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=2, C2=2, W=0.5)
    fit_var_list, best_pos = pso_solver.update_ndim()
    # Reconstruct best plan
    h1, s1, tr1, fu1, h2, s2, tr2, fu2, h3, s3, tr3, fu3 = best_pos[0]
    assigned = []
    explosions = []
    for uav, h, s, tr, fu in zip(uav_names, [h1, h2, h3], [s1, s2, s3], [tr1, tr2, tr3], [fu1, fu2, fu3]):
        v_uav = uav_velocity_from_heading(s, h)
        t_expl, expl_pos = explosion_position(FY_pos[uav], v_uav, tr, fu)
        ints = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=dt_refine)
        assigned.append({
            "uav": uav, "heading": h, "speed": s, "t_release": tr, "fuse": fu,
            "t_explosion": t_expl, "explosion_pos": expl_pos, "intervals": ints
        })
        explosions.append((t_expl, expl_pos))
    total = joint_shield_time(explosions)
    union_all = merge_intervals([i for a in assigned for i in a["intervals"]])  # Recompute union for consistency

    # Save to result2.xlsx
    rows = []
    for a in assigned:
        rows.append({
            "uav": a["uav"], "t_release": a["t_release"], "fuse_delay": a["fuse"],
            "t_explosion": a["t_explosion"],
            "explosion_x": float(a["explosion_pos"][0]), "explosion_y": float(a["explosion_pos"][1]),
            "explosion_z": float(a["explosion_pos"][2]),
            "intervals": ";".join([f"{s:.6f}-{e:.6f}" for s, e in a["intervals"]]),
            "marginal": total_time(a["intervals"])  # Approx marginal for each drop
        })
    df = pd.DataFrame(rows)
    df_summary = pd.DataFrame([{"total_shield": total}])
    with pd.ExcelWriter("result2.xlsx") as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
    return {"assigned": assigned, "union": union_all}

# --------------------------
# Problem 5: 5 UAVs, each up to 3 munitions, interfere M1,M2,M3
# Greedy across all candidate explosions; respect:
# - Once a UAV's heading & speed are chosen (first assigned drop), they are locked for that UAV.
# - Each UAV up to 3 drops; min_drop_interval between drops on same UAV.
# Objective: maximize SUM of total shielding times across M1,M2,M3 (sum of per-missile union durations).
# --------------------------
def problem5_greedy(
    heading_span=math.pi*2/3, heading_steps=7, speed_steps=4,
    t_release_grid=None, fuse_grid=None, dt_coarse=0.1, dt_refine=0.02,
    max_rounds=15
):
    if t_release_grid is None:
        t_release_grid = np.linspace(0.0, 70.0, 71)  # 1s steps
    if fuse_grid is None:
        fuse_grid = np.linspace(0.5, 40.0, 40)

    uav_list = list(FY_pos.keys())
    missiles = {"M1": M1_pos0, "M2": M2_pos0, "M3": M3_pos0}
    missile_names = list(missiles.keys())

    # Pre-generate candidate explosions (coarse filter): only keep candidates that produce some coarse coverage
    print("[Problem5] Generating candidate explosions (coarse filter)...")
    candidates = []  # each candidate: dict with uav, heading, speed, t_release, fuse, t_explosion, explosion_pos, coarse_flags_per_missile
    for uav in uav_list:
        base_heading = math.atan2((fake_target - FY_pos[uav])[1], (fake_target - FY_pos[uav])[0])
        headings = np.linspace(base_heading - heading_span/2, base_heading + heading_span/2, heading_steps)
        speeds = np.linspace(uav_speed_bounds[0], uav_speed_bounds[1], speed_steps)
        for heading in headings:
            for speed in speeds:
                v = uav_velocity_from_heading(speed, heading)
                for tr in t_release_grid:
                    for fu in fuse_grid:
                        t_expl, expl_pos = explosion_position(FY_pos[uav], v, tr, fu)
                        # coarse check per missile (fast sampling dt_coarse)
                        times = np.arange(t_expl, t_expl + cloud_duration + dt_coarse*0.5, dt_coarse)
                        if times.size == 0:
                            continue
                        Ms_all = {m: np.array([missile_pos(missiles[m], t) for t in times]) for m in missile_names}
                        Cs = np.array([cloud_center_at(t, expl_pos, t_expl) for t in times])
                        coarse_flags = {}
                        any_hit = False
                        for m in missile_names:
                            flags = segment_sphere_intersect_times(Ms_all[m], true_target, Cs, cloud_radius)
                            if np.any(flags):
                                coarse_flags[m] = True
                                any_hit = True
                            else:
                                coarse_flags[m] = False
                        if any_hit:
                            candidates.append({
                                "uav": uav, "heading": heading, "speed": float(speed),
                                "t_release": float(tr), "fuse": float(fu), "t_explosion": float(t_expl),
                                "explosion_pos": expl_pos, "coarse_flags": coarse_flags
                            })
    print(f"[Problem5] Coarse candidates count: {len(candidates)}")

    # Greedy selection
    assigned = {uav: [] for uav in uav_list}  # assigned candidate dicts per uav
    locked_v = {uav: None for uav in uav_list}  # locked (heading,speed) tuple after first assignment
    per_uav_count = {uav: 0 for uav in uav_list}
    per_uav_release_times = {uav: [] for uav in uav_list}
    existing_intervals_per_missile = {m: [] for m in missile_names}

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        best_candidate = None
        best_marginal = 0.0
        # consider each candidate that doesn't violate locked_v constraint and per-uav capacity
        for cand in candidates:
            uav = cand["uav"]
            if per_uav_count[uav] >= max_per_uav_problem5:
                continue
            # if locked_v is set for this uav, candidate must match it (within small tol)
            if locked_v[uav] is not None:
                locked_heading, locked_speed = locked_v[uav]
                if abs(cand["heading"] - locked_heading) > 1e-6 or abs(cand["speed"] - locked_speed) > 1e-6:
                    continue
            # release time conflict with same-uav assigned times
            tr = cand["t_release"]
            bad = False
            for et in per_uav_release_times[uav]:
                if abs(et - tr) < min_drop_interval - 1e-9:
                    bad = True; break
            if bad: continue
            # refine candidate intervals per missile (dt_refine)
            intervals_per_m = {}
            for m in missile_names:
                if not cand["coarse_flags"].get(m, False):
                    intervals_per_m[m] = []
                else:
                    ints = find_shield_intervals_for_explosion(missiles[m], cand["explosion_pos"], cand["t_explosion"], dt_sample=dt_refine)
                    intervals_per_m[m] = ints
            # compute marginal gain in total sum shielding (sum across all missiles)
            new_totals = {}
            for m in missile_names:
                new_union = merge_intervals(existing_intervals_per_missile[m] + intervals_per_m[m])
                new_totals[m] = total_time(new_union)
            new_sum = sum(new_totals.values())
            old_sum = sum(total_time(existing_intervals_per_missile[m]) for m in missile_names)
            marginal = new_sum - old_sum
            if marginal > best_marginal + 1e-9:
                best_marginal = marginal
                best_candidate = {"cand": cand, "intervals_per_m": intervals_per_m, "marginal": marginal, "new_totals": new_totals}
        if best_candidate is None:
            print(f"[Problem5] Greedy stopped at round {rounds}, no positive marginal found.")
            break
        # accept best_candidate
        cand = best_candidate["cand"]
        uav = cand["uav"]
        assigned[uav].append(cand)
        per_uav_count[uav] += 1
        per_uav_release_times[uav].append(cand["t_release"])
        # lock v if first assignment
        if locked_v[uav] is None:
            locked_v[uav] = (cand["heading"], cand["speed"])
        # update existing intervals per missile
        for m in missile_names:
            existing_intervals_per_missile[m] = merge_intervals(existing_intervals_per_missile[m] + best_candidate["intervals_per_m"][m])
        cur_total_sum = sum(total_time(existing_intervals_per_missile[m]) for m in missile_names)
        print(f"[Problem5] Round {rounds}: pick {uav} tr={cand['t_release']}, fuse={cand['fuse']}, marginal={best_candidate['marginal']:.6f}, total_sum={cur_total_sum:.6f}")
    # save result3.xlsx
    rows = []
    for uav in uav_list:
        for i, a in enumerate(assigned[uav], start=1):
            # recompute refined intervals per missile for output
            intervals_per_m = {}
            for m in missile_names:
                ints = find_shield_intervals_for_explosion(missiles[m], a["explosion_pos"], a["t_explosion"], dt_sample=dt_refine)
                intervals_per_m[m] = ";".join([f"{s:.6f}-{e:.6f}" for s,e in ints]) if ints else ""
            rows.append({
                "uav": uav, "drop_idx": i, "t_release": a["t_release"], "fuse_delay": a["fuse"],
                "t_explosion": a["t_explosion"],
                "explosion_x": float(a["explosion_pos"][0]), "explosion_y": float(a["explosion_pos"][1]), "explosion_z": float(a["explosion_pos"][2]),
                "intervals_M1": intervals_per_m["M1"], "intervals_M2": intervals_per_m["M2"], "intervals_M3": intervals_per_m["M3"]
            })
    df = pd.DataFrame(rows)
    per_m = []
    for m in missile_names:
        per_m.append({"missile": m, "total_shield": total_time(existing_intervals_per_missile[m]), "intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in existing_intervals_per_missile[m]])})
    df2 = pd.DataFrame(per_m)
    with pd.ExcelWriter("result3.xlsx") as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df2.to_excel(writer, sheet_name="per_missile", index=False)
        pd.DataFrame([{"total_sum": sum(total_time(existing_intervals_per_missile[m]) for m in missile_names)}]).to_excel(writer, sheet_name="summary", index=False)

    return {"assigned": assigned, "per_missile_intervals": existing_intervals_per_missile}


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    # p1 = problem1()
    # print("问题1结果:")
    # print("  爆炸位置:", p1["explosion_pos"])
    # print("  爆炸时刻:", p1["t_explosion"])
    # print("  遮蔽时间区间 (M1):", p1["intervals"])
    # print("  总遮蔽时间 (M1):", p1["total_shield_time"])

    # print("\n")

    # p2_result = problem2_pso_optimize(
    #     heading_span=math.pi*2/3,
    #     uav_name="FY1",            # 无人机名称
    #     iter_num=200,              # PSO迭代次数
    #     pop_size=100                # PSO种群规模
    # )
    p2_result = problem2_de_optimize()
    if p2_result:
        print("Problem2 best candidate summary:")
        print("  speed:", p2_result["speed"], "heading_deg:", math.degrees(p2_result["heading"]))
        print("  t_release:", p2_result["t_release"], "fuse:", p2_result["fuse_delay"], "total_shield:", p2_result["total_shield_time"])
        # save plot
        t_expl = p2_result["t_explosion"]
        ts = np.linspace(t_expl, t_expl + cloud_duration, 600)
        missile_z = [missile_pos(M1_pos0, t)[2] for t in ts]
        cloud_z = [cloud_center_at(t, p2_result["explosion_pos"], t_expl)[2] for t in ts]
        plt.figure(figsize=(8,4))
        plt.plot(ts, missile_z, label="Missile Z")
        plt.plot(ts, cloud_z, label="Cloud Z")
        for s,e in p2_result["intervals"]:
            plt.axvspan(s, e, color='gray', alpha=0.4)
        plt.legend(); plt.xlabel("t (s)"); plt.ylabel("altitude (m)")
        plt.title("Problem2 best candidate (z vs t)")
        plt.tight_layout()
        plt.savefig("problem2_best.png")
        print("Saved problem2_best.png")
    else:
        print("Problem2 found no positive candidate.")
    print("\n")

    # result = problem3_pso_FY1_three()
    # print("Problem 3 completed. Results saved to result1.xlsx")
    # print(f"Best total shield time: {result['total']:.6f} s")
    # print(f"Heading: {math.degrees(result['heading']):.2f}°, Speed: {result['speed']:.2f} m/s")
    # for drop in result["assigned"]:
    #     print(f"Drop {drop['drop_idx']}: t_release={drop['t_release']:.2f}s, fuse_delay={drop['fuse_delay']:.2f}s")


    # result = problem4_pso_three_uavs_one_each()
    # print("Problem 4 completed. Results saved to result2.xlsx")
    # print(f"Total shield time: {result['union']:.6f} s")
    # for drop in result["assigned"]:
    #     print(f"UAV {drop['uav']}: t_release={drop['t_release']:.2f}s, fuse_delay={drop['fuse']:.2f}s, intervals={drop['intervals']}")
