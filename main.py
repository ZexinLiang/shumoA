import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.optimize import differential_evolution

#基本参数
g = 9.80  # 重力加速度
cloud_descent_rate = 3.0  # 云层沉降速度
cloud_radius = 10.0  # 云层半径
cloud_duration = 20.0  # 云层持续时长
missile_speed = 300.0  # 导弹速度
real_target_radius = 7.0  # 真实目标半径
real_target_height = 10.0  # 真实目标高度
uav_speed_bounds = (70.0, 140.0)# 无人机速度限制
min_drop_interval = 1.0  # 间隔最小1s
max_per_uav_problem5 = 3 # 每架无人机最多携带的云雾弹数量

#位置坐标参数
M1_pos0 = np.array([20000.0, 0.0, 2000.0])
M2_pos0 = np.array([19000.0, 600.0, 2100.0])
M3_pos0 = np.array([18000.0, -600.0, 1900.0])
#无人机初始位置
FY_pos = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0])
}
#目标物：真/假
fake_target = np.array([0.0, 0.0, 0.0])
true_target = np.array([0.0, 200.0, 0.0])  # 真目标下底心 (0,200,0)

#计算功能函数
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v.copy()
#计算导弹向量速度
def missile_velocity(missile_pos0: np.ndarray) -> np.ndarray:
    return unit(fake_target - missile_pos0) * missile_speed
#计导弹位置
def missile_pos(missile_pos0: np.ndarray, t: float) -> np.ndarray:
    return missile_pos0 + missile_velocity(missile_pos0) * t
#计算无人机速度向量
def uav_velocity_from_heading(speed: float, heading_rad: float) -> np.ndarray:
    return np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0]) * speed
#计算爆炸位置
def explosion_position(uav_pos0: np.ndarray, v_uav: np.ndarray, t_release: float, fuse_delay: float) -> Tuple[float, np.ndarray]:
    t_expl = float(t_release + fuse_delay)
    release_pos = uav_pos0 + v_uav * t_release
    dt = float(fuse_delay)
    disp = v_uav * dt + np.array([0.0, 0.0, -0.5 * g * dt * dt])
    return t_expl, release_pos + disp
#计算云层位置
def cloud_center_at(t: float, explosion_pos: np.ndarray, t_explosion: float) -> np.ndarray:
    return explosion_pos + np.array([0.0, 0.0, -cloud_descent_rate * (t - t_explosion)])

#--------------------------------------
#模型
#--------------------------------------

#判断单一线段与球体是否相交
def segment_sphere_intersect_single(P: np.ndarray, Q: np.ndarray, C: np.ndarray, R: float) -> bool:
    # print(f"Checking segment P={P} to Q={Q} against sphere C={C}, R={R}")
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
#判断多条线段与多个球体是否相交
def segment_sphere_intersect_times(Ps: np.ndarray, Q: np.ndarray, Cs: np.ndarray, R: float) -> np.ndarray:
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
#针对真实目标物，计算遮蔽时间区间，只使用与单一云层
#针对目标为圆柱体情况
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
        # print(refined)
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
#合并区间
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
#根据区间计算时间总长
def total_time(intervals: List[Tuple[float,float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)
#针对多个云层，计算联合遮蔽时间
def joint_shield_time(explosions, dt_sample=0.02, radius=cloud_radius):
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
def ang_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d) < 1e-9
#problems
#问题1
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
    print("问题1：有效遮蔽时长为：", tot,"秒\n")

#问题2
# #采用PSO
def problem2(uav_name="FY1",iter_num=120,pop_size=200):
    heading_min = -math.pi
    heading_max = math.pi
    speed_min, speed_max = 110.0, 140.0
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0.0, 40.0
    dim = 4
    x_max = np.array([heading_max, speed_max, t_release_max, fuse_max])
    x_min = np.array([heading_min, speed_min, t_release_min, fuse_min])
    max_vel = (x_max - x_min) / 10
    #优化函数
    def fitness(x):
        heading, speed, t_release, fuse_delay = x
        v_uav = uav_velocity_from_heading(speed, heading)
        t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, t_release, fuse_delay)
        # 如果爆炸点高度小于0，直接返回极差适应度
        if expl_pos[2] < 0:
            return 1e6
        intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.5)
        tot = total_time(intervals)
        return -tot
    from pso import PSO
    pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=2, C2=1.5, W=0.91)
    fit_var_list, best_pos = pso_solver.update_ndim()
    best_heading, best_speed, best_t_release, best_fuse = best_pos
    v_uav = uav_velocity_from_heading(best_speed, best_heading)
    release_pos = FY_pos[uav_name] + v_uav * best_t_release
    t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, best_t_release, best_fuse)
    intervals = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=0.01)
    tot = total_time(intervals)
    print("问题2：最佳方案为：")
    print(f"无人机：{uav_name}, 航向角（角度）：{best_heading/math.pi*180}°, 速度：{best_speed}m/s, 投放时间：{best_t_release}s, 引信延时：{best_fuse:.2f}s")
    print(f"投放位置：({release_pos[0]}, {release_pos[1]}, {release_pos[2]}), 引爆位置：({expl_pos[0]}, {expl_pos[1]}, {expl_pos[2]})")
    print(f"有效遮蔽时长为：{tot}秒\n")
#问题3
def problem3(uav_name="FY1",iter_num=10,pop_size=500,dt_refine=0.1):
    #第一问使用参数
    heading_min = 0
    heading_max = 2*math.pi
    speed_min, speed_max = 70,140
    t_release_min, t_release_max = 0, 60
    fuse_min, fuse_max =0,40
    dim = 8
    x_max = np.array([heading_max, speed_max, t_release_max, fuse_max, t_release_max, fuse_max, t_release_max, fuse_max])
    x_min = np.array([heading_min, speed_min, t_release_min, fuse_min, t_release_min, fuse_min, t_release_min, fuse_min])
    max_vel = (x_max - x_min) / 10
    #优化函数
    def fitness(x):
        heading, speed, tr1, fu1, tr2, fu2, tr3, fu3 = x
        trs = sorted([tr1, tr2, tr3])
        fus = [fu1, fu2, fu3]
        if (trs[1] - trs[0] < min_drop_interval - 1e-9) or (trs[2] - trs[1] < min_drop_interval - 1e-9):
            return 1e6
        v_uav = uav_velocity_from_heading(speed, heading)
        explosions = []
        for tr, fu in zip(trs, fus):
            t_expl, expl_pos = explosion_position(FY_pos[uav_name], v_uav, tr, fu)
            if expl_pos[2] < 0:
                return 1e6
            explosions.append((t_expl, expl_pos))
        tot = joint_shield_time(explosions)
        return -tot
    from pso import PSO
    #第一轮使用较大范围搜索
    #pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=2, C2=2, W=1.3)
    #第二轮使用较小范围精细搜索
    first_pos = [179.649/180*math.pi, 139.992, 0.006, 3.609, 3.658, 5.391, 5.577, 6.055]
    pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=1.5, C2=1.7, W=0.3,init_positions=[first_pos])
    # pso_solver = PSO(dim, pop_size, iter_num, x_max, x_min, max_vel, tol=-1e9, fitness=fitness, C1=1.5, C2=1.7, W=0.3)
    fit_var_list, best_pos = pso_solver.update_ndim()
    best_heading, best_speed, tr1, fu1, tr2, fu2, tr3, fu3 = best_pos
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
    print(best_plan)
#问题4
def problem4(
    uav_names=("FY1", "FY2", "FY3"),
    iter_num=20,
    pop_size=50,
    dt_refine=0.1,
    debug=True
):
    # ---- Bounds ----
    heading_mins = 0,0,0  # 允许全方位航向
    heading_maxs = 2*math.pi,2*math.pi,2*math.pi
    speed_min, speed_max = uav_speed_bounds
    t_release_min, t_release_max = 0.0, 70.0
    fuse_min, fuse_max = 0, 40.0  # 避免 0
    dim = 12  # 每个 UAV: heading, speed, t_release, fuse
    x_max = np.array([
        heading_maxs[0], speed_max, t_release_max, fuse_max,
        heading_maxs[1], speed_max, t_release_max, fuse_max,
        heading_maxs[2], speed_max, t_release_max, fuse_max
    ])
    x_min = np.array([
        heading_mins[0], 120, t_release_min, fuse_min,
        heading_mins[1], speed_min, t_release_min, fuse_min,
        heading_mins[2], speed_min, t_release_min, fuse_min
    ])
    max_vel = (x_max - x_min) / 20
    def wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    def intersect_intervals(A, B):
        if not A or not B:
            return []
        A_sorted = sorted(A, key=lambda x: x[0])
        B_sorted = sorted(B, key=lambda x: x[0])
        i, j = 0, 0
        out = []
        while i < len(A_sorted) and j < len(B_sorted):
            a1, a2 = A_sorted[i]
            b1, b2 = B_sorted[j]
            s = max(a1, b1)
            e = min(a2, b2)
            if e > s + 1e-12:
                out.append((s, e))
            if a2 < b2:
                i += 1
            else:
                j += 1
        return out
    first_call = {"done": False}
    def fitness(x):
        # normalize input to 1D vector
        x = np.ravel(x).astype(float)
        if not first_call["done"] and debug:
            print("[DEBUG] fitness called with shape:", x.shape)
            first_call["done"] = True

        if x.size != dim:
            return 1e6

        h1, s1, tr1, fu1, h2, s2, tr2, fu2, h3, s3, tr3, fu3 = x

        # wrap headings properly
        h1 = wrap_angle(h1)
        h2 = wrap_angle(h2)
        h3 = wrap_angle(h3)

        # bounds check
        for s, tr, fu in [(s1, tr1, fu1), (s2, tr2, fu2), (s3, tr3, fu3)]:
            if not (speed_min <= s <= speed_max and t_release_min <= tr <= t_release_max and fuse_min <= fu <= fuse_max):
                return 1e6

        explosions = []
        for uav, h, s, tr, fu in zip(
            uav_names, [h1, h2, h3], [s1, s2, s3], [tr1, tr2, tr3], [fu1, fu2, fu3]
        ):
            v_uav = uav_velocity_from_heading(s, h)
            t_expl, expl_pos = explosion_position(FY_pos[uav], v_uav, tr, fu)
            if expl_pos[2] < 0:
                return 1e6  # invalid explosion underground
            explosions.append((t_expl, expl_pos))

        # Use a coarser dt during PSO to speed up evaluation (but >= dt_refine)
        tot = joint_shield_time(explosions, dt_sample=max(0.05, dt_refine))

        # We minimize fitness; want to maximize tot -> return -tot
        return -tot
    first_value = [6.379/180*math.pi, 80.014-4, 0.437, 0.750,-39.999/180*math.pi, 138.944, 13.179, 2.336,80.120/180*math.pi, 139.113, 20.341, 2.155]
    # ---- Run PSO ----
    from pso import PSO
    pso_solver = PSO(
        dim, pop_size, iter_num,
        x_max, x_min, max_vel,
        tol=-1e9, fitness=fitness,
        C1=1.9, C2=1.2, W=0.9,
        init_positions=[first_value]
    )
    fit_var_list, best_pos = pso_solver.update_ndim()
    # ---- 解最优 ----
    best = np.ravel(best_pos).astype(float)
    if best.size != dim:
        raise RuntimeError(f"best_pos shape unexpected: {np.shape(best_pos)} -> ravel -> {best.size} (expected {dim})")
    h1, s1, tr1, fu1, h2, s2, tr2, fu2, h3, s3, tr3, fu3 = best
    # final wrap
    h1, h2, h3 = wrap_angle(h1), wrap_angle(h2), wrap_angle(h3)
    assigned, explosions = [], []
    for uav, h, s, tr, fu in zip(
        uav_names, [h1, h2, h3], [s1, s2, s3], [tr1, tr2, tr3], [fu1, fu2, fu3]
    ):
        v_uav = uav_velocity_from_heading(s, h)
        t_expl, expl_pos = explosion_position(FY_pos[uav], v_uav, tr, fu)
        ints = find_shield_intervals_for_explosion(M1_pos0, expl_pos, t_expl, dt_sample=dt_refine)
        assigned.append({
            "uav": uav, "heading": float(h), "speed": float(s),
            "t_release": float(tr), "fuse": float(fu),
            "t_explosion": float(t_expl), "explosion_pos": [float(expl_pos[0]), float(expl_pos[1]), float(expl_pos[2])],
            "intervals": [(float(a), float(b)) for a, b in ints]
        })
        explosions.append((float(t_expl), np.array(expl_pos, dtype=float)))

    # compute union of all intervals and final total shield time
    all_intervals = [iv for a in assigned for iv in a["intervals"]]
    union_all = merge_intervals(all_intervals)
    total = total_time(union_all)
    # engagement window (based on explosions)
    if explosions:
        engagement_start = min(t for t, _ in explosions)
        engagement_end = max(t + cloud_duration for t, _ in explosions)
    else:
        engagement_start, engagement_end = None, None
    engagement_duration = (engagement_end - engagement_start) if (engagement_start is not None and engagement_end is not None) else 0.0
    coverage_rate = total / engagement_duration if engagement_duration > 0 else 0.0
    per_rows = []
    for a in assigned:
        u = a["uav"]
        indiv_total = total_time(a["intervals"])
        # union without this UAV
        other_intervals = [iv for b in assigned if b["uav"] != u for iv in b["intervals"]]
        union_without = merge_intervals(other_intervals)
        total_without = total_time(union_without)
        marginal = max(0.0, total - total_without)  # net added shield time by this UAV
        overlap = max(0.0, indiv_total - marginal)  # amount of its own intervals overlapped by others
        per_rows.append({
            "uav": u,
            "heading": a["heading"],
            "speed": a["speed"],
            "t_release": a["t_release"],
            "fuse": a["fuse"],
            "t_explosion": a["t_explosion"],
            "explosion_x": a["explosion_pos"][0],
            "explosion_y": a["explosion_pos"][1],
            "explosion_z": a["explosion_pos"][2],
            "individual_shield_time": indiv_total,
            "marginal_contribution": marginal,
            "overlap_with_others": overlap
        })

    per_df = pd.DataFrame(per_rows)
    # pairwise overlaps
    pair_rows = []
    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            Ai = assigned[i]["intervals"]
            Aj = assigned[j]["intervals"]
            inter = intersect_intervals(Ai, Aj)
            overlap_time = total_time(inter)
            pair_rows.append({
                "uav_a": assigned[i]["uav"],
                "uav_b": assigned[j]["uav"],
                "overlap_time": overlap_time,
                "intersection_intervals": ";".join([f"{s:.6f}-{e:.6f}" for s, e in inter])
            })
    pair_df = pd.DataFrame(pair_rows)
    # all individual intervals table
    interval_rows = []
    for a in assigned:
        for s, e in a["intervals"]:
            interval_rows.append({
                "uav": a["uav"],
                "start": s,
                "end": e,
                "duration": e - s
            })
    intervals_df = pd.DataFrame(interval_rows)
    # union intervals df
    union_rows = []
    for s, e in union_all:
        union_rows.append({"start": s, "end": e, "duration": e - s})
    union_df = pd.DataFrame(union_rows)
    # gaps (non-shield intervals between union intervals) relative to engagement window
    gaps = []
    if union_all:
        for k in range(len(union_all) - 1):
            gaps.append((union_all[k][1], union_all[k + 1][0]))
    gaps_rows = [{"start": s, "end": e, "duration": (e - s)} for s, e in gaps]
    gaps_df = pd.DataFrame(gaps_rows)
    # explosion times df
    exp_rows = []
    for a in assigned:
        exp_rows.append({"uav": a["uav"], "t_explosion": a["t_explosion"]})
    exp_df = pd.DataFrame(exp_rows)
    # explosion distribution stats
    explosion_times = [a["t_explosion"] for a in assigned]
    if explosion_times:
        expl_mean = float(np.mean(explosion_times))
        expl_std = float(np.std(explosion_times))
        expl_min = float(np.min(explosion_times))
        expl_max = float(np.max(explosion_times))
    else:
        expl_mean = expl_std = expl_min = expl_max = 0.0
    # summary
    summary = {
        "total_shield_time": total,
        "engagement_start": engagement_start,
        "engagement_end": engagement_end,
        "engagement_duration": engagement_duration,
        "coverage_rate": coverage_rate,
        "num_uavs": len(assigned),
        "explosion_time_mean": expl_mean,
        "explosion_time_std": expl_std,
        "explosion_time_min": expl_min,
        "explosion_time_max": expl_max
    }
    if debug:
        print("== Problem4 result summary ==")
        print("Total shield time:", total)
        print("Engagement window:", engagement_start, "->", engagement_end, " duration:", engagement_duration)
        print("Coverage rate:", coverage_rate)
        print("Explosion times (mean, std, min, max):", expl_mean, expl_std, expl_min, expl_max)
        print("Per-UAV contributions:\n", per_df)
        print("Union intervals:\n", union_df)
        print("Gaps:\n", gaps_df)
        print("Pairwise overlaps:\n", pair_df)
    return {
        "assigned": assigned,
        "union": union_all,
        "total_shield_time": total,
        "engagement_window": (engagement_start, engagement_end),
        "coverage_rate": coverage_rate,
        "per_uav_df": per_df,
        "intervals_df": intervals_df,
        "union_df": union_df,
        "gaps_df": gaps_df,
        "pairwise_df": pair_df,
        "explosion_times": explosion_times
    }

# ---------------------- Problem 5 修正版 ----------------------
def problem5_greedy(
    heading_span=math.pi,
    heading_steps=15,
    speed_steps=8,
    t_release_grid=None,
    fuse_grid=None,
    dt_coarse=0.3,
    dt_refine=0.10,
    max_rounds=20,
    coarse_topk_per_uav=500
):
    if t_release_grid is None:
        t_release_grid = np.linspace(0.0, 70.0, 71)
    if fuse_grid is None:
        fuse_grid = np.linspace(0.5, 40.0, 40)

    uav_list = list(FY_pos.keys())
    missiles = {"M1": M1_pos0, "M2": M2_pos0, "M3": M3_pos0}
    missile_names = list(missiles.keys())

    candidates = []
    print("[Problem5] Generating candidate explosions (coarse filter)...")
    for uav in uav_list:
        base_heading = math.atan2((fake_target - FY_pos[uav])[1], (fake_target - FY_pos[uav])[0])
        h_half = heading_span / 2.0
        headings = np.linspace(0, 2*math.pi, heading_steps)
        try:
            speed_min_glob, speed_max_glob = uav_speed_bounds
        except Exception:
            speed_min_glob, speed_max_glob = 70.0, 140.0
        speeds = np.linspace(speed_min_glob, speed_max_glob, speed_steps)

        for heading in headings:
            for speed in speeds:
                v = uav_velocity_from_heading(speed, heading)
                for tr in t_release_grid:
                    for fu in fuse_grid:
                        t_expl, expl_pos = explosion_position(FY_pos[uav], v, tr, fu)
                        times = np.arange(t_expl, t_expl + cloud_duration + dt_coarse * 0.5, dt_coarse)
                        if times.size == 0:
                            continue
                        Ms_all = {m: np.array([missile_pos(missiles[m], tt) for tt in times]) for m in missile_names}
                        Cs = np.array([cloud_center_at(tt, expl_pos, t_expl) for tt in times])
                        coarse_flags = {}
                        any_hit = False
                        for m in missile_names:
                            try:
                                flags = segment_sphere_intersect_times(Ms_all[m], true_target, Cs, cloud_radius)
                                hit = np.any(flags)
                            except Exception:
                                hit = False
                                for idx_t, tt in enumerate(times):
                                    mpos = Ms_all[m][idx_t]
                                    cpos = Cs[idx_t]
                                    if 'segment_sphere_intersect_single' in globals():
                                        if segment_sphere_intersect_single(mpos, true_target, cpos, cloud_radius):
                                            hit = True
                                            break
                                    else:
                                        raise RuntimeError("[problem5_greedy] 没有可用的 segment_sphere_intersect 函数")
                            coarse_flags[m] = bool(hit)
                            if hit:
                                any_hit = True
                        if any_hit:
                            candidates.append({
                                "uav": uav, "heading": float(heading), "speed": float(speed),
                                "t_release": float(tr), "fuse": float(fu), "t_explosion": float(t_expl),
                                "explosion_pos": expl_pos, "coarse_flags": coarse_flags,
                                "picked": False, "_intervals_per_m": None
                            })
    print(f"[Problem5] Coarse candidates count: {len(candidates)}")

    if coarse_topk_per_uav is not None and len(candidates) > 0:
        new_cands = []
        for uav in uav_list:
            cands_uav = [c for c in candidates if c["uav"] == uav]
            cands_uav_sorted = sorted(cands_uav, key=lambda cc: sum(1 for v in cc["coarse_flags"].values() if v), reverse=True)
            new_cands.extend(cands_uav_sorted[:coarse_topk_per_uav])
        candidates = new_cands
        print(f"[Problem5] After top-K prune per-uav, candidates count: {len(candidates)}")

    assigned = {uav: [] for uav in uav_list}
    locked_v = {uav: None for uav in uav_list}
    per_uav_count = {uav: 0 for uav in uav_list}
    per_uav_release_times = {uav: [] for uav in uav_list}
    existing_intervals_per_missile = {m: [] for m in missile_names}

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        best_candidate_info = None
        best_marginal = 0.0

        for cand in candidates:
            if cand.get("picked", False):
                continue
            uav = cand["uav"]
            if per_uav_count[uav] >= max_per_uav_problem5:
                continue
            if locked_v[uav] is not None:
                locked_heading, locked_speed = locked_v[uav]
                if ang_diff(cand["heading"], locked_heading) > 1e-3 or abs(cand["speed"] - locked_speed) > 1e-6:
                    continue
            tr = cand["t_release"]
            conflict = any(abs(et - tr) < (min_drop_interval - 1e-9) for et in per_uav_release_times[uav])
            if conflict:
                continue

            if cand["_intervals_per_m"] is None:
                intervals_per_m = {}
                for m in missile_names:
                    if not cand["coarse_flags"].get(m, False):
                        intervals_per_m[m] = []
                    else:
                        ints = find_shield_intervals_for_explosion(missiles[m], cand["explosion_pos"], cand["t_explosion"], dt_sample=dt_refine)
                        intervals_per_m[m] = ints if ints is not None else []
                cand["_intervals_per_m"] = intervals_per_m
            else:
                intervals_per_m = cand["_intervals_per_m"]

            new_totals = {}
            old_sum = 0.0
            for m in missile_names:
                old = existing_intervals_per_missile[m]
                new_union = merge_intervals(old + intervals_per_m[m])
                new_totals[m] = total_time(new_union)
                old_sum += total_time(old)
            new_sum = sum(new_totals.values())
            marginal = new_sum - old_sum

            if marginal > best_marginal + 1e-12:
                best_marginal = marginal
                best_candidate_info = {"cand": cand, "intervals_per_m": intervals_per_m, "marginal": marginal, "new_totals": new_totals}

        if best_candidate_info is None:
            print(f"[Problem5] Greedy stopped at round {rounds}, no positive marginal found.")
            break

        cand = best_candidate_info["cand"]
        cand["picked"] = True
        uav = cand["uav"]
        assigned[uav].append(cand)
        per_uav_count[uav] += 1
        per_uav_release_times[uav].append(cand["t_release"])
        if locked_v[uav] is None:
            locked_v[uav] = (cand["heading"], cand["speed"])
        for m in missile_names:
            existing_intervals_per_missile[m] = merge_intervals(existing_intervals_per_missile[m] + best_candidate_info["intervals_per_m"][m])

        cur_total_sum = sum(total_time(existing_intervals_per_missile[m]) for m in missile_names)
        print(f"[Problem5] Round {rounds}: pick {uav} tr={cand['t_release']}, fuse={cand['fuse']}, marginal={best_candidate_info['marginal']:.6f}, total_sum={cur_total_sum:.6f}")

    rows = []
    for uav in uav_list:
        for i, a in enumerate(assigned[uav], start=1):
            intervals_per_m = a.get("_intervals_per_m")
            if intervals_per_m is None:
                intervals_per_m = {}
                for m in missile_names:
                    ints = find_shield_intervals_for_explosion(missiles[m], a["explosion_pos"], a["t_explosion"], dt_sample=dt_refine)
                    intervals_per_m[m] = ints if ints is not None else []
            rows.append({
                "uav": uav,
                "drop_idx": i,
                "t_release": a["t_release"],
                "fuse_delay": a["fuse"],
                "t_explosion": a["t_explosion"],
                "explosion_x": float(a["explosion_pos"][0]),
                "explosion_y": float(a["explosion_pos"][1]),
                "explosion_z": float(a["explosion_pos"][2]),
                "intervals_M1": ";".join([f"{s:.6f}-{e:.6f}" for s,e in intervals_per_m["M1"]]) if intervals_per_m.get("M1") else "",
                "intervals_M2": ";".join([f"{s:.6f}-{e:.6f}" for s,e in intervals_per_m["M2"]]) if intervals_per_m.get("M2") else "",
                "intervals_M3": ";".join([f"{s:.6f}-{e:.6f}" for s,e in intervals_per_m["M3"]]) if intervals_per_m.get("M3") else ""
            })
    df = pd.DataFrame(rows)

    per_m = []
    for m in missile_names:
        merged_intervals = merge_intervals(existing_intervals_per_missile[m])
        per_m.append({
            "missile": m,
            "total_shield": total_time(merged_intervals),
            "intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in merged_intervals])
        })
    df2 = pd.DataFrame(per_m)

    summary_val = sum(total_time(existing_intervals_per_missile[m]) for m in missile_names)
    with pd.ExcelWriter("result3.xlsx") as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df2.to_excel(writer, sheet_name="per_missile", index=False)
        pd.DataFrame([{ "total_sum": summary_val }]).to_excel(writer, sheet_name="summary", index=False)

    return {"assigned": assigned, "per_missile_intervals": existing_intervals_per_missile}

if __name__ == "__main__":
    #设定随机数种子
    np.random.seed(42)
    random.seed(42)
    # problem1()
    # problem2()
    # problem3()
    # problem4()
    problem5_greedy()