"""
smoke_strategy_all_problems.py

完整求解 A题（烟幕干扰弹的投放策略）问题 1-5 的 Python 脚本（单文件）。
依赖: numpy, pandas, math, itertools
保存输出为: result1.xlsx, result2.xlsx, result3.xlsx

说明:
- 将题目给定常量硬编码为 DEFAULTS 里的值（可改）。
- 物理模型:
  * 导弹：匀速直线朝假目标(0,0,0)
  * 无人机：等高度匀速直线（飞行方向、速度可设），释放弹丸后弹丸按抛体运动自由下落（不考虑空气阻力）
  * 弹丸起爆瞬间产生球形云团；云团以 3 m/s 匀速下降；有效时间 20 s，半径 10 m
  * 覆盖判定：当导弹中心与云团中心距离 <= 10 m 时认为被遮蔽
- 优化方法:
  * 问题2：对飞行方向角、速度、释放时间、引信延迟做粗网格搜索并局部细化（耗时与精度可调）
  * 问题3-5：使用贪心+网格搜索（每次选择能最大增加遮蔽时间的弹丸配置），满足题目约束（例如同机投放间隔 >= 1s，最多投放数等）
"""

import numpy as np
import math
import itertools
import pandas as pd
from typing import List, Tuple, Dict

# --------------------------
# DEFAULT PARAMETERS (from PDF)
# --------------------------
DEFAULTS = {
    "g": 9.80665,  # m/s^2
    "cloud_descent_rate": 3.0,  # m/s
    "cloud_radius": 10.0,  # m
    "cloud_effective_duration": 20.0,  # s after explosion
    "missile_speed": 300.0,  # m/s
    "uav_speed_default": 120.0,  # m/s (FY1 in Problem1)
    "uav_speed_bounds": (70.0, 140.0),  # allowed range
    "min_drop_interval": 1.0,  # s between drops from same UAV
    "task_receive_time": 0.0,  # we treat t=0 when radar detects (as problem statement)
    # initial positions (from PDF)
    "M1_pos0": np.array([20000.0, 0.0, 2000.0]),
    "M2_pos0": np.array([19000.0, 600.0, 2100.0]),
    "M3_pos0": np.array([18000.0, -600.0, 1900.0]),
    "FY1_pos0": np.array([17800.0, 0.0, 1800.0]),
    "FY2_pos0": np.array([12000.0, 1400.0, 1400.0]),
    "FY3_pos0": np.array([6000.0, -3000.0, 700.0]),
    "FY4_pos0": np.array([11000.0, 2000.0, 1800.0]),
    "FY5_pos0": np.array([13000.0, -2000.0, 1300.0]),
    "fake_target": np.array([0.0, 0.0, 0.0]),
    # default fuse delay (if not optimizing): set None to treat as variable
    "default_fuse_delay": None  # if None, we allow fuse as optimization var; if number, we fix it
}

# --------------------------
# Utility functions
# --------------------------
def unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec.copy()
    return vec / n

def missile_velocity_vector(missile_pos0: np.ndarray, fake_target: np.ndarray, missile_speed: float) -> np.ndarray:
    return unit(fake_target - missile_pos0) * missile_speed

def uav_velocity_vector(uav_pos0: np.ndarray, fake_target: np.ndarray, speed: float) -> np.ndarray:
    # UAV flies towards fake target direction projected in XY plane, constant altitude (z velocity = 0)
    dir_xy = unit((fake_target - uav_pos0)[:2])
    return np.array([dir_xy[0], dir_xy[1], 0.0]) * speed

# Given release time t_rel and fuse_delay (time from release to explosion), compute explosion time and position
def compute_explosion(uav_pos0: np.ndarray, v_uav: np.ndarray, t_release: float,
                      fuse_delay: float, g: float) -> Tuple[float, np.ndarray]:
    """
    Returns (t_explosion, explosion_position)
    explosion_position = position of munition at t = t_release + fuse_delay (ballistic: initial vel = v_uav)
    """
    t_explosion = t_release + fuse_delay
    # UAV position at release
    release_pos = uav_pos0 + v_uav * t_release
    dt = fuse_delay
    # projectile horizontal velocity = v_uav (z component zero)
    # vertical displacement under gravity: -0.5 * g * dt^2 (initial vertical vel 0)
    disp = v_uav * dt + np.array([0.0, 0.0, -0.5 * g * dt * dt])
    explosion_pos = release_pos + disp
    return t_explosion, explosion_pos

# Given missile motion and cloud center motion, compute intervals in [t_start, t_end] where distance <= radius
def shielding_intervals_for_pair(
    missile_pos0: np.ndarray, v_missile: np.ndarray,
    explosion_pos: np.ndarray, t_explosion: float,
    cloud_descent_rate: float, cloud_radius: float,
    cloud_effective_duration: float,
    t_window: Tuple[float, float] = None
) -> List[Tuple[float, float]]:
    """
    Returns list of (t_start, t_end) intervals (absolute times) within cloud_effective_window where missile is within cloud radius.
    Analytical solution: distance^2(t) = quadratic in t. Solve quadratic inequality.
    """
    # cloud center: c(t) = explosion_pos + k*(t - t_explosion), where k = (0,0,-cloud_descent_rate)
    k = np.array([0.0, 0.0, -cloud_descent_rate])
    # Let's represent c(t) = c0 + k*t with c0 = explosion_pos - k * t_explosion
    c0 = explosion_pos - k * t_explosion
    # missile m(t) = missile_pos0 + v_missile * t
    # difference d(t) = m(t) - c(t) = (v_missile - k) * t + (missile_pos0 - c0)
    a_vec = v_missile - k
    b_vec = missile_pos0 - c0
    A = float(np.dot(a_vec, a_vec))
    B = 2.0 * float(np.dot(a_vec, b_vec))
    C = float(np.dot(b_vec, b_vec) - cloud_radius**2)
    t_effect_start = t_explosion
    t_effect_end = t_explosion + cloud_effective_duration
    if t_window:
        # intersect with user-specified time window (useful if we want global clamp)
        t_effect_start = max(t_effect_start, t_window[0])
        t_effect_end = min(t_effect_end, t_window[1])
    intervals = []
    eps = 1e-12
    if t_effect_start >= t_effect_end:
        return []
    if abs(A) < eps:
        # linear or constant
        if abs(B) < eps:
            # constant C <= 0 ? then always; else never
            if C <= 0:
                intervals.append((t_effect_start, t_effect_end))
        else:
            t_root = -C / B
            if B > 0:
                # inequality <=0 holds for t <= t_root
                t0 = max(-1e9, t_effect_start)
                t1 = min(t_root, t_effect_end)
                if t0 < t1:
                    intervals.append((t0, t1))
            else:
                # B < 0 -> holds for t >= t_root
                t0 = max(t_root, t_effect_start)
                t1 = min(1e9, t_effect_end)
                if t0 < t1:
                    intervals.append((t0, t1))
    else:
        disc = B*B - 4*A*C
        if disc < 0:
            # check single point
            t_mid = 0.5*(t_effect_start + t_effect_end)
            if A*t_mid*t_mid + B*t_mid + C <= 0:
                intervals.append((t_effect_start, t_effect_end))
        else:
            sqrt_disc = math.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2*A)
            t2 = (-B + sqrt_disc) / (2*A)
            low = min(t1, t2)
            high = max(t1, t2)
            if A > 0:
                # inside [low, high]
                t0 = max(low, t_effect_start)
                t1 = min(high, t_effect_end)
                if t0 < t1:
                    intervals.append((t0, t1))
            else:
                # outside [low, high], two pieces
                # piece 1
                t0 = max(t_effect_start, -1e9)
                t1 = min(low, t_effect_end)
                if t0 < t1:
                    intervals.append((t0, t1))
                # piece 2
                t0 = max(high, t_effect_start)
                t1 = min(1e9, t_effect_end)
                if t0 < t1:
                    intervals.append((t0, t1))
    # merge and clamp
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = []
    for s,e in intervals_sorted:
        s_clamped = max(s, t_effect_start)
        e_clamped = min(e, t_effect_end)
        if s_clamped < e_clamped:
            if not merged:
                merged.append([s_clamped, e_clamped])
            else:
                if s_clamped <= merged[-1][1] + 1e-9:
                    merged[-1][1] = max(merged[-1][1], e_clamped)
                else:
                    merged.append([s_clamped, e_clamped])
    return [(s,e) for s,e in merged]

def total_shield_time_from_intervals(intervals: List[Tuple[float,float]]) -> float:
    return sum(max(0.0, e - s) for s,e in intervals)

# --------------------------
# Problem 1: direct computation (as in your earlier run)
# --------------------------
def problem_1_compute(defaults=DEFAULTS, verbose=True):
    g = defaults["g"]
    cloud_descent_rate = defaults["cloud_descent_rate"]
    cloud_radius = defaults["cloud_radius"]
    cloud_effective_duration = defaults["cloud_effective_duration"]
    missile_speed = defaults["missile_speed"]
    uav_speed = defaults["uav_speed_default"]
    t_release = 1.5
    fuse_delay = 3.6

    M1 = defaults["M1_pos0"]
    FY1 = defaults["FY1_pos0"]
    fake = defaults["fake_target"]

    v_missile = missile_velocity_vector(M1, fake, missile_speed)
    v_uav = uav_velocity_vector(FY1, fake, uav_speed)

    t_explosion, explosion_pos = compute_explosion(FY1, v_uav, t_release, fuse_delay, g)

    intervals = shielding_intervals_for_pair(
        missile_pos0=M1, v_missile=v_missile,
        explosion_pos=explosion_pos, t_explosion=t_explosion,
        cloud_descent_rate=cloud_descent_rate, cloud_radius=cloud_radius,
        cloud_effective_duration=cloud_effective_duration
    )
    total = total_shield_time_from_intervals(intervals)
    if verbose:
        print("=== Problem 1 result ===")
        print(f"t_release = {t_release}, fuse_delay = {fuse_delay}, t_explosion = {t_explosion}")
        print(f"explosion_pos = {explosion_pos}")
        if intervals:
            for s,e in intervals:
                print(f"shield interval: {s:.6f} -> {e:.6f}, duration = {e-s:.6f} s")
        else:
            print("no shielding intervals")
        print(f"Total shielding time = {total:.6f} s")
    return {
        "t_release": t_release,
        "fuse_delay": fuse_delay,
        "t_explosion": t_explosion,
        "explosion_pos": explosion_pos,
        "intervals": intervals,
        "total": total,
        "v_missile": v_missile,
        "v_uav": v_uav
    }

# --------------------------
# Problem 2: optimize FY1 flight direction, speed, release time, fuse_delay to maximize shielding time for M1
# Approach: grid search then local refine.
# Variables:
#  - uav_speed in [70,140]
#  - heading_angle (in XY) - we param by unit heading toward fake target rotated by delta angle in [-pi/2, pi/2]
#    but to be general allow full circle; but search around direct-to-target direction is sufficient.
#  - t_release in [0, T_MAX] (T_MAX chosen reasonably, e.g., 80s)
#  - fuse_delay in [0.5, 40] (but cloud effective window limited to fuse+20)
# We'll do coarse grid then refine around top candidates
# --------------------------
def problem_2_optimize(defaults=DEFAULTS, coarse_steps=8, t_release_grid=None, fuse_grid=None, verbose=True):
    print("Starting Problem 2 optimization (coarse grid). This may take some seconds depending on grid sizes.")
    g = defaults["g"]
    cloud_descent_rate = defaults["cloud_descent_rate"]
    cloud_radius = defaults["cloud_radius"]
    cloud_effective_duration = defaults["cloud_effective_duration"]
    missile_speed = defaults["missile_speed"]
    M1 = defaults["M1_pos0"]
    FY1 = defaults["FY1_pos0"]
    fake = defaults["fake_target"]

    if t_release_grid is None:
        # reasonable window: between 0 and 70s (missile ~66s), step coarse
        t_release_grid = np.linspace(0.0, 70.0, 15)
    if fuse_grid is None:
        fuse_grid = np.linspace(0.5, 40.0, 16)

    # param grids
    speed_min, speed_max = defaults["uav_speed_bounds"]
    speeds = np.linspace(speed_min, speed_max, coarse_steps)
    # heading: param as angle offset relative to direct-to-target direction projected in XY
    direct_dir_xy = unit((fake - FY1)[:2])
    base_angle = math.atan2(direct_dir_xy[1], direct_dir_xy[0])
    angle_offsets = np.linspace(-math.pi/2, math.pi/2, coarse_steps)  # search ±90deg
    angles = base_angle + angle_offsets

    v_missile = missile_velocity_vector(M1, fake, missile_speed)

    best = {"total": 0.0}
    # coarse search
    for speed in speeds:
        for angle in angles:
            # build v_uav from angle (ensure z=0)
            v_uav = np.array([math.cos(angle), math.sin(angle), 0.0]) * speed
            for t_release in t_release_grid:
                # ensure release_pos not too far? we allow any t_release
                for fuse_delay in fuse_grid:
                    t_explosion, explosion_pos = compute_explosion(FY1, v_uav, t_release, fuse_delay, g)
                    intervals = shielding_intervals_for_pair(M1, v_missile, explosion_pos, t_explosion,
                                                            cloud_descent_rate, cloud_radius, cloud_effective_duration)
                    total = total_shield_time_from_intervals(intervals)
                    if total > best["total"]:
                        best = {
                            "total": total,
                            "speed": float(speed),
                            "angle": float(angle),
                            "t_release": float(t_release),
                            "fuse_delay": float(fuse_delay),
                            "v_uav": v_uav,
                            "explosion_pos": explosion_pos,
                            "intervals": intervals
                        }
    # local refine around best (finer grid)
    if best["total"] > 0:
        print("Coarse best total shielding:", best["total"])
        # refine near best speed/angle/t_release/fuse
        speed_vals = np.linspace(max(speed_min, best["speed"] - 20), min(speed_max, best["speed"] + 20), 11)
        angle_vals = np.linspace(best["angle"] - 0.3, best["angle"] + 0.3, 11)
        t_release_vals = np.linspace(max(0.0, best["t_release"] - 5.0), best["t_release"] + 5.0, 21)
        fuse_vals = np.linspace(max(0.1, best["fuse_delay"] - 5.0), best["fuse_delay"] + 5.0, 21)
        best2 = best.copy()
        for speed in speed_vals:
            for angle in angle_vals:
                v_uav = np.array([math.cos(angle), math.sin(angle), 0.0]) * speed
                for t_release in t_release_vals:
                    for fuse_delay in fuse_vals:
                        t_explosion, explosion_pos = compute_explosion(FY1, v_uav, t_release, fuse_delay, g)
                        intervals = shielding_intervals_for_pair(M1, v_missile, explosion_pos, t_explosion,
                                                                cloud_descent_rate, cloud_radius, cloud_effective_duration)
                        total = total_shield_time_from_intervals(intervals)
                        if total > best2["total"]:
                            best2 = {
                                "total": total,
                                "speed": float(speed),
                                "angle": float(angle),
                                "t_release": float(t_release),
                                "fuse_delay": float(fuse_delay),
                                "v_uav": v_uav,
                                "explosion_pos": explosion_pos,
                                "intervals": intervals
                            }
        best = best2

    if verbose:
        print("=== Problem 2 result (approx) ===")
        if best["total"] <= 0:
            print("No shielding achieved with search ranges.")
        else:
            print(f"Total shielding time (approx): {best['total']:.6f} s")
            print(f"UAV speed: {best['speed']:.3f} m/s")
            ang_deg = math.degrees(best['angle'])
            print(f"UAV heading angle (abs): {ang_deg:.3f} deg (0 deg = +x axis)")
            print(f"t_release: {best['t_release']:.3f} s, fuse_delay: {best['fuse_delay']:.3f} s")
            print(f"explosion_pos: {best['explosion_pos']}")
            for s,e in best['intervals']:
                print(f"  shielding interval: {s:.6f} -> {e:.6f}  duration {e-s:.6f}")
    return best

# --------------------------
# Multi-munition planners for Problems 3-5
# Strategy:
#  - We'll use a greedy incremental approach:
#    For each new munition to assign, search over candidate params (t_release, fuse_delay) for given UAV and target missile
#    and pick the one that gives maximum marginal increase in total shielding against the target set (for problem5).
#  - Candidate parameter grid can be adjusted for speed/angle if we allow changing UAV speed/direction (here FY1 is fixed in problems 3/4).
#  - For constraints: respect min_drop_interval for same-UAV drops and max per-UAV counts.
# This is heuristic but reasonable and deterministic.
# --------------------------
def generate_candidate_params(uav_pos0: np.ndarray, fake: np.ndarray, uav_speed: float,
                              t_release_grid: np.ndarray, fuse_grid: np.ndarray):
    # produce tuples (t_release, fuse_delay) and v_uav computed from default heading-to-target
    v_uav = uav_velocity_vector(uav_pos0, fake, uav_speed)
    for t_rel in t_release_grid:
        for fuse in fuse_grid:
            yield (t_rel, fuse, v_uav)

def plan_three_munitions_single_uav(FY_pos0: np.ndarray, defaults=DEFAULTS, uav_speed=None,
                                    n_munitions=3, tgrid=None, fusegrid=None, verbose=True):
    """
    Problem 3: FY1 投 3 枚 用于干扰 M1。
    Returns list of drops with fields: t_release, fuse_delay, explosion_pos, intervals, marginal_gain
    """
    if uav_speed is None:
        uav_speed = defaults["uav_speed_default"]
    if tgrid is None:
        tgrid = np.linspace(0.0, 70.0, 71)  # 1s step
    if fusegrid is None:
        fusegrid = np.linspace(0.5, 40.0, 41)  # 1s step
    M1 = defaults["M1_pos0"]
    v_missile = missile_velocity_vector(M1, defaults["fake_target"], defaults["missile_speed"])
    assigned = []  # each entry dict
    occupied_release_times = []  # to ensure min interval
    current_total = 0.0
    for m in range(n_munitions):
        best_local = None
        for t_rel in tgrid:
            # enforce min interval with existing assigned releases
            ok = True
            for tr in occupied_release_times:
                if abs(tr - t_rel) < defaults["min_drop_interval"] - 1e-9:
                    ok = False
                    break
            if not ok:
                continue
            for fuse in fusegrid:
                t_explosion, explosion_pos = compute_explosion(FY_pos0, uav_velocity_vector(FY_pos0, defaults["fake_target"], uav_speed),
                                                                t_rel, fuse, defaults["g"])
                intervals = shielding_intervals_for_pair(M1, v_missile, explosion_pos, t_explosion,
                                                        defaults["cloud_descent_rate"], defaults["cloud_radius"],
                                                        defaults["cloud_effective_duration"])
                gain = total_shield_time_from_intervals(intervals)
                # marginal gain relative to current set: however overlaps reduce marginal; compute union
                # compute union of existing intervals and candidate, to find new total
                existing = []
                for a in assigned:
                    existing.extend(a["intervals"])
                # merge existing
                union = merge_intervals(existing)
                new_union = merge_intervals(union + intervals)
                new_total = total_shield_time_from_intervals(new_union)
                marginal = new_total - current_total
                if best_local is None or marginal > best_local["marginal"]:
                    best_local = {
                        "t_release": float(t_rel),
                        "fuse_delay": float(fuse),
                        "t_explosion": t_explosion,
                        "explosion_pos": explosion_pos,
                        "intervals": intervals,
                        "marginal": marginal,
                        "new_total": new_total
                    }
        if best_local is None:
            break
        # accept best_local
        assigned.append(best_local)
        occupied_release_times.append(best_local["t_release"])
        current_total = best_local["new_total"]
        if verbose:
            print(f"Assigned munition {m+1}: t_release={best_local['t_release']}, fuse={best_local['fuse_delay']}, marginal gain={best_local['marginal']:.6f}, new total={best_local['new_total']:.6f}")
    # return assigned and union intervals
    all_intervals = []
    for a in assigned:
        all_intervals.extend(a["intervals"])
    merged = merge_intervals(all_intervals)
    return {
        "assigned": assigned,
        "union_intervals": merged,
        "total_shield": total_shield_time_from_intervals(merged)
    }

def merge_intervals(intervals: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = ints[0]
    for s,e in ints[1:]:
        if s <= cur_e + 1e-9:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s,e
    merged.append((cur_s, cur_e))
    return merged

# Problem 4: three UAVs each drop 1 munition (FY1,FY2,FY3)
def plan_three_uavs_one_each(defaults=DEFAULTS, tgrid=None, fusegrid=None, verbose=True):
    if tgrid is None:
        tgrid = np.linspace(0.0, 70.0, 71)
    if fusegrid is None:
        fusegrid = np.linspace(0.5, 40.0, 41)
    M1 = defaults["M1_pos0"]
    v_missile = missile_velocity_vector(M1, defaults["fake_target"], defaults["missile_speed"])
    uavs = [("FY1", defaults["FY1_pos0"]), ("FY2", defaults["FY2_pos0"]), ("FY3", defaults["FY3_pos0"])]
    assigned = []
    current_total = 0.0
    existing_intervals = []
    for name, pos in uavs:
        best_local = None
        for t_rel in tgrid:
            for fuse in fusegrid:
                t_explosion, explosion_pos = compute_explosion(pos, uav_velocity_vector(pos, defaults["fake_target"], defaults["uav_speed_default"]),
                                                               t_rel, fuse, defaults["g"])
                intervals = shielding_intervals_for_pair(M1, v_missile, explosion_pos, t_explosion,
                                                        defaults["cloud_descent_rate"], defaults["cloud_radius"],
                                                        defaults["cloud_effective_duration"])
                new_union = merge_intervals(existing_intervals + intervals)
                new_total = total_shield_time_from_intervals(new_union)
                marginal = new_total - current_total
                if best_local is None or marginal > best_local["marginal"]:
                    best_local = {
                        "uav": name,
                        "pos": pos,
                        "t_release": float(t_rel),
                        "fuse_delay": float(fuse),
                        "t_explosion": t_explosion,
                        "explosion_pos": explosion_pos,
                        "intervals": intervals,
                        "marginal": marginal,
                        "new_total": new_total
                    }
        if best_local is None:
            continue
        assigned.append(best_local)
        existing_intervals = merge_intervals(existing_intervals + best_local["intervals"])
        current_total = total_shield_time_from_intervals(existing_intervals)
        if verbose:
            print(f"Assigned {best_local['uav']} drop: t_release={best_local['t_release']}, fuse={best_local['fuse_delay']}, marginal gain={best_local['marginal']:.6f}")
    return {
        "assigned": assigned,
        "union_intervals": existing_intervals,
        "total_shield": total_shield_time_from_intervals(existing_intervals)
    }

# Problem 5: up to 5 UAVs each up to 3 munitions, interfere M1,M2,M3
# We'll do greedy: at each step consider all possible (uav, t_release, fuse, target_missile) candidates that satisfy per-uav capacity and min interval; pick candidate with largest marginal increase in sum shielding across M1,M2,M3.
def plan_multi_uavs_multi_missiles(defaults=DEFAULTS, uav_list=None, max_per_uav=3, total_rounds=15,
                                   tgrid=None, fusegrid=None, verbose=True):
    if uav_list is None:
        uav_list = [
            ("FY1", defaults["FY1_pos0"]),
            ("FY2", defaults["FY2_pos0"]),
            ("FY3", defaults["FY3_pos0"]),
            ("FY4", defaults["FY4_pos0"]),
            ("FY5", defaults["FY5_pos0"])
        ]
    if tgrid is None:
        tgrid = np.linspace(0.0, 70.0, 71)  # 1s steps
    if fusegrid is None:
        fusegrid = np.linspace(0.5, 40.0, 41)
    missiles = [("M1", defaults["M1_pos0"]), ("M2", defaults["M2_pos0"]), ("M3", defaults["M3_pos0"])]
    missile_vs = {name: missile_velocity_vector(pos, defaults["fake_target"], defaults["missile_speed"]) for name,pos in missiles}
    # state
    assigned = {uav_name: [] for uav_name,_ in uav_list}  # list of dicts per uav
    per_uav_counts = {uav_name:0 for uav_name,_ in uav_list}
    existing_intervals_per_missile = {name: [] for name,_ in missiles}
    current_total = 0.0
    # greedy rounds
    rounds = 0
    while rounds < total_rounds:
        rounds += 1
        best_candidate = None
        # iterate candidates: for each UAV with capacity, for each missile, for each t_rel/fuse try
        for uav_name, uav_pos in uav_list:
            if per_uav_counts[uav_name] >= max_per_uav:
                continue
            # ensure release times not too close to existing ones on same uav
            existing_times = [a["t_release"] for a in assigned[uav_name]]
            for t_rel in tgrid:
                # check min interval
                ok = True
                for et in existing_times:
                    if abs(et - t_rel) < defaults["min_drop_interval"] - 1e-9:
                        ok = False
                        break
                if not ok:
                    continue
                for fuse in fusegrid:
                    t_explosion, explosion_pos = compute_explosion(uav_pos, uav_velocity_vector(uav_pos, defaults["fake_target"], defaults["uav_speed_default"]),
                                                                   t_rel, fuse, defaults["g"])
                    for missile_name, missile_pos in missiles:
                        v_missile = missile_vs[missile_name]
                        intervals = shielding_intervals_for_pair(missile_pos, v_missile, explosion_pos, t_explosion,
                                                                defaults["cloud_descent_rate"], defaults["cloud_radius"],
                                                                defaults["cloud_effective_duration"])
                        # compute marginal gain across all missiles if we add this candidate by merging intervals
                        new_totals = {}
                        marginal_sum = 0.0
                        for m_name, m_pos in missiles:
                            existing = existing_intervals_per_missile[m_name].copy()
                            if m_name == missile_name:
                                new_union = merge_intervals(existing + intervals)
                            else:
                                new_union = merge_intervals(existing)
                            new_totals[m_name] = total_shield_time_from_intervals(new_union)
                        new_sum = sum(new_totals.values())
                        marginal = new_sum - current_total
                        if best_candidate is None or marginal > best_candidate["marginal"]:
                            best_candidate = {
                                "uav_name": uav_name,
                                "uav_pos": uav_pos,
                                "t_release": float(t_rel),
                                "fuse_delay": float(fuse),
                                "t_explosion": t_explosion,
                                "explosion_pos": explosion_pos,
                                "target_missile": missile_name,
                                "intervals": intervals,
                                "marginal": marginal,
                                "new_totals": new_totals
                            }
        if best_candidate is None or best_candidate["marginal"] <= 1e-9:
            # no improvement possible
            if verbose:
                print("No candidate gives positive marginal gain; stopping greedy allocation.")
            break
        # accept best candidate
        uavn = best_candidate["uav_name"]
        assigned[uavn].append(best_candidate)
        per_uav_counts[uavn] += 1
        # update existing_intervals_per_missile for the target missile
        tgt = best_candidate["target_missile"]
        existing_intervals_per_missile[tgt] = merge_intervals(existing_intervals_per_missile[tgt] + best_candidate["intervals"])
        # update current total sum
        current_total = sum(total_shield_time_from_intervals(existing_intervals_per_missile[m]) for m,_ in missiles)
        if verbose:
            print(f"Round {rounds}: assigned to {uavn} targeting {tgt} at t_rel={best_candidate['t_release']}, fuse={best_candidate['fuse_delay']}, marginal sum gain={best_candidate['marginal']:.6f}, total_sum={current_total:.6f}")
    # finalize: prepare summary
    summary = {"assigned": assigned, "existing_intervals_per_missile": existing_intervals_per_missile, "total_sum": current_total}
    return summary

# --------------------------
# Helpers to export results to Excel
# --------------------------
def save_result1_excel(plan3_result, filename="result1.xlsx"):
    # plan3_result: result from plan_three_munitions_single_uav
    rows = []
    for i,a in enumerate(plan3_result["assigned"], start=1):
        rows.append({
            "muniton_idx": i,
            "uav": "FY1",
            "t_release": a["t_release"],
            "fuse_delay": a["fuse_delay"],
            "t_explosion": a["t_explosion"],
            "explosion_x": a["explosion_pos"][0],
            "explosion_y": a["explosion_pos"][1],
            "explosion_z": a["explosion_pos"][2],
            "shield_intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in a["intervals"]]),
            "marginal_gain": a["marginal"]
        })
    df = pd.DataFrame(rows)
    df_summary = pd.DataFrame([{"total_shield": plan3_result["total_shield"]}])
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

def save_result2_excel(plan4_result, filename="result2.xlsx"):
    rows = []
    for a in plan4_result["assigned"]:
        rows.append({
            "uav": a["uav"],
            "t_release": a["t_release"],
            "fuse_delay": a["fuse_delay"],
            "t_explosion": a["t_explosion"],
            "explosion_x": a["explosion_pos"][0],
            "explosion_y": a["explosion_pos"][1],
            "explosion_z": a["explosion_pos"][2],
            "shield_intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in a["intervals"]]),
            "marginal_gain": a["marginal"]
        })
    df = pd.DataFrame(rows)
    df_summary = pd.DataFrame([{"total_shield": plan4_result["total_shield"]}])
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

def save_result3_excel(plan5_result, filename="result3.xlsx"):
    rows = []
    for uav, items in plan5_result["assigned"].items():
        for idx,a in enumerate(items, start=1):
            rows.append({
                "uav": uav,
                "idx": idx,
                "target_missile": a["target_missile"],
                "t_release": a["t_release"],
                "fuse_delay": a["fuse_delay"],
                "t_explosion": a["t_explosion"],
                "explosion_x": a["explosion_pos"][0],
                "explosion_y": a["explosion_pos"][1],
                "explosion_z": a["explosion_pos"][2],
                "shield_intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in a["intervals"]]),
                "marginal": a["marginal"]
            })
    df = pd.DataFrame(rows)
    # also save per-missile summaries
    per_miss = []
    for m,intervals in plan5_result["existing_intervals_per_missile"].items():
        per_miss.append({"missile": m, "total_shield": total_shield_time_from_intervals(intervals),
                         "intervals": ";".join([f"{s:.6f}-{e:.6f}" for s,e in intervals])})
    df2 = pd.DataFrame(per_miss)
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="drops", index=False)
        df2.to_excel(writer, sheet_name="per_missile", index=False)
        pd.DataFrame([{"total_sum": plan5_result["total_sum"]}]).to_excel(writer, sheet_name="summary", index=False)

# --------------------------
# Main: run problems 1-5, save excel files
# --------------------------
def main():
    print("Running all problems (1-5). This may take up to a few minutes depending on grid sizes.")
    # Problem 1
    p1 = problem_1_compute(DEFAULTS, verbose=True)
    # Problem 2
    p2 = problem_2_optimize(DEFAULTS, coarse_steps=9, verbose=True)

    # Problem 3: FY1 drop 3 munitions to interfere M1
    print("\nPlanning Problem 3 (FY1 drop 3 munitions) ...")
    p3 = plan_three_munitions_single_uav(DEFAULTS["FY1_pos0"], defaults=DEFAULTS, uav_speed=DEFAULTS["uav_speed_default"],
                                         n_munitions=3,
                                         tgrid=np.linspace(0.0,70.0,71), fusegrid=np.linspace(0.5,40.0,41),
                                         verbose=True)
    save_result1_excel(p3, filename="result1.xlsx")
    print("Saved Problem 3 result to result1.xlsx")

    # Problem 4: FY1 FY2 FY3 each drop 1 munition to interfere M1
    print("\nPlanning Problem 4 (FY1,FY2,FY3 each drop 1) ...")
    p4 = plan_three_uavs_one_each(DEFAULTS, tgrid=np.linspace(0.0,70.0,71), fusegrid=np.linspace(0.5,40.0,41), verbose=True)
    save_result2_excel(p4, filename="result2.xlsx")
    print("Saved Problem 4 result to result2.xlsx")

    # Problem 5: 5 UAVs each up to 3 munitions, interfere M1,M2,M3
    print("\nPlanning Problem 5 (5 UAVs up to 3 munitions each, target M1,M2,M3) ...")
    uav_list = [
        ("FY1", DEFAULTS["FY1_pos0"]),
        ("FY2", DEFAULTS["FY2_pos0"]),
        ("FY3", DEFAULTS["FY3_pos0"]),
        ("FY4", DEFAULTS["FY4_pos0"]),
        ("FY5", DEFAULTS["FY5_pos0"])
    ]
    p5 = plan_multi_uavs_multi_missiles(DEFAULTS, uav_list=uav_list, max_per_uav=3, total_rounds=15,
                                       tgrid=np.linspace(0.0,70.0,71), fusegrid=np.linspace(0.5,40.0,41),
                                       verbose=True)
    save_result3_excel(p5, filename="result3.xlsx")
    print("Saved Problem 5 result to result3.xlsx")

    print("\nAll done. Summary:")
    print(f"Problem1 total shielding (single munition by FY1): {p1['total']:.6f} s")
    print(f"Problem2 approx best shielding found: {p2.get('total', 0.0):.6f} s")
    print(f"Problem3 total shielding (FY1 x3): {p3['total_shield']:.6f} s (saved result1.xlsx)")
    print(f"Problem4 total shielding (FY1,FY2,FY3 each x1): {p4['total_shield']:.6f} s (saved result2.xlsx)")
    print(f"Problem5 total shielding sum across M1/M2/M3: {p5['total_sum']:.6f} s (saved result3.xlsx)")

if __name__ == "__main__":
    main()
