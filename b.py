# 保存为 smoke_problem1_2_segment_shield.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# -----------------------------
# Problem constants (from PDF)
# -----------------------------
g = 9.80665
cloud_descent_rate = 3.0
cloud_radius = 10.0
cloud_duration = 20.0
missile_speed = 300.0

M1_pos0 = np.array([20000., 0., 2000.])
FY1_pos0 = np.array([17800., 0., 1800.])
fake_target = np.array([0.,0.,0.])
true_target = np.array([0.,200.,0.])   # 真目标下底圆心 (0,200,0)

# -----------------------------
# helper functions
# -----------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def missile_velocity():
    return unit(fake_target - M1_pos0) * missile_speed

def missile_pos(t):
    return M1_pos0 + missile_velocity() * t

def uav_velocity_from_heading(speed, heading_angle):
    # heading_angle in radians, absolute in XY plane (0 = +x)
    return np.array([math.cos(heading_angle), math.sin(heading_angle), 0.0]) * speed

def explosion_pos_from_release(uav_pos0, v_uav, t_release, fuse_delay):
    release_pos = uav_pos0 + v_uav * t_release
    dt = fuse_delay
    disp = v_uav * dt + np.array([0.0, 0.0, -0.5 * g * dt * dt])
    return release_pos + disp

def cloud_center_at_time(t, explosion_pos, t_explosion):
    # valid for t >= t_explosion
    return explosion_pos + np.array([0.0, 0.0, -cloud_descent_rate * (t - t_explosion)])

# segment-sphere intersection (standard quadratic) for segment P->Q and sphere (C,R)
def segment_sphere_intersect_batch(Ps, Qs, C, R):
    # Ps, Qs: arrays shape (N,3), checks N segments P->Q against same sphere (C,R)
    # returns boolean array length N
    d = Qs - Ps                             # Nx3
    f = Ps - C                              # Nx3
    a = np.einsum('ij,ij->i', d, d)         # N
    b = 2.0 * np.einsum('ij,ij->i', f, d)
    c = np.einsum('ij,ij->i', f, f) - R*R
    result = np.zeros(a.shape[0], dtype=bool)
    eps = 1e-12
    # handle degenerate a ~ 0 (segment length almost zero)
    deg = a < eps
    if np.any(deg):
        result[deg] = (c[deg] <= 0.0)
    nondeg = ~deg
    if np.any(nondeg):
        disc = b[nondeg]**2 - 4*a[nondeg]*c[nondeg]
        pos_disc = disc >= 0
        idxs = np.nonzero(nondeg)[0][pos_disc]
        if idxs.size > 0:
            sd = np.sqrt(disc[pos_disc])
            # compute t roots and check if any root in [0,1]
            b_sub = b[idxs]; a_sub = a[idxs]; sd_sub = sd
            t1 = (-b_sub - sd_sub) / (2*a_sub)
            t2 = (-b_sub + sd_sub) / (2*a_sub)
            result[idxs] = ( (t1 >= 0) & (t1 <= 1) ) | ( (t2 >= 0) & (t2 <= 1) )
    return result

# 计算遮蔽时间区间
def find_shield_intervals_by_sampling(explosion_pos, t_explosion, dt_sample=0.01, radius=cloud_radius):
    t0 = t_explosion
    t1 = t_explosion + cloud_duration
    times = np.arange(t0, t1 + dt_sample*0.5, dt_sample) #采样时间点
    # 计算所有采样时间点的导弹位置
    Ms = M1_pos0[None,:] + missile_velocity()[None,:] * times[:,None]   # shape (T,3)
    # 计算所有采样时间点的云中心位置
    Cs = explosion_pos[None,:] + np.array([0.,0.,-cloud_descent_rate]) * (times - t_explosion)[:,None]
    # 计算所有采样时间点的遮蔽标志
    Ps = Ms
    Qs = np.tile(true_target[None,:], (len(times),1))
    # 判断每个时间点的遮蔽情况
    flags = segment_sphere_intersect_batch(Ps, Qs, Cs, radius)
    # 计算遮蔽时间区间
    intervals = []
    in_int = False
    cur_s = None
    for i,t in enumerate(times):
        if flags[i] and not in_int:
            in_int = True
            cur_s = t
        if (not flags[i]) and in_int:
            in_int = False
            cur_e = times[i-1]
            intervals.append((cur_s, cur_e))
    if in_int:
        intervals.append((cur_s, times[-1]))
    # 找到的粗略区间再细化端点，使用二分法
    refined = []
    for s,e in intervals:
        left = max(t0, s - dt_sample); right = s
        for _ in range(25):
            mid = 0.5*(left+right)
            P_mid = M1_pos0 + missile_velocity() * mid
            C_mid = cloud_center_at_time(mid, explosion_pos, t_explosion)
            if segment_sphere_intersect_batch(P_mid[None,:], np.tile(true_target[None,:],(1,1)), C_mid, radius)[0]:
                right = mid
            else:
                left = mid
        s_ref = right
        left = e; right = min(t1, e + dt_sample)
        for _ in range(25):
            mid = 0.5*(left+right)
            P_mid = M1_pos0 + missile_velocity() * mid
            C_mid = cloud_center_at_time(mid, explosion_pos, t_explosion)
            if segment_sphere_intersect_batch(P_mid[None,:], np.tile(true_target[None,:],(1,1)), C_mid, radius)[0]:
                left = mid
            else:
                right = mid
        e_ref = left
        refined.append((s_ref, e_ref))
    return refined

# 问题1
def problem1(dt_sample=0.01, verbose=True):
    uav_speed = 120.0 #无人机速度
    t_release = 1.5     #投放时间
    fuse = 3.6      #引信延时
    #计算引爆位置和时间
    v_uav = uav_velocity_from_heading(uav_speed, math.atan2((fake_target-FY1_pos0)[1], (fake_target-FY1_pos0)[0]))
    t_expl = t_release + fuse
    expl_pos = explosion_pos_from_release(FY1_pos0, v_uav, t_release, fuse)
    #计算遮蔽时间区间
    intervals = find_shield_intervals_by_sampling(expl_pos, t_expl, dt_sample=dt_sample)
    total = sum(e-s for s,e in intervals)
    if verbose:
        print("问题1结果:")
        print("爆炸位置:", expl_pos, "爆炸时间:", t_expl)
        print("遮蔽时间区间:", intervals)
        print("总遮蔽时间:", total, "s")
    return {"explosion_pos":expl_pos,"t_explosion":t_expl,"intervals":intervals,"total":total}

# -----------------------------
# Problem 2: vectorized search (coarse-to-fine)
# - coarse search over heading offsets, UAV speed, t_release and fuse (coarse grids)
# - keep top K candidates, then refine each candidate with finer grids
# vectorized evaluation: for each candidate explosion we evaluate the time-window samples in batch (fast)
# -----------------------------
def search_problem2(
    heading_center=None, heading_span=math.pi*2/3, heading_steps=13,
    speed_bounds=(70,140), speed_steps=6,
    t_release_range=(0,70), t_release_step=1.0,
    fuse_range=(0.5,40), fuse_step=1.0,
    dt_sample_coarse=0.05,
    top_k=10,
    refine_params=None
):
    if heading_center is None:
        heading_center = math.atan2((fake_target - FY1_pos0)[1], (fake_target - FY1_pos0)[0])
    # build candidate grids
    headings = np.linspace(heading_center - heading_span/2, heading_center + heading_span/2, heading_steps)
    speeds = np.linspace(speed_bounds[0], speed_bounds[1], speed_steps)
    t_releases = np.arange(t_release_range[0], t_release_range[1]+1e-6, t_release_step)
    fuses = np.arange(fuse_range[0], fuse_range[1]+1e-6, fuse_step)
    # precompute missile times grid for cloud sample window length
    # We'll evaluate each candidate explosion by sampling times [t_exp, t_exp+cloud_duration] with dt_sample_coarse
    results = []
    total_candidates = headings.size * speeds.size * t_releases.size * fuses.size
    print(f"Total candidates (coarse): {total_candidates}")
    cand_count = 0
    start = time.time()
    for heading in headings:
        for speed in speeds:
            v_uav = uav_velocity_from_heading(speed, heading)
            for t_rel in t_releases:
                for fuse in fuses:
                    t_exp = t_rel + fuse
                    expl_pos = explosion_pos_from_release(FY1_pos0, v_uav, t_rel, fuse)
                    # sample times for this candidate
                    ts = np.arange(t_exp, t_exp + cloud_duration + dt_sample_coarse*0.5, dt_sample_coarse)
                    # compute Ms and Cs arrays
                    Ms = M1_pos0[None,:] + missile_velocity()[None,:] * ts[:,None]
                    Cs = expl_pos[None,:] + np.array([0.,0.,-cloud_descent_rate]) * (ts - t_exp)[:,None]
                    Ps = Ms
                    Qs = np.tile(true_target[None,:], (len(ts),1))
                    flags = segment_sphere_intersect_batch(Ps, Qs, Cs, cloud_radius)
                    if flags.any():
                        # merge contiguous true segments to get coarse intervals (then compute coarse total)
                        idxs = np.nonzero(flags)[0]
                        # find groups of consecutive indices
                        groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
                        coarse_intervals = [(ts[g[0]], ts[g[-1]]) for g in groups]
                        coarse_total = sum(e-s for s,e in coarse_intervals)
                    else:
                        coarse_intervals = []
                        coarse_total = 0.0
                    if coarse_total > 0:
                        results.append({
                            "heading": heading, "speed": float(speed), "t_release": float(t_rel), "fuse": float(fuse),
                            "t_explosion": float(t_exp), "explosion_pos": expl_pos, "coarse_total": float(coarse_total),
                            "coarse_intervals": coarse_intervals
                        })
                    cand_count += 1
    elapsed = time.time() - start
    print(f"Coarse search done ({cand_count} candidates) in {elapsed:.1f}s, found {len(results)} positive candidates")
    if not results:
        return {"coarse":[], "refined":[]}
    # keep top_k by coarse_total
    results_sorted = sorted(results, key=lambda x: -x["coarse_total"])
    top = results_sorted[:top_k]
    refined = []
    # refine each top candidate with finer sampling and endpoint bisection
    for r in top:
        expl_pos = r["explosion_pos"]
        t_exp = r["t_explosion"]
        intervals = find_shield_intervals_by_sampling(expl_pos, t_exp, dt_sample=refine_params.get("dt_sample", 0.01))
        total = sum(e-s for s,e in intervals)
        rr = r.copy()
        rr.update({"intervals": intervals, "total": total})
        refined.append(rr)
    refined_sorted = sorted(refined, key=lambda x: -x["total"])
    return {"coarse": results_sorted, "refined": refined_sorted}

# -----------------------------
# Example usage and quick-run
# -----------------------------
if __name__ == "__main__":
    # Problem 1 quick compute
    p1 = problem1(dt_sample=0.01, verbose=True)

    # # Problem 2 quick coarse->refine search (default parameters are conservative so this runs quickly)
    # refine_params = {"dt_sample": 0.01}
    # # coarse grid relatively coarse to finish in reasonable time; you can increase grids for better optimum
    # res2 = search_problem2(
    #     heading_steps=13, speed_steps=6,
    #     t_release_step=1.0, fuse_step=1.0,
    #     dt_sample_coarse=0.05, top_k=8, refine_params=refine_params
    # )
    # # print top refined candidate(s)
    # print("Top refined candidates (problem2):")
    # for i, cand in enumerate(res2["refined"][:5], start=1):
    #     print(f"{i}) speed={cand['speed']}, heading={math.degrees(cand['heading']):.2f}deg, t_release={cand['t_release']}, fuse={cand['fuse']}, total={cand['total']:.6f}, intervals={cand['intervals']}")

    # # optional: save a plot of best candidate if any
    # if res2["refined"]:
    #     best = res2["refined"][0]
    #     t_exp = best["t_explosion"]
    #     ts = np.linspace(t_exp, t_exp + cloud_duration, 800)
    #     missile_z = (M1_pos0[2] + missile_velocity()[2] * ts)
    #     cloud_z = best["explosion_pos"][2] + (-cloud_descent_rate) * (ts - t_exp)
    #     plt.figure(figsize=(8,4))
    #     plt.plot(ts, missile_z, label="missile z")
    #     plt.plot(ts, cloud_z, label="cloud z")
    #     for s,e in best["intervals"]:
    #         plt.axvspan(s,e,color='gray',alpha=0.4)
    #     plt.legend(); plt.xlabel("t (s)"); plt.ylabel("alt (m)")
    #     plt.title("Best candidate (problem2) z vs t")
    #     plt.tight_layout()
    #     plt.savefig("problem2_best_segment_shield.png")
    #     print("Saved plot problem2_best_segment_shield.png")
