import matplotlib.pyplot as plt
import numpy as np

def draw_cylinder(ax, center_x, center_y, radius, height, color, alpha=0.5):
    """绘制一个底面在地面的圆柱体"""
    z = np.linspace(0, height, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center_x + radius * np.cos(theta_grid)
    y_grid = center_y + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

def plot_scene(clouds, missiles, target_pos=(0,200), target_height=1000):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 画云团
    for i, (x, y, h) in enumerate(clouds):
        draw_cylinder(ax, x, y, radius=1000, height=h, color="gray", alpha=0.4)

    # 画真实目标
    tx, ty = target_pos
    th = target_height
    draw_cylinder(ax, tx, ty, radius=700, height=th, color="red", alpha=0.6)

    # 画导弹
    for i, (mx, my, mz) in enumerate(missiles):
        ax.scatter(mx, my, mz, color="blue", s=5)
        ax.plot([mx, 0], [my, 0], [mz, 0], color="blue", linewidth=2, label="Missile Path" if i==0 else "")

    # 画原点 (导弹终点)
    ax.scatter(0, 0, 0, color="black", s=20, marker="x", label="Missile End (0,0,0)")

    # 坐标范围
    ax.set_xlim(0, 20000)
    ax.set_ylim(-5000, 5000)
    ax.set_zlim(0, 2500)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.legend()
    plt.show()


# 示例
clouds = [
    (17901.562405354172, 12.491939453173309, 1797.222195864321)
    # (40, 30, 20),
    # (70, 20, 25)
]

missiles = [
    (20000.0, 0.0, 2000.0)
]

plot_scene(clouds, missiles, target_pos=(50,50), target_height=7)
