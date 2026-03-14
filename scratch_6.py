import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ===================== 参数（保留你的修改） =====================
α1, α2, α3, α4 = 0.25, 0.08, 0.16, 0.05
μ1, μ2, μ3, μ4 = 0.035, 0.025, 0.015, 0.05
b, γ, K = 0.6, 0.8, 0.02

β = 0.005
c = 0.1
μ_L = 0.02
ω = 0.01

E0, Q0, P0, A0 = 0.01, 0.008, 0.005, 0.001014
L0_list = [0, 50, 100, 200, 300]
colors = ['k', 'r', 'b', 'g', 'orange']

t_span = (0, 27)
t_eval = np.linspace(0, 27, 28)


# ===================== 模型（无修改） =====================
def model(t, y):
    E, Q, P, A, L = y
    dE_dt = b * A - (α1 + μ1) * E
    dQ_dt = α1 * E - (α2 + μ2) * Q
    dP_dt = α2 * Q - (α3 + μ3) * P
    dA_dt = γ * A * (1 - A / K) + α3 * P - (α4 + μ4) * A - β * L * A
    dL_dt = c * β * L * A - (μ_L + ω) * L
    return [dE_dt, dQ_dt, dP_dt, dA_dt, dL_dt]


# ===================== 计算结果（无修改） =====================
data = {}
for L0 in L0_list:
    sol = solve_ivp(model, t_span, [E0, Q0, P0, A0, L0], t_eval=t_eval, method='RK45')
    data[L0] = sol.y

# ===================== 绘图：标题紧贴+左对齐优化 =====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 4,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'legend.fontsize': 4,
    'axes.labelpad': 0.1,
    'axes.titlepad': 0.0,  # 标题完全贴轴（关键）
    'xtick.labelsize': 3,
    'ytick.labelsize': 3,
})

fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])


# ---------------- 通用函数：背景浅化 + 网格线安全设置 ----------------
def setup_3d_ax(ax):
    """统一设置3D轴样式：背景透明 + 网格线极细极淡（兼容所有matplotlib版本）"""
    # 面板背景（几乎透明）
    ax.xaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.fill = True
    ax.yaxis.pane.set_facecolor('lightgray')
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.fill = True
    ax.zaxis.pane.set_facecolor('lightgray')
    ax.zaxis.pane.set_alpha(0.1)

    # 核心修复：补充linestyle键，避免KeyError + 网格线极细极淡
    ax.xaxis._axinfo["grid"] = {
        'linewidth': 0.1,
        'color': '#eeeeee',
        'alpha': 0.1,
        'linestyle': '-'  # 必须补充的键（默认实线）
    }
    ax.yaxis._axinfo["grid"] = {
        'linewidth': 0.1,
        'color': '#eeeeee',
        'alpha': 0.05,
        'linestyle': '-'
    }
    ax.zaxis._axinfo["grid"] = {
        'linewidth': 0.1,
        'color': '#eeeeee',
        'alpha': 0.05,
        'linestyle': '-'
    }

    # 刻度紧贴
    ax.tick_params(pad=0, length=1, labelsize=3)


def fill_curve_3d(ax, x, y, z, color, alpha=0.2):
    """3D曲线下方填充"""
    verts = []
    for j in range(len(x) - 1):
        verts.append([
            (x[j], y[j], 0),
            (x[j + 1], y[j + 1], 0),
            (x[j + 1], y[j + 1], z[j + 1]),
            (x[j], y[j], z[j])
        ])
    poly = Poly3DCollection(verts, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_collection3d(poly)


# -------- 左大图：蚜虫成虫 A --------
axA = fig.add_subplot(gs[:, 0], projection='3d')
for i, L0 in enumerate(L0_list):
    x = L0 * np.ones_like(t_eval)
    y = t_eval
    z = data[L0][3]
    axA.plot(x, y, z, color=colors[i], lw=1, label=f'$L_0={L0}$')
    fill_curve_3d(axA, x, y, z, colors[i], alpha=0.1)
axA.set_xlabel('Initial Release ($L_0$)')
axA.set_ylabel('Time (day)')
axA.set_zlabel('Adult Aphid Density ($A$)')
# 标题：左对齐(ha='left') + 底部对齐(va='bottom') + 最左侧(x=0.0) + 紧贴顶部(y=1.0)
axA.set_title('(A) Adult Aphis Glycines $A(t)$', fontsize=5, fontweight='bold',
              ha='left', va='bottom', x=0.0, y=1.1)
axA.legend(loc='upper left')
axA.view_init(elev=15, azim=-45)
setup_3d_ax(axA)

# -------- 右上小图：蚜虫卵 E --------
axE = fig.add_subplot(gs[0, 1], projection='3d')
for i, L0 in enumerate(L0_list):
    x = L0 * np.ones_like(t_eval)
    y = t_eval
    z = data[L0][0]
    axE.plot(x, y, z, color=colors[i], lw=1)
    fill_curve_3d(axE, x, y, z, colors[i], alpha=0.1)
axE.set_xlabel('$L_0$')
axE.set_ylabel('Time')
axE.set_zlabel('1st-2nd Instar Nymphs Density ($E$)')
axE.set_title('(B) 1st-2nd Instar Nymphs $E(t)$', fontsize=4, fontweight='bold',
              ha='left', va='bottom', x=0.0, y=1.0)
axE.view_init(elev=15, azim=-45)
setup_3d_ax(axE)

# -------- 右上小图：蚜虫幼虫 Q --------
axQ = fig.add_subplot(gs[0, 2], projection='3d')
for i, L0 in enumerate(L0_list):
    x = L0 * np.ones_like(t_eval)
    y = t_eval
    z = data[L0][1]
    axQ.plot(x, y, z, color=colors[i], lw=1)
    fill_curve_3d(axQ, x, y, z, colors[i], alpha=0.1)
axQ.set_xlabel('$L_0$')
axQ.set_ylabel('Time')
axQ.set_zlabel('3rd Instar Nymph Density ($Q$)')
axQ.set_title('(C) 3rd Instar Nymph $Q(t)$', fontsize=4, fontweight='bold',
              ha='left', va='bottom', x=0.0, y=1.0)
axQ.view_init(elev=15, azim=-45)
setup_3d_ax(axQ)

# -------- 右下小图：蚜虫蛹 P --------
axP = fig.add_subplot(gs[1, 1], projection='3d')
for i, L0 in enumerate(L0_list):
    x = L0 * np.ones_like(t_eval)
    y = t_eval
    z = data[L0][2]
    axP.plot(x, y, z, color=colors[i], lw=1)
    fill_curve_3d(axP, x, y, z, colors[i], alpha=0.1)
axP.set_xlabel('$L_0$')
axP.set_ylabel('Time')
axP.set_zlabel('4th Instar Nymph Density ($P$)')
axP.set_title('(D) 4th Instar Nymph $P(t)$', fontsize=4, fontweight='bold',
              ha='left', va='bottom', x=0.0, y=1.0)
axP.view_init(elev=15, azim=-45)
setup_3d_ax(axP)

# -------- 右下小图：异色瓢虫 L --------
axL = fig.add_subplot(gs[1, 2], projection='3d')
for i, L0 in enumerate(L0_list):
    x = L0 * np.ones_like(t_eval)
    y = t_eval
    z = data[L0][4]
    axL.plot(x, y, z, color=colors[i], lw=1)
    fill_curve_3d(axL, x, y, z, colors[i], alpha=0.1)
axL.set_xlabel('$L_0$')
axL.set_ylabel('Time')
axL.set_zlabel('Harmonia Axyridis Density ($L$)')
axL.set_title('(E) Harmonia Axyridis $L(t)$', fontsize=4, fontweight='bold',
              ha='left', va='bottom', x=0.0, y=1.0)
axL.view_init(elev=15, azim=-45)
setup_3d_ax(axL)

# 紧凑布局
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig('5_3D_plots_final.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
