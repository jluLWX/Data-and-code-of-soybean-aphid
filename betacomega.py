import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time

# %% 1. 固定所有基础参数（大豆蚜-异色瓢虫模型）
# 1.1 大豆蚜发育/存活参数（补充α4定义，修复NameError）
α1 = 0.25; α2 = 0.08; α3 = 0.16; α4 = 0.3461  # 补充α4，与原模型一致
μ1 = 0.035; μ2 = 0.025; μ3 = 0.015; μ4 = 0.05;
b = 0.6; γ = 0.8; K = 0.02; μ_L = 0.02  # 异色瓢虫基础死亡率
# 1.2 固定初始条件
N12_0 = 0.01; N3_0 = 0.008; N4_0 = 0.005; A0 = 0.001014; L0_fixed = 100
t_full = np.arange(1, 28)  # 1-27天（与原模型时间范围一致）
# 1.3 待分析参数梯度（各5个值，符合生物意义）
β_list = [0.001, 0.003, 0.005, 0.007, 0.009]  # 捕食率β
c_list = [0.05, 0.08, 0.1, 0.12, 0.15]        # 转化效率c
ω_list = [0.01, 0.015, 0.02, 0.025, 0.03]     # 异色瓢虫额外死亡率ω

# %% 2. 大豆蚜-异色瓢虫动力学模型（兼容β/c/ω三类参数分析）
def aphid_ladybug_model(t, y, beta, c, omega):
    """
    模型变量：y = [N12(1-2龄若虫), N3(3龄若虫), N4(4龄若虫), A(成虫), L(异色瓢虫)]
    """
    N12, N3, N4, A, L = y
    # 微分方程组（与原模型一致）
    dN12_dt = b * A - (α1 + μ1) * N12
    dN3_dt = α1 * N12 - (α2 + μ2) * N3
    dN4_dt = α2 * N3 - (α3 + μ3) * N4
    dA_dt = γ * A * (1 - A / K) + α3 * N4 - (α4 + μ4) * A - beta * L * A
    dL_dt = c * beta * L * A - (μ_L + omega) * L
    return [dN12_dt, dN3_dt, dN4_dt, dA_dt, dL_dt]

# %% 3. 批量计算函数（兼容β/c/ω三类参数）
def calculate_A_L_for_param(param_list, param_type):
    """
    计算不同参数下的A(t)和L(t)
    param_type: 'beta'/'c'/'omega'
    返回：(A_results列表, L_results列表)
    """
    A_results = []
    L_results = []
    start_time = time.time()
    for param in param_list:
        y0 = [N12_0, N3_0, N4_0, A0, L0_fixed]
        # 根据参数类型固定其他参数（基准值：β=0.005, c=0.1, ω=0.02）
        if param_type == 'beta':
            sol = solve_ivp(
                fun=lambda t,y: aphid_ladybug_model(t, y, beta=param, c=0.1, omega=0.02),
                t_span=(1, 27), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        elif param_type == 'c':
            sol = solve_ivp(
                fun=lambda t,y: aphid_ladybug_model(t, y, beta=0.005, c=param, omega=0.02),
                t_span=(1, 27), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        elif param_type == 'omega':
            sol = solve_ivp(
                fun=lambda t,y: aphid_ladybug_model(t, y, beta=0.005, c=0.1, omega=param),
                t_span=(1, 27), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        # 插值保证时间序列一致（与参考代码对齐）
        A_fit = interp1d(sol.t, sol.y[3], kind='linear')(t_full)
        L_fit = interp1d(sol.t, sol.y[4], kind='linear')(t_full)
        A_results.append(A_fit)
        L_results.append(L_fit)
    print(f"所有{param_type}参数的A(t)和L(t)计算完成，运行时间：{time.time() - start_time:.2f}秒")
    return A_results, L_results

# 执行三类参数计算
A_beta, L_beta = calculate_A_L_for_param(β_list, 'beta')
A_c, L_c = calculate_A_L_for_param(c_list, 'c')
A_omega, L_omega = calculate_A_L_for_param(ω_list, 'omega')

# %% 4. 3行2列组图可视化（完全对齐参考样式，修复\omega转义问题）
# 全局样式配置（与参考代码一致）
plt.rcParams['font.family'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'

# 1. 创建3行2列布局（与参考代码一致的尺寸）
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, wspace=0.25, hspace=0.4)

# 2. 样式配置（配色、线型、线宽与参考代码完全一致）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 蓝、橙、绿、红、紫
linestyles = ['-', '--', '-.', ':', '-']  # 差异化线型
linewidths = [2.2, 2.2, 2.2, 2.5, 2.8]    # 粗细区分
# 图例标签（修复\omega转义：改为\\omega）
labels_beta = [f'$β = {b}$' for b in β_list]
labels_c = [f'$c = {c}$' for c in c_list]
labels_omega = [f'$\\omega = {ω}$' for ω in ω_list]  # 修复转义错误
# Y轴标签（学术化表述）
y_label_A = 'Adult Soybean Aphid Density (individuals/m²)'
y_label_L = 'Harmonia axyridis Density (individuals/m²)'

# 3. 第1行：β的影响（隐藏Y轴文字）
ax1 = axes[0, 0]  # β对A(t)
for A_fit, color, ls, lw, label in zip(A_beta, colors, linestyles, linewidths, labels_beta):
    ax1.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax1.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax1.set_title('(A) Effect of $β$ on Adult Soybean Aphids ($A(t)$)', fontweight='bold', loc='left', pad=15)
ax1.legend(loc='upper left', ncol=1)
ax1.set_xticks(t_full[::4])  # 每4天一个刻度
ax1.set_xlim(1, 27); ax1.set_ylim(bottom=0)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax1.set_yticklabels([])  # 取消Y轴刻度文字

ax2 = axes[0, 1]  # β对L(t)
for L_fit, color, ls, lw, label in zip(L_beta, colors, linestyles, linewidths, labels_beta):
    ax2.plot(t_full, L_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax2.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax2.set_title('(B) Effect of $β$ on Harmonia axyridis ($L(t)$)', fontweight='bold', loc='left', pad=15)
ax2.legend(loc='upper right', ncol=1)
ax2.set_xticks(t_full[::4])
ax2.set_xlim(1, 27); ax2.set_ylim(bottom=0)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax2.set_yticklabels([])  # 取消Y轴刻度文字

# 4. 第2行：c的影响（保留Y轴文字）
ax3 = axes[1, 0]  # c对A(t)
for A_fit, color, ls, lw, label in zip(A_c, colors, linestyles, linewidths, labels_c):
    ax3.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax3.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax3.set_ylabel(y_label_A, fontsize=13, fontweight='bold')  # 保留Y轴文字
ax3.set_title('(C) Effect of $c$ on Adult Soybean Aphids ($A(t)$)', fontweight='bold', loc='left', pad=15)
ax3.legend(loc='upper left', ncol=1)
ax3.set_xticks(t_full[::4])
ax3.set_xlim(1, 27); ax3.set_ylim(bottom=0)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

ax4 = axes[1, 1]  # c对L(t)
for L_fit, color, ls, lw, label in zip(L_c, colors, linestyles, linewidths, labels_c):
    ax4.plot(t_full, L_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax4.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax4.set_ylabel(y_label_L, fontsize=13, fontweight='bold')  # 保留Y轴文字
ax4.set_title('(D) Effect of $c$ on Harmonia axyridis ($L(t)$)', fontweight='bold', loc='left', pad=15)
ax4.legend(loc='upper right', ncol=1)
ax4.set_xticks(t_full[::4])
ax4.set_xlim(1, 27); ax4.set_ylim(bottom=0)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

# 5. 第3行：ω的影响（隐藏Y轴文字，修复\omega转义）
ax5 = axes[2, 0]  # ω对A(t)
for A_fit, color, ls, lw, label in zip(A_omega, colors, linestyles, linewidths, labels_omega):
    ax5.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax5.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax5.set_title('(E) Effect of $\\omega$ on Adult Soybean Aphids ($A(t)$)', fontweight='bold', loc='left', pad=15)  # 修复转义
ax5.legend(loc='upper left', ncol=1)
ax5.set_xticks(t_full[::4])
ax5.set_xlim(1, 27); ax5.set_ylim(bottom=0)
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
ax5.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax5.set_yticklabels([])  # 取消Y轴刻度文字

ax6 = axes[2, 1]  # ω对L(t)
for L_fit, color, ls, lw, label in zip(L_omega, colors, linestyles, linewidths, labels_omega):
    ax6.plot(t_full, L_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax6.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax6.set_title('(F) Effect of $\\omega$ on Harmonia axyridis ($L(t)$)', fontweight='bold', loc='left', pad=15)  # 修复转义
ax6.legend(loc='upper right', ncol=1)
ax6.set_xticks(t_full[::4])
ax6.set_xlim(1, 27); ax6.set_ylim(bottom=0)
ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
ax6.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax6.set_yticklabels([])  # 取消Y轴刻度文字

# 6. 保存高清图（SCI标准，多格式）
plt.savefig('Beta_C_Omega_Effect_on_A_L_3x2_Final.svg', dpi=600, format='svg', bbox_inches='tight')
plt.savefig('Beta_C_Omega_Effect_on_A_L_3x2_Final.png', dpi=600, format='png', bbox_inches='tight')
plt.savefig('Beta_C_Omega_Effect_on_A_L_3x2_Final.eps', dpi=600, format='eps', bbox_inches='tight')

# 显示图像
plt.show()
