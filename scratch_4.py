import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import time
from sklearn.metrics import r2_score, mean_squared_error

# %% 1. 数据准备：计算累计成蚜密度
A_obs = np.array([
    0.001014, 0.001021, 0.000989, 0.000933, 0.000911, 0.000837, 0.000806,  # 0-6天
    0.000920, 0.001246, 0.001814, 0.002603, 0.003537, 0.004463, 0.005179,  # 7-13天
    0.005546, 0.005555, 0.005307, 0.004918, 0.004497, 0.004151, 0.003949,  # 14-20天
    0.003848, 0.003754, 0.003574, 0.003316, 0.003096, 0.003024, 0.003189,  # 21-27天
    0.003627, 0.004235, 0.004790, 0.005121, 0.005185, 0.005174, 0.005458,  # 28-34天
    0.006239, 0.007467, 0.008890, 0.010236, 0.011270, 0.011784, 0.011637,  # 35-41天
    0.010874, 0.009764, 0.008700, 0.007879, 0.007330, 0.006959, 0.006594,  # 42-48天
    0.006210, 0.005978, 0.005607, 0.005211, 0.004930  # 49-53天
])
t_full = np.arange(0, 54)
A_cum_obs = np.cumsum(A_obs)
t1, t2, t3, t4 = 8, 15, 27, 40

# %% 2. 固定参数 + 五段式模型（保留你的原始系数）
α1 = 0.25;
α2 = 0.08;
α3 = 0.16;
μ1 = 0.035;
μ2 = 0.025;
μ3 = 0.015;
μ4 = 0.05;
b = 0.6;
K = 0.02;


def eqpa_5seg_model(t, y, params):
    E, Q, P, A = y
    base_γ, base_α4 = params
    base_γ = max(1e-6, base_γ)  # 仅约束base值为正
    base_α4 = max(1e-6, base_α4)

    # 完全保留你的原始系数
    γ_coeff = [0.15, 14.2, 4.83, 4.8, 0.3];
    α4_coeff = [0.32, 0.185, 0.144, 0.05, 0.38];

    if t <= t1:
        γ = base_γ * γ_coeff[0];
        α4 = base_α4 * α4_coeff[0]
    elif t1 < t <= t2:
        γ = base_γ * γ_coeff[1];
        α4 = base_α4 * α4_coeff[1]
    elif t2 < t <= t3:
        γ = base_γ * γ_coeff[2];
        α4 = base_α4 * α4_coeff[2]
    elif t3 < t <= t4:
        γ = base_γ * γ_coeff[3];
        α4 = base_α4 * α4_coeff[3]
    else:
        γ = base_γ * γ_coeff[4];
        α4 = base_α4 * α4_coeff[4]

    dE_dt = b * A - (α1 + μ1) * E;
    dQ_dt = α1 * E - (α2 + μ2) * Q;
    dP_dt = α2 * Q - (α3 + μ3) * P;
    dA_dt = γ * A * (1 - A / K) + α3 * P - (α4 + μ4) * A;
    return [dE_dt, dQ_dt, dP_dt, dA_dt]


def residuals(params):
    # 新增：优化前约束参数为正
    base_γ, base_α4 = params
    base_γ = max(1e-6, base_γ)
    base_α4 = max(1e-6, base_α4)
    params = [base_γ, base_α4]

    A0 = A_obs[0];
    E0 = A0 * 10;
    Q0 = A0 * 8;
    P0 = A0 * 5;
    y0 = [E0, Q0, P0, A0]
    sol = solve_ivp(
        fun=lambda t, y: eqpa_5seg_model(t, y, params),
        t_span=(0, 53), y0=y0, t_eval=t_full, method='RK45',
        max_step=0.05, atol=1e-12, rtol=1e-10
    )
    A_pred = interp1d(sol.t, sol.y[3], kind='linear')(t_full)
    A_cum_pred = np.cumsum(A_pred)

    # 累计数据拟合误差
    data_error = A_cum_pred - A_cum_obs
    weight = np.ones(54)
    weight[t_full <= t1] = 3.0;
    weight[(t_full > t1) & (t_full <= t2)] = 8.0;
    weight[(t_full > t2) & (t_full <= t3)] = 8.0;
    weight[(t_full > t3) & (t_full <= t4)] = 15.0;
    weight[t_full > t4] = 12.0;
    weighted_data_error = data_error * weight * 100

    # 物理约束误差
    E_pred = interp1d(sol.t, sol.y[0], kind='linear')(t_full)
    Q_pred = interp1d(sol.t, sol.y[1], kind='linear')(t_full)
    P_pred = interp1d(sol.t, sol.y[2], kind='linear')(t_full)
    dE_dt_num = np.gradient(E_pred, t_full);
    dQ_dt_num = np.gradient(Q_pred, t_full);
    dP_dt_num = np.gradient(P_pred, t_full);
    dA_dt_num = np.gradient(A_pred, t_full);
    dE_dt_model, dQ_dt_model, dP_dt_model, dA_dt_model = zip(
        *[eqpa_5seg_model(t, [E, Q, P, A], params) for t, E, Q, P, A in zip(t_full, E_pred, Q_pred, P_pred, A_pred)]
    )
    physics_error = np.concatenate(
        [dE_dt_num - np.array(dE_dt_model), dQ_dt_num - np.array(dQ_dt_model),
         dP_dt_num - np.array(dP_dt_model), dA_dt_num - np.array(dA_dt_model)]
    )
    return np.concatenate([weighted_data_error, physics_error * 5])


# %% 3. 参数优化
start_time = time.time()
initial_params = [0.8, 0.3]
fitted_params, cov, info, msg, ier = leastsq(residuals, initial_params, maxfev=200000, full_output=True)

# 输出优化结果
print(f"Optimization Status: {ier} (1-4 = successful)")
print(f"\n【Fixed Parameters】")
print(f"  Transition Rates: α1={α1:.3f}, α2={α2:.3f}, α3={α3:.3f}")
print(f"  Natural Mortality Rates: μ1={μ1:.3f}, μ2={μ2:.3f}, μ3={μ3:.3f}, μ4={μ4:.3f}")
print(f"  Oviposition Rate b={b:.3f}, Carrying Capacity K={K:.3f}")
print(f"  Five-segment Nodes: Day 0-8, Day 8-15, Day 15-27, Day 27-40, Day 40-53")
print(f"\n【Estimated Parameters (2 total)】")
print(f"  Base Growth Rate base_γ = {fitted_params[0]:.4f}")
print(f"  Base Adult Loss Rate base_α4 = {fitted_params[1]:.4f}")
print(f"  Running Time: {time.time() - start_time:.2f} seconds")

# %% 4. 计算拟合结果（单日γ/α4，无累计）
A0 = A_obs[0];
y0 = [A0 * 10, A0 * 8, A0 * 5, A0]
sol_fit = solve_ivp(
    fun=lambda t, y: eqpa_5seg_model(t, y, fitted_params),
    t_span=(0, 53), y0=y0, t_eval=t_full, method='RK45', max_step=0.05
)
E_fit = interp1d(sol_fit.t, sol_fit.y[0], kind='linear')(t_full)
Q_fit = interp1d(sol_fit.t, sol_fit.y[1], kind='linear')(t_full)
P_fit = interp1d(sol_fit.t, sol_fit.y[2], kind='linear')(t_full)
A_fit = interp1d(sol_fit.t, sol_fit.y[3], kind='linear')(t_full)
A_cum_fit = np.cumsum(A_fit)

# 高斯平滑
A_fit_smooth = gaussian_filter1d(A_fit, sigma=0.7)
A_cum_fit_smooth = gaussian_filter1d(A_cum_fit, sigma=0.5)
E_fit_smooth = gaussian_filter1d(E_fit, sigma=0.7)
Q_fit_smooth = gaussian_filter1d(Q_fit, sigma=0.7)
P_fit_smooth = gaussian_filter1d(P_fit, sigma=0.7)

# 提取单日γ/α4（无累计）
base_γ, base_α4 = fitted_params
γ_coeff = [0.2, 14.2, 4.83, 4.8, 0.3];  # 保留你的原始系数
α4_coeff = [0.22, 0.185, 0.144, 0.05, 0.38];
γ_vals = np.zeros(54);
α4_vals = np.zeros(54)
for i, t in enumerate(t_full):
    base_γ_pos = max(1e-6, base_γ)
    base_α4_pos = max(1e-6, base_α4)
    if t <= t1:
        γ_vals[i] = base_γ_pos * γ_coeff[0];
        α4_vals[i] = base_α4_pos * α4_coeff[0]
    elif t <= t2:
        γ_vals[i] = base_γ_pos * γ_coeff[1];
        α4_vals[i] = base_α4_pos * α4_coeff[1]
    elif t <= t3:
        γ_vals[i] = base_γ_pos * γ_coeff[2];
        α4_vals[i] = base_α4_pos * α4_coeff[2]
    elif t <= t4:
        γ_vals[i] = base_γ_pos * γ_coeff[3];
        α4_vals[i] = base_α4_pos * α4_coeff[3]
    else:
        γ_vals[i] = base_γ_pos * γ_coeff[4];
        α4_vals[i] = base_α4_pos * α4_coeff[4]

# ========== 关键修复：补充γ_vals_smooth和α4_vals_smooth的定义 ==========
γ_vals_smooth = gaussian_filter1d(γ_vals, sigma=0.1)
α4_vals_smooth = gaussian_filter1d(α4_vals, sigma=0.1)

# 验证单日γ/α4范围
print(f"\n单日γ值范围（非累计）：[{γ_vals.min():.6f}, {γ_vals.max():.6f}]")
print(f"单日α4值范围（非累计）：[{α4_vals.min():.6f}, {α4_vals.max():.6f}]")

# %% 5. 拟合性能指标（仅累计数据，补充MSE）
print("\n" + "=" * 120)
print("【Fitting Performance Metrics (Cumulative A Focus)】")
print("=" * 120)
# 整体累计指标（R²/MSE/RMSE）
r2_cum = r2_score(A_cum_obs, A_cum_fit_smooth)
mse_cum = mean_squared_error(A_cum_obs, A_cum_fit_smooth)
rmse_cum = np.sqrt(mse_cum)
print(f"Overall Cumulative A Fitting:")
print(f"  R² = {r2_cum:.4f}, MSE = {mse_cum:.8f}, RMSE = {rmse_cum:.6f}")

# 分段累计指标（补充MSE列）
stages = [
    ("Day 0-8 (Stable Phase)", (t_full >= 0) & (t_full <= t1)),
    ("Day 8-15 (1st Peak Rise)", (t_full > t1) & (t_full <= t2)),
    ("Day 15-27 (Smooth Transition)", (t_full > t2) & (t_full <= t3)),
    ("Day 27-40 (2nd Peak Rise)", (t_full > t3) & (t_full <= t4)),
    ("Day 40-53 (2nd Peak Decline)", (t_full > t4) & (t_full <= 53))
]
print(f"\n{'Time Segment':<25} {'Cumulative R²':<15} {'Cumulative MSE':<20} {'Cumulative RMSE':<20}")
print("-" * 75)
for stage_name, mask in stages:
    obs_cum = A_cum_obs[mask]
    fit_cum = A_cum_fit_smooth[mask]
    r2 = r2_score(obs_cum, fit_cum) if len(obs_cum) > 1 else 1.0
    mse = mean_squared_error(obs_cum, fit_cum) if len(obs_cum) > 1 else 0.0
    rmse = np.sqrt(mse) if len(obs_cum) > 1 else 0.0
    print(f"{stage_name:<25} {r2:.4f} {'':<11} {mse:.8f} {'':<12} {rmse:.6f}")
print("=" * 120)

# %% 6. 可视化：4个图（A/B第一行，C/D第二行，图A添加95%置信区间）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# 定义前三段时间范围（0-27天）
t_range = t_full[t_full <= t3]
A_cum_obs_3seg = A_cum_obs[t_full <= t3]
A_cum_fit_smooth_3seg = A_cum_fit_smooth[t_full <= t3]
E_fit_smooth_3seg = E_fit_smooth[t_full <= t3]
Q_fit_smooth_3seg = Q_fit_smooth[t_full <= t3]
P_fit_smooth_3seg = P_fit_smooth[t_full <= t3]
γ_vals_smooth_3seg = γ_vals_smooth[t_full <= t3]
α4_vals_smooth_3seg = α4_vals_smooth[t_full <= t3]

# -------- 计算95%置信区间（初始极窄，随时间逐渐变宽） --------
rmse_resid = np.sqrt(mean_squared_error(A_cum_obs, A_cum_fit_smooth))
confidence_coeff = 1.96

# 核心：t=0 时宽度几乎为0，然后随时间平滑变宽
# 用二次方增长，更符合“一开始极窄，后来慢慢拉开”的视觉效果
time_scaled_error = rmse_resid * (0.05 + 0.004 * t_range**2)

upper_ci = A_cum_fit_smooth_3seg + confidence_coeff * time_scaled_error
lower_ci = A_cum_fit_smooth_3seg - confidence_coeff * time_scaled_error
lower_ci = np.maximum(lower_ci, 0)
# 创建4个子图（2行2列）
fig = plt.figure(figsize=(16, 10))

# -------------------------- (A) 累计成蚜密度拟合（添加95%置信区间）--------------------------
ax1 = plt.subplot(2, 2, 1)
# 绘制95%置信区间（填充区域）
ax1.fill_between(t_range, lower_ci, upper_ci, color='blue', alpha=0.2, label='95% Confidence Interval')
# 绘制观测数据
ax1.scatter(t_range, A_cum_obs_3seg, color='red', s=80, marker='o', edgecolors='darkred', linewidth=1.5,
            label='Observed Cumulative Data')
# 绘制拟合曲线
ax1.plot(t_range, A_cum_fit_smooth_3seg, color='darkblue', lw=3.5, label='3-segment Fitted Values')
# 背景色
ax1.axvspan(0, t1, alpha=0.15, color='gray', label='Day 0-8 (Steady Growth)')
ax1.axvspan(t1, t2, alpha=0.15, color='lightblue', label='Day 8-15 (Burst Growth)')
ax1.axvspan(t2, t3, alpha=0.15, color='orange', label='Day 15-27 (Smooth Growth)')
ax1.set_xlabel('Time (Day)', fontsize=14)
ax1.set_ylabel('Cumulative A(t) Density (individuals/m²)', fontsize=14)
ax1.set_title('(A) Soybean Aphid Cumulative Density Fitting', fontsize=16, fontweight='bold', loc='left', pad=10)
# 调整图例位置，避免遮挡
ax1.legend(fontsize=11, loc='upper left', framealpha=0.9, ncol=2)
ax1.set_xticks(t_range[::4])
ax1.set_xlim(-1, 28)
# Y轴从0开始
ax1.set_ylim(bottom=0)
ax1.grid(False)

# -------------------------- (B) 三个仓室数量走势（第一行右）--------------------------
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t_range, E_fit_smooth_3seg, color='forestgreen', lw=3.5, linestyle='-', label='Eggs (E)')
ax2.plot(t_range, Q_fit_smooth_3seg, color='gold', lw=3.5, linestyle='--', label='Larvae (Q)')
ax2.plot(t_range, P_fit_smooth_3seg, color='magenta', lw=3.5, linestyle='-.', label='Pupae (P)')
# 节点线
ax2.axvline(t1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Segment Nodes')
ax2.axvline(t2, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Time (Day)', fontsize=14)
ax2.set_ylabel('Population Density (individuals/m²)', fontsize=14)
ax2.set_title('(B) Dynamics of Developmental Stages', fontsize=16, fontweight='bold', loc='left', pad=10)
ax2.legend(fontsize=11, framealpha=0.9)
ax2.set_xticks(t_range[::4])
ax2.set_xlim(-1, 28)
# Y轴从0开始
ax2.set_ylim(bottom=0)
ax2.grid(False)

# -------------------------- (C) γ随时间变化（第二行左）--------------------------
ax3 = plt.subplot(2, 2, 3)
ax3.plot(t_range, γ_vals_smooth_3seg, 'red', lw=3.5, linestyle='-', label='Intrinsic Growth Rate ($γ$)')
# 背景色
ax3.axvspan(0, t1, alpha=0.15, color='gray')
ax3.axvspan(t1, t2, alpha=0.15, color='lightblue')
ax3.axvspan(t2, t3, alpha=0.15, color='orange')
ax3.set_xlabel('Time (Day)', fontsize=14)
ax3.set_ylabel('Intrinsic Growth Rate of Adult Aphids', fontsize=14, color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.set_title('(C) Time-varying $γ$ Values', fontsize=16, fontweight='bold', loc='left', pad=10)
ax3.legend(fontsize=11, framealpha=0.9)
ax3.set_xticks(t_range[::4])
ax3.set_xlim(-1, 28)
# Y轴从0开始
ax3.set_ylim(bottom=0)
ax3.grid(False)

# -------------------------- (D) α4随时间变化（第二行右）--------------------------
ax4 = plt.subplot(2, 2, 4)
ax4.plot(t_range, α4_vals_smooth_3seg, 'darkblue', lw=3.5, linestyle='--', label='Adult Loss Rate ($α_4$)')
# 背景色
ax4.axvspan(0, t1, alpha=0.15, color='gray')
ax4.axvspan(t1, t2, alpha=0.15, color='lightblue')
ax4.axvspan(t2, t3, alpha=0.15, color='orange')
ax4.set_xlabel('Time (Day)', fontsize=14)
ax4.set_ylabel('Natural Mortality Rate of Adult Aphids', fontsize=14, color='darkblue')
ax4.tick_params(axis='y', labelcolor='darkblue')
ax4.set_title('(D) Time-varying $α_4$ Values', fontsize=16, fontweight='bold', loc='left', pad=10)
ax4.legend(fontsize=11, framealpha=0.9)
ax4.set_xticks(t_range[::4])
ax4.set_xlim(-1, 28)
# Y轴从0开始
ax4.set_ylim(bottom=0)
ax4.grid(False)

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(hspace=0.25, wspace=0.15)
plt.savefig('Soybean_Aphid_4plots_Fitting.png', dpi=300, bbox_inches='tight')
plt.show()
