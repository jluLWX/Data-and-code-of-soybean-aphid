import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

# ===================== 数据 =====================
obs_dates_str = [
    '17-Jul', '20-Jul', '23-Jul', '26-Jul', '29-Jul',
    '2-Aug', '5-Aug', '7-Aug', '10-Aug', '14-Aug'
]
obs_aphid_raw = np.array([3,6,3,5,24,35,17,17,23,8])
obs_rel_days = np.array([0,3,6,9,12,15,18,20,23,27])
full_rel_days = np.arange(0, 28)

# ===================== 插值 + 7天平均 =====================
interp_func = interpolate.CubicSpline(obs_rel_days, obs_aphid_raw)
full_aphid_interp = interp_func(full_rel_days)
full_aphid_interp[full_aphid_interp < 0] = 0

aphid_interp_series = pd.Series(full_aphid_interp)
aphid_ma7 = aphid_interp_series.rolling(
    window=7, center=True, min_periods=1
).mean()
full_aphid_ma7 = aphid_ma7.values
area = 5000

# ===================== 画图（核心：标题在外部上方+高度一致+图表丰富化）=====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.9
})

fig = plt.figure(figsize=(12, 6))  # 固定画布尺寸

# ------------------- 左：表格（标题在外部上方左上角）-------------------
# 轴位置：[左, 下, 宽, 高] → 顶部留空放标题（height=0.8，top=0.08+0.8=0.88）
ax_table = fig.add_axes([0.03, 0.08, 0.30, 0.80])
ax_table.axis('off')

# (A) 标题：在表格轴的外部上方左上角（基于画布坐标，精准对齐）
fig.text(
    0.03, 0.90,  # 画布坐标：x=0.03（和表格左边界对齐），y=0.90（表格顶部上方）
    '(A) Observed Data of Soybean Aphids',
    fontsize=11, fontweight='bold',
    va='bottom', ha='left'  # 底部对齐表格顶部，左对齐表格左边界
)

# 表格：填满轴区域，左上角对齐
table_data = [[d, str(v)] for d, v in zip(obs_dates_str, obs_aphid_raw)]
table = ax_table.table(
    cellText=table_data,
    colLabels=['Date', 'Aphid count'],
    cellLoc='center',
    loc='center',  # 表格在轴内居中
    colWidths=[0.6, 0.4],
    bbox=[0.0, 0.0, 1.0, 1.0]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.0)

# 表头样式
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# ------------------- 右：曲线图（丰富化优化）-------------------
# 轴位置：[左, 下, 宽, 高] → bottom=0.08、height=0.80（和表格完全一致）
ax1 = fig.add_axes([0.38, 0.08, 0.58, 0.80])
ax2 = ax1.twinx()

# (B) 标题：在曲线图轴的外部上方左上角（和(A)同y坐标，x对齐图左边界）
fig.text(
    0.38, 0.90,  # 画布坐标：x=0.38（和图左边界对齐），y=0.90（和(A)同高度）
    '(B) Cleaning and Standardization of Monitoring Data',
    fontsize=11, fontweight='bold',
    va='bottom', ha='left'  # 底部对齐图顶部，左对齐图左边界
)

# 1. 分段背景色（按蚜虫数量趋势划分阶段）
# 低发期（0-9天）、高发期（9-15天）、衰退期（15-27天）
ax1.axvspan(0, 9, alpha=0.1, color='lightgray', label='Low Incidence Period')
ax1.axvspan(9, 15, alpha=0.15, color='lightcoral', label='Peak Incidence Period')
ax1.axvspan(15, 27, alpha=0.1, color='lightblue', label='Decline Period')

# 2. 绘制曲线（优化样式）
# 插值曲线：虚线+轻微透明
ax1.plot(full_rel_days, full_aphid_interp, color='black', linestyle='--',
         linewidth=1.2, label='Interpolated Soybean Aphid', alpha=0.7)
# 7天平均：实线+加粗+深蓝色
ax1.plot(full_rel_days, full_aphid_ma7, color='#0047AB',
         linewidth=2.5, label='7-day Moving Average', alpha=0.9)
# 观测点：更大尺寸+渐变色+边框
ax1.scatter(obs_rel_days, obs_aphid_raw, color='#D55E00', s=60,
            zorder=10, label='Observed', edgecolor='black', linewidth=1)

# 3. 关键节点标注（峰值、谷值）
peak_day = full_rel_days[np.argmax(full_aphid_ma7)]
peak_value = full_aphid_ma7[np.argmax(full_aphid_ma7)]
ax1.annotate(f'Peak: {peak_value:.1f} ind',
             xy=(peak_day, peak_value), xytext=(5, 10),
             textcoords='offset points', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

# 4. 观测值标注优化（移除边框+调大字体）
for d, v in zip(obs_rel_days, obs_aphid_raw):
    ax1.annotate(str(v), (d, v), xytext=(0, 10), textcoords='offset points',
                 fontsize=10, ha='center', zorder=11)  # 移除bbox参数（边框），字体调大到10

# 5. 坐标轴优化
y1_max = np.max(obs_aphid_raw) * 1.3  # 增加顶部留白
y2_max = y1_max / area
ax1.set_xlabel('Time (Day)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Soybean aphid abundance (individuals)', fontsize=10, fontweight='bold')
ax2.set_ylabel(r'Density (individuals/ m²)', color='#0047AB', fontsize=10, fontweight='bold')

# Y轴刻度优化
ax1.set_xticks(np.arange(0, 28, 3))  # 每3天一个刻度，更密集
ax1.set_yticks(np.arange(0, y1_max+1, 5))  # 每5个个体一个刻度
ax2.set_yticks(np.arange(0, y2_max+0.001, 0.001))  # 密度刻度细化

# 轴样式优化
ax2.tick_params(axis='y', labelcolor='#0047AB', width=1.2)
ax2.spines['right'].set_color('#0047AB')
ax2.spines['right'].set_linewidth(1.2)
#ax1.spines['top'].set_visible(False)  # 隐藏顶部边框
ax2.spines['top'].set_visible(False)

# 6. 网格线（轻量级，不杂乱）
ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
ax1.set_axisbelow(True)  # 网格线在曲线下方

# 7. 图例优化（多列布局+位置调整）
ax1.legend(loc='upper right', ncol=2, fontsize=8.5, columnspacing=1.0, handletextpad=0.5)

# 8. 补充统计信息文本框
stats_text = f"""Key Statistics:
• Peak abundance: {peak_value:.1f} individuals
• Peak density: {peak_value/area:.6f} ind/m²
• Total observation days: 27
• Area: {area} m²"""
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# 坐标轴范围
ax1.set_xlim(0, 27)
ax1.set_ylim(0, y1_max)
ax2.set_ylim(0, y2_max)

# 保存+显示（避免裁剪标题）
plt.savefig('Table+Plot_SCI_0-27days_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 数据输出 =====================
print("="*60)
print("7天滑动平均数据（0-27天）")
print("="*60)
print(f"数据时间范围：0 - 27 天")
print(f"27天插值值：{full_aphid_interp[-1]:.2f} 头")
print(f"27天7天移动平均：{full_aphid_ma7[-1]:.2f} 头")
print(f"27天蚜虫密度（移动平均）：{full_aphid_ma7[-1]/area:.6f} individuals/m²")
print(f"峰值出现时间：第 {peak_day} 天，峰值数量：{peak_value:.2f} 头")

df_ma7 = pd.DataFrame({
    'Day': full_rel_days,
    'Interpolated_Aphid': full_aphid_interp,
    '7day_Moving_Average': full_aphid_ma7,
    'Density_MA7': full_aphid_ma7 / area
})
df_ma7.to_csv('soybean_aphid_ma7_0-27days.csv', index=False)
print("\n✅ 数据已导出到：soybean_aphid_ma7_0-27days.csv")
