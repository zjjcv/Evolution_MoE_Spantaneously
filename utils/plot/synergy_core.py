import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline

# 设置绘图风格
sns.set_theme(style="white")
# 使用系统可用的字体（DejaVu Sans 是 Linux 默认字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']

def plot_synergistic_core(csv_path, is_layer_level=False):
    # 1. 加载数据
    df = pd.read_csv(csv_path)

    if is_layer_level:
        # 层级别绘图：只需要层级演变图
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # 归一化横轴和纵轴到 [0, 1]
        x_norm = (df['Layer'] - df['Layer'].min()) / (df['Layer'].max() - df['Layer'].min())
        y_min, y_max = df['Syn_Red_Rank'].min(), df['Syn_Red_Rank'].max()
        y_norm = (df['Syn_Red_Rank'] - y_min) / (y_max - y_min)

        # 增强平滑：使用更大的移动平均窗口
        window_size = max(4, len(x_norm) // 5)  # 增大窗口
        y_ma = np.convolve(y_norm, np.ones(window_size)/window_size, mode='same')

        # 使用更多的插值点数，使曲线更细腻
        x_smooth = np.linspace(x_norm.min(), x_norm.max(), 200)  # 增加到200点

        # 使用Savitzky-Golay滤波器进一步平滑
        from scipy.signal import savgol_filter
        y_smooth = savgol_filter(y_ma, window_length=min(15, len(y_ma)//2*2+1), polyorder=3)

        # 最后进行样条插值
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(x_norm, y_smooth, k=3)
        y_final = spline(x_smooth)

        # 绘制平滑曲线
        ax.plot(x_smooth, y_final, color='#d62728', lw=3.5)

        # 绘制原始数据点（带黑色边框）
        ax.plot(x_norm, y_norm, 'o', color='#d62728', markersize=8,
                markeredgecolor='black', markeredgewidth=2, zorder=5)

        # 填充区域
        ax.fill_between(x_smooth, y_final, 0, color='#d62728', alpha=0.25)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title('Gemma3-4B: Layer-wise Synergistic Core Profile', fontsize=14, pad=15)
        ax.set_xlabel('Normalized Layer Index', fontsize=12)
        ax.set_ylabel('Normalized Syn-Red Rank', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        output_path = './results/Gemma3-4B-Instruct/layer_synergy_profile.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"🚀 绘图完成！请查看 {output_path}")
        plt.show()

    else:
        # 头级别绘图：热力图 + 层级演变图
        heatmap_data = df.pivot(index='Head', columns='Layer', values='Syn_Red_Rank')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

        # --- 图 A: Syn-Red Rank 热力图（轻度高斯模糊） ---
        # 对热力图数据进行轻度高斯模糊处理
        heatmap_array = heatmap_data.values
        # 使用sigma=0.8进行轻度模糊，平滑趋势同时保留细节
        heatmap_blurred = gaussian_filter(heatmap_array, sigma=1.2)

        sns.heatmap(heatmap_blurred, cmap='RdBu_r', center=0, ax=ax1,
                    cbar_kws={'label': 'Syn-Red Rank Difference'},
                    xticklabels=heatmap_data.columns,
                    yticklabels=heatmap_data.index)
        ax1.set_title('Distribution of Synergistic and Redundant Heads', fontsize=14, pad=15)
        ax1.set_xlabel('Layer Index (Bottom to Top)', fontsize=12)
        ax1.set_ylabel('Head Index', fontsize=12)

        # --- 图 B: 层级演变图（高度平滑） ---
        layer_avg = df.groupby('Layer')['Syn_Red_Rank'].mean().reset_index()

        # 归一化横轴和纵轴到 [0, 1]
        x_norm = (layer_avg['Layer'] - layer_avg['Layer'].min()) / (layer_avg['Layer'].max() - layer_avg['Layer'].min())
        y_min, y_max = layer_avg['Syn_Red_Rank'].min(), layer_avg['Syn_Red_Rank'].max()
        y_norm = (layer_avg['Syn_Red_Rank'] - y_min) / (y_max - y_min)

        # 增强平滑：使用更大的移动平均窗口
        window_size = max(4, len(x_norm) // 5)  # 增大窗口
        y_ma = np.convolve(y_norm, np.ones(window_size)/window_size, mode='same')

        # 使用更多的插值点数，使曲线更细腻
        x_smooth = np.linspace(x_norm.min(), x_norm.max(), 200)  # 增加到200点

        # 使用Savitzky-Golay滤波器进一步平滑
        from scipy.signal import savgol_filter
        y_smooth = savgol_filter(y_ma, window_length=min(15, len(y_ma)//2*2+1), polyorder=3)

        # 最后进行样条插值
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(x_norm, y_smooth, k=3)
        y_final = spline(x_smooth)

        # 绘制平滑曲线
        ax2.plot(x_smooth, y_final, color='#d62728', lw=3.5)

        # 绘制原始数据点（带黑色边框）
        ax2.plot(x_norm, y_norm, 'o', color='#d62728', markersize=8,
                markeredgecolor='black', markeredgewidth=2, zorder=5)

        # 填充区域
        ax2.fill_between(x_smooth, y_final, 0, color='#d62728', alpha=0.25)

        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Layer-wise Synergistic Core Profile', fontsize=14, pad=15)
        ax2.set_xlabel('Normalized Layer Index', fontsize=12)
        ax2.set_ylabel('Normalized Syn-Red Rank', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig('./results/Gemma3-4B-Instruct/figure2_gemma_syn_red.png', dpi=300)
        print("🚀 绘图完成！请查看 ./results/Gemma3-4B-Instruct/figure2_gemma_syn_red.png")
        plt.show()

if __name__ == "__main__":
    # Gemma3头级别结果（per-head，与论文一致）
    plot_synergistic_core("./results/Gemma3-4B-Instruct/head_syn_red_ranks.csv", is_layer_level=False)