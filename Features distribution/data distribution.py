import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ======================
# 全局设置
# ======================
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 全局字体设置
    'axes.titlesize': 12,  # 标题字号
    'axes.labelsize': 10,  # 坐标轴标签字号
    'xtick.labelsize': 8,  # X轴刻度字号
    'ytick.labelsize': 8,  # Y轴刻度字号
    'legend.fontsize': 8,  # 图例字号
    'axes.linewidth': 1.0,  # 坐标轴线宽
    'xtick.direction': 'in',  # 刻度线方向
    'ytick.direction': 'in',
    'xtick.major.size': 4,  # 主刻度长度
    'ytick.major.size': 4,
    'xtick.minor.size': 2,  # 次刻度长度
    'ytick.minor.size': 2,
})

# 定义配色方案
COLOR_HIST = '#4C72B0'  # 柔和的蓝色
COLOR_BOX = '#DD8452'  # 柔和的橙色


# ======================
# 数据加载函数
# ======================
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")


# ======================
# 分布分析函数
# ======================
def analyze_distributions(df, fig_save_path='generation'):
    """
    1. 垂直排列箱线图和直方图
    2. 使用科研配色方案
    3. 标注格式
    4. 添加分布统计信息
    """
    report = pd.DataFrame(index=df.columns,
                          columns=['dtype', 'unique', 'missing',
                                   'skewness', 'kurtosis', 'iqr'])

    for col in df.columns:
        data = df[col].dropna()
        dtype_kind = data.dtype.kind

        # 更新报告数据
        report.loc[col, 'dtype'] = dtype_kind
        report.loc[col, 'unique'] = data.nunique()
        report.loc[col, 'missing'] = df[col].isna().sum()

        if dtype_kind in 'fiu':
            report.loc[col, 'skewness'] = data.skew()
            report.loc[col, 'kurtosis'] = data.kurtosis()
            q1, q3 = np.percentile(data, [25, 75])
            report.loc[col, 'iqr'] = q3 - q1

        # ======================
        # 新建图表布局
        # ======================
        fig = plt.figure(figsize=(6, 4.5))
        gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)

        # 箱线图 (上方)
        ax_box = plt.subplot(gs[0])
        sns.boxplot(x=data, ax=ax_box, color=COLOR_BOX, width=0.4,
                    linewidth=1, flierprops=dict(marker='o', markersize=3))

        # 隐藏x轴标签
        ax_box.set(xlabel='', yticks=[])
        plt.setp(ax_box.get_xticklabels(), visible=False)

        # 直方图 (下方)
        ax_hist = plt.subplot(gs[1], sharex=ax_box)
        sns.histplot(data, kde=True, ax=ax_hist, color=COLOR_HIST,
                     edgecolor='white', linewidth=0.5, bins='auto')

        # ======================
        # 添加统计信息标注
        # ======================
        stats_text = (
            f'Skewness: {data.skew():.2f}\n'
            f'Kurtosis: {data.kurtosis():.2f}\n'
            f'n = {len(data):,}'
        )
        ax_hist.text(0.98, 0.85, stats_text,
                     transform=ax_hist.transAxes,
                     ha='right', va='top',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # ======================
        # 统一格式设置
        # ======================
        # 主标题
        fig.suptitle(f'Distribution Analysis: {col}', y=0.95,
                     fontsize=12, fontweight='bold')

        # 坐标轴标签
        ax_hist.set_xlabel('Value', labelpad=5, fontweight='semibold')
        ax_hist.set_ylabel('Frequency', labelpad=5, fontweight='semibold')

        # 网格线设置
        ax_hist.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax_hist.grid(True, axis='x', linestyle=':', alpha=0.2)

        # 调整布局
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}/{col}_distribution.png', dpi=300, bbox_inches='tight')
        #plt.show()

    return report


# ======================
# 执行主程序
# ======================
if __name__ == '__main__':
    df = load_data(r"E:\物理模型结合机器学习\data1_20250324_泛化集.xlsx")
    report = analyze_distributions(df)
    report.to_csv('./generation/distribution_report.csv', encoding='utf-8-sig')
    print("分析完成！请查看生成的报告和可视化图形。")