import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ================= 科研图表全局参数 =================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelcolor': '#363636',
    'figure.dpi': 300,
    'figure.constrained_layout.use': True
})
plt.rc('font', family='Times New Roman')  # 强制全局设置

# ================= 数据预处理 =================
data = pd.read_excel("Dataset_combined_process_corrlation.xlsx").iloc[1:]
df = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)


# ================= 热力图生成 =================
def advanced_correlation_plot(corr_matrix, figsize=(9, 7)):


    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=20)


    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))


    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={
            'size': 12,
            'color': 'black',
            'va': 'center',
            'bbox': dict(boxstyle='round', facecolor='white', alpha=0.4)
        },
        linewidths=0.6,
        linecolor='white',
        cbar_kws={
            'label': 'Correlation Coefficient',
            'shrink': 0.75,
            'ticks': np.linspace(-1, 1, 5),
            'drawedges': False                        # 颜色条描边
        },
        square=True,
        ax=ax
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        fontstyle='italic',
        color='#404040'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontstyle='italic',
        color='#404040'
    )


    ax.set_title(
        'Spearman Correlation Matrix\n',
        fontweight='semibold',
        color='#1A1A1A',
        pad=20
    )


    cbar = heatmap.collections[0].colorbar
    cbar.outline.set_edgecolor('#707070')
    cbar.ax.tick_params(width=0.8, length=3)
    cbar.set_label(
        label='Spearman Correlation',
        weight='semibold',
        labelpad=12
    )

    return fig


if __name__ == "__main__":

    spearman_corr = df.corr(method='spearman')

    fig = advanced_correlation_plot(spearman_corr)

    fig.savefig("Spearman_Correlation_Datacombined.png", dpi=300)

    #plt.show()
