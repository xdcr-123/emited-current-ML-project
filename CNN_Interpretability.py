import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import shap
from itertools import combinations


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})
sns.set_style("whitegrid",{'font.family': 'Times New Roman'})
#plt.rc('font', family='Times New Roman')  # 强制全局设置

# 加载模型和标准化器
model = load_model('CNN_transfer_learning_model.h5', custom_objects={'mse': metrics.MeanSquaredError})  # 加载训练好的模型
scaler = joblib.load('CNN_transfer_scaler.joblib')  # 加载标准化器

# 准备数据
data = pd.read_excel("Dataset_combined.xlsx")   #用合并后的数据分析
X = data.iloc[:, :10].values
y = data.iloc[:, -2:].values
feature_names = data.columns[:10]
X_scaled = scaler.transform(X)
X_reshaped = X_scaled.reshape(-1, 1, 10, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)


# 第i个目标变量（索引从0开始）,控制输出变量
output_index = 0

#============================================
# SHAP分析
#============================================
'''shap.initjs()
background = X_train[:1000]
explainer = shap.DeepExplainer(model, background)
n_samples = 200  # 分析前200个测试样本
shap_values = explainer.shap_values(X_test[:n_samples], check_additivity=False)
print('shap_values的形状为：', shap_values[:,:,:,:,:].shape)  #（样本数200,高度1,宽度10,通道数1,目标2）
# 提取第i(0-1)个输出变量的SHAP值
shap_values_output = np.array(shap_values[:,:,:,:,output_index]).squeeze()  #压缩形状n_samples*特征数10, 控制输出量
print('shap_values_output的形状为：',shap_values_output.shape)
# 获取对应的原始特征值
X_test_reshaped = X_test[:n_samples].reshape(-1, 10)
X_test_original = scaler.inverse_transform(X_test_reshaped)

# ========== SHAP散点图==========
plt.figure(figsize=(20, 12), dpi=300)

# 创建渐变色映射
cmap = plt.colormaps['viridis']
colors = cmap(np.linspace(0, 1, n_samples))

for j, feature in enumerate(feature_names):
    ax = plt.subplot(2, 5, j + 1)

    # 使用渐变色散点图
    scatter = ax.scatter(X_test_original[:, j], shap_values_output[:, j],
                         c=X_test_original[:, j], cmap='viridis',
                         alpha=0.8, edgecolor='w', linewidth=0.5,
                         s=50, zorder=2)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Feature Value', fontsize=20,labelpad=0)
    cbar.ax.tick_params(labelsize=20, pad=0)

    # 趋势线
    sns.regplot(x=X_test_original[:, j], y=shap_values_output[:, j],
                scatter=False, ci=95,  #95%的置信区间
                line_kws={'color': '#d62728', 'lw': 1.5, 'alpha': 0.9},
                truncate=True)

    # 坐标轴
    ax.set_xlabel(f'{feature}', fontsize=20, labelpad=0)
    ax.set_ylabel('SHAP Value', fontsize=20, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=20, pad=0)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    # 统计信息
    corr = np.corrcoef(X_test_original[:, j], shap_values_output[:, j])[0, 1]
    ax.text(0.97, 0.10, f'ρ = {corr:.2f}',
            transform=ax.transAxes, ha='right',
            fontsize=20,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# 标题和布局调整
plt.suptitle('SHAP Value Relationships with Feature Values (emission current)',
             y=1.0, fontsize=24, fontweight='semibold')
plt.tight_layout(pad=2, h_pad=1, w_pad=1)  # 增加子图间距
plt.subplots_adjust(bottom=0.15, top=0.95,left=0.05,right=0.95)
plt.savefig('CNN_shap_scatterplots_Je_500.png', bbox_inches='tight', dpi=300)'''
#plt.show()

# ========== SHAP Beeswarm图 ==========
# 转换数据格式
'''expected_value_np = explainer.expected_value[output_index].numpy()  #第i个输出
X_test_original_np = np.array(X_test_original)

# 创建Explanation对象
shap_exp = shap.Explanation(
    values=shap_values_output,
    base_values=expected_value_np,
    data=X_test_original_np,
    feature_names=feature_names.tolist(),
    output_names=["Target1"]
)

# 配置绘图参数
plt.figure(figsize=(10, 6), dpi=300)

# 绘制Beeswarm图
shap.plots.beeswarm(
    shap_exp,
    max_display=10,
    color=plt.colormaps['viridis'],
    axis_color="#333333",
    alpha=0.8,
    show=False
)

# 增强可视化效果
ax = plt.gca()

# 其他样式设置
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True, which='major', linestyle='--', alpha=0.3)

plt.title("SHAP Feature Importance Summary (emission current)",
         fontsize=12, pad=15, fontweight='semibold')
plt.xlabel("SHAP Value Impact", fontsize=12, labelpad=10)
plt.ylabel("Features", fontsize=12, labelpad=10)

# 优化布局并保存
plt.tight_layout()
plt.savefig('CNN_SHAP_beeswarm_Je.png',
           bbox_inches='tight',
           dpi=300,
           transparent=False)
#plt.show()'''

#============================================
# 置换重要性计算（全局分析）
#============================================
'''def calc_permutation_importance(model, X, y, n_repeats=5):
    y_pred = model.predict(X)
    baseline = mean_squared_error(y, y_pred[:, output_index])
    importance = np.zeros(X.shape[2])  # 获取特征数量

    for i in range(X.shape[2]):
        X_perturbed = X.copy()
        # 打乱特征时保持其他维度
        X_perturbed[:, 0, i, 0] = np.random.permutation(X_perturbed[:, 0, i, 0])
        y_pred_perturbed = model.predict(X_perturbed)
        importance[i] = baseline - mean_squared_error(y, y_pred_perturbed[:, output_index])

    return importance


importance = calc_permutation_importance(model, X_test, y_test[:, output_index])

# ========== 置换重要性 ==========
plt.figure(figsize=(5, 4), dpi=300)
# 创建颜色渐变
norm = plt.Normalize(importance.min(), importance.max())
colors = plt.cm.viridis(norm(importance))
sorted_idx = importance.argsort()
plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color=colors)
plt.title('Permutation Feature Importance (emission current)',
          fontsize=16, pad=15, fontweight='semibold')
plt.xlabel('Mean Squared Error Difference', fontsize=16, labelpad=8)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('CNN_permutation_importance_Je.png', bbox_inches='tight', dpi=300)'''
#plt.show()


#============================================
# 单样本梯度分析对照（局部分析）
#============================================
sample_indices = [99, 199, 299, 399, 499]  # 第20,40,60,80,100个样本
# 创建画布和子图（5行1列）
fig, axes = plt.subplots(5, 1, figsize=(12, 12), dpi=300)
plt.subplots_adjust(hspace=0)  # 调整子图间距

for idx, (sample_index, ax) in enumerate(zip(sample_indices, axes)):
    
    sample = tf.convert_to_tensor(X_test[sample_index:sample_index + 1], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(sample)
        prediction = model(sample)
        output_value = prediction[0, output_index]

    # 计算梯度
    grads = tape.gradient(output_value, sample)

    if grads is None:
        print(f"无法可视化：梯度为None，样本索引 {sample_index}")
        continue

    # 提取并处理梯度数据
    hm_data = grads.numpy()[0, 0, :, 0]
    df_grad = pd.DataFrame(hm_data.reshape(1, -1),
                          columns=feature_names,
                          index=[f"Sample #{sample_index + 1}"])

    # 创建对称颜色归一化
    vmax = np.abs(hm_data).max()
    norm = plt.Normalize(-vmax, vmax)

    # 热力图配置
    ax = sns.heatmap(
        df_grad,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        annot_kws={
            'fontsize': 16,
            'fontweight': 'bold',
            'color': "black"
        },
        cbar_kws={
            'label': 'Gradient Magnitude',
            'shrink': 0.6,
            'aspect': 10,
            'ticks': np.linspace(-vmax, vmax, 5)
        },
        center=20,
        norm=norm,
        square=True,
        linewidths=0.5,
        linecolor='whitesmoke',
        xticklabels=feature_names if idx == 4 else [],  # 仅最后子图显示特征名
        yticklabels=False,
        ax=ax
    )

    # 颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Gradient Magnitude', fontsize=14)

    # 标题与标注
    prediction_value = prediction[0, output_index].numpy()
    ax.set_title(
        f"Emission Current | Pred: {prediction_value:.2f} | Sample: {sample_index + 1}",
        fontsize=16,
        pad=12,
        loc='left'
    )

    # X轴标签（仅最后一个子图）
    if idx == 4:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='right',
            fontsize=20,
            fontstyle='italic',
            rotation_mode='anchor'
        )
        ax.xaxis.set_ticks_position('bottom')
        # 更改坐标轴刻度的长短
        ax.tick_params(axis='x', length=30)
    else:
        ax.xaxis.set_ticks_position('none')

    # 添加统计信息
    gradient_stats = f"Max: {hm_data.max():.2f} | Min: {hm_data.min():.2f} | Mean: {hm_data.mean():.2f}"
    ax.text(
        0.5, -0.35,
        gradient_stats,
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.5,
            edgecolor='lightgray'
        )
    )

# 保存输出
'''plt.savefig('CNN_multisamples_gradient_analysis_Je_50.png',
            bbox_inches='tight',
            dpi=300,
            transparent=False)'''
plt.show()

# ============================================
# 部分依赖图 (Partial Dependence Plots)
# ============================================
# 获取训练集原始数据
'''X_train_reshaped_2d = X_train.reshape(-1, 10)
X_train_original = scaler.inverse_transform(X_train_reshaped_2d)

def partial_dependence_1d(model, scaler, feature_idx, X_train_original, output_index, n_grid=500):
    """计算一维部分依赖"""
    # 计算特征范围
    feat_min = X_train_original[:, feature_idx].min()
    feat_max = X_train_original[:, feature_idx].max()

    # 生成网格并标准化
    grid_original = np.linspace(feat_min, feat_max, n_grid)
    grid_scaled = (grid_original - scaler.mean_[feature_idx]) / scaler.scale_[feature_idx]

    # 构建样本
    samples = np.zeros((n_grid, 10))
    samples[:, feature_idx] = grid_scaled

    # 预测（确保输入形状与模型匹配）
    preds = model.predict(samples.reshape(-1, 1, 10, 1))
    pdp = preds[:, output_index]  # 直接使用每个网格点的预测值

    return grid_original, pdp


def partial_dependence_2d(model, scaler, feature_indices, X_train_original, output_index, n_grid=500):
    """计算二维部分依赖"""
    fi, fj = feature_indices

    # 生成双特征网格
    grid_i = np.linspace(X_train_original[:, fi].min(), X_train_original[:, fi].max(), n_grid)
    grid_j = np.linspace(X_train_original[:, fj].min(), X_train_original[:, fj].max(), n_grid)

    # 标准化处理
    gi_scaled = (grid_i - scaler.mean_[fi]) / scaler.scale_[fi]
    gj_scaled = (grid_j - scaler.mean_[fj]) / scaler.scale_[fj]

    # 创建网格样本
    xx, yy = np.meshgrid(gi_scaled, gj_scaled)
    samples = np.zeros((n_grid ** 2, 10))
    samples[:, fi] = xx.ravel()
    samples[:, fj] = yy.ravel()

    # 预测并重塑形状
    preds = model.predict(samples.reshape(-1, 1, 10, 1))
    pdp = preds[:, output_index].reshape(n_grid, n_grid)  # 直接重塑为网格形状

    return grid_i, grid_j, pdp


# ========== 一维部分依赖图 ==========
plt.figure(figsize=(20, 16), dpi=300)
for j, feature in enumerate(feature_names):
    ax = plt.subplot(2, 5, j + 1)

    # 计算PDP
    grid, pdp = partial_dependence_1d(model, scaler, j, X_train_original, output_index)

    # 可视化设置
    sns.lineplot(x=grid, y=pdp, color="#2b83ba", lw=2, ax=ax)
    ax.fill_between(grid, pdp, alpha=0.2, color="#2b83ba")

    # 样式增强
    ax.set_xlabel(feature, fontsize=20, labelpad=2)
    ax.set_ylabel("PDP", fontsize=20, labelpad=2)
    ax.tick_params(axis='both', labelsize=20, pad=1)
    ax.grid(True, linestyle=':', alpha=0.3)

    # 标注特征范围
    ax.text(0.98, 0.95, f"[{grid.min():.1f}, {grid.max():.1f}]",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=20, bbox=dict(facecolor='white', alpha=0.8))

plt.suptitle(f"1D Partial Dependence Plots (emission current)",
             y=1.02, fontsize=24, fontweight='semibold')
plt.tight_layout(pad=2.5, h_pad=3, w_pad=3)
plt.savefig('CNN_pdp_1d_Je_500.png', bbox_inches='tight', dpi=300)
#plt.show()

# ========== 二维部分依赖图 ==========
# 选择重要性最高的前4个特征（置换重要性按负值排序，取最小的4个）
sorted_idx = np.argsort(importance)  # 置换重要性负值越大越重要，直接升序排列
top_features = sorted_idx[:4]

# 生成所有两两组合
feature_pairs = list(combinations(top_features, 2))
n_pairs = len(feature_pairs)

# 创建子图画布
plt.figure(figsize=(20, 15), dpi=300)
plt.subplots_adjust(hspace=0.3, wspace=0.5)

# 遍历所有特征对
for i, (feat1, feat2) in enumerate(feature_pairs, 1):
    ax = plt.subplot(2, 3, i)  # 2行3列布局

    # 计算二维PDP
    grid_x, grid_y, pdp_2d = partial_dependence_2d(
        model, scaler, [feat1, feat2], X_train_original, output_index, n_grid=30
    )

    # 创建网格
    XX, YY = np.meshgrid(grid_x, grid_y)

    # 绘制等高线图
    contour = ax.contourf(XX, YY, pdp_2d, levels=15, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax, label='PDP', shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('PDP', size=20)

    # 设置坐标标签
    ax.set_xlabel(feature_names[feat1], fontsize=20, labelpad=5)
    ax.set_ylabel(feature_names[feat2], fontsize=20, labelpad=5)
    ax.tick_params(axis='both', labelsize=20)

    # 添加统计信息
    stats_text = (
        f"Max: {pdp_2d.max():.2f}\n"
        f"Min: {pdp_2d.min():.2f}\n"
        f"Range: {pdp_2d.ptp():.2f}"
    )
    ax.text(0.98, 0.15, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=20,
            bbox=dict(facecolor='white', alpha=0.8))

    # 设置子图标题
    ax.set_title(f"PDP: {feature_names[feat1]} vs {feature_names[feat2]}",
                 fontsize=20, pad=10)

# 添加总标题
plt.suptitle("2D Partial Dependence Plots for Top 4 Important Features (emission current)",
             fontsize=24, y=0.98,  fontweight='semibold')

plt.savefig('CNN_pdp_2d_Je_500.png', bbox_inches='tight', dpi=300)'''
#plt.show()