import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization

# ======================
# 1. 加载预训练模型和归一化器
# ======================
pretrained_model = load_model('CNN_model_smallsize.h5', custom_objects={'mse': metrics.MeanSquaredError})
#pretrained_scaler = joblib.load('CNN_scaler_small_size.joblib')

# ======================
# 2. 加载并预处理新数据
# ======================
# 加载原始数据和新数据
old_data = pd.read_excel(r"E:\物理模型结合机器学习\dataset_smallsize.xlsx")
new_data = pd.read_excel(r"E:\物理模型结合机器学习\dataset_largesize.xlsx")

# 合并数据集
combined_data = pd.concat([old_data, new_data], axis=0)
# 将合并后的数据保存到Excel文件
'''combined_data.to_excel(r"Dataset_combined.xlsx",
                      index=False,  # 不保存行索引
                      sheet_name='Combined Data')  # 设置工作表名称

print("数据已成功保存到 combined_dataset.xlsx")'''

# 分割特征和目标
X_combined = combined_data.iloc[:, :10].values
y_combined = combined_data.iloc[:, -2:].values

# 创建新的归一化器预处理特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)  # 新的数据分布，新的归一化

# 重塑数据为CNN输入格式
X_reshaped = X_scaled.reshape(-1, 1, 10, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_combined,
    test_size=0.2,
    random_state=42
)

# ======================
# 3. 迁移学习模型配置
# ======================
# 冻结所有层进行微调
for layer in pretrained_model.layers:
    layer.trainable = False

# 仅解冻输出层（最后一个Dense层）
#pretrained_model.layers[-1].trainable = True
# 解冻所有Dense层（包括输出层）
for layer in pretrained_model.layers:
    if isinstance(layer, Dense):
        layer.trainable = True

# 使用学习率调度器（提升训练稳定性）
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0059,
    decay_steps=1000,
    decay_rate=0.9)#0.9


# 使用更小的学习率进行微调
custom_adam = Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# 编译模型（添加MAE指标监控）
pretrained_model.compile(
    optimizer=custom_adam,
    loss='mse',
    metrics=['mae']  # 添加平均绝对误差指标
)

# 打印模型结构
pretrained_model.summary()

#早停
early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证集的损失
    patience=20,  # 当验证损失在 20 个 epochs 内没有改善时停止
    restore_best_weights=True  # 停止时恢复最佳的权重
)

history = pretrained_model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=136,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping]
)

# ======================
# 5. 模型评估与保存
# ======================
# 保存完整模型
pretrained_model.save('CNN_transfer_learning_model.h5')
print("已保存完整迁移学习模型：CNN_transfer_learning_model.h5")

# 保存模型权重
pretrained_model.save_weights('CNN_transfer_weights.weights.h5')
print("已保存迁移学习权重：CNN_transfer_weights.weights.h5")

# 保存归一化器
joblib.dump(scaler, 'CNN_transfer_scaler.joblib')
print("已保存归一化器：CNN_transfer_scaler.joblib")

def evaluate_model(X, y_true, dataset_type):
    y_pred = pretrained_model.predict(X)

    # 为每个目标计算指标
    results = {}
    for target in [0, 1]:
        mse = mean_squared_error(y_true[:, target], y_pred[:, target])
        mae = mean_absolute_error(y_true[:, target], y_pred[:, target])
        r2 = r2_score(y_true[:, target], y_pred[:, target])

        results[f'Target {target + 1}'] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

    # 打印结果
    print(f"\n{dataset_type} Evaluation:")
    for target, metrics in results.items():
        print(f"【{target}】")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"R2:  {metrics['R2']:.4f}")
        print("-" * 30)

    return y_pred, results  # 修改返回值为预测值和指标字典

# 评估训练集和测试集
print("\n" + "=" * 50)
train_pred, train_Loss_results = evaluate_model(X_train, y_train, "Training Set")
print("\n" + "=" * 50)
test_pred, test_Loss_results = evaluate_model(X_test, y_test, "Test Set")

# ======================
# 6. 可视化训练过程
# ======================
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")
plt.rcParams.update({
    'font.family': 'serif',  # 正式字体
    'font.serif': 'Times New Roman',
    'axes.labelsize': 12,  # 统一标签大小
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4
})

# ======== 1. 损失曲线 ========
fig1, ax1 = plt.subplots(figsize=(6, 5))

# 使用渐变色填充区间
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

ax1.plot(epochs, train_loss,
         color='#1f77b4',
         linewidth=2,
         marker='o',
         markersize=4,
         markevery=int(len(epochs) / 10),
         label='Training Loss')
ax1.plot(epochs, val_loss,
         color='#ff7f0e',
         linewidth=2,
         linestyle='--',
         marker='s',
         markersize=4,
         markevery=int(len(epochs) / 10),
         label='Validation Loss')

# 添加辅助元素
ax1.set_title('Transfer Learning Training Process', pad=20)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(visible=True, which='both', linestyle='--', alpha=0.3)
ax1.legend(frameon=True, shadow=True, fancybox=True)

# ======== 2. 预测效果可视化 ========
def create_prediction_plot(ax, y_true, y_pred, target_name, color, r2, mse, mae):
    # 传入指标
    metrics_text = f'$R^2 = {r2:.3f}$\n$MSE = {mse:.3f}$\n$MAE = {mae:.3f}$'

    # 散点图增强
    sc = ax.scatter(y_true, y_pred,
                    c=color,
                    alpha=0.6,
                    edgecolors='w',
                    linewidths=0.3,
                    s=60,
                    zorder=2)

    # 展平数据
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算趋势线
    coef = np.polyfit(y_true_flat, y_pred_flat, 1)
    trend_line = np.poly1d(coef)
    x_vals = np.array([y_true_flat.min(), y_true_flat.max()])
    x_range = np.linspace(x_vals[0], x_vals[1], 100)

    # 计算置信区间
    '''n = len(y_true_flat)
    if n > 2:
        residuals = y_pred_flat - trend_line(y_true_flat)
        mse_resid = np.sum(residuals ** 2) / (n - 2)
        se = np.sqrt(mse_resid)
        x_mean = np.mean(y_true_flat)
        Sxx = np.sum((y_true_flat - x_mean) ** 2)
        std_err = se * np.sqrt(1 / n + (x_range - x_mean) ** 2 / Sxx)
        t_val = stats.t.ppf(0.975, n - 2)
        conf_interval = t_val * std_err
        ax.fill_between(x_range,
                        trend_line(x_range) - conf_interval,
                        trend_line(x_range) + conf_interval,
                        color='#d62728', alpha=0.2, label='95% CI')'''

    # 绘制趋势线
    ax.plot(x_vals, trend_line(x_vals),
            color='#d62728',
            linewidth=2,
            zorder=3,
            label='Linear Fit')

    # 绘制y=x参考线
    ax.plot(x_vals, x_vals,
            color='black',
            linestyle='--',
            linewidth=1.5,
            zorder=1,
            label='y=x')

    # 添加指标文本
    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.8,
                      edgecolor='lightgray'))

    # 坐标轴设置
    ax.set_xlabel(f'True Values - {target_name}', labelpad=10)
    ax.set_ylabel(f'Predicted Values - {target_name}', labelpad=10)
    ax.set_title(f'{target_name} Prediction', pad=15)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_aspect('equal', adjustable='box')

    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.80),
              frameon=True, edgecolor='black')

# 绘制子图时传入预计算的指标
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'wspace': 0.25})

# 目标1的指标获取
target1_metrics = test_Loss_results['Target 1']
create_prediction_plot(ax2,
                       y_test[:, 0], test_pred[:, 0],
                       'emission current', '#1f77b4',
                       r2=target1_metrics['R2'],
                       mse=target1_metrics['MSE'],
                       mae=target1_metrics['MAE'])

# 目标2的指标获取
target2_metrics = test_Loss_results['Target 2']
create_prediction_plot(ax3,
                       y_test[:, 1], test_pred[:, 1],
                       'collection current', '#ff7f0e',
                       r2=target2_metrics['R2'],
                       mse=target2_metrics['MSE'],
                       mae=target2_metrics['MAE'])

# ======== 保存设置 ========
fig1.savefig('Transfer_training_loss_largesize.png', dpi=300, bbox_inches='tight')
fig2.savefig('Transfer_测试集散点图_largelsize.png', dpi=300, bbox_inches='tight')
plt.show()