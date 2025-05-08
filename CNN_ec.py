import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import scipy.stats as stats


# ======================
# 1. 数据读取与预处理
# ======================
data = pd.read_excel(r"E:\物理模型结合机器学习\dataset_smallsize.xlsx")

# 分割特征和目标
X = data.iloc[:, :10].values  # 前10列是特征
y = data.iloc[:, -2:].values  # 最后2列是目标

# 数据标准化（仅对特征）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 重塑数据为图像格式：(样本数, 高度=1, 宽度=10, 通道=1)
X_reshaped = X_scaled.reshape(-1, 1, 10, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

# ======================
# 2. 构建CNN模型
# ======================
'''model = Sequential([
    # 第一个卷积层（移除激活函数，添加BN后再激活）
    Conv2D(512, (1, 3), use_bias=False, input_shape=(1, 10, 1)),  # 输入形状需根据数据格式确认
    BatchNormalization(),
    Activation('relu'),
    # 池化层
    AveragePooling2D((1, 2)),
    # 第二个卷积层
    Conv2D(512, (1, 2), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    # 展平
    Flatten(),
    # 全连接层
    Dense(256, use_bias=False),  # 添加隐藏层增强非线性,512
    BatchNormalization(),
    Activation('relu'),
    Dense(128, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    # 输出层
    Dense(2)
])'''
model = Sequential([
    # 第一个卷积层（使用1x3的卷积核，相当于每次处理3个连续特征）,处理完后宽度为8,512
    Conv2D(490, (1,3), activation='relu', input_shape=(1, 10, 1)),

    # 池化层（窗口大小1x2，每次压缩宽度维度），处理完宽度为4
    AveragePooling2D((1, 2)),

    # 第二个卷积层（使用更小的1x2卷积核），处理完宽度为3,512
    Conv2D(332, (1, 2), activation='relu'),

    # 展平后接全连接层
    Flatten(),
    Dense(96, activation='relu'),  # 添加隐藏层增强非线性,512
    Dense(48, activation='relu'),  # 添加隐藏层增强非线性128
    Dense(24, activation='relu'),  # 添加隐藏层增强非线性64
    #Dense(64, activation='relu'),  # 添加隐藏层增强非线性64
    #Dense(64, activation='relu'),  # 添加隐藏层增强非线性64
    # 输出层
    Dense(2)
])

# 使用学习率调度器（提升训练稳定性）
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0059,
    decay_steps=1000,
    decay_rate=0.9)#0.9

# 编译模型（添加MAE指标监控）
model.compile(
    optimizer=Adam(learning_ratezai=lr_schedule),
    loss='mse',
    metrics=['mae']  # 添加平均绝对误差指标
)

# 打印模型结构
model.summary()

#早停
early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证集的损失
    patience=20,  # 当验证损失在 20 个 epochs 内没有改善时停止
    restore_best_weights=True  # 停止时恢复最佳的权重
)

# ======================
# 3. 训练模型
# ======================
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=136,#128
    validation_split=0.2,#0.2
    verbose=1,
    callbacks=[early_stopping]
)


# ======================
# 4. 评估模型
# ======================
def evaluate_model(X, y_true, dataset_type):
    y_pred = model.predict(X)

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
# 5. 可视化结果
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
ax1.set_title('Small Size Model Pretraining Process', pad=20)
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
fig1.savefig('CNN_training_loss_smallsize.png', dpi=300, bbox_inches='tight')
fig2.savefig('CNN_测试集散点图_smallsize.png', dpi=300, bbox_inches='tight')


# 保存训练好的模型（HDF5格式）
'''model.save('CNN_model_smallsize.h5')
model.save_weights('CNN_weights_smallsize.weights.h5')
# 保存标准化器
joblib.dump(scaler, 'CNN_scaler_smallsize.joblib')'''
plt.show()