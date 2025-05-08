'''import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from keras.layers import Dense

# ======================
# 1. 加载预训练模型
# ======================
pretrained_model_path = 'CNN_model_smallsize.h5'

# ======================
# 2. 加载数据
# ======================
old_data = pd.read_excel(r"E:\物理模型结合机器学习\dataset_smallsize.xlsx")
new_data = pd.read_excel(r"E:\物理模型结合机器学习\dataset_largesize.xlsx")

# 打乱新数据并初始化参数
new_data_shuffled = new_data.sample(frac=1, random_state=42).reset_index(drop=True)
total_new_samples = len(new_data_shuffled)
newdata_batch_size = 50
num_batches = (total_new_samples + newdata_batch_size - 1) // newdata_batch_size

# 初始化累积数据和结果存储
accumulated_new_data = pd.DataFrame()
all_results = []

# ======================
# 3. 逐步迁移学习
# ======================
for batch_idx in range(num_batches):
    print(f"\n=== 当前进度: 第 {batch_idx + 1}/{num_batches} 次抽样 ===")
    print(f"已使用新数据样本数: {len(accumulated_new_data)}/{len(new_data)}")

    # 合并数据
    current_batch = new_data_shuffled.iloc[batch_idx * newdata_batch_size: (batch_idx + 1) * newdata_batch_size]
    accumulated_new_data = pd.concat([accumulated_new_data, current_batch], ignore_index=True)
    combined_data = pd.concat([old_data, accumulated_new_data], axis=0)

    # 数据预处理
    X_combined = combined_data.iloc[:, :10].values
    y_combined = combined_data.iloc[:, -2:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    X_reshaped = X_scaled.reshape(-1, 1, 10, 1)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_combined,
        test_size=0.2,
        random_state=42
    )

    # ======================
    # 4. 模型配置
    # ======================
    # 重新加载预训练模型
    model = load_model(pretrained_model_path, custom_objects={'mse': metrics.MeanSquaredError})

    # 冻结所有层，解冻Dense层
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True

    # 学习率调度器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0059,
        decay_steps=1000,
        decay_rate=0.9
    )

    # 模型编译
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae']
    )

    # 早停设置
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # ======================
    # 5. 模型训练
    # ======================
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=136,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )


    # ======================
    # 6. 模型评估
    # ======================
    def evaluate(X, y, name):
        y_pred = model.predict(X)
        results = {}
        for target in [0, 1]:
            results[f'Target{target + 1}'] = {
                'MSE': mean_squared_error(y[:, target], y_pred[:, target]),
                'MAE': mean_absolute_error(y[:, target], y_pred[:, target]),
                'R2': r2_score(y[:, target], y_pred[:, target])
            }
        print(f"\n{name}评估结果:")
        for target, metrics in results.items():
            print(f"{target}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
        return results


    train_results = evaluate(X_train, y_train, "训练集")
    test_results = evaluate(X_test, y_test, "测试集")

    # 存储结果
    all_results.append({
        'batch': batch_idx + 1,
        'total_samples': len(combined_data),
        'train': train_results,
        'test': test_results
    })

# ======================
# 7. 结果可视化
# ======================
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 设置科研绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 全局字体设置
    'font.size': 12,                   # 基础字号
    'axes.labelsize': 14,             # 坐标轴标签字号
    'axes.titlesize': 16,             # 标题字号
    'xtick.labelsize': 12,            # X轴刻度字号
    'ytick.labelsize': 12,            # Y轴刻度字号
    'legend.fontsize': 12,            # 图例字号
    'legend.frameon': True,           # 图例边框
    'legend.framealpha': 0.8,         # 图例透明度
    'axes.linewidth': 1.5,            # 坐标轴线宽
    'lines.linewidth': 2,             # 数据线宽
    'lines.markersize': 8,            # 标记尺寸
    'xtick.direction': 'in',          # 刻度线方向
    'ytick.direction': 'in',          # 刻度线方向
    'xtick.major.size': 6,            # 主刻度长度
    'ytick.major.size': 6,            # 主刻度长度
    'xtick.minor.size': 3,            # 次刻度长度
    'ytick.minor.size': 3             # 次刻度长度
})

# 准备绘图数据
batches = [res['batch'] for res in all_results]
test_mse_1 = [res['test']['Target1']['MSE'] for res in all_results]
test_mse_2 = [res['test']['Target2']['MSE'] for res in all_results]

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制趋势线
line1, = ax.plot(batches, test_mse_1,
                color='#1f77b4',
                marker='o',
                linestyle='-',
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)

line2, = ax.plot(batches, test_mse_2,
                color='#d62728',
                marker='s',
                linestyle='--',
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)

# 设置坐标轴
ax.set_xlabel('Training Iterations', labelpad=10)
ax.set_ylabel('Test MSE', labelpad=10)
ax.set_title('Model Performance Progression with Increasing Large Size Data', pad=20)

# 设置刻度
ax.set_xticks(batches)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# 设置网格
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# 添加图例
legend = ax.legend(
    handles=[line1, line2],
    labels=['Emission Current', 'Collection Current'],
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    ncol=1,
    shadow=False,
    edgecolor='black',
    title='Current Type:'
)
legend.get_title().set_fontsize('12')

# 优化布局
plt.tight_layout(pad=2.0)

plt.savefig('mse_progression.png',  dpi=300, bbox_inches='tight')
plt.show()'''

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import time

# ======================
# 1. 初始化配置
# ======================
# 文件路径配置
pretrained_model_path = 'CNN_model_smallsize.h5'
old_data_path = r"E:\物理模型结合机器学习\dataset_smallsize.xlsx"
new_data_path = r"E:\物理模型结合机器学习\dataset_largesize.xlsx"
generalization_data_path = r"E:\物理模型结合机器学习\data1_20250324_泛化集.xlsx"

# 训练参数
newdata_batch_size = 50
train_batch_size = 136
initial_learning_rate = 0.0059
epochs = 300

# ======================
# 2. 加载数据
# ======================
old_data = pd.read_excel(old_data_path)
new_data = pd.read_excel(new_data_path).sample(frac=1, random_state=42)
generalization_data = pd.read_excel(generalization_data_path)

# 准备泛化集
X_gen = generalization_data.iloc[:, :10].values
y_gen = generalization_data.iloc[:, -2:].values

# ======================
# 3. 初始化记录
# ======================
results = []
total_batches = (len(new_data) + newdata_batch_size - 1) // newdata_batch_size

# ======================
# 4. 主训练循环
# ======================
accumulated_new = pd.DataFrame()
current_model = load_model(pretrained_model_path, custom_objects={'mse': metrics.MeanSquaredError})

for batch_idx in range(total_batches):
    print(f"\n=== 第 {batch_idx + 1}/{total_batches} 次迭代 ===")

    # 4.1 合并新数据
    batch_data = new_data.iloc[batch_idx * newdata_batch_size: (batch_idx + 1) * newdata_batch_size]
    accumulated_new = pd.concat([accumulated_new, batch_data], ignore_index=True)
    combined_data = pd.concat([old_data, accumulated_new], axis=0)

    # 4.2 数据预处理
    X = combined_data.iloc[:, :10].values
    y = combined_data.iloc[:, -2:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(-1, 1, 10, 1)

    # 4.3 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42
    )

    # 4.4 配置迁移学习模型
    for layer in current_model.layers:
        layer.trainable = False
    for layer in current_model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )

    current_model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae']
    )

    # 4.5 模型训练
    history = current_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=train_batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20)]
    )

    # 4.6 测试集评估
    test_pred = current_model.predict(X_test)
    test_mse = [
        mean_squared_error(y_test[:, 0], test_pred[:, 0]),  # Target1
        mean_squared_error(y_test[:, 1], test_pred[:, 1])  # Target2
    ]

    # 4.7 泛化集预测
    X_gen_scaled = scaler.transform(X_gen)  # 使用当前归一化器
    X_gen_reshaped = X_gen_scaled.reshape(-1, 1, 10, 1)
    start_time = time.time()
    gen_pred = current_model.predict(X_gen_reshaped)
    inference_time = time.time() - start_time

    gen_mse = [
        mean_squared_error(y_gen[:, 0], gen_pred[:, 0]),
        mean_squared_error(y_gen[:, 1], gen_pred[:, 1])
    ]

    # 4.8 记录结果
    results.append({
        'batch': batch_idx + 1,
        'total_samples': len(combined_data),
        'test_mse': test_mse,
        'gen_mse': gen_mse,
        'inference_time': inference_time
    })
    print(f"泛化集推理时间: {inference_time:.4f}s")
    print(f"测试集MSE: {test_mse} | 泛化集MSE: {gen_mse}")

# ======================
# 5. 结果可视化
# ======================
# 设置绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 全局字体设置
    'font.size': 12,                   # 基础字号
    'axes.labelsize': 14,             # 坐标轴标签字号
    'axes.titlesize': 16,             # 标题字号
    'xtick.labelsize': 12,            # X轴刻度字号
    'ytick.labelsize': 12,            # Y轴刻度字号
    'legend.fontsize': 12,            # 图例字号
    'legend.frameon': True,           # 图例边框
    'legend.framealpha': 0.8,         # 图例透明度
    'axes.linewidth': 1.5,            # 坐标轴线宽
    'lines.linewidth': 2,             # 数据线宽
    'lines.markersize': 8,            # 标记尺寸
    'xtick.direction': 'in',          # 刻度线方向
    'ytick.direction': 'in',          # 刻度线方向
    'xtick.major.size': 6,            # 主刻度长度
    'ytick.major.size': 6,            # 主刻度长度
    'xtick.minor.size': 3,            # 次刻度长度
    'ytick.minor.size': 3             # 次刻度长度
})

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制MSE曲线
batches = [res['batch'] for res in results]
test_mse1 = [res['test_mse'][0] for res in results]
test_mse2 = [res['test_mse'][1] for res in results]
gen_mse1 = [res['gen_mse'][0] for res in results]
gen_mse2 = [res['gen_mse'][1] for res in results]

# 测试集结果（左轴）
ax1.plot(batches, test_mse1, 'o--', color='#1f77b4', label='Test Emission',
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)
ax1.plot(batches, test_mse2, 's--', color='#ff7f0e', label='Test Collection',
                markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)
ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('Test MSE')
ax1.grid(True, linestyle='--', alpha=0.6)

# 泛化集结果（右轴）
ax2 = ax1.twinx()
ax2.plot(batches, gen_mse1, 'o-', color='#2ca02c', label='Gen Emission',markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)
ax2.plot(batches, gen_mse2, 's-', color='#d62728', label='Gen Collection',markersize=8,
                markerfacecolor='white',
                markeredgewidth=1.5)
ax2.set_ylabel('Generalization MSE')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.15),
           ncol=4,
           frameon=True)

plt.title('Model Performance Progression with Incremental Training')
plt.xticks(batches)
plt.tight_layout()
plt.savefig('mse_progression.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 6. 保存最终模型
# ======================
'''current_model.save('final_incremental_model.h5')
joblib.dump(scaler, 'final_incremental_scaler.joblib')
print("训练完成，最终模型已保存！")'''