import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import seaborn as sns
from pandas.plotting import parallel_coordinates
import time
from datetime import timedelta

# ======================
# 1. 数据读取与预处理
# ======================
data = pd.read_excel("E:\物理模型结合机器学习\dataset.xlsx")
X = data.iloc[:, :10].values
y = data.iloc[:, -2:].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape(-1, 1, 10, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)


# ======================
# 2. 贝叶斯优化超参数
# ======================
def cnn_bayesian_optimization(conv1_filters, conv1_kernel_size, conv2_filters,
                              conv2_kernel_size, dense_units, learning_rate, batch_size):
    # 参数类型转换
    params = {
        'conv1_filters': int(conv1_filters),
        'conv1_kernel_size': int(conv1_kernel_size),
        'conv2_filters': int(conv2_filters),
        'conv2_kernel_size': int(conv2_kernel_size),
        'dense_units': int(dense_units),
        'learning_rate': max(learning_rate, 1e-5),  # 防止学习率过小
        'batch_size': int(batch_size)
    }

    # 清除之前的计算图
    tf.keras.backend.clear_session()

    # 构建动态模型
    model = Sequential([
        Conv2D(params['conv1_filters'], (1, params['conv1_kernel_size']),
               activation='relu', input_shape=(1, 10, 1)),
        AveragePooling2D((1, 2)),
        Conv2D(params['conv2_filters'], (1, params['conv2_kernel_size']), activation='relu'),
        Flatten(),
        Dense(params['dense_units'], activation='relu'),
        Dense(params['dense_units'] // 2, activation='relu'),
        Dense(params['dense_units'] // 4, activation='relu'),
        Dense(2)
    ])

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )

    # 早停机制
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=params['batch_size'],
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stopping]
    )

    # 返回最佳验证损失（负值用于最大化目标）
    return -min(history.history['val_loss'])


# 定义参数空间
pbounds = {
    'conv1_filters': (64, 512),
    'conv1_kernel_size': (2, 4),
    'conv2_filters': (64, 512),
    'conv2_kernel_size': (2, 3),
    'dense_units': (64, 256),
    'learning_rate': (1e-4, 0.1),
    'batch_size': (64, 256)
}


# 定义贝叶斯优化过程跟踪器
class OptimizationProgressTracker:
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.start_time = time.time()

    def update(self, event, optimizer):
        self.current_iteration += 1
        elapsed_time = time.time() - self.start_time
        avg_time_per_iter = elapsed_time / self.current_iteration
        remaining_time = avg_time_per_iter * (self.total_iterations - self.current_iteration)

        print(f"\n=== 优化进度 [{self.current_iteration}/{self.total_iterations}] ===")
        print(f"已用时: {timedelta(seconds=int(elapsed_time))}")
        print(f"预计剩余时间: {timedelta(seconds=int(remaining_time))}")

        if optimizer.max is not None:
            best_loss = -optimizer.max['target']
            print(f"当前最佳验证损失: {best_loss:.4f}")
            print("最佳参数组合:")
            for key, value in optimizer.max['params'].items():
                if key in ['conv1_kernel_size', 'conv2_kernel_size']:
                    print(f"  {key}: {int(value)}")
                elif key == 'learning_rate':
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {int(value)}")



# 初始化优化器
optimizer = BayesianOptimization(
    f=cnn_bayesian_optimization,
    pbounds=pbounds,
    random_state=42
)

# 设置优化参数
init_points = 5
n_iter = 25
total_iterations = init_points + n_iter

# 初始化进度跟踪器
progress_tracker = OptimizationProgressTracker(total_iterations)

# 打印初始信息
print("==============================================")
print(f"开始贝叶斯优化")
print(f"总迭代次数: {total_iterations} (初始随机 {init_points} 次 + 贝叶斯优化 {n_iter} 次)")
print("==============================================")

# 定义进度回调函数
#def optimization_progress_callback(event, optimizer):
  # progress_tracker.update_progress(optimizer)

# 订阅进度事件
#optimizer.subscribe(Events.OPTIMIZATION_STEP, optimization_progress_callback, callback=None)
optimizer.subscribe(Events.OPTIMIZATION_STEP, progress_tracker)
# 运行优化
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter
)

# 最终结果
print("\n优化完成！最佳参数组合：")
for key, value in optimizer.max['params'].items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {int(value)}")


# ======================
# 4. 可视化
# ======================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

# 转换优化结果为DataFrame
results = []
for res in optimizer.res:
    params = res['params']
    params['target'] = -res['target']  # 转换为实际损失值
    results.append(params)
params_df = pd.DataFrame(results)

# 收敛曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(params_df)+1), params_df['target'],
         marker='o', color='#2C75B6', linewidth=2, markersize=8)
plt.xlabel('Iteration Number', fontweight='bold')
plt.ylabel('Validation Loss (MSE)', fontweight='bold')
plt.title('Bayesian Optimization Convergence', fontweight='bold', pad=20)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig('bayes_convergence.png',dpi=300)
#plt.show()

# 参数与目标值关系
for param in pbounds.keys():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=params_df, x=param, y='target',
                    palette='viridis', alpha=0.8, edgecolor='k', s=80)
    plt.title(f'Parameter Impact: {param}', fontweight='bold', pad=15)
    plt.xlabel(param.capitalize().replace('_', ' '), fontweight='bold')
    plt.ylabel('Validation Loss', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'param_{param}.png',dpi=300)
    #plt.show()

# 平行坐标图
fig, ax = plt.subplots(figsize=(10, 6))
numeric_cols = params_df.columns.drop('target')

# 数据归一化
params_normalized = params_df.copy()
params_normalized[numeric_cols] = (params_df[numeric_cols] - params_df[numeric_cols].min()) / \
                                 (params_df[numeric_cols].max() - params_df[numeric_cols].min())

# 创建颜色映射
colors = plt.cm.viridis((params_normalized['target'] - params_normalized['target'].min()) /
                       (params_normalized['target'].max() - params_normalized['target'].min()))

# 绘制平行坐标
for i in range(len(params_normalized)):
    ax.plot(params_normalized.columns[:-1], params_normalized.iloc[i, :-1],
             color=colors[i], alpha=0.6, linewidth=1.5)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap='viridis',
    norm=plt.Normalize(vmin=params_df['target'].min(), vmax=params_df['target'].max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.05)
cbar.set_label('Validation Loss', fontweight='bold')

ax.set_title('Parallel Coordinates Visualization', fontweight='bold', pad=20)
ax.set_xlabel('Hyperparameters', fontweight='bold')
ax.set_ylabel('Normalized Value', fontweight='bold')
plt.xticks(rotation=45, ha='right')
ax.xaxis.grid(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('parallel_coordinates.png',dpi=300)
#plt.show()