import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

time_start = time.time()  # 记录开始时间

# ======================
# 1. 加载模型和预处理参数
# ======================
model = load_model('CNN_model_smallsize.h5', custom_objects={'mse': metrics.MeanSquaredError})  # 加载训练好的模型
scaler = joblib.load('CNN_scaler_smallsize.joblib')  # 加载标准化器


# ======================
# 2. 新数据预处理
# ======================
# 读取新数据
new_data = pd.read_excel(r"E:\物理模型结合机器学习\data1_20250324_泛化集.xlsx")
X_new = new_data.iloc[:, :10].values
y_true = new_data.iloc[:,-2:].values
y_true_df=pd.DataFrame(y_true, columns=['emitted current', 'collection current'])
# 应用相同的标准化
X_scaled = scaler.transform(X_new)

# 重塑为CNN输入格式
X_reshaped = X_scaled.reshape(-1, 1, 10, 1)

# ======================
# 3. 进行预测
# ======================
predictions = model.predict(X_reshaped)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print("预测时间为：", time_sum)

# ======================
# 4. 保存预测结果
# ======================
# 将预测结果转为DataFrame
result_df = pd.DataFrame(predictions,
                        columns=['emitted current', 'collection current'])

# 保存到Excel文件
result_df.to_excel('CNN_predictions_results.xlsx', index=False)
print("预测结果已保存至 CNN_predictions_results.xlsx")

# ======================
#5.绘图
# ======================
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.4,'font.family': 'Times New Roman'})
plt.figure(figsize=(12, 6))

# 定义通用格式参数
scatter_kws = {'alpha':0.7, 'color':'#2E86C1', 's':60, 'edgecolor':'w'}
line_kws = {'color':'#E74C3C', 'lw':2.5, 'alpha':0.8}

# 绘制发射电流对比图
plt.subplot(1, 2, 1)
ax1 = sns.regplot(x=y_true_df['emitted current'],
                y=result_df['emitted current'],
                scatter_kws=scatter_kws,
                line_kws=line_kws)

# 计算统计指标
y_true_emit = y_true_df['emitted current']
y_pred_emit = result_df['emitted current']
r2_emit = r2_score(y_true_emit, y_pred_emit)
rmse_emit = np.sqrt(mean_squared_error(y_true_emit, y_pred_emit))
pearson_emit = stats.pearsonr(y_true_emit, y_pred_emit)[0]

# 添加参考线
min_val = min(y_true_emit.min(), y_pred_emit.min())
max_val = max(y_true_emit.max(), y_pred_emit.max())
ax1.plot([min_val, max_val], [min_val, max_val], '--', color='#7D3C98', alpha=0.6)

# 添加统计信息
stats_text = (f'$R^2$ = {r2_emit:.3f}\n'
            f'RMSE = {rmse_emit:.3f}\n'
            f'Pearson = {pearson_emit:.3f}')
ax1.text(0.05, 0.80, stats_text,
        transform=ax1.transAxes,
        #fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))

ax1.set_xlabel('True Emission Current (a.u.)',  fontweight='semibold')
ax1.set_ylabel('Predicted Emission Current (a.u.)',  fontweight='semibold')
ax1.set_title('Emission Current Comparison',  fontweight='bold')

# 绘制收集电流对比图
plt.subplot(1, 2, 2)
ax2 = sns.regplot(x=y_true_df['collection current'],
                y=result_df['collection current'],
                scatter_kws=scatter_kws,
                line_kws=line_kws)

# 计算统计指标
y_true_coll = y_true_df['collection current']
y_pred_coll = result_df['collection current']
r2_coll = r2_score(y_true_coll, y_pred_coll)
rmse_coll = np.sqrt(mean_squared_error(y_true_coll, y_pred_coll))
pearson_coll = stats.pearsonr(y_true_coll, y_pred_coll)[0]

# 添加参考线
min_val = min(y_true_coll.min(), y_pred_coll.min())
max_val = max(y_true_coll.max(), y_pred_coll.max())
ax2.plot([min_val, max_val], [min_val, max_val], '--', color='#7D3C98', alpha=0.6)

# 添加统计信息
stats_text = (f'$R^2$ = {r2_coll:.3f}\n'
            f'RMSE = {rmse_coll:.3f}\n'
            f'Pearson = {pearson_coll:.3f}')
ax2.text(0.05, 0.80, stats_text,
        transform=ax2.transAxes,
        #fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))

ax2.set_xlabel('True Collection Current (a.u.)',  fontweight='semibold')
ax2.set_ylabel('Predicted Collection Current (a.u.)',  fontweight='semibold')
ax2.set_title('Collection Current Comparison',  fontweight='bold')


plt.tight_layout()
plt.savefig('泛化CNN_true_vs_pred_scatter_smallsize.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')
plt.show()