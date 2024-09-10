import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,roc_auc_score,roc_curve,auc

# 读取数据
df = pd.read_excel('Q1-cleaned-data.xlsx')

# 数据处理
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

#设置日期索引频率但不填充缺失值
df = df.asfreq('D', method=None)
print("数据缺失情况")
print(df.isnull().sum())

# 绘制数据
df.plot(figsize=(12, 5))
plt.
plt.xlabel('日期')
plt.ylabel('门诊收入')
plt.title('门诊收入随时间变化')
plt.legend
plt.show(
    
)

# 选择特征和目标变量
target = df['门诊收入OBS_T01_MZSR68']
features = df[['门诊患者人次数', '药品总收入', '当日病房收入']]

# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = target[:train_size], target[train_size:]
train_exog, test_exog = features[:train_size], features[train_size:]

# 构建SARIMA模型
model = SARIMAX(train, 
                order=(2, 0, 2), 
                seasonal_order=(1, 1, 1, 12), 
                exog=train_exog)
model_fit = model.fit(disp=False)

# 计算训练MSE
train_predictions = model_fit.fittedvalues
train_mse = mean_squared_error(train, train_predictions)

# 预测测试集
test_predictions = model_fit.forecast(steps=len(test), exog=test_exog)

# 计算测试MSE
test_mse = mean_squared_error(test, test_predictions)

# 可视化训练MSE和测试MSE
plt.figure(figsize=(10, 5))
plt.plot(train.index, (train - train_predictions)**2, label='Train MSE')
plt.plot(test.index, (test - test_predictions)**2, label='Test MSE')
plt.xlabel('Date')
plt.ylabel('MSE')
plt.title('Train and Test MSE over Time')
plt.legend()
plt.show()
