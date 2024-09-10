import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import itertools

# 读取数据
df = pd.read_excel('康复医学科一病房数据.xlsx')

# 选择特征和目标变量
features = df[['门诊患者人次数', '药品总收入', '当日病房收入']]
target = df['门诊收入OBS_T01_MZSR68']

# 处理缺失值（例如，用中位数填充）
features = features.fillna(features.median())
target = target.fillna(target.median())

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 创建特征数据框架
features_df = pd.DataFrame(features_scaled, columns=['门诊患者人次数', '药品总收入', '当日病房收入'], index=df.index)

# 划分训练和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
train_features = features_df.iloc[:train_size]
test_features = features_df.iloc[train_size:]
train_target = target.iloc[:train_size]
test_target = target.iloc[train_size:]

# 确保目标变量和外生变量对齐
train_target = train_target.reset_index(drop=True)
test_target = test_target.reset_index(drop=True)
train_features = train_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

# 定义网格搜索的参数范围
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
m = 12  # 季节性周期
param_combinations = list(itertools.product(p, d, q))
seasonal_combinations = list(itertools.product(P, D, Q, [m]))

# 初始化最佳参数和指标
best_accuracy = -np.inf
best_f1 = -np.inf
best_auc = -np.inf
best_params_accuracy = None
best_params_f1 = None
best_params_auc = None

# 进行网格搜索
for param in param_combinations:
    for seasonal_param in seasonal_combinations:
        try:
            model = SARIMAX(train_target,
                            order=param,
                            seasonal_order=seasonal_param,
                            exog=train_features,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            
            # 预测测试集
            forecast = model_fit.forecast(steps=len(test_target), exog=test_features)
            
            # 将预测值和实际值转换为分类标签
            actual_labels = (test_target.diff().fillna(0) > 0).astype(int)
            predicted_labels = (forecast.diff().fillna(0) > 0).astype(int)
            
            # 计算分类指标
            accuracy = accuracy_score(actual_labels, predicted_labels)
            f1 = f1_score(actual_labels, predicted_labels)
            auc = roc_auc_score(actual_labels, forecast)
            
            # 更新最佳参数和指标
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params_accuracy = (param, seasonal_param)
                
            if f1 > best_f1:
                best_f1 = f1
                best_params_f1 = (param, seasonal_param)
                
            if auc > best_auc:
                best_auc = auc
                best_params_auc = (param, seasonal_param)
                
            print(f'SARIMA{param}x{seasonal_param} - AIC:{model_fit.aic}, Accuracy: {accuracy}, F1: {f1}, AUC: {auc}')

        except Exception as e:
            print(f"Error for SARIMA{param}x{seasonal_param}: {e}")
            continue

print(f'最佳准确率参数: {best_params_accuracy}, 最佳准确率: {best_accuracy}')
print(f'最佳F1参数: {best_params_f1}, 最佳F1: {best_f1}')
print(f'最佳AUC参数: {best_params_auc}, 最佳AUC: {best_auc}')

# 使用最佳模型进行预测
# 根据需要选择最佳指标的模型进行预测
best_model = SARIMAX(train_target,
                     order=best_params_accuracy[0],  # 使用最佳准确率的参数
                     seasonal_order=best_params_accuracy[1],
                     exog=train_features)
best_model_fit = best_model.fit(disp=False)
best_forecast = best_model_fit.forecast(steps=len(test_target), exog=test_features)
