import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import itertools

# ��ȡ����
df = pd.read_excel('����ҽѧ��һ��������.xlsx')

# ѡ��������Ŀ�����
features = df[['���ﻼ���˴���', 'ҩƷ������', '���ղ�������']]
target = df['��������OBS_T01_MZSR68']

# ����ȱʧֵ�����磬����λ����䣩
features = features.fillna(features.median())
target = target.fillna(target.median())

# ��׼������
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# �����������ݿ��
features_df = pd.DataFrame(features_scaled, columns=['���ﻼ���˴���', 'ҩƷ������', '���ղ�������'], index=df.index)

# ����ѵ���Ͳ��Լ�
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
train_features = features_df.iloc[:train_size]
test_features = features_df.iloc[train_size:]
train_target = target.iloc[:train_size]
test_target = target.iloc[train_size:]

# ȷ��Ŀ�������������������
train_target = train_target.reset_index(drop=True)
test_target = test_target.reset_index(drop=True)
train_features = train_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

# �������������Ĳ�����Χ
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
m = 12  # ����������
param_combinations = list(itertools.product(p, d, q))
seasonal_combinations = list(itertools.product(P, D, Q, [m]))

# ��ʼ����Ѳ�����ָ��
best_accuracy = -np.inf
best_f1 = -np.inf
best_auc = -np.inf
best_params_accuracy = None
best_params_f1 = None
best_params_auc = None

# ������������
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
            
            # Ԥ����Լ�
            forecast = model_fit.forecast(steps=len(test_target), exog=test_features)
            
            # ��Ԥ��ֵ��ʵ��ֵת��Ϊ�����ǩ
            actual_labels = (test_target.diff().fillna(0) > 0).astype(int)
            predicted_labels = (forecast.diff().fillna(0) > 0).astype(int)
            
            # �������ָ��
            accuracy = accuracy_score(actual_labels, predicted_labels)
            f1 = f1_score(actual_labels, predicted_labels)
            auc = roc_auc_score(actual_labels, forecast)
            
            # ������Ѳ�����ָ��
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

print(f'���׼ȷ�ʲ���: {best_params_accuracy}, ���׼ȷ��: {best_accuracy}')
print(f'���F1����: {best_params_f1}, ���F1: {best_f1}')
print(f'���AUC����: {best_params_auc}, ���AUC: {best_auc}')

# ʹ�����ģ�ͽ���Ԥ��
# ������Ҫѡ�����ָ���ģ�ͽ���Ԥ��
best_model = SARIMAX(train_target,
                     order=best_params_accuracy[0],  # ʹ�����׼ȷ�ʵĲ���
                     seasonal_order=best_params_accuracy[1],
                     exog=train_features)
best_model_fit = best_model.fit(disp=False)
best_forecast = best_model_fit.forecast(steps=len(test_target), exog=test_features)
