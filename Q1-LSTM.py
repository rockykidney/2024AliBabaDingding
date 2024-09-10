import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,roc_auc_score,roc_curve,auc

# ��ȡ����
df = pd.read_excel('Q1-cleaned-data.xlsx')

# ���ݴ���
df['����'] = pd.to_datetime(df['����'])
df.set_index('����', inplace=True)

#������������Ƶ�ʵ������ȱʧֵ
df = df.asfreq('D', method=None)
print("����ȱʧ���")
print(df.isnull().sum())

# ��������
df.plot(figsize=(12, 5))
plt.
plt.xlabel('����')
plt.ylabel('��������')
plt.title('����������ʱ��仯')
plt.legend
plt.show(
    
)

# ѡ��������Ŀ�����
target = df['��������OBS_T01_MZSR68']
features = df[['���ﻼ���˴���', 'ҩƷ������', '���ղ�������']]

# ����ѵ�����Ͳ��Լ�
train_size = int(len(df) * 0.8)
train, test = target[:train_size], target[train_size:]
train_exog, test_exog = features[:train_size], features[train_size:]

# ����SARIMAģ��
model = SARIMAX(train, 
                order=(2, 0, 2), 
                seasonal_order=(1, 1, 1, 12), 
                exog=train_exog)
model_fit = model.fit(disp=False)

# ����ѵ��MSE
train_predictions = model_fit.fittedvalues
train_mse = mean_squared_error(train, train_predictions)

# Ԥ����Լ�
test_predictions = model_fit.forecast(steps=len(test), exog=test_exog)

# �������MSE
test_mse = mean_squared_error(test, test_predictions)

# ���ӻ�ѵ��MSE�Ͳ���MSE
plt.figure(figsize=(10, 5))
plt.plot(train.index, (train - train_predictions)**2, label='Train MSE')
plt.plot(test.index, (test - test_predictions)**2, label='Test MSE')
plt.xlabel('Date')
plt.ylabel('MSE')
plt.title('Train and Test MSE over Time')
plt.legend()
plt.show()
