import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
# ��ȡ����
df = pd.read_excel('����һδԭʼ����.xlsx', sheet_name=0)
data = df['��������OBS_T01_MZSR68'].fillna(df['��������OBS_T01_MZSR68'].mean())
# ����ƽ����
result = adfuller(data)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
# �����������в����Ż�
import itertools
# ���������Χ
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
m = 12 # ��������
# ���ɲ������
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, [m]))

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
best_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(data,
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)

            results = mod.fit()

            print(f'SARIMA{param}x{param_seasonal}12 - AIC:{results.aic}')

            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_model = results

        except Exception as e:
            print(e)
            continue

print(f'Best SARIMA{best_pdq}x{best_seasonal_pdq}12 - AIC:{best_aic}')
