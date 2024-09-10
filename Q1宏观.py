import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ��ȡ����
data = pd.read_excel("����ҽѧ��һ��������.xlsx")

# # Ϊ����ʾ�����ﴴ��һ��ʾ�� DataFrame
# data = pd.DataFrame({
#     '����': pd.date_range(start='2024-01-01', periods=100, freq='D'),
#     '���ﻼ���˴���': range(100),
#     'ҩƷ������': range(100, 200),
#     '���ղ�������': range(200, 300),
#     '��������OBS_T01_MZSR68': range(300, 400)
# })

# ȷ������������������
data['����'] = pd.to_datetime(data['����'])

# �������ͼ
plt.figure(figsize=(10, 8))
correlation_matrix = data[['���ﻼ���˴���', 'ҩƷ������', '���ղ�������', '��������OBS_T01_MZSR68']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('����֮������������ͼ')
plt.show()

# ����ͼ
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['���ﻼ���˴���', 'ҩƷ������', '���ղ�������', '��������OBS_T01_MZSR68']])
plt.title('������������ͼ')
plt.xlabel('����')
plt.ylabel('ֵ')
plt.show()
