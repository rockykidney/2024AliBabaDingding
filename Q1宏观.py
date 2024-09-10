import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_excel("康复医学科一病房数据.xlsx")

# # 为了演示，这里创建一个示例 DataFrame
# data = pd.DataFrame({
#     '日期': pd.date_range(start='2024-01-01', periods=100, freq='D'),
#     '门诊患者人次数': range(100),
#     '药品总收入': range(100, 200),
#     '当日病房收入': range(200, 300),
#     '门诊收入OBS_T01_MZSR68': range(300, 400)
# })

# 确保日期列是日期类型
data['日期'] = pd.to_datetime(data['日期'])

# 相关热力图
plt.figure(figsize=(10, 8))
correlation_matrix = data[['门诊患者人次数', '药品总收入', '当日病房收入', '门诊收入OBS_T01_MZSR68']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('变量之间的相关性热力图')
plt.show()

# 箱型图
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['门诊患者人次数', '药品总收入', '当日病房收入', '门诊收入OBS_T01_MZSR68']])
plt.title('各变量的箱型图')
plt.xlabel('变量')
plt.ylabel('值')
plt.show()
