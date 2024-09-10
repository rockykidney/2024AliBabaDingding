
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
df=pd.read_excel('数据.xlsx')
print(df.head())


#df['日期']=pd.to_datetime(df['日期']) #将日期列转换为日期类型
#df.set_index(df['日期']) #将日期列设置为索引
df=df[df["当日病房收入对应科室"]=="康复医学科一病房"] #筛选出康复医学科一病房的数据
df = df[(df["门诊患者人次数"] >= 0) &(df["门诊收入OBS_T01_MZSR68"]>=0) & (df["当日病房收入"] >= 0) & (df["药品总收入"] >= 0)]# 筛选出门诊患者人次数、门诊收入、当日病房收入、药品总收入大于等于0的数据
#df= df.replace([np.inf, -np.inf], np.nan)    # 将数据框df中的正无穷大和负无穷大替换为NaN可有可无 caonimade
print(df)  
df.to_excel("Q1-cleaned-data.xlsx", index=False) #将筛选清洗后的数据保存到新的Excel文件中