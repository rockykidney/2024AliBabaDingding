
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
df=pd.read_excel('����.xlsx')
print(df.head())


#df['����']=pd.to_datetime(df['����']) #��������ת��Ϊ��������
#df.set_index(df['����']) #������������Ϊ����
df=df[df["���ղ��������Ӧ����"]=="����ҽѧ��һ����"] #ɸѡ������ҽѧ��һ����������
df = df[(df["���ﻼ���˴���"] >= 0) &(df["��������OBS_T01_MZSR68"]>=0) & (df["���ղ�������"] >= 0) & (df["ҩƷ������"] >= 0)]# ɸѡ�����ﻼ���˴������������롢���ղ������롢ҩƷ��������ڵ���0������
#df= df.replace([np.inf, -np.inf], np.nan)    # �����ݿ�df�е��������͸�������滻ΪNaN���п��� caonimade
print(df)  
df.to_excel("Q1-cleaned-data.xlsx", index=False) #��ɸѡ��ϴ������ݱ��浽�µ�Excel�ļ���