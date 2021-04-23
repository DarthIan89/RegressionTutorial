import pandas as pd
import numpy as np
import statsmodels.api as sm

df_adv = pd.read_csv('ESE_Data.csv')

X = df_adv[['Attendance', 'MSE', 'HRS']]
y = df_adv[['ESE']]
X= sm.add_constant(X)
est = sm.OLS(y,X).fit()
print(est.summary())
print(est.params)
