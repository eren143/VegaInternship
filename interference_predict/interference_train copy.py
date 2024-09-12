import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display as d
from statsmodels import api as sm
np.seterr(divide='ignore', invalid='ignore')

df = pd.read_csv("/Users/erenanbar/Desktop/internship_project/data/ticketData_vs_max_interference_202308141556.csv")
df.drop('day', axis = 1, inplace = True)
df = df.groupby(['interface','modelName','max_intf_score']).agg('sum').reset_index()
df['ticket_prob'] = df['ticket_size']/df['total_size']

isoR = IsotonicRegression(
        y_min=0,
        y_max=1,
        increasing=True,
        out_of_bounds="clip")

df_grouped = df.groupby(['interface','modelName'])
for interface_model, data in df_grouped:
    interface, modelName = interface_model[0],interface_model[1]
    x_train = data['max_intf_score']
    y_train = data['ticket_prob']
    isoR_intscore = isoR.fit(x_train, y_train)
    train_prediction = isoR_intscore.predict(x_train)
    train_prediction_smoothed = sm.nonparametric.lowess(
    train_prediction, x_train, frac=0.3)[:, 1]
    df.loc[(df['interface'] == interface) & (df['modelName'] == modelName), "prob_isoR"] = train_prediction
    df.loc[(df['interface'] == interface) & (df['modelName'] == modelName), "prob_isoR_smoothed"] = train_prediction_smoothed
    
#enter interface and model to plot
interface_model = (0,'DLink DIR-853/ET')

score_column = df_grouped.get_group((interface_model[0],interface_model[1]))['max_intf_score']
predict_column = df_grouped.get_group((interface_model[0],interface_model[1]))['prob_isoR_smoothed']
train_column = df_grouped.get_group((interface_model[0],interface_model[1]))['ticket_prob']

plt.plot(score_column,train_column,label='prob_train')
plt.plot(score_column,predict_column,label='prob_predict')
plt.xlabel('intf_score')
plt.ylabel('prob')
plt.title('interface: ' + str(interface_model[0]) + ' Model name: ' + interface_model[1] )
plt.legend()
plt.show()