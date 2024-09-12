import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from IPython import display as d

df = pd.read_csv("/Users/erenanbar/Desktop/internship_project/data/ticketData_vs_max_interference_202308141556.csv")
df.drop('day', axis = 1, inplace = True)

grouped = df.groupby(['interface','modelName','max_intf_score'])

prob_df = pd.DataFrame(columns = ['interface','model','score','ticket_prob'])

for interface_model_score, data in grouped:
    interface = interface_model_score[0]
    model = interface_model_score[1]
    score = interface_model_score[2]
    ticket_prob = (data['ticket_size'].sum()/data['total_size'].sum())
    row = [interface,model,score,ticket_prob]
    prob_df.loc[len(prob_df)] = row
    
prob_grouped = prob_df.groupby(['interface','model'])

interface_model = (0,'Arcadyan eLife Connect C1AA')

score_column = prob_grouped.get_group((interface_model[0],interface_model[1]))['score']
prob_column = prob_grouped.get_group((interface_model[0],interface_model[1])) ['ticket_prob']





X_Y_Spline = make_interp_spline(score_column, prob_column)
 
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(score_column.min(), score_column.max(), 14)
Y_ = X_Y_Spline(X_)


plt.plot(X_,Y_,label='ticket_prob')
plt.xlabel('intf_score')
plt.ylabel('prob')
plt.title('interface: ' + str(interface_model[0]) + ' Model name: ' + interface_model[1] )
plt.legend()
plt.show()